'''
Two-sample tests for VECTOR scores, so a scorer does not have to collapse to one number.

Why this exists: the autoencoder produces a 64 x 16 x 16 quantised code per image and the
project's statistic throws all of it away except the overall fraction of +1s. A JPEG byte count
beats that scalar (FINDINGS.md 2.4), which is unsurprising -- one number is one number. The
question worth asking is whether the CODE carries more than the byte count does, and answering
it needs tests that work in R^d.

Three of them, all distribution-free:

  energy_test  Szekely-Rizzo energy distance. Consistent against any difference in
               distribution, no kernel to tune, scale-sensitive (so standardise first).
  mmd_test     Maximum mean discrepancy with an RBF kernel, bandwidth by the median heuristic.
  c2st         Classifier two-sample test. Train a classifier to tell the samples apart; if it
               beats chance out of sample, they differ. Slower, but the accuracy is directly
               interpretable as an effect size and it says WHICH direction the difference is
               in, which the others do not.

Energy and MMD both depend on the pooled samples only through a pairwise matrix, so the
permutation null reuses one precomputed matrix and reindexes it. That is what makes an exact
p-value affordable: 200 permutations on n=1000 per side is a couple of seconds, not minutes.

Both use the V-statistic form (the diagonal is included). It is biased for the population
quantity, but the permutation null is computed the same way, so the TEST is still exact.
'''
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import binomtest


def _as_2d(X):
    X = np.asarray(X, dtype=np.float64)
    return X[:, None] if X.ndim == 1 else X


def standardise(X, Y):
    '''
    Centre and scale by the POOLED statistics.

    Necessary for energy distance, which is not scale-free: without it a latent channel with a
    wide range would dominate the statistic purely because of its units. Using the pooled mean
    and sd (rather than each sample's own) keeps the transform identical for both samples, so
    it cannot manufacture or hide a difference between them.
    '''
    X, Y = _as_2d(X), _as_2d(Y)
    pooled = np.vstack([X, Y])
    mu = pooled.mean(axis=0)
    sd = pooled.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, (Y - mu) / sd


def _pooled_matrix(X, Y, kind='euclidean'):
    pooled = np.vstack([X, Y])
    if kind == 'euclidean':
        return cdist(pooled, pooled, 'euclidean'), len(X)
    if kind == 'sqeuclidean':
        return cdist(pooled, pooled, 'sqeuclidean'), len(X)
    raise ValueError(kind)


def _energy_from(M, ix, iy):
    xx = M[np.ix_(ix, ix)].mean()
    yy = M[np.ix_(iy, iy)].mean()
    xy = M[np.ix_(ix, iy)].mean()
    return 2 * xy - xx - yy


def _mmd_from(K, ix, iy):
    xx = K[np.ix_(ix, ix)].mean()
    yy = K[np.ix_(iy, iy)].mean()
    xy = K[np.ix_(ix, iy)].mean()
    return xx + yy - 2 * xy


def _permutation_p(M, n1, n_total, stat_fn, observed, n_permutations, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n_total)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(idx)
        if stat_fn(M, idx[:n1], idx[n1:]) >= observed:
            count += 1
    return (1.0 + count) / (1.0 + n_permutations)


def energy_test(X, Y, n_permutations=200, seed=0, scale=True):
    '''Energy distance + permutation p-value. Returns (statistic, p_value).'''
    X, Y = (standardise(X, Y) if scale else (_as_2d(X), _as_2d(Y)))
    M, n1 = _pooled_matrix(X, Y, 'euclidean')
    n_total = len(M)
    observed = _energy_from(M, np.arange(n1), np.arange(n1, n_total))
    p = _permutation_p(M, n1, n_total, _energy_from, observed, n_permutations, seed)
    return float(observed), float(p)


def mmd_test(X, Y, n_permutations=200, seed=0, scale=True, gamma=None):
    '''
    MMD^2 with an RBF kernel + permutation p-value. Returns (statistic, p_value).

    gamma defaults to the median heuristic: 1 / median(pairwise squared distance), computed on
    the POOLED sample so the permutation null uses the same kernel as the observed statistic.
    '''
    X, Y = (standardise(X, Y) if scale else (_as_2d(X), _as_2d(Y)))
    D2, n1 = _pooled_matrix(X, Y, 'sqeuclidean')
    if gamma is None:
        med = np.median(D2[D2 > 0]) if np.any(D2 > 0) else 1.0
        gamma = 1.0 / med
    K = np.exp(-gamma * D2)
    n_total = len(K)
    observed = _mmd_from(K, np.arange(n1), np.arange(n1, n_total))
    p = _permutation_p(K, n1, n_total, _mmd_from, observed, n_permutations, seed)
    return float(observed), float(p)


def c2st(X, Y, seed=0, test_fraction=0.5, classifier='logistic'):
    '''
    Classifier two-sample test. Returns (accuracy, p_value, n_test).

    Label one sample 0 and the other 1, fit a classifier on a TRAIN split, and test its
    accuracy on a held-out split against chance with an exact binomial test. Chance is 0.5 only
    when the two samples are the same size, which is why this refuses unequal inputs.

    The accuracy IS the effect size: 0.5 means indistinguishable, 1.0 means trivially separable.
    That is far easier to report than a divergence in nats.

    A single held-out split, not cross-validation, and that is deliberate. The binomial test
    needs the predictions to be independent given the fitted model, which holds for one clean
    test split but NOT for out-of-fold predictions -- those share training data through the
    folds. Measured on a true null, the cross-validated version rejects about 9% of the time at
    a nominal 5%; the held-out version below lands on 5%. The cost is that half the data goes
    to training, so this has less power than the permutation-based tests at the same n.
    '''
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, Y = _as_2d(X), _as_2d(Y)
    if len(X) != len(Y):
        raise ValueError(
            f'c2st needs equal sample sizes so chance is 0.5, got {len(X)} and {len(Y)}')

    data = np.vstack([X, Y])
    labels = np.concatenate([np.zeros(len(X)), np.ones(len(Y))]).astype(int)

    if classifier == 'logistic':
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, random_state=seed))
    elif classifier == 'forest':
        model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    else:
        raise ValueError(f'unknown classifier {classifier!r}')

    train_x, test_x, train_y, test_y = train_test_split(
        data, labels, test_size=test_fraction, stratify=labels, random_state=seed)
    model.fit(train_x, train_y)
    correct = int((model.predict(test_x) == test_y).sum())
    n = len(test_y)
    p = binomtest(correct, n, 0.5, alternative='greater').pvalue
    return float(correct / n), float(p), n


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    d = 64
    same_a = rng.normal(0, 1, (400, d))
    same_b = rng.normal(0, 1, (400, d))
    # a difference in ONE channel out of 64 -- invisible to a mean over channels
    shifted = rng.normal(0, 1, (400, d))
    shifted[:, 7] += 0.5

    for label, (A, B) in [('same', (same_a, same_b)), ('one channel shifted',
                                                       (same_a, shifted))]:
        e, ep = energy_test(A, B, n_permutations=200)
        m, mp = mmd_test(A, B, n_permutations=200)
        acc, cp, _ = c2st(A, B)
        print(f'{label:22s} energy={e:.4f} p={ep:.3g} | mmd={m:.5f} p={mp:.3g} '
              f'| c2st acc={acc:.3f} p={cp:.3g}')
