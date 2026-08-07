'''
Two-sample comparisons between two 1-D samples of scores.

Two families live here:

  divergences  - KL, Jensen-Shannon, Wasserstein. Effect sizes. No p-value on their own.
  tests        - KS, Cramer-von Mises, Anderson-Darling. Statistic AND p-value, no binning.

Anything binned goes through samples_to_pmf, which smooths the *counts* rather than the
densities. This matters: the old code added a fixed 1e-6 to `density=True` values, whose
magnitude depends on the width of the score axis, so the resulting KL was not invariant to
rescaling the score and moved ~2.9 nats per decade of that constant. Smoothing counts is
scale-free (alpha=0.5 is the Krichevsky-Trofimov estimator).

For a genuinely distribution-free p-value on *any* statistic, use permutation_test.

THE ANALYTIC P-VALUES ARE CONSERVATIVE AT SMALL n. Measured on 500 null draws at n=200, the KS
p-values skew high (mean 0.548 where uniform would give 0.5) and the false-positive rate lands at
3.6% rather than the nominal 5%; by n=1000 the deviation is gone. That is safe -- it will not
invent significance -- but it costs power.

Two things contribute, and it is worth not confusing them. The dominant term at small n is
scipy's asymptotic KS approximation: perfectly continuous synthetic samples reproduce the n=200
behaviour almost exactly (mean 0.539, uniformity p=6.5e-06). Separately, these scores are
genuinely DISCRETE -- a ratio of integer code counts, granularity 1/n_codes, so 1/16384 at
im_size=256 and only 1/256 at im_size=32 -- and the resulting ties become the dominant term once
n is large enough for the asymptotic approximation to be good.

permutation_test is exact under both, so prefer it when the effect is marginal. See FINDINGS.md 2.7.
'''
import numpy as np
from scipy.stats import (entropy, ks_2samp, cramervonmises_2samp, anderson_ksamp,
                         wasserstein_distance)


def samples_to_pmf(sample1, sample2, num_bins=50, alpha=0.5):
    '''
    Two samples -> two smoothed PMFs over a shared binning.

    alpha is added to the counts (not the densities) before normalising, so no bin is ever
    empty and the result does not depend on the units of the score. alpha=0.5 is
    Krichevsky-Trofimov; alpha=1 is Laplace; alpha=0 is the raw MLE (and will give an
    infinite KL if any bin of q is empty).
    '''
    sample1 = np.asarray(sample1).ravel()
    sample2 = np.asarray(sample2).ravel()
    range_vals = (min(sample1.min(), sample2.min()),
                  max(sample1.max(), sample2.max()))
    counts1, bins = np.histogram(sample1, bins=num_bins, range=range_vals)
    counts2, _ = np.histogram(sample2, bins=num_bins, range=range_vals)

    p = counts1 + alpha
    q = counts2 + alpha
    return p / p.sum(), q / q.sum(), bins


def KL(sample1, sample2, num_bins=50, alpha=0.5):
    '''KL(sample1 || sample2) in nats. Asymmetric: sample1 is the target/reference.'''
    p, q, _ = samples_to_pmf(sample1, sample2, num_bins=num_bins, alpha=alpha)
    return float(entropy(pk=p, qk=q))


def JS(sample1, sample2, num_bins=50, alpha=0.5):
    '''
    Jensen-Shannon distance: symmetric, and bounded in [0, 1] when using log base 2.
    Easier to report than KL because it cannot run away when the supports barely overlap.
    '''
    p, q, _ = samples_to_pmf(sample1, sample2, num_bins=num_bins, alpha=alpha)
    m = 0.5 * (p + q)
    js_div = 0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2)
    return float(np.sqrt(max(js_div, 0.0)))


def wasserstein(sample1, sample2):
    '''1st Wasserstein (earth mover) distance. No binning, in units of the score.'''
    return float(wasserstein_distance(np.asarray(sample1).ravel(),
                                      np.asarray(sample2).ravel()))


def KS(sample1, sample2):
    '''Two-sample Kolmogorov-Smirnov. Returns (statistic, p-value). Sensitive to the bulk.'''
    res = ks_2samp(np.asarray(sample1).ravel(), np.asarray(sample2).ravel())
    return float(res.statistic), float(res.pvalue)


def CVM(sample1, sample2):
    '''Two-sample Cramer-von Mises. Returns (statistic, p-value). Integrates the whole CDF.'''
    res = cramervonmises_2samp(np.asarray(sample1).ravel(),
                               np.asarray(sample2).ravel())
    return float(res.statistic), float(res.pvalue)


def AD(sample1, sample2):
    '''
    Two-sample Anderson-Darling. Returns (statistic, p-value). Weights the tails more than
    KS does. scipy floors/caps the p-value at [0.001, 0.25] and warns at the edges.
    '''
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = anderson_ksamp([np.asarray(sample1).ravel(),
                              np.asarray(sample2).ravel()])
    return float(res.statistic), float(res.pvalue)


def permutation_test(sample1, sample2, statistic=KL, n_permutations=1000, seed=0,
                     **stat_kwargs):
    '''
    Distribution-free p-value for ANY two-sample statistic, including the binned ones.

    Pools the two samples, reshuffles, re-splits at the original size, and recomputes the
    statistic to build the null distribution of "these came from the same distribution".

    Returns (observed, p_value, null_distribution). The p-value uses the (1 + count) / (1 + n)
    form so it is never exactly 0 -- with n_permutations=1000 the smallest reportable p is
    1/1001 ~= 1e-3.

    This is what turns a KL number into an accept/reject decision, and the quantile of the
    null is the threshold the method needs.
    '''
    sample1 = np.asarray(sample1).ravel()
    sample2 = np.asarray(sample2).ravel()
    observed = statistic(sample1, sample2, **stat_kwargs)
    if isinstance(observed, tuple):   # a test that returns (stat, p) - use the statistic
        observed = observed[0]
        _stat = lambda a, b: statistic(a, b, **stat_kwargs)[0]
    else:
        _stat = lambda a, b: statistic(a, b, **stat_kwargs)

    pooled = np.concatenate([sample1, sample2])
    n1 = len(sample1)
    rng = np.random.default_rng(seed)

    null = np.empty(n_permutations)
    for i in range(n_permutations):
        rng.shuffle(pooled)
        null[i] = _stat(pooled[:n1], pooled[n1:])

    p_value = (1.0 + np.sum(null >= observed)) / (1.0 + n_permutations)
    return float(observed), float(p_value), null


# registry so get_scores(test=[...]) can look these up by name
DIVERGENCES = {
    'KL': KL,
    'JS': JS,
    'wasserstein': wasserstein,
}

TESTS = {
    'KS': KS,
    'CVM': CVM,
    'AD': AD,
}


def compare(sample1, sample2, which='all', num_bins=50, alpha=0.5,
            n_permutations=0, seed=0):
    '''
    Run every divergence and test at once. Returns a flat dict.

    which: 'all', or a list of names from DIVERGENCES / TESTS.
    n_permutations: if > 0, also compute a permutation p-value for each divergence
                    (the tests already have their own).
    '''
    if which == 'all':
        names = list(DIVERGENCES) + list(TESTS)
    elif isinstance(which, str):
        names = [which]
    else:
        names = list(which)

    results = {}
    for name in names:
        if name in DIVERGENCES:
            func = DIVERGENCES[name]
            kwargs = {'num_bins': num_bins, 'alpha': alpha} if name in ('KL', 'JS') else {}
            if n_permutations > 0:
                obs, p, _ = permutation_test(sample1, sample2, statistic=func,
                                             n_permutations=n_permutations, seed=seed,
                                             **kwargs)
                results[name] = obs
                results[f'{name}_p'] = p
            else:
                results[name] = func(sample1, sample2, **kwargs)
        elif name in TESTS:
            stat, p = TESTS[name](sample1, sample2)
            results[name] = stat
            results[f'{name}_p'] = p
        else:
            raise ValueError(
                f'unknown comparison {name!r}, expected one of '
                f'{list(DIVERGENCES) + list(TESTS)}')
    return results


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    same_a = rng.normal(0.5, 0.012, 4000)
    same_b = rng.normal(0.5, 0.012, 4000)
    spike = rng.normal(0.5, 0.0015, 4000)

    print('--- same distribution (should accept) ---')
    for k, v in compare(same_a, same_b, n_permutations=200).items():
        print(f'  {k:16s} {v:.4g}')

    print('--- broad vs spike (should reject) ---')
    for k, v in compare(same_a, spike, n_permutations=200).items():
        print(f'  {k:16s} {v:.4g}')

    print('--- scale invariance check (x1000 on the score axis) ---')
    print(f'  KL as is    {KL(same_a, spike):.4f}')
    print(f'  KL x1000    {KL(same_a * 1000, spike * 1000):.4f}')
