'''
Turn a comparison spec into two arrays of scores, drawn from the cache.

This is the only place that knows how a spec ({dataset, labels, partition}) becomes a sample,
so every experiment draws its data the same way and the control means the same thing everywhere.

The cache stores each dataset in a fixed random row order, which is what makes this cheap:
  - a class subset is a boolean mask, and the surviving rows are still in random order
  - partition 'a'/'b' are the two halves of that order, so they share no images at all
  - a sample of n is n rows chosen without replacement
'''
import numpy as np

from experiments import score_cache


def resolve(spec, scorer, im_size):
    '''spec -> the pool of scores it can draw from (a view into the cached array)'''
    cached = score_cache.load(spec['dataset'], scorer, im_size)
    scores, labels = cached['scores'], cached['labels']

    if spec.get('labels', 'all') != 'all':
        keep = np.isin(labels, spec['labels'])
        scores = scores[keep]

    partition = spec.get('partition')
    if partition is not None:
        half = len(scores) // 2
        if partition == 'a':
            scores = scores[:half]
        elif partition == 'b':
            scores = scores[half:]
        else:
            raise ValueError(f"partition must be None, 'a' or 'b', got {partition!r}")
    return scores


def pool_sizes(comparison, scorer, im_size):
    '''(n available on the target side, n available on the test side)'''
    _, target, test, _ = comparison
    return len(resolve(target, scorer, im_size)), len(resolve(test, scorer, im_size))


def max_n(comparison, scorer, im_size):
    return min(pool_sizes(comparison, scorer, im_size))


def draw(rng, pool, n):
    '''n scores without replacement'''
    if n > len(pool):
        raise ValueError(f'asked for {n} from a pool of {len(pool)}')
    idx = rng.choice(len(pool), size=n, replace=False)
    return pool[idx]


def draw_pair(rng, target_pool, test_pool, n):
    return draw(rng, target_pool, n), draw(rng, test_pool, n)


def draw_contaminated(rng, clean_pool, contaminant_pool, n, fraction):
    '''
    n scores of which round(fraction * n) come from the contaminant.

    Mixing at the score level is exactly equivalent to mixing the images and scoring the
    mixture, because the scorer is per-image. That equivalence is what makes the whole
    contamination sweep free.
    '''
    n_bad = int(round(fraction * n))
    n_good = n - n_bad
    parts = [draw(rng, clean_pool, n_good)]
    if n_bad > 0:
        parts.append(draw(rng, contaminant_pool, n_bad))
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out
