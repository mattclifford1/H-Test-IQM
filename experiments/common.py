'''
Shared configuration for every experiment. Paths, the comparison list, the scorer list.

Nothing here runs anything. Import it, don't execute it.

The comparison list is the spine of the whole results section: each entry names a pair of
datasets (or a pair of subsets of one dataset) and says whether a correct method should reject
the null. `control-disjoint` is the only 'accept' row and it is the one that matters -- if that
rejects, nothing else in the table means anything.
'''
import os
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'results')
FIGURES_DIR = os.path.join(ROOT, 'figures')
CACHE_DIR = os.path.join(RESULTS_DIR, 'score_cache')

ALPHA = 0.05

# every experiment that subsamples uses this as its base seed
SEED = 0


# --- the comparisons -------------------------------------------------------------------------
# (name, target spec, test spec, expected verdict)
#
# A spec is {dataset, labels, partition}. partition 'a'/'b' take disjoint halves of the same
# cached pool, which is how the control is made honest -- a different seed only re-shuffles and
# leaves chance-level overlap. See FINDINGS.md 3.4.
COMPARISONS = [
    ('control-disjoint',
     dict(dataset='CIFAR_10', labels='all', partition='a'),
     dict(dataset='CIFAR_10', labels='all', partition='b'),
     'accept'),
    ('cifar10-vs-cifar100',
     dict(dataset='CIFAR_10', labels='all'),
     dict(dataset='CIFAR_100', labels='all'),
     'reject'),
    ('cifar-vs-imagenet64',
     dict(dataset='CIFAR_10', labels='all'),
     dict(dataset='IMAGENET64_VAL', labels='all'),
     'reject'),
    ('cifar-vs-dtd',
     dict(dataset='CIFAR_10', labels='all'),
     dict(dataset='DTD', labels='all'),
     'reject'),
    ('cifar-vs-oneclass',
     dict(dataset='CIFAR_10', labels='all', partition='a'),
     dict(dataset='CIFAR_10', labels=[1], partition='b'),
     'reject'),
    ('cifar-vs-uniform',
     dict(dataset='CIFAR_10', labels='all'),
     dict(dataset='UNIFORM', labels='all'),
     'reject'),
]

# display order for tables and figures
ORDER = [c[0] for c in COMPARISONS]
ORDER_BY_DIFFICULTY = ['control-disjoint', 'cifar10-vs-cifar100', 'cifar-vs-dtd',
                       'cifar-vs-imagenet64', 'cifar-vs-oneclass', 'cifar-vs-uniform']

# the scorer under test, then the controls. Order is the table's column order.
SCORERS = [
    'entropy-2-mse',    # the perceptual autoencoder -- the method under test
    'jpeg_bytes',       # baseline: compressibility
    'pixel_std',        # baseline: dispersion of pixel values
    'pixel_entropy',    # baseline: grey-level histogram entropy
    'BRISQUE',          # baseline: a classical no-reference IQM (slow, ~34 images/sec)
]

# the autoencoder checkpoints that actually work as a scalar scorer. centres=5 is excluded --
# counts_per_emb_feature_flat has a bare TODO there and returns raw embeddings rather than a
# ratio (FINDINGS.md 3.10) -- and `mae` is excluded because models/README.md says those weights
# are broken.
AE_CHECKPOINTS = [
    'entropy-2-mse', 'entropy-2-ssim', 'entropy-2-nlpd',
    'entropy-2-mse-u', 'entropy-2-ssim-u', 'entropy-2-nlpd-u',
]

IM_SIZES = [32, 64, 128, 256]
DEFAULT_IM_SIZE = 256


# --- helpers ---------------------------------------------------------------------------------

def out_path(name, ext='csv'):
    '''results/<name>.<ext>, with the directory made'''
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, f'{name}.{ext}')


def fig_path(name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    return os.path.join(FIGURES_DIR, name)


def seed_for(*parts):
    '''
    A reproducible seed sequence from arbitrary labels.

    Python's hash() is salted per process (PYTHONHASHSEED), so using it here would make every
    run irreproducible in a way that is invisible until someone tries to rerun a cell.
    '''
    out = []
    for p in parts:
        if isinstance(p, str):
            out.append(zlib.crc32(p.encode()))
        else:
            out.append(int(p))
    return out


def subsample(rng, scores, n):
    '''n scores drawn without replacement'''
    if n > len(scores):
        raise ValueError(f'asked for {n} scores from a pool of {len(scores)}')
    return scores[rng.choice(len(scores), size=n, replace=False)]


def wilson_ci(k, n, z=1.96):
    '''
    Wilson interval for a detection rate. Normal-approximation intervals are useless here
    because the interesting cells sit at 0% and 100%, where they have zero width.
    '''
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
