'''
Score every dataset once, per scorer, per resolution -- then never score again.

WHY THIS EXISTS
    Scoring is a pure function of (image, scorer, im_size), but every experiment used to
    re-run the whole dataset -> scorer path for every repeat. That is what limited the first
    power curve to 10 repeats, which quantises a detection rate to 10% and makes the control
    row look like a 10-20% false-positive rate when it is really 1-2 hits out of 10.

    Cache the scores and the sampling happens in numpy. 1000 repeats becomes free, and the
    contamination sweep becomes free too: mixing images at fraction f is exactly the same
    thing as drawing (1-f)*n scores from one cached array and f*n from another.

WHAT IS STORED
    results/score_cache/<dataset>__<scorer>__<im_size>.npz with
        scores   (N,) float32   -- or (N, 64) for the spacial autoencoder variant
        labels   (N,) int64     -- numerical class label, so class subsets are a mask
        indices  (N,) int64     -- row in the dataset's meta_data csv, for traceability

    Row order is a FIXED RANDOM PERMUTATION of the dataset (seeded by CACHE_SEED), which is
    what makes the rest of the code simple: scores[:n] is already a random sample, and
    scores[:h] / scores[h:] are two disjoint random halves. Because the permutation depends
    only on the dataset, row i is the same image for every scorer and every resolution, so a
    small cache is a prefix of a large one and the columns of a table are comparable.

    The cache is derived data and is gitignored. Rebuilding is the cost below.

COST (RTX 3090, measured)
    entropy-2-*, pixel_*, jpeg_bytes   seconds to a couple of minutes per dataset
    BRISQUE                            ~34 images/sec, so it is capped at a smaller pool

Usage:
    python -m experiments.score_cache                 # build everything the experiments need
    python -m experiments.score_cache --list          # show what is already cached
    python -m experiments.score_cache --datasets CIFAR_10 --scorers jpeg_bytes --im-sizes 32
'''
import os
import sys
import time
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from h_test_IQM.datasets import DATA_LOADER, TOTAL_INSTANCES
from h_test_IQM.scorers import SCORERS as SCORER_REGISTRY
from h_test_IQM.pipeline.main import _get_scorer

from experiments.common import CACHE_DIR, SCORERS, AE_CHECKPOINTS, IM_SIZES

# fixed forever: changing it invalidates every cached file and every result built on one
CACHE_SEED = 12345

NORMALISE = (0, 1)

# how many images to cache per dataset. Bigger pools mean more independent repeats; the only
# cost is build time. CIFAR-10 gets the full set because it is the target in every comparison
# and the class-restricted subsets are a tenth of it.
POOL = {
    'CIFAR_10': 60000,
    'CIFAR_100': 30000,
    'IMAGENET64_VAL': 30000,
    'DTD': 5640,
    'UNIFORM': 30000,
}

# BRISQUE is ~30x slower than anything else, so it gets a smaller pool. Since the row order is
# a dataset-level permutation, its pool is a strict prefix of the others -- same images.
SCORER_POOL_CAP = {'BRISQUE': 20000}

DATASETS = list(POOL)


def pool_size(dataset, scorer):
    n = min(POOL[dataset], TOTAL_INSTANCES[dataset])
    return min(n, SCORER_POOL_CAP.get(scorer, n))


def cache_file(dataset, scorer, im_size):
    return os.path.join(CACHE_DIR, f'{dataset}__{scorer}__{int(im_size)}.npz')


def is_cached(dataset, scorer, im_size):
    return os.path.exists(cache_file(dataset, scorer, im_size))


_WARNED_NONFINITE = set()


def load(dataset, scorer, im_size):
    '''
    -> dict with scores / labels / indices. Raises if it was never built.

    Non-finite scores are dropped here, with the labels and indices masked alongside them.
    BRISQUE genuinely returns NaN on a handful of images -- 2 of 5640 DTD images at 32px, where
    the MSCN normalisation divides by an almost-zero local variance. It is a real scorer
    failure, not corruption, but a single NaN turns every downstream KS and KL into NaN, so it
    cannot be left in. The raw .npz keeps what the scorer actually produced; the filtering
    happens on the way out and is reported once per file.
    '''
    path = cache_file(dataset, scorer, im_size)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'no cache for {dataset} / {scorer} / {im_size}. Build it with:\n'
            f'  python -m experiments.score_cache --datasets {dataset} '
            f'--scorers {scorer} --im-sizes {int(im_size)}')
    with np.load(path) as f:
        scores, labels, indices = f['scores'], f['labels'], f['indices']

    finite = np.isfinite(scores)
    if finite.ndim == 2:            # the 64-D variant: drop a row if any channel is bad
        finite = finite.all(axis=1)
    if not finite.all():
        key = (dataset, scorer, im_size)
        if key not in _WARNED_NONFINITE:
            _WARNED_NONFINITE.add(key)
            n_bad = int((~finite).sum())
            print(f'  note: dropped {n_bad}/{len(finite)} non-finite scores from '
                  f'{dataset} / {scorer} / {im_size}px')
        scores, labels, indices = scores[finite], labels[finite], indices[finite]

    return {'scores': scores, 'labels': labels, 'indices': indices}


def _permutation(dataset, n):
    '''the fixed row order for this dataset -- same for every scorer and resolution'''
    total = TOTAL_INSTANCES[dataset]
    rng = np.random.default_rng(CACHE_SEED)
    return rng.permutation(total)[:n]


def _make_loader(dataset, indices, batch_size, device):
    '''
    A DataLoader over exactly `indices`, in that order, with image caching OFF.

    cache_data=False matters: DTD at 256x256 is 4.4 GB of decoded float32 and we only ever
    make one pass. shuffle=False matters too -- the row order IS the permutation, and the
    labels have to stay aligned with the scores.
    '''
    ds = DATA_LOADER[dataset](normalise=NORMALISE,
                              indicies_to_use=list(indices),
                              image_dict={},
                              cache_data=False,
                              device='cpu')
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def build_dataset(dataset, scorers, im_size, device='cuda', batch_size=32, force=False):
    '''
    Build every missing (dataset, scorer, im_size) cache in ONE pass over the images.

    One pass because decoding and resizing dominate for the cheap scorers, so scoring five
    of them per batch costs barely more than scoring one.
    '''
    todo = [s for s in scorers if force or not is_cached(dataset, s, im_size)]
    if not todo:
        return {}

    os.makedirs(CACHE_DIR, exist_ok=True)

    # each scorer may want a different pool size -- take one pass over the largest, and write
    # out the shorter prefixes for the rest
    wanted = {s: pool_size(dataset, s) for s in todo}
    n_max = max(wanted.values())
    indices = _permutation(dataset, n_max)
    loader = _make_loader(dataset, indices, batch_size, device)

    models = {s: _get_scorer(s, (im_size, im_size), device) for s in todo}
    out = {s: [] for s in todo}
    labels = []

    t0 = time.time()
    desc = f'{dataset} @ {im_size}px'
    for batch in tqdm(loader, desc=desc, leave=False):
        img, _, label = batch
        labels.append(np.asarray(label).reshape(-1))
        seen = sum(len(l) for l in labels)
        for s in todo:
            if seen - len(labels[-1]) >= wanted[s]:
                continue           # this scorer's pool is already full
            out[s].append(np.asarray(models[s](img)))

    labels = np.concatenate(labels)[:n_max].astype(np.int64)
    secs = time.time() - t0

    written = {}
    for s in todo:
        scores = np.concatenate(out[s], axis=0)[:wanted[s]].astype(np.float32)
        n = len(scores)
        np.savez_compressed(cache_file(dataset, s, im_size),
                            scores=scores,
                            labels=labels[:n],
                            indices=indices[:n].astype(np.int64))
        written[s] = n
        shape = 'x'.join(str(d) for d in scores.shape)
        print(f'  {dataset:16s} {s:18s} {im_size:>3d}px  n={n:<6d} ({shape})')
    print(f'  -> {dataset} @ {im_size}px done in {secs/60:.1f} min')
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*', default=DATASETS)
    ap.add_argument('--scorers', nargs='*', default=None,
                    help='default: the 5 table scorers + the 6 AE checkpoints + the 64-D one')
    ap.add_argument('--im-sizes', nargs='*', type=int, default=None,
                    help='default: 256 for everything, plus 32/64/128 for the fast scorers')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--force', action='store_true', help='rebuild even if cached')
    ap.add_argument('--list', action='store_true', help='show what is cached and exit')
    args = ap.parse_args()

    if args.list:
        os.makedirs(CACHE_DIR, exist_ok=True)
        files = sorted(os.listdir(CACHE_DIR))
        if not files:
            print('score cache is empty')
            return
        total = 0
        for f in files:
            path = os.path.join(CACHE_DIR, f)
            mb = os.path.getsize(path) / 1e6
            total += mb
            with np.load(path) as d:
                shape = 'x'.join(str(x) for x in d['scores'].shape)
            print(f'  {f:52s} {shape:>12s}  {mb:6.1f} MB')
        print(f'  {len(files)} files, {total:.0f} MB')
        return

    # what to build. BRISQUE at every resolution is an hour on its own, so by default it only
    # gets the two resolutions the tables actually compare.
    if args.scorers is not None and args.im_sizes is not None:
        plan = [(s, z) for s in args.scorers for z in args.im_sizes]
    else:
        scorers = args.scorers
        fast = [s for s in SCORERS if s != 'BRISQUE']
        extra = [c for c in AE_CHECKPOINTS if c not in SCORERS] + ['entropy-2-mse-64d']
        if scorers is None:
            plan = ([(s, z) for s in fast for z in (args.im_sizes or IM_SIZES)]
                    + [('BRISQUE', z) for z in (args.im_sizes or [32, 256])]
                    + [(s, 256) for s in extra])
        else:
            plan = [(s, z) for s in scorers for z in (args.im_sizes or [256])]

    by_size = {}
    for s, z in plan:
        by_size.setdefault(z, []).append(s)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('cuda not available, falling back to cpu')
        args.device = 'cpu'

    t0 = time.time()
    for im_size in sorted(by_size):
        for dataset in args.datasets:
            build_dataset(dataset, by_size[im_size], im_size,
                          device=args.device, batch_size=args.batch_size,
                          force=args.force)
            sys.stdout.flush()
    print(f'\ncache build finished in {(time.time()-t0)/60:.1f} min -> {CACHE_DIR}')


if __name__ == '__main__':
    main()
