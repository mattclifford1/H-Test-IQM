'''
EXPERIMENT 8 -- a featuriser ladder: what is the untrained encoder detecting?

Pre-registered in experiments/PREREGISTRATION.md (commit e3c3ee8, before this file existed).
Read that first: it holds the predictions, the check and how each outcome is read.

exp7 found an untrained autoencoder encoder beats the same architecture trained on natural
images or on noise at detecting CIFAR-10 vs CIFAR-100. This puts it on a ladder of featurisers
ordered by how much they know about images, all through the same held-out logistic C2ST:

    1   colour moments                         6-D   32 px
    2   random projection of raw pixels        64-D  32 px    seeds 0-2
    3   untrained AE encoder, occupancy        64-D  256 px   seeds 0-2  (exp7's winner)
    3q  untrained AE encoder, activation       64-D  256 px   seeds 0-2  (no quantiser)
    4   natural-trained AE mse-2, occupancy    64-D  256 px
    4q  natural-trained AE mse-2, activation   64-D  256 px
    5   untrained ResNet-18                    512-D 224 px   seeds 0-2
    5m  untrained ResNet-18, 64 of 512 chans   64-D           one channel subset per seed
    6   ImageNet ResNet-18                     512-D 224 px
    6m  ImageNet ResNet-18, 64 of 512 chans    64-D           channel subsets 0-2

plus jpeg_bytes (scalar KS) as the reference. Every null redraws its partition on every repeat
(FINDINGS.md 2.9.8).

Parts:
    null       CIFAR-10 vs CIFAR-10, redrawn partition, n=1000, 1000 repeats
    primary    CIFAR-10 vs CIFAR-100, n in {100, 250, 500, 1000}, 200 repeats
    classdrop  k = 9 and its k = 10 null, redrawn partition, n=2000, 500 draws

Output: results/exp8_featuriser_ladder.csv -- one row per (part, arm, n, k)

Usage:  python -m experiments.exp8_featuriser_ladder [--parts null primary classdrop] [--jobs 8]
'''
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

from h_test_IQM.pipeline.h_tests import KS
from h_test_IQM.pipeline.multivariate import c2st
from experiments.common import ALPHA, SEED, out_path, wilson_ci, seed_for
from experiments import score_cache

SUBSET_DIM = 64
NUM_CLASSES = 10

# (rung, member, scorer, im_size, channel-subset seed or None, test)
ARMS = (
    [('1', 'colour', 'colour-moments', 32, None, 'C2ST')]
    + [('2', f's{s}', f'random-pixels-s{s}-64d', 32, None, 'C2ST') for s in range(3)]
    + [('3', f's{s}', f'entropy-2-random-s{s}-64d', 256, None, 'C2ST') for s in range(3)]
    + [('3q', f's{s}', f'entropy-2-random-s{s}-act64d', 256, None, 'C2ST') for s in range(3)]
    + [('4', 'mse', 'entropy-2-mse-64d', 256, None, 'C2ST')]
    + [('4q', 'mse', 'entropy-2-mse-act64d', 256, None, 'C2ST')]
    + [('5', f's{s}', f'resnet18-random-s{s}', 224, None, 'C2ST') for s in range(3)]
    + [('5m', f's{s}', f'resnet18-random-s{s}', 224, s, 'C2ST') for s in range(3)]
    + [('6', 'imagenet', 'resnet18-imagenet', 224, None, 'C2ST')]
    + [('6m', f'sub{s}', 'resnet18-imagenet', 224, s, 'C2ST') for s in range(3)]
    + [('ref', 'jpeg', 'jpeg_bytes', 256, None, 'KS')]
)

RUNG_NAME = {
    '1': 'colour moments', '2': 'random pixel projection',
    '3': 'untrained AE (occupancy)', '3q': 'untrained AE (activation)',
    '4': 'natural AE mse-2 (occupancy)', '4q': 'natural AE mse-2 (activation)',
    '5': 'untrained ResNet-18 (512)', '5m': 'untrained ResNet-18 (64 ch)',
    '6': 'ImageNet ResNet-18 (512)', '6m': 'ImageNet ResNet-18 (64 ch)',
    'ref': 'JPEG bytes (scalar KS)',
}


def _load(dataset, scorer, im_size, subset):
    c = score_cache.load(dataset, scorer, im_size)
    scores, labels = c['scores'], c['labels']
    if subset is not None:
        # a seeded channel subset, fixed per (scorer, subset) and never fitted to data
        rng = np.random.default_rng(seed_for(SEED, 'exp8-subset', scorer, subset))
        keep = np.sort(rng.choice(scores.shape[1], size=SUBSET_DIM, replace=False))
        scores = scores[:, keep]
    return scores, labels


def _test(a, b, test, rng):
    if test == 'KS':
        stat, p = KS(a, b)
    else:
        stat, p, _ = c2st(a, b, seed=int(rng.integers(2**31)))
    return stat, p


def _row(part, arm, n, hits, total, effects, k=None, comparison=''):
    rung, member, scorer, im_size, subset, test = arm
    lo, hi = wilson_ci(hits, total)
    return dict(part=part, comparison=comparison, rung=rung, rung_name=RUNG_NAME[rung],
                member=member, scorer=scorer, im_size=im_size,
                subset=-1 if subset is None else subset, test=test, n=n, k_classes=k,
                repeats=total, detect=hits / total, detect_lo=lo, detect_hi=hi,
                effect_mean=float(np.mean(effects)), effect_sd=float(np.std(effects)))


# --- parts (each runs one arm; parallelised over arms in main) --------------------------------

def run_null(arm, n=1000, repeats=1000):
    '''CIFAR-10 against itself, a fresh disjoint partition every repeat'''
    scores, _ = _load('CIFAR_10', arm[2], arm[3], arm[4])
    rng = np.random.default_rng(seed_for(SEED, 'exp8', 'null', *map(str, arm)))
    hits, eff = 0, []
    for _ in range(repeats):
        idx = rng.choice(len(scores), size=2 * n, replace=False)
        stat, p = _test(scores[idx[:n]], scores[idx[n:]], arm[5], rng)
        hits += p < ALPHA
        eff.append(stat)
    # not 'null': pandas reads that string back as a missing value
    return [_row('null_redrawn', arm, n, hits, repeats, eff, comparison='cifar10-vs-cifar10')]


def run_primary(arm, sizes=(100, 250, 500, 1000), repeats=200):
    a_pool, _ = _load('CIFAR_10', arm[2], arm[3], arm[4])
    b_pool, _ = _load('CIFAR_100', arm[2], arm[3], arm[4])
    rows = []
    for n in sizes:
        rng = np.random.default_rng(seed_for(SEED, 'exp8', 'primary', n, *map(str, arm)))
        hits, eff = 0, []
        for _ in range(repeats):
            a = a_pool[rng.choice(len(a_pool), size=n, replace=False)]
            b = b_pool[rng.choice(len(b_pool), size=n, replace=False)]
            stat, p = _test(a, b, arm[5], rng)
            hits += p < ALPHA
            eff.append(stat)
        rows.append(_row('primary', arm, n, hits, repeats, eff,
                         comparison='cifar10-vs-cifar100'))
    return rows


def run_classdrop(arm, n=2000, draws=500):
    '''target: n rows of all of CIFAR-10; test: n of the REMAINING rows, k classes kept'''
    scores, labels = _load('CIFAR_10', arm[2], arm[3], arm[4])
    rows = []
    for k in (9, 10):
        rng = np.random.default_rng(seed_for(SEED, 'exp8', 'classdrop', k, *map(str, arm)))
        dropped_list = list(range(NUM_CLASSES)) if k == 9 else [None]
        per = draws // len(dropped_list)
        hits, eff = 0, []
        for dropped in dropped_list:
            for _ in range(per):
                perm = rng.permutation(len(scores))
                target, rest = perm[:n], perm[n:]
                if dropped is not None:
                    rest = rest[labels[rest] != dropped]
                stat, p = _test(scores[target], scores[rest[:n]], arm[5], rng)
                hits += p < ALPHA
                eff.append(stat)
        rows.append(_row('classdrop', arm, n, hits, per * len(dropped_list), eff, k=k,
                         comparison='cifar-k-of-10'))
    return rows


PARTS = {'null': run_null, 'primary': run_primary, 'classdrop': run_classdrop}


def _say(r):
    sys.stdout.write(f'  {r["part"]:9s} rung {r["rung"]:3s} {r["member"]:8s} {r["scorer"]:28s}'
                     f' n={r["n"]:<5d} k={str(r["k_classes"]):4s} detect={r["detect"]:6.1%} '
                     f'[{r["detect_lo"]:.3f}, {r["detect_hi"]:.3f}] '
                     f'effect={r["effect_mean"]:.4f}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', nargs='*', default=list(PARTS))
    ap.add_argument('--jobs', type=int, default=8)
    args = ap.parse_args()

    # one BLAS thread per worker, so N workers do not fight over the same cores
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from joblib import Parallel, delayed

    t0 = time.time()
    try:
        old = pd.read_csv(out_path('exp8_featuriser_ladder'))
        written = [{'null': 'null_redrawn'}.get(p, p) for p in args.parts]
        rows = old[~old['part'].isin(written)].to_dict('records')
    except FileNotFoundError:
        rows = []

    for part in args.parts:
        print(f'\n=== {part}', flush=True)
        results = Parallel(n_jobs=args.jobs)(delayed(PARTS[part])(arm) for arm in ARMS)
        for arm_rows in results:
            for r in arm_rows:
                _say(r)
                rows.append(r)
        sys.stdout.flush()
        pd.DataFrame(rows).to_csv(out_path('exp8_featuriser_ladder'), index=False)

    print(f'\n{len(rows)} cells in {(time.time() - t0) / 60:.1f} min '
          f'-> {out_path("exp8_featuriser_ladder")}')


if __name__ == '__main__':
    main()
