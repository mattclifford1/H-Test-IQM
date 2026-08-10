'''
EXPERIMENT 4 -- how much class coverage is missing before it shows up?

The original one-class experiment removed 9 of CIFAR-10's 10 classes -- an enormous shift, and
a single point on what should be a curve. This sweeps it: the test sample keeps k of the 10
classes, k = 1..10, and the target keeps all of them.

    k = 10  is exactly the control (both sides are all-class CIFAR-10, disjoint halves), so the
            right-hand end of every curve is a built-in false-positive check
    k = 1   is the old one-class comparison

A monotone curve is the claim the drafts want to make -- "adding a class adds density, and the
method sees it" -- and this is the cheapest way to test it. If it is NOT monotone, that is worth
more than if it is: it would mean the score is picking up something about particular classes
rather than about coverage.

Which k classes are kept matters, so each k averages over several random class subsets rather
than always keeping classes 0..k-1. The spread across subsets is reported as subset_sd, and it
is the interesting quantity: if it is large, "how many classes" is the wrong axis and "which
classes" is the real story.

Output: results/exp4_class_count.csv -- one row per (scorer, k).

Usage:  python -m experiments.exp4_class_count [--n 2000]
'''
import sys
import time
import argparse
import itertools

import numpy as np
import pandas as pd

from h_test_IQM.pipeline.h_tests import KS, KL
from experiments.common import (SCORERS, ALPHA, SEED, DEFAULT_IM_SIZE, out_path,
                                wilson_ci, seed_for)
from experiments import sampling, score_cache

NUM_CLASSES = 10
TARGET = dict(dataset='CIFAR_10', labels='all', partition='a')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2000, help='images per side')
    ap.add_argument('--subsets', type=int, default=20,
                    help='random class subsets per k')
    ap.add_argument('--repeats', type=int, default=25, help='minimum draws per subset')
    ap.add_argument('--draws', type=int, default=500,
                    help='total draws per k -- repeats per subset is raised to hit this, '
                         'so every k is estimated to the same precision')
    ap.add_argument('--scorers', nargs='*', default=SCORERS)
    ap.add_argument('--im-size', type=int, default=DEFAULT_IM_SIZE)
    args = ap.parse_args()

    rows = []
    t0 = time.time()

    for scorer in args.scorers:
        try:
            target_pool = sampling.resolve(TARGET, scorer, args.im_size)
        except FileNotFoundError as e:
            print(f'  SKIP {scorer} -- {str(e).splitlines()[0]}')
            continue

        for k in range(1, NUM_CLASSES + 1):
            rng = np.random.default_rng(seed_for(SEED, scorer, 'classcount', k))
            # all C(10,k) subsets when there are few, otherwise a random sample of them
            all_subsets = list(itertools.combinations(range(NUM_CLASSES), k))
            if len(all_subsets) > args.subsets:
                pick = rng.choice(len(all_subsets), size=args.subsets, replace=False)
                subsets = [all_subsets[i] for i in pick]
            else:
                subsets = all_subsets

            # Equalise the work per k. C(10,k) collapses to 1 at k=10 and 10 at k=9, so a
            # fixed repeats-per-subset would leave the control row resting on `repeats`
            # draws while the middle of the sweep gets `subsets * repeats`. The control is
            # the one cell that has to be precise, so spend the draws where the subsets ran out.
            repeats = max(args.repeats, -(-args.draws // len(subsets)))

            per_subset_ks, hits, total, kls, ks_all = [], 0, 0, [], []
            skipped = 0
            for classes in subsets:
                spec = dict(dataset='CIFAR_10', labels=list(classes), partition='b')
                test_pool = sampling.resolve(spec, scorer, args.im_size)
                if len(test_pool) < args.n:
                    skipped += 1
                    continue
                subset_ks = []
                for _ in range(repeats):
                    a, b = sampling.draw_pair(rng, target_pool, test_pool, args.n)
                    s, p = KS(a, b)
                    subset_ks.append(s)
                    ks_all.append(s)
                    kls.append(KL(a, b))
                    hits += p < ALPHA
                    total += 1
                per_subset_ks.append(np.mean(subset_ks))

            if total == 0:
                print(f'  SKIP {scorer} k={k} -- pools too small for n={args.n}')
                continue
            lo, hi = wilson_ci(hits, total)
            rows.append(dict(
                scorer=scorer, k_classes=k, n=args.n, im_size=args.im_size,
                subsets=len(per_subset_ks), draws=total,
                detect=hits / total, detect_lo=lo, detect_hi=hi,
                KS_mean=float(np.mean(ks_all)), KS_sd=float(np.std(ks_all)),
                subset_sd=float(np.std(per_subset_ks)) if len(per_subset_ks) > 1 else 0.0,
                KL_mean=float(np.mean(kls))))
            sys.stdout.write(
                f'  {scorer:16s} k={k:<3d} KS={np.mean(ks_all):.4f} '
                f'(between-subset sd {rows[-1]["subset_sd"]:.4f})  '
                f'detect={hits/total:6.1%}\n')
            sys.stdout.flush()
        pd.DataFrame(rows).to_csv(out_path('exp4_class_count'), index=False)

    pd.DataFrame(rows).to_csv(out_path('exp4_class_count'), index=False)
    print(f'\n{len(rows)} cells in {(time.time()-t0)/60:.1f} min '
          f"-> {out_path('exp4_class_count')}")


if __name__ == '__main__':
    main()
