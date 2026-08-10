'''
EXPERIMENT 2 -- graded shift: how BIG a contamination can be detected, with how many images?

Every other comparison in this project is a fixed pair of datasets, so it can only ever answer
"different: yes or no". Nothing measured a shift of controllable size. This does.

The test sample is CIFAR-10 with a fraction f of its images replaced by another dataset:

    test = (1 - f) * n images from CIFAR-10   +   f * n images from the contaminant
    target = n images from a disjoint half of CIFAR-10

f = 0 is exactly the control, so the f=0 column is a built-in calibration check on every row.
Sweeping f and n gives the surface that answers the question the drafts actually pose: given a
collection of n images, how much foreign material can be sitting in it before this notices?

Contaminants, easiest to hardest:
    UNIFORM          noise. The sanity check -- should be caught at tiny f.
    DTD              textures. A different image domain.
    IMAGENET64_VAL   natural images at a different native resolution. The realistic case.
    CIFAR_100        natural images, same source, different classes. The hard case.

Mixing at the score level is exactly equivalent to mixing images and then scoring, because the
scorer is per-image -- which is why this whole surface costs seconds off the cache.

Output: results/exp2_contamination.csv -- one row per (scorer, contaminant, n, f).

Usage:  python -m experiments.exp2_contamination [--repeats 500]
'''
import sys
import time
import argparse

import numpy as np
import pandas as pd

from h_test_IQM.pipeline.h_tests import KS
from experiments.common import (SCORERS, ALPHA, SEED, DEFAULT_IM_SIZE, out_path,
                                wilson_ci, seed_for)
from experiments import sampling

CLEAN = dict(dataset='CIFAR_10', labels='all', partition='a')
CLEAN_TEST = dict(dataset='CIFAR_10', labels='all', partition='b')

CONTAMINANTS = {
    'uniform': dict(dataset='UNIFORM', labels='all'),
    'dtd': dict(dataset='DTD', labels='all'),
    'imagenet64': dict(dataset='IMAGENET64_VAL', labels='all'),
    'cifar100': dict(dataset='CIFAR_100', labels='all'),
}

FRACTIONS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
SAMPLE_SIZES = [100, 250, 500, 1000, 2000, 4000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=500)
    ap.add_argument('--scorers', nargs='*', default=SCORERS)
    ap.add_argument('--im-size', type=int, default=DEFAULT_IM_SIZE)
    ap.add_argument('--fractions', nargs='*', type=float, default=FRACTIONS)
    ap.add_argument('--sizes', nargs='*', type=int, default=SAMPLE_SIZES)
    args = ap.parse_args()

    rows = []
    t0 = time.time()

    for scorer in args.scorers:
        try:
            clean_target = sampling.resolve(CLEAN, scorer, args.im_size)
            clean_test = sampling.resolve(CLEAN_TEST, scorer, args.im_size)
        except FileNotFoundError as e:
            print(f'  SKIP {scorer} -- {str(e).splitlines()[0]}')
            continue

        for cname, cspec in CONTAMINANTS.items():
            try:
                dirty = sampling.resolve(cspec, scorer, args.im_size)
            except FileNotFoundError as e:
                print(f'  SKIP {scorer} / {cname} -- {str(e).splitlines()[0]}')
                continue

            for n in args.sizes:
                if n > min(len(clean_target), len(clean_test)):
                    continue
                for f in args.fractions:
                    n_bad = int(round(f * n))
                    if n_bad > len(dirty):
                        continue        # not enough contaminant to build this cell
                    rng = np.random.default_rng(
                        seed_for(SEED, scorer, cname, n, int(f * 10000)))
                    hits = 0
                    stats = []
                    for _ in range(args.repeats):
                        a = sampling.draw(rng, clean_target, n)
                        b = sampling.draw_contaminated(rng, clean_test, dirty, n, f)
                        s, p = KS(a, b)
                        stats.append(s)
                        hits += p < ALPHA
                    lo, hi = wilson_ci(hits, args.repeats)
                    rows.append(dict(
                        scorer=scorer, contaminant=cname, n=n, fraction=f,
                        n_contaminating=n_bad, repeats=args.repeats,
                        im_size=args.im_size,
                        detect=hits / args.repeats, detect_lo=lo, detect_hi=hi,
                        KS_mean=float(np.mean(stats))))
                sys.stdout.write(f'  {scorer:16s} {cname:11s} n={n:<5d} ' + ' '.join(
                    f"{r['fraction']:g}:{r['detect']:.0%}"
                    for r in rows if r['scorer'] == scorer
                    and r['contaminant'] == cname and r['n'] == n) + '\n')
                sys.stdout.flush()
            pd.DataFrame(rows).to_csv(out_path('exp2_contamination'), index=False)

    pd.DataFrame(rows).to_csv(out_path('exp2_contamination'), index=False)
    print(f'\n{len(rows)} cells in {(time.time()-t0)/60:.1f} min '
          f"-> {out_path('exp2_contamination')}")


if __name__ == '__main__':
    main()
