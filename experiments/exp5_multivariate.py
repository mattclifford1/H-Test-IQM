'''
EXPERIMENT 5 -- does keeping the code help? Scalar vs 64-D autoencoder scores.

The project's statistic compresses a 64 x 16 x 16 quantised latent to ONE number: the overall
fraction of +1s. A JPEG byte count beats that number on every comparison (FINDINGS.md 2.4).
That is not really a fair fight for the autoencoder -- it is one scalar against another, and the
autoencoder's scalar throws away everything the code knows about WHICH channels fired.

So: rerun the comparisons keeping one +1-ratio per latent channel (scorer `entropy-2-mse-64d`,
which is the already-implemented but never-used `spacial=True` path) and test in R^64.

    representation      test
    ---------------------------------------------------------------------------
    AE scalar           KS          the current method
    AE scalar           C2ST        same data, so the test is not the reason for any gap
    AE 64-D             C2ST        does the code carry more than its own mean?
    AE 64-D             energy      "  (permutation, no classifier to blame)
    AE 64-D             MMD         "
    jpeg_bytes scalar   KS          the baseline that currently wins

If the 64-D tests beat the scalar at the same n, the perceptual scorer is worth keeping and the
paper's framing survives. If they do not, the code genuinely carries no more usable signal than
its own mean does, and that is a result in itself -- it says the occupancy statistic, not the
autoencoder, is the bottleneck.

Energy and MMD are O(n^2) per permutation, so they run at fewer repeats than the cheap tests.
C2ST is the one that carries the power comparison because it applies unchanged to 1-D and 64-D.

Output: results/exp5_multivariate.csv       detection rates for the cheap tests
        results/exp5_multivariate_perm.csv  energy / MMD, fewer repeats

Usage:  python -m experiments.exp5_multivariate [--repeats 200]
'''
import sys
import time
import argparse

import numpy as np
import pandas as pd

from h_test_IQM.pipeline.h_tests import KS
from h_test_IQM.pipeline.multivariate import energy_test, mmd_test, c2st
from experiments.common import (COMPARISONS, ALPHA, SEED, DEFAULT_IM_SIZE, out_path,
                                wilson_ci, seed_for)
from experiments import sampling

SCALAR = 'entropy-2-mse'
VECTOR = 'entropy-2-mse-64d'
BASELINE = 'jpeg_bytes'

SAMPLE_SIZES = [250, 500, 1000]


def _draw(rng, comparison, scorer, im_size, n):
    _, target, test, _ = comparison
    tp = sampling.resolve(target, scorer, im_size)
    sp = sampling.resolve(test, scorer, im_size)
    return sampling.draw_pair(rng, tp, sp, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=200, help='for KS and C2ST')
    ap.add_argument('--perm-repeats', type=int, default=20, help='for energy and MMD')
    ap.add_argument('--permutations', type=int, default=199)
    ap.add_argument('--im-size', type=int, default=DEFAULT_IM_SIZE)
    ap.add_argument('--sizes', nargs='*', type=int, default=SAMPLE_SIZES)
    ap.add_argument('--skip-perm', action='store_true')
    args = ap.parse_args()

    rows, perm_rows = [], []
    t0 = time.time()

    for name, target, test, expected in COMPARISONS:
        comparison = (name, target, test, expected)
        for n in args.sizes:
            # --- cheap tests: KS and C2ST, on scalar and on 64-D -----------------------
            arms = [
                ('AE-scalar', SCALAR, 'KS'),
                ('AE-scalar', SCALAR, 'C2ST'),
                ('AE-64d', VECTOR, 'C2ST'),
                ('jpeg-scalar', BASELINE, 'KS'),
                ('jpeg-scalar', BASELINE, 'C2ST'),
            ]
            for rep_name, scorer, test_name in arms:
                rng = np.random.default_rng(seed_for(SEED, name, rep_name, test_name, n))
                try:
                    tp = sampling.resolve(target, scorer, args.im_size)
                    sp = sampling.resolve(test, scorer, args.im_size)
                except FileNotFoundError as e:
                    print(f'  SKIP {name} / {rep_name} -- {str(e).splitlines()[0]}')
                    continue
                if n > min(len(tp), len(sp)):
                    continue

                hits, effects = 0, []
                for _ in range(args.repeats):
                    a, b = sampling.draw_pair(rng, tp, sp, n)
                    if test_name == 'KS':
                        stat, p = KS(a, b)
                    else:
                        stat, p, _ = c2st(a, b, seed=int(rng.integers(2**31)))
                    hits += p < ALPHA
                    effects.append(stat)
                lo, hi = wilson_ci(hits, args.repeats)
                rows.append(dict(
                    comparison=name, expected=expected, representation=rep_name,
                    dims=1 if rep_name != 'AE-64d' else 64,
                    test=test_name, n=n, repeats=args.repeats, im_size=args.im_size,
                    detect=hits / args.repeats, detect_lo=lo, detect_hi=hi,
                    effect_mean=float(np.mean(effects)),
                    effect_sd=float(np.std(effects))))
                sys.stdout.write(
                    f'  {name:22s} n={n:<5d} {rep_name:12s} {test_name:5s} '
                    f'detect={hits/args.repeats:6.1%}  effect={np.mean(effects):.4f}\n')
                sys.stdout.flush()
            pd.DataFrame(rows).to_csv(out_path('exp5_multivariate'), index=False)

        # --- expensive permutation tests on the 64-D code, at the largest n -----------
        if args.skip_perm:
            continue
        n = max(args.sizes)
        rng = np.random.default_rng(seed_for(SEED, name, 'perm'))
        try:
            tp = sampling.resolve(target, VECTOR, args.im_size)
            sp = sampling.resolve(test, VECTOR, args.im_size)
        except FileNotFoundError:
            continue
        if n > min(len(tp), len(sp)):
            continue
        for test_name, fn in [('energy', energy_test), ('MMD', mmd_test)]:
            hits, stats = 0, []
            for _ in range(args.perm_repeats):
                a, b = sampling.draw_pair(rng, tp, sp, n)
                stat, p = fn(a, b, n_permutations=args.permutations,
                             seed=int(rng.integers(2**31)))
                hits += p < ALPHA
                stats.append(stat)
            lo, hi = wilson_ci(hits, args.perm_repeats)
            perm_rows.append(dict(
                comparison=name, expected=expected, representation='AE-64d', dims=64,
                test=test_name, n=n, repeats=args.perm_repeats,
                permutations=args.permutations, im_size=args.im_size,
                detect=hits / args.perm_repeats, detect_lo=lo, detect_hi=hi,
                effect_mean=float(np.mean(stats))))
            sys.stdout.write(
                f'  {name:22s} n={n:<5d} {"AE-64d":12s} {test_name:6s} '
                f'detect={hits/args.perm_repeats:6.1%}\n')
            sys.stdout.flush()
        pd.DataFrame(perm_rows).to_csv(out_path('exp5_multivariate_perm'), index=False)

    pd.DataFrame(rows).to_csv(out_path('exp5_multivariate'), index=False)
    if perm_rows:
        pd.DataFrame(perm_rows).to_csv(out_path('exp5_multivariate_perm'), index=False)
    print(f'\n{len(rows)} + {len(perm_rows)} cells in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
