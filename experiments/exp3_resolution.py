'''
EXPERIMENT 3 -- the main table, at every resolution, for every scorer.

Two jobs in one sweep.

  1. THE MAIN TABLE. Every comparison x every scorer, size-matched, with a real p-value.
     Unlike results/prior/density_grid.csv this reports the mean over many independent draws
     rather than a single one, so the numbers come with an error bar and a rejection RATE
     rather than a single accept/reject.

  2. THE RESOLUTION CONFOUND. Everything in this project is resized to 256x256 before scoring,
     which upsamples CIFAR 8x (FINDINGS.md 3.6). That matters twice over:

       - the autoencoder's discriminative power RISES with im_size while the control stays
         flat, so the headline result partly depends on an undocumented hardcode;
       - CIFAR-vs-DTD and CIFAR-vs-ImageNet64 compare datasets with different NATIVE
         resolutions, so at 256 they are partly resolution comparisons for any scorer.

     Running the full table at 32/64/128/256 separates the two. The 32px table is the honest
     resolution-matched one for the CIFAR-native comparisons; the 256px table is what the
     earlier results used.

     Caveat worth stating in the paper: DTD images are variable-size and its loader resizes to
     256 on read, so "DTD at 32" is 256 -> 32, not a true native-resolution sample.

Output: results/exp3_resolution.csv -- one row per (comparison, scorer, im_size).

Usage:  python -m experiments.exp3_resolution [--repeats 100] [--n 4000]
'''
import sys
import time
import argparse

import numpy as np
import pandas as pd

from h_test_IQM.pipeline.h_tests import KS, CVM, AD, KL, JS, wasserstein
from experiments.common import (COMPARISONS, SCORERS, IM_SIZES, ALPHA, SEED,
                                out_path, wilson_ci, seed_for)
from experiments import sampling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4000, help='images per side')
    ap.add_argument('--repeats', type=int, default=100)
    ap.add_argument('--scorers', nargs='*', default=SCORERS)
    ap.add_argument('--im-sizes', nargs='*', type=int, default=IM_SIZES)
    args = ap.parse_args()

    rows = []
    t0 = time.time()

    for im_size in args.im_sizes:
        for scorer in args.scorers:
            for name, target, test, expected in COMPARISONS:
                try:
                    tp = sampling.resolve(target, scorer, im_size)
                    sp = sampling.resolve(test, scorer, im_size)
                except FileNotFoundError as e:
                    print(f'  SKIP {scorer:16s} {name:22s} @{im_size} -- '
                          f'{str(e).splitlines()[0]}')
                    continue

                n = min(args.n, len(tp), len(sp))
                rng = np.random.default_rng(seed_for(SEED, name, scorer, im_size))
                acc = {k: [] for k in ['KS', 'KS_p', 'CVM', 'CVM_p', 'AD', 'AD_p',
                                       'KL', 'JS', 'wasserstein']}
                for _ in range(args.repeats):
                    a, b = sampling.draw_pair(rng, tp, sp, n)
                    for key, fn in [('KS', KS), ('CVM', CVM), ('AD', AD)]:
                        stat, p = fn(a, b)
                        acc[key].append(stat)
                        acc[f'{key}_p'].append(p)
                    acc['KL'].append(KL(a, b))
                    acc['JS'].append(JS(a, b))
                    acc['wasserstein'].append(wasserstein(a, b))

                k = int((np.asarray(acc['KS_p']) < ALPHA).sum())
                lo, hi = wilson_ci(k, args.repeats)
                row = dict(comparison=name, scorer=scorer, im_size=im_size,
                           expected=expected, n=n, repeats=args.repeats,
                           reject_rate=k / args.repeats,
                           reject_lo=lo, reject_hi=hi)
                for key, vals in acc.items():
                    row[key] = float(np.mean(vals))
                    row[f'{key}_sd'] = float(np.std(vals))
                row['KS_p_median'] = float(np.median(acc['KS_p']))
                rows.append(row)

                verdict = 'REJECT' if k / args.repeats > 0.5 else 'accept'
                sys.stdout.write(
                    f"  {im_size:>3d}px {scorer:16s} {name:22s} "
                    f"KS={row['KS']:.4f}+-{row['KS_sd']:.4f} "
                    f"reject={k/args.repeats:5.0%} {verdict}\n")
                sys.stdout.flush()
            pd.DataFrame(rows).to_csv(out_path('exp3_resolution'), index=False)

    pd.DataFrame(rows).to_csv(out_path('exp3_resolution'), index=False)
    print(f'\n{len(rows)} cells in {(time.time()-t0)/60:.1f} min '
          f"-> {out_path('exp3_resolution')}")


if __name__ == '__main__':
    main()
