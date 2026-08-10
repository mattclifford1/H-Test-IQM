'''
EXPERIMENT 1 -- statistical power: how many images before the method notices?

A p-value at n=4000 only says "yes, these differ". The question a data curator actually has is
how much data they need before a shift of a given size becomes detectable. So: sweep n, repeat
each cell many times, and report the fraction of repeats that reject at alpha=0.05.

The `control-disjoint` row is not a power measurement -- it is the FALSE-POSITIVE rate, and it
has to sit near 5%. If it does not, every other row is meaningless.

WHAT IS NEW HERE (vs results/prior/power_curve.csv)
  - 1000 repeats instead of 10. The old version re-scored every image on every repeat, which
    capped it at 10 and quantised the detection rate to 10% -- the control row read 10-20%,
    which is 1-2 hits out of 10 and perfectly consistent with 5%. It looked like a calibration
    problem and was not one.
  - every scorer, not just the autoencoder. The old curve was entropy-2-mse only, so its
    "n needed" numbers do not apply to the baseline that actually wins the main table.

Output: results/exp1_power_curve.csv -- one row per (comparison, scorer, n), aggregated,
        plus results/exp1_power_curve_raw.csv.gz with every repeat.

Usage:  python -m experiments.exp1_power_curve [--repeats 1000]
'''
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

from h_test_IQM.pipeline.h_tests import KS, CVM, KL
from experiments.common import (COMPARISONS, SCORERS, ALPHA, SEED, DEFAULT_IM_SIZE,
                                out_path, wilson_ci, seed_for)
from experiments import sampling, score_cache

SAMPLE_SIZES = [25, 50, 100, 200, 500, 1000, 2000, 4000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=1000)
    ap.add_argument('--scorers', nargs='*', default=SCORERS)
    ap.add_argument('--im-size', type=int, default=DEFAULT_IM_SIZE)
    ap.add_argument('--sizes', nargs='*', type=int, default=SAMPLE_SIZES)
    args = ap.parse_args()

    rows, raw = [], []
    t0 = time.time()

    for scorer in args.scorers:
        for comparison in COMPARISONS:
            name, target, test, expected = comparison
            try:
                target_pool = sampling.resolve(target, scorer, args.im_size)
                test_pool = sampling.resolve(test, scorer, args.im_size)
            except FileNotFoundError as e:
                print(f'  SKIP {scorer:16s} {name:22s} -- {e}'.split('\n')[0])
                continue
            cap = min(len(target_pool), len(test_pool))

            for n in args.sizes:
                if n > cap:
                    continue
                rng = np.random.default_rng(seed_for(SEED, name, scorer, n))
                ks_p, cvm_p, ks_stat, kl = [], [], [], []
                for rep in range(args.repeats):
                    a, b = sampling.draw_pair(rng, target_pool, test_pool, n)
                    s, p = KS(a, b)
                    ks_stat.append(s)
                    ks_p.append(p)
                    cvm_p.append(CVM(a, b)[1])
                    kl.append(KL(a, b))
                ks_p = np.asarray(ks_p)
                cvm_p = np.asarray(cvm_p)
                k = int((ks_p < ALPHA).sum())
                lo, hi = wilson_ci(k, len(ks_p))
                rows.append(dict(
                    comparison=name, scorer=scorer, expected=expected, n=n,
                    repeats=args.repeats, im_size=args.im_size,
                    detect_KS=k / len(ks_p), detect_KS_lo=lo, detect_KS_hi=hi,
                    detect_CVM=float((cvm_p < ALPHA).mean()),
                    KS_mean=float(np.mean(ks_stat)), KS_sd=float(np.std(ks_stat)),
                    KL_mean=float(np.mean(kl)),
                    n_target_pool=len(target_pool), n_test_pool=len(test_pool)))
                raw.append(pd.DataFrame(dict(comparison=name, scorer=scorer, n=n,
                                             repeat=np.arange(len(ks_p)),
                                             KS=ks_stat, KS_p=ks_p, CVM_p=cvm_p, KL=kl)))
                sys.stdout.write(
                    f'  {scorer:16s} {name:22s} n={n:<5d} '
                    f'detect={k/len(ks_p):6.1%} [{lo:.1%}, {hi:.1%}]\n')
                sys.stdout.flush()
            pd.DataFrame(rows).to_csv(out_path('exp1_power_curve'), index=False)

    pd.DataFrame(rows).to_csv(out_path('exp1_power_curve'), index=False)
    pd.concat(raw).to_csv(out_path('exp1_power_curve_raw', 'csv.gz'),
                          index=False, compression='gzip')
    print(f'\n{len(rows)} cells in {(time.time()-t0)/60:.1f} min '
          f"-> {out_path('exp1_power_curve')}")


if __name__ == '__main__':
    main()
