'''
False-positive rate of the test under the null.

The power curve measures the control over only 10 repeats per cell, which cannot resolve a 5%
rejection rate -- 0/10 and 2/10 are both perfectly consistent with alpha=0.05. Calibration is
the one thing that has to be right before any p-value in the paper means anything, so it gets
its own run with enough repeats to actually measure it.

Draws BOTH sides from disjoint halves of CIFAR-10, so the null is true by construction, and
reports the observed rejection rate with a binomial confidence interval. If KS_p is uniform on
[0,1] under the null -- which is what "calibrated" means -- the rate should sit at alpha.

Results go to results/calibration.csv.

Usage:  python -m experiments.run_calibration [--repeats 500]
'''
import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import binomtest, kstest

from h_test_IQM.pipeline import get_scores
from experiments.prior.run_density_grid import RESULTS_DIR

SAMPLE_SIZES = [200, 1000, 4000]
ALPHA = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=500)
    ap.add_argument('--scorer', default='entropy-2-mse')
    ap.add_argument('--im-size', type=int, default=256)
    ap.add_argument('--out', default=os.path.join(RESULTS_DIR, 'calibration.csv'))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pool_half = 60000 // 2
    preloaded, rows = {}, []

    for n in SAMPLE_SIZES:
        for rep in range(args.repeats):
            out = get_scores(
                dataset_target='CIFAR_10', dataset_test='CIFAR_10',
                partition_target='a', partition_test='b',
                dataset_proportion_target=n / pool_half,
                dataset_proportion_test=n / pool_half,
                scorer=args.scorer, test=['KS', 'CVM'],
                im_size=(args.im_size, args.im_size),
                device='cuda', seed=rep, _print=False, preloaded_ims=preloaded)
            preloaded = out['preloaded_ims']
            rows.append(dict(n=n, repeat=rep, scorer=args.scorer, **out['results']))

        df = pd.DataFrame(rows)
        cell = df[df.n == n]
        k = int((cell.KS_p < ALPHA).sum())
        ci = binomtest(k, len(cell)).proportion_ci()
        # is the p-value itself uniform under the null? the stronger check
        uni = kstest(cell.KS_p, 'uniform').pvalue
        sys.stdout.write(
            f'n={n:<5d} false positives {k}/{len(cell)} = {k/len(cell):.1%} '
            f'[{ci.low:.1%}, {ci.high:.1%}]   p-value uniformity KS p={uni:.3f}\n')
        sys.stdout.flush()
        df.to_csv(args.out, index=False)

    print(f'\nnominal alpha = {ALPHA:.0%}. The interval should contain it, and the uniformity '
          f'p-value should NOT be small.\nwrote {len(rows)} rows -> {args.out}')


if __name__ == '__main__':
    main()
