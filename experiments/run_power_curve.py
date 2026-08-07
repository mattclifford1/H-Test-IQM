'''
Detection power vs sample size.

With n=4000 per side a KS test rejects almost anything real, so a single p-value says little
beyond "yes, these differ". The question a data curator actually has is:

    how many images do I need before this method notices a shift of this size?

So: sweep n, repeat each cell over several seeds, and record the fraction of repeats that
reject at alpha=0.05. That is the detection rate -- statistical power. The control row is the
false-positive rate and MUST sit near alpha; if it does not, the test is miscalibrated and
every other row is meaningless.

This turns the method into the data-collection stopping criterion the drafts propose, and it
is the figure that makes the approach useful rather than merely correct. See FINDINGS.md 5.2.9.

Results go to results/power_curve.csv -- one row per (comparison, n, repeat).

Usage:  python experiments/run_power_curve.py [--repeats 10]
'''
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

from h_test_IQM.pipeline import get_scores
from experiments.run_density_grid import COMPARISONS, pool_size, RESULTS_DIR

SAMPLE_SIZES = [50, 100, 200, 500, 1000, 2000, 4000]
ALPHA = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=10,
                    help='independent draws per (comparison, n) cell')
    ap.add_argument('--scorer', default='entropy-2-mse')
    ap.add_argument('--im-size', type=int, default=256)
    ap.add_argument('--out', default=os.path.join(RESULTS_DIR, 'power_curve.csv'))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    preloaded, rows = {}, []

    for name, target, test, expected in COMPARISONS:
        cap = min(pool_size(target), pool_size(test))
        for n in SAMPLE_SIZES:
            if n > cap:
                continue
            t0 = time.time()
            for rep in range(args.repeats):
                try:
                    out = get_scores(
                        dataset_target=target['dataset'], dataset_test=test['dataset'],
                        target_labels=target['labels'], test_labels=test['labels'],
                        partition_target=target.get('partition'),
                        partition_test=test.get('partition'),
                        dataset_proportion_target=n / pool_size(target),
                        dataset_proportion_test=n / pool_size(test),
                        scorer=args.scorer, test=['KS', 'CVM', 'KL'],
                        im_size=(args.im_size, args.im_size),
                        device='cuda', seed=rep, _print=False,
                        preloaded_ims=preloaded)
                    preloaded = out['preloaded_ims']
                    rows.append(dict(
                        comparison=name, expected=expected, n=n, repeat=rep,
                        scorer=args.scorer,
                        n_target=len(out['scores_target']),
                        n_test=len(out['scores_test']),
                        **out['results']))
                except Exception as e:
                    rows.append(dict(comparison=name, n=n, repeat=rep,
                                     error=f'{type(e).__name__}: {e}'))
            df = pd.DataFrame(rows)
            cell = df[(df.comparison == name) & (df.n == n)]
            rate = float((cell['KS_p'] < ALPHA).mean()) if 'KS_p' in cell else float('nan')
            sys.stdout.write(
                f'{name:22s} n={n:<5d} detect={rate:5.0%} '
                f'({args.repeats} reps, {time.time()-t0:.0f}s)\n')
            sys.stdout.flush()
            df.to_csv(args.out, index=False)

    # summary: detection rate per (comparison, n)
    df = pd.DataFrame(rows)
    if 'KS_p' in df:
        piv = (df.assign(reject=df['KS_p'] < ALPHA)
                 .pivot_table(index='comparison', columns='n', values='reject'))
        print('\ndetection rate at alpha=0.05 (KS)\n')
        print((piv * 100).round(0).to_string())
        print('\ncontrol-disjoint is the false-positive rate: it should sit near 5%.')
    print(f'\nwrote {len(rows)} rows -> {args.out}')


if __name__ == '__main__':
    main()
