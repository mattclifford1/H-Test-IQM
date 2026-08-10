'''
Resolution control.

pipeline/main.py used to hardcode im_size=(256, 256), so every CIFAR image was bilinearly
UPSAMPLED 8x from 32x32 before it reached the autoencoder -- which destroys exactly the
high-frequency content a compression autoencoder's code responds to. The datasets being
compared have different native resolutions (CIFAR 32, ImageNet64 64, DTD variable, uniform
noise generated at 32), so some of the measured "distribution difference" may simply be a
resolution difference. See FINDINGS.md 3.6.

This sweeps im_size and asks whether the result survives at native resolution.

Results go to results/resolution_sweep.csv.

Usage:  python experiments/run_resolution_sweep.py [--n 4000]
'''
import os
import sys
import time
import argparse
import pandas as pd

from h_test_IQM.pipeline import get_scores
from experiments.prior.run_density_grid import COMPARISONS, proportion_for, RESULTS_DIR

# 32 is CIFAR's native size; 256 is what every published number in this repo used.
# The autoencoder is fully convolutional so it accepts all of these, but its latent grid
# scales with the input (64 x H/16 x W/16), so the score is an average over 4x as many
# code positions at each doubling -- expect the variance of the score to shrink with size.
IM_SIZES = [32, 64, 128, 256]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4000)
    ap.add_argument('--permutations', type=int, default=1000)
    ap.add_argument('--scorer', default='entropy-2-mse')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(RESULTS_DIR, 'resolution_sweep.csv'))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    preloaded, rows = {}, []

    for im_size in IM_SIZES:
        for name, target, test, expected in COMPARISONS:
            t0 = time.time()
            try:
                out = get_scores(
                    dataset_target=target['dataset'], dataset_test=test['dataset'],
                    target_labels=target['labels'], test_labels=test['labels'],
                    partition_target=target.get('partition'),
                    partition_test=test.get('partition'),
                    dataset_proportion_target=proportion_for(target, args.n),
                    dataset_proportion_test=proportion_for(test, args.n),
                    scorer=args.scorer, test='all',
                    n_permutations=args.permutations,
                    im_size=(im_size, im_size),
                    device='cuda', seed=args.seed, _print=False,
                    preloaded_ims=preloaded)
                preloaded = out['preloaded_ims']
                row = dict(comparison=name, scorer=args.scorer, expected=expected,
                           im_size=im_size,
                           n_target=len(out['scores_target']),
                           n_test=len(out['scores_test']),
                           target_score_std=float(out['scores_target'].std()),
                           test_score_std=float(out['scores_test'].std()),
                           seed=args.seed, secs=round(time.time() - t0, 1),
                           **out['results'])
            except Exception as e:
                row = dict(comparison=name, scorer=args.scorer, im_size=im_size,
                           error=f'{type(e).__name__}: {e}')
            rows.append(row)
            got = row.get('KS_p')
            sys.stdout.write(
                f"im_size={im_size:<4d} {name:22s} "
                f"KL={row.get('KL', float('nan')):8.4f} "
                f"KS={row.get('KS', float('nan')):.4f} "
                f"p={got if got is not None else float('nan'):<10.3g} "
                f"{'' if got is None else ('REJECT' if got < 0.05 else 'accept')}\n")
            sys.stdout.flush()
            pd.DataFrame(rows).to_csv(args.out, index=False)

    print(f'\nwrote {len(rows)} rows -> {args.out}')


if __name__ == '__main__':
    main()
