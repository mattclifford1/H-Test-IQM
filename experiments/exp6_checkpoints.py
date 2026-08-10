'''
EXPERIMENT 6 -- does the training objective matter? All six usable autoencoder checkpoints.

Every result in this project uses ONE checkpoint, `mse-2`, and nothing has ever checked whether
that choice matters. models/save_nets has fifteen. Six of them work as a scalar scorer:

    {mse, ssim, nlpd} x {natural, uniform}, centres = 2

The other nine are excluded for stated reasons: centres=5 has a bare TODO in
counts_per_emb_feature_flat and returns raw embeddings rather than a ratio (FINDINGS.md 3.10),
and models/README.md says the `mae` weights do not work.

The interesting axis is `-u`: those encoders were trained on UNIFORM NOISE, not natural images.
The project's premise is that an autoencoder trained to reconstruct natural images has absorbed
natural-image statistics, and that this is why its code occupancy is perceptually meaningful. If
the noise-trained encoders discriminate just as well, that premise is wrong -- the statistic
would be measuring something about the architecture and the quantiser, not about natural images.

That is not an idle worry: the sibling project ~/projects/percept_reduce found uniform-trained
encoders recovering ~91% of MSE-trained performance on a downstream probe.

Output: results/exp6_checkpoints.csv -- one row per (comparison, checkpoint).

Usage:  python -m experiments.exp6_checkpoints [--repeats 100]
'''
import sys
import time
import argparse

import numpy as np
import pandas as pd

from h_test_IQM.pipeline.h_tests import KS, KL
from experiments.common import (COMPARISONS, AE_CHECKPOINTS, ALPHA, SEED,
                                DEFAULT_IM_SIZE, out_path, wilson_ci, seed_for)
from experiments import sampling

# for reference in the same table -- the baseline the autoencoders have to beat
REFERENCE = ['jpeg_bytes']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4000)
    ap.add_argument('--repeats', type=int, default=100)
    ap.add_argument('--im-size', type=int, default=DEFAULT_IM_SIZE)
    ap.add_argument('--scorers', nargs='*', default=AE_CHECKPOINTS + REFERENCE)
    args = ap.parse_args()

    rows = []
    t0 = time.time()

    for scorer in args.scorers:
        trained_on = 'uniform noise' if scorer.endswith('-u') else 'natural images'
        for name, target, test, expected in COMPARISONS:
            try:
                tp = sampling.resolve(target, scorer, args.im_size)
                sp = sampling.resolve(test, scorer, args.im_size)
            except FileNotFoundError as e:
                print(f'  SKIP {scorer:18s} {name:22s} -- {str(e).splitlines()[0]}')
                continue
            n = min(args.n, len(tp), len(sp))
            rng = np.random.default_rng(seed_for(SEED, scorer, name, 'ckpt'))

            hits, ks, kls = 0, [], []
            for _ in range(args.repeats):
                a, b = sampling.draw_pair(rng, tp, sp, n)
                stat, p = KS(a, b)
                hits += p < ALPHA
                ks.append(stat)
                kls.append(KL(a, b))
            lo, hi = wilson_ci(hits, args.repeats)
            rows.append(dict(
                comparison=name, scorer=scorer, trained_on=trained_on,
                expected=expected, n=n, repeats=args.repeats, im_size=args.im_size,
                reject_rate=hits / args.repeats, reject_lo=lo, reject_hi=hi,
                KS=float(np.mean(ks)), KS_sd=float(np.std(ks)),
                KL=float(np.mean(kls))))
            sys.stdout.write(
                f'  {scorer:18s} {name:22s} KS={np.mean(ks):.4f} '
                f'reject={hits/args.repeats:5.0%}\n')
            sys.stdout.flush()
        pd.DataFrame(rows).to_csv(out_path('exp6_checkpoints'), index=False)

    pd.DataFrame(rows).to_csv(out_path('exp6_checkpoints'), index=False)
    print(f'\n{len(rows)} cells in {(time.time()-t0)/60:.1f} min '
          f"-> {out_path('exp6_checkpoints')}")


if __name__ == '__main__':
    main()
