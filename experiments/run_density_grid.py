'''
The main experiment: {dataset comparison} x {scorer} -> divergences + hypothesis tests.

Answers two questions at once.

  1. Does the density comparison detect a distribution shift, with a real p-value?
     (Every comparison is SIZE-MATCHED here -- the original one-class experiment compared
     n=24000 against n=2400, which confounds the effect with a sample-size bias in the
     binned divergence. See FINDINGS.md 3.5.)

  2. Does the perceptual autoencoder beat statistics that know nothing about perception?
     If jpeg_bytes separates these datasets as well as entropy-2-mse does, that is the
     first thing a reviewer will notice. See FINDINGS.md 5.1.4.

Results go to results/density_grid.csv -- one row per (comparison, scorer).

Usage:  python experiments/run_density_grid.py [--n 4000] [--quick]
'''
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

from h_test_IQM.pipeline import get_scores
from h_test_IQM.datasets import TOTAL_INSTANCES
from h_test_IQM.datasets.torch_loaders import get_classes

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'results')


# each entry is (name, target spec, test spec, what we expect)
COMPARISONS = [
    ('control-disjoint',
     dict(dataset='CIFAR_10', labels='all', partition='a'),
     dict(dataset='CIFAR_10', labels='all', partition='b'),
     'accept'),
    ('cifar10-vs-cifar100',
     dict(dataset='CIFAR_10', labels='all'),
     dict(dataset='CIFAR_100', labels='all'),
     'reject'),
    ('cifar-vs-imagenet64',
     dict(dataset='CIFAR_10', labels='all'),
     dict(dataset='IMAGENET64_VAL', labels='all'),
     'reject'),
    ('cifar-vs-dtd',
     dict(dataset='CIFAR_10', labels='all'),
     dict(dataset='DTD', labels='all'),
     'reject'),
    ('cifar-vs-oneclass',
     dict(dataset='CIFAR_10', labels='all'),
     dict(dataset='CIFAR_10', labels=[1]),
     'reject'),
    ('cifar-vs-uniform',
     dict(dataset='CIFAR_10', labels='all'),
     dict(dataset='UNIFORM', labels='all'),
     'reject'),
]

SCORERS = [
    'entropy-2-mse',    # the perceptual autoencoder -- the method under test
    'jpeg_bytes',       # baseline: compressibility
    'pixel_std',        # baseline: dispersion of pixel values
    'pixel_entropy',    # baseline: grey-level histogram entropy
    'BRISQUE',          # baseline: a classical no-reference IQM (slow)
]


def pool_size(spec):
    '''how many images this spec can draw from, after label filter and partition'''
    if spec['labels'] == 'all':
        n = TOTAL_INSTANCES[spec['dataset']]
    else:
        labels = get_classes(dataset=spec['dataset'], numerical=True)
        n = sum(1 for l in labels if l in spec['labels'])
    if spec.get('partition') is not None:
        n = n // 2
    return n


def proportion_for(spec, n_wanted):
    '''the dataset_proportion that yields n_wanted images from this spec'''
    pool = pool_size(spec)
    if pool < n_wanted:
        raise ValueError(
            f"{spec['dataset']} (labels={spec['labels']}) only has {pool} images, "
            f'cannot draw {n_wanted}')
    return n_wanted / pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4000,
                    help='images per side (both sides always equal)')
    ap.add_argument('--permutations', type=int, default=1000)
    ap.add_argument('--im-size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--quick', action='store_true',
                    help='n=300, no BRISQUE, 100 permutations -- for a smoke test')
    ap.add_argument('--out', default=os.path.join(RESULTS_DIR, 'density_grid.csv'))
    args = ap.parse_args()

    scorers = SCORERS
    if args.quick:
        args.n, args.permutations = 300, 100
        scorers = [s for s in SCORERS if s != 'BRISQUE']

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    preloaded, rows = {}, []

    for scorer in scorers:
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
                    scorer=scorer, test='all', n_permutations=args.permutations,
                    im_size=(args.im_size, args.im_size),
                    device='cuda', seed=args.seed, _print=False,
                    preloaded_ims=preloaded)
                preloaded = out['preloaded_ims']
                row = dict(comparison=name, scorer=scorer, expected=expected,
                           n_target=len(out['scores_target']),
                           n_test=len(out['scores_test']),
                           im_size=args.im_size, permutations=args.permutations,
                           seed=args.seed, secs=round(time.time() - t0, 1),
                           **out['results'])
            except Exception as e:
                row = dict(comparison=name, scorer=scorer, expected=expected,
                           error=f'{type(e).__name__}: {e}')
            rows.append(row)
            got = row.get('KS_p')
            verdict = '' if got is None else (
                'REJECT' if got < 0.05 else 'accept')
            sys.stdout.write(
                f"{scorer:16s} {name:22s} "
                f"KL={row.get('KL', float('nan')):8.4f} "
                f"KS={row.get('KS', float('nan')):.4f} "
                f"p={got if got is not None else float('nan'):<10.3g} "
                f"{verdict:7s} ({row.get('secs', 0)}s)\n")
            sys.stdout.flush()
            pd.DataFrame(rows).to_csv(args.out, index=False)   # save as we go

    print(f'\nwrote {len(rows)} rows -> {args.out}')


if __name__ == '__main__':
    main()
