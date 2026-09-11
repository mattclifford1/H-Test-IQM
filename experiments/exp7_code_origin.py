'''
EXPERIMENT 7 -- does the structure of the code come from natural images?

Pre-registered in experiments/PREREGISTRATION.md (committed before this file existed). Read
that first: it holds the predictions, the decision rules and how each outcome will be read.

exp5 showed the 64-D per-channel code detects CIFAR-10 vs CIFAR-100 at 88% where its own mean
manages 41.5% -- on ONE encoder, natural-trained mse-2. exp6 showed noise-trained encoders
match natural-trained ones -- on the SCALAR only. This crosses them. Nine encoders, all at the
64-D code:

    natural   mse-2, ssim-2, nlpd-2
    noise     mse-2-u, ssim-2-u, nlpd-2-u          (trained on uniform noise)
    random    the same architecture UNTRAINED, seeds 0, 1, 2

The random group is the null: an untrained encoder's code is the sign pattern of a random
convolutional projection, i.e. "architecture + quantiser, no learning".

Parts, in the order they run:

    checks    the three pre-registered checks -- pipeline identity, non-degenerate random
              codes -- written to exp7_checks.csv. Calibration (check 3) is part `control`.
    control   64-D C2ST on the disjoint control, n=1000, 1000 repeats, every encoder; and the
              scalar KS control at n=4000, 1000 repeats (the ssim-2-u decision rule)
    primary   64-D C2ST, CIFAR-10 vs CIFAR-100 and one-class, n in {250, 500, 1000}, 200 reps
    classdrop 64-D C2ST (and scalar KS) at k=9 classes, n=2000, 500 draws over the 10 subsets
    scalar    scalar KS, CIFAR-10 vs CIFAR-100, n=4000, 100 reps -- exp6 plus a random baseline

The scalar for every encoder is the row mean of its 64-D code. That is identical to the cached
scalar by construction (every channel has the same number of spatial elements), and check 1
verifies it on the checkpoints that have both, which is what licenses using it for the random
encoders that have no cached scalar.

Output: results/exp7_code_origin.csv  one row per (part, comparison, encoder, test, n)
        results/exp7_checks.csv       the pre-registered checks

Usage:  python -m experiments.exp7_code_origin [--parts checks control primary classdrop scalar]
'''
import sys
import time
import argparse

import numpy as np
import pandas as pd

from h_test_IQM.pipeline.h_tests import KS
from h_test_IQM.pipeline.multivariate import c2st
from experiments.common import (COMPARISONS, ALPHA, SEED, out_path, wilson_ci, seed_for)
from experiments import sampling, score_cache

IM_SIZE = 256

GROUPS = {
    'natural': ['entropy-2-mse-64d', 'entropy-2-ssim-64d', 'entropy-2-nlpd-64d'],
    'noise': ['entropy-2-mse-u-64d', 'entropy-2-ssim-u-64d', 'entropy-2-nlpd-u-64d'],
    'random': ['entropy-2-random-s0-64d', 'entropy-2-random-s1-64d',
               'entropy-2-random-s2-64d'],
}
ENCODERS = [e for g in GROUPS.values() for e in g]
GROUP_OF = {e: g for g, es in GROUPS.items() for e in es}
# the objective each encoder was trained with, so natural and noise can be paired
OBJECTIVE = {e: e.split('-')[2] for e in GROUPS['natural'] + GROUPS['noise']}
OBJECTIVE.update({e: 'none' for e in GROUPS['random']})

# the checkpoints that also have a cached scalar, for check 1
SCALAR_OF = {e: e[:-len('-64d')] for e in GROUPS['natural'] + GROUPS['noise']}

REFERENCE = 'jpeg_bytes'
COMP = {c[0]: c for c in COMPARISONS}

NUM_CLASSES = 10
CLASSDROP_TARGET = dict(dataset='CIFAR_10', labels='all', partition='a')


def _pools(comparison, encoder, scalar=False):
    _, target, test, _ = comparison
    tp = sampling.resolve(target, encoder, IM_SIZE)
    sp = sampling.resolve(test, encoder, IM_SIZE)
    if scalar and tp.ndim == 2:
        tp, sp = tp.mean(axis=1), sp.mean(axis=1)
    return tp, sp


def _run_cell(rng, tp, sp, n, repeats, test):
    '''detection rate of `test` over `repeats` draws of size n from each pool'''
    hits, effects = 0, []
    for _ in range(repeats):
        a, b = sampling.draw_pair(rng, tp, sp, n)
        if test == 'KS':
            stat, p = KS(a, b)
        else:
            stat, p, _ = c2st(a, b, seed=int(rng.integers(2**31)))
        hits += p < ALPHA
        effects.append(stat)
    return hits, effects


def _row(part, comparison, expected, encoder, representation, test, n, hits, repeats,
         effects, **extra):
    lo, hi = wilson_ci(hits, repeats)
    return dict(part=part, comparison=comparison, expected=expected, encoder=encoder,
                group=GROUP_OF.get(encoder, 'reference'),
                objective=OBJECTIVE.get(encoder, 'reference'),
                representation=representation, test=test, n=n, repeats=repeats,
                im_size=IM_SIZE, detect=hits / repeats, detect_lo=lo, detect_hi=hi,
                effect_mean=float(np.mean(effects)), effect_sd=float(np.std(effects)),
                **extra)


def _say(r):
    sys.stdout.write(
        f'  {r["part"]:9s} {r["comparison"]:20s} {r["encoder"]:26s} {r["representation"]:7s}'
        f' {r["test"]:5s} n={r["n"]:<5d} detect={r["detect"]:6.1%} '
        f'[{r["detect_lo"]:.3f}, {r["detect_hi"]:.3f}]  effect={r["effect_mean"]:.4f}\n')
    sys.stdout.flush()


# --- parts ------------------------------------------------------------------------------------

def part_checks():
    '''checks 1 and 2 of the pre-registration'''
    rows = []
    for enc in ENCODERS:
        for dataset in ('CIFAR_10', 'CIFAR_100'):
            code = score_cache.load(dataset, enc, IM_SIZE)['scores']
            row = dict(encoder=enc, group=GROUP_OF[enc], dataset=dataset, n=len(code),
                       mean_occupancy=float(code.mean()),
                       min_channel_occupancy=float(code.mean(axis=0).min()),
                       max_channel_occupancy=float(code.mean(axis=0).max()),
                       constant_channels=int((code.std(axis=0) == 0).sum()),
                       identity_max_abs_diff=np.nan)
            if enc in SCALAR_OF:
                scalar = score_cache.load(dataset, SCALAR_OF[enc], IM_SIZE)['scores']
                row['identity_max_abs_diff'] = float(np.abs(code.mean(axis=1) - scalar).max())
            rows.append(row)
            print(f'  check  {enc:26s} {dataset:10s} occupancy={row["mean_occupancy"]:.3f} '
                  f'constant_channels={row["constant_channels"]:2d} '
                  f'identity_diff={row["identity_max_abs_diff"]:.2e}')
    return pd.DataFrame(rows)


def part_control(repeats_c2st=1000, repeats_ks=1000):
    '''check 3 -- calibration -- and the ssim-2-u scalar decision rule'''
    rows = []
    comp = COMP['control-disjoint']
    for enc in ENCODERS:
        rng = np.random.default_rng(seed_for(SEED, 'exp7', 'control', enc, 'c2st'))
        tp, sp = _pools(comp, enc)
        hits, eff = _run_cell(rng, tp, sp, 1000, repeats_c2st, 'C2ST')
        rows.append(_row('control', comp[0], comp[3], enc, '64d', 'C2ST', 1000, hits,
                         repeats_c2st, eff))
        _say(rows[-1])
    for enc in ENCODERS + [REFERENCE]:
        rng = np.random.default_rng(seed_for(SEED, 'exp7', 'control', enc, 'ks'))
        tp, sp = _pools(comp, enc, scalar=True)
        hits, eff = _run_cell(rng, tp, sp, 4000, repeats_ks, 'KS')
        rows.append(_row('control', comp[0], comp[3], enc, 'scalar', 'KS', 4000, hits,
                         repeats_ks, eff))
        _say(rows[-1])
    return rows


def part_primary(sizes=(250, 500, 1000), repeats=200):
    rows = []
    for name in ('cifar10-vs-cifar100', 'cifar-vs-oneclass'):
        comp = COMP[name]
        for n in sizes:
            for enc in ENCODERS:
                rng = np.random.default_rng(seed_for(SEED, 'exp7', 'primary', name, enc, n))
                tp, sp = _pools(comp, enc)
                hits, eff = _run_cell(rng, tp, sp, n, repeats, 'C2ST')
                rows.append(_row('primary', name, comp[3], enc, '64d', 'C2ST', n, hits,
                                 repeats, eff))
                _say(rows[-1])
            rng = np.random.default_rng(seed_for(SEED, 'exp7', 'primary', name, REFERENCE, n))
            tp, sp = _pools(comp, REFERENCE)
            hits, eff = _run_cell(rng, tp, sp, n, repeats, 'KS')
            rows.append(_row('primary', name, comp[3], REFERENCE, 'scalar', 'KS', n, hits,
                             repeats, eff))
            _say(rows[-1])
    return rows


def part_classdrop(n=2000, draws=500):
    '''k = 9: drop one CIFAR-10 class from the test side. 10 subsets, draws spread evenly.'''
    rows = []
    per_subset = draws // NUM_CLASSES
    arms = ([(enc, '64d', 'C2ST') for enc in ENCODERS]
            + [(enc, 'scalar', 'KS') for enc in ENCODERS]
            + [(REFERENCE, 'scalar', 'KS')])
    for enc, rep, test in arms:
        rng = np.random.default_rng(seed_for(SEED, 'exp7', 'classdrop', enc, rep, test))
        target = sampling.resolve(CLASSDROP_TARGET, enc, IM_SIZE)
        hits, eff = 0, []
        for dropped in range(NUM_CLASSES):
            kept = [c for c in range(NUM_CLASSES) if c != dropped]
            spec = dict(dataset='CIFAR_10', labels=kept, partition='b')
            test_pool = sampling.resolve(spec, enc, IM_SIZE)
            tp, sp = target, test_pool
            if rep == 'scalar' and tp.ndim == 2:
                tp, sp = tp.mean(axis=1), sp.mean(axis=1)
            h, e = _run_cell(rng, tp, sp, n, per_subset, test)
            hits += h
            eff += e
        rows.append(_row('classdrop', 'cifar-k9-of-10', 'reject', enc, rep, test, n, hits,
                         per_subset * NUM_CLASSES, eff))
        _say(rows[-1])
    return rows


def part_scalar(n=4000, repeats=100):
    rows = []
    comp = COMP['cifar10-vs-cifar100']
    for enc in ENCODERS + [REFERENCE]:
        rng = np.random.default_rng(seed_for(SEED, 'exp7', 'scalar', enc))
        tp, sp = _pools(comp, enc, scalar=True)
        hits, eff = _run_cell(rng, tp, sp, n, repeats, 'KS')
        rows.append(_row('scalar', comp[0], comp[3], enc, 'scalar', 'KS', n, hits,
                         repeats, eff))
        _say(rows[-1])
    return rows


PARTS = {'control': part_control, 'primary': part_primary,
         'classdrop': part_classdrop, 'scalar': part_scalar}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', nargs='*', default=['checks'] + list(PARTS))
    args = ap.parse_args()

    t0 = time.time()
    if 'checks' in args.parts:
        checks = part_checks()
        checks.to_csv(out_path('exp7_checks'), index=False)

    # merge with any rows already on disk from parts not re-run this time, so one part can be
    # repeated without discarding the others
    try:
        old = pd.read_csv(out_path('exp7_code_origin'))
        old = old[~old['part'].isin(args.parts)]
        rows = old.to_dict('records')
    except FileNotFoundError:
        rows = []

    for part in args.parts:
        if part == 'checks':
            continue
        print(f'\n=== {part}')
        rows += PARTS[part]()
        pd.DataFrame(rows).to_csv(out_path('exp7_code_origin'), index=False)

    print(f'\n{len(rows)} cells in {(time.time() - t0) / 60:.1f} min '
          f'-> {out_path("exp7_code_origin")}')


if __name__ == '__main__':
    main()
