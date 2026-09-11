'''
EXPERIMENT 7 -- post-hoc diagnostics. EXPLORATORY: none of this was pre-registered.

Written after exp7 returned two things that needed explaining before they could be written up.

1. The scalar KS control scattered in BOTH directions. At n = 4000, 1000 repeats, 6 of 10
   scorers' Wilson intervals excluded 5% -- some conservative (~3%), some anti-conservative
   (ssim-u 8.7%, nlpd 7.5%) -- while the ten averaged 5.2%. The pre-registered ssim-u rule
   fired. The suspicion is the control's design rather than the scorers: `control-disjoint`
   compares two FIXED halves of the cached pool (30k each), and every repeat re-draws from
   those same two halves. At n = 4000 a draw is 13% of its half, so the realised difference
   between the halves is a fixed offset shared by all 1000 repeats. The rate measured is then
   conditional on that one split, not the unconditional false-positive rate.

   Two tests of that:
     redraw     re-randomise the partition on every repeat -- draw 2n from the whole pool
                without replacement and split it. This is the unconditional null. If the
                explanation is right, the scatter collapses to roughly binomial around 5% (or
                slightly under -- the scalar takes discrete values, and ties make KS
                conservative).
     split_ks   the KS between the two full fixed halves, per scorer. If the explanation is
                right, scorers whose halves happen to differ more reject more often.

2. The untrained encoders were the strongest detectors. One candidate explanation is simply
   dimension: natural-trained encoders leave 2-11 channels dead, and a code with more live,
   less-correlated channels gives the classifier more to work with. `dims` reports the live
   channel count and the participation ratio (effective dimension) of each code's covariance
   on CIFAR-10, next to the C2ST accuracy on CIFAR-10 vs CIFAR-100 at n = 1000.

3. If (1) is right, the class-drop comparison inherits the same problem -- it too draws from
   fixed halves (target from half a, test from half b), at n = 2000, 6.7% of a half -- and so
   does exp4, whose k = 9 numbers the pre-registration quoted. `classdrop_redraw` re-does it
   with a fresh disjoint partition per repeat, and adds k = 10 under the same protocol so the
   comparison carries its own unconditional false-positive rate.

Output: results/exp7_diagnostics.csv

Usage:  python -m experiments.exp7_diagnostics
'''
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from h_test_IQM.pipeline.h_tests import KS
from h_test_IQM.pipeline.multivariate import c2st
from experiments.common import ALPHA, SEED, out_path, wilson_ci, seed_for
from experiments import score_cache
from experiments.exp7_code_origin import ENCODERS, GROUP_OF, REFERENCE, IM_SIZE


def _scalar_pool(scorer, dataset='CIFAR_10'):
    s = score_cache.load(dataset, scorer, IM_SIZE)['scores']
    return s.mean(axis=1) if s.ndim == 2 else s


def redraw_control(scorer, n, repeats, test, scalar):
    '''unconditional null: a fresh random partition of the whole pool on every repeat'''
    pool = score_cache.load('CIFAR_10', scorer, IM_SIZE)['scores']
    if scalar and pool.ndim == 2:
        pool = pool.mean(axis=1)
    rng = np.random.default_rng(seed_for(SEED, 'exp7diag', 'redraw', scorer, test, n))
    hits = 0
    for _ in range(repeats):
        idx = rng.choice(len(pool), size=2 * n, replace=False)
        a, b = pool[idx[:n]], pool[idx[n:]]
        if test == 'KS':
            _, p = KS(a, b)
        else:
            _, p, _ = c2st(a, b, seed=int(rng.integers(2**31)))
        hits += p < ALPHA
    return hits


def redraw_classdrop(scorer, rep, test, k, n=2000, draws=500):
    '''
    exp4 / exp7's class drop, re-done with a fresh disjoint partition on every repeat.

    The target is n rows drawn from all of CIFAR-10; the test is n rows drawn from the
    REMAINING rows, restricted to the k kept classes. k = 10 is then an unconditional null for
    this exact protocol, which the fixed-partition version never had. For k = 9 the draws are
    spread evenly over the 10 possible dropped classes.
    '''
    cached = score_cache.load('CIFAR_10', scorer, IM_SIZE)
    scores, labels = cached['scores'], cached['labels']
    if rep == 'scalar' and scores.ndim == 2:
        scores = scores.mean(axis=1)
    rng = np.random.default_rng(seed_for(SEED, 'exp7diag', 'classdrop', scorer, rep, test, k))
    dropped_list = list(range(10)) if k == 9 else [None]
    per = draws // len(dropped_list)
    hits = 0
    for dropped in dropped_list:
        for _ in range(per):
            perm = rng.permutation(len(scores))
            target, rest = perm[:n], perm[n:]
            if dropped is not None:
                rest = rest[labels[rest] != dropped]
            a, b = scores[target], scores[rest[:n]]
            if test == 'KS':
                _, p = KS(a, b)
            else:
                _, p, _ = c2st(a, b, seed=int(rng.integers(2**31)))
            hits += p < ALPHA
    return hits, per * len(dropped_list)


def participation_ratio(code):
    '''(sum of eigenvalues)^2 / sum of squared eigenvalues -- the effective dimension'''
    ev = np.clip(np.linalg.eigvalsh(np.cov(code, rowvar=False)), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def main():
    t0 = time.time()
    fixed = pd.read_csv(out_path('exp7_code_origin'))
    rows = []

    # --- 1. the control: fixed split vs redrawn split -------------------------------------
    for scorer in ENCODERS + [REFERENCE]:
        for rep, test, n, repeats in (('scalar', 'KS', 4000, 1000), ('64d', 'C2ST', 1000, 500)):
            if rep == '64d' and scorer == REFERENCE:
                continue
            hits = redraw_control(scorer, n, repeats, test, scalar=(rep == 'scalar'))
            lo, hi = wilson_ci(hits, repeats)
            f = fixed[(fixed.part == 'control') & (fixed.encoder == scorer)
                      & (fixed.representation == rep)].iloc[0]
            # realised KS between the two fixed halves (scalar only -- that is where the
            # conditional offset was seen)
            split_ks = np.nan
            if rep == 'scalar':
                pool = _scalar_pool(scorer)
                half = len(pool) // 2
                split_ks = float(KS(pool[:half], pool[half:])[0])
            rows.append(dict(diagnostic='control', scorer=scorer,
                             group=GROUP_OF.get(scorer, 'reference'), representation=rep,
                             test=test, n=n, repeats=repeats,
                             fixed_fp=f.detect, fixed_lo=f.detect_lo, fixed_hi=f.detect_hi,
                             redraw_fp=hits / repeats, redraw_lo=lo, redraw_hi=hi,
                             split_ks=split_ks))
            r = rows[-1]
            sys.stdout.write(f'  control {scorer:26s} {rep:6s} fixed={r["fixed_fp"]:6.1%} '
                             f'redraw={r["redraw_fp"]:6.1%} [{lo:.3f}, {hi:.3f}] '
                             f'split_ks={split_ks:.4f}\n')
            sys.stdout.flush()

    ctl = pd.DataFrame(rows)
    sc = ctl[ctl.representation == 'scalar']
    rho, p = spearmanr(sc.split_ks, sc.fixed_fp)
    print(f'\n  scalar: spearman(split_ks, fixed_fp) = {rho:.2f} (p = {p:.3g}, '
          f'{len(sc)} scorers)')
    for name, col in (('fixed', 'fixed'), ('redraw', 'redraw')):
        miss = ((sc[f'{col}_lo'] > ALPHA) | (sc[f'{col}_hi'] < ALPHA)).sum()
        print(f'  scalar {name:6s}: mean FP {sc[f"{col}_fp"].mean():.3%}, sd across scorers '
              f'{sc[f"{col}_fp"].std():.3%}, intervals excluding 5%: {miss}/{len(sc)}')

    # --- 2. dimension vs power --------------------------------------------------------------
    prim = fixed[(fixed.part == 'primary') & (fixed.comparison == 'cifar10-vs-cifar100')
                 & (fixed.n == 1000) & (fixed.representation == '64d')]
    for enc in ENCODERS:
        code = score_cache.load('CIFAR_10', enc, IM_SIZE)['scores']
        live = int((code.std(axis=0) > 0).sum())
        pr = participation_ratio(code[:, code.std(axis=0) > 0])
        acc = float(prim[prim.encoder == enc].effect_mean.iloc[0])
        rows.append(dict(diagnostic='dims', scorer=enc, group=GROUP_OF[enc],
                         live_channels=live, participation_ratio=pr, c2st_acc_n1000=acc))
        print(f'  dims   {enc:26s} live={live:2d} participation_ratio={pr:6.2f} '
              f'c2st_acc={acc:.4f}')
    dims = pd.DataFrame([r for r in rows if r['diagnostic'] == 'dims'])
    rho, p = spearmanr(dims.participation_ratio, dims.c2st_acc_n1000)
    print(f'\n  spearman(participation_ratio, c2st_acc) = {rho:.2f} (p = {p:.3g}, 9 encoders)')

    # --- 3. the class drop, with the partition redrawn on every repeat -----------------------
    arms = ([(e, '64d', 'C2ST') for e in ENCODERS] + [(e, 'scalar', 'KS') for e in ENCODERS]
            + [(REFERENCE, 'scalar', 'KS')])
    for scorer, rep, test in arms:
        for k in (9, 10):
            hits, total = redraw_classdrop(scorer, rep, test, k)
            lo, hi = wilson_ci(hits, total)
            rows.append(dict(diagnostic='classdrop_redraw', scorer=scorer,
                             group=GROUP_OF.get(scorer, 'reference'), representation=rep,
                             test=test, n=2000, k_classes=k, repeats=total,
                             redraw_fp=hits / total if k == 10 else np.nan,
                             redraw_detect=hits / total if k == 9 else np.nan,
                             redraw_lo=lo, redraw_hi=hi))
            print(f'  classdrop {scorer:26s} {rep:6s} k={k:<2d} rate={hits / total:6.1%} '
                  f'[{lo:.3f}, {hi:.3f}]')
            sys.stdout.flush()

    pd.DataFrame(rows).to_csv(out_path('exp7_diagnostics'), index=False)
    print(f'\ndone in {(time.time() - t0) / 60:.1f} min -> {out_path("exp7_diagnostics")}')


if __name__ == '__main__':
    main()
