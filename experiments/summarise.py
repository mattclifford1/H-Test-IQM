'''
Turn results/*.csv into the tables that go in the paper. Read-only -- runs nothing.

Usage:  python -m experiments.summarise           # everything that exists
        python -m experiments.summarise --only exp1 exp3
        python -m experiments.summarise > results/SUMMARY.txt
'''
import os
import argparse

import numpy as np
import pandas as pd

from experiments.common import RESULTS_DIR, ORDER_BY_DIFFICULTY, SCORERS, ALPHA

FMT = lambda x: f'{x:.4f}'
PCT = lambda x: '' if pd.isna(x) else f'{x*100:.0f}'


def _load(name, prior=False):
    path = os.path.join(RESULTS_DIR, 'prior' if prior else '', name)
    return pd.read_csv(path) if os.path.exists(path) else None


def _head(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


def _reindex(df, cols=None):
    df = df.reindex([c for c in ORDER_BY_DIFFICULTY if c in df.index])
    if cols is not None:
        df = df[[c for c in cols if c in df.columns]]
    return df


# --- the new experiments ---------------------------------------------------------------
def exp1():
    d = _load('exp1_power_curve.csv')
    if d is None:
        return
    _head('EXP 1 -- POWER: detection rate (%) at alpha=0.05')
    # exp1 reports a rate per test, so the KS column is detect_KS; every other experiment has
    # a single `detect`.
    col = 'detect' if 'detect' in d else 'detect_KS'
    for scorer in [s for s in SCORERS if s in set(d.scorer)]:
        sub = d[d.scorer == scorer]
        print(f'\n-- {scorer}  ({int(sub.repeats.iloc[0])} repeats/cell) --')
        piv = _reindex(sub.pivot_table(index='comparison', columns='n', values=col))
        print(piv.to_string(float_format=PCT))
    print('\nsmallest n reaching 80% detection:')
    print(f'  {"comparison":24s}' + ''.join(f'{s:>16s}' for s in SCORERS if s in set(d.scorer)))
    for comp in [c for c in ORDER_BY_DIFFICULTY if c in set(d.comparison)]:
        line = f'  {comp:24s}'
        for scorer in [s for s in SCORERS if s in set(d.scorer)]:
            cell = d[(d.comparison == comp) & (d.scorer == scorer)].sort_values('n')
            hit = cell[cell.detect >= 0.8] if 'detect' in cell else cell[cell.detect_KS >= 0.8]
            if len(hit):
                line += f'{int(hit.n.iloc[0]):>16d}'
            else:
                line += f'{">" + str(int(cell.n.max())) if len(cell) else "-":>16s}'
        print(line)
    print('\n  control-disjoint is the FALSE-POSITIVE rate, not power -- it must stay near 5%.')


def exp2():
    d = _load('exp2_contamination.csv')
    if d is None:
        return
    _head('EXP 2 -- CONTAMINATION: detection rate (%) vs fraction contaminated')
    for scorer in [s for s in SCORERS if s in set(d.scorer)]:
        for cont in d[d.scorer == scorer].contaminant.unique():
            sub = d[(d.scorer == scorer) & (d.contaminant == cont)]
            print(f'\n-- {scorer} contaminated with {cont} --')
            print(sub.pivot_table(index='n', columns='fraction', values='detect')
                     .to_string(float_format=PCT))
    print('\n  f=0 is the control, so that column is a false-positive rate and must sit near 5%.')
    print('\nsmallest detectable contamination at 80% power:')
    for scorer in [s for s in SCORERS if s in set(d.scorer)]:
        for cont in d[d.scorer == scorer].contaminant.unique():
            sub = d[(d.scorer == scorer) & (d.contaminant == cont)]
            bits = []
            for n, cell in sub.groupby('n'):
                hit = cell[cell.detect >= 0.8].sort_values('fraction')
                bits.append(f'n={n}: ' + (f'{hit.fraction.iloc[0]:g}' if len(hit) else '--'))
            print(f'  {scorer:16s} {cont:11s} ' + '  '.join(bits))


def exp3():
    d = _load('exp3_resolution.csv')
    if d is None:
        return
    scorers = [s for s in SCORERS if s in set(d.scorer)]
    for im_size in sorted(d.im_size.unique()):
        sub = d[d.im_size == im_size]
        _head(f'EXP 3 -- MAIN TABLE at {im_size}px  '
              f'(n={int(sub.n.max())}/side, {int(sub.repeats.iloc[0])} draws, mean KS)')
        print(_reindex(sub.pivot_table(index='comparison', columns='scorer', values='KS'),
                       scorers).to_string(float_format=FMT))
        print(f'\n-- rejection rate (% of draws with p < {ALPHA}) --')
        print(_reindex(sub.pivot_table(index='comparison', columns='scorer',
                                       values='reject_rate'), scorers)
              .to_string(float_format=PCT))

    _head('EXP 3 -- RESOLUTION: KS vs im_size')
    for scorer in scorers:
        sub = d[d.scorer == scorer]
        print(f'\n-- {scorer} --')
        print(_reindex(sub.pivot_table(index='comparison', columns='im_size', values='KS'))
              .to_string(float_format=FMT))
    print('\n  CIFAR is natively 32px. A comparison whose KS grows with im_size is being')
    print('  helped by interpolated detail, not by the data (FINDINGS.md 2.5).')


def exp4():
    d = _load('exp4_class_count.csv')
    if d is None:
        return
    _head('EXP 4 -- CLASS COUNT: test sample keeps k of 10 CIFAR-10 classes')
    print('\n-- mean KS --')
    print(d.pivot_table(index='k_classes', columns='scorer', values='KS_mean')
           .reindex(columns=[s for s in SCORERS if s in set(d.scorer)])
           .to_string(float_format=FMT))
    print('\n-- detection rate (%) --')
    print(d.pivot_table(index='k_classes', columns='scorer', values='detect')
           .reindex(columns=[s for s in SCORERS if s in set(d.scorer)])
           .to_string(float_format=PCT))
    print('\n-- spread of KS ACROSS class subsets (if this rivals the k effect, "which')
    print('   classes" matters more than "how many") --')
    print(d.pivot_table(index='k_classes', columns='scorer', values='subset_sd')
           .reindex(columns=[s for s in SCORERS if s in set(d.scorer)])
           .to_string(float_format=FMT))
    print('\n  k=10 is the control: both sides are all-class CIFAR-10, so it must sit near 5%.')


def exp5():
    d = _load('exp5_multivariate.csv')
    if d is None:
        return
    _head('EXP 5 -- MULTIVARIATE: does keeping all 64 latent channels help?')
    d = d.assign(arm=d.representation + ' / ' + d.test)
    for n in sorted(d.n.unique()):
        sub = d[d.n == n]
        print(f'\n-- detection rate (%) at n={n} --')
        print(_reindex(sub.pivot_table(index='comparison', columns='arm', values='detect'))
              .to_string(float_format=PCT))
    p = _load('exp5_multivariate_perm.csv')
    if p is not None:
        print(f'\n-- permutation tests on the 64-D code, n={int(p.n.max())} --')
        print(_reindex(p.pivot_table(index='comparison', columns='test', values='detect'))
              .to_string(float_format=PCT))
    print('\n  AE-scalar/C2ST vs AE-64d/C2ST is the comparison that matters: same test, same')
    print('  images, the only difference is whether the code was collapsed to one number.')


def exp6():
    d = _load('exp6_checkpoints.csv')
    if d is None:
        return
    _head('EXP 6 -- CHECKPOINTS: does the training objective matter?')
    order = [c for c in d.scorer.unique()]
    print('\n-- mean KS --')
    print(_reindex(d.pivot_table(index='comparison', columns='scorer', values='KS'), order)
          .to_string(float_format=FMT))
    print('\n-- rejection rate (%) --')
    print(_reindex(d.pivot_table(index='comparison', columns='scorer',
                                 values='reject_rate'), order).to_string(float_format=PCT))
    nat = d[(d.trained_on == 'natural images') & (d.scorer.str.startswith('entropy'))]
    uni = d[d.trained_on == 'uniform noise']
    if len(nat) and len(uni):
        real = [c for c in ORDER_BY_DIFFICULTY if c != 'control-disjoint']
        n_ks = nat[nat.comparison.isin(real)].groupby('comparison').KS.mean()
        u_ks = uni[uni.comparison.isin(real)].groupby('comparison').KS.mean()
        ratio = (u_ks / n_ks).mean()
        print(f'\n  noise-trained encoders retain {ratio:.0%} of the natural-trained KS on')
        print('  average across the real comparisons. If that is near 100%, the "it learned')
        print('  natural-image statistics" premise does not survive.')


# --- the first pass, kept for the record -----------------------------------------------
def prior():
    d = _load('density_grid.csv', prior=True)
    if d is None:
        return
    _head('PRIOR PASS (results/prior/) -- single-draw main table, n=4000, 256px')
    print(_reindex(d.pivot_table(index='comparison', columns='scorer', values='KS'), SCORERS)
          .to_string(float_format=FMT))
    c = _load('calibration.csv', prior=True)
    if c is not None:
        from scipy.stats import binomtest, kstest
        print('\n-- calibration, 500 null draws per n --')
        for n, cell in c.groupby('n'):
            k = int((cell.KS_p < ALPHA).sum())
            ci = binomtest(k, len(cell)).proportion_ci()
            uni = kstest(cell.KS_p, 'uniform').pvalue
            print(f'  n={n:<5d} {k:>3d}/{len(cell)} = {k/len(cell):5.1%} '
                  f'[{ci.low:.1%}, {ci.high:.1%}]   mean p={cell.KS_p.mean():.3f}   '
                  f'uniformity p={uni:.3g}')
        print('\n  A high mean p and a tiny uniformity p mean the test is CONSERVATIVE --')
        print('  safe, but it costs power. Two causes cross over: the asymptotic KS')
        print('  approximation dominates at small n, ties in the discrete score at large n.')
        print('  See FINDINGS.md 2.7.')


SECTIONS = {'exp1': exp1, 'exp2': exp2, 'exp3': exp3, 'exp4': exp4,
            'exp5': exp5, 'exp6': exp6, 'prior': prior}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', default=list(SECTIONS))
    args = ap.parse_args()
    for key in args.only:
        if key not in SECTIONS:
            raise SystemExit(f'unknown section {key!r}, expected any of {list(SECTIONS)}')
        SECTIONS[key]()


if __name__ == '__main__':
    main()
