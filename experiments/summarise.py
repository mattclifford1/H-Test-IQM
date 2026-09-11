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


def exp7():
    d = _load('exp7_code_origin.csv')
    if d is None:
        return
    _head('EXP 7 -- CODE ORIGIN: natural vs noise-trained vs untrained, at the 64-D code')
    groups = ['natural', 'noise', 'random']
    short = lambda e: e.replace('entropy-2-', '').replace('-64d', '')

    # the pre-registered primary comparison
    prim = d[(d.part == 'primary') & (d.representation == '64d')]
    for comp in ('cifar10-vs-cifar100', 'cifar-vs-oneclass'):
        cell = prim[prim.comparison == comp]
        print(f'\n-- {comp}: detection rate (%), 64-D C2ST, 200 repeats --')
        t = cell.pivot_table(index='encoder', columns='n', values='detect')
        t.index = [short(e) for e in t.index]
        print(t.to_string(float_format=PCT))
        g = cell.groupby(['group', 'n']).detect.mean().unstack()
        print('   group means:')
        print(g.reindex(groups).to_string(float_format=PCT))

    # P1: natural vs noise, objective-matched pairs, at n = 1000
    c = prim[(prim.comparison == 'cifar10-vs-cifar100') & (prim.n == 1000)]
    nat = c[c.group == 'natural'].set_index('objective').detect
    noi = c[c.group == 'noise'].set_index('objective').detect
    ran = c[c.group == 'random'].set_index('encoder').detect
    print('\n-- the registered predictions, CIFAR-10 vs CIFAR-100, n = 1000 --')
    gap = nat - noi
    print(f'  P1 natural - noise: group means {nat.mean():.1%} vs {noi.mean():.1%} '
          f'({100 * (nat.mean() - noi.mean()):+.1f} pts); per objective '
          + ', '.join(f'{o} {100 * v:+.1f}' for o, v in gap.items()))
    print(f'  P2 natural - random: {nat.mean():.1%} vs {ran.mean():.1%} '
          f'({100 * (nat.mean() - ran.mean()):+.1f} pts); random seeds '
          + ', '.join(f'{short(e)} {v:.1%}' for e, v in ran.items()))
    c5 = prim[(prim.comparison == 'cifar10-vs-cifar100') & (prim.n == 500)]
    print('  (n = 1000 saturates the random group; at n = 500 the group means are '
          + ', '.join(f'{g} {c5[c5.group == g].detect.mean():.1%}' for g in groups) + ')')

    # calibration and the rest
    ctl = d[d.part == 'control']
    print('\n-- control-disjoint false-positive rate (%), fixed partitions --')
    t = ctl.pivot_table(index='encoder', columns=['representation', 'test', 'n'],
                        values='detect')
    t.index = [short(e) for e in t.index]
    print(t.to_string(float_format=lambda x: f'{x*100:.1f}'))

    cd = d[d.part == 'classdrop']
    print('\n-- k = 9 class drop, n = 2000, fixed partitions: detection (%) --')
    t = cd.pivot_table(index='encoder', columns='representation', values='detect')
    t.index = [short(e) for e in t.index]
    print(t.to_string(float_format=lambda x: f'{x*100:.1f}'))

    sc = d[d.part == 'scalar']
    print('\n-- scalar KS, CIFAR-10 vs CIFAR-100, n = 4000: detection (%) and mean KS --')
    t = sc.set_index('encoder')[['detect', 'effect_mean']]
    t.index = [short(e) for e in t.index]
    print(t.to_string(formatters={'detect': lambda x: f'{x*100:.0f}',
                                  'effect_mean': FMT}))

    x = _load('exp7_diagnostics.csv')
    if x is None:
        return
    _head('EXP 7 -- POST-HOC DIAGNOSTICS (exploratory, not pre-registered)')
    ctl = x[x.diagnostic == 'control']
    print('\n-- control: fixed partition vs a partition redrawn every repeat (FP %) --')
    t = ctl.set_index(['scorer', 'representation'])[['fixed_fp', 'redraw_fp', 'split_ks']]
    print(t.to_string(formatters={'fixed_fp': lambda v: f'{v*100:.1f}',
                                  'redraw_fp': lambda v: f'{v*100:.1f}',
                                  'split_ks': lambda v: '' if pd.isna(v) else f'{v:.4f}'}))
    dims = x[x.diagnostic == 'dims'].set_index('scorer')[
        ['live_channels', 'participation_ratio', 'c2st_acc_n1000']]
    print('\n-- effective dimension of each code (CIFAR-10) vs C2ST accuracy --')
    print(dims.to_string(float_format=lambda v: f'{v:.3f}'))
    cr = x[x.diagnostic == 'classdrop_redraw']
    if len(cr):
        print('\n-- class drop with the partition redrawn every repeat (%) --')
        cr = cr.assign(rate=cr.redraw_detect.fillna(cr.redraw_fp))
        t = cr.pivot_table(index='scorer', columns=['representation', 'k_classes'],
                           values='rate')
        print(t.to_string(float_format=lambda v: f'{v*100:.1f}'))


def exp8():
    d = _load('exp8_featuriser_ladder.csv')
    if d is None:
        return
    _head('EXP 8 -- FEATURISER LADDER: what is the untrained encoder detecting?')
    order = ['ref', '2', '1', '3q', '4', '4q', '5', '3', '5m', '6', '6m']
    names = d.drop_duplicates('rung').set_index('rung').rung_name

    nul = d[d.part == 'null_redrawn']
    print('\n-- null, CIFAR-10 vs CIFAR-10, redrawn partition, n = 1000, 1000 repeats --')
    for _, r in nul.iterrows():
        flag = '' if r.detect_lo <= ALPHA <= r.detect_hi else '   <-- excludes 5%'
        print(f'  rung {r.rung:3s} {r.member:8s} {r.detect:6.1%} '
              f'[{r.detect_lo:.4f}, {r.detect_hi:.4f}]{flag}')

    pr = d[d.part == 'primary']
    print('\n-- CIFAR-10 vs CIFAR-100: detection rate (%), rung means (members in brackets) --')
    for rung in order:
        cell = pr[pr.rung == rung]
        if not len(cell):
            continue
        line = f'  {rung:3s} {names[rung]:30s}'
        for n, c in cell.groupby('n'):
            mem = '/'.join(f'{v*100:.0f}' for v in c.detect)
            line += f'  n={n:<4d} {c.detect.mean()*100:5.1f}' + (f' ({mem})' if len(c) > 1 else '')
        print(line)

    print('\n-- C2ST accuracy on CIFAR-10 vs CIFAR-100 (rung means) --')
    t = pr[pr.test == 'C2ST'].groupby(['rung', 'n']).effect_mean.mean().unstack()
    print(t.reindex([r for r in order if r in t.index]).to_string(float_format=FMT))

    cd = d[d.part == 'classdrop']
    print('\n-- class drop, n = 2000, redrawn: detection at k = 9 and the k = 10 null (%) --')
    for rung in order:
        c9 = cd[(cd.rung == rung) & (cd.k_classes == 9)]
        c10 = cd[(cd.rung == rung) & (cd.k_classes == 10)]
        if not len(c9):
            continue
        print(f'  {rung:3s} {names[rung]:30s} k=9 {c9.detect.mean()*100:5.1f} '
              f'[{c9.detect_lo.min()*100:.1f}, {c9.detect_hi.max()*100:.1f}]'
              f'  k=10 {c10.detect.mean()*100:4.1f}')

    # the registered predictions, at n = 500
    m = pr[pr.n == 500].groupby('rung').detect
    lo, hi = m.min(), m.max()
    mean = m.mean()

    def ahead(a, b):
        gap = 100 * (mean[a] - mean[b])
        every = lo[a] > hi[b]
        return f'{mean[a]:.1%} vs {mean[b]:.1%} ({gap:+.1f} pts), every member ahead: {every}'
    print('\n-- the registered predictions, n = 500 --')
    print('  P1  rung 6 vs 3   :', ahead('6', '3'), '| matched 64-D 6m vs 3:', ahead('6m', '3'))
    print('  P2  rung 3 vs 1   :', ahead('3', '1'))
    print('  P3  rung 3 vs 2   :', ahead('3', '2'))
    print('  P4  rung 6 vs 5   :', ahead('6', '5'), '| matched 64-D 6m vs 5m:', ahead('6m', '5m'))
    print('  Q   quantiser     : untrained occupancy vs activation', ahead('3', '3q'))
    print('                      natural occupancy vs activation  ', ahead('4', '4q'))
    j = cd[(cd.rung == 'ref') & (cd.k_classes == 9)].iloc[0]
    r6 = cd[(cd.rung == '6') & (cd.k_classes == 9)].iloc[0]
    print(f'  P5  class drop    : rung 6 {r6.detect:.1%} [{r6.detect_lo:.3f}, {r6.detect_hi:.3f}] '
          f'vs JPEG {j.detect:.1%} [{j.detect_lo:.3f}, {j.detect_hi:.3f}] '
          f'(registered JPEG reference 32.2% [0.283, 0.364])')


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
            'exp5': exp5, 'exp6': exp6, 'exp7': exp7, 'exp8': exp8, 'prior': prior}


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
