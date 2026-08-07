'''
Turn results/*.csv into the tables that go in the paper. Read-only -- runs nothing.

Usage:  python -m experiments.summarise
'''
import os
import numpy as np
import pandas as pd

from experiments.run_density_grid import RESULTS_DIR

ORDER = ['control-disjoint', 'cifar10-vs-cifar100', 'cifar-vs-dtd',
         'cifar-vs-imagenet64', 'cifar-vs-oneclass', 'cifar-vs-uniform']
SCORER_ORDER = ['entropy-2-mse', 'BRISQUE', 'jpeg_bytes', 'pixel_std', 'pixel_entropy']
ALPHA = 0.05


def _load(name):
    path = os.path.join(RESULTS_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def _reindex(df, cols=None):
    df = df.reindex([c for c in ORDER if c in df.index])
    if cols is not None:
        df = df[[c for c in cols if c in df.columns]]
    return df


def main():
    fmt = lambda x: f'{x:.4f}'

    d = _load('density_grid.csv')
    if d is not None:
        print('=' * 78)
        print('MAIN TABLE  (n=4000 per side, size-matched, im_size=256)')
        print('=' * 78)
        for metric in ['KS', 'KL']:
            print(f'\n-- {metric} --')
            print(_reindex(d.pivot_table(index='comparison', columns='scorer',
                                         values=metric), SCORER_ORDER)
                  .to_string(float_format=fmt))
        print('\n-- reject at alpha=0.05? (control must accept) --')
        d = d.assign(rej=d.KS_p < ALPHA)
        print(_reindex(d.pivot_table(index='comparison', columns='scorer', values='rej'),
                       SCORER_ORDER)
              .replace({1.0: 'REJECT', 0.0: 'accept'}).to_string())

    for fname, label in [('resolution_sweep.csv', 'entropy-2-mse'),
                         ('resolution_sweep_jpeg.csv', 'jpeg_bytes')]:
        r = _load(fname)
        if r is None:
            continue
        print('\n' + '=' * 78)
        print(f'RESOLUTION SWEEP -- {label}  (KS statistic)')
        print('=' * 78)
        print(_reindex(r.pivot_table(index='comparison', columns='im_size', values='KS'))
              .to_string(float_format=fmt))

    p = _load('power_curve.csv')
    if p is not None and 'KS_p' in p:
        print('\n' + '=' * 78)
        print('POWER -- detection rate at alpha=0.05, % of repeats rejecting')
        print('=' * 78)
        piv = _reindex(p.assign(rej=p.KS_p < ALPHA)
                        .pivot_table(index='comparison', columns='n', values='rej')) * 100
        print(piv.to_string(float_format=lambda x: f'{x:.0f}'))
        print('\nsmallest n reaching 80% detection:')
        for comp in piv.index:
            row = piv.loc[comp].dropna()
            hit = row[row >= 80]
            got = int(hit.index[0]) if len(hit) else None
            print(f'  {comp:22s} {got if got else ">" + str(int(row.index.max()))}')
        print('\n(control-disjoint is the false-positive rate -- it should stay near 5%)')

    c = _load('calibration.csv')
    if c is not None:
        from scipy.stats import binomtest, kstest
        print('\n' + '=' * 78)
        print('CALIBRATION -- false-positive rate under a true null')
        print('=' * 78)
        for n, cell in c.groupby('n'):
            k = int((cell.KS_p < ALPHA).sum())
            ci = binomtest(k, len(cell)).proportion_ci()
            uni = kstest(cell.KS_p, 'uniform').pvalue
            print(f'  n={n:<5d} {k:>3d}/{len(cell)} = {k/len(cell):5.1%} '
                  f'[{ci.low:.1%}, {ci.high:.1%}]   mean p={cell.KS_p.mean():.3f}   '
                  f'uniformity p={uni:.3g}')
        print('\n  nominal alpha=5%. A high mean p and a tiny uniformity p mean the test is')
        print('  CONSERVATIVE -- safe (it never over-rejects), but it costs power. Two causes')
        print('  cross over: the asymptotic KS approximation dominates at small n, ties in the')
        print('  discrete score dominate at large n. See FINDINGS.md 2.7.')


if __name__ == '__main__':
    main()
