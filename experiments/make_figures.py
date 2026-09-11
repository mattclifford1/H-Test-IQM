'''
Every figure, regenerated from results/*.csv. Read-only over the results; runs no experiments.

Before this existed, every plot in the drafts was made by hand in a notebook and could not be
regenerated (FINDINGS.md 5.4). Each figure below is written to figures/ as both PNG (for
looking at) and PDF (for LaTeX).

  fig1_power_curve        detection rate vs n. THE headline figure.
  fig2_contamination      detection rate over (contamination fraction x n).
  fig3_ecdf               the score distributions as ECDFs, with the KS statistic drawn on.
  fig4_scorer_comparison  every scorer on every comparison, against the control's null band.
  fig5_resolution         how much of the effect is the 256x256 upsample.
  fig6_calibration        null p-values against uniform -- the honesty check.

Usage:  python -m experiments.make_figures [--only fig3]
'''
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiments.common import (RESULTS_DIR, FIGURES_DIR, ORDER_BY_DIFFICULTY, SCORERS,
                                ALPHA, COMPARISONS, DEFAULT_IM_SIZE, fig_path)

plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'font.size': 9,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
})

# one colour per comparison, kept identical across every figure
COMPARISON_COLOURS = {
    'control-disjoint': '#888888',
    'cifar10-vs-cifar100': '#1f77b4',
    'cifar-vs-dtd': '#2ca02c',
    'cifar-vs-imagenet64': '#ff7f0e',
    'cifar-vs-oneclass': '#d62728',
    'cifar-vs-uniform': '#9467bd',
}
SCORER_COLOURS = {
    'entropy-2-mse': '#d62728',     # the method under test -- always red
    'jpeg_bytes': '#1f77b4',
    'pixel_std': '#2ca02c',
    'pixel_entropy': '#ff7f0e',
    'BRISQUE': '#9467bd',
}
NICE = {
    'control-disjoint': 'control (disjoint halves)',
    'cifar10-vs-cifar100': 'CIFAR-10 vs CIFAR-100',
    'cifar-vs-dtd': 'CIFAR-10 vs DTD',
    'cifar-vs-imagenet64': 'CIFAR-10 vs ImageNet64',
    'cifar-vs-oneclass': 'CIFAR-10 vs one class',
    'cifar-vs-uniform': 'CIFAR-10 vs uniform noise',
    'entropy-2-mse': 'autoencoder (entropy-2-mse)',
    'jpeg_bytes': 'JPEG bytes/pixel',
    'pixel_std': 'pixel std',
    'pixel_entropy': 'pixel entropy',
    'BRISQUE': 'BRISQUE',
}


def _load(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        path = os.path.join(RESULTS_DIR, 'prior', name)
    return pd.read_csv(path) if os.path.exists(path) else None


def _save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(fig_path(f'{name}.{ext}'))
    plt.close(fig)
    print(f'  wrote figures/{name}.png / .pdf')


# --- fig 1 -------------------------------------------------------------------------------
def fig1_power_curve():
    d = _load('exp1_power_curve.csv')
    if d is None:
        print('  skip fig1 -- no exp1_power_curve.csv')
        return
    scorers = [s for s in SCORERS if s in set(d.scorer)]
    fig, axes = plt.subplots(1, len(scorers), figsize=(3.2 * len(scorers), 3.4),
                             sharey=True, squeeze=False)
    for ax, scorer in zip(axes[0], scorers):
        sub = d[d.scorer == scorer]
        for comp in ORDER_BY_DIFFICULTY:
            cell = sub[sub.comparison == comp].sort_values('n')
            if cell.empty:
                continue
            c = COMPARISON_COLOURS[comp]
            ax.plot(cell.n, cell.detect_KS * 100, marker='o', ms=3, color=c, lw=1.4,
                    label=NICE[comp])
            ax.fill_between(cell.n, cell.detect_KS_lo * 100, cell.detect_KS_hi * 100,
                            color=c, alpha=0.15, lw=0)
        ax.axhline(80, color='k', ls=':', lw=0.8)
        ax.axhline(ALPHA * 100, color='k', ls='--', lw=0.8)
        ax.set_xscale('log')
        ax.set_xlabel('images per side ($n$)')
        ax.set_title(NICE.get(scorer, scorer), fontsize=9)
        ax.set_ylim(-3, 103)
    axes[0][0].set_ylabel('detection rate (%)')
    handles = [Line2D([], [], color=COMPARISON_COLOURS[c], marker='o', ms=3,
                      label=NICE[c]) for c in ORDER_BY_DIFFICULTY]
    handles += [Line2D([], [], color='k', ls=':', lw=0.8, label='80% power'),
                Line2D([], [], color='k', ls='--', lw=0.8, label=r'$\alpha=5\%$')]
    fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.16))
    fig.suptitle('How many images before the test notices?', y=1.02)
    _save(fig, 'fig1_power_curve')


# --- fig 2 -------------------------------------------------------------------------------
def fig2_contamination(scorers=('entropy-2-mse', 'jpeg_bytes')):
    d = _load('exp2_contamination.csv')
    if d is None:
        print('  skip fig2 -- no exp2_contamination.csv')
        return
    scorers = [s for s in scorers if s in set(d.scorer)]
    contaminants = [c for c in ['uniform', 'dtd', 'imagenet64', 'cifar100']
                    if c in set(d.contaminant)]
    fig, axes = plt.subplots(len(scorers), len(contaminants),
                             figsize=(2.6 * len(contaminants), 2.5 * len(scorers)),
                             squeeze=False)
    for i, scorer in enumerate(scorers):
        for j, cont in enumerate(contaminants):
            ax = axes[i][j]
            cell = d[(d.scorer == scorer) & (d.contaminant == cont)]
            piv = cell.pivot_table(index='n', columns='fraction', values='detect')
            im = ax.imshow(piv.values * 100, origin='lower', aspect='auto',
                           vmin=0, vmax=100, cmap='magma')
            ax.set_xticks(range(len(piv.columns)))
            ax.set_xticklabels([f'{c:g}' for c in piv.columns], fontsize=6, rotation=90)
            ax.set_yticks(range(len(piv.index)))
            ax.set_yticklabels(piv.index, fontsize=6)
            ax.grid(False)
            # the 80%-power boundary -- the practically useful line
            if piv.values.max() > 0.8:
                ax.contour(piv.values * 100, levels=[80], colors='cyan', linewidths=1.2)
            if i == 0:
                ax.set_title(cont, fontsize=9)
            if j == 0:
                ax.set_ylabel(f'{NICE.get(scorer, scorer)}\n\nimages $n$', fontsize=8)
            if i == len(scorers) - 1:
                ax.set_xlabel('contaminated fraction $f$', fontsize=8)
    cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
    cbar.set_label('detection rate (%)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle('How much foreign data can hide in a collection of $n$ images?\n'
                 'cyan contour = 80% power', y=1.02, fontsize=10)
    _save(fig, 'fig2_contamination')


# --- fig 3 -------------------------------------------------------------------------------
def fig3_ecdf(scorer='entropy-2-mse', n=4000, im_size=DEFAULT_IM_SIZE):
    '''
    ECDFs rather than histograms, because the KS statistic IS the largest vertical gap between
    them. The plot then shows the statistic instead of merely illustrating it, and it sidesteps
    the bin-width question that a histogram invites.
    '''
    from experiments import sampling
    from h_test_IQM.pipeline.h_tests import KS

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.4), sharey=True)
    for ax, comp in zip(axes.ravel(), ORDER_BY_DIFFICULTY):
        spec = [c for c in COMPARISONS if c[0] == comp][0]
        _, target, test, _ = spec
        try:
            rng = np.random.default_rng(0)
            tp = sampling.resolve(target, scorer, im_size)
            sp = sampling.resolve(test, scorer, im_size)
        except FileNotFoundError:
            ax.set_visible(False)
            continue
        m = min(n, len(tp), len(sp))
        a, b = sampling.draw_pair(rng, tp, sp, m)
        stat, p = KS(a, b)

        grid = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 1000)
        fa = np.searchsorted(np.sort(a), grid, side='right') / len(a)
        fb = np.searchsorted(np.sort(b), grid, side='right') / len(b)
        ax.plot(grid, fa, color='k', lw=1.4, label='target')
        ax.plot(grid, fb, color=COMPARISON_COLOURS[comp], lw=1.4, label='test')

        # the KS statistic, drawn where it actually occurs
        k = int(np.argmax(np.abs(fa - fb)))
        ax.vlines(grid[k], min(fa[k], fb[k]), max(fa[k], fb[k]),
                  color='k', lw=2.5, alpha=0.7)
        ax.annotate(f'$D$={stat:.3f}', xy=(grid[k], (fa[k] + fb[k]) / 2),
                    xytext=(6, 0), textcoords='offset points', fontsize=8, va='center')
        ax.set_title(f'{NICE[comp]}\n$p$ = {p:.2g}   ($n$ = {m})', fontsize=8)
        ax.legend(fontsize=7, loc='lower right')
    for ax in axes[-1]:
        ax.set_xlabel('score')
    for ax in axes[:, 0]:
        ax.set_ylabel('cumulative fraction')
    fig.suptitle(f'Score distributions as ECDFs -- {NICE.get(scorer, scorer)} at '
                 f'{im_size}px.  The bar is the KS statistic.', y=1.0, fontsize=10)
    fig.tight_layout()
    _save(fig, 'fig3_ecdf')


# --- fig 4 -------------------------------------------------------------------------------
def fig4_scorer_comparison(im_size=DEFAULT_IM_SIZE):
    d = _load('exp3_resolution.csv')
    if d is None:
        print('  skip fig4 -- no exp3_resolution.csv')
        return
    d = d[d.im_size == im_size]
    scorers = [s for s in SCORERS if s in set(d.scorer)]
    comps = [c for c in ORDER_BY_DIFFICULTY if c in set(d.comparison) and
             c != 'control-disjoint']

    fig, ax = plt.subplots(figsize=(9, 3.8))
    width = 0.8 / len(scorers)
    x = np.arange(len(comps))
    for i, scorer in enumerate(scorers):
        vals, errs = [], []
        for comp in comps:
            cell = d[(d.scorer == scorer) & (d.comparison == comp)]
            vals.append(float(cell.KS.iloc[0]) if len(cell) else np.nan)
            errs.append(float(cell.KS_sd.iloc[0]) if len(cell) else 0.0)
        ax.bar(x + i * width, vals, width, yerr=errs, capsize=1.5,
               color=SCORER_COLOURS.get(scorer, None), label=NICE.get(scorer, scorer),
               error_kw=dict(lw=0.7))

    # the null band: what the control produces on the same n. Anything inside it is noise.
    ctrl = d[d.comparison == 'control-disjoint']
    if len(ctrl):
        hi = float((ctrl.KS + 2 * ctrl.KS_sd).max())
        ax.axhspan(0, hi, color='k', alpha=0.08, lw=0)
        ax.text(len(comps) - 0.5, hi, ' control (null) band', va='bottom', ha='right',
                fontsize=7, color='0.3')

    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels([NICE[c] for c in comps], fontsize=8)
    ax.set_ylabel('KS statistic $D$')
    ax.set_title(f'Every scorer on every comparison, {im_size}px, '
                 f'$n$ = {int(d.n.max())} per side\n'
                 'the autoencoder (red) is not the best column', fontsize=9)
    ax.legend(fontsize=8, ncol=3)
    _save(fig, 'fig4_scorer_comparison')


# --- fig 5 -------------------------------------------------------------------------------
def fig5_resolution():
    d = _load('exp3_resolution.csv')
    if d is None:
        print('  skip fig5 -- no exp3_resolution.csv')
        return
    comps = [c for c in ORDER_BY_DIFFICULTY if c in set(d.comparison)]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.2), sharex=True)
    for ax, comp in zip(axes.ravel(), comps):
        for scorer in [s for s in SCORERS if s in set(d.scorer)]:
            cell = d[(d.comparison == comp) & (d.scorer == scorer)].sort_values('im_size')
            if cell.empty:
                continue
            ax.plot(cell.im_size, cell.KS, marker='o', ms=3, lw=1.3,
                    color=SCORER_COLOURS.get(scorer), label=NICE.get(scorer, scorer))
            ax.fill_between(cell.im_size, cell.KS - cell.KS_sd, cell.KS + cell.KS_sd,
                            color=SCORER_COLOURS.get(scorer), alpha=0.12, lw=0)
        ax.set_xscale('log', base=2)
        ax.set_xticks(sorted(d.im_size.unique()))
        ax.set_xticklabels([str(s) for s in sorted(d.im_size.unique())])
        ax.set_title(NICE[comp], fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('im_size (px)')
    for ax in axes[:, 0]:
        ax.set_ylabel('KS statistic $D$')
    axes[0][0].legend(fontsize=7)
    fig.suptitle('How much of the effect is the 256x256 upsample?\n'
                 'CIFAR is natively 32px -- everything to its right is interpolated detail',
                 y=1.0, fontsize=10)
    fig.tight_layout()
    _save(fig, 'fig5_resolution')


# --- fig 6 -------------------------------------------------------------------------------
def fig6_calibration():
    '''
    The honesty check. Under a true null the p-values must be uniform, so the ECDF of the
    p-values must sit on the diagonal. Sagging BELOW it means the test is conservative: safe,
    because it cannot invent significance, but it costs power.
    '''
    raw = None
    path = os.path.join(RESULTS_DIR, 'exp1_power_curve_raw.csv.gz')
    if os.path.exists(path):
        raw = pd.read_csv(path)
        raw = raw[raw.comparison == 'control-disjoint']
    if raw is None or raw.empty:
        cal = _load('calibration.csv')
        if cal is None:
            print('  skip fig6 -- no null p-values')
            return
        raw = cal.assign(scorer='entropy-2-mse')

    scorers = [s for s in SCORERS if s in set(raw.scorer)]
    fig, axes = plt.subplots(1, len(scorers), figsize=(2.6 * len(scorers), 2.9),
                             sharey=True, squeeze=False)
    sizes = sorted(raw.n.unique())
    cmap = plt.get_cmap('viridis')
    for ax, scorer in zip(axes[0], scorers):
        sub = raw[raw.scorer == scorer]
        for i, n in enumerate(sizes):
            p = np.sort(sub[sub.n == n].KS_p.values)
            if len(p) == 0:
                continue
            ax.plot(p, np.arange(1, len(p) + 1) / len(p),
                    color=cmap(i / max(1, len(sizes) - 1)), lw=1.1, label=f'n={n}')
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
        ax.set_xlabel('$p$-value')
        ax.set_title(NICE.get(scorer, scorer), fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0][0].set_ylabel('cumulative fraction')
    axes[0][-1].legend(fontsize=6, ncol=2)
    fig.suptitle('Null $p$-values against uniform. Below the diagonal = conservative.',
                 y=1.04, fontsize=10)
    _save(fig, 'fig6_calibration')


# --- fig 7 -------------------------------------------------------------------------------
ARM_STYLE = {
    # (colour, linestyle) -- the autoencoder is red in every figure, and the two AE arms are
    # distinguished by dash, so the eye reads "same scorer, different statistic"
    'AE-64d / C2ST':      ('#d62728', '-'),
    'AE-scalar / KS':     ('#d62728', ':'),
    'AE-scalar / C2ST':   ('#d62728', '--'),
    'jpeg-scalar / KS':   ('#1f77b4', ':'),
    'jpeg-scalar / C2ST': ('#1f77b4', '--'),
}
# draw order, back to front. Most of these curves saturate at 100%, so whatever is plotted
# last is the only one visible up there -- and AE-64d is the line the figure exists to show.
ARM_ORDER = ['jpeg-scalar / C2ST', 'jpeg-scalar / KS',
             'AE-scalar / C2ST', 'AE-scalar / KS', 'AE-64d / C2ST']


def fig7_multivariate():
    '''
    The result that reverses fig4. fig4 says a JPEG byte count beats the autoencoder; that is
    one hand-picked scalar against another. Keeping one +1-ratio per latent channel instead of
    averaging the whole code into a single number is what the comparison should have been.
    '''
    d = _load('exp5_multivariate.csv')
    if d is None:
        print('  skip fig7 -- no exp5_multivariate.csv')
        return
    d = d.assign(arm=d.representation + ' / ' + d.test)
    comps = [c for c in ORDER_BY_DIFFICULTY if c in set(d.comparison)]

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.4), sharey=True, sharex=True)
    for ax, comp in zip(axes.ravel(), comps):
        sub = d[d.comparison == comp]
        for z, arm in enumerate([a for a in ARM_ORDER if a in set(sub.arm)]):
            cell = sub[sub.arm == arm].sort_values('n')
            colour, ls = ARM_STYLE[arm]
            lead = arm == 'AE-64d / C2ST'
            # Most arms saturate at 100%, so overlapping curves hide each other completely.
            # The jpeg baselines go down as a wide pale band and the autoencoder lines sit on
            # top of it -- both stay readable where they coincide, without faking an offset.
            base = arm.startswith('jpeg')
            ax.plot(cell.n, cell.detect * 100, marker='o', ms=3.5 if lead else 3,
                    lw=3.4 if base else (2.2 if lead else 1.4),
                    alpha=0.30 if base else 1.0,
                    solid_capstyle='round',
                    color=colour, ls=ls, label=arm, zorder=3 + z)
            ax.fill_between(cell.n, cell.detect_lo * 100, cell.detect_hi * 100,
                            color=colour, alpha=0.12, lw=0)
        ax.axhline(80, color='k', ls=':', lw=0.8)
        ax.axhline(ALPHA * 100, color='k', ls='--', lw=0.8)
        ax.set_xscale('log')
        ax.set_ylim(-3, 103)
        ax.set_title(NICE[comp], fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('images per side ($n$)')
    for ax in axes[:, 0]:
        ax.set_ylabel('detection rate (%)')
    handles = [Line2D([], [], color=ARM_STYLE[a][0], ls=ARM_STYLE[a][1], marker='o', ms=3,
                      lw=3.4 if a.startswith('jpeg') else (2.2 if a == 'AE-64d / C2ST' else 1.4),
                      alpha=0.30 if a.startswith('jpeg') else 1.0, label=a)
               for a in reversed(ARM_ORDER)]
    handles += [Line2D([], [], color='k', ls=':', lw=0.8, label='80% power'),
                Line2D([], [], color='k', ls='--', lw=0.8, label=r'$\alpha=5\%$')]
    fig.suptitle('Keeping the code beats averaging it away\n'
                 'thick red = 64-D latent occupancy; dotted red = the same code as one scalar',
                 y=1.005, fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.005),
               fontsize=8)
    _save(fig, 'fig7_multivariate')


# --- fig 8 -------------------------------------------------------------------------------
GROUP_COLOURS = {'natural': '#2ca02c', 'noise': '#ff7f0e', 'random': '#7f7f7f',
                 'reference': '#1f77b4'}
GROUP_LABEL = {'natural': 'trained on natural images', 'noise': 'trained on uniform noise',
               'random': 'untrained (random init)', 'reference': 'JPEG bytes (scalar KS)'}


def _short(enc):
    return enc.replace('entropy-2-', '').replace('-64d', '')


def fig8_code_origin():
    '''
    exp7: nine encoders at the 64-D code, coloured by what each was trained on, so "does
    training on natural images matter" is read off the colour ordering. Panels (c) and (d)
    use the post-hoc redrawn-partition protocol (exp7_diagnostics) -- the fixed-partition
    control reports a rate conditional on one split, see FINDINGS.md.
    '''
    d = _load('exp7_code_origin.csv')
    x = _load('exp7_diagnostics.csv')
    if d is None or x is None:
        print('  skip fig8 -- needs exp7_code_origin.csv and exp7_diagnostics.csv')
        return

    # 2 x 2 so it stays legible at a single column width in the write-up
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.0))
    axes = axes.ravel()
    sizes = sorted(d[d.part == 'primary'].n.unique())

    def _lines(ax, comp, value, reference=True):
        sub = d[(d.part == 'primary') & (d.comparison == comp)]
        for enc, cell in sub.groupby('encoder'):
            g = cell.group.iloc[0]
            if g == 'reference' and not reference:
                continue
            cell = cell.sort_values('n')
            ax.plot(cell.n, cell[value] * (100 if value == 'detect' else 1), marker='o',
                    ms=3, lw=3.4 if g == 'reference' else 1.4,
                    alpha=0.35 if g == 'reference' else 0.9, color=GROUP_COLOURS[g])
        ax.set_xscale('log')
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(n) for n in sizes])
        ax.minorticks_off()
        ax.set_xlabel('images per side ($n$)')

    # (a) the primary comparison: detection rate
    ax = axes[0]
    _lines(ax, 'cifar10-vs-cifar100', 'detect')
    ax.axhline(80, color='k', ls=':', lw=0.8)
    ax.axhline(ALPHA * 100, color='k', ls='--', lw=0.8)
    ax.set_ylim(-3, 103)
    ax.set_ylabel('detection rate (%)')
    ax.set_title('(a) CIFAR-10 vs CIFAR-100\n64-D C2ST, 200 repeats', fontsize=9)

    # (b) one-class saturates at 100% for every encoder, so show the effect size instead
    ax = axes[1]
    _lines(ax, 'cifar-vs-oneclass', 'effect_mean', reference=False)
    ax.axhline(0.5, color='k', ls='--', lw=0.8)
    ax.set_ylabel('C2ST held-out accuracy')
    ax.set_title('(b) CIFAR-10 vs one class\n(detection is 100% everywhere)', fontsize=9)

    # (c) k = 9 class drop, redrawn partitions: 64-D bar, scalar tick, JPEG band
    cr = x[x.diagnostic == 'classdrop_redraw']
    ax = axes[2]
    encs = [e for e in cr.scorer.unique() if e != 'jpeg_bytes']
    for i, enc in enumerate(encs):
        g = cr[cr.scorer == enc].group.iloc[0]
        v = cr[(cr.scorer == enc) & (cr.representation == '64d') & (cr.k_classes == 9)].iloc[0]
        s = cr[(cr.scorer == enc) & (cr.representation == 'scalar')
               & (cr.k_classes == 9)].iloc[0]
        ax.bar(i, v.redraw_detect * 100, color=GROUP_COLOURS[g], width=0.7)
        ax.errorbar(i, v.redraw_detect * 100,
                    yerr=[[100 * (v.redraw_detect - v.redraw_lo)],
                          [100 * (v.redraw_hi - v.redraw_detect)]],
                    color='k', lw=0.8, capsize=2)
        ax.plot(i, s.redraw_detect * 100, marker='_', ms=12, mew=2, color='k')
    j = cr[(cr.scorer == 'jpeg_bytes') & (cr.k_classes == 9)]
    if len(j):
        ax.axhline(j.redraw_detect.iloc[0] * 100, color=GROUP_COLOURS['reference'], lw=3.4,
                   alpha=0.35)
    ax.axhline(ALPHA * 100, color='k', ls='--', lw=0.8)
    ax.set_xticks(range(len(encs)))
    ax.set_xticklabels([_short(e) for e in encs], rotation=60, fontsize=7)
    ax.set_ylim(0, 60)
    ax.set_ylabel('detection rate (%)')
    ax.set_title('(c) drop 1 of 10 classes, $n=2000$\nbar = 64-D, tick = scalar', fontsize=9)

    # (d) calibration: fixed partition (filled) vs redrawn every repeat (hollow)
    ctl = x[(x.diagnostic == 'control') & (x.representation == '64d')]
    ax = axes[3]
    for i, (_, r) in enumerate(ctl.iterrows()):
        c = GROUP_COLOURS[r.group]
        for off, fp, lo, hi, face in ((-0.17, r.fixed_fp, r.fixed_lo, r.fixed_hi, c),
                                      (0.17, r.redraw_fp, r.redraw_lo, r.redraw_hi, 'white')):
            ax.errorbar(i + off, fp * 100, yerr=[[100 * (fp - lo)], [100 * (hi - fp)]],
                        fmt='o', ms=4, color=c, mfc=face, capsize=2, lw=1)
    ax.axhline(ALPHA * 100, color='k', ls='--', lw=0.8)
    ax.set_xticks(range(len(ctl)))
    ax.set_xticklabels([_short(e) for e in ctl.scorer], rotation=60, fontsize=7)
    ax.set_ylim(0, 10)
    ax.set_ylabel('false-positive rate (%)')
    ax.set_title('(d) calibration, 64-D C2ST, $n=1000$\nfilled = fixed split, '
                 'hollow = redrawn', fontsize=9)

    handles = [Line2D([], [], color=GROUP_COLOURS[g], lw=3.4 if g == 'reference' else 1.4,
                      alpha=0.35 if g == 'reference' else 1.0, label=GROUP_LABEL[g])
               for g in ('natural', 'noise', 'random', 'reference')]
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.5)
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.005),
               fontsize=8.5)
    _save(fig, 'fig8_code_origin')


FIGURES = {
    'fig1': fig1_power_curve,
    'fig2': fig2_contamination,
    'fig3': fig3_ecdf,
    'fig4': fig4_scorer_comparison,
    'fig5': fig5_resolution,
    'fig6': fig6_calibration,
    'fig7': fig7_multivariate,
    'fig8': fig8_code_origin,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', default=list(FIGURES))
    args = ap.parse_args()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for key in args.only:
        if key not in FIGURES:
            raise SystemExit(f'unknown figure {key!r}, expected any of {list(FIGURES)}')
        FIGURES[key]()
    print(f'\nfigures -> {FIGURES_DIR}')


if __name__ == '__main__':
    main()
