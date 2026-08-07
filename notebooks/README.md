# Notebooks

Worked examples and the original (Dec 2024) experiments. Everything is driven from
`h_test_IQM.pipeline.get_scores`.

> **The notebooks are no longer the source of truth for results.** The August 2026 re-run lives in
> [`../experiments/`](../experiments) and writes to [`../results/`](../results/README.md) as CSV.
> Use those for any number you intend to quote — see [`../FINDINGS.md`](../FINDINGS.md) §2.2–2.5.
>
> The output cells below were produced **before** the methodological fixes (KL smoothing, size
> matching, `dataset_proportion` semantics) and their numbers are superseded. They are kept
> because they are the record of what produced the figures currently in the Overleaf drafts.

Start with **`about_pipeline.ipynb`** — the worked example, with `get_scores(help=True)` output.

## `dataset_experiments/densities/` — the paper

One KL per dataset pair, from 50-bin histograms of the scores. `scorer='entropy-2-mse'`, no
distortion. **This is the line of experiments the current draft is built on.**

| notebook | comparison | KL | paper figure |
|---|---|---|---|
| `labels/CIFAR_control.ipynb` | CIFAR vs CIFAR, disjoint subsample (**control**) | 0.0090 | `CIFAR_subsamples.png` |
| `cifar_vs_imagenet.ipynb` | CIFAR vs ImageNet64-val | 0.0429 | `CIFAR_IMAGENET_SCORES.png` |
| `labels/CIFAR_one_class.ipynb` | CIFAR (all) vs CIFAR (class 1) | 0.136 | `CIFAR_one_class.png` |
| `AE_cifar_vs_unifrom.ipynb` | CIFAR vs uniform noise | 4.71 | `CIFAR_UNIFORM_SCORES.png` |

The ordering is what the hypothesis predicts and the control sits ~15× below the weakest real
effect. Caveats before quoting any of it: the KL *values* are dominated by an arbitrary smoothing
constant (`FINDINGS.md` §3.1), the one-class row is not sample-size matched (§3.5), and the
"20% CIFAR" in the draft is really 8% (§3.4).

## `dataset_experiments/difference_in_difference/` — the negative result

Add noise, take the KL between a dataset and its noised self, repeat over 100 seeds, and compare
the *distribution of KLs* between two datasets.

| notebook | comparison | state |
|---|---|---|
| `same_dataset/CIFAR_differences.ipynb` | CIFAR vs CIFAR (**control**) | 100 runs, overlaps as expected → `CIFAR_VS_CIFAR.png` |
| `uniform/cifar_vs_uniform.ipynb` | CIFAR vs uniform noise | 100 runs, **does not separate** → `CIFAR_VS_UNIFORM.png` |
| `different_labels/CIFAR_differences_one_class.ipynb` | CIFAR all vs CIFAR one class | 100 runs, partial separation → `CIFAR_VS_oneclass.png` |
| `different_datasets/caltech_differences.ipynb` | Caltech-256 vs Caltech-101 | **aborted at 9/100** — no result |

Failing to separate CIFAR from uniform noise is the reason the Jan-2025 draft demotes this whole
section. Three things to know before reading these figures:

- **The two legend entries are swapped in every one of these notebooks** — including in the three
  figures as they appear in the drafts (`FINDINGS.md` §3.2). In `CIFAR_VS_oneclass.png` this
  inverts the reading.
- `same_dataset/`, `different_labels/` and `different_datasets/` use `run_multiple_pipelines`,
  which noises **only the test side**; `uniform/` uses `run_multiple_pipelines_diff`, which noises
  **both**. That is the entire 10× scale gap between the figures, and it means they are not
  comparable (§3.3).
- Target and test are given the same seed, so they are the *same images* — the KL only reflects
  the noise draw (§2.2).

## `dev/` — exploratory, not results

- `AE_representation.ipynb` — first pass at the AE-scorer comparison, superseded by `densities/`.
- `entropy_net_scoring.ipynb` — poking at the entropy model's outputs directly.
- `iqm_distortion_plot_distributions.ipynb` — the **older** approach: per-image IQM sensitivity to
  ε-noise via `pipeline/distributions.py`, using classical IQMs rather than the AE code. This is
  the only consumer of `pipeline/distributions.py`, which is otherwise dead.
- `kodak_test_epsilon_vis.ipynb` — visualising ε-noise on Kodak to pick the "barely noticeable"
  ε=1 used by `epsilon_noise`.

Note `dataset_experiments/AE_representation-results.ipynb` existed at commit `03e9871` and is no
longer in `HEAD`; recover it with `git show 03e9871:notebooks/dataset_experiments/AE_representation-results.ipynb`
if it held anything worth keeping.

## Adding an experiment

Follow `densities/cifar_vs_imagenet.ipynb` for a single comparison, or
`difference_in_difference/uniform/cifar_vs_uniform.ipynb` for a repeated-seed run. Pass
`shift_seed_test=1` unless you specifically want target and test to be the same images, and
`dev=True` while iterating.
