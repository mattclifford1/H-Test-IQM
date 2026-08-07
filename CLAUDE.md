# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this project is

**`H-Test-IQM`** — a research codebase for one question:

> Given a collected image dataset, can we decide whether it is representative of a target
> distribution — without assuming anything parametric about image data?

The approach: image data is too complex for a direct distributional test, so push both datasets
through a **perceptual scorer** that reduces each image to a scalar. That gives two 1-D samples
which ordinary two-sample statistics can compare. The scorer is Alex Hepburn's compressive
autoencoder (`EntropyLimitedModel`) with its latent hard-quantised to two centres `{-1, +1}`; an
image's score is **the fraction of `+1`s in its flattened code**. The premise is that an AE
trained to reconstruct natural images has absorbed natural-image statistics, so code occupancy is
a perceptually meaningful summary.

Pipeline: `dataset → (optional distortion) → scorer → score distribution → comparison`.

Intended uses: a stopping criterion for data collection; measuring how close two datasets are;
checking that a newly added class actually adds density.

Author: Matt Clifford <matt.clifford@bristol.ac.uk> (University of Bristol).

**Read `FINDINGS.md` before changing anything.** It has the results, the confirmed bugs, and the
prioritised list of what to run next. Do not quote a number from this repo without checking it
there first — §3.1 in particular explains why the KL values are not currently quotable.

## Repo map

```
h_test_IQM/
  pipeline/main.py         get_scores() — THE entry point. dataset→distortion→scorer→compare.
                           get_scores(help=True) prints the accepted arguments.
  pipeline/multiple.py     run_multiple_pipelines{,_diff}() — repeat a config over N seeds
  pipeline/distributions.py  DEAD except for notebooks/dev/ — the older per-image
                           IQM-sensitivity approach, superseded by the scorer pipeline
  scorers/__init__.py      the SCORERS registry — only 'entropy-2-mse' and 'BRISQUE' are wired up
  scorers/entropy_AE.py    the AE code-occupancy scorer (flat and `spacial`; `centers=5` is a TODO)
  scorers/torch_scorers.py base class: resize + to-device + no_grad, for torch models
  scorers/numpy_scorers.py base class: same, for per-image numpy models (BRISQUE)
  distortions/__init__.py  the TRANSFORMS registry: 'epsilon_noise', 'gaussian_noise', None
  distortions/additive_noise.py  unit-hypersphere ε-noise and Gaussian noise, with a
                           reject-if-clipping-ate-the-noise loop
  datasets/__init__.py     DATA_LOADER / TOTAL_INSTANCES / DATASET_PROPORTIONS registries.
                           CIFAR_100 and DTD have loaders but are NOT registered here.
  datasets/abstract_dataset.py  generic_loader — csv of filenames+labels, images on disk, RAM cache
  datasets/<NAME>/         one dir per dataset: VARS.py, downloader.py, loader.py.
                           Data self-downloads into <NAME>/raw_data/ (gitignored).
  datasets/torch_loaders.py  get_all_loaders() — split, subsample, wrap in a DataLoader
  datasets/uniform.py      uniform-noise "dataset" — the negative control
  models/compressive_AE_model.py  EntropyLimitedModel (N=128, M=64, GDN, hard-quantised latent)
  models/save_nets/        15 checkpoints: {mse,ssim,nlpd,mae} × {2,5} centres × {natural,-u}.
                           GITIGNORED — not recoverable from this repo. Only mse-2 is ever used.
                           models/README.md says the `mae` weights don't work: ignore them.
  metrics/                 classical IQMs (SSIM/MS-SSIM, NLPD, LPIPS, DISTS, PSNR, BRISQUE).
                           Only BRISQUE is reachable from the pipeline.
  pipeline/h_tests.py      the two-sample statistics: KL/JS/Wasserstein (effect sizes) and
                           KS/CVM/AD (statistic + p-value), plus permutation_test() for a
                           distribution-free p-value on any of them. compare() runs the lot.
  scorers/baseline_scorers.py  pixel_std / pixel_entropy / jpeg_bytes — the non-perceptual
                           controls. These currently BEAT the autoencoder (FINDINGS.md 2.4).
experiments/               scripts that produce results/*.csv. Run as modules:
  run_density_grid.py        {6 comparisons} x {5 scorers} -> every statistic. The main table.
  run_resolution_sweep.py    the same comparisons across im_size 32/64/128/256
  run_power_curve.py         detection rate vs sample size (statistical power)
results/                   committed CSVs + logs, one row per run. THIS is the record now,
                           not notebook output cells.
notebooks/
  about_pipeline.ipynb     start here — the worked example of get_scores
  dataset_experiments/     the experiments behind the paper. See notebooks/README.md.
  dev/                     scratch//exploratory, not results
noise_autoencoders/        the AE training code as received, kept for reference.
                           Its save_nets/ duplicates models/save_nets/ byte for byte.
                           NOT used at runtime — models/ is what the pipeline imports.
```

## Current state

Last commit `111560d` (2024-12-13); the working tree is clean and the project has been dormant
since. Four density comparisons produce the expected ordering (control 0.009 < ImageNet 0.043 <
one-class 0.136 < uniform noise 4.71) and that is the short paper. The difference-in-difference
line of experiments does not separate and the newest Overleaf draft demotes it. See `FINDINGS.md`
§2 for the numbers and §5 for what to run next.

The drafts live in `~/Repos/Overleaf/percept-reduce/` — three separate Overleaf git repos.
**`Perceptual-Tests-Densities` (2025-01-13) is the current one**; the other two are superseded or
brainstorm.

## Key facts (these are load-bearing)

- **`dataset_proportion` is not the fraction of the dataset.** `get_all_loaders` splits
  `[0.4, 0.3, 0.3]` and only ever uses the first split, then applies `dataset_proportion` on top.
  `dataset_proportion=1` → 40% of the data; `=0.2` → 8%. The drafts get this wrong.
- **Target and test share a seed by default.** `shift_seed_test=0` means `get_indicies` returns
  *identical* indices for both — same images. Pass `shift_seed_test=1` for a genuinely disjoint
  subsample. The density notebooks do; `run_multiple_pipelines` does not.
- **Everything is resized to 256×256** by `main.py:130`, so CIFAR is upsampled 8× before scoring.
  `base_scorer_torch`'s own default is 128. This is an unexplained hardcode and a confound.
- **`dev=True` shrinks the data and silently overrides any `dataset_proportion` you passed.** The
  comment above that block says the opposite of what the code does.
- **The score is centred on 0.5 for every dataset.** All the signal is in the dispersion, not the
  mean. Do not reach for a mean-difference test.
- Distortions may **reject** an image (ε-noise, when clipping ate the noise). Rejected images are
  dropped, so `n` is not exactly the requested size; `transform.num_rejected` reports it.
- `models/save_nets/*.pth` is gitignored. If those files are lost the project cannot run — there
  is no download script for them.

## Environment

```bash
conda activate h_data          # the env with the editable install (torch 2.4.0+cu121, numpy 2.0.1)
```

`h_data` is the working env — `~/anaconda3/envs/h_data/bin/python` already has `h_test_IQM`
installed editable against this directory. `h_dev` is an older equivalent. There is no `h_test`
env despite what `README.md` used to imply. Setup from scratch is in `README.md`.

Datasets self-download on first use into `h_test_IQM/datasets/<NAME>/raw_data/` (gitignored).
**ImageNet64 does not** — it is expected pre-extracted at `~/datasets/ImageNet64/{train,val}/`
with a `meta_data.csv`.

## Running things

```python
from h_test_IQM.pipeline import get_scores
get_scores(help=True)          # prints every accepted argument

outputs = get_scores(
    dataset_target='CIFAR_10', dataset_test='UNIFORM',
    scorer='entropy-2-mse', test=['KL', 'plot_hist'],
    dataset_proportion_target=0.2, dataset_proportion_test=0.2,
    shift_seed_test=1, device='cuda', dev=False)
# -> {'scores_target', 'scores_test', 'results', 'preloaded_ims'}
```

Pass `dev=True` for a fast smoke test on a tiny slice. Thread `preloaded_ims` between calls to
reuse the RAM image cache — this is what makes repeated runs tolerable.

A full-data run is minutes; a 100-seed `run_multiple_pipelines` is 7–10 minutes on GPU and up to
an hour for Caltech. Nothing is checkpointed, so a kernel restart loses the lot.

## Before you change code

- **Results live only in notebook output cells.** There is no results directory and no CSV. If you
  re-run a notebook you destroy the record of what produced the committed figures, and the figures
  in the Overleaf drafts were saved by hand and cannot be regenerated automatically. Preserve
  outputs, or add persistence first (`FINDINGS.md` §5.4).
- **There are no tests.** Verify changes by running `dev=True` end-to-end.
- Registries are the extension points: add a scorer to `scorers/__init__.py:SCORERS`, a distortion
  to `distortions/__init__.py:TRANSFORMS`, a dataset to all three dicts in `datasets/__init__.py`.
- If you touch `samples_to_pdf` or `get_sample_from_scorer`, read `FINDINGS.md` §3.1 and §3.10 —
  both have known defects and naive fixes will change every published number.
- `../H-Test-IQM-bug-fix` was a stale duplicate clone of this repo and was **deleted on
  2026-08-06** after verifying it held nothing local-only. Do not recreate it; work on a branch
  here instead. See `README.md`.

## Style

Match what is there: plain functions and small ABC + registry-dict classes, no framework, no type
annotations, `snake_case`, module-level `if __name__ == '__main__':` blocks used as smoke tests.
Comments are sparse and explain *why*. Do not add a config system, a CLI, or a class hierarchy —
the notebooks are the interface.
