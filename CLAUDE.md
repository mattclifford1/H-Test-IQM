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
  pipeline/multivariate.py energy distance / MMD / classifier two-sample test, for VECTOR
                           scores. Only exp5 uses these.
  scorers/baseline_scorers.py  pixel_std / pixel_entropy / jpeg_bytes — the non-perceptual
                           controls. These currently BEAT the autoencoder (FINDINGS.md 2.4).
experiments/               scripts that produce results/*.csv. Run as modules. See
                           experiments/README.md — it is the index and says the run order.
  common.py                  paths, COMPARISONS, SCORERS, ALPHA, seeding. Import, don't run.
  score_cache.py             STEP 0. Scores every dataset once per (scorer, im_size) to
                             results/score_cache/*.npz. Everything else reads that, never
                             an image — which is what makes 1000 repeats affordable.
  sampling.py                the only place a comparison spec becomes two samples
  exp1..exp6_*.py            the six experiments, in the order to run them
  make_figures.py            CSVs -> figures/fig1..fig7 as *.png and *.pdf
  summarise.py               CSVs -> the text tables
  prior/                     the first (August 2026) pass, kept because FINDINGS §2 quotes
                             it. Re-scores images per call. Superseded by exp1-exp6.
results/                   committed CSVs, one row per cell. THIS is the record, not notebook
                           output cells. score_cache/ is gitignored derived data; prior/ holds
                           the first pass.
figures/                   generated by make_figures.py. Never edit by hand.
notebooks/
  about_pipeline.ipynb     start here — the worked example of get_scores
  dataset_experiments/     the experiments behind the paper. See notebooks/README.md.
  dev/                     scratch//exploratory, not results
noise_autoencoders/        the AE training code as received, kept for reference.
                           Its save_nets/ duplicates models/save_nets/ byte for byte.
                           NOT used at runtime — models/ is what the pipeline imports.
```

## Current state

Last commit is `d5afa8d`; the August 2026 work on top of it is **uncommitted**. The project was
dormant from 2024-12-13 until then.

The original claim was that four density comparisons produce the expected ordering
(control 0.009 < ImageNet 0.043 < one-class 0.136 < uniform noise 4.71), and that is the short
paper. The six-experiment suite (`FINDINGS.md` §2b) has largely rewritten it:

- the autoencoder's **scalar** loses to a JPEG byte count (§2.4), but its **64-D code beats it
  88% to 19%** on CIFAR-10 vs CIFAR-100 (§2.9.5) — the averaging is the defect, not the scorer;
- the dependence on the 8× upsample is a **binomial floor** (§2.9.3): at 32px the latent is 256
  bits, so the occupancy score cannot have an sd below 0.031 and every dataset sits on it;
- checkpoints trained on **uniform noise** work as well as natural-trained ones — 100.1%
  recovery (§2.9.6), so the stated premise is unsupported;
- `mse-2`, the only checkpoint ever used, is the **worst** of the three natural-trained ones;
- `cifar-vs-imagenet64` is substantially a **resampling artifact** (§2.9.3).

The difference-in-difference line does not separate and the newest Overleaf draft demotes it.
The pipeline itself is sound — calibrated, 4.1–5.1% false positives against a nominal 5% at 1000
repeats per cell. See §5.1 for what the results now force.

The drafts live in `~/Repos/Overleaf/percept-reduce/` — three separate Overleaf git repos.
**`Perceptual-Tests-Densities` (2025-01-13) is the current one**; the other two are superseded or
brainstorm.

## Key facts (these are load-bearing)

- **`dataset_proportion` IS now the fraction of the dataset** — but it was not when the drafts
  were written. `get_all_loaders` used to apply an unused `[0.4, 0.3, 0.3]` split first, so
  `dataset_proportion=1` meant 40% of the data and `=0.2` meant 8%. That split was removed
  (FINDINGS.md §3.4). **Any n quoted in the Overleaf drafts is 0.4× what it claims.**
- **Target and test share a seed by default.** `shift_seed_test=0` means `get_indicies` returns
  *identical* indices for both — same images. Pass `shift_seed_test=1` for a genuinely disjoint
  subsample, or `partition_target='a'` / `partition_test='b'` for guaranteed disjoint halves.
  The density notebooks shift the seed; `run_multiple_pipelines` does not.
- **Everything is resized to 256×256** by `main.py:130`, so CIFAR is upsampled 8× before scoring.
  `base_scorer_torch`'s own default is 128. This is load-bearing, not cosmetic: the AE
  downsamples by 16, so a 32px input gives a 2×2×64 latent = 256 bits, and the occupancy score
  cannot have an sd below √(0.25/256) = 0.031. Every dataset sits on that floor at 32px,
  including uniform noise, and the scorer has no power at all. 256px buys 16 384 bits.
  See `FINDINGS.md` §2.9.3 before changing `im_size` anywhere.
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

- **`results/*.csv` is the record**, written by `experiments/`. Notebook output cells are not —
  re-running a notebook destroys whatever was in it. The figures in the Overleaf drafts were
  saved by hand and cannot be regenerated; everything in `figures/` can, via `make_figures.py`.
- **There are no tests.** Verify changes by running `dev=True` end-to-end, or by running any
  `exp*.py` with small `--repeats`. Most modules have a `__main__` self-test.
- **If you change a scorer or a loader, delete the affected `results/score_cache/*.npz`.** The
  cache is keyed on `(dataset, scorer, im_size)` only — it cannot tell that the code that
  produced it changed, so a stale entry will silently poison every experiment downstream.
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
