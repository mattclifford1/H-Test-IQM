# H-Test-IQM

Using perceptual image quality metrics to decide whether an image dataset is representative of a
target distribution.

Image data is too complex for a direct distributional test, so this project pushes both datasets
through a **perceptual scorer** that reduces each image to a scalar. That gives two 1-D samples
which ordinary two-sample statistics can compare. The scorer is a compressive autoencoder whose
latent is hard-quantised to two centres `{-1, +1}`; an image's score is the fraction of `+1`s in
its code. The premise is that an autoencoder trained to reconstruct natural images has absorbed
natural-image statistics, so code occupancy is a perceptually meaningful summary.

```
dataset → (optional distortion) → scorer → score distribution → comparison
```

Intended uses: a stopping criterion for data collection; measuring how close two datasets are;
checking that a newly added class actually adds density.

## Documentation

| file | what's in it |
|---|---|
| **[FINDINGS.md](FINDINGS.md)** | **Results, confirmed bugs, and the prioritised list of what to run next.** Read this first. |
| [CLAUDE.md](CLAUDE.md) | Repo map, load-bearing conventions, and gotchas. Useful to humans too. |
| [results/README.md](results/README.md) | What each results CSV contains and how to regenerate it. |
| [notebooks/README.md](notebooks/README.md) | What each experiment notebook does and which paper figure it produced. |
| [h_test_IQM/pipeline/README.md](h_test_IQM/pipeline/README.md) | The original pipeline design notes. |
| [h_test_IQM/models/README.md](h_test_IQM/models/README.md) | Which autoencoder checkpoint is which. |

Paper drafts are in `~/Repos/Overleaf/percept-reduce/` — three separate Overleaf git projects.
**`Perceptual-Tests-Densities` (2025-01-13) is the current one.**

## Setting up

```bash
git clone https://github.com/mattclifford1/H-Test-IQM
cd H-Test-IQM
conda create -n h_data python=3.10 -y
conda activate h_data
```

Install PyTorch for GPU if required, e.g.

```bash
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Then install as an editable package:

```bash
pip install -e .
```

> **Existing machine:** the env is already built as **`h_data`** — just `conda activate h_data`.
> (`h_dev` is an older equivalent. There is no `h_test` env.)

### Data

Most datasets download and unpack themselves on first use into
`h_test_IQM/datasets/<NAME>/raw_data/` (gitignored): CIFAR-10, CIFAR-100, Caltech-101,
Caltech-256, DTD, MNIST. Kodak is committed. Uniform noise is generated on the fly.

**ImageNet64 is the exception** — it is expected pre-extracted at
`~/datasets/ImageNet64/{train,val}/` with an `images/` folder and a `meta_data.csv`.

### Model weights

`h_test_IQM/models/save_nets/*.pth` holds the 15 autoencoder checkpoints and is **gitignored** —
they are not recoverable from this repo, and there is no download script. If you are setting up on
a new machine you need to copy them across. `noise_autoencoders/save_nets/` is a byte-identical
duplicate, kept alongside the training code as received.

## Running an experiment

```python
from h_test_IQM.pipeline import get_scores
get_scores(help=True)          # prints every accepted argument

outputs = get_scores(
    dataset_target='CIFAR_10',
    dataset_test='UNIFORM',
    scorer='entropy-2-mse',
    test=['KL', 'plot_hist'],
    dataset_proportion_target=0.2,
    dataset_proportion_test=0.2,
    shift_seed_test=1,          # otherwise target and test get the SAME images
    device='cuda',
    dev=False,                  # dev=True -> tiny slice, for a smoke test
)
```

`test` accepts effect sizes (`'KL'`, `'JS'`, `'wasserstein'`), real two-sample tests that return
a p-value (`'KS'`, `'CVM'`, `'AD'`), `'all'`, and `'plot_hist'`. Add `n_permutations=1000` for a
distribution-free p-value on the effect sizes too.

To repeat a configuration over many seeds, use
`h_test_IQM.pipeline.multiple.run_multiple_pipelines{,_diff}`. Worked examples are in
[`notebooks/`](notebooks/README.md) — start with `about_pipeline.ipynb`.

### The full experiments

```bash
python -m experiments.run_density_grid --n 4000 --permutations 1000   # main table
python -m experiments.run_resolution_sweep --n 4000                   # im_size control
python -m experiments.run_power_curve --repeats 10                    # detection vs sample size
python -m experiments.run_calibration --repeats 500                   # false-positive rate
```

Each writes a CSV to [`results/`](results/README.md) incrementally, so a crash costs only the
current row. Add `--quick` to `run_density_grid` for a smoke test.

### Two things that will bite you

- **`dataset_proportion` is not the fraction of the dataset.** An unused `[0.4, 0.3, 0.3]` split
  is applied first, so `dataset_proportion=1` gives 40% of the data and `=0.2` gives 8%.
- **Target and test share a seed by default**, which means *identical* images, not two
  subsamples. Pass `shift_seed_test=1` for a disjoint draw.

Both, and the rest, are in [CLAUDE.md](CLAUDE.md#key-facts-these-are-load-bearing).

## Extending

The three extension points are registry dicts:

| add a… | to |
|---|---|
| scorer | `h_test_IQM/scorers/__init__.py` → `SCORERS` |
| distortion | `h_test_IQM/distortions/__init__.py` → `TRANSFORMS` |
| dataset | `h_test_IQM/datasets/__init__.py` → `DATA_LOADER`, `TOTAL_INSTANCES`, `DATASET_PROPORTIONS` (all three) |

A new dataset needs a `VARS.py` / `downloader.py` / `loader.py` trio; copy `CIFAR_10/` and
subclass `generic_loader`.

## Related

- **`~/projects/percept_reduce`** — sibling project by the same author on whether training an
  autoencoder with a perceptual loss makes its latent space more useful downstream. Shares the
  compressive-AE lineage and the Overleaf parent directory; the codebases are independent. Its
  finding that noise-trained encoders recover ~91% of MSE performance is directly relevant here
  (`FINDINGS.md` §7).
- **`~/projects/H-Test-IQM-bug-fix`** — **deleted (2026-08-06); nothing was lost.** It was a stale
  duplicate clone of this repo: the same GitHub remote checked out at a detached `24fb5cc`
  (2024-09-20), an ancestor of `main`. Every file in it was either identical to, or an older
  version of, a file here — its `noise_sphere.py` became `distortions/additive_noise.py`, its
  `entropy_encoder.py` became `scorers/entropy_AE.py`, and its five notebooks were earlier copies
  of `notebooks/dev/`. Verified before deletion: no local-only commits, no stashes, no untracked
  files, and a single uncommitted one-line default swap (`test='plot_hist'` → `test='KL'`) that
  `main` has long since superseded. Recorded here so it does not get recreated.
