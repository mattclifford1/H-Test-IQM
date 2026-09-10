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

The project uses [uv](https://docs.astral.sh/uv/). Install it if you have not
(`curl -LsSf https://astral.sh/uv/install.sh | sh`), then:

```bash
git clone https://github.com/mattclifford1/H-Test-IQM
cd H-Test-IQM
uv sync
```

That is the whole setup. `uv sync` builds `.venv/` from `uv.lock` — Python 3.12, the pinned
dependencies, a CUDA 12.1 PyTorch build, and `h_test_IQM` itself installed editable. No conda,
no separate PyTorch install step.

Run things through `uv run`, which activates the env for you:

```bash
uv run python -m experiments.exp1_power_curve --repeats 10
uv run jupyter notebook
```

or activate it the usual way if you prefer: `source .venv/bin/activate`.

> **GPU:** `pyproject.toml` pulls torch/torchvision/torchaudio from the `cu121` wheel index, which
> only carries linux and windows wheels. On macOS, or for a CPU-only box, delete the
> `[[tool.uv.index]]` and `[tool.uv.sources]` blocks and re-run `uv lock && uv sync` to take the
> default builds from PyPI.

> **Versions are pinned** in `pyproject.toml` to what `results/` and `FINDINGS.md` were produced
> with. Change them deliberately. `uv.lock` is committed and is the reproducible record; use
> `uv sync --frozen` to install exactly it.

> **Migrating from conda:** the old `h_data` env still exists and still works, but `setup.py` and
> `requirements.txt` are gone — `pyproject.toml` replaces both. `h_data` is no longer the
> supported path; use `.venv`.

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

Build the score cache once, then everything else is cheap:

```bash
python -m experiments.score_cache             # ~1-2 h, once. All the rest depend on it.

python -m experiments.exp1_power_curve        # how many images to detect each shift
python -m experiments.exp2_contamination      # detection vs contamination fraction
python -m experiments.exp3_resolution         # main table at 32/64/128/256 px
python -m experiments.exp4_class_count        # keep k of 10 classes, k = 1..10
python -m experiments.exp5_multivariate       # 64-D latent code vs the scalar
python -m experiments.exp6_checkpoints        # all six usable AE checkpoints

python -m experiments.summarise               # the tables
python -m experiments.make_figures            # figures/fig*.png and .pdf
```

Results are in [FINDINGS.md](FINDINGS.md) §2b. Three that change how you should read the rest of
this repo:

- the autoencoder's **scalar** loses to a JPEG byte count, but its **64-D code** beats it 88% to
  19% on the hardest comparison — the averaging is the problem, not the scorer;
- the autoencoder's dependence on `im_size` is a **binomial floor**, not perception: at 32px its
  latent is 256 bits, so the occupancy score cannot have an sd below 0.031, and every dataset
  sits on that floor;
- checkpoints trained on **uniform noise** discriminate as well as ones trained on natural
  images (100.1%), so the "it learned natural-image statistics" premise is unsupported.

Each writes a CSV to [`results/`](results/README.md) incrementally, so a crash costs only the
current row. See [`experiments/README.md`](experiments/README.md) for what each one is for and
why the cache changes what is affordable.

### Two things that will bite you

- **Target and test share a seed by default**, which means *identical* images, not two
  subsamples. Pass `shift_seed_test=1` for a disjoint draw — or better, use
  `partition_target='a'` / `partition_test='b'`, which are disjoint by construction.
- **Everything is resized to 256×256 before scoring**, so CIFAR is upsampled 8×. That is a real
  confound, not a detail: the autoencoder's discriminative power rises with `im_size` while the
  control stays flat (FINDINGS.md §2.5). Pass `im_size=(32, 32)` for native resolution.

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
