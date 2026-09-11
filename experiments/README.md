# experiments/

Every number in `results/` and every figure in `figures/` is produced by a script in here.
Nothing is computed in a notebook any more.

Run everything as a module from the repo root, through `uv run` (see the top-level `README.md`
for setup — `uv sync` once):

```bash
uv run python -m experiments.score_cache        # do this first -- everything else depends on it
uv run python -m experiments.exp1_power_curve
```

---

## The order to run things

| step | script | produces | cost |
|---|---|---|---|
| 0 | `score_cache.py` | `results/score_cache/*.npz` | ~1–2 h once |
| 1 | `exp1_power_curve.py` | `results/exp1_power_curve.csv` | minutes |
| 2 | `exp2_contamination.py` | `results/exp2_contamination.csv` | minutes |
| 3 | `exp3_resolution.py` | `results/exp3_resolution.csv` | minutes |
| 4 | `exp4_class_count.py` | `results/exp4_class_count.csv` | minutes |
| 5 | `exp5_multivariate.py` | `results/exp5_multivariate*.csv` | ~30 min |
| 6 | `exp6_checkpoints.py` | `results/exp6_checkpoints.csv` | minutes |
| 7 | `exp7_code_origin.py` | `results/exp7_code_origin.csv`, `exp7_checks.csv` | ~5 min (+5 min cache build) — pre-registered in `PREREGISTRATION.md` |
| 7b | `exp7_diagnostics.py` | `results/exp7_diagnostics.csv` | ~5 min — exploratory |
| — | `summarise.py` | the tables, on stdout | seconds |
| — | `make_figures.py` | `figures/fig*.png` and `.pdf` | seconds |

Step 0 is the only expensive one, and it is the reason the rest are cheap.

---

## Step 0 is the whole trick

`score_cache.py` scores every dataset once per (scorer, resolution) and stores the resulting
1-D array to disk. After that, no experiment touches an image.

That matters because scoring is a *pure function* of `(image, scorer, im_size)`, but the first
pass at these experiments re-ran the entire `dataset → scorer` path for every repeat. That is
what limited the original power curve to 10 repeats, which quantises a detection rate to 10%
and made the control row read "10–20% false positives" when it was really 1–2 hits out of 10.

With the cache:

- **1000 repeats is free**, so detection rates get real confidence intervals;
- **the contamination sweep is free**, because mixing images at fraction *f* is exactly the
  same operation as drawing `(1-f)·n` scores from one cached array and `f·n` from another;
- **the class-count sweep is free**, because a class subset is a boolean mask on the cache.

Each dataset is cached in a **fixed random row order**, which is what keeps the rest of the code
simple: `scores[:n]` is already a random sample, and `scores[:h]` / `scores[h:]` are two
disjoint random halves. The permutation depends only on the dataset, so row *i* is the same
image for every scorer and every resolution — a small cache is a prefix of a large one, and
table columns are comparable.

The cache is derived data and is gitignored. `python -m experiments.score_cache --list` shows
what is built.

---

## The six experiments

Each script's docstring says what it is for and what it would mean if the answer came back the
other way. Read those first — they are the argument, this table is just the index.

**`exp1_power_curve.py`** — how many images before a shift becomes detectable. Detection rate
vs *n*, every comparison × every scorer, 1000 repeats. The `control-disjoint` row is the
false-positive rate, not power, and it has to sit near 5%.

**`exp2_contamination.py`** — the graded version. CIFAR-10 with a fraction *f* of its images
replaced by another dataset. This is the one that turns the method into the data-collection
stopping criterion the drafts propose; everything else is a fixed pair of datasets and can only
answer "different: yes/no". The *f* = 0 column is a built-in calibration check.

**`exp3_resolution.py`** — the main table at 32/64/128/256 px. Everything here is resized to
256 before scoring, which upsamples CIFAR 8×; the autoencoder's power rises with that number
while the control stays flat. Also separates a real distribution difference from a *native
resolution* difference in the DTD and ImageNet comparisons.

**`exp4_class_count.py`** — the test sample keeps *k* of 10 CIFAR-10 classes, *k* = 1…10.
*k* = 10 is the control. Averaged over several random class subsets, because *which* classes
are kept may matter more than how many — `subset_sd` in the output is that check.

**`exp5_multivariate.py`** — the autoencoder's scalar is a fraction of `+1`s over the whole
64 × 16 × 16 code. This keeps one ratio per latent channel (64-D) and tests in R^64 with
energy distance, MMD, and a classifier two-sample test. It is the only experiment that could
overturn the finding that a JPEG byte count beats the autoencoder.

**`exp6_checkpoints.py`** — all six usable autoencoder checkpoints, `{mse,ssim,nlpd}` ×
`{natural, uniform-noise}` training. The `-u` ones were trained on *noise*; if they work as
well, the "it learned natural-image statistics" premise does not survive.

---

## What they found

Written up in full in [FINDINGS.md](../FINDINGS.md) §2b. The short version:

| exp | result |
|---|---|
| 1 | calibrated at 1000 repeats — 4.1–5.1% false positives across the n grid |
| 2 | ~5% foreign data detectable in 4000 images with `jpeg_bytes`; the AE scalar needs 10–100% |
| 3 | the AE's resolution dependence is a **binomial floor**: at 32px the latent is 256 bits |
| 4 | dropping one class of ten — `jpeg_bytes` 31%, AE scalar 8.8% against a 6% control |
| 5 | **the 64-D code beats `jpeg_bytes` 88% to 19%** on CIFAR-10 vs CIFAR-100 |
| 6 | noise-trained encoders match natural-trained ones at **100.1%** |

exp5 is the one that reverses the "a JPEG byte count beats the autoencoder" finding: it beats
the autoencoder's *scalar*, not the autoencoder.

---

## Shared modules

| file | what it holds |
|---|---|
| `common.py` | paths, `COMPARISONS`, `SCORERS`, `ALPHA`, seeding, Wilson intervals |
| `score_cache.py` | build and load the cached scores |
| `sampling.py` | the only place that turns a comparison spec into two samples |
| `summarise.py` | CSVs → the text tables |
| `make_figures.py` | CSVs → `figures/` |

`COMPARISONS` in `common.py` is the spine of the results section. A spec is
`{dataset, labels, partition}`; `partition='a'`/`'b'` take disjoint halves of the same cached
pool, which is how the control is made honest — a *different seed* only reshuffles and leaves
chance-level overlap (FINDINGS.md §3.4).

---

## `prior/`

The first pass (August 2026), kept because it produced the numbers quoted in FINDINGS.md §2.
Those scripts re-score images on every call and write to `results/prior/`. They are superseded
by `exp1`–`exp6` and should not be used for new work; `exp3` in particular reports the same
main table with error bars instead of a single draw.

One methodological difference to know about: `cifar-vs-oneclass` now draws its two sides from
disjoint halves (`partition='a'`/`'b'`). The prior version did not, so the one-class test
sample could share images with the target — which makes the two samples more similar and the
test *more conservative*, not less.
