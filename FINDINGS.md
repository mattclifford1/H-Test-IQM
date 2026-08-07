# Findings

State of the `H-Test-IQM` project. Originally reconstructed from the dormant repo at commit
`111560d` (2024-12-13); **§2.2 and §2.4–2.7 are new results from the August 2026 re-run** after the
methodological fixes in §3 were applied.

Every number is quoted from a **committed notebook output cell**, from a CSV in `results/`, or
was **recomputed here** (with `~/anaconda3/envs/h_data/bin/python`). Nothing is read off a figure.

> **Headline, after the fixes and the re-run.**
>
> 1. **The method works and is calibrated.** With real hypothesis tests in place, the disjoint
>    control accepts (p = 0.20–0.98 across all five scorers), every genuine shift rejects at
>    n = 4000, and the measured false-positive rate is 3.6% [2.1%, 5.6%] against a nominal 5% —
>    mildly conservative at small n, never over-rejecting (§2.7). There is also now a power curve
>    saying how many images each kind of shift needs (§2.6). That part is defensible.
> 2. **But the perceptual autoencoder loses to a JPEG byte count on every comparison tested**
>    (§2.4). This is the control the paper most needed and it does not go the way the framing
>    assumes.
> 3. **And the autoencoder only works because of an accident** — the hardcoded 8× upsample. At
>    CIFAR's native 32×32 it fails to detect class removal at all (§2.5).
>
> None of this kills the project, but it does redirect it: the defensible contribution is the
> *pipeline* (score → 1-D two-sample test → calibrated p-value), not the choice of scorer. See §5.

---

## 1. What the project is

Given a target image dataset and a test image dataset, decide whether they come from the same
distribution — without assuming anything parametric about image data.

The mechanism: push both datasets through a **perceptual scorer** that reduces each image to a
scalar, giving two 1-D samples that can be compared with ordinary statistics. The scorer is
Alex Hepburn's compressive autoencoder (`EntropyLimitedModel`) whose latent is hard-quantised to
2 centres `{-1, +1}`; the score for an image is **the fraction of `+1`s in its flattened code**.
The claim is that an AE trained to reconstruct natural images has absorbed natural-image
statistics, so the code-occupancy statistic is a perceptually meaningful summary.

The pipeline is `dataset → (optional distortion) → scorer → score distribution → comparison`,
which is `h_test_IQM/pipeline/main.py:get_scores`.

Intended uses, from the newest draft (`Perceptual-Tests-Densities`):

- a stopping criterion for data collection;
- measuring how close two datasets are, and searching for similar datasets;
- checking that a newly added concept/class actually adds density.

## 2. Results

### 2.1 Density comparison, as originally run (Dec 2024) — superseded by §2.2

One KL per dataset pair, from 50-bin normalised histograms of the scores,
`scorer='entropy-2-mse'`, no distortion. All from committed notebook outputs.

| comparison | KL(target‖test) | n target | n test | notebook |
|---|---|---|---|---|
| CIFAR-10 vs CIFAR-10, disjoint subsample (**control**) | **0.0090** | 4 800 | 4 800 | `densities/labels/CIFAR_control.ipynb` |
| CIFAR-10 vs ImageNet64-val | **0.0429** | 24 000 | 20 000 | `densities/cifar_vs_imagenet.ipynb` |
| CIFAR-10 (all) vs CIFAR-10 (class 1 only) | **0.136** | 24 000 | 2 400 | `densities/labels/CIFAR_one_class.ipynb` |
| CIFAR-10 vs uniform noise | **4.71** | 4 800 | 4 800 | `densities/AE_cifar_vs_unifrom.ipynb` |

**The ordering is exactly what the hypothesis predicts** — control ≪ different natural dataset <
class-restricted < noise — and the control is ~15× below the weakest real effect. The ordering
survives the re-run; the *values* do not (the smoothing constant inflated them, §3.1, and the
one-class row was not size-matched, §3.5). Use §2.2 instead.

Two things the figures show that the table does not:

- **The score is centred on 0.5 in every case.** Uniform noise does not shift the mean, it
  collapses the *variance* (`CIFAR_UNIFORM_SCORES.png`: CIFAR is a broad bump, uniform is a
  spike at 0.500). The `+1`-ratio is a mean over ~16 000 near-symmetric code positions, so the
  CLT pins it at 0.5 and **all the discriminative signal lives in the dispersion**. Worth saying
  out loud in the paper — it means a mean-based two-sample test would find nothing, and it
  suggests the statistic is close to "how much high-frequency structure survives quantisation".
- **Removing classes narrows and slightly left-shifts the distribution** rather than moving it
  (`CIFAR_one_class.png`), consistent with the same reading.

### 2.2 Density comparison, re-run with real hypothesis tests (Aug 2026)

`results/density_grid.csv`, `experiments/run_density_grid.py`. n = 4000 **per side, size-matched
everywhere**, `im_size=256`, 1000 permutations. `entropy-2-mse` column:

| comparison | KL | KS | KS p-value | verdict |
|---|---|---|---|---|
| control (CIFAR-a vs CIFAR-b, disjoint partition) | 0.0088 | 0.0240 | 0.20 | **accept** ✓ |
| CIFAR-10 vs CIFAR-100 | 0.0134 | 0.0372 | 7.8e-3 | reject |
| CIFAR-10 vs DTD | 0.0230 | 0.0450 | 6.1e-4 | reject |
| CIFAR-10 vs ImageNet64-val | 0.0448 | 0.1190 | 4.4e-25 | reject |
| CIFAR-10 (all) vs CIFAR-10 (class 1) | 0.0768 | 0.1520 | 1.0e-40 | reject |
| CIFAR-10 vs uniform noise | 1.863 | 0.3120 | 2.3e-172 | reject |

**This is the result the paper should report.** The control accepts and everything else rejects,
with a real p-value rather than an uncalibrated divergence. Three things changed versus §2.1:

- **Uniform noise: KL 4.71 → 1.86.** The old value was inflated by the smoothing constant (§3.1).
- **One class: KL 0.136 → 0.077.** Roughly half the original effect was the n=24000-vs-2400
  sample-size mismatch (§3.5). The effect survives size-matching, but it is half the size.
- **CIFAR-10 vs CIFAR-100 and vs DTD are new** — both loaders were written but never registered
  (§3.9). CIFAR-10 vs CIFAR-100 is the hardest pair (two similar natural 32×32 datasets) and is
  detected, marginally, at n=4000.

### 2.3 Difference-in-difference — a genuine negative result

Add noise, take the KL between a dataset and its noised self, repeat over 100 seeds, and compare
the resulting *distributions of KL* between two datasets. Idea: dataset identity should show up
in noise sensitivity even if the raw densities are hard to compare.

**It does not separate.** `CIFAR_VS_UNIFORM.png` shows CIFAR's and uniform noise's KL
distributions almost completely overlapping (both ≈0.01–0.09) — on the one comparison that
should have been trivial. `CIFAR_VS_oneclass.png` shows partial separation, `CIFAR_VS_CIFAR.png`
(the control) overlaps as it should.

The Jan-2025 draft already reflects this: the section heading is *"Difference in Difference --
Probably not to be included for the short paper"*. **That call looks right.** Two structural
reasons why it was never going to work as coded:

1. Target and test are handed **the same seed**, so `get_indicies` returns **identical image
   indices** (verified: `t[:n] == s[:n]` is `True` for seeds 0, 1, 2). Target and test are
   literally the same pictures. The KL therefore measures only the difference between two draws
   of Gaussian noise — a Monte-Carlo noise floor with no dataset-level sampling variation in it.
2. Noise is applied to **both** sides in the uniform experiment and to **only the test side** in
   the CIFAR/one-class experiments (§3.3), so the two families are not on the same scale
   (KL ≈ 0.01–0.09 vs ≈ 0.2–0.55) and must not be read side by side.

So the negative result is currently a statement about a specific broken configuration, not about
the difference-in-difference idea. Fix (1) and (2) before abandoning the idea — or abandon it and
say only that the density comparison in §2.1 is stronger and simpler.

### 2.4 The baselines beat the perceptual autoencoder

`results/density_grid.csv`. Same six comparisons, same n = 4000, five scorers. **KS statistic**
(higher = better separation; the control row should be near zero for all of them):

| comparison | entropy-2-mse | BRISQUE | jpeg_bytes | pixel_std | pixel_entropy |
|---|---|---|---|---|---|
| control (should not separate) | 0.024 | 0.011 | 0.021 | 0.023 | 0.015 |
| CIFAR-10 vs CIFAR-100 | 0.037 | 0.037 | 0.028 ᶰ | **0.051** | **0.076** |
| CIFAR-10 vs DTD | 0.045 | 0.867 | **0.958** | 0.128 | 0.169 |
| CIFAR-10 vs ImageNet64 | 0.119 | **0.920** | 0.674 | 0.087 | 0.067 |
| CIFAR-10 vs one class | 0.152 | 0.200 | **0.252** | 0.209 | **0.252** |
| CIFAR-10 vs uniform | 0.312 | **1.000** | **1.000** | 0.531 | 0.662 |

ᶰ = the only cell that fails to reject. **Every scorer's control accepts**, so the pipeline is
calibrated regardless of scorer — that part is solid.

But the autoencoder is **last or second-to-last on five of six comparisons**. A JPEG byte count
separates CIFAR from uniform noise *perfectly* (KS = 1.000, every image on the right side) where
the autoencoder manages 0.312.

The obvious defence is that the baselines are just reading **native resolution** — CIFAR is
upsampled 8× to reach 256 and is therefore blurry, while DTD and ImageNet are downsampled and
stay sharp. That defence is **half right**, and testing it is what `resolution_sweep_jpeg.csv` is
for. Splitting the comparisons by whether the two sides share a native resolution:

- **Resolution differs** (DTD, ImageNet64): jpeg's KS collapses from 0.958 → 0.299 and 0.674 →
  0.072 when everything is scored at 32×32. So yes — those two cells are largely a resolution
  artifact and should not be quoted as evidence for anything.
- **Resolution matched** (CIFAR-100, one-class, uniform — all natively 32×32): jpeg's KS is
  essentially **flat** across `im_size` (one-class 0.295 → 0.252; uniform 1.000 at every size).
  Those wins are real content sensitivity, not an artifact.

And on exactly those resolution-matched comparisons, scored at native 32×32 where neither method
is handicapped:

| comparison (all native 32×32) | entropy-2-mse KS (p) | jpeg_bytes KS (p) |
|---|---|---|
| control | 0.024 (0.19) accept | 0.015 (0.79) accept |
| CIFAR-10 vs CIFAR-100 | 0.013 (0.91) **miss** | 0.018 (0.52) **miss** |
| CIFAR-10 vs one class | 0.020 (0.43) **miss** | **0.295 (4e-154)** |
| CIFAR-10 vs uniform | 0.048 (1.8e-4) | **1.000 (0)** |

**A JPEG byte count is a strictly better dataset-shift detector than the perceptual autoencoder,
on every comparison, at every resolution tested.** At native resolution the autoencoder misses
class removal entirely while JPEG detects it overwhelmingly.

This is the control the drafts never ran, and it has to be in the paper. It does not make the
work worthless — but it does mean the claim "perceptual metrics give you a privileged space for
dataset comparison" is not supported by these experiments, and a reviewer running `jpeg_bytes`
would find this immediately.

### 2.5 The autoencoder's power comes from the upsampling accident

`results/resolution_sweep.csv`. Same comparisons, `entropy-2-mse`, sweeping `im_size`. KS:

| comparison | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| control | 0.024 | 0.018 | 0.021 | 0.024 |
| CIFAR-10 vs CIFAR-100 | 0.013 ᶰ | 0.022 ᶰ | 0.028 ᶰ | 0.037 |
| CIFAR-10 vs DTD | 0.035 | 0.047 | 0.040 | 0.045 |
| CIFAR-10 vs ImageNet64 | 0.037 | 0.048 | 0.069 | 0.119 |
| CIFAR-10 vs one class | 0.020 ᶰ | 0.057 | 0.095 | 0.152 |
| CIFAR-10 vs uniform | 0.048 | 0.068 | 0.178 | 0.310 |

ᶰ = fails to reject. **Discrimination increases monotonically with input size on every
comparison**, and the control stays flat at ~0.02 throughout — so this is real added power, not
inflated noise. At CIFAR's native 32×32 the method **fails to detect class removal** (p = 0.43)
and barely detects uniform noise (KS = 0.048).

Part of this is a within-statistic averaging effect: the latent grid is 64 × H/16 × W/16, so the
score averages 256 codes at im_size=32 and 16 384 at 256, and the per-image score std duly falls
0.0300 → 0.0177 → 0.0129 → 0.0127. But that **plateaus by 128** while the power keeps climbing
from 128 → 256, so noise reduction is not the whole story — the autoencoder is also reading
structure in the interpolated content that is not present in the original image.

Either way the conclusion is uncomfortable and should be stated plainly: **the hardcoded 256 was
never justified in the code, and the method's apparent effectiveness depends on it.**

### 2.6 How many images you need — the practically useful result

`results/power_curve.csv`, `experiments/run_power_curve.py`. `entropy-2-mse`, `im_size=256`,
10 independent draws per cell, **detection rate = % of repeats rejecting at α = 0.05**:

| comparison | n=50 | 100 | 200 | 500 | 1000 | 2000 | 4000 | n for 80% |
|---|---|---|---|---|---|---|---|---|
| control (true null) | 10 | 20 | 10 | 0 | 0 | 10 | 10 | — (never) |
| CIFAR-10 vs uniform noise | **100** | 100 | 100 | 100 | 100 | 100 | 100 | **≤ 50** |
| CIFAR-10 vs one class | 20 | 20 | **100** | 100 | 100 | 100 | 100 | **200** |
| CIFAR-10 vs ImageNet64 | 30 | 40 | 70 | **90** | 100 | 100 | 100 | **500** |
| CIFAR-10 vs DTD | 0 | 0 | 0 | 10 | 50 | **80** | 100 | **2000** |
| CIFAR-10 vs CIFAR-100 | 10 | 0 | 0 | 20 | 30 | 70 | **90** | **4000** |

**This is the figure that makes the method useful rather than merely correct**, and it is what the
drafts' "stopping criterion for data collection" actually cashes out to: *uniform noise is caught
with 50 images; a missing class needs 200; a different natural dataset needs 500–2000; telling
CIFAR-10 from CIFAR-100 needs 4000 and is still not saturated.* Every curve is monotone in n and
the control stays flat, which is exactly the shape a working test should produce.

Two caveats. **DTD is oddly hard** (nothing until n=500) despite being the most visually distinct
dataset here — consistent with its weak KS at `im_size=256` (0.045, the smallest of all six), and
another sign the scalar statistic is discarding most of the signal. And the **control row is
measured over only 10 repeats**, which cannot resolve a 5% rate — hence §2.7.

### 2.7 The test is calibrated, and mildly conservative at small n

`results/calibration.csv`, `experiments/run_calibration.py`. Both sides drawn from disjoint halves
of CIFAR-10, so the null is true by construction. 500 repeats per row:

| n | false positives | rate | 95% CI | p-value uniformity |
|---|---|---|---|---|
| 200 | 18/500 | 3.6% | [2.1%, 5.6%] | p = 4.5e-06 ✗ |
| 1000 | 18/500 | 3.6% | [2.1%, 5.6%] | p = 0.127 ✓ |
| 4000 | 19/500 | 3.8% | [2.3%, 5.9%] | p = 0.007 ✗ |

**The false-positive rate is fine at every size** — stable at 3.6–3.8%, CI always containing the
nominal 5%, never over-rejecting. But the uniformity of the null p-values fails at n=200, passes
at n=1000, and fails again at n=4000. A single mechanism cannot produce that shape; two can.

I initially attributed this to **ties** — the score is a ratio of integer code counts, so it is
discrete with granularity exactly 1/16384 at `im_size=256`, and 400 pooled samples do produce only
**314 unique values**. That attribution is wrong, and the check that settles it is worth keeping.
Feeding *perfectly continuous* synthetic samples (no autoencoder, no ties) through the same test
reproduces the n=200 behaviour almost exactly:

| n | data | mean null p | uniformity p | FP rate |
|---|---|---|---|---|
| 200 | continuous | 0.539 | 6.5e-06 | 3.8% |
| 200 | rounded to 1/16384 | 0.525 | 0.013 | 3.8% |
| 1000 | continuous | 0.502 | 0.434 | 5.0% |
| 1000 | rounded to 1/16384 | 0.547 | 2.6e-04 | 4.0% |

Compare the n=200 continuous row (mean 0.539, uniformity 6.5e-06) against the measured AE row
(0.548, 4.5e-06): essentially identical. **So the n=200 conservatism is scipy's asymptotic KS
approximation being conservative at small samples, not the discreteness of the score.**

And the synthetic table predicts the odd fail/pass/fail shape exactly. At n=1000 continuous data
passes uniformity (0.434) while rounded data fails (2.6e-04) — so by n=1000 the approximation
error has died away and **ties** have become the dominant term. The real data follows: uniformity
recovers at n=1000 (0.127) as the approximation error vanishes, then degrades again at n=4000
(0.007) as the tie effect grows with sample size. Two mechanisms crossing over, in the order the
synthetic control predicts.

Three consequences for the paper:

1. **Nothing in §2.2–2.6 is inflated by this.** A conservative test under-rejects, so every
   rejection reported stands and the §2.6 power numbers are, if anything, slight underestimates.
2. **Prefer the permutation p-value for anything marginal.** `h_tests.permutation_test` is exact
   under both small samples and ties; CIFAR-10-vs-CIFAR-100 (p = 7.8e-3) is the cell where it
   could matter. This also removes the need to argue about which effect dominates.
3. **The discreteness still deserves a sentence**, because it gets worse at low resolution: at
   `im_size=32` there are only 256 codes, so the score takes at most 257 distinct values — which
   compounds the loss of power already visible in §2.5.

### 2.8 What was never actually built

The July-2024 `Perceptual-Tests-Ideas` draft lists the non-parametric tests to use —
Kolmogorov–Smirnov, Mann–Whitney U, Cramér–von Mises, Anderson–Darling. **None of them were
implemented.** The only comparison in the code is a KL point estimate. There is no null
distribution, no p-value, no threshold, and therefore **no hypothesis test anywhere in a project
called H-Test-IQM**. This is the single biggest gap between the draft and the code (§5.1).

---

## 3. Bugs and methodological problems

Ordered by how much they affect a publishable number.

### 3.1 The KL estimator is not scale-invariant — **FIXED**

`pipeline/main.py:215 samples_to_pdf` builds histograms with `density=True`, then, **only if some
bin is empty**, adds `1e-6` and renormalises. Three separate problems:

- `density=True` values are `counts / (n · bin_width)`, so their magnitude depends on the width of
  the score axis. Adding a *fixed* `1e-6` to them means the smoothing strength silently changes
  with the score range. Recomputed on a CIFAR-like vs uniform-like pair: rescaling the score axis
  by 1000 moves the KL from **10.57 to 6.22**. A divergence must not depend on the units of the
  variable.
- The result is essentially a readout of the arbitrary constant. Same data, only `1e-6` varied:

  | epsilon | 1e-2 | 1e-4 | **1e-6** | 1e-8 | 1e-10 |
  |---|---|---|---|---|---|
  | KL | 4.77 | 7.67 | **10.57** | 13.47 | 16.36 |

  ≈2.9 nats per decade. With 43 of 50 bins empty in the spiked (uniform) histogram, the reported
  **KL = 4.71 for CIFAR vs uniform is mostly a count of empty bins**, not a measure of divergence.
- Because the branch is conditional, runs with an empty bin and runs without are smoothed
  differently, so KLs are not comparable across experiments.

**Fix:** smooth the *counts*, not the densities — `p = (counts + 0.5) / (counts + 0.5).sum()`,
applied unconditionally. Scale-free, standard (Krichevsky–Trofimov), and on the test pair above
gives a stable **3.63**. Better still, drop histogram-KL for a 1-D sample and use a statistic that
needs no binning at all (§5.1).

The §2.1 *ordering* survives all of this. The *values* do not — do not quote 4.71.

### 3.2 Every difference-in-difference figure has its legend entries swapped — **FIXED**

In all four DiD notebooks:

```python
dist1, dist2, all_bins, one_bins = samples_to_pdf(test2, test1, num_bins=50)
plot_hist(dist1, all_bins, name=run1['name'])   # dist1 is built from test2
plot_hist(dist2, one_bins, name=run2['name'])   # dist2 is built from test1
```

`samples_to_pdf(sample1, sample2)` returns `dist1` from `sample1`. `sample1` is `test2`, so
`dist1` is run **2**'s data — plotted under run **1**'s name. The arguments were swapped to get
the bin range from the wider sample and the labels were never swapped back.

**This affects `CIFAR_VS_CIFAR.png`, `CIFAR_VS_UNIFORM.png` and `CIFAR_VS_oneclass.png` as they
appear in the Overleaf drafts.** In `CIFAR_VS_oneclass.png` the curve labelled "CIFAR all" is
actually "CIFAR one" and vice versa — which inverts the reading of that figure. Regenerate all
three before they go anywhere near a submission.

### 3.3 The two experiment families measure different things under one name — **FIXED**

`pipeline/multiple.py:run_multiple_pipelines` (used by `same_dataset/`, `different_labels/`,
`different_datasets/`) **never sets `transform_target`**, so it defaults to `None`: clean target
vs noised test. `run_multiple_pipelines_diff` (used by `uniform/`) sets both. That is the whole
explanation for the 10× scale gap between `CIFAR_VS_CIFAR.png` (0.2–0.55) and `CIFAR_VS_UNIFORM.png`
(0.01–0.09) — noise on both sides largely cancels. Both are captioned "difference in difference".

### 3.4 `dataset_proportion` is not the fraction of the dataset it looks like — **FIXED**

`get_all_loaders` splits with `props=[0.4, 0.3, 0.3]` and only ever uses the **first (0.4) split**;
`dataset_proportion` is then applied *on top of that*. So `dataset_proportion=1` is 40% of the
dataset and `dataset_proportion=0.2` is **8%**, not 20%. The sample counts confirm it: CIFAR at
`1` gives 24 000 = 60 000 × 0.4, and at `0.2` gives 4 800 = 60 000 × 0.08.

The drafts say *"Target is the 20% CIFAR dataset"* and *"CIFAR-10 with 10% of the data"*. Both
are wrong by a factor of 2.5 — they are 8% and 4%. Fix the prose or make the val/test splits
opt-out.

### 3.5 The one-class comparison confounds the effect with sample size — **FIXED**

`CIFAR_one_class.ipynb` compares n=24 000 against n=2 400. Histogram-KL is biased upward when the
two samples have different sizes (the smaller sample has more empty and more noisy bins), and
that bias is amplified by §3.1. The notebook contains a commented-out line that would have
matched the sizes:

```python
dataset_proportion_target = dataset_proportion_test*(len(test_labels)/len(target_labels))
dataset_proportion_target = 1   # <- this overwrites it
```

The control (§2.1 row 1) *is* size-matched, so the comparison between the control and this row is
not clean. Rerun size-matched — the effect will very likely survive, but as it stands 0.136 vs
0.009 is not a controlled contrast.

### 3.6 CIFAR is upsampled 8× before it is scored — **NOW CONFIGURABLE, AND IT MATTERS**

`pipeline/main.py:130` constructs every scorer with `im_size=(256, 256)`, and
`base_scorer_torch._preprocess_image` bilinearly resizes to that. CIFAR-10 is 32×32, so **every
CIFAR image is upsampled 8× before it reaches the autoencoder**, which destroys exactly the
high-frequency content a compression AE's code responds to. Note that `entropy_AE.py`'s own
`__main__` and `base_scorer_torch`'s default both use `(128, 128)` — the 256 is a hardcoded
override in the pipeline with no comment.

`models/README.md` is written for the 128 setting — *"8x8 -- flatten this ... 64 embedding"*.
Verified: at `im_size=128` the latent is `(64, 8, 8)` = 4 096 codes, exactly as documented; at the
256 the pipeline actually uses it is `(64, 16, 16)` = **16 384 codes**. So the project's own
documentation describes a configuration that no experiment was run in.

This is a confound across the whole results table, because the datasets have different native
resolutions (CIFAR 32, ImageNet64 64, Caltech/DTD variable, uniform noise generated at 32).
**Some of the measured "distribution difference" may be a resolution difference.** Cheapest
control available and it should be run before the paper: repeat §2.1 at 32, 64 and 128.

### 3.7 String labels crash — **FIXED**

`datasets/torch_loaders.py:21` calls `loader.get_labels()`; the method is called
`get_str_labels()` (`abstract_dataset.py:162`, `ImageNet64.py:53`). Any call passing string class
names raises `AttributeError`. Numerical labels work, which is why the notebooks never hit it.

### 3.8 `dataset='KODAK'` raises `KeyError` — **FIXED**

`get_all_loaders` special-cases lowercase `'kodak'` (`torch_loaders.py:39`) but `DATA_LOADER`,
`DATASET_PROPORTIONS` and the `get_scores(help=True)` text all use `'KODAK'`. `TOTAL_INSTANCES`
has no `KODAK` entry at all, so the documented spelling falls through to
`TOTAL_INSTANCES['KODAK']` → `KeyError`.

### 3.9 CIFAR-100 and DTD are unreachable — **FIXED**

Both have complete `loader.py` / `downloader.py` / `VARS.py`, and the draft lists them as
available datasets — but neither is in `DATA_LOADER` / `TOTAL_INSTANCES` /
`DATASET_PROPORTIONS` in `datasets/__init__.py`. Three lines each to wire up.
Caltech-101/256 *are* registered but have never appeared in a finished experiment (§4).

### 3.10 Scores are duplicated for any 2-D scorer — **FIXED**

`pipeline/main.py:252`:

```python
if len(score.shape) == 2:
    for s in score:
        scores.append(s)
    scores.append(score)   # appends the whole batch again
```

Latent, not active: the only registered scorers return 1-D. It fires the moment `centers=5` or
`spacial=True` is used — which is exactly the direction §5.2 points in.

### 3.11 Mutable default argument

`abstract_dataset_torch.__init__(self, ..., image_dict={})`. Every loader constructed without an
explicit `image_dict` shares one dict for the lifetime of the process. `torch_loaders` always
passes one explicitly so it is currently harmless, but it will silently leak images between
datasets the first time someone constructs a loader directly.

### 3.12 Smaller things

- `pipeline/main.py:161` disables the image cache for the test dataloader
  (`pre_loaded_images={}` with the real argument commented out). Every test run re-reads and
  re-processes from disk — this is most of the 4–6 s/iteration in the DiD runs.
- `main.py:93` comment says *"change to full dataset size if not in dev mode"*; the code does the
  opposite (`dev=True` shrinks, and silently overrides any user-supplied proportion).
- `'KL' in test` is a substring match against a string — works, but `test='KL_something'` would
  silently match.
- `different_datasets/caltech_differences.ipynb`: `max(1, run1['dataset_proportion']*2)` should be
  `min(...)`; `max` pins it to 1 whatever the input.
- `UNIFORM_LOADER.__getitem__` generates a **fresh** random image on every access, so it is not a
  fixed dataset and is unaffected by `seed` — two "different subsamples" of UNIFORM are just two
  fresh draws.
- `datasets/CIFAR_10/__init__,py` — comma instead of a dot. Works only because of namespace
  packages.
- `brisque_scorer.py`'s docstring is copy-pasted from `entropy_AE.py` and describes the wrong model.
- `pipeline/distributions.py` is dead except for `notebooks/dev/iqm_distortion_plot_distributions.ipynb`
  (the older per-image IQM-sensitivity approach, superseded by the scorer pipeline).

---

## 4. Experiments that were started and never finished

- **`different_datasets/caltech_differences.ipynb`** — aborted at 9/100 iterations at ~40 s each
  (≈1 h remaining). Caltech-256 vs Caltech-101 is the only *natural-vs-natural, different-source*
  DiD comparison and it has no result.
- **Caltech-101/256, MNIST** are registered and downloadable but appear in no finished experiment.
- **CIFAR-100, DTD** — loaders written, never wired up (§3.9), never run.
- **`notebooks/dataset_experiments/AE_representation-results.ipynb`** existed at commit `03e9871`
  and is not in `HEAD`. Recover with `git show 03e9871:notebooks/dataset_experiments/AE_representation-results.ipynb`
  if it held results worth keeping.
- **The other 14 autoencoders are unused.** `save_nets/` has `{mse,ssim,nlpd,mae} × {2,5} centres
  × {natural,uniform}`, and `models/README.md` calls `nlpd-2/5-u` "interesting" — but `SCORERS`
  registers only `entropy-2-mse`, so every result in the project comes from one checkpoint.
  (`models/README.md` also notes the `mae` weights don't work and should be ignored.)

---

## 5. What to do next

### 5.0 Done in the August 2026 pass

- ✅ **Real hypothesis tests** — KS / Cramér–von Mises / Anderson–Darling, plus JS and
  Wasserstein, in `pipeline/h_tests.py`. `get_scores(test='all')` runs the lot.
- ✅ **Permutation null** — `h_tests.permutation_test` gives a distribution-free p-value for any
  statistic, including the binned ones.
- ✅ **KL estimator fixed** (§3.1) — smoothing moved onto the counts, verified scale-invariant.
- ✅ **DiD legends fixed** (§3.2) in all four notebooks; the stale outputs were cleared.
- ✅ **Baseline controls run** (§2.4) — and they *beat* the autoencoder. See below.
- ✅ **Resolution control run** (§2.5) — and it changes the interpretation. See below.
- ✅ **Size-matched re-run** of every comparison (§2.2).
- ✅ **Power curve** (§2.6) — the n needed to detect each shift.
- ✅ **Calibration measured** over 500 null draws (§2.7).
- ✅ **CIFAR-100, DTD, KODAK registered**; string labels, the 2-D score duplication, and the
  `dataset_proportion` semantics all fixed.
- ✅ **Results persist to `results/*.csv`** instead of living in notebook output cells.

### 5.1 Tier 1 — what the August results now force

1. **Decide what the paper claims, because "perceptual metrics are the right space" is no longer
   supported.** §2.4 shows a JPEG byte count beating the autoencoder on every comparison at every
   resolution. Two honest options:
   - **Reframe around the pipeline.** The contribution becomes the *procedure* — reduce images to
     a scalar with any scorer, run a calibrated two-sample test, report a p-value and the sample
     size needed. The scorer is a pluggable component, and the paper reports that a trivial one
     wins. This is defensible, and the baseline table becomes a feature rather than a threat.
   - **Rescue the perceptual scorer** by not throwing away the code (§5.2.6). The current
     statistic compresses 16 384 quantised codes to one scalar; a multivariate comparison is the
     obvious next move and is where the AE could plausibly beat a byte count.
   Either way the baseline table has to be *in* the paper.
2. **Justify or drop `im_size=256`.** §2.5 shows the method's power depends on an undocumented
   8× upsample and largely evaporates at native resolution. If 256 is kept it needs an argument;
   if not, the headline results weaken substantially.
3. **Report the conservatism.** Calibration is measured (§2.7): the false-positive rate is fine
   at 3.6% [2.1%, 5.6%], but the null p-values are non-uniform because the score is discrete, so
   the tests are conservative. Say so, and use permutation p-values for marginal cells.
4. **Match native resolution in every comparison, or report both.** CIFAR-vs-DTD and
   CIFAR-vs-ImageNet at `im_size=256` are substantially resolution comparisons for *any* scorer;
   the honest version either downsamples everything to 32 or reports both and says which is which.

### 5.2 Tier 2 — cheap and likely to strengthen the result

6. **Stop throwing away the code.** The current statistic compresses a 64×16×16 quantised code to
   one scalar. Try (a) per-channel `+1`-ratios → a 64-D vector, compared with MMD, energy
   distance, or a classifier two-sample test (C2ST); (b) `spacial=True`, already implemented in
   `entropy_AE.py` and never used; (c) `centers=5` — note §3.10 fires here and
   `counts_per_emb_feature_flat` has a bare `# TODO: center 5` that returns raw embeddings.
   A C2ST also gives a p-value and an interpretable effect size (accuracy above 0.5) for free.
7. **Sensitivity to the scorer.** Rerun §2.2 across the other 14 checkpoints — especially the
   uniform-trained ones. If `nlpd-2-u` (trained on *noise*, with a perceptual loss) works as well
   as `mse-2` trained on natural images, that is a much more interesting paper than the current
   one, and the sibling project `~/projects/percept_reduce` found exactly that pattern
   (uniform-noise-trained encoders recovering ~91% of MSE performance on a downstream probe).
8. **A dataset × dataset KL/p-value matrix.** Wire up CIFAR-100 and DTD (§3.9) and produce one
   heatmap over {CIFAR-10, CIFAR-100, Caltech-101, Caltech-256, DTD, ImageNet64, MNIST, uniform}.
   Single figure, carries the whole "measure how close two datasets are" claim, and the loaders
   already exist.
9. **A power curve — this is the practically useful deliverable.** Contaminate CIFAR with a
   fraction *f* ∈ {0, 0.01, 0.02, 0.05, 0.1, 0.2} of uniform noise (or of another dataset) and
   plot detection rate vs *f* vs *n*. It answers "how big a shift can this detect with how many
   images", which is what turns the method into the data-collection stopping criterion the
   drafts propose.

### 5.3 Tier 3 — new directions the code supports

10. **Class-count sweep.** §2.1 only removes 9 of 10 classes. Sweep 1→10 classes retained; a
    monotone KL curve would be a strong, cheap figure.
11. **Re-examine difference-in-difference properly** — give target and test disjoint subsamples
    (`shift_seed_test`), apply the transform on one side consistently, then decide. Or drop it,
    per the Jan-2025 draft.
12. **PixelCNN / explicit density model** as a scorer, per the Ideas draft — the natural
    comparison against the AE-code statistic.

### 5.4 Engineering

13. Fix §3.7–3.12 (all small).
14. Notebooks recompute everything from scratch and store results only in output cells; nothing is
    saved to disk. Have `run_multiple_pipelines*` write a CSV per run (config + results, like
    `percept_reduce/saves/`) so a 10-minute run is not lost to a kernel restart, and so figures
    can be regenerated without recomputation.
15. There are no tests. A single smoke test (`dev=True` end-to-end on CIFAR for each registered
    scorer and transform) would have caught §3.7 and §3.8.
16. ~~`../H-Test-IQM-bug-fix` can be deleted~~ — **done 2026-08-06**, nothing lost. See `README.md`.

---

## 6. The three Overleaf drafts

In `~/Repos/Overleaf/percept-reduce/`. All three are separate Overleaf git projects.

| draft | last commit | what it is |
|---|---|---|
| `Perceptual-Tests-Ideas` | 2024-12-16 | July-2024 brainstorm. Three framings of the idea, the list of non-parametric tests (§2.3), the ε-noise/IQM-Vis framing, and the open questions. Not a paper — but it is where the unimplemented ideas live. |
| `Perceptual-Tests-2` | 2024-12-13 | First full write-up. Superseded. |
| **`Perceptual-Tests-Densities`** | **2025-01-13** | **The current draft.** Prose tightened, a "Uses" section added, additive-noise/encodings sections commented out, and difference-in-difference demoted to *"probably not to be included for the short paper"*. Start here. |

The `Densities` draft is a short paper built on §2.1 alone, which — given §2.2 — is the right
call. What it still needs, in order: an actual test with a p-value (§5.1.1–2), the baseline
controls (§5.1.4), and the corrected figures (§3.1–3.2).

Neither `main.tex` has a bibliography — "cite Alex", "cite ?Branco?" are still inline
placeholders. The autoencoder is Alex Hepburn's; the reference needs chasing before submission.

## 7. Relationship to `~/projects/percept_reduce`

A separate, independently useful codebase by the same author, sharing this project's Overleaf
parent directory and the same compressive-AE lineage. It asks whether training an autoencoder
with a *perceptual* loss makes its latent space more useful downstream, and whether that reduces
the data needed. It has its own `FINDINGS.md` / `TODO.md`.

The two projects do not import each other, but two of its results bear directly on this one:

- encoders trained on **pure uniform noise** recover ~91% of full-data MSE performance on a
  downstream probe — i.e. a lot of what looks like "learned natural-image statistics" is just
  conv-stack architecture. That is a direct warning for §5.1.4 and §5.2.7 here.
- perceptual losses reshape the latent space substantially versus MSE. Since every result in this
  project comes from the single `mse-2` checkpoint, the scorer axis is unexplored.
