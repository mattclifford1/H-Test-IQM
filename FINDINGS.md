# Findings

State of the `H-Test-IQM` project. Originally reconstructed from the dormant repo at commit
`111560d` (2024-12-13); **§2.2 and §2.4–2.7 are from the August 2026 re-run** after the
methodological fixes in §3 were applied, and **§2b is the six-experiment suite** built on the
score cache, which supersedes §2.4 and §2.5.

Every number is quoted from a **committed notebook output cell**, from a CSV in `results/`, or
was **recomputed here** (with `~/anaconda3/envs/h_data/bin/python`). Nothing is read off a figure.

> **Headline, after the fixes, the re-run, and the six-experiment suite (§2b).**
>
> 1. **The method works and is calibrated.** With real hypothesis tests in place, the disjoint
>    control accepts, every genuine shift rejects at n = 4000, and the false-positive rate is
>    4.1–5.1% against a nominal 5% across the whole n grid at 1000 repeats per cell — mildly
>    conservative at small n, never over-rejecting (§2.7, §2.9.1). There is a power curve saying
>    how many images each shift needs (§2.9.1) and a contamination sweep giving the smallest
>    detectable fraction of foreign data (§2.9.2). That part is defensible.
> 2. **The autoencoder's *scalar* loses to a JPEG byte count — but the autoencoder does not.**
>    Averaging the 64×16×16 latent into one number is what costs it. Keeping one `+1`-ratio per
>    channel and testing in R^64 takes CIFAR-10 vs CIFAR-100 from 41.5% to **88%** detection
>    where JPEG bytes manage 19% (§2.9.5). §2.4 is real but is a finding about the statistic,
>    not the scorer.
> 3. **The upsampling dependence has a mechanism, and it is not perception.** At 32px the latent
>    is only 256 bits, so the occupancy score cannot have a standard deviation below the binomial
>    floor of 0.031 — and every dataset, uniform noise included, sits exactly on it. The 8×
>    upsample buys latent elements to average over, nothing more (§2.9.3).
> 4. **The stated premise fails outright.** Autoencoders trained on *uniform noise* discriminate
>    these datasets as well as ones trained on natural images — 100.1% recovery (§2.9.6). The
>    statistic is not measuring absorbed natural-image statistics.
> 5. **Two of the four headline comparisons are confounded.** `cifar-vs-imagenet64` is largely a
>    resampling artifact (§2.9.3), and `cifar-vs-oneclass` — the only resampling-clean one — gives
>    the autoencoder *no signal at all* at native resolution.
>
> The redirect: the defensible contributions are the **pipeline** (score → two-sample test →
> calibrated p-value) and the **multivariate result** (§2.9.5), which is the one place the
> perceptual scorer beats the trivial baselines. The premise, the scalar statistic, and the
> checkpoint choice all need rewriting. See §5.

> **Update 2026-09-11 — exp7 (§2.9.7) changes points 2 and 4.** Crossing the two was
> pre-registered (`experiments/PREREGISTRATION.md`) and it came out a way no row of the
> registered reading table anticipated:
>
> - **An untrained, randomly initialised encoder is the best detector of all.** On CIFAR-10 vs
>   CIFAR-100 at n = 500, 64-D C2ST: untrained **82.5%**, natural-trained 49.3%, noise-trained
>   35.2%, JPEG bytes 10.5%. Every random seed beats every trained encoder, and the same
>   ordering holds on the effect size of every other comparison.
> - So point 2 needs restating: the multivariate win is **not** evidence for the perceptual
>   autoencoder. It is evidence for *a random convolutional projection, quantised and kept as
>   a vector*. Training on anything costs detection power; training on natural images costs
>   less than training on noise.
> - Point 4's "100.1% recovery" was a mean dominated by easy and resampling-confounded
>   comparisons. On the one hard, clean pair, natural training beats noise training
>   (88% vs 67% at the 64-D code) — so the premise is not simply false. But it is beside the
>   point, since the untrained encoder beats both.
> - The class-drop use case is **not** rescued by the code: every 64-D encoder detects a dropped
>   class 13–20% of the time at n = 2000, against 32% for JPEG bytes.
> - Methodological (§2.9.8): the disjoint-partition control used everywhere reports a
>   false-positive rate *conditional on one fixed split*. Redrawn per repeat, every scorer is
>   calibrated. The project's calibration claims survive, but the protocol should change.

> **Update 2026-09-11 — exp8 (§2.9.9), the featuriser ladder.** Pre-registered; every
> prediction held. An **ImageNet-pretrained ResNet-18** detects CIFAR-10 vs CIFAR-100 88.5% of
> the time at n = 100, where the untrained AE encoder manages 16% and needs n ≈ 500 to reach
> 80%. It is also the first featuriser to beat JPEG bytes at dropping one class (42–53% vs
> 29%). Untrained *convolutional* features — the AE's or a ResNet's — are a solid middle rung
> (76–84% at n = 500), far above colour moments (29%) or a random pixel projection (22%). So
> exp7's "untrained beats trained" is about **reconstruction training**, not learning: the
> perceptual autoencoder is dominated on every comparison by an off-the-shelf network.

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

`results/prior/density_grid.csv`, `experiments/prior/run_density_grid.py`. n = 4000 **per side, size-matched
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

`results/prior/density_grid.csv`. Same six comparisons, same n = 4000, five scorers. **KS statistic**
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

`results/prior/resolution_sweep.csv`. Same comparisons, `entropy-2-mse`, sweeping `im_size`. KS:

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

`results/prior/power_curve.csv`, `experiments/prior/run_power_curve.py`. `entropy-2-mse`, `im_size=256`,
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

`results/prior/calibration.csv`, `experiments/prior/run_calibration.py`. Both sides drawn from disjoint halves
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

## 2b. The six-experiment suite (August 2026, second pass)

Everything below comes from `results/exp*.csv`, written by `experiments/exp1..exp6`. They all
read `results/score_cache/`, so every cell is a resample of scores computed once — which is what
makes 1000 repeats per cell affordable where the first pass could only manage 10.

**These supersede §2.4 and §2.5.** §2.4's "a JPEG byte count beats the autoencoder" survives as
a statement about *the scalar*, but §2.9.5 shows it does not survive as a statement about the
autoencoder. §2.5's upsampling result is confirmed and given a mechanism in §2.9.3.

### 2.9.1 Power (exp1) — calibrated at 1000 repeats

False-positive rate on `control-disjoint`, `entropy-2-mse`, 1000 repeats per cell:

| n | 25 | 50 | 100 | 200 | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|---|---|---|
| FP | 4.1% | 4.3% | **1.9%** | 4.1% | 4.6% | 5.1% | 4.7% | 4.6% |

Seven of eight sit on 5%. The n = 100 cell is conservative because the KS statistic is discrete
at small n, so the attainable p-values straddle 0.05 rather than hitting it — expected behaviour
of the test, consistent with §2.7, and it makes small-n power look slightly worse than a
perfectly-calibrated test would.

Smallest *tested* n reaching 80% detection (grid: 25, 50, 100, 200, 500, 1000, 2000, 4000):

| comparison | entropy-2-mse | jpeg_bytes | pixel_std | pixel_entropy | BRISQUE |
|---|---|---|---|---|---|
| control-disjoint | >4000 | >4000 | >4000 | >4000 | >4000 |
| cifar10-vs-cifar100 | 4000 | 4000 | 4000 | **1000** | 2000 |
| cifar-vs-dtd | 4000 | **25** | 500 | 200 | **25** |
| cifar-vs-imagenet64 | 500 | **25** | 500 | 2000 | **25** |
| cifar-vs-oneclass | 200 | 100 | 100 | 100 | 200 |
| cifar-vs-uniform | 50 | **25** | 25 | 25 | 25 |

Figure: `figures/fig1_power_curve.png`.

### 2.9.2 Contamination (exp2) — the stopping-criterion result

CIFAR-10 with a fraction *f* of its images replaced. Smallest *f* reaching 80% power at
n = 4000, over 500 repeats per cell:

| contaminant | jpeg_bytes | entropy-2-mse |
|---|---|---|
| uniform noise | 5% | 10% |
| DTD | 5% | 100% |
| ImageNet64 | 5% | 50% |
| CIFAR-100 | never | never |

All 1081 cells have an f = 0 false-positive rate near 5%, so the sweep is calibrated throughout —
that column is a built-in null replicated 24 times. Neither scorer detects CIFAR-100
contamination at any fraction, which is a fair negative: those distributions really are close.

This is the experiment that turns the method into the data-collection stopping criterion the
drafts propose. Everything else compares two fixed datasets and can only answer "different:
yes/no". Figure: `figures/fig2_contamination.png`.

### 2.9.3 Resolution (exp3) — the mechanism behind §2.5

KS statistic by `im_size`, 100 repeats, n = 4000. The control is flat (0.0172 → 0.0188), so the
rise is signal, not a null that drifts:

| comparison | 32px | 64px | 128px | 256px |
|---|---|---|---|---|
| **entropy-2-mse** | | | | |
| cifar10-vs-cifar100 | 0.0243 | 0.0344 | 0.0406 | 0.0478 |
| cifar-vs-dtd | 0.0305 | 0.0501 | 0.0494 | 0.0448 |
| cifar-vs-imagenet64 | 0.0404 | 0.0389 | 0.0731 | 0.1082 |
| cifar-vs-oneclass | 0.0186 | 0.0444 | 0.0782 | 0.1453 |
| cifar-vs-uniform | 0.0360 | 0.0780 | 0.1949 | 0.3099 |
| **jpeg_bytes** | | | | |
| cifar-vs-dtd | 0.3045 | 0.6407 | 0.8500 | 0.9567 |
| cifar-vs-imagenet64 | 0.0922 | 0.8019 | 0.6057 | 0.6678 |
| cifar-vs-oneclass | 0.2846 | 0.2552 | 0.2536 | 0.2462 |
| cifar-vs-uniform | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**The mechanism is a binomial floor, not perception.** The autoencoder downsamples by 16, so a
32px input produces a 2×2×64 latent — 256 bits. The score is the fraction of `+1`s among them,
so its standard deviation cannot fall below √(0.25/256) = 0.0312. Measured per-image standard
deviations at 32px:

| dataset | 32px sd | 256px sd |
|---|---|---|
| CIFAR_10 | 0.0303 | 0.0125 |
| CIFAR_100 | 0.0297 | 0.0135 |
| IMAGENET64_VAL | 0.0294 | 0.0120 |
| DTD | 0.0279 | 0.0131 |
| UNIFORM | 0.0310 | 0.0032 |

Every dataset sits on the floor, uniform noise included, and the statistic takes only **64
distinct values across 60 000 images** (1518 at 256px). At 256px the latent is 16×16×64 = 16 384
bits, the floor drops to 0.0039, and real structure appears — uniform noise collapses to 0.0032
while natural images spread to 0.012–0.013. That dispersion gap is the entire signal, consistent
with the score being centred on 0.5 for every dataset.

So §2.5 is right that the power depends on the upsample, but the reason is not that upsampling
adds perceptual information — it is that **the occupancy statistic needs enough latent elements
to average over**, and 8× upsampling is how it accidentally gets them. This is a statement about
the estimator, not about natural-image statistics, and it predicts the effect should be absent
for any scorer whose output is not a proportion over code elements. `jpeg_bytes` confirms that
prediction: its one-class detection is *flat* in resolution (0.2846 → 0.2462).

**Two consequences for the headline table.**

1. `cifar-vs-oneclass` is the only comparison with no resampling asymmetry at all — both sides
   are CIFAR-10 at the same resolution. At native 32px the autoencoder scores **0.0186 against a
   control of 0.0172**, i.e. nothing, while `jpeg_bytes` scores 0.2846.
2. `cifar-vs-imagenet64` is substantially a **resampling artifact**. At 32px (CIFAR native,
   ImageNet downsampled) `jpeg_bytes` finds them nearly indistinguishable at 0.0922; at 64px
   (CIFAR upsampled 2×, ImageNet native) it jumps to 0.8019. A 9× jump from changing only the
   resampling. Only the 32px row of that comparison is trustworthy.

Figures: `figures/fig5_resolution.png`, `figures/fig4_scorer_comparison.png`.

### 2.9.4 Class count (exp4) — the weakest case for the method

The test sample keeps *k* of 10 CIFAR-10 classes; *k* = 10 is the control. n = 2000, 500 draws
per *k* (repeats are scaled up as C(10,k) shrinks, so every *k* is estimated to the same
precision — without that, *k* = 10 rests on a single subset).

Detection at **k = 9** — dropping exactly one class of ten:

| scorer | k = 9 | k = 10 (control) |
|---|---|---|
| jpeg_bytes | 31.4% | 3.0% |
| BRISQUE | 31.4% | 4.4% |
| pixel_entropy | 29.0% | 4.4% |
| pixel_std | 21.6% | 3.8% |
| **entropy-2-mse** | **8.8%** | 6.0% |

The autoencoder is barely above its own false-positive rate. This is exactly the "check that a
newly added class actually adds density" use case from §1, and on it the perceptual scorer is
close to useless while a JPEG byte count works.

Also: between-subset sd at *k* = 1 is 0.0394 against a mean KS of 0.0798. **Which** classes are
kept matters about as much as how many — any single-subset result at small *k* is not reportable.

### 2.9.5 Multivariate (exp5) — this reverses §2.4

The project's statistic averages a 64×16×16 quantised latent into one number. Keeping one
`+1`-ratio per latent *channel* (64-D, the already-implemented `spacial=True` path) and testing
in R^64 with a classifier two-sample test, at n = 1000, 200 repeats:

| comparison | AE 64-D / C2ST | AE scalar / KS | jpeg scalar / KS |
|---|---|---|---|
| control-disjoint | 4.5% | 5.0% | 5.0% |
| **cifar10-vs-cifar100** | **88.0%** | 41.5% | **19.0%** |
| cifar-vs-dtd | 100% | 30.0% | 100% |
| cifar-vs-imagenet64 | 100% | 100% | 100% |
| cifar-vs-oneclass | 100% | 100% | 100% |
| cifar-vs-uniform | 100% | 100% | 100% |

Energy distance and MMD on the same 64-D vectors also reach 100% on every real comparison
(`results/exp5_multivariate_perm.csv`).

**The scalar was the bottleneck, not the autoencoder.** The same forward pass, at the same n,
goes from 41.5% to 100% on ImageNet64 and 30% to 100% on DTD purely by not averaging the code
away. And on `cifar10-vs-cifar100` — the hardest comparison, the only one free of resampling
confounds, and the one that most resembles the intended use case — the 64-D code reaches 88%
where a JPEG byte count manages 19%.

So §2.4's finding is real but narrower than it was stated: **a JPEG byte count beats the
autoencoder's scalar, not the autoencoder.** The perceptual scorer wins where the task is
genuinely perceptual, once it is allowed to keep its representation.

Caveat: the energy/MMD control ran only 20 repeats (2/20 rejections). That is consistent with 5%
but does not demonstrate it — the Wilson interval spans roughly [3%, 30%]. The C2ST arms carry
the calibration evidence at 200 repeats. C2ST here uses a single held-out split, **not**
cross-validation: out-of-fold predictions are not independent given the fitted model, and the
cross-validated version measured a 9% false-positive rate at a nominal 5%. The held-out version
measures 5.7% on 300 null draws.

Figure: `figures/fig7_multivariate.png`.

### 2.9.6 Checkpoints (exp6) — the premise does not survive

All six usable checkpoints, `{mse, ssim, nlpd} × {natural, uniform-noise}`, n = 4000, 100
repeats. Mean KS across the five real comparisons:

| checkpoint | trained on natural | trained on uniform noise | recovery |
|---|---|---|---|
| entropy-2-mse | 0.1310 | 0.1521 | 116% |
| entropy-2-nlpd | 0.1794 | 0.1718 | 96% |
| entropy-2-ssim | 0.1999 | 0.1870 | 94% |
| **overall** | **0.1701** | **0.1703** | **100.1%** |

**Autoencoders trained on uniform noise discriminate these datasets exactly as well as ones
trained on natural images.** The premise in §1 — "an AE trained to reconstruct natural images has
absorbed natural-image statistics, so the code-occupancy statistic is perceptually meaningful" —
is not supported. Whatever the statistic measures, it is a property of the architecture and the
quantiser. The sibling `percept_reduce` project found ~91% recovery on a downstream probe (§7);
here it is 100%.

Second, more immediately actionable: **`mse-2`, the checkpoint used for every result in this
project, is the worst of the three natural-trained ones** (0.1310 against ssim's 0.1999, better
on four of five comparisons). That choice was never tested until now.

> **Qualified 2026-09-11 (§2.9.7).** The 100.1% averages KS over five comparisons, and the
> average is carried by the easy ones — uniform noise at KS ≈ 0.5 — and by DTD and ImageNet64,
> which §2.9.3 shows are resampling artifacts. On CIFAR-10 vs CIFAR-100, the only comparison
> that is both hard and resampling-clean, natural training is ahead: reject rates 1.00 vs 0.33
> (nlpd), 1.00 vs 0.75 (ssim), 0.98 vs 0.97 (mse), mean KS 0.052 vs 0.037 — about 70%
> recovery. exp7 confirms the gap at the 64-D code. The `ssim-2-u` control's 14% in the table
> above is explained in §2.9.8: it is the fixed split, not the scorer.

### 2.9.7 Code origin (exp7, pre-registered) — an untrained encoder is the best detector

**Question.** exp5 ran the 64-D code on one natural-trained encoder; exp6 ran the
noise-vs-natural comparison on the scalar only. Does the code's detection power come from
training on natural images? Pre-registered 2026-09-11 in `experiments/PREREGISTRATION.md`
(commit `5490998`, before the script or its caches existed), with five predictions, three
checks and a table saying how each outcome would be read.

**Design.** Nine encoders, same architecture, 256 px, 64-D per-channel `+1`-ratio code:
natural-trained `{mse, ssim, nlpd}-2`, noise-trained `{mse, ssim, nlpd}-2-u`, and the
architecture **untrained** at seeds 0, 1, 2. The untrained code is the sign pattern of a random
convolutional projection — the encoder ends in a plain conv and quantises to the nearer of
{−1, +1}; the sigmoid is on the decoder. Held-out C2ST (the exp5 protocol), 200 repeats.
`experiments/exp7_code_origin.py`; `results/exp7_code_origin.csv`; `figures/fig8_code_origin`.
Build 5 min, experiment 4 min.

**CIFAR-10 vs CIFAR-100, detection rate (%):**

| encoder | n = 250 | 500 | 1000 |
|---|---|---|---|
| natural `mse` | 28 | 55 | 92 |
| natural `ssim` | 22 | 46 | 84 |
| natural `nlpd` | 22 | 47 | 88 |
| noise `mse-u` | 16 | 32 | 66 |
| noise `ssim-u` | 22 | 46 | 75 |
| noise `nlpd-u` | 15 | 28 | 60 |
| **untrained, seed 0** | **55** | **94** | **100** |
| **untrained, seed 1** | **46** | **82** | **100** |
| **untrained, seed 2** | **32** | **72** | **99** |
| JPEG bytes (scalar KS) | 11 | 10 | 20 |
| *group mean: natural / noise / untrained* | *24 / 18 / 44* | *49 / 35 / 82* | *88 / 67 / 100* |

Against the registration:

- **P1 confirmed** — natural ahead of noise by 21.2 pts at n = 1000, in all three
  objective-matched pairs (mse +26.0, ssim +9.0, nlpd +28.5).
- **P2 refuted, in the opposite direction** — the untrained group is ahead of the
  natural-trained one, by 11.5 pts at n = 1000 where it saturates and by 33 pts at n = 500,
  with every random seed above every trained encoder. The registered expectation was the
  reverse.
- **P4 confirmed directionally, not in substance** — see the class drop below.
- The outcome, untrained > natural > noise, is **not a row of the registered reading table**,
  and is reported as found.

**The ordering is not specific to one comparison.** CIFAR-10 vs one class saturates at 100%
detection for every encoder, so detection says nothing there, but the effect size — C2ST
held-out accuracy at n = 1000 — orders the groups identically: untrained 0.713–0.750,
natural 0.687–0.703, noise 0.666–0.674, no overlap between groups. The scalar arm agrees too:
at n = 4000 the untrained encoders' *mean occupancy* alone gives KS 0.066–0.078 on CIFAR-10
vs CIFAR-100, against 0.048–0.054 for natural-trained and 0.028–0.046 for noise-trained.

**The class drop is not rescued.** Dropping one of ten CIFAR-10 classes, n = 2000, partitions
redrawn every repeat (§2.9.8), k = 10 null alongside:

| | natural 64-D | noise 64-D | untrained 64-D | JPEG bytes |
|---|---|---|---|---|
| k = 9 detection | 15.4–17.0% | 12.6–14.6% | 18.8–20.2% | **32.2%** |
| k = 10 null | 3.2–7.2% | 3.2–5.6% | 4.6–5.6% | 5.0% |

Same ordering, but every code lands at about half of what a JPEG byte count manages. The
64-D code roughly doubles the natural `mse` scalar (17.0% vs 9.0%), which is what P4
registered, and it does not come close to making "does a new class add density" a use case
this method serves. Scalar class-drop detection is also a lottery across random seeds —
33.4%, 5.8% and 27.2% for seeds 0, 1, 2 — which is the clearest demonstration that one
mean-occupancy number is at the mercy of which projection it happens to be.

**Checks, and where the registration was not followed as written.**

- Check 1 (the 64-D row mean equals the cached scalar) passed with a maximum difference of
  exactly 0.0 — the caches are row-aligned and the random encoders' scalars are legitimate.
- Check 2 failed as worded: the untrained encoders have 1–3 constant channels. The criterion
  was mis-specified — natural-trained encoders have *more* dead channels (mse 2, nlpd 8, ssim
  11) and noise-trained have none — so it was not applied. Live-channel counts are below.
- Check 3 failed for `random-s0` (3.6% [2.6%, 4.9%], conservative). The result holds with it
  excluded; §2.9.8 shows it is calibrated under a redrawn partition (5.2%).
- The `ssim-2-u` decision rule fired (8.7% [7.1%, 10.6%]); it is excluded from scalar
  conclusions. §2.9.8 shows the cause was the fixed split (redrawn: 4.8%).

**Why might an untrained encoder win? (exploratory, `exp7_diagnostics.py`)** Part of it is
dimension. Participation ratio of each code's covariance on CIFAR-10:

| | live channels | effective dimension | C2ST acc. |
|---|---|---|---|
| natural mse / ssim / nlpd | 62 / 53 / 56 | 2.69 / 2.46 / 2.88 | 0.547 / 0.542 / 0.545 |
| noise mse-u / ssim-u / nlpd-u | 64 / 64 / 64 | 1.51 / 1.93 / 2.04 | 0.534 / 0.539 / 0.531 |
| untrained s0 / s1 / s2 | 61 / 62 / 63 | 2.84 / 2.30 / 3.18 | 0.586 / 0.574 / 0.565 |

Effective dimension tracks accuracy (Spearman 0.67, p = 0.05, 9 encoders) but does not
explain it: seed 1 has a lower effective dimension than natural `mse` or `nlpd` and beats
both. Noise-trained codes are the striking column — all 64 channels alive but nearly
one-dimensional, i.e. heavily redundant. A reading consistent with everything here, **not
tested**: reconstruction training spends the code on what reconstruction needs and discards
the rest; training on noise commits the code hardest; an untrained projection stays
uncommitted and so keeps more of the low-level statistics that distinguish two datasets. That
is also consistent with random features being a standard tool in kernel two-sample testing.

**What it means for the project.** The natural-image premise is not simply false — at the
representation that carries the signal, natural training beats noise training on the hardest
comparison. But it no longer matters, because no training beats both. The contribution that
survives is a calibrated two-sample pipeline on a **random convolutional featuriser, kept
multivariate** — which invites the question every reviewer will ask first: how does it compare
with other featurisers, from trivial colour statistics up to pretrained networks? That is the
next experiment (§5.1).

### 2.9.8 The disjoint-partition control reports a conditional rate (exploratory)

`control-disjoint` compares the two fixed halves of the cached CIFAR-10 pool (30 000 each), and
every repeat re-draws from those same two halves. At n = 4000 a draw is 13% of its half, so the
realised difference between the halves is a fixed offset shared by every repeat. The
false-positive rate measured is then the rate **conditional on that one split**, not the
unconditional rate.

Evidence, from `exp7_diagnostics.py`, scalar KS at n = 4000, 1000 repeats, 10 scorers:

- The realised KS between each scorer's two full halves predicts its fixed-split
  false-positive rate: **Spearman 0.86, p = 0.001**. Scorers whose halves happen to differ more
  reject more — both of the "anomalous" ones (`ssim-2-u` 8.7%, `nlpd-2` 7.5%) have the largest
  split differences.
- Redrawing the partition on every repeat: spread across scorers falls from **2.0%** to
  **0.42%** — now under the binomial 0.69% — the mean goes 5.2% → 4.3% (slightly conservative,
  as expected from ties in a discrete statistic), and intervals excluding 5% fall from 7/10 to
  1/10. `ssim-2-u`: 8.7% → 4.8%.
- The 64-D C2ST controls at n = 1000 (3% of a half) were already much closer to 5% under the
  fixed split — exactly as the account predicts, since the offset shrinks with n / pool.

**Consequences.** Nothing previously claimed as calibrated turns out not to be: under a
redrawn partition every scorer sits at or just under 5%. exp4's class-drop numbers, which used
fixed partitions, move little when redone (mse scalar 8.8% → 9.0%, JPEG 31.4% → 32.2%). But
exp6's 14% for `ssim-2-u` was this effect, and any future control should redraw the partition
per repeat — the fix is ten lines (`exp7_diagnostics.redraw_control`), and it should replace
`control-disjoint` in `common.py`.

### 2.9.9 The featuriser ladder (exp8, pre-registered) — pretrained features dominate

**Question.** exp7's untrained encoder could be winning because the difference is low-level,
because reconstruction training specifically discards information, or because untrained
convolutional features really are competitive. Pre-registered 2026-09-11
(`experiments/PREREGISTRATION.md`, commit `e3c3ee8`, before the script or its scorers existed).

**Design.** Every featuriser through the same held-out logistic C2ST, every null redrawn per
repeat (§2.9.8). `experiments/exp8_featuriser_ladder.py`, featurisers in
`h_test_IQM/scorers/featurisers.py`; `results/exp8_featuriser_ladder.csv`;
`figures/fig9_featuriser_ladder`. Caches under 5 min, experiment 4 min.

**CIFAR-10 vs CIFAR-100, detection rate (%) — rung means, seeds or channel subsets in
brackets:**

| rung | featuriser | n = 100 | 250 | 500 | 1000 |
|---|---|---|---|---|---|
| ref | JPEG bytes (scalar KS) | 3 | 10 | 13 | 15 |
| 2 | random pixel projection, 64-D | 8 | 11 | 22 (22/18/24) | 49 |
| 1 | colour moments, 6-D | 9 | 17 | 29 | 53 |
| 3q | untrained AE, unquantised | 12 | 24 | 51 (56/46/51) | 90 |
| 4 | natural AE `mse-2`, occupancy | 12 | 23 | 54 | 89 |
| 4q | natural AE `mse-2`, unquantised | 14 | 29 | 66 | 97 |
| 5 | untrained ResNet-18, 512-D | 14 | 43 | 76 (80/76/72) | 99 |
| 3 | **untrained AE, occupancy (exp7's winner)** | 16 | 44 | **81** (93/82/68) | 100 |
| 5m | untrained ResNet-18, 64 channels | 16 | 38 | 84 (84/89/80) | 99 |
| 6m | ImageNet ResNet-18, 64 channels | 54 | 98 | 100 | 100 |
| 6 | **ImageNet ResNet-18, 512-D** | **89** | **100** | 100 | 100 |

Rung 3 replicates exp7 (93/82/68 against 94/82/72 there, on independent draws).

**Against the registration — every prediction held**, landing in the registered row "rung 6
ahead of rung 3 and of rung 5":

- **P1** ImageNet beats the untrained AE: +19.3 pts at n = 500, every member, also at matched
  64-D. The gap is widest at small n — 88.5% vs 16.0% at n = 100.
- **P2** not colour: rung 3 beats colour moments by 51.7 pts. **P3** convolution matters: it
  beats a random pixel projection by 59.0 pts.
- **P4** ImageNet beats the *untrained* ResNet: +24.0 pts (512-D), +15.8 (64-D). exp7's
  "untrained beats trained" does not generalise to ImageNet pretraining.
- **P5** the class drop: ImageNet 42.4% [38.1, 46.8] against JPEG 29.4% [25.6, 33.5] — the
  first featuriser to clear it, with 64-channel subsets reaching 49.6–55.4%. Nothing below
  rung 6 clears JPEG.
- **P6** 23 of 23 redrawn nulls calibrated (4.0–6.3%).
- The pre-registered identity check failed as first run and passed under matched conditions:
  re-scoring at batch 64 flipped single code elements (max 1/256, 0.05% of cells); at the
  cache's batch size of 32 the match is exact. The code is unchanged — cuDNN picks different
  algorithms by batch size, and 0.017% of activations sit within 1e-5 of the quantiser
  boundary. **Batch size is part of a cached code's definition**; keep it at 32.

**The quantiser (registered with no prediction).** For the untrained encoder, sign
quantisation *helps* a lot: occupancy 80.7% vs unquantised activation 50.8% at n = 500, in
every seed. For the natural-trained encoder it *hurts*: 54.0% vs 65.5%. On the class drop the
unquantised version wins for both. No account is offered; it is recorded.

**Exploratory observations, not registered:**

- **Sample efficiency is the headline number.** ImageNet features at n = 100 (88.5%) outperform
  the untrained AE at n = 500 (80.7%) — a 5× saving in images — and the perceptual AE at
  n = 500 (54.0%) by a wider margin still.
- **Untrained convolutional features are architecture-robust.** The untrained AE and the
  untrained ResNet land within about 6 points of each other at every n, and at n = 250–500
  detect roughly three times as often as the pixel-level rungs. "Random convolutional features" is a real, reproducible middle rung — just not
  the top one.
- **Dimension against n matters for the C2ST.** 64-channel subsets beat full 512-D features on
  the class drop (53.1% vs 42.4%) and for the untrained ResNet at n = 500, but lose badly at
  n = 100 (54% vs 89%). The logistic C2ST trains on n rows, so its regularisation is doing
  real work and has never been tuned. Any featuriser comparison is partly a comparison of
  how well a fixed classifier copes with that featuriser's dimension.
- **Relation to prior work.** A C2ST on pretrained deep features is, as far as I recall, the
  setting of Lopez-Paz & Oquab, *Revisiting classifier two-sample tests* (ICLR 2017) — so rung
  6's strength is expected rather than a discovery. **To verify and cite.** What this study
  adds is the calibration, the sample-size characterisation, and the negative result for
  reconstruction-trained perceptual features.

**What it means for the project.** The perceptual autoencoder is dominated on every
comparison run here by an off-the-shelf ImageNet ResNet-18, and beaten even by the same
architecture untrained. The natural-image premise was the wrong axis: what matters is what a
representation was trained *for*, and reconstruction is a poor objective for detecting
distribution shift. A paper that survives this is a practical one — *how many images does it
take to detect a dataset shift, and with what features* — with calibrated tests, a sample-size
table per featuriser, the negative result for reconstruction features, and the
fixed-partition lesson of §2.9.8. It needs more dataset pairs than CIFAR-10/100 before it is
one.

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

Then, in the second pass (§2b):

- ✅ **Score cache built** — every dataset scored once per (scorer, resolution), so experiments
  resample cached arrays and never touch an image. This is what made 1000 repeats per cell
  affordable; the first pass managed 10.
- ✅ **Six experiments run** — power, contamination, resolution, class count, multivariate,
  checkpoints (`experiments/exp1..exp6`, `results/exp*.csv`).
- ✅ **Seven figures regenerable** from the CSVs via `experiments/make_figures.py`.
- ✅ **Multivariate tests implemented** — energy distance, MMD, C2ST in
  `h_test_IQM/pipeline/multivariate.py`, all checked against a true null.

### 5.1 Tier 1 — what the results now force

> **Added 2026-09-11, after exp7 (§2.9.7).** Items 1 and 2 below are superseded in part: the
> multivariate result is real but belongs to the *random* encoder, not the trained one. What
> exp7 forces, in order:
>
> **Done 2026-09-11 — see §2.9.9. 0a below is complete; the next list follows it.**
>
> 0a. **exp8 — a featuriser ladder.** Same pipeline (64-D-ish feature → C2ST, redrawn
>     partitions), featurisers ordered by how much they know about images: per-channel colour
>     moments (6-D, knows nothing) → random projection of raw pixels (no convolution) →
>     random conv encoder (this) → trained AE → pretrained ImageNet network features. It says
>     *what* the untrained encoder is detecting. If colour moments match it, CIFAR-10 vs
>     CIFAR-100 is a low-level-statistics difference and the story is about C2ST, not the
>     featuriser. If pretrained features beat it, the random encoder is a cheap baseline
>     rather than a finding. Either way it is the comparison a reviewer asks for first. The
>     colour-moment and random-pixel scorers need an image pass; CIFAR at 32 px is minutes.
> 0b. **More random seeds, and the quantiser.** Three seeds spread 72–94% at n = 500. Ten would
>     say whether the untrained advantage is a property of the architecture or of lucky
>     draws — and scoring the *unquantised* random features would say whether the sign
>     quantiser helps or hurts.
> 0c. **Replace `control-disjoint` with a redrawn partition** (§2.9.8) in `common.py`, and
>     re-run exp1's calibration row under it.
> 0d. **A comparison between the one-class and CIFAR-100 difficulty levels.** One-class
>     saturates at 100% for every encoder and CIFAR-10 vs CIFAR-100 separates them; the
>     contamination sweep (exp2) at the 64-D code would give the missing intermediate rungs.
>
> **After exp8, in order:**
>
> 1. **More dataset pairs.** Everything in exp7–8 rests on CIFAR-10 vs CIFAR-100 plus a class
>    drop. The practical-study framing needs a spread of shift types and sizes — the
>    contamination sweep (exp2) with the ladder's featurisers is the cheapest route to a
>    graded set, and CIFAR-10.1 / CINIC-10 would add natural "same task, new collection"
>    shifts if they can be sourced.
> 2. **Tune, or replace, the classifier in the C2ST.** Dimension against n visibly matters
>    (§2.9.9). Cross-validated regularisation strength inside the training half, or an MMD
>    with a learned or median-heuristic kernel, before any featuriser ranking is quoted as
>    final.
> 3. **A stronger pretrained rung.** ResNet-18 already saturates at n = 250; one self-supervised
>    featuriser (e.g. a DINO/CLIP image encoder, if available offline) and a larger supervised
>    one would say whether the ceiling is the featuriser or the comparison.
> 4. **Check and cite the prior art** (Lopez-Paz & Oquab 2017 and the FID/two-sample-testing
>    literature) before writing any framing sentence.

The §2b suite answered questions 1 and 6 of the previous list. What it forces instead:

1. **Rebuild the paper around the multivariate result (§2.9.5).** This is now the strongest
   claim available and the only place the perceptual scorer beats the trivial baselines: 88% vs
   19% on CIFAR-10 vs CIFAR-100, the hardest and cleanest comparison. The scalar occupancy
   statistic should be presented as *the thing that was wrong*, with the JPEG baseline as the
   evidence — §2.4 becomes a motivating negative result rather than a threat.
2. **Drop or rewrite the natural-image-statistics premise (§2.9.6).** Noise-trained encoders
   match natural-trained ones at 100.1%. The premise as stated in §1 cannot appear in the paper.
   Either drop the perceptual justification and present the AE as a generic learned featuriser,
   or investigate *why* the architecture alone suffices — the latter is a more interesting paper
   and connects directly to `percept_reduce` (§7).
3. **Switch checkpoint, or explain the choice.** `mse-2` is the worst of the three natural-trained
   checkpoints; `ssim-2` scores 0.1999 against its 0.1310 (§2.9.6). Every result in the project
   uses `mse-2` by inheritance, not by test. Re-run §2b's headline cells on `ssim-2`.
4. **Retire or heavily caveat `cifar-vs-imagenet64` (§2.9.3).** It is substantially a resampling
   artifact — a 9× swing in the JPEG KS from changing only which side gets upsampled. Report the
   32px row or drop the comparison.
5. **Report `im_size` as an estimator parameter, not a preprocessing detail (§2.9.3).** The
   binomial-floor argument means the right framing is "the occupancy statistic needs ≥ N latent
   elements", which is a property of the method that can be stated and defended, unlike an
   unexplained 8× upsample. It also predicts the effect vanishes for non-proportion scorers,
   which `jpeg_bytes` confirms.
6. **Report the conservatism.** The false-positive rate is fine (4.1–5.1% across the n grid,
   §2.9.1), but the KS statistic is discrete, so the n = 100 cell sits at 1.9%. Say so, and use
   permutation p-values for marginal cells.

### 5.2 Tier 2 — cheap and likely to strengthen the result

6. ✅ **Done — stop throwing away the code** (§2.9.5). Per-channel `+1`-ratios → 64-D, compared
   with C2ST, energy distance and MMD. This is now the project's best result. Still untried:
   `centers=5` — §3.10 fires here and `counts_per_emb_feature_flat` has a bare `# TODO: center 5`
   that returns raw embeddings. And the full `spacial` grid (64×16×16) rather than per-channel
   means, which discards spatial layout.
7. ✅ **Done — sensitivity to the checkpoint** (§2.9.6). Answer: the objective barely matters and
   the training *data* does not matter at all. The remaining 9 checkpoints are excluded for
   stated reasons (`centers=5` is unimplemented, `mae` weights do not work).
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
