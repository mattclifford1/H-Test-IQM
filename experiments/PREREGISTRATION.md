# Pre-registrations

Predictions written down **before** the experiment that tests them is run, and not edited
afterwards. A result that is easy to rationalise once seen — "the premise holds, narrowly" —
is exactly the kind that needs this. Git history is the timestamp: each registration is
committed before its script exists.

---

# exp7 — does the structure of the code come from natural images?

Registered **2026-09-11**, before `experiments/exp7_code_origin.py` was written or any of its
caches were built.

## Why this experiment

Two results in `FINDINGS.md` point in opposite directions and have never been crossed:

- **exp5** — keeping the 64-D per-channel code instead of its mean takes CIFAR-10 vs CIFAR-100
  from 41.5% to 88% detection (n = 1000). Run on **one** encoder: `mse-2`, natural-trained.
- **exp6** — encoders trained on uniform noise discriminate as well as natural-trained ones,
  "100.1% recovery". Run on the **scalar** only.

Re-reading `results/exp6_checkpoints.csv` before registering this, the 100.1% is a mean of KS
over five comparisons, and it is dominated by the easy ones (uniform noise, KS ≈ 0.5) and by
two that §2.9.3 shows are resampling artifacts (DTD, ImageNet64). On the one comparison that is
both hard and resampling-clean, CIFAR-10 vs CIFAR-100 at n = 4000, natural training is ahead
for two of three objectives:

| objective | natural reject | noise reject |
|---|---|---|
| mse | 0.98 | 0.97 |
| nlpd | 1.00 | 0.33 |
| ssim | 1.00 | 0.75 |

Mean KS 0.0522 natural vs 0.0367 noise — about 70% recovery, not 100%. So the premise may
fail only where anything works, and partially survive where the method is needed. exp7 asks
the question at the representation that actually carries the signal.

## Design

**Encoders (9), all at 256 px, 64-D per-channel `+1`-ratio code:**

| group | encoders |
|---|---|
| natural | `mse-2`, `ssim-2`, `nlpd-2` |
| noise | `mse-2-u`, `ssim-2-u`, `nlpd-2-u` |
| random | the same architecture, **untrained**, three seeds (0, 1, 2) |

The random group is the null this experiment is built around. The encoder ends in a plain
convolution and quantises to the nearer of the fixed centres {−1, +1}; the sigmoid is on the
*decoder*. So an untrained code is the sign pattern of a random convolutional projection —
non-degenerate, and exactly "architecture + quantiser, no learning". Three seeds, because one
random initialisation is one draw.

**Comparisons:**

| comparison | role |
|---|---|
| `control-disjoint` | false-positive rate, **1000 repeats** at n = 1000 |
| `cifar10-vs-cifar100` | **primary** — hardest comparison, resampling-symmetric (both native 32 px) |
| `cifar-vs-oneclass` | resampling-clean; secondary |
| k = 9 class drop | the "does a new class add density" use case where the scalar scored 8.8% (exp4) |

**Test:** held-out classifier two-sample test (C2ST, logistic regression, the exp5 protocol).
n ∈ {250, 500, 1000}, 200 repeats, for the primary and one-class comparisons. k = 9 at
n = 2000 with 500 draws spread over the 10 possible dropped classes, matching exp4.

**Scalar arm, for completeness:** each encoder's scalar (the mean of its 64-D code, which is
identical to the cached scalar by construction — see Checks) tested with KS on CIFAR-10 vs
CIFAR-100 at n = 4000, 100 repeats. This completes exp6 with the random baseline it lacked.

**Reference arm:** `jpeg_bytes` scalar, KS, same cells.

## Checks that must pass before any result is read

1. **Pipeline identity.** For every checkpoint with both caches, the row mean of the 64-D code
   equals the cached scalar to float32 precision. If it does not, the caches are not
   row-aligned and nothing downstream is valid.
2. **Random codes are non-degenerate.** Mean occupancy of every random encoder lies in
   [0.2, 0.8], and no channel is constant across the CIFAR-10 pool.
3. **Calibration.** Every encoder's 64-D control rejects at a rate whose Wilson 95% interval
   (1000 repeats) contains 5%. An encoder that fails this is excluded from interpretation and
   reported as such.

## The ssim-u anomaly — a decision rule, not a prediction

`ssim-2-u`'s scalar control rejected 14% of true nulls in exp6 (Wilson [8.5%, 22.1%], 100
repeats) where every other checkpoint sits on 5%. I do not know why. exp7 re-runs that exact
cell at 1000 repeats. **If the Wilson interval still excludes 5%, `ssim-2-u` is excluded from
every scalar conclusion** and the cause is investigated separately (first suspect: ties in a
discrete statistic). If it includes 5%, exp6's 14% is recorded as a 100-repeat fluctuation.

## Predictions

All on the primary comparison, CIFAR-10 vs CIFAR-100, 64-D C2ST, at n = 1000 unless stated.
A group's detection rate is the mean over its three members. "Ahead" means the group means
differ by more than **10 percentage points** *and* the gap holds in at least **2 of the 3**
objective-matched pairs (for natural vs noise) or all three random seeds (for comparisons with
random).

**P1 — natural training helps at the representation that matters.** The natural group is
ahead of the noise group. This extends the scalar pattern above to the 64-D code. It fails if
the groups are within 10 points, or noise is ahead.

**P2 — the random encoder is not a strong detector.** The natural group is ahead of the
random group. I expect this less confidently than P1: `~/projects/percept_reduce` found
untrained encoders account for most of a downstream probe's accuracy, and a random
convolutional projection is a known-reasonable featuriser.

**P3 — noise vs random: no prediction.** Whether training on noise does anything beyond
initialisation is the open question, and it is what separates the two failure stories ("image
content does not matter" from "nothing about training matters").

**P4 — the 64-D code rescues the class-drop use case.** At k = 9, n = 2000, natural `mse-2`
64-D detects well above the scalar's 8.8% (exp4). Directional only: I do not predict a size.

**P5 — calibration holds.** Check 3 passes for all nine encoders.

## How the outcome will be read

| primary-comparison outcome | reading |
|---|---|
| natural ahead of both noise and random | the premise survives in a narrow form: natural training shapes *which channels fire*, even though the mean occupancy is indifferent to it |
| natural ≈ noise ≈ random | the 64-D result is architecture + sign quantisation. The honest framing is "a random convolutional featuriser + C2ST is a strong dataset-shift test", and "perceptual" goes |
| natural ≈ noise, both ahead of random | training matters, the training *images* do not |
| anything else | reported as found, without a story fitted to it |

Whichever row it lands in, that row is what gets written up.

---

## Outcome (added 2026-09-11, after running — the registration above is unedited)

Full numbers: `python -m experiments.summarise --only exp7`, written up in `FINDINGS.md`
§2.9.7. Scored against what was registered:

| | outcome |
|---|---|
| Check 1 — pipeline identity | **passed** exactly: max difference 0.0 on all 12 checkpoint × dataset pairs |
| Check 2 — non-degenerate random codes | **failed as worded.** Occupancy is fine (0.43–0.45), but the random encoders have 1–3 constant channels. The criterion was mis-specified: the natural-trained encoders under test have *more* (mse 2, nlpd 8, ssim 11). Proceeded anyway, and say so here; live-channel counts are reported beside every result |
| Check 3 / P5 — calibration | **failed for 1 of 9.** `random-s0` rejected 3.6% [2.6%, 4.9%] — conservative. Excluded from interpretation per the rule; every conclusion below holds without it |
| ssim-u decision rule | **fired**: 8.7% [7.1%, 10.6%] at 1000 repeats. `ssim-2-u` excluded from scalar conclusions |
| **P1** natural ahead of noise | **confirmed.** 88.2% vs 67.0% (+21.2 pts) at n = 1000; ahead in all three objective pairs (mse +26.0, ssim +9.0, nlpd +28.5) |
| **P2** natural ahead of random | **refuted, in the opposite direction.** Random 99.7% vs natural 88.2% at n = 1000 (saturated); at n = 500, 82.5% vs 49.3%, every random seed above every trained encoder. Holds with `random-s0` excluded |
| P3 noise vs random | no prediction registered. Random is far ahead |
| **P4** 64-D rescues k = 9 | **confirmed directionally, not in substance.** Natural mse 64-D 15.2% vs its scalar 7.4% — but JPEG bytes reach 30.8% |

The outcome, untrained > natural > noise, is not a row of the reading table. It falls under
"anything else", and is reported as found.

Two things were done after seeing the results, and they are **exploratory**, in
`experiments/exp7_diagnostics.py`. Both change how two of the rows above should be read, not
what was registered:

- The fixed-partition control measures a rate *conditional on one split*. Redrawing the
  partition every repeat brings `ssim-2-u` to 4.8% and `random-s0` to 5.2% — so both
  exclusions above were conservative, and the scorers are calibrated.
- The class drop inherits the same problem, so it was re-run with redrawn partitions and its
  own k = 10 null. Natural mse 64-D: 17.0% vs scalar 9.0% — P4's reading is unchanged.
