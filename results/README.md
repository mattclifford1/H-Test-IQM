# results/

One CSV per experiment, named after the script that wrote it. **This is the record** — not
notebook output cells, which is where results used to live and where they were destroyed by any
re-run (FINDINGS.md §5.4).

Regenerate the text tables with:

```bash
python -m experiments.summarise                    # everything
python -m experiments.summarise --only exp1 exp3   # one section
python -m experiments.summarise > results/SUMMARY.txt
```

## Current results

| file | written by | one row per |
|---|---|---|
| `exp1_power_curve.csv` | `exp1_power_curve.py` | comparison × scorer × n |
| `exp1_power_curve_raw.csv.gz` | ” | every individual repeat |
| `exp2_contamination.csv` | `exp2_contamination.py` | scorer × contaminant × n × fraction |
| `exp3_resolution.csv` | `exp3_resolution.py` | comparison × scorer × im_size |
| `exp4_class_count.csv` | `exp4_class_count.py` | scorer × k classes retained |
| `exp5_multivariate.csv` | `exp5_multivariate.py` | comparison × representation × test × n |
| `exp5_multivariate_perm.csv` | ” | the energy / MMD cells (fewer repeats) |
| `exp6_checkpoints.csv` | `exp6_checkpoints.py` | comparison × autoencoder checkpoint |
| `exp7_code_origin.csv` | `exp7_code_origin.py` | part × comparison × encoder × test × n (pre-registered) |
| `exp7_checks.csv` | ” | the pre-registered checks, per encoder × dataset |
| `exp7_diagnostics.csv` | `exp7_diagnostics.py` | post-hoc diagnostics — **exploratory** |
| `exp8_featuriser_ladder.csv` | `exp8_featuriser_ladder.py` | part × featuriser arm × n × k (pre-registered) |
| `SUMMARY.txt` | `summarise.py` | — the rendered tables |

The headline numbers these produced are written up in [FINDINGS.md](../FINDINGS.md) §2b. In
short: the pipeline is calibrated (4.1–5.1% false positives across the whole n grid), the
autoencoder's *scalar* loses to a JPEG byte count but its *64-D code* beats it 88% to 19% on the
hardest comparison, and checkpoints trained on uniform noise work exactly as well as ones
trained on natural images.

## Columns you will see everywhere

- `detect` / `reject_rate` — fraction of repeats rejecting at α = 0.05. For
  `control-disjoint` and for `fraction = 0` this is the **false-positive rate**, not power, and
  it must sit near 5%.
- `detect_lo` / `detect_hi` — Wilson 95% interval. Normal-approximation intervals are useless
  here because the interesting cells sit at 0% and 100%, where they have zero width.
- `KS` — the Kolmogorov–Smirnov statistic, the largest gap between the two ECDFs. `KS_sd` is
  the spread across repeats, not a standard error.
- `KL` — in nats, binned, with Krichevsky–Trofimov smoothing on the **counts**. Comparable
  across rows here; **not** comparable to any KL printed before August 2026 (FINDINGS.md §3.1).
- `n` — images per side. Both sides are always size-matched.

## `score_cache/`

Cached per-image scores, `<dataset>__<scorer>__<im_size>.npz`. **Gitignored** — derived data,
rebuild with `python -m experiments.score_cache`. See `experiments/README.md` for why it exists
and what the row order guarantees.

## `prior/`

The first pass (August 2026): `density_grid.csv`, `resolution_sweep*.csv`, `power_curve.csv`,
`calibration.csv`, and their logs. Superseded by `exp1`–`exp6`, kept because FINDINGS.md §2
quotes them and because `calibration.csv` is still the only dedicated 500-draw calibration run.

`prior/power_curve.csv` in particular should not be quoted: 10 repeats per cell, one scorer.
Use `exp1_power_curve.csv`.
