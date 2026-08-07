# Results

One CSV per experiment, written incrementally by the scripts in `experiments/`. Each `.log` is
the console output of the run that produced the CSV next to it.

**These files are the record.** Before this existed, results lived only in notebook output cells,
so re-running a notebook destroyed the evidence for the committed figures.

| file | produced by | one row per |
|---|---|---|
| `density_grid.csv` | `run_density_grid.py` | (comparison, scorer) — the main table |
| `resolution_sweep.csv` | `run_resolution_sweep.py` | (comparison, im_size) for `entropy-2-mse` |
| `resolution_sweep_jpeg.csv` | `run_resolution_sweep.py --scorer jpeg_bytes` | as above, for the JPEG baseline |
| `power_curve.csv` | `run_power_curve.py` | (comparison, n, repeat) — detection rate vs sample size |

## Columns

Shared: `comparison`, `scorer`, `expected` (accept/reject), `n_target`, `n_test`, `seed`, `secs`.

Statistics, all computed on the two 1-D score samples:

| column | meaning |
|---|---|
| `KL`, `JS`, `wasserstein` | effect sizes. `KL`/`JS` are binned (`num_bins`, `alpha`); Wasserstein is not. |
| `KS`, `CVM`, `AD` | two-sample test statistics |
| `KS_p`, `CVM_p`, `AD_p` | their analytic p-values — **no binning, no smoothing constant** |
| `KL_p`, `JS_p`, `wasserstein_p` | permutation p-values (present when `permutations > 0`) |

`AD_p` is floored at 0.001 and capped at 0.25 by scipy — do not read it as a precise value at
the edges. Permutation p-values are floored at `1/(permutations+1)`.

Reading a row: `KS_p < 0.05` means the two datasets are distinguishable at that sample size.
The `control-disjoint` row is the false-positive check and must **accept**.

## Reproducing

```bash
conda activate h_data
python -m experiments.run_density_grid --n 4000 --permutations 1000
python -m experiments.run_resolution_sweep --n 4000
python -m experiments.run_power_curve --repeats 10
```

Add `--quick` to `run_density_grid` for a fast smoke test (n=300, no BRISQUE). BRISQUE is the
slow scorer — roughly 4 minutes per comparison at n=4000, everything else is seconds.

Interpretation and caveats are in [`../FINDINGS.md`](../FINDINGS.md) §2.
