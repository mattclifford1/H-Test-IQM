# Experiments

Scripts that produce the CSVs in [`../results/`](../results/README.md). Run them as modules from
the repo root so the shared imports resolve:

```bash
conda activate h_data
python -m experiments.run_density_grid --n 4000 --permutations 1000
```

| script | question it answers |
|---|---|
| `run_density_grid.py` | Does the method detect each kind of shift, and **does the perceptual autoencoder beat trivial baselines?** {6 comparisons} × {5 scorers}. |
| `run_resolution_sweep.py` | How much of the result is a **resolution artifact** rather than a distribution difference? Same comparisons across `im_size` 32/64/128/256. |
| `run_power_curve.py` | **How many images do you need** to detect a shift of a given size? Detection rate vs n. |
| `run_calibration.py` | Is the test **calibrated** — is the false-positive rate actually α? The one thing that must be right before any p-value means anything. |

`run_density_grid.py` owns the shared definitions: `COMPARISONS` (the six dataset pairs) and
`proportion_for` / `pool_size` (the size-matching helpers). The other scripts import from it, so
adding a comparison in one place adds it everywhere.

## Conventions these scripts enforce

- **Both sides are always size-matched.** Binned divergences are biased upward when the two
  samples differ in size, which is what made the original one-class result look twice as large as
  it is (`FINDINGS.md` §3.5).
- **The control uses disjoint partitions** (`partition_target='a'`, `partition_test='b'`), not
  two different seeds. Different seeds only re-shuffle and give chance-level overlap; `'a'`/`'b'`
  at the same seed share no images at all.
- **Results are written after every row**, so a crash or a kernel restart costs one row.
- **Errors are captured into an `error` column** rather than aborting the grid.

## Adding a comparison

Append to `COMPARISONS` in `run_density_grid.py`:

```python
('cifar-vs-mnist',
 dict(dataset='CIFAR_10', labels='all'),
 dict(dataset='MNIST',    labels='all'),
 'reject'),
```

The dataset must be registered in all three dicts in `h_test_IQM/datasets/__init__.py`, and
`pool_size` must be able to count it — that is automatic for anything with a `meta_data.csv`.

## Adding a scorer

Register it in `h_test_IQM/scorers/__init__.py:SCORERS` and add its name to `SCORERS` in
`run_density_grid.py`. Anything callable as `(im_size, device) -> model` works; numpy-based
per-image scorers should subclass `base_scorer_numpy` and be wrapped in `init_numpy_scorer`.
