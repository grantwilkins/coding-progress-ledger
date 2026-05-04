# Reproducibility

This project guarantees byte-stable artifacts and seeded model training. The
contract:

## Seeds
- `coding_estimator.io.set_global_seed(seed)` seeds `random`, `numpy`, and
  `PYTHONHASHSEED`. Every script that builds a dataset or trains a model
  must call it before any RNG use.
- All `sklearn` estimators must be constructed with `random_state=seed`.
- Any library that consults its own global RNG must be re-seeded explicitly;
  do not assume our `set_global_seed` reaches it.

## Stable writers
All artifacts go through `coding_estimator.io`:
- `write_parquet`: zstd, no statistics, dictionary encoding, columns sorted
  alphabetically, rows sorted by `sort_by` (default: every column).
- `write_csv`: utf-8, LF endings, no index, columns sorted, rows sorted.
- `write_json`: `sort_keys=True`, `indent=2`, trailing newline.

## What not to rely on
- Dict iteration order for anything user-facing (sort first).
- Set ordering anywhere (sort before serializing).
- Floating point equality across NumPy/BLAS versions; assertions on
  predictions should use `np.testing.assert_allclose` with an explicit tol.
- Pandas' default index in artifacts: always pass `index=False`.
