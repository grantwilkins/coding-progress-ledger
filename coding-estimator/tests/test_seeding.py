"""Two pipeline invocations with the same seed must yield identical predictions."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from coding_estimator.io import set_global_seed


def _run(seed: int) -> np.ndarray:
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(40, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    model = LogisticRegression(random_state=seed, solver="liblinear")
    model.fit(X, y)
    X_test = rng.normal(size=(10, 4))
    return model.predict_proba(X_test)[:, 1]


def test_same_seed_same_predictions() -> None:
    a = _run(42)
    b = _run(42)
    np.testing.assert_array_equal(a, b)


def test_different_seed_different_predictions() -> None:
    a = _run(0)
    b = _run(1)
    assert not np.array_equal(a, b)


def test_set_global_seed_seeds_legacy_numpy_and_random() -> None:
    """set_global_seed itself must drive reproducibility for code paths
    that consult numpy's legacy global RNG or the `random` module."""
    import random as py_random

    set_global_seed(123)
    a_np = np.random.rand(5)
    a_py = [py_random.random() for _ in range(5)]
    set_global_seed(123)
    b_np = np.random.rand(5)
    b_py = [py_random.random() for _ in range(5)]
    np.testing.assert_array_equal(a_np, b_np)
    assert a_py == b_py
