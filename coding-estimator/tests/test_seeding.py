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
