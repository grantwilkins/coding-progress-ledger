"""
Claim:
  The workload mix is specified by *token share*, not job count. For model m
  with declared token_share s_m and clipped mean session length mu_m,
      counts_m  proportional to  s_m / mu_m,
  so the realized share of tokens per model matches s_m. model_token_shares(
  flagship_share=s) pins the flagship to s and rescales the other models
  proportionally to their base shares, keeping the vector normalized.

Plausible wrong implementations:
- counts proportional to s_m directly (drop the / mu_m): tokens then over-weight
  long-session models, so the flagship's realized token share blows past s_m.
- weight realized share by job count instead of tokens: flagship looks like a
  ~quarter of the mix, not its declared majority.
- flagship_share rescales the others uniformly (equal shares) rather than
  proportionally, breaking their relative ratios.
- flagship_share pins s but forgets to renormalize, so shares do not sum to 1.
- monotonicity knob wired to the wrong model index.
"""

from __future__ import annotations

import numpy as np

from instance import (FLAGSHIP_IDX, MODELS, build_instance, model_token_shares)


def _realized_token_share(inst) -> np.ndarray:
    tot = inst.T.sum()
    return np.array([inst.T[inst.model_idx == i].sum() / tot for i in range(len(MODELS))])


def test_realized_token_share_matches_declared():
    # If counts dropped the / mean-session-length, the flagship (longest
    # sessions) would realize far more than its 0.55 declared share.
    inst = build_instance(total_jobs=40_000, seed=3)
    np.testing.assert_allclose(_realized_token_share(inst), model_token_shares(), atol=0.03)


def test_flagship_is_token_majority_but_job_minority():
    inst = build_instance(total_jobs=40_000, seed=3)
    tok = _realized_token_share(inst)
    job = np.array([(inst.model_idx == i).mean() for i in range(len(MODELS))])
    assert tok.argmax() == FLAGSHIP_IDX            # largest token share
    assert job[FLAGSHIP_IDX] < tok[FLAGSHIP_IDX]   # but underweight in jobs (long sessions)


def test_flagship_share_pins_and_rescales_proportionally():
    base = model_token_shares()
    s = 0.7
    out = model_token_shares(flagship_share=s)
    assert np.isclose(out.sum(), 1.0)
    assert np.isclose(out[FLAGSHIP_IDX], s)
    rest = np.delete(np.arange(len(MODELS)), FLAGSHIP_IDX)
    # proportional rescale preserves the ratios among the non-flagship models
    # (a uniform rescale would equalize them); check the full vector matches.
    expected = base[rest] * (1.0 - s) / base[rest].sum()
    np.testing.assert_allclose(out[rest], expected)


def test_realized_flagship_share_monotone_in_knob():
    realized = []
    for s in (0.3, 0.5, 0.7, 0.9):
        inst = build_instance(total_jobs=30_000, seed=5, flagship_share=s)
        realized.append(_realized_token_share(inst)[FLAGSHIP_IDX])
    assert all(a < b for a, b in zip(realized, realized[1:]))
    np.testing.assert_allclose(realized, [0.3, 0.5, 0.7, 0.9], atol=0.03)
