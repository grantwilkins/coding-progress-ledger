"""Closed-form materialization costs for vagrant.

Four formulas (see docs and TASKS.md § Workstream C):

    kv_transfer_s    = 8 * T * kv_bytes_per_token / link_bps
    context_replay_s = T / dst_prefill_tok_s
    text_transfer_s  = 8 * text_bytes / link_bps
    artifact_copy_s  = 8 * artifact_bytes / link_bps

T cancels in the kv-vs-replay comparison; the crossover is in bandwidth:

    B* = 8 * kv_bytes_per_token * dst_prefill_tok_s

The MVP allowed-modes per state layer:

    prompt_context (tokens > 0)  -> {kv_transfer, context_replay}
    workspace      (bytes > 0)   -> {artifact_copy}

`warm_reuse` is intentionally NOT a formula here — it's a placement-state-aware
concept handled in policy code (skip the cost call when the state is already
materialized at dst by a prior placement decision).
"""
from __future__ import annotations

from .events import (
    MATERIALIZATION_MODES as _ALL_MODES,
)
from .manifest import StateObject
from .profiles import LinkProfile, ProfileBundle, SiteProfile

KV_TRANSFER = "kv_transfer"
CONTEXT_REPLAY = "context_replay"
TEXT_TRANSFER = "text_transfer"
ARTIFACT_COPY = "artifact_copy"

MVP_MODES = (KV_TRANSFER, CONTEXT_REPLAY, TEXT_TRANSFER, ARTIFACT_COPY)
assert set(MVP_MODES).issubset(set(_ALL_MODES)), "MVP_MODES must be a subset of events.MATERIALIZATION_MODES"


def kv_transfer_cost(tokens: int, kv_bytes_per_token: int, link_bps: float) -> float:
    if link_bps <= 0:
        raise ValueError("link_bps must be positive")
    return 8.0 * tokens * kv_bytes_per_token / link_bps


def context_replay_cost(tokens: int, dst_prefill_tok_s: float) -> float:
    if dst_prefill_tok_s <= 0:
        raise ValueError("dst_prefill_tok_s must be positive")
    return tokens / dst_prefill_tok_s


def text_transfer_cost(text_bytes: int, link_bps: float) -> float:
    if link_bps <= 0:
        raise ValueError("link_bps must be positive")
    return 8.0 * text_bytes / link_bps


def artifact_copy_cost(artifact_bytes: int, link_bps: float) -> float:
    if link_bps <= 0:
        raise ValueError("link_bps must be positive")
    return 8.0 * artifact_bytes / link_bps


def bandwidth_crossover_bps(kv_bytes_per_token: int, dst_prefill_tok_s: float) -> float:
    """Return the link bandwidth at which kv_transfer_s == context_replay_s.

    Below this bps, context_replay is cheaper. Above this bps, kv_transfer is cheaper.
    """
    return 8.0 * kv_bytes_per_token * dst_prefill_tok_s


def allowed_modes_for_state(state: StateObject) -> tuple[str, ...]:
    """Mode allowability is a function of state layer in the MVP."""
    if state.layer in ("prompt_context", "model_execution"):
        return (KV_TRANSFER, CONTEXT_REPLAY)
    if state.layer == "workspace":
        return (ARTIFACT_COPY,)
    if state.layer == "memory":
        return (TEXT_TRANSFER,)
    return ()


def materialize_cost(
    state: StateObject,
    mode: str,
    src_site: str,
    dst_site: str,
    bundle: ProfileBundle,
) -> float:
    """Cost in seconds to materialize `state` at `dst_site` given source `src_site`.

    Same-site (src == dst): no transfer; only `CONTEXT_REPLAY` is feasible for
    `prompt_context` (you still pay local prefill). For `workspace` and `memory`
    layers, same-site is treated as a local read with zero cost.

    Different sites: each mode dispatches to its closed-form formula below.
    """
    dst = bundle.site(dst_site)

    if src_site == dst_site:
        if mode == CONTEXT_REPLAY:
            return context_replay_cost(state.tokens, dst.prefill_tok_s)
        if mode in (TEXT_TRANSFER, ARTIFACT_COPY) and state.layer in ("workspace", "memory"):
            return 0.0
        raise ValueError(f"mode {mode!r} requires src != dst for state layer {state.layer!r}")

    link = bundle.link(src_site, dst_site)
    if mode == KV_TRANSFER:
        return kv_transfer_cost(state.tokens, bundle.model.kv_bytes_per_token, link.effective_bps)
    if mode == CONTEXT_REPLAY:
        return context_replay_cost(state.tokens, dst.prefill_tok_s)
    if mode == TEXT_TRANSFER:
        if state.bytes is None:
            raise ValueError(f"text_transfer requires state.bytes; state {state.state_id!r} has none")
        return text_transfer_cost(state.bytes, link.effective_bps)
    if mode == ARTIFACT_COPY:
        if state.bytes is None:
            raise ValueError(f"artifact_copy requires state.bytes; state {state.state_id!r} has none")
        return artifact_copy_cost(state.bytes, link.effective_bps)
    raise ValueError(f"unknown materialization mode: {mode!r}")


def choose_min_cost_mode(
    state: StateObject,
    src_site: str,
    dst_site: str,
    bundle: ProfileBundle,
    allowed_modes: tuple[str, ...] | None = None,
) -> tuple[str, float]:
    """Return (mode, cost_s) for the cheapest feasible mode at (src, dst).

    `allowed_modes=None` defaults to `allowed_modes_for_state(state)`.
    Tie-break: the first mode in `candidates` (insertion order of the tuple)
    wins on equal cost, so callers control the tie via tuple ordering.
    Hard-fails if no feasible mode is found.
    """
    candidates = allowed_modes if allowed_modes is not None else allowed_modes_for_state(state)
    if not candidates:
        raise ValueError(f"no allowed modes for state layer {state.layer!r}")
    best: tuple[str, float] | None = None
    for mode in candidates:
        try:
            cost = materialize_cost(state, mode, src_site, dst_site, bundle)
        except ValueError:
            continue
        if best is None or cost < best[1]:
            best = (mode, cost)
    if best is None:
        raise ValueError(
            f"no feasible mode for state {state.state_id!r} (layer={state.layer!r}, "
            f"tokens={state.tokens}, bytes={state.bytes})"
        )
    return best
