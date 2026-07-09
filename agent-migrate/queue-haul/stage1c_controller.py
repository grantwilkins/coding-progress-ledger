from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import http.client
import json
import math
import random
import signal
import subprocess
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

import stage1b_drain_sink as b
from dispatch import Event, solve
from impact import Movement, compute
from instance import JobPopulation
from node_knee import evaluate_node_expected_w, solve_live_greedy, solve_power_function_lp, solve_random_jobs
from power import BETA_BYTES_PER_TOK, ETA_BYTES_PER_TOK, PoolPower, rho_replay

SCHEMA = "queue-haul-stage1c-v1"
LIVE_SCHEMA = "queue-haul-stage1c-live-v1"
MANIFEST_SCHEMA = "queue-haul-stage1c-session-manifest-v1"
PROFILE_SCHEMA = "queue-haul-stage1c-live-profile-v1"
LIVE_ARTIFACTS = (
    "gpu_power.csv", "events.jsonl", "power_summary.csv", "power_trace.png",
    "source_power.png", "sink_power.png", "delay_summary.csv", "delay_summary.png",
    "ell_power5s.csv", "ell_power5s.png", "request_counts.csv", "proxy_audit.csv",
)
WORDS_PER_TOKEN = 0.75
LIVE_A100_P_IDLE_W = 67.12041959182154
LIVE_A100_P_BUSY_W = 390.0
LIVE_A100_LOG_SHAPE = 8.55
LIVE_A100_C_PREFILL_J_PER_TOK = 0.028197803044670608
LIVE_A100_C_DECODE_J_PER_TOK = 0.25745287802176875


def default_fixture() -> dict:
    return {
        "schema": "queue-haul-stage1c-fixture-v1",
        "deadline_s": 60.0,
        "target_w": "all",
        "constants": {
            "eta_bytes_per_tok": 4096.0,
            "lambda_src_bytes_per_s": 125_000_000.0,
            "mu_bytes_per_s": 1_000_000_000.0,
        },
        "sessions": [
            {"id": "r0", "T": 1024, "ell_pre": 0.12, "ell_dec": 0.02, "c_replay_s": 2.0, "c_transfer_s": 12.0, "words": 768},
            {"id": "k0", "T": 4096, "ell_pre": 0.13, "ell_dec": 0.02, "c_replay_s": 12.0, "c_transfer_s": 4.0, "words": 1024},
        ],
    }


def load_fixture(path: Path | None) -> dict:
    return json.loads(path.read_text()) if path else default_fixture()


def active_sessions(fixture: dict) -> list[dict]:
    sessions = fixture["sessions"]
    if not sessions:
        raise ValueError("fixture has no sessions")
    for s in sessions:
        if s.get("state", "active") != "active":
            raise ValueError("stage1c fixture sessions must all be active")
    return sessions


def build_population(fixture: dict) -> JobPopulation:
    sessions = active_sessions(fixture)
    n = len(sessions)
    T = np.array([s["T"] for s in sessions], dtype=float)
    ell_pre = np.array([s["ell_pre"] for s in sessions], dtype=float)
    ell_dec = np.array([s["ell_dec"] for s in sessions], dtype=float)
    eta = float(fixture["constants"].get("eta_bytes_per_tok", 4096.0))
    classes = np.array([s.get("session_class", "agentic_tool_loop") for s in sessions], dtype=object)
    return JobPopulation(
        np.where(classes == "agentic_tool_loop", "agentic", "chat"),
        classes,
        np.array(["active"] * n, dtype=object),
        classes == "reasoning_chat",
        T,
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.ones(n, dtype=bool),
        ell_pre,
        ell_dec,
        eta * T,
        "bf16",
        0.35,
        np.array([s.get("source_node", 0) for s in sessions], dtype=int),
    )


def build_plan(fixture: dict):
    sessions = active_sessions(fixture)
    pool = PoolPower(mean_context_tokens=float(np.mean([s["T"] for s in sessions])))
    const = fixture["constants"]
    move = Movement(
        lambda_src=float(const.get("lambda_src_bytes_per_s", 125_000_000.0)),
        mu_in=float(const.get("mu_bytes_per_s", 1_000_000_000.0)),
        dest_prefill_util=0.0,
        dest_ingest_util=0.0,
    )
    pop = build_population(fixture)
    base = compute(pop, pool, move)
    T = pop.T.astype(float)
    eta = float(const.get("eta_bytes_per_tok", 4096.0))
    imp = replace(
        base,
        b_replay=BETA_BYTES_PER_TOK * T,
        b_transfer=eta * T,
        c_replay=np.array([s.get("c_replay_s", base.c_replay[i]) for i, s in enumerate(sessions)], dtype=float),
        c_transfer=np.array([s.get("c_transfer_s", base.c_transfer[i]) for i, s in enumerate(sessions)], dtype=float),
    )
    event = Event(D=float(fixture["deadline_s"]), dest_nodes=1, spare_frac=1.0, tau_src=0.0, tau_pre=0.0, tau_in=0.0)
    target = fixture.get("target_w", "all")
    s_star = float(np.sum(imp.dp_certified)) if target == "all" else float(target)
    plan = solve(pop, pool, imp, s_star, event, move, integer=True, mode="sf")
    if not plan.feasible or plan.shortfall != 0:
        raise RuntimeError(f"solver failed target: feasible={plan.feasible} shortfall={plan.shortfall}")
    if not np.all(np.isclose(plan.y, np.round(plan.y))):
        raise RuntimeError("non-integer plan returned for integer solve")
    return pop, pool, imp, event, move, plan, s_star


def planned_sessions(fixture: dict) -> list[dict]:
    pop, _pool, imp, _event, _move, plan, s_star = build_plan(fixture)
    sessions = active_sessions(fixture)
    rows = []
    for i, s in enumerate(sessions):
        if plan.y[i] <= 0.5:
            continue
        action = "R" if plan.y_R[i] > plan.y_S[i] else "S"
        planned = float(imp.c_replay[i] if action == "R" else imp.c_transfer[i])
        density = float(imp.dp_certified[i] / planned)
        rows.append({
            "id": s["id"],
            "action": action,
            "fixture_index": i,
            "T": float(pop.T[i]),
            "planned_finish_s": planned,
            "dp_certified_w": float(imp.dp_certified[i]),
            "density": density,
            "words": int(s.get("words", min(max(pop.T[i], 256), 2048))),
        })
    rows.sort(key=lambda r: (-r["density"], r["fixture_index"]))
    actions = {r["action"] for r in rows}
    if actions != {"R", "S"}:
        raise RuntimeError(f"stage1c proof fixture must produce replay and KV actions, got {sorted(actions)}")
    for rank, row in enumerate(rows):
        row["dispatch_rank"] = rank
    return rows


def plan_summary(fixture: dict) -> dict:
    _pop, _pool, imp, event, move, plan, s_star = build_plan(fixture)
    return {
        "schema": "queue-haul-stage1c-plan-v1",
        "deadline_s": event.D,
        "target_w": s_star,
        "movement": {"lambda_src_bytes_per_s": move.lambda_src, "mu_bytes_per_s": move.mu_in},
        "solver": {"method": plan.method, "feasible": plan.feasible, "shortfall_w": plan.shortfall, "shed_guaranteed_w": plan.shed_guaranteed},
        "sessions": planned_sessions(fixture),
        "costs": {"c_replay": imp.c_replay.tolist(), "c_transfer": imp.c_transfer.tolist()},
    }


def run_selected_session(cfg: b.Config, run_root: Path, row: dict, t0: float, prewarmed: dict[str, dict] | None = None) -> dict:
    proxy_log = run_root / "proxy_bytes.csv"
    source_log = run_root / "source.log"
    sink_log = run_root / "sink.log"
    prompt = b.prompt_text(f"stage1c-{row['id']}", row["words"])
    before = b.proxy_counts(proxy_log)
    retrieved0 = b.count_needle(sink_log, "Retrieved")
    source = (prewarmed or {}).get(row["id"])
    sink = b.post_chat(cfg, cfg.api_proxy_port, prompt, 4)
    b.check_chat(sink, f"sink dispatch {row['id']}")
    time.sleep(2)
    delta = b.count_delta(before, b.proxy_counts(proxy_log))
    if row["action"] == "S":
        if b.count_needle(sink_log, "Retrieved") <= retrieved0:
            raise RuntimeError(f"{row['id']} did not retrieve KV on sink")
        if delta.get("kv/target_to_client", 0) <= 0:
            raise RuntimeError(f"{row['id']} had no KV bytes")
        if source and source["content"].strip() != sink["content"].strip():
            raise RuntimeError(f"{row['id']} source/sink deterministic outputs differ")
    else:
        if delta.get("api/client_to_target", 0) <= 0:
            raise RuntimeError(f"{row['id']} had no replay API bytes")
        if delta.get("kv/target_to_client", 0) > 2_000_000:
            raise RuntimeError(f"{row['id']} replay pulled too many KV bytes")
    return {
        **row,
        "endpoint": f"http://{cfg.host}:{cfg.api_proxy_port}/v1/chat/completions",
        "actual_start_s": sink["start_ts"] - t0,
        "actual_end_s": sink["end_ts"] - t0,
        "http_status": sink["status"],
        "prompt_sha256": sink["prompt_sha256"],
        "content": sink["content"],
        "proxy_delta": delta,
        "deadline_met": sink["end_ts"] - t0 <= row.get("deadline_s", float("inf")),
    }


def check_manifest(manifest: dict) -> None:
    if manifest.get("schema") != SCHEMA:
        raise ValueError("bad schema")
    solver = manifest["solver"]
    if not solver["feasible"] or solver["shortfall_w"] != 0:
        raise ValueError("solver did not meet target")
    sessions = manifest["sessions"]
    if {s["action"] for s in sessions} != {"R", "S"}:
        raise ValueError("proof must include replay and KV actions")
    ranks = [s["dispatch_rank"] for s in sessions]
    if ranks != list(range(len(ranks))):
        raise ValueError("dispatch ranks are not contiguous")
    starts = [s.get("actual_start_s", i) for i, s in enumerate(sessions)]
    ends = [s.get("actual_end_s", starts[i]) for i, s in enumerate(sessions)]
    if any(starts[i] < ends[i - 1] for i in range(1, len(sessions))):
        raise ValueError("dispatch execution is not serial by rank")
    for s in sessions:
        if s["http_status"] != 200 or not s["deadline_met"]:
            raise ValueError(f"session failed: {s['id']}")
        delta = s["proxy_delta"]
        if s["action"] == "S" and delta.get("kv/target_to_client", 0) <= 0:
            raise ValueError(f"KV action lacks KV bytes: {s['id']}")
        if s["action"] == "R" and delta.get("api/client_to_target", 0) <= 0:
            raise ValueError(f"replay action lacks API bytes: {s['id']}")
    if not manifest["smoke2"]["acceptance"]["ok"]:
        raise ValueError("smoke2 gate failed")


def proof(cfg: b.Config, run_root: Path, fixture: dict, mbps: float, extra: list[str]) -> Path:
    stack = b.start_stack(cfg, run_root, mbps, extra)
    try:
        summary = plan_summary(fixture)
        b.start_sink(stack, cfg, extra)
        prewarmed = {}
        for row in summary["sessions"]:
            if row["action"] == "S":
                prompt = b.prompt_text(f"stage1c-{row['id']}", row["words"])
                prewarmed[row["id"]] = b.warm_source(cfg, run_root, prompt, f"source warm {row['id']}")[0]
        deadline = float(fixture["deadline_s"])
        t0 = time.time()
        rows = []
        for row in summary["sessions"]:
            rows.append(run_selected_session(cfg, run_root, {**row, "deadline_s": deadline}, t0, prewarmed))
        manifest = {**summary, "schema": SCHEMA, "smoke2": {"acceptance": {"ok": True, "covered_by_controller_sessions": True, "live_source_sink": True}}, "sessions": rows}
        manifest["acceptance"] = {
            "all_completed_by_deadline": all(r["deadline_met"] for r in rows),
            "actions": sorted({r["action"] for r in rows}),
            "ok": True,
        }
        check_manifest(manifest)
        (run_root / "controller_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    finally:
        b.stop_stack(stack)
    return run_root


def _jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                row["_line"] = n
                yield row


def _field(row: dict, *names: str, default=None):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    if default is not None:
        return default
    raise ValueError(f"missing field; tried {names}; line={row.get('_line')}")


def _num(row: dict, *names: str, default=None) -> float:
    return float(_field(row, *names, default=default))


def _parse_ts(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    for fmt in (None, "%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            if fmt is None:
                return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            pass
    return float(text)


def _row_time(row: dict) -> float:
    for key in ("timestamp", "ts", "created_at", "started_at", "start_time"):
        if key in row and row[key] is not None:
            return _parse_ts(row[key])
    for ev in row.get("timing_events", []):
        for key in ("timestamp", "ts", "time", "at", "emitted_at"):
            if key in ev and ev[key] is not None:
                return _parse_ts(ev[key])
    raise ValueError(f"TraceLab row has no timestamp/timing_events time: line={row.get('_line')}")


def _trace_tokens(row: dict) -> dict:
    total = int(_num(row, "input_tokens_total", "input_tokens", "prompt_tokens", default=0))
    prefix = int(_num(row, "prefix_tokens", "cached_input_tokens", "cache_read_input_tokens", default=0))
    append = int(_num(row, "newly_append_tokens", "append_tokens", default=max(1, total - prefix)))
    output = int(_num(row, "output_tokens", "completion_tokens", "generated_tokens", default=32))
    if total <= 0:
        raise ValueError(f"TraceLab row has no positive input token count: line={row.get('_line')}")
    return {"total": total, "prefix": max(0, prefix), "append": max(1, append), "output": max(1, output)}


def _words(tag: str, tokens: int) -> str:
    n = max(1, int(tokens * WORDS_PER_TOKEN))
    return f"{tag} " + " ".join("x" for _ in range(n))


def session_prompt(session: dict, turn_index: int, replay_nonce: str | None = None) -> str:
    turns = session.get("turns", [])
    upto = turns[: max(1, min(turn_index + 1, len(turns)))]
    cap = int(session["served_T"])
    budget = max(256, cap - 1024)
    selected = []
    for i in range(len(upto) - 1, -1, -1):
        take = min(int(upto[i]["append_tokens"]), budget)
        if take <= 0:
            break
        selected.append((i, take))
        budget -= take
    selected.reverse()
    turn_tokens = sum(t for _, t in selected)
    profile_tokens = max(128, min(cap - turn_tokens - 512, cap // 2))
    parts = []
    if replay_nonce:
        parts.append(f"Replay nonce {replay_nonce}.")
    parts.append(f"Synthetic TraceLab-sized coding-agent session {session['id']}.")
    parts.append("Stable task state " + _words(f"state_{session['id']}", profile_tokens))
    for i, tokens in selected:
        parts.append(f"User follow-up {i}: " + _words(f"turn_{session['id']}_{i}", tokens))
    parts.append("Reply with one concise progress sentence.")
    return "\n".join(parts)


def tracelab_manifest(path: Path, n_sessions: int, seed: int, max_model_len: int = 32768,
                      decode_margin: int = 512, min_turns: int = 3, min_context_tokens: int = 2048,
                      baseline_s: float = 120.0, settle_s: float = 120.0) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in _jsonl(path):
        sid = str(_field(row, "session_id", "session", "conversation_id", "trace_key"))
        groups.setdefault(sid, []).append(row)
    sessions = []
    served_cap = max_model_len - decode_margin
    for sid, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: (_row_time(r), r.get("_line", 0)))
        if len(rows) < min_turns:
            continue
        good = []
        for r in rows:
            try:
                good.append((r, _row_time(r), _trace_tokens(r)))
            except ValueError:
                pass
        if len(good) < min_turns:
            continue
        rows, times, toks = zip(*good)
        span = times[-1] - times[0]
        if span <= 0:
            continue
        original_T = max(t["total"] for t in toks)
        if original_T < min_context_tokens:
            continue
        served_T = min(original_T, served_cap)
        if served_T < min_context_tokens:
            continue
        output = int(np.clip(np.median([t["output"] for t in toks]), 8, decode_margin))
        rate = max(1e-4, (len(rows) - 1) / span)
        service_s = served_T / float(rho_replay(served_T)) + output / 80.0
        ell = float(np.clip(rate * service_s, 0.005, 0.50))
        turns = []
        for i, (row, tok) in enumerate(zip(rows, toks)):
            turns.append({
                "round": i,
                "gap_s": 0.0 if i == 0 else max(0.0, times[i] - times[i - 1]),
                "append_tokens": tok["append"],
                "prefix_tokens": tok["prefix"],
                "input_tokens_total": tok["total"],
                "decode_tokens": tok["output"],
            })
        sessions.append({
            "id": sid,
            "session_class": "agentic_tool_loop",
            "state": "active",
            "T": served_T,
            "served_T": served_T,
            "original_T": original_T,
            "turn_rate_hz": rate,
            "decode_tokens": output,
            "ell_pre": ell,
            "ell_dec": min(0.50, ell / 2),
            "source_node": 0,
            "words": int(served_T * WORDS_PER_TOKEN),
            "turns": turns,
        })
    if len(sessions) < n_sessions:
        raise ValueError(f"need {n_sessions} TraceLab sessions after filtering, got {len(sessions)}")
    rng = random.Random(seed)
    picked = rng.sample(sessions, n_sessions) if len(sessions) > n_sessions else sessions
    for i, s in enumerate(picked):
        s["source_node"] = 0
        s["manifest_rank"] = i
    return {
        "schema": MANIFEST_SCHEMA,
        "source": {"type": "tracelab", "path": str(path), "seed": seed},
        "deadline_s": 300.0,
        "target_w": "all",
        "baseline_s": baseline_s,
        "settle_s": settle_s,
        "max_model_len": max_model_len,
        "decode_margin": decode_margin,
        "power_curve": "log",
        "constants": {
            "eta_bytes_per_tok": ETA_BYTES_PER_TOK,
            "lambda_src_bytes_per_s": 125_000_000.0,
            "mu_bytes_per_s": 1_000_000_000.0,
        },
        "sessions": picked,
    }


def live_sessions(manifest: dict) -> list[dict]:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("bad live session manifest schema")
    sessions = manifest.get("sessions", [])
    if not sessions:
        raise ValueError("manifest has no sessions")
    for s in sessions:
        for key in ("id", "served_T", "ell_pre", "ell_dec", "turn_rate_hz", "turns"):
            if key not in s:
                raise ValueError(f"session {s.get('id')} missing {key}")
        if s.get("state", "active") != "active":
            raise ValueError(f"session {s['id']} is not active")
        if len(s["turns"]) < 1:
            raise ValueError(f"session {s['id']} has no turns")
    return sessions




def profile_token_points(sessions: list[dict], max_points: int = 3) -> list[int]:
    vals = sorted({int(s["served_T"]) for s in sessions})
    if len(vals) <= max_points:
        return vals
    idxs = np.linspace(0, len(vals) - 1, max_points).round().astype(int)
    return [vals[int(i)] for i in dict.fromkeys(idxs)]


def _profile_session(tokens: int) -> dict:
    return {
        "id": f"profile_{tokens}",
        "served_T": int(tokens),
        "decode_tokens": 1,
        "turns": [{"append_tokens": max(1, int(tokens) - 1024)}],
    }


def _profile_row(action: str, tokens: int, result: dict, delta: dict) -> dict:
    return {
        "action": action,
        "tokens": int(tokens),
        "first_token_s": float(result["first_token_ts"] - result["start_ts"]),
        "completion_s": float(result["end_ts"] - result["start_ts"]),
        "proxy_delta": delta,
    }


def calibrate_live_profile(cfg: b.Config, run_root: Path, sessions: list[dict], mbps: float, max_points: int = 3) -> dict:
    rows, proxy_log = [], run_root / "proxy_bytes.csv"
    for tokens in profile_token_points(sessions, max_points):
        prompt = session_prompt(_profile_session(tokens), 0)
        before = b.proxy_counts(proxy_log)
        replay = stream_chat(cfg, cfg.api_proxy_port, f"Replay profile {tokens} {time.time()}.\n{prompt}", 1)
        if replay["status"] != 200:
            raise RuntimeError(f"profile replay failed for {tokens}: {replay['response_text']}")
        delta = b.count_delta(before, b.proxy_counts(proxy_log))
        if delta.get("api/client_to_target", 0) <= 0:
            raise RuntimeError(f"profile replay had no API bytes for {tokens}")
        rows.append(_profile_row("R", tokens, replay, delta))

        b.warm_source(cfg, run_root, prompt, f"profile source warm {tokens}")
        before = b.proxy_counts(proxy_log)
        kv = stream_chat(cfg, cfg.api_proxy_port, prompt, 1)
        if kv["status"] != 200:
            raise RuntimeError(f"profile KV failed for {tokens}: {kv['response_text']}")
        time.sleep(2)
        delta = b.count_delta(before, b.proxy_counts(proxy_log))
        if delta.get("kv/target_to_client", 0) <= 0:
            raise RuntimeError(f"profile KV had no KV bytes for {tokens}")
        rows.append(_profile_row("S", tokens, kv, delta))
    return {"schema": PROFILE_SCHEMA, "created_ts": time.time(), "mbps": mbps, "points": rows}


def ensure_live_profile(cfg: b.Config, run_root: Path, profile_path: Path | None, manifest: dict, mbps: float) -> tuple[dict, Path]:
    path = profile_path or run_root / "live_profile.json"
    if path.exists():
        profile = json.loads(path.read_text())
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        profile = calibrate_live_profile(cfg, run_root, live_sessions(manifest), mbps)
        path.write_text(json.dumps(profile, indent=2, sort_keys=True))
    if profile.get("schema") != PROFILE_SCHEMA:
        raise ValueError("bad live profile schema")
    return profile, path


def _profile_cost(profile: dict, action: str, tokens: int) -> float:
    pts = sorted((int(r["tokens"]), float(r["completion_s"])) for r in profile["points"] if r["action"] == action)
    if not pts:
        raise ValueError(f"profile missing action {action}")
    if len(pts) == 1 or tokens <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if tokens <= x1:
            return y0 + (y1 - y0) * (tokens - x0) / max(1, x1 - x0)
    return pts[-1][1] * tokens / max(1, pts[-1][0])


def apply_live_profile(manifest: dict, profile: dict) -> dict:
    if profile.get("schema") != PROFILE_SCHEMA:
        raise ValueError("bad live profile schema")
    sessions = []
    for s in live_sessions(manifest):
        tokens = int(s["served_T"])
        sessions.append({
            **s,
            "c_replay_s": _profile_cost(profile, "R", tokens),
            "c_transfer_s": _profile_cost(profile, "S", tokens),
        })
    return {**manifest, "sessions": sessions, "profile": {"schema": profile["schema"], "mbps": profile.get("mbps"), "points": profile["points"]}}

def build_live_population(manifest: dict) -> JobPopulation:
    sessions = live_sessions(manifest)
    patched = {**manifest, "sessions": [{**s, "T": s["served_T"]} for s in sessions]}
    return build_population(patched)


def _live_model(manifest: dict, deadline_s: float | None = None, target_frac: float | None = None):
    sessions = live_sessions(manifest)
    pop = build_live_population(manifest)
    const = manifest["constants"]
    power = manifest.get("power", {})
    pool = PoolPower(
        p_idle_w=float(power.get("p_idle_w", LIVE_A100_P_IDLE_W)),
        p_busy_w=float(power.get("p_busy_w", LIVE_A100_P_BUSY_W)),
        rho_star=max(float(pop.ell.sum()), 1e-9),
        mean_context_tokens=float(np.mean(pop.T)),
        c_prefill_j_per_tok=float(power.get("c_prefill_j_per_tok", LIVE_A100_C_PREFILL_J_PER_TOK)),
        c_decode_j_per_tok=float(power.get("c_decode_j_per_tok", LIVE_A100_C_DECODE_J_PER_TOK)),
        power_curve=manifest.get("power_curve", "log"),
        log_shape=float(power.get("log_shape", LIVE_A100_LOG_SHAPE)),
    )
    move = Movement(
        lambda_src=float(const.get("lambda_src_bytes_per_s", 125_000_000.0)),
        mu_in=float(const.get("mu_bytes_per_s", 1_000_000_000.0)),
        dest_prefill_util=0.0,
        dest_ingest_util=0.0,
    )
    base = compute(pop, pool, move)
    imp = replace(
        base,
        c_replay=np.array([s.get("c_replay_s", base.c_replay[i]) for i, s in enumerate(sessions)], dtype=float),
        c_transfer=np.array([s.get("c_transfer_s", base.c_transfer[i]) for i, s in enumerate(sessions)], dtype=float),
    )
    event = Event(D=float(manifest["deadline_s"] if deadline_s is None else deadline_s), dest_nodes=1, spare_frac=1.0, tau_src=0.0, tau_pre=0.0, tau_in=0.0)
    full_w = evaluate_node_expected_w(pop, pool, np.ones(len(pop)))
    target = manifest.get("target_w", "all")
    target_w = float(target_frac) * full_w if target_frac is not None else full_w if target == "all" else float(target)
    return sessions, pop, pool, move, imp, event, full_w, target_w


def _policy_result(policy: str, pop: JobPopulation, pool: PoolPower, imp, target_w: float, event: Event, move: Movement, seed: int):
    if policy == "lp":
        return solve_power_function_lp(pop, pool, imp, target_w, event, move)
    if policy == "random":
        return solve_random_jobs(pop, pool, imp, target_w, event, move, seed=seed)
    if policy == "greedy":
        return solve_live_greedy(pop, pool, imp, target_w, event, move)
    raise ValueError(f"unknown live policy {policy!r}")


def _row_action(result, imp, i: int) -> str:
    if result.y_R[i] + result.y_S[i] <= 1e-9:
        return "R" if imp.c_replay[i] <= imp.c_transfer[i] else "S"
    return "R" if result.y_R[i] >= result.y_S[i] else "S"


def _policy_order(policy: str, result, pop: JobPopulation, pool: PoolPower, imp, seed: int) -> list[int]:
    selected = set(np.flatnonzero(result.y > 1e-9))
    if policy == "random":
        return [int(i) for i in np.random.default_rng(seed).permutation(len(pop)) if i in selected]
    if policy == "greedy":
        y, out = np.zeros(len(pop)), []
        while selected:
            base = evaluate_node_expected_w(pop, pool, y)
            scored = []
            for i in selected:
                yy = y.copy()
                yy[i] = 1.0
                action = _row_action(result, imp, int(i))
                cost = imp.c_replay[i] if action == "R" else imp.c_transfer[i]
                scored.append((-(evaluate_node_expected_w(pop, pool, yy) - base) / max(cost, 1e-12), int(i)))
            i = sorted(scored)[0][1]
            y[i] = 1.0
            out.append(i)
            selected.remove(i)
        return out
    return [int(i) for i in sorted(selected, key=lambda j: (-(result.y_R[j] + result.y_S[j]), -pop.ell[j], j))]


def live_plan_summary(manifest: dict, policy: str = "lp", seed: int = 0,
                      deadline_s: float | None = None, target_frac: float | None = None) -> dict:
    sessions, pop, pool, move, imp, event, full_w, target_w = _live_model(manifest, deadline_s, target_frac)
    result = _policy_result(policy, pop, pool, imp, target_w, event, move, seed)
    rows, cumulative = [], np.zeros(len(pop))
    for i in _policy_order(policy, result, pop, pool, imp, seed):
        action = _row_action(result, imp, i)
        cumulative[i] = 1.0
        predicted = evaluate_node_expected_w(pop, pool, cumulative)
        rows.append({
            "id": sessions[i]["id"],
            "fixture_index": i,
            "dispatch_rank": len(rows),
            "action": action,
            "y_R": float(result.y_R[i]),
            "y_S": float(result.y_S[i]),
            "planned_finish_s": float(imp.c_replay[i] if action == "R" else imp.c_transfer[i]),
            "predicted_cumulative_source_drop_w": float(predicted),
            "served_T": int(pop.T[i]),
        })
        if predicted >= target_w:
            break
    planned_w = evaluate_node_expected_w(pop, pool, cumulative)
    target_ratio = target_w / full_w if full_w > 0 else math.nan
    return {
        "schema": "queue-haul-stage1c-live-plan-v1",
        "policy": policy,
        "deadline_s": event.D,
        "target_frac": float(target_frac if target_frac is not None else target_ratio),
        "target_w": target_w,
        "full_source_drop_w": full_w,
        "planned_source_drop_w": planned_w,
        "planned_shortfall_w": max(0.0, target_w - planned_w),
        "planned_hit": planned_w >= target_w - 1e-6 * max(target_w, 1.0),
        "power_curve": {"name": pool.power_curve, "log_shape": pool.log_shape, "p_idle_w": pool.p_idle_w, "p_busy_w": pool.p_busy_w, "rho_star": pool.rho_star},
        "movement": {"lambda_src_bytes_per_s": move.lambda_src, "mu_bytes_per_s": move.mu_in},
        "profile": manifest.get("profile"),
        "solver": {"method": result.method, "feasible": result.true_expected_feasible, "shortfall_w": result.expected_shortfall_w, "cost_s": result.cost},
        "sessions": rows,
    }


def stream_chat(cfg: b.Config, port: int, prompt: str, max_tokens: int) -> dict:
    body = json.dumps({"model": cfg.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0, "stream": True})
    t0 = time.time()
    conn = http.client.HTTPConnection(cfg.host, port, timeout=900)
    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    content, first = [], None
    if resp.status != 200:
        text = resp.read().decode(errors="ignore")
        conn.close()
        return {"status": resp.status, "content": "", "start_ts": t0, "first_token_ts": None, "end_ts": time.time(), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "response_text": text[:500]}
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line or not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        chunk = json.loads(data)
        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content") or ""
        if delta and first is None:
            first = time.time()
        content.append(delta)
    conn.close()
    t1 = time.time()
    return {"status": resp.status, "content": "".join(content), "start_ts": t0, "first_token_ts": first or t1, "end_ts": t1, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "response_text": ""}


class JsonlLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", buffering=1)
        self.lock = threading.Lock()
        self.closed = False

    def write(self, kind: str, **row) -> None:
        with self.lock:
            if not self.closed:
                self.handle.write(json.dumps({"ts": time.time(), "kind": kind, **row}, sort_keys=True) + "\n")

    def close(self) -> None:
        with self.lock:
            self.closed = True
            self.handle.close()


class SessionWorker:
    def __init__(self, cfg: b.Config, session: dict, port: int, log: JsonlLog, seed: int):
        self.cfg, self.session, self.port, self.log = cfg, session, port, log
        self.rng = random.Random(seed)
        self.cond = threading.Condition()
        self.pending = 0
        self.turn = 0
        self.paused = False
        self.stopped = False
        self.in_flight = False
        self.last_prompt = session_prompt(session, 0)
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        self.threads = [threading.Thread(target=self._arrivals, daemon=True), threading.Thread(target=self._serve, daemon=True)]
        for t in self.threads:
            t.start()

    def _arrivals(self) -> None:
        rate = float(self.session["turn_rate_hz"])
        while True:
            due = time.monotonic() + self.rng.expovariate(rate)
            with self.cond:
                while not self.stopped:
                    left = due - time.monotonic()
                    if left <= 0:
                        break
                    self.cond.wait(left)
                if self.stopped:
                    return
                if self.pending:
                    self.log.write("turn_drop", session_id=self.session["id"])
                else:
                    self.pending = 1
                    self.log.write("turn_arrival", session_id=self.session["id"])
                self.cond.notify_all()

    def _serve(self) -> None:
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.stopped or (self.pending and not self.paused))
                if self.stopped:
                    return
                self.pending = 0
                idx = self.turn
                self.turn += 1
                self.in_flight = True
            prompt = session_prompt(self.session, idx)
            self.log.write("request_start", session_id=self.session["id"], turn=idx, port=self.port)
            result = stream_chat(self.cfg, self.port, prompt, int(self.session["decode_tokens"]))
            if result["status"] == 200:
                self.last_prompt = prompt
            self.log.write("request_end", session_id=self.session["id"], turn=idx, port=self.port, status=result["status"], first_token_ts=result["first_token_ts"], end_ts=result["end_ts"])
            with self.cond:
                self.in_flight = False
                self.cond.notify_all()

    def pause_boundary(self, timeout_s: float = 900.0) -> None:
        deadline = time.monotonic() + timeout_s
        with self.cond:
            self.paused = True
            self.cond.notify_all()
            while self.in_flight:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"timed out pausing {self.session['id']}")
                self.cond.wait(left)

    def switch_to(self, port: int) -> None:
        with self.cond:
            self.port = port

    def resume(self) -> None:
        with self.cond:
            self.paused = False
            self.cond.notify_all()

    def stop(self) -> None:
        with self.cond:
            self.stopped = True
            self.cond.notify_all()
        for t in self.threads:
            t.join(timeout=5)


def nvsmi_cmd(ms: int) -> list[str]:
    return ["nvidia-smi", "--query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-lms", str(ms)]


def start_nvsmi(path: Path, ms: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", buffering=1)
    handle.write("timestamp,index,power_w,util_gpu,memory_mib\n")
    return subprocess.Popen(nvsmi_cmd(ms), stdout=handle, stderr=subprocess.STDOUT, start_new_session=True), handle


def stop_nvsmi(proc: subprocess.Popen, handle) -> None:
    if proc.poll() is None:
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    handle.close()


def power_summary_rows(path: Path, windows: dict[str, tuple[float, float]]) -> list[dict]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append({"ts": _parse_ts(row["timestamp"]), "gpu": int(row["index"]), "power_w": float(row["power_w"])})
    out = []
    for phase, (lo, hi) in windows.items():
        for gpu in sorted({r["gpu"] for r in rows}):
            vals = [r["power_w"] for r in rows if r["gpu"] == gpu and lo <= r["ts"] <= hi]
            out.append({"phase": phase, "gpu": gpu, "samples": len(vals), "power_mean_w": float(np.mean(vals)) if vals else math.nan})
    return out


def write_power_summary(power_csv: Path, out_csv: Path, windows: dict[str, tuple[float, float]]) -> list[dict]:
    rows = power_summary_rows(power_csv, windows)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, ["phase", "gpu", "samples", "power_mean_w"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _power_points(power_csv: Path) -> list[tuple[float, int, float]]:
    with power_csv.open() as f:
        return [(_parse_ts(r["timestamp"]), int(r["index"]), float(r["power_w"])) for r in csv.DictReader(f)]


def write_power_plot(power_csv: Path, out_png: Path, dispatch_rows: list[dict], gpu: int | None = None) -> None:
    import matplotlib.pyplot as plt
    rows = _power_points(power_csv)
    if gpu is not None:
        rows = [r for r in rows if r[1] == gpu]
    if not rows:
        raise ValueError("empty power trace")
    t0 = min(r[0] for r in rows)
    fig, ax = plt.subplots(figsize=(8, 3))
    for g in sorted({r[1] for r in rows}):
        xs = [r[0] - t0 for r in rows if r[1] == g]
        ys = [r[2] for r in rows if r[1] == g]
        ax.plot(xs, ys, label=f"gpu{g}")
    for row in dispatch_rows:
        if "move_start_ts" in row:
            ax.axvline(row["move_start_ts"] - t0, color="k", alpha=0.25, linewidth=0.8)
    ax.set(xlabel="seconds", ylabel="power W")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _live_ell_by_gpu(manifest: dict, ts: float) -> dict[int, float]:
    per_session = {s["id"]: float(s["ell_pre"]) + float(s["ell_dec"]) for s in manifest["input_manifest"]["sessions"]}
    moved = sum(per_session[r["id"]] for r in manifest.get("sessions", []) if float(r.get("switch_end_ts", r.get("move_end_ts", r["move_start_ts"]))) <= ts)
    total = sum(per_session.values())
    return {0: max(0.0, total - moved), 1: moved}


def ell_power5s_rows(power_csv: Path, manifest: dict, bucket_s: float = 5.0) -> list[dict]:
    if bucket_s <= 0:
        raise ValueError("bucket_s must be positive")
    points = _power_points(power_csv)
    if not points:
        raise ValueError("empty power trace")
    windows = manifest.get("windows", {})
    if windows:
        lo = min(float(v[0]) for v in windows.values())
        hi = max(float(v[1]) for v in windows.values())
        points = [p for p in points if lo <= p[0] <= hi]
    else:
        lo, hi = min(p[0] for p in points), max(p[0] for p in points)
    if not points:
        raise ValueError("power trace has no samples inside live windows")
    if hi <= lo:
        hi = lo + bucket_s
    grouped: dict[tuple[int, int], list[float]] = {}
    for ts, gpu, power in points:
        rel = min(max(ts, lo), hi - 1e-9) - lo
        grouped.setdefault((int(rel // bucket_s), gpu), []).append(power)
    rows = []
    for (bucket, gpu), vals in sorted(grouped.items()):
        start = lo + bucket * bucket_s
        end = min(start + bucket_s, hi)
        ell = _live_ell_by_gpu(manifest, start + (end - start) / 2).get(gpu, 0.0)
        rows.append({
            "bucket": bucket,
            "bucket_start_s": start - lo,
            "bucket_end_s": end - lo,
            "gpu": gpu,
            "node": "source" if gpu == 0 else "sink" if gpu == 1 else f"gpu{gpu}",
            "ell": ell,
            "power_mean_w": float(np.mean(vals)),
            "samples": len(vals),
        })
    return rows


def write_ell_power5s(power_csv: Path, manifest: dict, out_csv: Path, out_png: Path, bucket_s: float = 5.0) -> list[dict]:
    import matplotlib.pyplot as plt
    rows = ell_power5s_rows(power_csv, manifest, bucket_s)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, ["bucket", "bucket_start_s", "bucket_end_s", "gpu", "node", "ell", "power_mean_w", "samples"])
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    for node in ("source", "sink"):
        part = [r for r in rows if r["node"] == node]
        if part:
            ax.scatter([r["ell"] for r in part], [r["power_mean_w"] for r in part], s=18, alpha=0.75, label=node)
    ax.set(xlabel="ell", ylabel="5s average power W")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return rows


def write_delay_summary(rows: list[dict], out_csv: Path, out_png: Path) -> list[dict]:
    import matplotlib.pyplot as plt
    delays = [{"dispatch_rank": r["dispatch_rank"], "id": r["id"], "action": r["action"], "first_token_s": r["first_token_s"], "completion_s": r["completion_s"], "downtime_s": r.get("downtime_s", r["completion_s"])} for r in rows]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, ["dispatch_rank", "id", "action", "first_token_s", "completion_s", "downtime_s"])
        writer.writeheader()
        writer.writerows(delays)
    fig, ax = plt.subplots(figsize=(8, 3))
    xs = [d["dispatch_rank"] for d in delays]
    ax.bar([x - 0.25 for x in xs], [d["first_token_s"] for d in delays], width=0.25, label="TTFT")
    ax.bar(xs, [d["completion_s"] for d in delays], width=0.25, label="completion")
    ax.bar([x + 0.25 for x in xs], [d["downtime_s"] for d in delays], width=0.25, label="downtime")
    ax.set(xlabel="dispatch rank", ylabel="seconds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return delays


def request_count_rows(events_jsonl: Path, windows: dict[str, tuple[float, float]]) -> list[dict]:
    counts: dict[tuple[str, int], int] = {}
    with events_jsonl.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("kind") != "request_start":
                continue
            phase = next((k for k, (lo, hi) in windows.items() if lo <= float(row["ts"]) <= hi), None)
            if phase is not None:
                key = (phase, int(row["port"]))
                counts[key] = counts.get(key, 0) + 1
    return [{"phase": p, "port": port, "requests": n} for (p, port), n in sorted(counts.items())]


def write_request_counts(events_jsonl: Path, out_csv: Path, windows: dict[str, tuple[float, float]]) -> list[dict]:
    rows = request_count_rows(events_jsonl, windows)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, ["phase", "port", "requests"])
        writer.writeheader()
        writer.writerows(rows)
    return rows




def proxy_audit_rows(proxy_log: Path, windows: dict[str, tuple[float, float]], mbps: float) -> list[dict]:
    target = mbps * 1_000_000 / 8
    rows = b.proxy_rows(proxy_log)
    out = []
    for phase, (lo, hi) in windows.items():
        for route, direction in sorted(b.BILLED_DIRECTIONS):
            xs = [r for r in rows if r["route"] == route and r["direction"] == direction and r.get("billed") == "1" and lo <= float(r["ts"]) <= hi]
            if xs:
                ts = [float(r["ts"]) for r in xs]
                nbytes = sum(int(r["bytes"]) for r in xs)
                window = max(ts) - min(ts)
            else:
                nbytes, window = 0, 0.0
            rate = nbytes / window if window else 0.0
            out.append({
                "phase": phase,
                "route": route,
                "direction": direction,
                "bytes": nbytes,
                "window_s": window,
                "bytes_per_s": rate,
                "target_bytes_per_s": target,
                "target_ratio": rate / target if target else math.nan,
                "ok": int(window < 0.5 or rate <= 1.25 * target),
            })
    return out


def write_proxy_audit(proxy_log: Path, out_csv: Path, windows: dict[str, tuple[float, float]], mbps: float) -> list[dict]:
    rows = proxy_audit_rows(proxy_log, windows, mbps)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, ["phase", "route", "direction", "bytes", "window_s", "bytes_per_s", "target_bytes_per_s", "target_ratio", "ok"])
        writer.writeheader()
        writer.writerows(rows)
    bad = [r for r in rows if not r["ok"]]
    if bad:
        raise RuntimeError(f"proxy exceeded configured link: {bad[:2]}")
    return rows

def check_live_manifest(manifest: dict, run_root: Path) -> None:
    if manifest.get("schema") != LIVE_SCHEMA:
        raise ValueError("bad live schema")
    for name in LIVE_ARTIFACTS:
        if not (run_root / name).exists():
            raise ValueError(f"missing {name}")
    sessions = manifest.get("sessions", [])
    if not sessions:
        raise ValueError("live manifest has no sessions")
    ranks = [s["dispatch_rank"] for s in sessions]
    if ranks != list(range(len(ranks))):
        raise ValueError("dispatch ranks are not contiguous")
    for row in sessions:
        if row["http_status"] != 200 or row.get("first_token_s") is None:
            raise ValueError(f"session failed: {row['id']}")
        delta = row.get("proxy_delta", {})
        if row["action"] == "S" and delta.get("kv/target_to_client", 0) <= 0:
            raise ValueError(f"KV action lacks KV bytes: {row['id']}")
        if row["action"] == "R" and delta.get("api/client_to_target", 0) <= 0:
            raise ValueError(f"replay action lacks API bytes: {row['id']}")



def run_live_moves(cfg: b.Config, run_root: Path, sessions: list[dict], workers: dict[str, SessionWorker],
                   rows: list[dict], settle_s: float = 2.0, replay_concurrency: int = 1,
                   kv_concurrency: int | None = None) -> list[dict]:
    if replay_concurrency <= 0:
        raise ValueError("replay_concurrency must be positive")
    if kv_concurrency is not None and kv_concurrency <= 0:
        raise ValueError("kv_concurrency must be positive")
    gates = {"R": threading.Semaphore(replay_concurrency), "S": threading.Semaphore(kv_concurrency or max(1, len(rows)))}

    def move(row: dict) -> dict:
        with gates[row["action"]]:
            worker = workers[row["id"]]
            before = b.proxy_counts(run_root / "proxy_bytes.csv")
            move_start = time.time()
            prompt = worker.last_prompt if row["action"] == "S" else f"Replay cache bust {row['id']} {move_start}.\n{worker.last_prompt}"
            result = stream_chat(cfg, cfg.api_proxy_port, prompt, int(sessions[row["fixture_index"]]["decode_tokens"]))
            if result["status"] != 200:
                raise RuntimeError(f"sink move failed for {row['id']}: {result['response_text']}")
            switch_start = time.time()
            worker.pause_boundary()
            worker.switch_to(cfg.api_proxy_port)
            worker.resume()
            switch_end = time.time()
            if settle_s:
                time.sleep(settle_s)
            delta = b.count_delta(before, b.proxy_counts(run_root / "proxy_bytes.csv"))
            completion = switch_end - move_start
            return {
                **row,
                "warm_move": True,
                "move_start_ts": move_start,
                "sink_ready_ts": result["end_ts"],
                "switch_start_ts": switch_start,
                "switch_end_ts": switch_end,
                "move_end_ts": switch_end,
                "http_status": result["status"],
                "first_token_s": result["first_token_ts"] - move_start,
                "completion_s": completion,
                "downtime_s": switch_end - switch_start,
                "deadline_met": completion <= float(row.get("deadline_s", math.inf)),
                "prompt_sha256": result["prompt_sha256"],
                "proxy_delta": delta,
            }

    with ThreadPoolExecutor(max_workers=max(1, len(rows))) as ex:
        return [f.result() for f in [ex.submit(move, row) for row in rows]]

def live_drain(cfg: b.Config, run_root: Path, manifest: dict, mbps: float, nvsmi_ms: int, extra: list[str],
               policy: str = "lp", seed: int = 0, deadline_s: float | None = None,
               target_frac: float | None = None, profile_path: Path | None = None,
               replay_concurrency: int = 1, kv_concurrency: int | None = None) -> Path:
    sessions = live_sessions(manifest)
    stack = b.start_stack(cfg, run_root, mbps, extra)
    events = JsonlLog(run_root / "events.jsonl")
    nvsmi = None
    workers: dict[str, SessionWorker] = {}
    try:
        b.start_sink(stack, cfg, extra)
        profile, used_profile_path = ensure_live_profile(cfg, run_root, profile_path, manifest, mbps)
        manifest = apply_live_profile(manifest, profile)
        sessions = live_sessions(manifest)
        nvsmi = start_nvsmi(run_root / "gpu_power.csv", nvsmi_ms)
        for i, session in enumerate(sessions):
            prompt = session_prompt(session, 0)
            result = stream_chat(cfg, cfg.src_port, prompt, int(session["decode_tokens"]))
            if result["status"] != 200:
                raise RuntimeError(f"source prewarm failed for {session['id']}: {result['response_text']}")
            events.write("prewarm", session_id=session["id"], status=result["status"])
            worker = SessionWorker(cfg, session, cfg.src_port, events, int(manifest.get("source", {}).get("seed", 0)) + i)
            worker.last_prompt = prompt
            worker.start()
            workers[session["id"]] = worker
        baseline_start = time.time()
        time.sleep(float(manifest.get("baseline_s", 120.0)))
        drain_start = time.time()
        summary = live_plan_summary(manifest, policy, seed, deadline_s, target_frac)
        move_rows = [{**row, "deadline_s": summary["deadline_s"]} for row in summary["sessions"]]
        rows = run_live_moves(cfg, run_root, sessions, workers, move_rows, replay_concurrency=replay_concurrency, kv_concurrency=kv_concurrency)
        drain_end = time.time()
        time.sleep(float(manifest.get("settle_s", 120.0)))
        post_end = time.time()
        for worker in workers.values():
            worker.stop()
        windows = {"baseline": (baseline_start, drain_start), "drain": (drain_start, drain_end), "post": (drain_end, post_end)}
        if nvsmi:
            stop_nvsmi(*nvsmi)
            nvsmi = None
        proxy_audit = write_proxy_audit(run_root / "proxy_bytes.csv", run_root / "proxy_audit.csv", windows, mbps)
        write_power_summary(run_root / "gpu_power.csv", run_root / "power_summary.csv", windows)
        write_power_plot(run_root / "gpu_power.csv", run_root / "power_trace.png", rows)
        write_power_plot(run_root / "gpu_power.csv", run_root / "source_power.png", rows, gpu=0)
        write_power_plot(run_root / "gpu_power.csv", run_root / "sink_power.png", rows, gpu=1)
        delays = write_delay_summary(rows, run_root / "delay_summary.csv", run_root / "delay_summary.png")
        request_counts = write_request_counts(run_root / "events.jsonl", run_root / "request_counts.csv", windows)
        write_ell_power5s(run_root / "gpu_power.csv", {**summary, "input_manifest": manifest, "sessions": rows, "windows": windows}, run_root / "ell_power5s.csv", run_root / "ell_power5s.png")
        out = {**summary, "schema": LIVE_SCHEMA, "input_manifest": manifest, "sessions": rows, "delay_summary": delays, "request_counts": request_counts, "proxy_audit": proxy_audit, "windows": windows, "profile_path": str(used_profile_path), "scheduler": {"warm_movement": True, "replay_concurrency": replay_concurrency, "kv_concurrency": kv_concurrency or max(1, len(move_rows))}, "acceptance": {"ok": True, "power_threshold_gated": False, "planned_hit": summary["planned_hit"], "proxy_audit_ok": all(r["ok"] for r in proxy_audit)}}
        check_live_manifest(out, run_root)
        (run_root / "controller_manifest.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    finally:
        for worker in workers.values():
            worker.stop()
        if nvsmi:
            stop_nvsmi(*nvsmi)
        events.close()
        b.stop_stack(stack)
    return run_root



def _fmt(x: float) -> str:
    return f"{float(x):g}".replace(".", "p")


def grid_run_name(policy: str, deadline_s: float, target_frac: float) -> str:
    return f"{policy}_D{_fmt(deadline_s)}_T{_fmt(target_frac)}"


def _power_summary_lookup(path: Path) -> dict[tuple[str, int], float]:
    with path.open() as f:
        return {(r["phase"], int(r["gpu"])): float(r["power_mean_w"]) for r in csv.DictReader(f)}


def grid_summary_row(run_root: Path) -> dict:
    manifest = json.loads((run_root / "controller_manifest.json").read_text())
    power = _power_summary_lookup(run_root / "power_summary.csv")
    delays = manifest.get("sessions") or manifest.get("delay_summary") or []
    if delays and all("move_start_ts" in d and "move_end_ts" in d for d in delays):
        t0 = min(float(d["move_start_ts"]) for d in delays)
        total_first = max(float(d["move_start_ts"]) + float(d["first_token_s"]) for d in delays) - t0
        total_completion = max(float(d["move_end_ts"]) for d in delays) - t0
    else:
        total_first = sum(float(d["first_token_s"]) for d in delays)
        total_completion = sum(float(d["completion_s"]) for d in delays)
    return {
        "policy": manifest["policy"],
        "deadline_s": float(manifest["deadline_s"]),
        "target_frac": float(manifest["target_frac"]),
        "target_w": float(manifest["target_w"]),
        "full_source_drop_w": float(manifest["full_source_drop_w"]),
        "planned_source_drop_w": float(manifest["planned_source_drop_w"]),
        "planned_shortfall_w": float(manifest["planned_shortfall_w"]),
        "planned_hit": bool(manifest["planned_hit"]),
        "deadline_hit": all(bool(d.get("deadline_met", float(d["completion_s"]) <= float(manifest["deadline_s"]))) for d in delays),
        "measured_source_drop_w": power.get(("baseline", 0), math.nan) - power.get(("post", 0), math.nan),
        "measured_sink_rise_w": power.get(("post", 1), math.nan) - power.get(("baseline", 1), math.nan),
        "total_first_token_s": float(total_first),
        "total_completion_s": float(total_completion),
        "max_completion_s": float(max((d["completion_s"] for d in delays), default=math.nan)),
        "sessions": len(manifest.get("sessions", [])),
        "run_root": str(run_root),
    }


def _write_grid_plot(rows: list[dict], out_png: Path, y_key: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for policy in dict.fromkeys(r["policy"] for r in rows):
        part = [r for r in rows if r["policy"] == policy]
        ax.scatter([r["target_frac"] for r in part], [r[y_key] for r in part], label=policy, alpha=0.8)
    ax.set(xlabel="target fraction", ylabel=ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def write_grid_summary(run_roots: list[Path], out_csv: Path, power_png: Path, delay_png: Path) -> list[dict]:
    rows = [grid_summary_row(p) for p in run_roots]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    _write_grid_plot(rows, power_png, "measured_source_drop_w", "measured source drop W")
    _write_grid_plot(rows, delay_png, "total_completion_s", "total completion delay s")
    return rows


def live_grid(cfg: b.Config, run_root: Path, manifest: dict, policies: list[str], deadlines: list[float],
              target_fracs: list[float], mbps: float, nvsmi_ms: int, baseline_s: float,
              settle_s: float, seed: int, extra: list[str], profile_path: Path | None = None,
              replay_concurrency: int = 1, kv_concurrency: int | None = None) -> Path:
    run_roots, base = [], {**manifest, "baseline_s": baseline_s, "settle_s": settle_s}
    profile_path = profile_path or run_root / "live_profile.json"
    for policy in policies:
        for D in deadlines:
            for frac in target_fracs:
                dst = run_root / grid_run_name(policy, D, frac)
                live_drain(cfg, dst, base, mbps, nvsmi_ms, extra, policy, seed, D, frac, profile_path, replay_concurrency, kv_concurrency)
                run_roots.append(dst)
    write_grid_summary(run_roots, run_root / "scenario_summary.csv", run_root / "grid_power_drop.png", run_root / "grid_delay.png")
    return run_root


def _csv_list(text: str, cast=str):
    return [cast(x) for x in text.split(",") if x]


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Queue-Haul Stage 1c controller proof")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fixture")
    plan_p = sub.add_parser("plan")
    plan_p.add_argument("--fixture", type=Path)
    proof_p = sub.add_parser("proof")
    b.add_common(proof_p)
    proof_p.add_argument("--fixture", type=Path)
    proof_p.add_argument("--run-root", type=Path, default=Path("queue-haul/runs/stage1c/proof"))
    proof_p.add_argument("--mbps", type=float, default=1000.0)
    proof_p.add_argument("extra_vllm_args", nargs=argparse.REMAINDER)
    check_p = sub.add_parser("check")
    check_p.add_argument("--run-root", type=Path, default=Path("queue-haul/runs/stage1c/proof"))
    check_p.add_argument("--manifest", type=Path)
    make_p = sub.add_parser("make-manifest")
    make_p.add_argument("--source", choices=("tracelab",), required=True)
    make_p.add_argument("--input", type=Path, required=True)
    make_p.add_argument("--out", type=Path, required=True)
    make_p.add_argument("--sessions", type=int, default=8)
    make_p.add_argument("--seed", type=int, default=0)
    make_p.add_argument("--max-model-len", type=int, default=32768)
    make_p.add_argument("--decode-margin", type=int, default=512)
    make_p.add_argument("--min-turns", type=int, default=3)
    make_p.add_argument("--min-context-tokens", type=int, default=2048)
    live_p = sub.add_parser("live-drain")
    b.add_common(live_p)
    live_p.add_argument("--manifest", type=Path, required=True)
    live_p.add_argument("--run-root", type=Path, default=Path("queue-haul/outputs/stage1c_live"))
    live_p.add_argument("--mbps", type=float, default=1000.0)
    live_p.add_argument("--nvsmi-ms", type=int, default=250)
    live_p.add_argument("--policy", choices=("lp", "random", "greedy"), default="lp")
    live_p.add_argument("--deadline-s", type=float)
    live_p.add_argument("--target-frac", type=float)
    live_p.add_argument("--seed", type=int, default=0)
    live_p.add_argument("--profile", type=Path)
    live_p.add_argument("--replay-concurrency", type=int, default=1)
    live_p.add_argument("--kv-concurrency", type=int, default=0)
    live_p.add_argument("extra_vllm_args", nargs=argparse.REMAINDER)
    grid_p = sub.add_parser("live-grid")
    b.add_common(grid_p)
    grid_p.add_argument("--manifest", type=Path, required=True)
    grid_p.add_argument("--run-root", type=Path, default=Path("queue-haul/outputs/stage1c_grid"))
    grid_p.add_argument("--mbps", type=float, default=1000.0)
    grid_p.add_argument("--nvsmi-ms", type=int, default=250)
    grid_p.add_argument("--policies", default="lp,random,greedy")
    grid_p.add_argument("--deadlines", default="10,30,120")
    grid_p.add_argument("--target-fracs", default="0.25,0.45,0.65")
    grid_p.add_argument("--baseline-s", type=float, default=120.0)
    grid_p.add_argument("--settle-s", type=float, default=120.0)
    grid_p.add_argument("--seed", type=int, default=0)
    grid_p.add_argument("--profile", type=Path)
    grid_p.add_argument("--replay-concurrency", type=int, default=1)
    grid_p.add_argument("--kv-concurrency", type=int, default=0)
    grid_p.add_argument("extra_vllm_args", nargs=argparse.REMAINDER)
    check_live_p = sub.add_parser("check-live")
    check_live_p.add_argument("--run-root", type=Path, default=Path("queue-haul/outputs/stage1c_live"))
    check_live_p.add_argument("--manifest", type=Path)
    plot_live_p = sub.add_parser("plot-live")
    plot_live_p.add_argument("--run-root", type=Path, default=Path("queue-haul/outputs/stage1c_live"))
    plot_live_p.add_argument("--manifest", type=Path)
    plot_live_p.add_argument("--bucket-s", type=float, default=5.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.cmd == "fixture":
        print(json.dumps(default_fixture(), indent=2, sort_keys=True))
    elif args.cmd == "plan":
        print(json.dumps(plan_summary(load_fixture(args.fixture)), indent=2, sort_keys=True))
    elif args.cmd == "proof":
        cfg = b.config_from_args(args)
        extra = args.extra_vllm_args[1:] if args.extra_vllm_args[:1] == ["--"] else args.extra_vllm_args
        print(proof(cfg, args.run_root, load_fixture(args.fixture), args.mbps, extra))
    elif args.cmd == "check":
        path = args.manifest or args.run_root / "controller_manifest.json"
        check_manifest(json.loads(path.read_text()))
        print(path)
    elif args.cmd == "make-manifest":
        manifest = tracelab_manifest(args.input, args.sessions, args.seed, args.max_model_len, args.decode_margin, args.min_turns, args.min_context_tokens)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        print(args.out)
    elif args.cmd == "live-drain":
        cfg = b.config_from_args(args)
        extra = args.extra_vllm_args[1:] if args.extra_vllm_args[:1] == ["--"] else args.extra_vllm_args
        print(live_drain(cfg, args.run_root, json.loads(args.manifest.read_text()), args.mbps, args.nvsmi_ms, extra, args.policy, args.seed, args.deadline_s, args.target_frac, args.profile, args.replay_concurrency, args.kv_concurrency or None))
    elif args.cmd == "live-grid":
        cfg = b.config_from_args(args)
        extra = args.extra_vllm_args[1:] if args.extra_vllm_args[:1] == ["--"] else args.extra_vllm_args
        print(live_grid(cfg, args.run_root, json.loads(args.manifest.read_text()), _csv_list(args.policies), _csv_list(args.deadlines, float), _csv_list(args.target_fracs, float), args.mbps, args.nvsmi_ms, args.baseline_s, args.settle_s, args.seed, extra, args.profile, args.replay_concurrency, args.kv_concurrency or None))
    elif args.cmd == "check-live":
        path = args.manifest or args.run_root / "controller_manifest.json"
        check_live_manifest(json.loads(path.read_text()), args.run_root)
        print(path)
    elif args.cmd == "plot-live":
        path = args.manifest or args.run_root / "controller_manifest.json"
        manifest = json.loads(path.read_text())
        write_ell_power5s(args.run_root / "gpu_power.csv", manifest, args.run_root / "ell_power5s.csv", args.run_root / "ell_power5s.png", args.bucket_s)
        write_request_counts(args.run_root / "events.jsonl", args.run_root / "request_counts.csv", manifest["windows"])
        check_live_manifest(manifest, args.run_root)
        print(args.run_root / "ell_power5s.png")


if __name__ == "__main__":
    main()
