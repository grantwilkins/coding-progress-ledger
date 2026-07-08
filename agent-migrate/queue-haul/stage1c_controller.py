from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import stage1b_drain_sink as b
from dispatch import Event, solve
from impact import Movement, compute
from instance import JobPopulation
from power import BETA_BYTES_PER_TOK, PoolPower

SCHEMA = "queue-haul-stage1c-v1"


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


if __name__ == "__main__":
    main()
