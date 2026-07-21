from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
import statistics
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

import migration_testbed as testbed
from destination_evaluation import reduce_bounds, reduce_loaded
from migration_profiler import JOB_CLASSES, object_hash, write_json


SCHEMA = "queue-haul-destination-campaign-v1"
TRACE_SCHEMA = "queue-haul-trace-v1"
EVIDENCE = {
    "service_envelopes": "measure",
    "loaded_migration_slowdown": "measure",
    "foreground_impact": "measure",
    "kv_correctness": "measure",
    "continuation": "measure",
    "wan_capacity": "public_constant",
    "kv_bytes": "derive",
    "trace_growth": "derive",
    "residency_horizon": "derive",
    "headroom_grid": "derive",
    "old_service_results": "prior_only",
    "old_migration_results": "prior_only",
}
IMAGE_SHA256 = "50e98f65de09ebfe196f270c8b5c595636853646eb5536dca92f27bd45c084ab"
SLO = {
    "normal": {"p90_ttft_s": 2, "p90_mean_tpot_s": 0.1},
    "emergency": {"p90_ttft_s": 10, "p90_mean_tpot_s": 0.25},
}


def audit_evidence(inventory: dict = EVIDENCE) -> dict:
    allowed = {"measure", "derive", "public_constant", "prior_only"}
    bad = sorted(set(inventory.values()) - allowed)
    if bad:
        raise ValueError(f"unclassified evidence: {bad}")
    return {
        "schema": SCHEMA,
        "inventory": inventory,
        "gpu_measurements": sorted(k for k, v in inventory.items() if v == "measure"),
    }


def make_plan() -> dict:
    jobs = [
        (1, "anchors_coding_frontier", (), (), False),
        (2, "interactive_agentic_frontier", ("preflight",), (1,), False),
        (3, "mixed_frontier_fit", ("frontier_1", "frontier_2"), (1, 2), False),
        (4, "loaded_migration", ("profile_frozen",), (3,), False),
        (5, "independent_validation", ("loaded_valid",), (4,), False),
        (6, "adaptive_repeats", ("repeat_needed",), (5,), True),
    ]
    return {
        "schema": SCHEMA,
        "image_sha256": IMAGE_SHA256,
        "gpu_pair_hour_budget": 72,
        "jobs": [
            {
                "shard": n,
                "name": name,
                "hours": 12,
                "requires": list(req),
                "after_shards": list(after),
                "conditional": conditional,
            }
            for n, name, req, after, conditional in jobs
        ],
        "service": {
            "directions": ["prefill", "decode", *JOB_CLASSES],
            "arrival": "open_loop_poisson",
            "arrival_sensitivity": "gamma_cv2",
            "warmup_s": 60,
            "hold_min_s": 180,
            "hold_max_s": 480,
            "completion_target": 200,
            "run_cap_s": 720,
            "initial_repeats": 3,
            "disagreement_repeats": 5,
            "radial_resolution": 0.05,
            "block_bootstrap_s": 30,
            "bootstrap_samples": 2000,
            "slos": SLO,
        },
        "migration": {
            "methods": ["replay", "kv_transfer"],
            "rho": [0, 0.5, 0.8, 0.95, "emergency_inside"],
            "emergency_inside_fraction": 0.95,
            "context_tokens": 16384,
            "bandwidth_gbps": 10,
            "heldout_context_tokens": 24576,
            "heldout_bandwidth_gbps": 5,
            "concurrency": 1,
            "repeats": 3,
            "paired_controls": True,
        },
    }


def validate_plan(plan: dict) -> None:
    if (
        plan.get("schema") != SCHEMA
        or sum(j["hours"] for j in plan["jobs"]) > plan["gpu_pair_hour_budget"]
    ):
        raise ValueError("campaign exceeds its A100-pair-hour budget")
    if (
        plan.get("image_sha256") != IMAGE_SHA256
        or plan["migration"]["rho"][-1] != "emergency_inside"
    ):
        raise ValueError("campaign runtime or emergency migration point changed")


def boundary_decision(labels: list[str]) -> str:
    if len(labels) not in (3, 5) or not set(labels) <= {"feasible", "infeasible"}:
        raise ValueError("boundary decisions need three or five valid runs")
    counts = {label: labels.count(label) for label in set(labels)}
    if (
        len(labels) == 3
        and len(counts) == 1
        or len(labels) == 5
        and max(counts.values()) >= 4
    ):
        return max(counts, key=counts.get)
    raise ValueError("boundary disagreement requires a four-of-five decision")


def check_gate(report: dict, phase: str) -> None:
    required = {
        "preflight": {
            "image_sha256": IMAGE_SHA256,
            "gpu_count": 2,
            "same_session_cache_hit": True,
            "cross_session_cache_hits": 0,
            "tokenizer_ok": True,
        },
        "frontier": {
            "false_feasible": 0,
            "median_radial_error_max": 0.15,
            "profile_frozen": True,
        },
        "loaded": {"paired_controls": True, "queue_gate": True, "loaded_valid": True},
        "validation": {"false_feasible": 0, "kv_correct": True, "continuation": True},
    }[phase]
    failed = [key for key, value in required.items() if report.get(key) != value]
    if failed:
        raise ValueError(f"{phase} gate failed: {', '.join(failed)}")


def _rates(
    rows: list[dict], metric: str, reducer
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    selected = [r for r in rows if r["metric"] == metric]
    contexts = sorted({float(r["context_tokens"]) for r in selected})
    if len(contexts) < 2 or any(
        len({r["run_id"] for r in selected if float(r["context_tokens"]) == x}) < 3
        for x in contexts
    ):
        raise ValueError(f"{metric} anchors need three runs at two contexts")
    return tuple(contexts), tuple(
        float(
            reducer(
                [
                    float(r["tokens_per_s"])
                    for r in selected
                    if float(r["context_tokens"]) == x
                ]
            )
        )
        for x in contexts
    )


def reduce_profile(
    anchors: list[dict],
    service: list[dict],
    loaded: list[dict],
    identity: dict,
    normals: list[list[float]],
) -> dict:
    bounds, migration = (
        reduce_bounds(service),
        reduce_loaded(loaded, identity["provenance"]),
    )
    profiles = {}
    for case, reducer in (("central", statistics.median), ("conservative", min)):
        profiles[case] = {
            "type_id": "gpt-oss-20b-a100-tp1",
            "compatibility": identity["compatibility"],
            "prefill": _rates(anchors, "prefill", reducer),
            "decode": _rates(anchors, "decode", reducer),
            "normals": normals,
            "bounds": bounds[case],
            "kv_capacity_tokens": identity["kv_capacity_tokens"],
            "loaded": {
                method: asdict(value) for method, value in migration[case].items()
            },
            "workload_prefill_fraction_range": identity[
                "workload_prefill_fraction_range"
            ],
            "provenance": identity["provenance"],
            "synthetic": False,
        }
    if any(
        c > k
        for mode in SLO
        for c, k in zip(
            profiles["conservative"]["bounds"][mode],
            profiles["central"]["bounds"][mode],
        )
    ):
        raise ValueError("conservative envelope exceeds central envelope")
    return {
        "schema": SCHEMA,
        "profiles": profiles,
        "input_sha256": object_hash([anchors, service, loaded, identity, normals]),
    }


def _read_table(path: Path) -> list[dict]:
    if path.suffix == ".csv":
        with path.open() as handle:
            return list(csv.DictReader(handle))
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _get_json(url: str, opener=urllib.request.urlopen) -> dict:
    with opener(url) as response:
        return json.load(response)


def fetch_dataset(
    dataset: str,
    out: Path,
    config: str | None = None,
    split: str | None = None,
    opener=urllib.request.urlopen,
) -> dict:
    encoded = urllib.parse.quote(dataset, safe="")
    info_url = f"https://huggingface.co/api/datasets/{encoded}"
    revision = _get_json(info_url, opener)["sha"]
    if config is None or split is None:
        choices = _get_json(
            f"https://datasets-server.huggingface.co/splits?dataset={encoded}", opener
        )["splits"]
        choice = next((x for x in choices if x["split"] == "train"), choices[0])
        config, split = config or choice["config"], split or choice["split"]
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": 100,
            }
        )
        page = _get_json(f"https://datasets-server.huggingface.co/rows?{query}", opener)
        batch, total = [x["row"] for x in page["rows"]], int(page["num_rows_total"])
        if not batch and offset < total:
            raise RuntimeError(f"dataset server stopped at {offset}/{total}")
        rows += batch
        offset += len(batch)
    if _get_json(info_url, opener)["sha"] != revision:
        raise RuntimeError(f"{dataset} changed during download")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    )
    metadata = {
        "dataset": dataset,
        "revision": revision,
        "config": config,
        "split": split,
        "rows": len(rows),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    write_json(out.with_suffix(out.suffix + ".metadata.json"), metadata)
    return metadata


def checksums(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    }


def write_checksums(root: Path) -> Path:
    path = root / "SHA256SUMS"
    path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums(root).items())
    )
    return path


def verify_checksums(root: Path) -> None:
    manifest = root / "SHA256SUMS"
    if not manifest.exists():
        raise ValueError("missing SHA256SUMS")
    expected = {
        name: digest
        for digest, name in (
            line.strip().split("  ", 1)
            for line in manifest.read_text().splitlines()
            if line.strip()
        )
    }
    actual = checksums(root)
    if expected != actual:
        raise ValueError("artifact checksum mismatch")


def submit(
    plan_path: Path,
    sbatch: Path,
    job_dir: Path,
    include_reserve: bool = False,
    run=subprocess.run,
) -> dict[int, str]:
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    ids = {}
    for job in plan["jobs"]:
        if job["conditional"] and not include_reserve:
            continue
        job_file = (job_dir / f"shard-{job['shard']}.sh").resolve()
        if not job_file.is_file() or not job_file.with_suffix(".sh.sha256").is_file():
            raise ValueError(f"missing immutable job file for shard {job['shard']}")
        expected = job_file.with_suffix(".sh.sha256").read_text().split()[0]
        if hashlib.sha256(job_file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"job file checksum changed for shard {job['shard']}")
        command = ["sbatch", "--parsable"]
        deps = [ids[n] for n in job["after_shards"]]
        if deps:
            command += [f"--dependency=afterok:{':'.join(deps)}"]
        command += [
            f"--export=ALL,QH_SHARD={job['shard']},QH_JOB_FILE={job_file},QH_CAMPAIGN_PLAN={plan_path.resolve()}",
            str(sbatch),
        ]
        result = run(command, check=True, capture_output=True, text=True)
        ids[job["shard"]] = result.stdout.strip().split(";")[0]
    return ids


def sync(remote: str, remote_root: str, local: Path, run=subprocess.run) -> None:
    local.mkdir(parents=True, exist_ok=True)
    source = f"{remote}:{remote_root.rstrip('/')}/"
    run(
        [
            "rsync",
            "-a",
            "--partial",
            "--checksum",
            source + "SHA256SUMS",
            str(local / "SHA256SUMS"),
        ],
        check=True,
    )
    run(
        ["rsync", "-a", "--partial", "--checksum", source, str(local) + "/"], check=True
    )
    verify_checksums(local)


def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def _messages(row: dict) -> list[dict]:
    value = next(
        (row[k] for k in ("messages", "trajectory", "events") if row.get(k)), None
    )
    value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(value, list):
        raise ValueError("trace row has no message list")
    return [
        {
            "role": str(m.get("role", m.get("type", ""))).lower(),
            "content": _text(m.get("content", m.get("message", m.get("text", "")))),
            **(
                {"timestamp": m[k]}
                if (
                    k := next(
                        (
                            x
                            for x in ("timestamp", "created_at", "time")
                            if m.get(x) is not None
                        ),
                        None,
                    )
                )
                else {}
            ),
        }
        for m in value
    ]


def _seconds(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return (
        __import__("datetime")
        .datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        .timestamp()
    )


def normalize_traces(
    rows: list[dict], source: str, revision: str, count_tokens
) -> list[dict]:
    out = []
    for index, row in enumerate(rows):
        messages = _messages(row)
        session_id = str(
            next(
                (
                    row[k]
                    for k in ("session_id", "id", "instance_id", "trace_id")
                    if row.get(k) is not None
                ),
                index,
            )
        )
        previous = 0
        for turn, stop in enumerate(
            i for i, m in enumerate(messages) if m["role"] == "assistant"
        ):
            prefix, answer = messages[:stop], messages[stop]
            total = count_tokens(prefix)
            if total < 1 or total > 32768:
                continue
            timestamp = next(
                (
                    m.get("timestamp")
                    for m in reversed(prefix)
                    if m.get("timestamp") is not None
                ),
                None,
            )
            out.append(
                {
                    "schema": TRACE_SCHEMA,
                    "source": source,
                    "revision": revision,
                    "license": "CC-BY-4.0",
                    "session_id": f"{source}:{session_id}",
                    "turn": turn,
                    "time_s": _seconds(timestamp) if timestamp is not None else None,
                    "input_tokens_total": total,
                    "newly_append_tokens": max(1, total - previous),
                    "output_tokens": max(1, count_tokens(answer["content"])),
                    "current_user_message_count": sum(
                        m["role"] == "user" for m in prefix
                    ),
                    "tool_message_count": sum(m["role"] == "tool" for m in prefix),
                    "reset": total < previous,
                    "content_sha256": hashlib.sha256(
                        json.dumps(messages[: stop + 1], sort_keys=True).encode()
                    ).hexdigest(),
                }
            )
            previous = total
    return out


def _sessions(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["session_id"], []).append(row)
    return grouped


def classify(rows: list[dict]) -> list[dict]:
    grouped = _sessions(rows)
    nvidia = {k for k in grouped if k.startswith("nvidia/")}
    timed = []
    for session_id, turns in grouped.items():
        times = sorted(r["time_s"] for r in turns if r["time_s"] is not None)
        if not session_id.startswith("nvidia/") and len(times) > 1:
            timed.append((session_id, (times[-1] - times[0]) / (len(times) - 1)))
    if len(timed) < 48 or len(nvidia) < 24:
        raise ValueError("need 48 timestamped Trace Commons and 24 NVIDIA sessions")
    interactive = {
        k for k, _ in sorted(timed, key=lambda x: (-x[1], x[0]))[: len(timed) // 2]
    }
    return [
        dict(
            row,
            job_class="agentic_tool_loop"
            if row["session_id"] in nvidia
            else "interactive_coding"
            if row["session_id"] in interactive
            else "coding",
        )
        for row in rows
    ]


def build_manifests(rows: list[dict], seed: int = 0) -> dict:
    rows = classify(rows)
    grouped = _sessions(rows)
    manifests = {}
    for job_class in JOB_CLASSES:
        sessions = [
            (sid, sorted(turns, key=lambda r: r["turn"]))
            for sid, turns in grouped.items()
            if turns[0]["job_class"] == job_class
        ]
        sessions.sort(
            key=lambda item: (
                max(r["input_tokens_total"] for r in item[1]),
                object_hash([seed, item[0]]),
            )
        )
        if len(sessions) < 24:
            raise ValueError(f"need 24 {job_class} sessions, found {len(sessions)}")
        chosen = [sessions[round(i * (len(sessions) - 1) / 23)] for i in range(24)]
        manifests[job_class] = {
            split: [
                sid
                for i, (sid, _) in enumerate(chosen)
                if ("fit", "fit", "tune", "validation")[i % 4] == split
            ]
            for split in ("fit", "tune", "validation")
        }
    return {
        "schema": SCHEMA,
        "trace_schema": TRACE_SCHEMA,
        "seed": seed,
        "rows_sha256": object_hash(
            sorted(rows, key=lambda r: (r["session_id"], r["turn"]))
        ),
        "splits": manifests,
    }


def token_counter(host: str, port: int, model: str):
    def count(value) -> int:
        payload = {"model": model, "add_generation_prompt": isinstance(value, list)}
        payload["messages" if isinstance(value, list) else "prompt"] = value
        result = testbed.http_json(host, port, "POST", "/tokenize", payload)
        if result.get("count") is None:
            raise RuntimeError("vLLM tokenizer returned no count")
        return int(result["count"])

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Destination measurement campaign")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-evidence")
    audit.add_argument("--out", type=Path, required=True)
    plan = sub.add_parser("make-plan")
    plan.add_argument("--out", type=Path, required=True)
    fetch = sub.add_parser("fetch-traces")
    fetch.add_argument("--out-dir", type=Path, required=True)
    fetch.add_argument("--trace-config")
    fetch.add_argument("--trace-split")
    fetch.add_argument("--nvidia-config")
    fetch.add_argument("--nvidia-split")
    submit_parser = sub.add_parser("submit-next")
    submit_parser.add_argument("--plan", type=Path, required=True)
    submit_parser.add_argument(
        "--sbatch",
        type=Path,
        default=Path(__file__).with_name("destination_campaign.sbatch"),
    )
    submit_parser.add_argument("--job-dir", type=Path, required=True)
    submit_parser.add_argument("--include-reserve", action="store_true")
    sync_parser = sub.add_parser("sync")
    sync_parser.add_argument("--remote", default=os.environ.get("QH_REMOTE"))
    sync_parser.add_argument("--remote-root", default=os.environ.get("QH_REMOTE_ROOT"))
    sync_parser.add_argument("--local", type=Path, required=True)
    checksum = sub.add_parser("checksums")
    checksum.add_argument("--root", type=Path, required=True)
    check = sub.add_parser("check")
    check.add_argument("--report", type=Path, required=True)
    check.add_argument(
        "--phase",
        required=True,
        choices=("preflight", "frontier", "loaded", "validation"),
    )
    reduce = sub.add_parser("reduce")
    for name in ("anchors", "service", "loaded", "identity"):
        reduce.add_argument(f"--{name}", type=Path, required=True)
    reduce.add_argument("--normals", type=json.loads, required=True)
    reduce.add_argument("--out", type=Path, required=True)
    build = sub.add_parser("build-manifests")
    build.add_argument("--trace-commons", type=Path, required=True)
    build.add_argument("--nvidia", type=Path, required=True)
    build.add_argument("--trace-revision", required=True)
    build.add_argument("--nvidia-revision", required=True)
    build.add_argument("--host", default="127.0.0.1")
    build.add_argument("--port", type=int, default=8000)
    build.add_argument("--model", default=testbed.MODEL)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit-evidence":
        write_json(args.out, audit_evidence())
        return
    if args.command == "make-plan":
        plan = make_plan()
        validate_plan(plan)
        write_json(args.out, plan)
        return
    if args.command == "fetch-traces":
        fetch_dataset(
            "trace-commons/agent-traces",
            args.out_dir / "trace-commons.jsonl",
            args.trace_config,
            args.trace_split,
        )
        fetch_dataset(
            "nvidia/SWE-Hero-openhands-trajectories",
            args.out_dir / "nvidia-swe-hero.jsonl",
            args.nvidia_config,
            args.nvidia_split,
        )
        return
    if args.command == "submit-next":
        print(
            json.dumps(
                submit(args.plan, args.sbatch, args.job_dir, args.include_reserve),
                sort_keys=True,
            )
        )
        return
    if args.command == "sync":
        if not args.remote or not args.remote_root:
            raise ValueError("sync needs QH_REMOTE and QH_REMOTE_ROOT")
        sync(args.remote, args.remote_root, args.local)
        return
    if args.command == "checksums":
        write_checksums(args.root)
        return
    if args.command == "check":
        check_gate(json.loads(args.report.read_text()), args.phase)
        return
    if args.command == "reduce":
        write_json(
            args.out,
            reduce_profile(
                _read_table(args.anchors),
                _read_table(args.service),
                _read_table(args.loaded),
                json.loads(args.identity.read_text()),
                args.normals,
            ),
        )
        return
    counter = token_counter(args.host, args.port, args.model)
    def load(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = normalize_traces(
        load(args.trace_commons),
        "trace-commons/agent-traces",
        args.trace_revision,
        counter,
    )
    rows += normalize_traces(
        load(args.nvidia),
        "nvidia/SWE-Hero-openhands-trajectories",
        args.nvidia_revision,
        counter,
    )
    write_json(args.out, {"manifest": build_manifests(rows, args.seed), "traces": rows})


if __name__ == "__main__":
    main()
