from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
import re
import statistics
import subprocess
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from pathlib import Path

import migration_testbed as testbed
from destination_evaluation import reduce_bounds, reduce_loaded
from migration_profiler import JOB_CLASSES, file_hash, object_hash, write_json


SCHEMA = "queue-haul-destination-campaign-v2"
MANIFEST_SCHEMA = "queue-haul-destination-campaign-v1"
TRACE_SCHEMA = "queue-haul-trace-v1"
NORMALIZER_VERSION = "queue-haul-normalizer-v2"
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
    "old_service_results": "reuse",
    "old_migration_results": "reuse",
}
IMAGE_SHA256 = "50e98f65de09ebfe196f270c8b5c595636853646eb5536dca92f27bd45c084ab"
BASELINE_PROFILE = Path(__file__).with_name("profiles") / "gpt_oss_20b_a100_tp1.json"
SLO = {
    "normal": {"p90_ttft_s": 2, "p90_mean_tpot_s": 0.1},
    "emergency": {"p90_ttft_s": 10, "p90_mean_tpot_s": 0.25},
}
SOURCES = {
    "trace-commons/agent-traces": (
        "CC-BY-4.0-compilation",
        "contributor-certified public repositories",
    ),
    "allenai/WildChat-1M": ("ODC-BY-1.0", "affirmative opt-in"),
    "nvidia/SWE-Hero-openhands-trajectories": (
        "CC-BY-4.0",
        "permissively licensed source repositories",
    ),
}


def audit_evidence(inventory: dict = EVIDENCE) -> dict:
    allowed = {"measure", "derive", "public_constant", "reuse"}
    bad = sorted(set(inventory.values()) - allowed)
    if bad:
        raise ValueError(f"unclassified evidence: {bad}")
    return {
        "schema": SCHEMA,
        "inventory": inventory,
        "gpu_measurements": sorted(k for k, v in inventory.items() if v == "measure"),
    }


def _manifest(path: Path) -> dict:
    value = json.loads(path.read_text())
    manifest = value.get("manifest", {})
    if manifest.get("schema") != MANIFEST_SCHEMA or not value.get("traces"):
        raise ValueError("campaign needs a complete content-free manifest")
    usable = {r["session_id"] for r in value["traces"]
              if not r.get("reset") and 256 <= int(r["input_tokens_total"]) <= 24576}
    for splits in manifest.get("splits", {}).values():
        if [len(splits.get(k, ())) for k in ("fit", "tune", "validation")] != [12, 6, 6]:
            raise ValueError("campaign manifest must preserve 12/6/6 splits")
        if not set().union(*map(set, splits.values())) <= usable:
            raise ValueError("campaign splits contain unusable session shapes")
    return value


def make_plan(manifest_path: Path) -> dict:
    manifest = _manifest(manifest_path)
    return {
        "schema": SCHEMA, "image_sha256": IMAGE_SHA256,
        "gpu_pair_hour_budget": 12, "reserve_pair_hour_limit": 12,
        "job": {"name": "mandatory", "hours": 12},
        "manifest": {"path": "content-free-manifest.json",
                     "sha256": object_hash(manifest)},
        "baseline_profile": {"path": "baseline-profile.json",
                             "sha256": file_hash(BASELINE_PROFILE)},
        "anchor_drift_limit": .15,
        "service": {
            "anchors": [4096, 16384, 24576],
            "directions": list(JOB_CLASSES),
            "arrival": "open_loop_poisson",
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
            "cache_block_tokens": 16,
            "slos": SLO,
        },
        "migration": {
            "methods": ["replay", "kv_transfer"],
            "rho": [0, 0.8, "emergency_inside"],
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
        or plan.get("job", {}).get("hours") != 12
        or plan.get("gpu_pair_hour_budget") != 12
        or plan.get("reserve_pair_hour_limit") != 12
    ):
        raise ValueError("campaign exceeds its A100-pair-hour budget")
    if (
        plan.get("image_sha256") != IMAGE_SHA256
        or plan.get("baseline_profile", {}).get("sha256") != file_hash(BASELINE_PROFILE)
        or plan["migration"]["rho"] not in (
            [0, .8, "emergency_inside"], [.8, "emergency_inside"]
        )
        or plan["migration"]["rho"][-1] != "emergency_inside"
        or plan.get("anchor_drift_limit") != .15
        or plan.get("service", {}).get("cache_block_tokens") != 16
    ):
        raise ValueError("campaign runtime or emergency migration point changed")


def prepare(manifest_path: Path, out: Path) -> dict:
    manifest, plan = _manifest(manifest_path), None
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "content-free-manifest.json", manifest)
    (out / "baseline-profile.json").write_bytes(BASELINE_PROFILE.read_bytes())
    plan = make_plan(out / "content-free-manifest.json")
    write_json(out / "plan.json", plan)
    job = out / "mandatory.sh"
    job.write_text("""#!/usr/bin/env bash
set -euo pipefail
uv run python queue-haul/destination_runner.py --plan "$QH_CAMPAIGN_PLAN" --run-root "$QH_RUN_ROOT"
""")
    job.chmod(0o755)
    job.with_suffix(".sh.sha256").write_text(file_hash(job) + "\n")
    write_json(out / "evidence.json", audit_evidence())
    write_checksums(out)
    return plan


def boundary_decision(labels: list[str]) -> str:
    if len(labels) not in (3, 5) or not set(labels) <= {"feasible", "infeasible"}:
        raise ValueError("boundary decisions need three or five valid runs")
    counts = {label: labels.count(label) for label in set(labels)}
    return max(counts, key=counts.get)


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


def acceptance_report(service: list[dict], loaded: list[dict]) -> dict:
    radial = [abs(float(r["predicted_bound"]) / float(r["actual_bound"]) - 1) for r in service]
    migration = [abs(float(r["predicted_s"]) / float(r["observed_s"]) - 1) for r in loaded]
    false = [r["cell"] for r in service if r["predicted_feasible"] and not r["actual_feasible"]]
    interactions = [r["cell"] for r, error in zip(loaded, migration) if error > .15]
    correctness = [r["cell"] for r in loaded if not r.get("correct", False)]
    if not radial or not migration:
        raise ValueError("acceptance needs service and migration validation rows")
    report = {
        "false_feasible": false,
        "median_radial_error": statistics.median(radial),
        "median_migration_error": statistics.median(migration),
        "loaded_interactions": interactions,
        "correctness_failures": correctness,
    }
    report["accepted"] = not false and report["median_radial_error"] <= .15 \
        and report["median_migration_error"] <= .15 and not correctness
    return report


def reserve_tasks(report: dict) -> list[dict]:
    tasks = [{"phase": "service", "cell": cell, "reason": "boundary_disagreement"}
             for cell in report.get("boundary_disagreements", [])]
    if report.get("false_feasible") or report.get("median_radial_error", 0) > .15:
        tasks += [{"phase": "service", "cell": cell, "reason": "facet_validation"}
                  for cell in report.get("false_feasible") or report.get("service_validation_cells", [])]
    tasks += [{"phase": "migration", "cell": cell, "reason": "interaction"}
              for cell in report.get("loaded_interactions", [])]
    tasks += [{"phase": "migration", "cell": cell, "reason": "correctness"}
              for cell in report.get("correctness_failures", [])]
    return list({object_hash(row): row for row in tasks}.values())


def prepare_reserve(report_path: Path, bundle: Path, out: Path) -> dict | None:
    tasks = reserve_tasks(json.loads(report_path.read_text()))
    if not tasks:
        return None
    source = json.loads((bundle / "plan.json").read_text())
    validate_plan(source)
    out.mkdir(parents=True, exist_ok=True)
    plan = {
        **source,
        "job": {"name": "reserve", "hours": 12},
        "migration": {**source["migration"], "rho": [0, .8, "emergency_inside"]},
        "reserve_tasks": tasks,
    }
    for name in ("content-free-manifest.json", "baseline-profile.json"):
        (out / name).write_bytes((bundle / name).read_bytes())
    write_json(out / "plan.json", plan)
    job = out / "reserve.sh"
    job.write_text("""#!/usr/bin/env bash
set -euo pipefail
uv run python queue-haul/destination_runner.py --plan "$QH_CAMPAIGN_PLAN" --run-root "$QH_RUN_ROOT"
""")
    job.chmod(0o755)
    job.with_suffix(".sh.sha256").write_text(file_hash(job) + "\n")
    write_checksums(out)
    return plan


def _read_table(path: Path) -> list[dict]:
    if path.suffix == ".csv":
        with path.open() as handle:
            return list(csv.DictReader(handle))
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _get_json(url: str, opener=urllib.request.urlopen) -> dict:
    for attempt in range(4):
        try:
            with opener(url) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code != 429 or attempt == 3:
                raise
            time.sleep(min(10, int(error.headers.get("Retry-After", "10"))))
    raise AssertionError("unreachable")


def fetch_dataset(
    dataset: str,
    out: Path,
    config: str | None = None,
    split: str | None = None,
    predicate=None,
    target: int | None = None,
    sample_pages: int = 128,
    opener=urllib.request.urlopen,
) -> dict:
    encoded = urllib.parse.quote(dataset, safe="")
    info_url = (
        f"https://huggingface.co/api/datasets/{urllib.parse.quote(dataset, safe='/')}"
    )
    revision = _get_json(info_url, opener)["sha"]
    metadata_path = out.with_suffix(out.suffix + ".metadata.json")
    if out.is_file() and metadata_path.is_file():
        cached = json.loads(metadata_path.read_text())
        if (
            (cached.get("dataset"), cached.get("revision")) == (dataset, revision)
            and (config is None or cached.get("config") == config)
            and (split is None or cached.get("split") == split)
            and cached.get("sha256") == hashlib.sha256(out.read_bytes()).hexdigest()
        ):
            return cached
    if config is None or split is None:
        choices = _get_json(
            f"https://datasets-server.huggingface.co/splits?dataset={encoded}", opener
        )["splits"]
        choice = next((x for x in choices if x["split"] == "train"), choices[0])
        config, split = config or choice["config"], split or choice["split"]

    def page(offset):
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": 100,
            }
        )
        result = _get_json(
            f"https://datasets-server.huggingface.co/rows?{query}", opener
        )
        return [x["row"] for x in result["rows"]], int(result["num_rows_total"])

    first, total = page(0)
    if target:
        offsets = sorted(
            {
                round(i * max(0, total - 100) / max(1, sample_pages - 1))
                for i in range(sample_pages)
            }
            - {0},
            key=lambda offset: object_hash([dataset, offset]),
        )
        batches, rows = [first], [row for row in first if predicate(row)]
        for offset in offsets:
            if len(rows) >= target:
                break
            time.sleep(1)
            batch, _ = page(offset)
            batches.append(batch)
            rows += [row for row in batch if predicate(row)]
        rows = sorted(rows, key=object_hash)[:target]
        if len(rows) < target:
            raise ValueError(
                f"need {target} eligible {dataset} rows, found {len(rows)}"
            )
        scanned = sum(map(len, batches))
    else:
        rows, offset = first, len(first)
        while offset < total:
            batch, _ = page(offset)
            if not batch:
                raise RuntimeError(f"dataset server stopped at {offset}/{total}")
            rows += batch
            offset += len(batch)
        scanned = len(rows)
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
        "source_rows": total,
        "scanned_rows": scanned,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    write_json(metadata_path, metadata)
    return metadata


def stable_sample(batches, predicate, target: int) -> list[dict]:
    selected = {}
    for batch in batches:
        rows = [
            json.loads(json.dumps(row, default=str)) for row in batch if predicate(row)
        ]
        selected.update((object_hash(row), row) for row in rows)
        selected = dict(sorted(selected.items())[:target])
    if len(selected) < target:
        raise ValueError(f"need {target} eligible rows, found {len(selected)}")
    return list(selected.values())


def fetch_parquet_sample(
    dataset: str,
    out: Path,
    predicate,
    target: int,
    config: str | None = None,
    split: str | None = None,
    opener=urllib.request.urlopen,
    run=subprocess.run,
) -> dict:
    info_url = (
        f"https://huggingface.co/api/datasets/{urllib.parse.quote(dataset, safe='/')}"
    )
    revision = _get_json(info_url, opener)["sha"]
    metadata_path = out.with_suffix(out.suffix + ".metadata.json")
    if out.is_file() and metadata_path.is_file():
        cached = json.loads(metadata_path.read_text())
        if (cached.get("dataset"), cached.get("revision")) == (
            dataset,
            revision,
        ) and cached.get("sha256") == hashlib.sha256(out.read_bytes()).hexdigest():
            return cached
    index = _get_json(
        "https://datasets-server.huggingface.co/parquet?"
        + urllib.parse.urlencode({"dataset": dataset}),
        opener,
    )["parquet_files"]
    files = [
        row
        for row in index
        if (config is None or row["config"] == config)
        and (split is None or row["split"] == split)
    ]
    if not files:
        raise ValueError(f"no matching Parquet shard for {dataset}")
    shard = sorted(files, key=lambda row: row["filename"])[
        int(revision[:16], 16) % len(files)
    ]
    parquet = out.with_suffix(".parquet")
    parquet.parent.mkdir(parents=True, exist_ok=True)
    if not parquet.is_file() or parquet.stat().st_size != int(shard["size"]):
        run(
            [
                "curl",
                "-fL",
                "--retry",
                "3",
                "--continue-at",
                "-",
                "-o",
                str(parquet),
                shard["url"],
            ],
            check=True,
        )
    import pyarrow.parquet as pq

    batches = (
        batch.to_pylist()
        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=1000)
    )
    rows = stable_sample(batches, predicate, target)
    if _get_json(info_url, opener)["sha"] != revision:
        raise RuntimeError(f"{dataset} changed during download")
    out.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    )
    metadata = {
        "dataset": dataset,
        "revision": revision,
        "config": shard["config"],
        "split": shard["split"],
        "shard": shard["filename"],
        "rows": len(rows),
        "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    write_json(metadata_path, metadata)
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


def submit(plan_path: Path, sbatch: Path, job_file: Path, run_root: Path,
           run=subprocess.run) -> str:
    verify_checksums(plan_path.parent)
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    job_file = job_file.resolve()
    checksum = job_file.with_suffix(".sh.sha256")
    if not job_file.is_file() or not checksum.is_file() \
            or file_hash(job_file) != checksum.read_text().split()[0]:
        raise ValueError("missing or changed immutable job file")
    if not sbatch.is_file():
        raise ValueError(f"missing sbatch file: {sbatch}")
    command = ["sbatch", "--parsable",
               f"--export=ALL,QH_JOB_FILE={job_file},QH_CAMPAIGN_PLAN={plan_path.resolve()},QH_RUN_ROOT={run_root.resolve()}",
               str(sbatch)]
    result = run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip().split(";")[0]


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
        (
            row[k]
            for k in ("messages", "trajectory", "events", "conversation")
            if row.get(k)
        ),
        None,
    )
    value = json.loads(value) if isinstance(value, str) else value
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("trace messages must be a list")
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


def renderable(messages: list[dict]) -> list[dict]:
    out = []
    for message in messages:
        role = message["role"] if message["role"] in {"assistant", "user"} else "user"
        if message["role"] == "system" and not out:
            role = "system"
        if role == "assistant" and (not out or out[-1]["role"] == "system"):
            role = "user"
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n" + message["content"]
        else:
            out.append({"role": role, "content": message["content"]})
    return out


def normalize_traces(
    rows: list[dict], source: str, revision: str, count_tokens
) -> list[dict]:
    if source not in SOURCES:
        raise ValueError(f"unapproved trace source: {source}")
    license_id, content_basis = SOURCES[source]
    out = []
    for index, row in enumerate(rows):
        messages = _messages(row)
        if not messages:
            continue
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
            original_prefix, answer = messages[:stop], messages[stop]
            rendered = renderable(original_prefix)
            if not rendered:
                continue
            total = count_tokens(rendered)
            if total < 1 or total > 32768:
                continue
            timestamp = next(
                (
                    m.get("timestamp")
                    for m in reversed(original_prefix)
                    if m.get("timestamp") is not None
                ),
                None,
            )
            out.append(
                {
                    "schema": TRACE_SCHEMA,
                    "source": source,
                    "revision": revision,
                    "dataset_license": license_id,
                    "content_basis": content_basis,
                    "rendering_contract": "gpt-oss-shape-v1",
                    "session_id": f"{source}:{session_id}",
                    "turn": turn,
                    "time_s": _seconds(timestamp) if timestamp is not None else None,
                    "input_tokens_total": total,
                    "newly_append_tokens": max(1, total - previous),
                    "output_tokens": max(1, count_tokens(answer["content"])),
                    "current_user_message_count": sum(
                        m["role"] == "user" for m in original_prefix
                    ),
                    "tool_message_count": sum(
                        m["role"] == "tool" for m in original_prefix
                    ),
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


def wildchat_coding(row: dict) -> bool:
    if row.get("toxic") or str(row.get("language", "")).lower() != "english":
        return False
    users = [m["content"] for m in _messages(row) if m["role"] == "user"]
    text = "\n".join(users).lower()
    signals = (
        r"\b(python|javascript|typescript|rust|java|c\+\+|sql|react|node\.js)\b",
        r"\b(function|method|compiler|exception|traceback|stack trace|unit test|api|database|regex)\b",
        r"(^|\n)\s*(def |class |import |select |insert |const |let |fn )",
    )
    return len(users) >= 2 and (
        "```" in text or sum(bool(re.search(x, text)) for x in signals) >= 2
    )


def nvidia_agentic(row: dict) -> bool:
    allowed = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"}
    roles = {m["role"] for m in _messages(row)}
    return row.get("license") in allowed and {"assistant", "tool"} <= roles


def classify(rows: list[dict]) -> list[dict]:
    grouped = _sessions(rows)
    classes = {
        session_id: "agentic_tool_loop"
        if session_id.startswith("nvidia/")
        else "interactive_coding"
        if session_id.startswith("allenai/")
        else "coding"
        for session_id in grouped
    }
    return [dict(row, job_class=classes[row["session_id"]]) for row in rows]


def build_manifests(rows: list[dict], seed: int = 0) -> dict:
    rows = classify(rows)
    grouped = _sessions(rows)
    manifests = {}
    for job_class in JOB_CLASSES:
        sessions = [
            (sid, sorted(turns, key=lambda r: r["turn"]))
            for sid, turns in grouped.items()
            if turns[0]["job_class"] == job_class
            and any(not r.get("reset") and 256 <= int(r["input_tokens_total"]) <= 24576
                    for r in turns)
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
        "schema": MANIFEST_SCHEMA,
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
        if isinstance(value, list):
            payload["chat_template_kwargs"] = {
                "reasoning_effort": "low",
                "enable_thinking": True,
            }
        result = testbed.http_json(host, port, "POST", "/tokenize", payload)
        if result.get("count") is None:
            raise RuntimeError("vLLM tokenizer returned no count")
        return int(result["count"])

    return count


def tokenizer_counter(tokenizer):
    def count(value) -> int:
        if not isinstance(value, list):
            return len(tokenizer(value, add_special_tokens=False)["input_ids"])
        encoded = tokenizer.apply_chat_template(
            value,
            tokenize=True,
            add_generation_prompt=True,
            reasoning_effort="low",
            enable_thinking=True,
        )
        return len(
            encoded if isinstance(encoded, (list, tuple)) else encoded["input_ids"]
        )

    return count


def local_token_counter(model: str, revision: str):
    from transformers import AutoTokenizer

    return tokenizer_counter(AutoTokenizer.from_pretrained(model, revision=revision))


def cached_normalize(path: Path, key: str, make) -> list[dict]:
    if path.is_file():
        cached = json.loads(path.read_text())
        if cached.get("key") == key:
            return cached["traces"]
    traces = make()
    write_json(path, {"key": key, "traces": traces})
    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Destination measurement campaign")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-evidence")
    audit.add_argument("--out", type=Path, required=True)
    plan = sub.add_parser("prepare")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--out-dir", type=Path, required=True)
    fetch = sub.add_parser("fetch-traces")
    fetch.add_argument("--out-dir", type=Path, required=True)
    fetch.add_argument("--trace-config")
    fetch.add_argument("--trace-split")
    fetch.add_argument("--wildchat-config")
    fetch.add_argument("--wildchat-split")
    fetch.add_argument("--nvidia-config")
    fetch.add_argument("--nvidia-split")
    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--plan", type=Path, required=True)
    submit_parser.add_argument(
        "--sbatch",
        type=Path,
        default=Path(__file__).with_name("destination_campaign.sbatch"),
    )
    submit_parser.add_argument("--job-file", type=Path, required=True)
    submit_parser.add_argument("--run-root", type=Path, required=True)
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
    reserve = sub.add_parser("prepare-reserve")
    reserve.add_argument("--report", type=Path, required=True)
    reserve.add_argument("--bundle", type=Path, required=True)
    reserve.add_argument("--out-dir", type=Path, required=True)
    accept = sub.add_parser("acceptance")
    accept.add_argument("--service", type=Path, required=True)
    accept.add_argument("--loaded", type=Path, required=True)
    accept.add_argument("--out", type=Path, required=True)
    build = sub.add_parser("build-manifests")
    build.add_argument("--trace-commons", type=Path, required=True)
    build.add_argument("--wildchat", type=Path, required=True)
    build.add_argument("--nvidia", type=Path, required=True)
    build.add_argument("--trace-revision", required=True)
    build.add_argument("--wildchat-revision", required=True)
    build.add_argument("--nvidia-revision", required=True)
    build.add_argument("--host", default="127.0.0.1")
    build.add_argument("--port", type=int, default=8000)
    build.add_argument("--model", default=testbed.MODEL)
    build.add_argument("--local-tokenizer-revision")
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit-evidence":
        write_json(args.out, audit_evidence())
        return
    if args.command == "prepare":
        prepare(args.manifest, args.out_dir)
        return
    if args.command == "fetch-traces":
        fetch_dataset(
            "trace-commons/agent-traces",
            args.out_dir / "trace-commons.jsonl",
            args.trace_config,
            args.trace_split,
        )
        fetch_parquet_sample(
            "allenai/WildChat-1M",
            args.out_dir / "wildchat-coding.jsonl",
            wildchat_coding,
            48,
            args.wildchat_config,
            args.wildchat_split,
        )
        fetch_dataset(
            "nvidia/SWE-Hero-openhands-trajectories",
            args.out_dir / "nvidia-swe-hero.jsonl",
            args.nvidia_config,
            args.nvidia_split,
            nvidia_agentic,
            48,
        )
        return
    if args.command == "submit":
        print(
            json.dumps(
                submit(args.plan, args.sbatch, args.job_file, args.run_root),
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
    if args.command == "prepare-reserve":
        prepare_reserve(args.report, args.bundle, args.out_dir)
        return
    if args.command == "acceptance":
        write_json(args.out, acceptance_report(_read_table(args.service), _read_table(args.loaded)))
        return
    counter = (
        local_token_counter(args.model, args.local_tokenizer_revision)
        if args.local_tokenizer_revision
        else token_counter(args.host, args.port, args.model)
    )

    def load(path):
        return [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]

    rows, source_counts = [], {}
    for path, source, revision in (
        (args.trace_commons, "trace-commons/agent-traces", args.trace_revision),
        (args.wildchat, "allenai/WildChat-1M", args.wildchat_revision),
        (args.nvidia, "nvidia/SWE-Hero-openhands-trajectories", args.nvidia_revision),
    ):
        raw = load(path)
        key = object_hash(
            [
                NORMALIZER_VERSION,
                file_hash(path),
                source,
                revision,
                args.model,
                args.local_tokenizer_revision or "vllm",
            ]
        )
        cache = args.out.with_name(f".{args.out.stem}-{source.split('/')[0]}.json")
        normalized = cached_normalize(
            cache, key, lambda: normalize_traces(raw, source, revision, counter)
        )
        rows += normalized
        source_counts[source] = {
            "raw_rows": len(raw),
            "sessions": len({r["session_id"] for r in normalized}),
            "turns": len(normalized),
        }
    write_json(
        args.out,
        {
            "manifest": build_manifests(rows, args.seed),
            "traces": rows,
            "source_counts": source_counts,
            "tokenizer": {
                "model": args.model,
                "revision": args.local_tokenizer_revision,
                "source": "transformers" if args.local_tokenizer_revision else "vllm",
            },
        },
    )


if __name__ == "__main__":
    main()
