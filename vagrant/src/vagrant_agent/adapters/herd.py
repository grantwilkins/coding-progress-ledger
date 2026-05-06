"""Herd episode adapter (Workstream K6).

Procedurally generates synthetic per-workflow manifests + a MobilityEpisode
JSON for K7's gauntlet fixtures. Each workflow has one prompt-context state
(system_prompt-like) + one workspace state (sized from a configurable
distribution) + one issue-text-like prompt-context state per workflow.

State-size distributions follow A1 audit's recommendation: target the
production regime, not the SWE-bench-pilot 33 MB shallow-clone regime.
Three distributions:

  tiny      median   30 MB workspace   (matches H5b real-bytes anchor)
  medium    median  500 MB workspace   (production agent with installed deps)
  monorepo  median    5 GB workspace   (large-codebase regime)

Per A2 audit: `home_asymmetry` parameter labels the scenario class.
  all_same: every workflow's src_site = first source site (single-source-evac)
  balanced: workflows split equally across source sites (distributed-origin)
  skewed:   80%/20% split (mixed)

Determinism: same (n_workflows, distribution, fraction, asymmetry, seed)
produces byte-identical output. Tests assert this.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path

from ..episode import MobilityEpisode, Workflow, dump_episode
from ..manifest import ServingGroupManifest, StateObject, WorkNode

WORKSPACE_DISTS: dict[str, tuple[int, int]] = {
    # (median_bytes, log2_sigma_bits) — log-normal-ish around the median.
    "tiny": (30_000_000, 2),         # ~30 MB ± factor of 4
    "medium": (500_000_000, 2),      # ~500 MB
    "monorepo": (5_000_000_000, 2),  # ~5 GB
}

PROMPT_TOKENS_DISTS: dict[str, tuple[int, int]] = {
    "small": (2_000, 1),     # ~2K tokens
    "medium": (10_000, 1),   # ~10K
    "large": (50_000, 1),    # ~50K
}


@dataclass(frozen=True)
class HerdSpec:
    n_workflows: int
    workspace_bytes_distribution: str = "tiny"  # "tiny"|"medium"|"monorepo"
    prompt_tokens_distribution: str = "medium"  # "small"|"medium"|"large"
    warm_cache_fraction: float = 0.0          # fraction of states warm at any dst
    home_asymmetry: str = "all_same"          # "all_same"|"balanced"|"skewed"
    seed: int = 0

    def __post_init__(self):
        if self.n_workflows < 1:
            raise ValueError(f"n_workflows must be >= 1; got {self.n_workflows}")
        if self.workspace_bytes_distribution not in WORKSPACE_DISTS:
            raise ValueError(f"unknown workspace dist {self.workspace_bytes_distribution!r}")
        if self.prompt_tokens_distribution not in PROMPT_TOKENS_DISTS:
            raise ValueError(f"unknown prompt dist {self.prompt_tokens_distribution!r}")
        if not 0.0 <= self.warm_cache_fraction <= 1.0:
            raise ValueError(f"warm_cache_fraction must be in [0, 1]; got {self.warm_cache_fraction}")
        if self.home_asymmetry not in ("all_same", "balanced", "skewed"):
            raise ValueError(f"unknown home_asymmetry {self.home_asymmetry!r}")


def build_herd_episode(
    spec: HerdSpec,
    *,
    source_sites: tuple[str, ...] = ("phoenix",),
    destination_sites: tuple[str, ...] = ("phoenix", "seattle"),
    episode_id: str | None = None,
) -> tuple[MobilityEpisode, dict[str, ServingGroupManifest]]:
    """Build a herd episode + per-workflow manifests in memory.

    Returns (episode, manifests) where manifests[workflow_id] is the
    ServingGroupManifest for that workflow. Does NOT write to disk —
    K7 fixtures call this and either dump the result for committed
    fixtures, or feed it directly to simulate_fluid for ephemeral runs.
    """
    rng = random.Random(spec.seed)
    workspace_median, ws_sigma = WORKSPACE_DISTS[spec.workspace_bytes_distribution]
    prompt_median, pr_sigma = PROMPT_TOKENS_DISTS[spec.prompt_tokens_distribution]

    workflows: list[Workflow] = []
    manifests: dict[str, ServingGroupManifest] = {}
    state_warmness: dict[str, list[str]] = {}

    # Shared system_prompt across all workflows (always warm at all sources).
    system_prompt_state_id = "system_prompt"

    # Pick warm states — flat fraction across all *cold* states once they
    # exist. We'll fill state_warmness AFTER manifests are built.

    for i in range(spec.n_workflows):
        wid = f"wf_{i:04d}"
        src = _assign_source_site(i, spec.n_workflows, spec.home_asymmetry, source_sites, rng)
        # Workspace state: log-normal-ish.
        ws_bytes = _lognormal_int(rng, workspace_median, ws_sigma)
        prompt_tokens = _lognormal_int(rng, prompt_median, pr_sigma)

        # Per-workflow states:
        #   system_prompt (shared across the herd; same state_id, same hash)
        #   issue_text_<wid>   (per-workflow prompt_context)
        #   workspace_<wid>    (per-workflow workspace, src-anchored)
        states = {
            "system_prompt": StateObject(
                state_id="system_prompt",
                content_hash="hash_system_prompt_v1",
                layer="prompt_context", lifetime="persistent",
                tokens=512, bytes=None, home_site=None,
            ),
            f"issue_text_{wid}": StateObject(
                state_id=f"issue_text_{wid}",
                content_hash=f"hash_issue_{wid}",
                layer="prompt_context", lifetime="shared",
                tokens=prompt_tokens, bytes=None, home_site=src,
            ),
            f"workspace_{wid}": StateObject(
                state_id=f"workspace_{wid}",
                content_hash=f"hash_ws_{wid}",
                layer="workspace", lifetime="private",
                tokens=0, bytes=ws_bytes, home_site=src,
            ),
        }
        # One node consuming all three states.
        node = WorkNode(
            node_id=f"n_{wid}", node_type="llm_call",
            parent_node_id=None, workflow_id=wid, label=None, status="complete",
            required_state=list(states), produced_state=[],
            session_id=wid,
        )
        manifests[wid] = ServingGroupManifest(
            workflow_id=wid, root_task=f"herd workflow {wid}",
            nodes={node.node_id: node}, state_objects=states, edges=[],
        )
        workflows.append(Workflow(
            workflow_id=wid, manifest_path=f"<inline:herd:{wid}>",
            src_site=src, deadline_s=None,
        ))

    # Warm-cache assignment: pick `warm_cache_fraction` of all (state, dst) pairs
    # at random and mark them warm.
    if spec.warm_cache_fraction > 0:
        all_pairs: list[tuple[str, str]] = []
        for manifest in manifests.values():
            for sid in manifest.state_objects:
                for dst in destination_sites:
                    all_pairs.append((sid, dst))
        # Sort for determinism, then sample.
        all_pairs.sort()
        n_warm = int(round(len(all_pairs) * spec.warm_cache_fraction))
        rng.shuffle(all_pairs)
        warm_pairs = all_pairs[:n_warm]
        for sid, dst in warm_pairs:
            state_warmness.setdefault(sid, []).append(dst)
    state_warmness_tuples = {sid: tuple(sorted(set(sites)))
                             for sid, sites in state_warmness.items()}

    if episode_id is None:
        # Deterministic episode_id from spec (so the JSON dump is byte-stable).
        h = hashlib.sha256(
            f"{spec.n_workflows}|{spec.workspace_bytes_distribution}|"
            f"{spec.prompt_tokens_distribution}|{spec.warm_cache_fraction}|"
            f"{spec.home_asymmetry}|{spec.seed}|{source_sites}|{destination_sites}"
            .encode()
        ).hexdigest()[:8]
        episode_id = f"herd_{spec.workspace_bytes_distribution}_n{spec.n_workflows}_{h}"

    episode = MobilityEpisode(
        episode_id=episode_id,
        source_sites=tuple(source_sites),
        destination_sites=tuple(destination_sites),
        workflows=tuple(workflows),
        state_warmness=state_warmness_tuples,
        capacities=None,
        trigger_t_s=0.0,
        notes=f"herd: {spec.n_workflows} workflows, workspace_dist={spec.workspace_bytes_distribution}, "
              f"warm={spec.warm_cache_fraction}, asymmetry={spec.home_asymmetry}, seed={spec.seed}",
    )
    return episode, manifests


def write_herd_fixture(
    spec: HerdSpec,
    out_path: str | Path,
    *,
    source_sites: tuple[str, ...] = ("phoenix",),
    destination_sites: tuple[str, ...] = ("phoenix", "seattle"),
    episode_id: str | None = None,
) -> tuple[MobilityEpisode, dict[str, ServingGroupManifest]]:
    """Build + dump a herd episode to JSON. Returns the same tuple as
    `build_herd_episode` so callers can use the in-memory manifests
    directly without re-loading."""
    ep, manifests = build_herd_episode(
        spec, source_sites=source_sites, destination_sites=destination_sites,
        episode_id=episode_id,
    )
    dump_episode(ep, out_path)
    return ep, manifests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lognormal_int(rng: random.Random, median: int, log2_sigma: float) -> int:
    """Sample a log-normal-ish positive int with a given median and rough
    spread. log2_sigma is the (geometric) standard deviation in log2-space:
    1 means ~factor-of-2 spread, 2 means ~factor-of-4."""
    # mu = ln(median); sigma = log2_sigma * ln(2)
    mu = math.log(median)
    sigma = log2_sigma * math.log(2.0)
    # rng.gauss is from random module; use it for log-normal sampling.
    val = math.exp(rng.gauss(mu, sigma))
    return max(1, int(val))


def _assign_source_site(
    workflow_idx: int, n_workflows: int, asymmetry: str,
    source_sites: tuple[str, ...], rng: random.Random,
) -> str:
    """Pick a source site for workflow `workflow_idx` per the asymmetry rule."""
    if not source_sites:
        raise ValueError("source_sites required")
    if asymmetry == "all_same" or len(source_sites) == 1:
        return source_sites[0]
    if asymmetry == "balanced":
        # Round-robin across source sites.
        return source_sites[workflow_idx % len(source_sites)]
    if asymmetry == "skewed":
        # 80% at first source, 20% distributed across the rest.
        if workflow_idx < int(0.8 * n_workflows):
            return source_sites[0]
        # Remaining 20% distributed round-robin across the rest.
        rest = source_sites[1:]
        if not rest:
            return source_sites[0]
        return rest[(workflow_idx - int(0.8 * n_workflows)) % len(rest)]
    raise ValueError(f"unknown asymmetry {asymmetry!r}")
