"""Workload anchors (Workstreams W1, W2, W3).

Each anchor pins down one production-shape workload family chosen because
it stresses a different state layer:

    W1  large-repo coding         (workspace dominates)
    W2  data / RAG / artifact      (artifact bytes dominate)
    W3  multi-agent fanout/fanin   (private transcripts + shared context)

An anchor carries three artifacts, all derived from a single
`WorkloadAnchor` value:

    * `state_layer_breakdown` — per-A1/S1 layer bytes plus mobility class
      (`globally_available` / `cheaply_rehydratable` / `must_move` /
      `can_be_recomputed` / `can_be_discarded`).  This is the qualitative
      claim about *why* the anchor stresses one resource and not another.

    * an episode + manifests built via `build_episode(...)`. The episode
      and manifests are deterministic in `seed`, ride on the existing
      herd adapter, and are the thing K4 / K9 actually consume.

    * a regime hypothesis (`reuse` / `state_locality` / `landing_pressure`
      / `multi_resource`).  `classify_regime(anchor, ...)` runs the strong
      per-site reuse baseline and `mixed_min_pressure` through K4 and
      reports the *observed* regime — if the observed and hypothesized
      regimes disagree, the anchor was miscalibrated and the bytes /
      capacities should be revisited (not the verdict).

Anchors do **not** invent a new event class, simulator, or solver. They
are episode constructors plus a small bytes-by-layer table; everything
downstream is the existing K4/K8/K9 code path.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

from .adapters.herd import HerdSpec, build_herd_episode
from .episode import MobilityEpisode, Workflow
from .fluid_sim import (
    ALL_RESOURCES,
    NETWORK,
    PREFILL,
    WORKSPACE,
    KV_MEMORY,
    SimulationResult,
    simulate_fluid,
)
from .manifest import ServingGroupManifest, StateObject, WorkNode
from .profiles import ProfileBundle
from .reconstitution import cache_reuse, mixed_min_pressure
from .resources import ResourceBudget
from .warmness import WarmnessMap


# ---------------------------------------------------------------------------
# Layer / regime vocabulary (S1 taxonomy, hard-coded for grep)
# ---------------------------------------------------------------------------

MOBILITY_CLASSES: tuple[str, ...] = (
    "globally_available",
    "cheaply_rehydratable",
    "must_move",
    "can_be_recomputed",
    "can_be_discarded",
)

REGIMES: tuple[str, ...] = (
    "reuse",
    "state_locality",
    "landing_pressure",
    "multi_resource",
)


@dataclass(frozen=True)
class StateLayer:
    """One row of an anchor's S1-taxonomy breakdown.

    `bytes_per_workflow` is the *per-workflow* synthetic byte count this
    layer contributes to the episode (so a 100-workflow anchor multiplies
    by 100 across the herd; the herd-level total is reported by
    `WorkloadAnchor.total_bytes_per_layer`).
    """
    name: str
    bytes_per_workflow: int
    mobility_class: str
    notes: str = ""

    def __post_init__(self) -> None:
        if self.mobility_class not in MOBILITY_CLASSES:
            raise ValueError(
                f"unknown mobility_class {self.mobility_class!r}; "
                f"expected one of {MOBILITY_CLASSES}"
            )
        if self.bytes_per_workflow < 0:
            raise ValueError(
                f"bytes_per_workflow must be >= 0; got {self.bytes_per_workflow}"
            )


# A builder takes (anchor, n_workflows, source_sites, destination_sites, seed)
# and returns a (MobilityEpisode, manifests) pair.
EpisodeBuilder = Callable[
    ["WorkloadAnchor", int, tuple[str, ...], tuple[str, ...], int],
    tuple[MobilityEpisode, dict[str, ServingGroupManifest]],
]


@dataclass(frozen=True)
class WorkloadAnchor:
    name: str
    description: str
    state_layers: tuple[StateLayer, ...]
    regime_hypothesis: str
    builder: EpisodeBuilder
    notes: str = ""

    def __post_init__(self) -> None:
        if self.regime_hypothesis not in REGIMES:
            raise ValueError(
                f"unknown regime_hypothesis {self.regime_hypothesis!r}; "
                f"expected one of {REGIMES}"
            )
        if not self.state_layers:
            raise ValueError(f"anchor {self.name!r} requires >=1 state layer")
        names = [layer.name for layer in self.state_layers]
        if len(names) != len(set(names)):
            raise ValueError(
                f"anchor {self.name!r} has duplicate layer names: {names}"
            )

    # ---- byte accounting ---------------------------------------------------

    def must_move_bytes_per_workflow(self) -> int:
        """Sum of `must_move` layers — the lower bound on cross-site bytes."""
        return sum(
            layer.bytes_per_workflow
            for layer in self.state_layers
            if layer.mobility_class == "must_move"
        )

    def total_bytes_per_workflow(self) -> int:
        return sum(layer.bytes_per_workflow for layer in self.state_layers)

    def total_bytes_per_layer(self, n_workflows: int) -> dict[str, int]:
        if n_workflows < 1:
            raise ValueError(f"n_workflows must be >= 1; got {n_workflows}")
        return {
            layer.name: layer.bytes_per_workflow * n_workflows
            for layer in self.state_layers
        }

    def build_episode(
        self,
        n_workflows: int,
        *,
        source_sites: tuple[str, ...] = ("phoenix",),
        destination_sites: tuple[str, ...] = ("seattle", "austin"),
        seed: int = 0,
    ) -> tuple[MobilityEpisode, dict[str, ServingGroupManifest]]:
        return self.builder(self, n_workflows, source_sites, destination_sites, seed)


# ---------------------------------------------------------------------------
# W1 — Large-repo coding fixture
# ---------------------------------------------------------------------------
#
# Modeled after a `pip install -e .`-ed scientific-Python repo
# (pandas/numpy/scikit-learn class) with a populated dependency cache, a
# warmed pytest cache, and a few touched files mid-iteration. The
# load-bearing claim: `must_move` bytes (uncommitted diff + tool output)
# are tiny relative to `cheaply_rehydratable` workspace state, but the
# workspace must arrive at the destination *somehow* before the agent can
# compile/import — so the regime is "state_locality": the strong reuse
# baseline already reuses the workspace once it exists at a site, and the
# question is whether mobility planning beats that.

_W1_LAYERS: tuple[StateLayer, ...] = (
    StateLayer(
        name="base_repo_checkout",
        bytes_per_workflow=350_000_000,    # ~350 MB scipy-class working tree
        mobility_class="globally_available",
        notes="upstream HEAD; rehydratable from origin at any site",
    ),
    StateLayer(
        name="dependency_cache",
        bytes_per_workflow=600_000_000,    # ~600 MB .venv with scipy/numpy/pandas
        mobility_class="cheaply_rehydratable",
        notes="rebuilt from pyproject.toml / requirements.txt",
    ),
    StateLayer(
        name="build_artifacts",
        bytes_per_workflow=80_000_000,     # __pycache__ + compiled extensions
        mobility_class="can_be_recomputed",
        notes="pytest sets these up; recomputation is fast at warm sites",
    ),
    StateLayer(
        name="uncommitted_diff",
        bytes_per_workflow=200_000,        # ~200 KB hand-edited patch
        mobility_class="must_move",
        notes="agent's in-progress edits; lost if not transported",
    ),
    StateLayer(
        name="tool_output_context",
        bytes_per_workflow=300_000,        # ~300 KB stdout/stderr in prompt
        mobility_class="must_move",
        notes="moves with KV/replay; counted as prompt tokens too",
    ),
    StateLayer(
        name="test_logs",
        bytes_per_workflow=20_000_000,     # ~20 MB pytest -v output
        mobility_class="can_be_discarded",
        notes="agent rarely re-reads; discardable on migration",
    ),
)


def _build_w1_episode(
    anchor: WorkloadAnchor,
    n_workflows: int,
    source_sites: tuple[str, ...],
    destination_sites: tuple[str, ...],
    seed: int,
) -> tuple[MobilityEpisode, dict[str, ServingGroupManifest]]:
    # Same post-hoc state.bytes rewrite pattern that k8_regime.make_k8_episode
    # uses. The herd adapter's manifests are constructed fresh per call, so
    # mutating them here is local — no shared/cached state is touched.
    spec = HerdSpec(
        n_workflows=n_workflows,
        workspace_bytes_distribution="tiny",
        prompt_tokens_distribution="medium",
        warm_cache_fraction=0.0,
        home_asymmetry="all_same",
        seed=seed,
    )
    episode, manifests = build_herd_episode(
        spec,
        source_sites=source_sites,
        destination_sites=destination_sites,
        episode_id=f"w1_largerepo_n{n_workflows}_s{seed}",
    )
    # Workspace bytes = layers that the workspace materially carries:
    # base repo + dep cache + build artifacts + uncommitted diff. Test logs
    # are `can_be_discarded` and excluded; tool_output is prompt-context.
    workspace_layers = {
        "base_repo_checkout",
        "dependency_cache",
        "build_artifacts",
        "uncommitted_diff",
    }
    workspace_bytes = sum(
        layer.bytes_per_workflow
        for layer in anchor.state_layers
        if layer.name in workspace_layers
    )
    # tool_output_context is `must_move` but rides with KV/replay, not the
    # workspace. We add it to the per-workflow prompt-context tokens so
    # the byte accounting actually flows into the episode rather than
    # being a paper-only entry. ~4 bytes/token English-prose conversion.
    tool_output_layer = next(
        (l for l in anchor.state_layers if l.name == "tool_output_context"),
        None,
    )
    extra_tokens = (
        _bytes_to_tokens(tool_output_layer.bytes_per_workflow)
        if tool_output_layer is not None else 0
    )
    for manifest in manifests.values():
        for state in manifest.state_objects.values():
            if state.layer == "workspace":
                state.bytes = workspace_bytes
                state.content_hash = (
                    f"{state.content_hash}:w1_largerepo:{workspace_bytes}"
                )
            elif (
                extra_tokens
                and state.layer == "prompt_context"
                and state.lifetime != "persistent"
                and state.state_id != "system_prompt"
            ):
                # Per-workflow prompt-context ("issue_text_<wid>"): grow it
                # by the tool_output_context bytes converted to tokens.
                state.tokens = state.tokens + extra_tokens
                state.content_hash = f"{state.content_hash}:w1_tool_output:{extra_tokens}"
    return replace(episode, notes=f"W1 large-repo coding anchor n={n_workflows}"), manifests


W1_LARGE_REPO_CODING = WorkloadAnchor(
    name="w1_large_repo_coding",
    description=(
        "Scientific-Python class repo with installed dependency cache + warmed "
        "pytest cache. Per-workflow workspace ~1 GB (base repo 350 MB + "
        "dep cache 600 MB + build artifacts 80 MB + uncommitted diff 0.2 MB)."
    ),
    state_layers=_W1_LAYERS,
    regime_hypothesis="state_locality",
    builder=_build_w1_episode,
    notes=(
        "Hypothesis: workspace bytes >> must_move bytes, but the workspace "
        "still needs to land somewhere. Strong per-site reuse should win once "
        "the destination is warm; the open question is how the FIRST workflow "
        "to land pays the cost."
    ),
)


# ---------------------------------------------------------------------------
# W2 — Data / RAG / artifact-heavy fixture
# ---------------------------------------------------------------------------
#
# Modeled after an analyst-style agent over a parquet/CSV bundle plus a
# retrieved-document corpus. Workspace is small; what dominates is the
# retrieved-document bundle (RAG) and the produced artifacts (cleaned
# intermediates, plots). Both are `must_move` if the agent's next step
# needs them — RAG retrievals are not globally cached at every site.

_W2_LAYERS: tuple[StateLayer, ...] = (
    StateLayer(
        name="base_data_bundle",
        bytes_per_workflow=500_000_000,   # ~500 MB parquet/CSV bundle
        mobility_class="globally_available",
        notes="public dataset; replicated to all sites in advance",
    ),
    StateLayer(
        name="retrieved_documents",
        bytes_per_workflow=1_500_000_000,  # ~1.5 GB retrieved doc bundle
        mobility_class="must_move",
        notes="result of vector-search at source; not cached at destination",
    ),
    StateLayer(
        name="cleaned_intermediates",
        bytes_per_workflow=400_000_000,    # ~400 MB cleaned parquet
        mobility_class="must_move",
        notes="produced by earlier nodes; recomputation is expensive",
    ),
    StateLayer(
        name="generated_plots",
        bytes_per_workflow=15_000_000,     # ~15 MB PNGs
        mobility_class="must_move",
        notes="agent's next step references the exact PNG; ship it",
    ),
    StateLayer(
        name="prompt_summaries",
        bytes_per_workflow=400_000,        # ~400 KB prompt context
        mobility_class="must_move",
        notes="moves with KV/replay; counted as prompt tokens too",
    ),
    StateLayer(
        name="vector_index_shards",
        bytes_per_workflow=8_000_000_000,  # ~8 GB FAISS shard
        mobility_class="globally_available",
        notes="shared index; replicated to all sites in advance",
    ),
)


_W2_GLOBAL_LAYER_TO_STATE: dict[str, str] = {
    "base_data_bundle": "global_data_bundle",
    "vector_index_shards": "global_vector_index",
}


def _build_w2_episode(
    anchor: WorkloadAnchor,
    n_workflows: int,
    source_sites: tuple[str, ...],
    destination_sites: tuple[str, ...],
    seed: int,
) -> tuple[MobilityEpisode, dict[str, ServingGroupManifest]]:
    # W2's dominant cross-site bytes are `must_move`, not
    # `cheaply_rehydratable`. The workspace state carries the
    # cleaned_intermediates + retrieved_documents + generated_plots sum;
    # `prompt_summaries` rides as extra prompt-context tokens (parallel
    # to how W1 handles tool_output_context); the two `globally_available`
    # layers (base_data_bundle, vector_index_shards) become real state
    # objects with initial warmness at every destination so K4 actually
    # sees them as zero-cost (instead of being unfalsifiable paper rows).
    spec = HerdSpec(
        n_workflows=n_workflows,
        workspace_bytes_distribution="tiny",
        prompt_tokens_distribution="medium",
        warm_cache_fraction=0.0,
        home_asymmetry="all_same",
        seed=seed,
    )
    episode, manifests = build_herd_episode(
        spec,
        source_sites=source_sites,
        destination_sites=destination_sites,
        episode_id=f"w2_dataartifact_n{n_workflows}_s{seed}",
    )
    layers_by_name = {layer.name: layer for layer in anchor.state_layers}
    workspace_must_move = (
        layers_by_name["retrieved_documents"].bytes_per_workflow
        + layers_by_name["cleaned_intermediates"].bytes_per_workflow
        + layers_by_name["generated_plots"].bytes_per_workflow
    )
    extra_prompt_tokens = _bytes_to_tokens(
        layers_by_name["prompt_summaries"].bytes_per_workflow,
    )
    # Inject globally_available state objects as workspace-layer states
    # with warmness at every destination + every source. They have real
    # bytes, so a wrong policy that does NOT honor warmness would surface
    # them as huge cross-site transfers (= falsifiable claim).
    state_warmness: dict[str, tuple[str, ...]] = {
        sid: tuple(sorted(set(destination_sites) | set(source_sites)))
        for sid in _W2_GLOBAL_LAYER_TO_STATE.values()
    }
    src = source_sites[0]
    for manifest in manifests.values():
        for state in manifest.state_objects.values():
            if state.layer == "workspace":
                state.bytes = workspace_must_move
                state.content_hash = (
                    f"{state.content_hash}:w2_dataartifact:{workspace_must_move}"
                )
            elif (
                extra_prompt_tokens
                and state.layer == "prompt_context"
                and state.state_id != "system_prompt"
            ):
                state.tokens = state.tokens + extra_prompt_tokens
                state.content_hash = (
                    f"{state.content_hash}:w2_prompt_summaries:{extra_prompt_tokens}"
                )
        for layer_name, sid in _W2_GLOBAL_LAYER_TO_STATE.items():
            manifest.state_objects[sid] = StateObject(
                state_id=sid,
                content_hash=f"hash_{sid}",
                layer="workspace",
                lifetime="persistent",
                tokens=0,
                bytes=layers_by_name[layer_name].bytes_per_workflow,
                home_site=src,
            )
            for node in manifest.nodes.values():
                if sid not in node.required_state:
                    node.required_state.append(sid)
    return replace(
        episode,
        state_warmness=state_warmness,
        notes=f"W2 data/RAG anchor n={n_workflows}",
    ), manifests


W2_DATA_RAG_HEAVY = WorkloadAnchor(
    name="w2_data_rag_heavy",
    description=(
        "Analyst-style agent over a parquet/CSV bundle plus a "
        "retrieved-document corpus. Per-workflow must_move payload ~1.9 GB "
        "(retrieved docs 1.5 GB + cleaned intermediates 0.4 GB)."
    ),
    state_layers=_W2_LAYERS,
    regime_hypothesis="state_locality",
    builder=_build_w2_episode,
    notes=(
        "Hypothesis: must_move bytes dominate, so artifact movement is the "
        "binding cost. globally_available index/data is pre-warmed at every "
        "destination so the regime is purely 'do the per-workflow artifacts "
        "fit the link?'"
    ),
)


# ---------------------------------------------------------------------------
# W3 — Multi-agent fanout / fanin fixture
# ---------------------------------------------------------------------------
#
# Modeled after a planner that spawns K parallel subagents on a shared
# task context, each producing a private transcript that is then merged
# by a reviewer. The state layers split into one shared (the planner's
# task context) and many private (per-subagent transcripts). The shared
# layer becomes a benchmark for grouping pressure: `cache_reuse` will
# correctly reuse the shared context across colocated subagents, which
# is exactly the L1 vs L2 question.

_W3_LAYERS: tuple[StateLayer, ...] = (
    StateLayer(
        name="shared_task_context",
        bytes_per_workflow=2_500_000,      # ~2.5 MB / ~600K tokens
        mobility_class="must_move",
        notes="planner's task context; shared across subagents",
    ),
    StateLayer(
        name="private_subagent_transcript",
        bytes_per_workflow=1_200_000,      # ~1.2 MB / ~300K tokens — per subagent
        mobility_class="must_move",
        notes="per-subagent prompt + tool output trace",
    ),
    StateLayer(
        name="subagent_workspace",
        bytes_per_workflow=4_000_000,      # ~4 MB shallow scratch per subagent
        mobility_class="cheaply_rehydratable",
        notes="shallow scratch per subagent; rebuildable",
    ),
    StateLayer(
        name="merge_review_buffer",
        bytes_per_workflow=600_000,        # ~600 KB / ~150K tokens summary buffer
        mobility_class="must_move",
        notes="reviewer's merge-context buffer",
    ),
    StateLayer(
        name="dependency_cache",
        bytes_per_workflow=200_000_000,    # ~200 MB shared dep cache
        mobility_class="globally_available",
        notes="shared dep cache; replicated at every site",
    ),
)

_W3_SUBAGENTS_PER_WORKFLOW = 4
_W3_GLOBAL_LAYER_TO_STATE: dict[str, str] = {
    "dependency_cache": "global_dep_cache",
}


def _build_w3_episode(
    anchor: WorkloadAnchor,
    n_workflows: int,
    source_sites: tuple[str, ...],
    destination_sites: tuple[str, ...],
    seed: int,
) -> tuple[MobilityEpisode, dict[str, ServingGroupManifest]]:
    # W3 differs in *manifest shape*: each workflow has K=4 subagent
    # nodes hanging off a planner node, with one shared task-context
    # state read by all subagent nodes (this is what makes grouping
    # matter — L2 sees the shared edge, L1 sees per-node placement).
    workspace_bytes = next(
        layer.bytes_per_workflow
        for layer in anchor.state_layers
        if layer.name == "subagent_workspace"
    )
    shared_tokens = _bytes_to_tokens(
        next(
            layer.bytes_per_workflow
            for layer in anchor.state_layers
            if layer.name == "shared_task_context"
        )
    )
    private_tokens = _bytes_to_tokens(
        next(
            layer.bytes_per_workflow
            for layer in anchor.state_layers
            if layer.name == "private_subagent_transcript"
        )
    )

    layers_by_name = {l.name: l for l in anchor.state_layers}
    merge_tokens = _bytes_to_tokens(
        layers_by_name["merge_review_buffer"].bytes_per_workflow,
    )
    dep_cache_bytes = layers_by_name["dependency_cache"].bytes_per_workflow

    workflows: list[Workflow] = []
    manifests: dict[str, ServingGroupManifest] = {}
    src = source_sites[0]
    for i in range(n_workflows):
        wid = f"wf_{i:04d}"
        merge_sid = f"merge_{wid}"
        dep_sid = _W3_GLOBAL_LAYER_TO_STATE["dependency_cache"]
        states: dict[str, StateObject] = {
            "system_prompt": StateObject(
                state_id="system_prompt",
                content_hash="hash_system_prompt_v1",
                layer="prompt_context", lifetime="persistent",
                tokens=512, bytes=None, home_site=None,
            ),
            f"shared_task_{wid}": StateObject(
                state_id=f"shared_task_{wid}",
                content_hash=f"hash_shared_{wid}",
                layer="prompt_context", lifetime="shared",
                tokens=shared_tokens, bytes=None, home_site=src,
            ),
            merge_sid: StateObject(
                state_id=merge_sid,
                content_hash=f"hash_merge_{wid}",
                layer="prompt_context", lifetime="shared",
                tokens=merge_tokens, bytes=None, home_site=src,
            ),
            # globally_available dep cache as a real workspace state; warmed
            # at every site so a wrong policy that ignores warmness would
            # surface this as a dep_cache_bytes-sized cross-site transfer.
            dep_sid: StateObject(
                state_id=dep_sid,
                content_hash="hash_global_dep_cache",
                layer="workspace", lifetime="persistent",
                tokens=0, bytes=dep_cache_bytes, home_site=src,
            ),
        }
        nodes: dict[str, WorkNode] = {
            f"planner_{wid}": WorkNode(
                node_id=f"planner_{wid}", node_type="llm_call",
                parent_node_id=None, workflow_id=wid, label="planner",
                status="complete",
                required_state=["system_prompt", f"shared_task_{wid}", dep_sid],
                produced_state=[],
                session_id=wid,
            ),
        }
        # K subagent nodes, each with its own private transcript +
        # workspace, but all reading the same shared_task_<wid>.
        for k in range(_W3_SUBAGENTS_PER_WORKFLOW):
            sid_priv = f"private_{wid}_s{k}"
            sid_ws = f"workspace_{wid}_s{k}"
            states[sid_priv] = StateObject(
                state_id=sid_priv,
                content_hash=f"hash_priv_{wid}_s{k}",
                layer="prompt_context", lifetime="private",
                tokens=private_tokens, bytes=None, home_site=src,
            )
            states[sid_ws] = StateObject(
                state_id=sid_ws,
                content_hash=f"hash_ws_{wid}_s{k}",
                layer="workspace", lifetime="private",
                tokens=0, bytes=workspace_bytes, home_site=src,
            )
            nodes[f"subagent_{wid}_s{k}"] = WorkNode(
                node_id=f"subagent_{wid}_s{k}",
                node_type="subagent",
                parent_node_id=f"planner_{wid}",
                workflow_id=wid, label=f"subagent_{k}",
                status="complete",
                required_state=[
                    "system_prompt", f"shared_task_{wid}", dep_sid,
                    sid_priv, sid_ws,
                ],
                produced_state=[],
                session_id=f"{wid}_s{k}",
            )
        # Reviewer node closes the fanin: reads merge buffer, shared task,
        # and every subagent's private transcript.
        nodes[f"reviewer_{wid}"] = WorkNode(
            node_id=f"reviewer_{wid}", node_type="llm_call",
            parent_node_id=f"planner_{wid}",
            workflow_id=wid, label="reviewer", status="complete",
            required_state=[
                "system_prompt", f"shared_task_{wid}", merge_sid,
                *(f"private_{wid}_s{k}" for k in range(_W3_SUBAGENTS_PER_WORKFLOW)),
            ],
            produced_state=[], session_id=f"{wid}_review",
        )
        manifests[wid] = ServingGroupManifest(
            workflow_id=wid, root_task=f"w3 fanout workflow {wid}",
            nodes=nodes, state_objects=states, edges=[],
        )
        workflows.append(Workflow(
            workflow_id=wid, manifest_path=f"<inline:w3:{wid}>",
            src_site=src, deadline_s=None,
        ))

    warm_sites = tuple(sorted(set(destination_sites) | set(source_sites)))
    state_warmness: dict[str, tuple[str, ...]] = {
        # globally_available dep cache: warm at every site (real bytes,
        # falsifiable claim — a policy ignoring warmness would surface
        # dep_cache_bytes worth of cross-site transfer).
        _W3_GLOBAL_LAYER_TO_STATE["dependency_cache"]: warm_sites,
    }
    episode = MobilityEpisode(
        episode_id=f"w3_fanout_n{n_workflows}_s{seed}",
        source_sites=source_sites,
        destination_sites=destination_sites,
        workflows=tuple(workflows),
        state_warmness=state_warmness,
        capacities=None,
        trigger_t_s=0.0,
        notes=(
            f"W3 multi-agent fanout anchor n={n_workflows}, "
            f"subagents_per_workflow={_W3_SUBAGENTS_PER_WORKFLOW}"
        ),
    )
    return episode, manifests


W3_MULTI_AGENT_FANOUT = WorkloadAnchor(
    name="w3_multi_agent_fanout",
    description=(
        f"Planner + {_W3_SUBAGENTS_PER_WORKFLOW} subagents + reviewer per "
        "workflow with a shared task context and a fanin merge step. "
        "Per-workflow must_move ~7 MB (shared 2.5 MB + "
        f"{_W3_SUBAGENTS_PER_WORKFLOW}×private 1.2 MB + merge buffer 0.6 MB). "
        "Workspace ~16 MB (4 MB shallow scratch × K subagents)."
    ),
    state_layers=_W3_LAYERS,
    regime_hypothesis="landing_pressure",
    builder=_build_w3_episode,
    notes=(
        "Hypothesis: N workflows × (planner + K subagents + reviewer) "
        "= N × (K+2) llm_calls competing for prefill at each destination "
        "is the binding cost. Cache_reuse correctly collapses each shared "
        "state to a single per-(workflow, dst) materialization — a "
        "structural test asserts that — but the AGGREGATE prefill demand "
        "across N workflows still dominates under any non-trivial cell. "
        "If a cell instead lands on `reuse`, the herd is small enough that "
        "the per-site cache cost is negligible and richer planning has no "
        "headroom; if it lands on `state_locality`, the workspace bytes "
        "drift indicates miscalibration."
    ),
)


# ---------------------------------------------------------------------------
# Registry + regime classification
# ---------------------------------------------------------------------------


ANCHORS: dict[str, WorkloadAnchor] = {
    W1_LARGE_REPO_CODING.name: W1_LARGE_REPO_CODING,
    W2_DATA_RAG_HEAVY.name: W2_DATA_RAG_HEAVY,
    W3_MULTI_AGENT_FANOUT.name: W3_MULTI_AGENT_FANOUT,
}


@dataclass(frozen=True)
class RegimeClassification:
    """Observed-regime read-out for one (anchor, capacity-cell) pair."""
    anchor_name: str
    n_workflows: int
    strong_reuse_p50_resume_s: float
    mixed_p50_resume_s: float
    dominant_bottleneck: str
    observed_regime: str
    matches_hypothesis: bool

    @property
    def mixed_vs_strong_gap_frac(self) -> float:
        if self.strong_reuse_p50_resume_s <= 0:
            return 0.0
        return (
            self.strong_reuse_p50_resume_s - self.mixed_p50_resume_s
        ) / self.strong_reuse_p50_resume_s


def classify_regime(
    anchor: WorkloadAnchor,
    bundle: ProfileBundle,
    budget: ResourceBudget,
    *,
    n_workflows: int = 8,
    source_sites: tuple[str, ...] = ("phoenix",),
    destination_sites: tuple[str, ...] = ("seattle", "austin"),
    seed: int = 0,
    significant_gap_threshold: float = 0.10,
) -> RegimeClassification:
    """Classify the observed regime by simulating the anchor.

    The dominant bottleneck and the mixed-vs-strong gap are read from K4
    output for the strong reuse baseline:

      * `state_locality`    bottleneck is workspace OR network — the
                            cross-site/artifact bytes themselves are the
                            binding cost (matches TASKS.md regime map's
                            "large workspace/artifact/state transfer").
      * `landing_pressure`  bottleneck is prefill — many concurrent
                            workflows saturate prefill capacity at the
                            destination.
      * `multi_resource`    no single bottleneck dominates AND mixed
                            beats strong by >= `significant_gap_threshold`
                            (mixed's resource balancing buys real time).
      * `reuse`             no significant gap and no binding resource —
                            per-site reuse already handles it.
    """
    episode, manifests = anchor.build_episode(
        n_workflows,
        source_sites=source_sites,
        destination_sites=destination_sites,
        seed=seed,
    )
    warmness = WarmnessMap.from_episode_seed(episode.state_warmness)

    strong_plan = cache_reuse(episode, manifests, bundle, warmness, budget)
    mixed_plan = mixed_min_pressure(episode, manifests, bundle, warmness, budget)
    strong_result = simulate_fluid(
        episode, manifests, strong_plan, bundle, warmness, budget,
    )
    mixed_result = simulate_fluid(
        episode, manifests, mixed_plan, bundle, warmness, budget,
    )

    bottleneck = _dominant_bottleneck(strong_result)
    gap = _gap_frac(strong_result.p50_resume_s(), mixed_result.p50_resume_s())
    # Under infinite capacity K4 still attributes a "bottleneck" axis to
    # every action — it just picks the first axis in the needed list when
    # all shares are infinite. So the bottleneck label only carries
    # information when there is observable contention; without it, the
    # strong-reuse plan ran at its wallclock lower bound on every action
    # and the regime is necessarily `reuse`.
    contended = _has_contention(strong_result)
    multi = _is_multi_resource(strong_result)
    if not contended and gap < significant_gap_threshold:
        observed = "reuse"
    elif multi and gap >= significant_gap_threshold:
        # Two or more resource axes carry substantial elapsed time AND
        # mixed buys real time over strong: that's multi_resource. Tested
        # before single-bottleneck branches so a workspace+network
        # co-saturation is not silently labeled state_locality.
        observed = "multi_resource"
    elif bottleneck == PREFILL:
        observed = "landing_pressure"
    elif bottleneck in (WORKSPACE, NETWORK):
        observed = "state_locality"
    elif gap >= significant_gap_threshold:
        observed = "multi_resource"
    else:
        observed = "reuse"

    return RegimeClassification(
        anchor_name=anchor.name,
        n_workflows=n_workflows,
        strong_reuse_p50_resume_s=strong_result.p50_resume_s(),
        mixed_p50_resume_s=mixed_result.p50_resume_s(),
        dominant_bottleneck=bottleneck,
        observed_regime=observed,
        matches_hypothesis=(observed == anchor.regime_hypothesis),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bytes_to_tokens(byte_count: int) -> int:
    """Rough bytes→tokens conversion at ~4 bytes/token (English prose).
    Used only for synthetic prompt-context sizing in W3.
    """
    if byte_count <= 0:
        return 0
    return max(1, byte_count // 4)


def _dominant_bottleneck(result: SimulationResult) -> str:
    """Time-weighted dominant bottleneck.

    Each action contributes its elapsed seconds to its bottleneck's bucket.
    A naive action-count tally would crown prefill whenever there are many
    small replays even if a single workspace transfer ate most of the
    wall clock — wrong-aggregation-level mistake. Time-weighting matches
    the regime intuition (where did the simulation actually spend time?).
    """
    weights: dict[str, float] = {}
    for action in result.actions:
        if action.bottleneck == "none":
            continue
        elapsed = max(action.finished_s - action.started_s, 0.0)
        weights[action.bottleneck] = weights.get(action.bottleneck, 0.0) + elapsed
    if not weights:
        return "none"
    return max(weights.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _gap_frac(strong_s: float, mixed_s: float) -> float:
    if strong_s <= 0:
        return 0.0
    return (strong_s - mixed_s) / strong_s


def _is_multi_resource(result: SimulationResult, share_threshold: float = 0.25) -> bool:
    """True if two or more bottleneck axes each carry at least
    `share_threshold` of the total elapsed-time mass. Distinguishes
    multi_resource regimes from single-resource bottlenecks."""
    weights: dict[str, float] = {}
    for action in result.actions:
        if action.bottleneck == "none":
            continue
        elapsed = max(action.finished_s - action.started_s, 0.0)
        weights[action.bottleneck] = weights.get(action.bottleneck, 0.0) + elapsed
    total = sum(weights.values())
    if total <= 0:
        return False
    above = [w for w in weights.values() if w / total >= share_threshold]
    return len(above) >= 2


def _has_contention(result: SimulationResult, tol: float = 0.05) -> bool:
    """True if any action's elapsed time exceeds its wallclock lower
    bound by more than `tol`. Under infinite capacity every action runs
    at its lower bound, so this returns False — the right signal that
    K4's per-action `bottleneck` label is uninformative."""
    for action in result.actions:
        elapsed = action.finished_s - action.started_s
        lb = action.wallclock_lower_bound_s
        if lb <= 0:
            continue
        if elapsed > lb * (1.0 + tol):
            return True
    return False
