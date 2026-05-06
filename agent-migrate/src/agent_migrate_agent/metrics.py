"""Metrics for agent_migrate plans and manifests.

Three MVP metrics:

- `cost_weighted_duplication_factor(plan)`:
      sum over rows of (cost_s * materialization_count)
    / sum over rows of (cost_s * 1)
  The numerator is the cost the policy actually paid (= `Plan.total_cost_s()`).
  The denominator is the lower_bound cost: one materialization per (state, site)
  pair in the plan's placement. For a policy that never duplicates within a
  component (D2), every row has count=1 and the factor is 1.0 exactly. For D1,
  rows with count>1 push the factor above 1.0.

- `repeated_prefix_fraction(manifest)`:
      sum_over_(state, consumer) [state.tokens if #consumers >= 2 else 0]
    / sum_over_(state, consumer) state.tokens
  Pure manifest_level metric; independent of policy.

- `state_layer_breakdown(plan)`:
  Dict mapping state_layer name to total cost paid for that layer across all
  materializations.

Reading from the audit CSV (E4) reproduces these metrics by construction; this
module is the canonical implementation, and the audit CSV is a sanity check.
"""
from __future__ import annotations

from collections import defaultdict

from .manifest import ServingGroupManifest
from .policies import Plan


def cost_weighted_duplication_factor(plan: Plan) -> float:
    if not plan.materializations:
        return 1.0
    paid = sum(m.cost_s * m.materialization_count for m in plan.materializations)
    ideal = sum(m.cost_s for m in plan.materializations)
    if ideal == 0:
        return 1.0
    return paid / ideal


def repeated_prefix_fraction(manifest: ServingGroupManifest) -> float:
    total = 0
    repeated = 0
    for state in manifest.state_objects.values():
        per_state = state.tokens * len(state.consumers)
        total += per_state
        if len(state.consumers) >= 2:
            repeated += per_state
    if total == 0:
        return 0.0
    return repeated / total


def state_layer_breakdown(plan: Plan, manifest: ServingGroupManifest) -> dict[str, float]:
    breakdown: dict[str, float] = defaultdict(float)
    for m in plan.materializations:
        layer = manifest.state_objects[m.state_id].layer
        breakdown[layer] += m.total_cost_s
    return dict(breakdown)


def non_trivial_shared_state_count(
    manifest: ServingGroupManifest,
    exclude: tuple[str, ...] = ("system_prompt", "issue_text"),
) -> int:
    """Count of state objects with >=2 consumers, excluding mandatory framework
    states (system_prompt, issue_text). On a SWE_agent F2 manifest this should
    be > 0; if it's 0, the trace has no non_trivial sharing structure beyond
    the trivial "everyone reads the system prompt" floor and downstream
    duplication_factor results will be uninformative."""
    return sum(
        1 for s in manifest.state_objects.values()
        if len(s.consumers) >= 2 and s.state_id not in exclude
    )
