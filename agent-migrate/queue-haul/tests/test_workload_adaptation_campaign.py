"""
Claim:
The workload-adaptation campaign generates the complete binary resource
factorial, keeps each workload draw and target paired across all eight states,
and conserves every source session across Replay, KV, and not moved.

Plausible wrong implementations:
- Clone selected legacy cases instead of generating the 2^3 truth table.
- Resample turns independently from their conversation template or change a pack by case.
- Let HBM alter only KV actions instead of destination residency for both methods.
- Change the target watts with destination constraints.
- Drop sessions from the three-part action accounting.
- Split width-8 validation rows from the same session set across fit and holdout.
- Validate migration timing against the 30-second outer deadline instead of
  its 25-second usable budget after the power window.
- Call a target hit from power shortfall alone when execution feasibility failed.
- Ignore the requested regional timing fit and measured route rates.
- Make a state depend on whether a tighter counterfactual was solved first.
- Apply a constrained factor to only one destination or retain a prefill-only label.
- Drop a migration method from contexts covered by the regional timing model.
- Apply the measured load factor to route bytes, HBM, or the idle timing anchor.
- Serialize route and endpoint work after fitting an end-to-end pipeline rate.
- Use endpoint replica-seconds as the action objective and thereby hide a
  bandwidth-dependent change in isolated migration duration.
- Reuse the fitted end-to-end pipeline rate as the physical link budget, leaving
  Germany effectively unconstrained.
- Apply the bandwidth bottleneck to only one destination or confuse Mbit/s with
  bytes/s.
- Widen context support when extending only the measured bandwidth boundary.
- Label a single-factor state despite no material paired action response.
- Summarize a power-targeted action mix by session count instead of phase load.
- Lose phase-load conservation when assigning selected sessions to actions.
- Use phase-load shares rather than session shares in the companion boxplot.
- Include intermediate joint cases instead of the three independent bottlenecks,
  all bottlenecked, and none bottlenecked.
- Use default Tukey whiskers instead of the declared 5th and 95th percentiles.
- Reverse prefill occupancy into available throughput or report aggregate rather
  than per-destination throughput.
- Confuse Mbit/s with bytes/s or fail to recover the measured endpoint cases.
- Resample OAT levels independently, vary both axes at once, drop not-moved
  sessions, or weight feasibility by sessions instead of plans.
"""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

import workload_adaptation_campaign as campaign
from pool_planner import candidate_table
from power_model import ExpectedPower
from profiles import ModelProfile


def test_factorial_is_exact_and_uses_only_declared_levels():
    cases = campaign.factorial_cases()

    assert campaign.FACTORS == ("hbm", "bandwidth", "dest_compute")
    assert campaign.LABELS[frozenset(("dest_compute",))] == "Dest. compute"
    assert len(cases) == 8
    assert {constraints for _, _, constraints in cases} == {
        frozenset(factor for factor, enabled in zip(campaign.FACTORS, flags)
                  if enabled)
        for flags in __import__("itertools").product((False, True), repeat=3)
    }
    for _, _, constraints in cases:
        values = campaign.state_values(constraints)
        assert values == {factor: campaign.LEVELS[factor][factor in constraints]
                          for factor in campaign.FACTORS}


def test_template_sampling_is_reproducible_and_keeps_whole_shapes():
    profile = ModelProfile.load(campaign.PROFILE)
    templates, workload = campaign.load_templates(campaign.MANIFEST, profile)

    first = campaign.sample_pack(templates, 28, 7)
    second = campaign.sample_pack(templates, 28, 7)

    assert first == second and len({session.session_id for session in first}) == 28
    valid = {(shape.context_tokens, shape.prompt_tokens,
              shape.output_tokens, 2 * shape.context_tokens)
             for shapes in templates.values() for shape in shapes}
    assert all((session.context_tokens, session.expected_f, session.expected_g,
                session.log_bytes) in valid for session in first)
    phase = profile.case().phase_power
    assert workload["phase_direction_excluded_states"] == 2
    assert all(phase.contains(
        1e-3 * shape.prompt_tokens / max(shape.prompt_tokens, shape.output_tokens),
        1e-3 * shape.output_tokens / max(shape.prompt_tokens, shape.output_tokens),
    ) for shapes in templates.values() for shape in shapes)


def test_pack_normalization_preserves_shape_and_sets_common_source_load():
    profile = ModelProfile.load(campaign.PROFILE)
    templates, _ = campaign.load_templates(campaign.MANIFEST, profile)
    raw = campaign.sample_pack(templates, 28, 7)
    scaled = campaign.normalize_pack(profile, raw)
    case = profile.case()

    assert np.isclose(sum(session.expected_f / case.F + session.expected_g / case.G
                          for session in scaled), campaign.SOURCE_LOAD)
    assert np.ptp([scaled[i].expected_f / raw[i].expected_f
                   for i in range(28)]) < 1e-15


def test_one_paired_draw_conserves_sessions_and_target():
    rows, _ = campaign.simulate(samples=1, seed=3)

    assert rows == campaign.simulate(samples=1, seed=3)[0]
    assert len(rows) == 8
    assert np.ptp([row["target_w"] for row in rows]) < 1e-9
    assert len({row["timing_fit_sha256"] for row in rows}) == 1
    assert len({row["power_bootstrap_index"] for row in rows}) == 1
    assert all(sum(row[f"{action}_count"] for action in campaign.ACTIONS) == 28
               for row in rows)
    assert all(np.isclose(sum(row[action] for action in campaign.ACTIONS), 1)
               for row in rows)
    assert all(np.isclose(sum(row[f"{action}_phase_load"]
                              for action in campaign.ACTIONS), 1)
               for row in rows)
    assert all(row["target_met"] == (
        row["feasible"] and row["power_shortfall_w"] <= campaign.POWER_TOLERANCE_W
    ) for row in rows)
    checks = campaign.factor_checks(rows)
    assert len(checks) == 12
    bandwidth = next(row for row in checks
                     if row["case_id"] == "bandwidth"
                     and row["factor"] == "bandwidth")
    assert bandwidth["utilization"] == next(
        row["route_utilization"] for row in rows
        if row["case_id"] == "bandwidth"
    )
    none = next(row for row in rows if row["case_id"] == "none")
    constrained = next(row for row in rows if row["case_id"] == "bandwidth")
    # Constraining a resource cannot raise the relaxation's value; the two
    # agree to machine precision when the constraint is not binding.
    assert constrained["fractional_lp_opportunity_w"] \
        <= none["fractional_lp_opportunity_w"] * (1 + 1e-12)
    assert not any(row["fractional_opportunity_worsened_on_release"]
                   for row in checks)


def test_oat_axes_cover_only_effective_resource_ranges():
    profile = ModelProfile.load(campaign.PROFILE)
    templates, _ = campaign.load_templates(campaign.MANIFEST, profile)
    pack = campaign.normalize_pack(profile, campaign.sample_pack(templates, 4, 3))
    fits = campaign.central_timing_fits()
    bandwidths, prefills, fixed_bandwidth, fixed_prefill, prefill_max, \
        model_rate = \
        campaign.oat_design(profile, levels=50)
    natural_bandwidth = max(campaign.physical_route_mbps().values())
    observed = [float(row["tokens_per_s"]) for row in json.loads(
        campaign.PREFILL_ANCHORS.read_text())["anchors"]
        if row["metric"] == "prefill"]

    assert bandwidths == pytest.approx(np.linspace(
        campaign.OAT_BANDWIDTH_LOWER_MBPS, natural_bandwidth, 50))
    assert prefills[0] == pytest.approx(
        (1 - campaign.OAT_DEST_COMPUTE[1]) * prefill_max)
    assert fixed_bandwidth == pytest.approx(natural_bandwidth)
    assert fixed_prefill == pytest.approx(np.median(observed))
    assert prefill_max == pytest.approx(max(observed))
    assert model_rate == pytest.approx(5839.4091324275805)
    assert bandwidths[0] < campaign.BANDWIDTH_BOTTLENECK_MBPS
    assert bandwidths[-1] == pytest.approx(natural_bandwidth)
    assert fixed_prefill in prefills
    assert prefills[-1] == pytest.approx(prefill_max)
    assert min(np.diff(prefills)) > .9 * (prefills[-1] - prefills[0]) / 49

    for point in ((bandwidths[0], prefills[0]), (bandwidths[-1], prefills[-1])):
        scenario, architecture, *_ = campaign.build_problem(
            profile, pack, frozenset(), 2 / 3, fits,
            bandwidth_mbps=point[0], prefill_tps=point[1],
        )
        assert scenario.deadline_s == scenario.end_s \
            == campaign.SCORING_DEADLINE_S == 30
        assert scenario.controller_delay_s == 0
        assert campaign.migration_budget_s(profile) == 25
        assert campaign.candidate_table(
            scenario, profile, architecture, "normal",
            campaign.ExpectedPower(scenario, profile),
        ).migration_horizon_s == 25


def test_oat_pairs_seeded_openhands_packs_across_resource_levels():
    rows, packs, raw, distribution, design = campaign.simulate_oat(
        packs=2, levels=3, seed=3, sessions=4)

    assert len(rows) == 2 * 3 * len(campaign.ACTIONS)
    assert len(packs) == 2
    assert len(raw) == 2 * 2 * 3
    assert len(distribution) == 2 * 3 * 5
    assert design["paired_draws"] == design["packs"] == 2
    assert (campaign.OAT_PACKS, campaign.OAT_SESSIONS,
            campaign.OAT_LEVELS) == (1000, 8, 50)
    assert design["sessions_per_pack"] == 4
    assert design["target_fraction"] == 1
    assert design["pack_seed_range"] == [4, 5]
    assert (design["scoring_deadline_s"], design["power_window_s"],
            design["migration_budget_s"]) == (30, 5, 25)
    assert design["controller_delay_s"] == 0
    assert {(row["scoring_deadline_s"], row["power_window_s"],
             row["controller_delay_s"], row["migration_budget_s"])
            for row in raw} == {(30, 5, 0, 25)}
    assert design["prefill_observations"]["context_tokens"] \
        == [4096, 16384, 24576]
    assert design["prefill_observations"]["repeats_per_context"] == 3
    assert design["prefill_observations"]["median_reducer"] \
        == "pooled median across contexts and repeats"
    assert design["prefill_observations"]["max_reducer"] \
        == "raw maximum across contexts and repeats"
    for sweep in ("bandwidth", "prefill"):
        for level in range(3):
            selected = [row for row in rows
                        if row["sweep"] == sweep and row["level"] == level]
            assert sum(row["session_count"] for row in selected) == 8
            assert sum(row["session_share"] for row in selected) \
                == pytest.approx(1)
            assert {row["plans"] for row in selected} == {2}
            density = [row for row in distribution
                       if row["sweep"] == sweep and row["level"] == level]
            assert sum(row["pack_count"] for row in density) == 2
            assert sum(row["pack_share"] for row in density) == pytest.approx(1)
            metric = "kv_transfer_count" if sweep == "bandwidth" \
                else "migrated_count"
            observed = [row["kv_transfer_count"] if sweep == "bandwidth" else
                        row["replay_count"] + row["kv_transfer_count"]
                        for row in raw
                        if row["sweep"] == sweep and row["level"] == level]
            assert {row["outcome"]: row["pack_count"] for row in density} == {
                outcome: observed.count(outcome) for outcome in range(5)}
            assert {row["metric"] for row in density} == {metric}
    assert len({row["prefill_available_tps"] for row in rows
                if row["sweep"] == "bandwidth"}) == 1
    assert len({row["bandwidth_cap_gbps"] for row in rows
                if row["sweep"] == "prefill"}) == 1
    shared = design["shared_operating_point"]
    for pack_id in (1, 2):
        selected = [row for row in raw if row["pack_id"] == pack_id]
        pack = next(row for row in packs if row["pack_id"] == pack_id)
        ids = pack["template_ids"].split(";")
        assert len(ids) == len(set(ids)) == 4
        assert all("openhands" in template_id.lower() for template_id in ids)
        assert all(sum(row[f"{action}_count"] for action in campaign.ACTIONS) == 4
                   for row in selected)
        assert all(row["target_met"] == (row["replay_count"] +
                                         row["kv_transfer_count"] == 4)
                   for row in selected)
        assert all(row["planned_shed_w"] == pytest.approx(
            row["initial_source_power_w"] - row["planned_source_power_w"])
                   for row in selected)
        assert all(not row["target_met"] or
                   row["predicted_migration_makespan_s"] <= 25
                   for row in selected)
        bandwidth = next(row for row in selected
                         if row["sweep"] == "bandwidth"
                         and row["level"] == shared["bandwidth_level"])
        prefill = next(row for row in selected if row["sweep"] == "prefill"
                       and row["level"] == shared["prefill_level"])
        fields = ("target_met", "failure",
                  *(f"{action}_count" for action in campaign.ACTIONS))
        assert {key: bandwidth[key] for key in fields} == {
            key: prefill[key] for key in fields}
    assert {row["metric"] for row in distribution
            if row["sweep"] == "bandwidth"} == {"kv_transfer_count"}
    assert {row["metric"] for row in distribution
            if row["sweep"] == "prefill"} == {"migrated_count"}


def test_oat_publication_grid_avoids_highs_solve_error():
    profile = ModelProfile.load(campaign.PROFILE)
    templates, _ = campaign.load_templates(campaign.MANIFEST, profile)
    manifest = json.loads(campaign.MANIFEST.read_text())
    unscaled, _ = campaign.sample_family_pack(
        templates, manifest, campaign.OAT_FAMILY, 8,
        campaign.DEFAULT_SEED + 325)
    pack = campaign.normalize_pack(profile, unscaled)
    fits = campaign.central_timing_fits()
    timing_hash = hashlib.sha256(
        json.dumps(fits, sort_keys=True).encode()).hexdigest()
    _, prefills, bandwidth, _, _, _ = campaign.oat_design(profile)

    row = campaign.run_case(
        profile, pack, "oat_prefill", "OAT prefill", frozenset(), 325, 1,
        fits, None, timing_hash, 0, bandwidth_mbps=bandwidth,
        prefill_tps=prefills[8])

    assert row["failure"] == "target_unmet"


def test_oat_refuses_a_lower_target():
    with pytest.raises(ValueError, match="must remain 1.0"):
        campaign.simulate_oat(packs=1, levels=3, sessions=4,
                              target_fraction=2 / 3)


def test_phase_load_action_mix_uses_power_weights_not_session_counts():
    sessions = tuple(SimpleNamespace(session_id=value) for value in "abc")
    moves = (SimpleNamespace(session_id="a", method="replay"),
             SimpleNamespace(session_id="b", method="kv_transfer"))
    power = SimpleNamespace(ell={"a": 1, "b": 3, "c": 6})

    assert campaign.phase_load_shares(sessions, moves, power) == {
        "replay": .1, "kv_transfer": .3, "not_moved": .6,
    }


def test_action_plots_use_only_the_three_single_bottlenecks_and_endpoints():
    assert campaign.DISPLAY_CASES == (
        "hbm", "bandwidth", "dest_compute",
        "bandwidth-dest_compute-hbm", "none",
    )

    rows = []
    for case_id in campaign.DISPLAY_CASES:
        for replicate, (replay, kv) in enumerate(zip(
                (0, 10, 20, 30, 40), (0, 0, 10, 10, 20))):
            rows.append({
                "case_id": case_id, "replicate": replicate, "sessions": 100,
                "replay_count": replay, "kv_transfer_count": kv,
                "not_moved_count": 100 - replay - kv,
                "replay_phase_load": .99, "kv_transfer_phase_load": .005,
                "not_moved_phase_load": .005,
            })
    rows.append({
        **rows[0], "case_id": "hbm-bandwidth", "replay_count": 100,
        "not_moved_count": 0,
    })

    summary = campaign.action_boxplot_statistics(rows)
    replay = next(row for row in summary
                  if row["case_id"] == "hbm" and row["action"] == "replay")

    assert len(summary) == 5 * 3
    assert tuple(dict.fromkeys(row["case_id"] for row in summary)) \
        == campaign.DISPLAY_CASES
    assert (replay["p05"], replay["p25"], replay["median"], replay["p75"],
            replay["p95"]) == pytest.approx((2, 10, 20, 30, 38))


def test_loaded_factor_transport_is_counted_and_in_context_run_is_paired():
    rows, _ = campaign.simulate(samples=1, seed=3)
    robust, workload = campaign.simulate(
        samples=1, seed=3, loaded_context_only=True,
    )

    assert [(row["timing_fit_sha256"], row["power_bootstrap_index"])
            for row in rows] == [
                (row["timing_fit_sha256"], row["power_bootstrap_index"])
                for row in robust]
    assert workload["loaded_factor_context_only"]
    assert workload["loaded_factor_outside_context_states"] > 0
    assert all(row["loaded_context_below_session_count"] == 0
               and row["loaded_context_above_session_count"] == 0
               and row["loaded_context_below_candidate_count"] == 0
               and row["loaded_context_above_candidate_count"] == 0
               and row["loaded_context_below_selected_count"] == 0
               and row["loaded_context_above_selected_count"] == 0
               for row in robust)
    summary = campaign.transport_summary(robust)
    assert summary["context"]["sessions"]["denominator"] == 28
    assert not summary["context"]["sessions"]["outside"]
    assert not summary["context"]["candidates"]["outside"]
    assert not summary["context"]["selected"]["outside"]


def test_bandwidth_transport_counts_the_sub_gigabit_east_route():
    rows, _ = campaign.simulate(samples=1, seed=3)
    by_case = {row["case_id"]: row for row in rows}

    assert by_case["none"]["loaded_bandwidth_below_pool_count"] == 0
    assert by_case["bandwidth"]["loaded_bandwidth_below_pool_count"] == 1
    assert by_case["bandwidth"]["loaded_bandwidth_below_candidate_count"] == 56
    assert not any(row["loaded_bandwidth_above_pool_count"] for row in rows)
    summary = campaign.transport_summary(rows)
    assert summary["bandwidth"]["pools"]["denominator"] == 16
    assert summary["bandwidth"]["pools"]["outside"] == 4


def test_case_results_do_not_depend_on_factorial_traversal():
    profile = ModelProfile.load(campaign.PROFILE)
    templates, _ = campaign.load_templates(campaign.MANIFEST, profile)
    pack = campaign.normalize_pack(profile, campaign.sample_pack(templates, 28, 5))
    rng = np.random.default_rng(5)
    fits, projected = campaign.ordered_timing_fit(
        profile, campaign.json.loads(campaign.TIMING_PARENT.read_text()),
        campaign.read_csv(campaign.TIMING), rng,
    )
    power_index = int(rng.integers(len(profile.case().phase_power.bootstrap)))
    profile = campaign.state_profile(profile, {
        "power_bootstrap_index": power_index, "service_multiplier": 1,
        "replay_multiplier": 1, "kv_multiplier": 1,
    })

    def solve(cases):
        return {case_id: campaign.run_case(
            profile, pack, case_id, label, constraints, 5, 2 / 3, fits,
            power_index, "paired", projected,
        ) for case_id, label, constraints in cases}

    forward = solve(campaign.factorial_cases())
    reverse = solve(reversed(campaign.factorial_cases()))
    fields = (*[f"{action}_count" for action in campaign.ACTIONS],
              "power_shortfall_w", "fractional_lp_opportunity_w", "target_met")
    assert all(tuple(forward[case][field] for field in fields) ==
               tuple(reverse[case][field] for field in fields)
               for case in forward)


def test_opportunity_and_rounded_release_outcomes_are_distinct():
    rows = []
    for case_id, label, constraints in campaign.factorial_cases():
        rows.append({
            "replicate": 0, "case_id": case_id, "bound_constraint": label,
            "replay_count": 1, "kv_transfer_count": 1, "not_moved_count": 1,
            "power_shortfall_w": 1.0 if not constraints else 0.0,
            "target_met": bool(constraints),
            "fractional_lp_opportunity_w": 10 - len(constraints),
            "hbm_utilization": 0, "route_utilization": 0,
            "service_utilization": 0,
        })

    hbm = next(row for row in campaign.factor_checks(rows)
                if row["case_id"] == "hbm")

    assert hbm["fractional_lp_opportunity_change_w"] == 1
    assert not hbm["fractional_opportunity_worsened_on_release"]
    assert not hbm["resource_near_capacity"]
    assert hbm["opportunity_reduced"] and hbm["active"]
    assert hbm["planner_shortfall_worsened_on_release"]


def test_width8_validation_split_is_by_whole_session_set():
    scenarios = campaign.pd.read_csv(campaign.WIDTH8_TIMING / "scenarios.csv")
    selected = scenarios[(scenarios.kind == "migration")
                         & (scenarios.status == "complete")]
    split = {}
    for row in selected.itertuples():
        digest = int(campaign.hashlib.sha256(row.session_set.encode()).hexdigest(), 16)
        split.setdefault(row.session_set, set()).add(
            "holdout" if digest % 3 == 0 else "development"
        )

    assert len(split) == 18
    assert all(len(values) == 1 for values in split.values())


def test_surface_validation_uses_grouped_splits_and_migration_budget():
    rows = campaign.validate_surface()
    grouped = {}
    for row in rows:
        grouped.setdefault((row["source"], row["session_set"]), set()).add(
            row["split"]
        )

    assert all(len(splits) == 1 for splits in grouped.values())
    assert all(row["migration_budget_s"] == 25 for row in rows)
    assert all(row["false_feasible"] == (
        row["predicted_s"] <= 25 < row["measured_s"]
    ) for row in rows)


def test_surface_validation_overlaps_route_with_shared_endpoint_work():
    scenario = SimpleNamespace(
        migration_s=10, scenario_id="hand", session_set="hand",
        method="mixed", bandwidth_mbps=1000, concurrency=2,
    )

    route_bound = campaign._surface_row(
        "hand", "hand", scenario, replay_s=4, kv_s=3,
        route_s=10, coupling=1, migration_budget_s=25,
    )
    endpoint_bound = campaign._surface_row(
        "hand", "hand", scenario, replay_s=4, kv_s=3,
        route_s=5, coupling=1, migration_budget_s=25,
    )

    assert route_bound["predicted_s"] == 10
    assert endpoint_bound["predicted_s"] == 7


def test_surface_validation_rejects_width8_false_feasibility(monkeypatch):
    original = campaign._surface_row

    def unsafe(*args, **kwargs):
        row = original(*args, **kwargs)
        if row["source"] == "width8":
            row["false_feasible"] = True
        return row

    monkeypatch.setattr(campaign, "_surface_row", unsafe)
    with pytest.raises(RuntimeError, match="false-feasible on width-8"):
        campaign.validate_surface()


def test_surface_summary_exposes_method_specific_error():
    rows = campaign.validate_surface()
    summary = campaign.validation_summary(rows)

    assert "width8/development" in summary
    assert "width8/development/kv_transfer" in summary
    assert "width8/development/mixed" in summary
    assert "width8/development/replay" in summary


def test_central_surface_uses_regional_timing_and_routes():
    fits = campaign.central_timing_fits()
    profile = ModelProfile.load(campaign.PROFILE)
    templates, _ = campaign.load_templates(campaign.MANIFEST, profile)
    scenario, architecture, routes, _ = campaign.build_problem(
        profile, campaign.normalize_pack(
            profile, campaign.sample_pack(templates, 28, 4)),
        frozenset(), 2 / 3, fits,
    )
    services = {pool.pool_id: pool.fluid_migration
                for pool in architecture.pools}

    links = {link.link_id: link.bytes_per_s for link in scenario.links}
    physical = campaign.physical_route_mbps()
    assert links == {
        **{f"link/{region}": physical[region] * 125_000
           for region in campaign.REGIONS},
        **{f"pipeline/{region}":
           fits[region]["effective_pipeline_mbps"]["natural"] * 125_000
           for region in campaign.REGIONS},
    }
    assert routes == {("source", region):
                      (f"link/{region}", f"pipeline/{region}")
                      for region in campaign.REGIONS}
    assert all(service.route_overlap for service in services.values())
    assert services["pool/east"].replay_speedup == \
        1 / fits["east"]["replay_compute_completion_factor"]
    summary = campaign.json.loads(campaign.TIMING_SUMMARY.read_text())
    assert summary["migration_gate_passed"] and not summary["kv_byte_mismatches"]
    assert all(
        row["coverage"] >= .9 and row["median_relative_error"] <= .1
        and (row["p90_relative_error"] <= .15
             or row["p90_absolute_error_s"] <= 1)
        for row in summary["held_out"].values()
    )


def test_bandwidth_state_caps_both_physical_routes_at_measured_floor():
    fits = campaign.central_timing_fits()
    profile = ModelProfile.load(campaign.PROFILE)
    templates, _ = campaign.load_templates(campaign.MANIFEST, profile)
    pack = campaign.normalize_pack(
        profile, campaign.sample_pack(templates, 28, 4),
    )
    scenario, architecture, routes, _ = campaign.build_problem(
        profile, pack, frozenset(("bandwidth",)), 2 / 3, fits,
    )
    links = {link.link_id: link.bytes_per_s for link in scenario.links}
    cap = campaign.BANDWIDTH_BOTTLENECK_MBPS * 125_000

    assert campaign.BANDWIDTH_BOTTLENECK_MBPS == \
        campaign.loaded_service_model()["validation_bandwidth_mbps"][0]
    assert {links[f"link/{region}"] for region in campaign.REGIONS} == {cap}
    assert all(links[f"pipeline/{region}"] ==
               fits[region]["effective_pipeline_mbps"]["controlled_40"] * 125_000
               for region in campaign.REGIONS)
    assert all(routes[("source", region)] ==
               (f"link/{region}", f"pipeline/{region}")
               for region in campaign.REGIONS)
    for q in architecture.types:
        raw = fits[q.type_id.rsplit("/", 1)[-1]]["migration_components"]
        for method, support in q.migration.items():
            assert support.context_range == tuple(raw[method]["context_range"])
            assert support.bandwidth_range_bytes_per_s[0] <= min(
                links[link] for link in routes[("source", q.type_id.rsplit("/", 1)[-1])]
            )
            assert not support.allow_extrapolation


def test_destination_load_scales_replay_endpoint_but_not_kv_or_idle_anchor():
    fits = campaign.central_timing_fits()
    profile = ModelProfile.load(campaign.PROFILE)
    templates, _ = campaign.load_templates(campaign.MANIFEST, profile)
    pack = campaign.normalize_pack(
        profile, campaign.sample_pack(templates, 28, 4),
    )

    scenario, architecture, _, _ = campaign.build_problem(
        profile, pack, frozenset(), 2 / 3, fits,
    )
    model = campaign.loaded_service_model()
    assert model["slowdown_at_rho_0"]["replay"] == 1
    assert model["slowdown_at_rho_0"]["kv_transfer"] == 1
    bandwidths = {link.link_id: link.bytes_per_s for link in scenario.links}
    for pool in architecture.pools:
        loaded = architecture.type_by_id[pool.type_id].loaded
        context = 8192
        bandwidth = min(bandwidths[link] for link in pool.route)
        assert loaded["replay"].worst(.95, .95, context, bandwidth) \
            > loaded["replay"].worst(.25, .25, context, bandwidth) > 1
        assert loaded["kv_transfer"].worst(.95, .95, context, bandwidth) == 1


def test_supported_pack_has_replay_and_kv_candidates_for_every_session():
    profile = ModelProfile.load(campaign.PROFILE)
    templates, _ = campaign.load_templates(campaign.MANIFEST, profile)
    pack = campaign.normalize_pack(
        profile, campaign.sample_pack(templates, 28, campaign.DEFAULT_SEED),
    )
    scenario, architecture, _, _ = campaign.build_problem(
        profile, pack, frozenset(), 2 / 3, campaign.central_timing_fits(),
    )
    table = candidate_table(
        scenario, profile, architecture, "normal",
        ExpectedPower(scenario, profile),
    )
    slots = set(campaign.product(range(len(pack)), range(len(architecture.pools))))

    for method in ("replay", "kv_transfer"):
        assert {(candidate.session, candidate.pool) for candidate in table.candidates
                if candidate.method == method} == slots
    dominance = campaign.method_dominance(table, len(architecture.pools))
    assert dominance["candidate_matched_pairs"] == len(slots)
    assert not any(dominance[name] for name in (
        "candidate_replay_only", "candidate_kv_only", "candidate_neither",
    ))
    assert sum(dominance[name] for name in (
        "candidate_replay_dominates", "candidate_kv_dominates",
        "candidate_equivalent", "candidate_incomparable",
    )) == len(slots)


def test_factor_levels_apply_to_both_destinations():
    fits = campaign.central_timing_fits()
    profile = ModelProfile.load(campaign.PROFILE)
    templates, _ = campaign.load_templates(campaign.MANIFEST, profile)
    pack = campaign.normalize_pack(profile, campaign.sample_pack(templates, 28, 4))
    states = {}
    for constraints in (frozenset(), frozenset(campaign.FACTORS)):
        scenario, architecture, _, _ = campaign.build_problem(
            profile, pack, constraints, 2 / 3, fits,
        )
        states[bool(constraints)] = (scenario, architecture)

    physical = campaign.physical_route_mbps()
    for constrained, timing_level, compute_level, hbm_level in (
        (False, "natural", .25, 0), (True, "controlled_40", .95, .98),
    ):
        scenario, architecture = states[constrained]
        rates = {link.link_id: link.bytes_per_s for link in scenario.links}
        assert rates == {
            **{
                f"link/{region}": (
                    campaign.BANDWIDTH_BOTTLENECK_MBPS
                    if constrained else physical[region]
                ) * 125_000
                for region in campaign.REGIONS
            },
            **{
                f"pipeline/{region}": fits[region]["effective_pipeline_mbps"][
                    timing_level] * 125_000
                for region in campaign.REGIONS
            },
        }
        for pool in architecture.pools:
            dtype, replica = architecture.type_by_id[pool.type_id], pool.replicas[0]
            pressure = max(
                np.asarray(dtype.normals) @ np.asarray(replica.baseline_work)
                / np.asarray(dtype.bounds["normal"])
            )
            resident = np.floor(
                hbm_level * dtype.kv_capacity_tokens / dtype.kv_block_tokens
            ) * dtype.kv_block_tokens
            assert np.isclose(pressure, compute_level)
            assert replica.baseline_kv_tokens == resident
        assert {pool.replicas[0].replica_id for pool in architecture.pools} == \
            set(campaign.REGIONS)


def test_single_factors_change_only_their_physical_columns():
    fits = campaign.central_timing_fits()
    profile = ModelProfile.load(campaign.PROFILE)
    templates, _ = campaign.load_templates(campaign.MANIFEST, profile)
    pack = campaign.normalize_pack(profile, campaign.sample_pack(templates, 28, 4))
    tables = {}
    for name, constraints in {
        "none": frozenset(), "bandwidth": frozenset(("bandwidth",)),
        "hbm": frozenset(("hbm",)),
        "dest_compute": frozenset(("dest_compute",)),
    }.items():
        scenario, architecture, _, _ = campaign.build_problem(
            profile, pack, constraints, 2 / 3, fits,
        )
        tables[name] = candidate_table(
            scenario, profile, architecture, "normal",
            ExpectedPower(scenario, profile),
        )

    def choices(table):
        return {(row.session, row.pool, row.method): (i, row)
                for i, row in enumerate(table.candidates)}

    base = choices(tables["none"])
    bandwidth = choices(tables["bandwidth"])
    hbm = choices(tables["hbm"])
    compute = choices(tables["dest_compute"])
    key = next(key for key in base if key[1:] == (0, "kv_transfer"))
    base_i, base_kv = base[key]
    bandwidth_i, bandwidth_kv = bandwidth[key]
    route = tables["none"].resource_names.index("route:link/east")
    route_bound = tables["bandwidth"].resource_names.index("route:link/east")

    assert bandwidth_kv.migration_work_s > base_kv.migration_work_s
    assert bandwidth_kv.objective_cost_s > base_kv.objective_cost_s
    assert tables["bandwidth"].resources[route_bound, bandwidth_i] \
        > tables["none"].resources[route, base_i]

    for method in ("replay", "kv_transfer"):
        key = next(key for key in base.keys() & hbm.keys()
                   if key[1:] == (0, method))
        base_i, base_row = base[key]
        hbm_i, hbm_row = hbm[key]
        base_resource = tables["none"].resource_names.index("kv:pool/east")
        hbm_resource = tables["hbm"].resource_names.index("kv:pool/east")
        assert hbm_row.duration_s == pytest.approx(base_row.duration_s)
        assert tables["hbm"].resources[hbm_resource, hbm_i] \
            > tables["none"].resources[base_resource, base_i]

    replay_key = next(key for key in base if key[1:] == (0, "replay"))
    kv_key = next(key for key in base if key[1:] == (0, "kv_transfer"))
    assert compute[replay_key][1].duration_s > base[replay_key][1].duration_s
    assert compute[kv_key][1].duration_s == pytest.approx(base[kv_key][1].duration_s)


def test_bootstrap_preserves_bandwidth_release_order():
    profile = ModelProfile.load(campaign.PROFILE)
    rows = campaign.read_csv(campaign.TIMING)
    parent = campaign.json.loads(campaign.TIMING_PARENT.read_text())
    rng = np.random.default_rng(7)

    for _ in range(128):
        fits, _ = campaign.ordered_timing_fit(profile, parent, rows, rng)
        assert all(
            fits[region]["effective_pipeline_mbps"]["natural"] >=
            fits[region]["effective_pipeline_mbps"]["controlled_40"]
            for region in campaign.REGIONS
        )


def test_action_mix_figure_has_half_column_canvas(tmp_path):
    rows = [{
        "case_id": case_id, "replay_phase_load": .4,
        "kv_transfer_phase_load": .3, "not_moved_phase_load": .3,
    } for case_id, _, _ in campaign.factorial_cases()]

    campaign.plot(rows, tmp_path / "mix")
    image = campaign.plt.imread(tmp_path / "mix.png")

    assert image.shape[:2] == (2.5 * campaign.plot_style.SAVE_DPI,
                               3.85 * campaign.plot_style.SAVE_DPI)
    assert len({campaign.plot_style.ACTION_HATCHES[action]
                for action in campaign.ACTIONS}) == len(campaign.ACTIONS)


def test_stacked_action_mix_uses_pooled_mean_phase_shares():
    rows = [{
        "case_id": case_id,
        **{f"{action}_phase_load": value for action, value in zip(
            campaign.ACTIONS, values,
        )},
    } for case_id in campaign.DISPLAY_CASES
        for values in ((.2, .3, .5), (.6, .1, .3))]

    expected = np.asarray((.4, .2, .4))[:, None]
    assert campaign.action_mix_means(rows) == pytest.approx(
        np.repeat(expected, len(campaign.DISPLAY_CASES), axis=1),
    )
