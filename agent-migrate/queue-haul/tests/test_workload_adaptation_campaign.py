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
- Validate against 30 seconds even though the migration horizon is 25 seconds.
- Call a target hit from power shortfall alone when execution feasibility failed.
- Ignore the requested regional timing fit and measured route rates.
- Make a state depend on whether a tighter counterfactual was solved first.
- Apply a constrained factor to only one destination or retain a prefill-only label.
- Drop a migration method from contexts covered by the regional timing model.
- Apply the measured load factor to route bytes, HBM, or the idle timing anchor.
"""

import numpy as np

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
    assert all(row["target_met"] == (
        row["feasible"] and row["power_shortfall_w"] <= campaign.POWER_TOLERANCE_W
    ) for row in rows)
    checks = campaign.factor_checks(rows)
    assert len(checks) == 12
    assert not any(row["fractional_opportunity_worsened_on_release"]
                   for row in checks)


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
            "hbm_utilization": 0, "migration_utilization": 0,
            "service_utilization": 0,
        })

    hbm = next(row for row in campaign.factor_checks(rows)
                if row["case_id"] == "hbm")

    assert hbm["fractional_lp_opportunity_change_w"] == 1
    assert not hbm["fractional_opportunity_worsened_on_release"]
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


def test_surface_validation_uses_grouped_splits_and_migration_horizon():
    rows = campaign.validate_surface()
    grouped = {}
    for row in rows:
        grouped.setdefault((row["source"], row["session_set"]), set()).add(
            row["split"]
        )

    assert all(len(splits) == 1 for splits in grouped.values())
    assert all(row["horizon_s"] == 25 for row in rows)
    assert all(row["false_feasible"] == (
        row["predicted_s"] <= 25 < row["measured_s"]
    ) for row in rows)


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

    assert {link.bytes_per_s for link in scenario.links} == {
        fits[region]["effective_pipeline_mbps"]["natural"] * 125_000
        for region in campaign.REGIONS
    }
    assert all(not service.route_overlap for service in services.values())
    assert services["pool/east"].kv_ingest_bytes_per_s == \
        fits["east"]["kv_ingest_lower_bound_bytes_per_s"]
    summary = campaign.json.loads(campaign.TIMING_SUMMARY.read_text())
    assert summary["migration_gate_passed"]
    assert all(row["coverage"] == 1 for row in summary["held_out"].values())


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
    assert campaign.method_dominance(table, len(architecture.pools)) == {
        "candidate_matched_pairs": len(slots),
        "candidate_replay_only": 0, "candidate_kv_only": 0,
        "candidate_neither": 0, "candidate_replay_dominates": len(slots),
        "candidate_kv_dominates": 0, "candidate_equivalent": 0,
        "candidate_incomparable": 0,
    }


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

    for constrained, bandwidth_level, compute_level, hbm_level in (
        (False, "natural", .25, 0), (True, "controlled_40", .95, .9),
    ):
        scenario, architecture = states[constrained]
        rates = {link.link_id: link.bytes_per_s for link in scenario.links}
        assert rates == {
            f"link/{region}": fits[region]["effective_pipeline_mbps"][
                bandwidth_level] * 125_000
            for region in campaign.REGIONS
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


def test_action_mix_figure_is_exactly_five_and_a_half_by_three(tmp_path):
    rows = [{
        "case_id": case_id, "replay": .4, "kv_transfer": .3,
        "not_moved": .3,
    } for case_id, _, _ in campaign.factorial_cases()]

    campaign.plot(rows, tmp_path / "mix")
    image = campaign.plt.imread(tmp_path / "mix.png")

    assert image.shape[:2] == (3 * campaign.plot_style.SAVE_DPI,
                               5.5 * campaign.plot_style.SAVE_DPI)
    assert len({campaign.plot_style.ACTION_HATCHES[action]
                for action in campaign.ACTIONS}) == len(campaign.ACTIONS)
