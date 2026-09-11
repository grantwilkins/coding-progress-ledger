"""Bounded solver scaling: shared physical constraints, repeated source ownership."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import highspy
import numpy as np
import scipy
from scipy.sparse import block_diag, csc_matrix, hstack, vstack

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pool_shed_priced_greedy import priced_greedy

CASES = ("coding-d30-b10", "coding_long-d120-b100")
FACTORS, REPEATS = (1, 4, 16, 64), 3
ABSOLUTE_TOLERANCE, MAX_ITERATIONS = .001, 10000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.dtype.str, value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def load(case):
    with np.load(ROOT / f"outputs/a100-greedy-quality-20260911/matrix/{case}.npz") as data:
        return dict(A=csc_matrix(data["matrix"]), b=data["capacity"].copy(),
                    g=data["gains"].copy(), debt=data["debt"].copy(), source_rows=data["replay"].shape[1])


def independent(base, factor):
    source = base["source_rows"]
    # Source ownership splits; destination, WAN and temporal rows remain shared.
    matrix = vstack((block_diag([base["A"][:source]] * factor, format="csc"),
                     hstack([base["A"][source:]] * factor, format="csc")), format="csc")
    return dict(A=matrix, b=np.r_[np.tile(base["b"][:source] / factor, factor), base["b"][source:]],
                g=np.tile(base["g"], factor), debt=np.tile(base["debt"], factor), source_rows=source * factor)


def normalized(problem):
    A, b, g = (problem[k] for k in ("A", "b", "g"))
    assert np.all(b > 0), "These frozen stress inputs require positive capacities"
    assert all(np.isfinite(x).all() and np.all(x >= 0) for x in (A.data, b, g, problem["debt"]))
    ids = np.flatnonzero(g > 0)
    B = A[:, ids].multiply((1 / b)[:, None]).tocsc()
    maximum = B.max(axis=0).toarray().ravel()
    assert len(ids) and np.all(maximum > 0)
    upper = 1 / maximum
    B.data *= np.repeat(upper, np.diff(B.indptr))
    B.sum_duplicates()
    B.eliminate_zeros()
    B.sort_indices()
    assert max(B.shape) < 2**31 and B.nnz < 2**31
    B.indptr, B.indices = np.asarray(B.indptr, np.int32), np.asarray(B.indices, np.int32)
    return B, g[ids] * upper, problem["debt"][ids] * upper, ids, upper


def certificate(A, b, g, chosen, prices):
    assert np.isfinite(chosen).all() and np.all(chosen >= 0)
    usage, objective = A @ chosen, float(g @ chosen)
    residual = float(max(0., np.max(usage / b, initial=0.) - 1))
    assert np.isfinite(usage).all() and residual <= 1e-8
    result = dict(objective=objective, max_relative_constraint_residual=residual,
                  max_constraint_residual=float(np.max(usage - b, initial=0.)),
                  positive_variables=int(np.count_nonzero(chosen > 0)),
                  variables_above_executor_mass_threshold=int(np.count_nonzero(chosen > 1e-10)),
                  primal_sha256=array_sha(chosen))
    if prices is None:
        return {**result, "upper_bound": None, "dual_certified": False}
    assert np.isfinite(prices).all() and np.all(prices >= 0)
    covered, bound = A.T @ prices, float(b @ prices)
    positive = g > 0
    deficit = float(max(0., np.max(1 - covered[positive] / g[positive], initial=0.)))
    assert np.isfinite(covered).all() and deficit <= 1e-10 and bound >= objective * (1 - 1e-10)
    return {**result, "upper_bound": bound, "absolute_gap": max(0., bound - objective),
            "relative_gap": max(0., (bound - objective) / bound) if bound else 0.,
            "max_relative_dual_deficit": deficit, "dual_certified": True, "dual_sha256": array_sha(prices)}


def reference(B, c):
    """Direct primary HiGHS LP; full input witnesses checked after tiny-entry filtering."""
    model, solver = highspy.HighsLp(), highspy.Highs()
    model.num_row_, model.num_col_ = B.shape
    scale = float(c.max())
    model.col_cost_, model.col_lower_, model.col_upper_ = -c / scale, np.zeros(len(c)), np.full(len(c), highspy.kHighsInf)
    model.row_lower_, model.row_upper_ = np.full(B.shape[0], -highspy.kHighsInf), np.ones(B.shape[0])
    model.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    model.a_matrix_.start_, model.a_matrix_.index_, model.a_matrix_.value_ = B.indptr, B.indices, B.data
    for key, value in (("output_flag", False), ("threads", 1), ("time_limit", 10.),
                       ("presolve", "on"), ("primal_feasibility_tolerance", 1e-10),
                       ("dual_feasibility_tolerance", 1e-9), ("small_matrix_value", 1e-12)):
        assert solver.setOptionValue(key, value) == highspy.HighsStatus.kOk
    imported = solver.passModel(model)
    assert imported != highspy.HighsStatus.kError
    if imported == highspy.HighsStatus.kWarning:
        warnings.warn("HiGHS import warning: witnesses are checked against every coefficient in the full supplied CSC matrix.")
    run_status = solver.run()
    assert run_status != highspy.HighsStatus.kError
    status, solution, detail = solver.getModelStatus(), solver.getSolution(), solver.getInfo()
    chosen = np.maximum(solution.col_value, 0.) if solution.value_valid else np.zeros(len(c))
    repair = max(1., float(np.max(B @ chosen, initial=0.)))
    chosen /= repair
    prices, multiplier = None, None
    if solution.dual_valid:
        prices = np.maximum(-np.asarray(solution.row_dual), 0.) * scale
        coverage = B.T @ prices
        if np.all(coverage > 0):
            multiplier = max(1., float(np.max(c / coverage)))
            prices *= multiplier
        else:
            prices = None
    info = certificate(B, np.ones(B.shape[0]), c, chosen, prices)
    optimal = status == highspy.HighsModelStatus.kOptimal
    assert not optimal or (prices is not None and info["absolute_gap"] <= 1e-8)
    return chosen, prices, {**info, "model_status": str(status), "run_status": str(run_status),
        "import_status": str(imported), "optimal": optimal, "solver_primal_available": bool(solution.value_valid),
        "solver_dual_available": bool(solution.dual_valid), "primal_repair_scale": repair,
        "dual_repair_scale": multiplier, "objective_cost_scale": scale,
        "simplex_iterations": int(detail.simplex_iteration_count), "ipm_iterations": int(detail.ipm_iteration_count)}


def run_case(case, factor, base, load_s):
    start = time.perf_counter()
    problem = independent(base, factor)
    assembly_s = time.perf_counter() - start
    start = time.perf_counter()
    B, c, debt, ids, upper = normalized(problem)
    preparation_s = time.perf_counter() - start
    inputs = dict(csc_data=B.data, csc_indices=B.indices, csc_indptr=B.indptr, csc_shape=np.asarray(B.shape),
                  gains=c, debt=debt, capacity=np.ones(B.shape[0]))
    frozen = {key: array_sha(value) for key, value in inputs.items()}
    repeats, best = [], {}
    for repeat in range(REPEATS):
        result = dict(repeat=repeat + 1, order=["native", "lp"] if repeat % 2 == 0 else ["lp", "native"])
        for name in result["order"]:
            start = time.perf_counter()
            if name == "native":
                chosen, info = priced_greedy(B, inputs["capacity"], c, np.zeros(len(c)), debt,
                    tolerance=1e-12, absolute_tolerance=ABSOLUTE_TOLERANCE,
                    max_iterations=MAX_ITERATIONS, return_prices=True)
                prices = info.pop("dual_prices")
            else:
                chosen, prices, info = reference(B, c)
            elapsed_s = time.perf_counter() - start
            start = time.perf_counter()
            raw_x = np.zeros(len(problem["g"]))
            raw_x[ids] = chosen * upper
            raw_y = None if prices is None else prices / problem["b"]
            checked = certificate(problem["A"], problem["b"], problem["g"], raw_x, raw_y)
            assert np.isclose(checked["objective"], info["objective"], rtol=1e-10, atol=1e-12)
            if prices is not None:
                assert np.isclose(checked["upper_bound"], info["upper_bound"], rtol=1e-10, atol=1e-12)
            result[name] = {**info, **checked, "elapsed_s": elapsed_s,
                            "external_original_unit_check_s": time.perf_counter() - start}
            assert frozen == {key: array_sha(value) for key, value in inputs.items()}, "Solver mutated shared inputs"
            if name not in best or checked["objective"] > best[name][0]:
                best[name] = (checked["objective"], repeat + 1, raw_x, raw_y)
        native, lp = result["native"], result["lp"]
        best_bound = min(x for x in (native["upper_bound"], lp["upper_bound"]) if x is not None)
        result["native_objective_shortfall"] = dict(
            signed_to_lp_primal=lp["objective"] - native["objective"],
            lower_bound=max(0., lp["objective"] - native["objective"]),
            upper_bound=max(0., best_bound - native["objective"]), lp_reference_optimal=lp["optimal"])
        assert lp["objective"] <= native["upper_bound"] + 1e-8
        if lp["upper_bound"] is not None:
            assert native["objective"] <= lp["upper_bound"] + 1e-8
        repeats.append(result)
    path = OUT / f"scaling-{case}-x{factor}.npz"
    witnesses = {f"{name}_{kind}": value for name, (_, _, x, y) in best.items()
                 for kind, value in (("primal", x), ("dual", np.array([]) if y is None else y))}
    np.savez_compressed(path, **inputs, **witnesses, original_capacity=problem["b"], original_gains=problem["g"],
                        retained_columns=ids, column_upper=upper, original_shape=np.asarray(problem["A"].shape))
    return dict(case=case, factor=factor, source_rows=problem["source_rows"],
        shared_rows=problem["A"].shape[0] - problem["source_rows"], source_capacity=float(problem["b"][:problem["source_rows"]].sum()),
        original_shape=list(problem["A"].shape), original_nnz=int(problem["A"].nnz),
        normalized_shape=list(B.shape), normalized_nnz=int(B.nnz), zero_gain_columns_removed=len(problem["g"]) - len(c),
        file_load_s=load_s, assembly_s=assembly_s, common_preparation_s=preparation_s,
        common_input_array_sha256=frozen, repeats=repeats,
        best_witness_repeat={name: value[1] for name, value in best.items()},
        witness_file=str(path.relative_to(ROOT)), witness_file_sha256=sha(path))


def provenance():
    binary = Path(importlib.util.find_spec("_queue_haul_native._queue_haul_native").origin)
    assert binary.suffix in (".so", ".pyd"), "Provenance must identify the compiled extension"
    files = [Path(__file__).resolve(), ROOT / "pool_shed_priced_greedy.py", binary,
             *(ROOT / f"native/{name}" for name in ("src/lib.rs", "src/packing.rs", "Cargo.toml", "Cargo.lock", "pyproject.toml", "rust-toolchain.toml")),
             *(ROOT / f"outputs/a100-greedy-quality-20260911/matrix/{case}.npz" for case in CASES)]
    return {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha(path) for path in files}


def main():
    path = OUT / "scaling.json"
    assert OUT.is_relative_to(ROOT), "Benchmark artifacts must stay within the repository"
    assert not path.exists(), "Preserve recorded timings; explicitly archive an earlier result before rerunning"
    OUT.mkdir(parents=True, exist_ok=True)
    sources = provenance()
    report = dict(complete=False, input_sha256=sources, records=[],
        environment=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
                         highspy=highspy.Highs().version(), threads=1, thread_env={key: os.environ[key] for key in
                             ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}),
        parameters=dict(cases=CASES, factors=FACTORS, serial_repeats=REPEATS, native_relative_tolerance=1e-12,
                        native_absolute_tolerance=ABSOLUTE_TOLERANCE, native_max_iterations=MAX_ITERATIONS, lp_time_limit_s=10.),
        scope=["Solver stress, not a datacenter simulation, new measured workload diversity, or a physical scaling forecast.",
            "Each block has independent source-ownership rows; every original destination, WAN, memory, standing and temporal row stays shared. Total source capacity is fixed; capacities split fractionally among repeated coefficient blocks.",
            "Both solvers receive the same canonical CSC matrix and primary objective. Zero-gain columns are removed and capacities/columns normalized once in common preparation. Native performs its own production validation and normalization inside its timed call.",
            "Native timing includes zero-incumbent allocation, native greedy seed, refinement, fill and its primal/dual certificate. Direct LP timing includes fresh model assembly, import, single-thread primary solve and full supplied-matrix witness checks; no stored solution, secondary solve or warm start is used.",
            "Common input loading, block assembly, preparation and external original-unit audits are reported separately. Python/module imports are outside solver timings. Three serial paired repeats alternate solver order; no GPU work is launched.",
            "Absolute native tolerance .001 is 0.1 percentage point of the fixed total-source objective, not a guarantee of 0.1 percent relative error. Actual relative gaps and budget exhaustion are retained.",
            "Returned positive-variable counts are recorded before and after the executor's 1e-10 mass threshold. They describe primal support, not measured execution-wave counts or end-to-end simulation runtime.",
            "HiGHS receives a 10-second solver limit and max-cost objective normalization. Tiny-coefficient import warnings are retained; primal and dual witnesses are checked against all input coefficients, repaired only by recorded feasibility/coverage scalings. Nonoptimal references remain explicitly qualified.",
            "Each NPZ retains common normalized inputs plus the best primal and its matching native dual, and best LP primal/dual when available. Original matrices are reproducible from the hashed archived input and source-ownership construction."])
    started = time.perf_counter()
    for case in CASES:
        before = time.perf_counter()
        base = load(case)
        load_s = time.perf_counter() - before
        for factor in FACTORS:
            record = run_case(case, factor, base, load_s)
            assert provenance() == sources, "Benchmark inputs changed during the run"
            report["records"].append(record)
            report["wall_s"] = time.perf_counter() - started
            path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({key: record[key] for key in ("case", "factor", "normalized_shape", "normalized_nnz")}), flush=True)
    report["complete"] = True
    report["unconverged_native_repeats"] = [[r["case"], r["factor"], t["repeat"]] for r in report["records"] for t in r["repeats"] if not t["native"]["converged"]]
    report["nonoptimal_lp_repeats"] = [[r["case"], r["factor"], t["repeat"]] for r in report["records"] for t in r["repeats"] if not t["lp"]["optimal"]]
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def test_source_blocks_share_global_rows_and_conserve_source_capacity():
    base = dict(A=csc_matrix([[1., 0.], [0., 2.], [3., 4.]]), b=np.array([2., 6., 7.]),
                g=np.array([.1, .2]), debt=np.array([.3, .4]), source_rows=2)
    expanded = independent(base, 4)
    assert expanded["A"].shape == (9, 8) and expanded["source_rows"] == 8
    assert expanded["b"][:8].sum() == base["b"][:2].sum()
    np.testing.assert_array_equal(expanded["A"][:8].toarray(), block_diag([base["A"][:2]] * 4).toarray())
    np.testing.assert_array_equal(expanded["A"][8:].toarray(), np.tile(base["A"][2:].toarray(), 4))
    np.testing.assert_array_equal(expanded["b"][8:], base["b"][2:])


def test_common_normalization_preserves_objective_and_original_dual():
    problem = dict(A=csc_matrix([[2., 0., 1.], [1., 3., 0.]]), b=np.array([4., 6.]),
                   g=np.array([.25, .5, 0.]), debt=np.zeros(3))
    B, c, _, ids, upper = normalized(problem)
    z, prices = np.array([.2, .3]), np.array([.5, 1.])
    raw = np.zeros(3)
    raw[ids] = z * upper
    np.testing.assert_allclose(B @ z, (problem["A"] @ raw) / problem["b"])
    assert np.isclose(c @ z, problem["g"] @ raw)
    checked = certificate(problem["A"], problem["b"], problem["g"], raw, prices / problem["b"])
    assert checked["dual_certified"] and np.isclose(checked["upper_bound"], prices.sum())
    assert B.has_canonical_format and B.indices.dtype == np.int32 and B.indptr.dtype == np.int32


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT, help="new result directory within the repository")
    OUT = parser.parse_args().out.resolve()
    main()
