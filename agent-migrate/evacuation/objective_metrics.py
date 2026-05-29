"""Evacuation metrics for one (inst, Stage1Result, Stage2Result) triple.

A "class" is a model x log-T bin (the n_bins aggregation in instance.py); u_q is
its evacuated fraction. Class percentiles are equal-weight per class so small,
expensive cohorts stay visible; the CDF (driver) is population-weighted.
"""

from __future__ import annotations

import numpy as np

from instance import T_MAX, T_MIN

# Log-T buckets; edges coincide with build_instance(n_bins=5), so each class
# falls cleanly into one bucket.
BUCKET_EDGES = np.logspace(np.log10(T_MIN), np.log10(T_MAX), 6)
BUCKET_LABELS = ("1k-4k", "4k-16k", "16k-64k", "64k-256k", "256k-1M")


def evac_fraction(inst, z):
    return 1.0 - z / inst.n


def bucket_idx(inst):
    return np.clip(np.digitize(inst.T, BUCKET_EDGES) - 1, 0, len(BUCKET_LABELS) - 1)


def evac_summary(inst, z):
    u = evac_fraction(inst, z)
    n = inst.n
    evac = float((n - z).sum())
    # State-aware evacuation: jobs differ by orders of magnitude in context tokens
    # (T) and KV bytes (eta*T), so weight the evacuated fraction by each.
    tok = inst.T * n
    kv = inst.eta * inst.T * n
    return {
        "total_evacuated": evac,
        "total_unmoved": float(z.sum()),
        "evacuated_fraction_total": evac / float(n.sum()),
        "token_weighted_evacuation": float((inst.T * (n - z)).sum() / tok.sum()),
        "kv_weighted_evacuation": float((inst.eta * inst.T * (n - z)).sum() / kv.sum()),
        "min_class_evacuated_fraction": float(u.min()),
        "p10_class_evacuated_fraction": float(np.percentile(u, 10)),
        "p50_class_evacuated_fraction": float(np.percentile(u, 50)),
        "p90_class_evacuated_fraction": float(np.percentile(u, 90)),
        "num_starved_classes": int((u < 1e-6).sum()),
    }


def by_bucket(inst, z):
    b = bucket_idx(inst)
    out = {}
    for k, label in enumerate(BUCKET_LABELS):
        m = b == k
        tot = float(inst.n[m].sum())
        if tot == 0:
            continue
        ev = float((inst.n[m] - z[m]).sum())
        out[label] = {
            "jobs_total": tot,
            "jobs_evacuated": ev,
            "evacuated_fraction": ev / tot,
            "mean_T": float(np.average(inst.T[m], weights=inst.n[m])),
        }
    return out


def by_model(inst, s1):
    out = {}
    for m, name in enumerate(inst.M_names):
        q = inst.model_idx == m
        tot = float(inst.n[q].sum())
        if tot == 0:
            continue
        out[name] = {
            "jobs_total": tot,
            "jobs_evacuated": float((inst.n[q] - s1.z[q]).sum()),
            "evacuated_fraction": float((inst.n[q] - s1.z[q]).sum() / tot),
            "replay_jobs": float(s1.x_R[q].sum()),
            "state_jobs": float(s1.x_S[q].sum()),
            "unmoved_jobs": float(s1.z[q].sum()),
        }
    return out


def pressure_summary(s2):
    vals = np.array(list(s2.pressures.values()))
    by_kind = lambda k: [v for n, v in s2.pressures.items() if n.startswith(k)]
    return {
        "phi_star": float(s2.phi_star),
        "max_net_pressure": float(max(by_kind("net|"), default=0.0)),
        "max_pfill_pressure": float(max(by_kind("pfill|"), default=0.0)),
        "max_ing_pressure": float(max(by_kind("ing|"), default=0.0)),
        "mean_pressure": float(vals.mean()),
        "p90_pressure": float(np.percentile(vals, 90)),
    }


def _mix(R, S):
    moved = R + S
    return {
        "total_replay": float(R),
        "total_state": float(S),
        "replay_fraction": float(R / moved) if moved > 0 else 0.0,
        "state_fraction": float(S / moved) if moved > 0 else 0.0,
    }


def action_mix(inst, x_R, x_S):
    out = {"overall": _mix(x_R.sum(), x_S.sum()), "by_model": {}, "by_bucket": {}}
    for m, name in enumerate(inst.M_names):
        q = inst.model_idx == m
        out["by_model"][name] = _mix(x_R[q].sum(), x_S[q].sum())
    b = bucket_idx(inst)
    for k, label in enumerate(BUCKET_LABELS):
        sel = b == k
        if sel.any():
            out["by_bucket"][label] = _mix(x_R[sel].sum(), x_S[sel].sum())
    return out


def metrics(inst, s1, s2):
    """Full nested metrics dict for one objective."""
    return {
        "objective": s1.objective,
        "alpha_star": s1.alpha_star,
        "U_star": s1.U_star,
        "Z_star": s1.Z_star,
        "evacuation_stage1": evac_summary(inst, s1.z),
        "evacuation_stage2": evac_summary(inst, s2.z),
        "by_bucket": by_bucket(inst, s1.z),
        "by_model": by_model(inst, s1),
        "pressure": pressure_summary(s2),
        "action_mix": action_mix(inst, s1.x_R, s1.x_S),
    }
