"""Invariance test: scale the measured constants by 10x/100x and see if the
formulation survives unchanged, and which row binds next."""
import json, sys
from dataclasses import replace
import numpy as np
import destination_bench as db
from destination import ContextRate, DestinationType
from profiles import ModelProfile
from power_model import ExpectedPower
from pool_planner import candidate_table, _lp

profile = ModelProfile.load(db.DEFAULT_MODEL)
manifest = json.loads(db.DEFAULT_MANIFEST.read_text())

def scale_rate(r, m):
    return ContextRate(r.contexts, tuple(x * m for x in r.rates))

def build(prof, sessions, replicas, pressure, kv_mult, prefill_mult, ingest_mult):
    arch = db.architecture(prof, sessions, replicas, pressure)
    q = arch.types[0]
    q2 = replace(q,
        kv_capacity_tokens=int(db.KV_CAPACITY_TOKENS * kv_mult),
        prefill=scale_rate(q.prefill, prefill_mult))
    pool = replace(arch.pools[0], type_id=q2.type_id)
    return replace(arch, types=(q2,), pools=(pool,))

def binding(t, sel, thresh=0.95):
    if not sel: return "none"
    u = np.asarray(t.resources[:, sorted(sel)].sum(1)).ravel()
    groups = {}
    for nm, v in zip(t.resource_names, u):
        k = nm.split(":")[0]
        groups[k] = max(groups.get(k, 0), v)
    hot = sorted((k for k, v in groups.items() if v >= thresh),
                 key=lambda k: -groups[k])
    return ",".join(f"{k}={groups[k]:.2f}" for k in hot) or \
           "slack(max %s=%.2f)" % max(groups.items(), key=lambda kv: kv[1])

rows = []
for jc in ("interactive_coding", "coding"):
    ratio = db.log_bytes_per_token(db.WORKLOADS[jc])
    packed, replicas = db.pack_source(
        db.sample_sessions(db.trace_shapes(manifest, jc), 10_000, 0, ratio), profile)
    print(f"\n===== {jc}: {replicas} source replicas =====", flush=True)
    for streams in (1, 2):
        prof_s = replace(profile, max_source_streams=streams)
        sc = db.scenario(prof_s, packed, replicas, db.Pressure())
        prof2 = db.extrapolate_replay(prof_s, sc.sessions, sc.deadline_s)
        power = ExpectedPower(sc, prof2, "central")
        target = max(0.0, power.power(True) - sc.power_limit_w)
        for kv_mult in (1, 10, 100):
            for pf_mult in (1, 100):
                arch = build(prof2, sc.sessions, replicas, db.Pressure(),
                             kv_mult, pf_mult, 1)
                t = candidate_table(sc, prof2, arch, "emergency", power)
                sel = _lp(t, target)
                gain = sum(t.candidates[i].gain_w for i in sel)
                b = binding(t, sel)
                rows.append(dict(workload=jc, streams=streams, kv_mult=kv_mult,
                                 prefill_mult=pf_mult, landed=len(sel),
                                 gain_w=round(gain), binding=b))
                print(f"  streams={streams}  KVx{kv_mult:<4d} prefillx{pf_mult:<4d} "
                      f"-> landed {len(sel):5d}  gain {gain:7.0f} W   binding: {b}",
                      flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
