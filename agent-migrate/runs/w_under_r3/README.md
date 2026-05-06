# W under R3 — anchor regime classification across model profiles

`w_under_r3.csv` reports per-(anchor, model_profile, cell) regime
classification. `w_under_r3_summary.json` lists the cells where the
observed regime differs from the `compact_kv` baseline. A flip means
model architecture is enough to move the anchor across the regime
map — the W-anchors must then carry per-profile hypothesis labels,
not just one.
