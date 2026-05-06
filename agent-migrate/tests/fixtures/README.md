# Test fixtures

## `swe_agent_pilot_s_07.json`

Single SWE-agent trajectory (~23 turns, 35 KB), copied from `coding-progress-ledger/external_data/swe_agent/pilot_cache/swe_agent_pilot_s_07.json` for agent-migrate's F2 adapter tests.

Selected because it is the **smallest pilot in the upstream cache that contains repeated tool-output content** (2 distinct tool outputs each appear ≥ 2× across the trajectory). That is the load-bearing third class of shared state for F2 — without it, F2's manifest collapses to "everyone reads the system prompt + issue text," which is structurally non-trivial but produces no inter-turn sharing graph. Of the 20 cached pilots, only 9 contain any repeated outputs; the F2 pre-flight critic flagged this as Risk 1.

Schema: `{instance_id, model_name, target, trajectory: [{role, text, system_prompt, mask, cutoff_date}], exit_status, generated_patch, eval_logs}`. `role` ∈ `{system, user, ai}`. The first turn is `system` and carries the SWE-agent system prompt; subsequent turns alternate `user` (issue text or tool output) and `ai` (LLM response with command).

License inherited from the upstream SWE-Bench / SWE-agent dataset.
