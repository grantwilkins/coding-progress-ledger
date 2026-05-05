# AGENTS.md

Write the most succinct code for each task. We want the least amount of code to solve the task and the most simple code without many comments.

Prioritize hard fails as compared to try/except blocks and things that suppress errors.

Always add and commit after completing each task. Add a descriptive commit message for every change.

Always run tests after every change using `uv run pytest`.

At the end of each task, please update `TASKS.md` to reflect work that was completed and add tasks that should be completed next.

## Repo-specific addenda

Vagrant imports `ledger_progress` from the sibling repo `../coding-progress-ledger/`. Do not fork it. The only permitted upstream change for the MVP is a ~10-line pass-through hook in `ledger_progress/core.py:apply_event` so unknown `event_type` strings append without raising. Anything bigger requires user approval.

Do not invent a new event class, a second JSONL serializer, or a second replay engine. Ride on `LedgerEvent`, `serialization.to_jsonl`, and `core.replay`.

Do not add ILP, queue simulation, capacity constraints, or extra policies in the MVP. The MVP is two policies (`request_level`, `shared_state_aware`), four cost formulas, one toy trace, one plot. See `TASKS.md` § Workstream E for the gate.

If a change crosses workstream boundaries or touches the reuse contract with `coding-progress-ledger`, stop and ask before coding.

Read `CLAUDE.md` for the full rule set and reuse contract; this file is the short form.
