# Together Replay Ensemble

Blind turn-by-turn ensemble estimates for a solved SWE-Agent trace.

- model: `openai/gpt-oss-120b`
- raw row index: `349`
- instance: `bihealth__biomedsheets-23`
- agents: 40
- parallel workers per turn: 40
- final mean fraction complete: 0.938
- final one-sigma band: [0.893, 0.983]
- dataset target label: `True`

The target label and evaluator logs were used only for selecting and documenting the solved trace, not in the per-turn prompts.

Artifacts:

- `agent_estimates.csv`: one scalar estimate per agent per turn.
- `aggregate_progress.csv`: mean, standard deviation, and one-sigma band by turn.
- `aggregate_progress.png`: aggregate progress curve.
- `agent_trajectories.png`: one trajectory per agent with the mean overlaid.
