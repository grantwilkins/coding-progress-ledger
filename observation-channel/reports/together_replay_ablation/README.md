# Together Replay Progress Ablation

Observer-derived progress measurements for one SWE-Agent trace.

- raw row index: `349`
- instance: `bihealth__biomedsheets-23`
- selected models: `openai/gpt-oss-20b, openai/gpt-oss-120b`
- selected prompt variants: `fraction_complete, remaining_work, goal_closeness`
- selected contexts: `task_and_trace`
- selected ensemble sizes: `1, 5, 10, 40`
- max agents queried per condition: 40
- parallel workers per turn: 40
- temperature: 0.2
- max tokens: 256
- dataset target label: `True`

This report runs directional single-axis ablations, not a full Cartesian grid.
Model variants use the main prompt/context at the max ensemble size; prompt variants use the main model/context at the max ensemble size; ensemble-size variants use the main model/prompt/context.
Smaller ensemble sizes are derived by prefix-subsetting agent IDs from the max-agent run.

These are observer-derived progress measurements, not ground-truth progress labels.
The target label and evaluator logs are not included in per-turn prompts.

Final progress by condition:

- `openai/gpt-oss-120b | fraction_complete | task_and_trace | n=1`: 0.700
- `openai/gpt-oss-120b | fraction_complete | task_and_trace | n=10`: 0.795
- `openai/gpt-oss-120b | fraction_complete | task_and_trace | n=40`: 0.779
- `openai/gpt-oss-120b | fraction_complete | task_and_trace | n=5`: 0.840
- `openai/gpt-oss-120b | goal_closeness | task_and_trace | n=40`: 0.840
- `openai/gpt-oss-120b | remaining_work | task_and_trace | n=40`: 0.522
- `openai/gpt-oss-20b | fraction_complete | task_and_trace | n=40`: 0.422

Artifacts:

- `agent_estimates.csv`: one scalar estimate per agent per turn.
- `aggregate_progress.csv`: mean, standard deviation, and one-sigma band by turn.
- `condition_summary.csv`: stability metrics by condition.
- `agreement_matrix.csv`: pairwise curve agreement by condition.
- `aggregate_progress.png`: main-condition aggregate progress curve.
- `agent_trajectories.png`: main-condition agent trajectories with the mean overlaid.
- `prompt_comparison.png`: prompt wording comparison.
- `model_comparison.png`: observer model comparison.
- `context_comparison.png`: context comparison.
- `ensemble_size_convergence.png`: ensemble-size curves and deviation from the max ensemble.
- `agreement_matrix.png`: pairwise correlation heatmap.
