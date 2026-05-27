# Together Replay Time Remaining

Turn-by-turn time-remaining estimates from multiple Together models for one SWE-Agent trace.

Trace:

- raw row index: `349`
- instance: `bihealth__biomedsheets-23`
- dataset target success: `True`
- exit status: `submitted`
- evaluator log contains passed tests: `True`

Models:

- `deepseek-ai/DeepSeek-V4-Pro`
- `Qwen/Qwen3.6-Plus`
- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen2.5-7B-Instruct-Turbo`

Artifacts:

- `time_estimates.csv`: one time and confidence estimate per model per turn.
- `remaining_time.png`: seconds-left curves with inverse-confidence error bars by model.
