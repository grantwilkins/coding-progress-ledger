# Queued replay migration power trace

Slurm job `37368991` ran eight LMCache replay migrations across two A100 80 GB GPUs while trace-shaped inference remained queued on the source, then switched queued inference to the destination. No LMCache bypass was used.

`run-37368991/` contains the native PowerTrace `power_100ms.csv`, phase markers, queue metrics and requests for all three foreground intervals, migration telemetry, the migration result, and rendered plots. `driver.submitted.py` is the exact job driver; `../driver.py` adds only portable postprocessing and the documented sampling-coverage gate.

Slurm recorded exit `1` only after the completed measurement because the submitted reducer rejected any cadence gap over 250 ms. The portable reducer instead hard-fails below 90% overall coverage or above a one-second blind spot.

The logger attempted 100 ms cadence. Median cadence was 100 ms with 94.54% overall coverage and 83 queries dropped by PowerTrace's 200 ms timeout. Source-heavy and overlap windows retained 230 and 101 ticks, respectively.

From the repository root, regenerate `power_summary.json` and both plots with:

```bash
uv run python queue-haul/outputs/live-power-oneoff/driver.py --reduce queue-haul/outputs/live-power-oneoff/run-37368991
```
