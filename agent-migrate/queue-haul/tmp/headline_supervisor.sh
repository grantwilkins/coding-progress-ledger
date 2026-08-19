#!/bin/sh
# Keeps the fleet-shed headline run alive independent of the launching session:
# waits for running workers, fills in missing shards (3 attempts), then reduces.
QH=/Users/grantwilkins/houdini/agent-migrate/queue-haul
PY=/Users/grantwilkins/houdini/.venv/bin/python
OUT=$QH/outputs/fleet-shed-frontier-a100-20260818
LOG=$QH/tmp/headline-supervisor
missing() { for s in $(seq 0 31); do
    [ -f "$OUT/$(printf 'shard-%02d-headline.csv' "$s")" ] || echo "$s"; done; }
while pgrep -f "run-shard" >/dev/null; do sleep 60; done
attempt=0
while [ -n "$(missing)" ] && [ $attempt -lt 3 ]; do
  attempt=$((attempt + 1))
  missing | xargs -P 8 -I% sh -c "$PY $QH/fleet_shed_frontier_campaign.py run-shard --shard % --subset headline > $LOG/shard-%.log 2>&1"
done
sleep 120
[ -f "$OUT/frontier.csv" ] || {
  $PY $QH/fleet_shed_frontier_campaign.py reduce > "$LOG/reduce.log" 2>&1 &&
  $PY $QH/plot_fleet_shed_frontier.py "$OUT" >> "$LOG/reduce.log" 2>&1
}
echo "done $(date)" >> "$LOG/supervisor.log"
