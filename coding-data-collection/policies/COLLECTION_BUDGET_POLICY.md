# Collection Budget Policy

Collection uses scout, commit, and replicate phases.

## Scout

Cheap protocol-valid task triage. Scout runs are not final estimator
evidence unless they follow the full run protocol.

Record:

```text
wall-clock
agent turns
tool calls
tokens
estimated cost
validation events
errors
progress drops
terminal outcome
```

## Commit

Full evidence-quality run on selected tasks.

Each committed run must respect pre-registered limits for:

```text
wall-clock
agent turns
tool calls
tokens
estimated dollar cost
container CPU/memory/disk
```

## Replicate

Replicate only scientifically valuable phenomena:

```text
high-progress failure
recovery after progress drop
validation failure then recovery
stuck loops
agent self-claim before verifier failure
model/arm disagreement
```

Do not replicate all tasks uniformly.

