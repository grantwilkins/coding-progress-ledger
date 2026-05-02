# Hermes — pilot sampling policy

## Target

5 traces, all from category `Terminal & Coding`. One config (`kimi`)
to keep the first wave homogenous.

## Why 5 (not 10/10 like SWE-agent)

Hermes has no `final_success` label. Balanced success/failure
sampling is impossible. The pilot's purpose is **feasibility**:
confirm the existing pipeline runs unchanged on a non-SWE source.
N=5 is sufficient for the qualitative gates in HP3 acceptance:
enum coverage, step-segmentation unambiguity, ≤1 new pitfall per
trace.

## Inclusion criteria (in order)

```text
I1. category == "Terminal & Coding"
I2. config == "kimi"
I3. conversation length >= 6  (at least system + human + 2 gpt + 2 tool)
I4. at least one <tool_call> block somewhere in the gpt turns
I5. id not previously sampled
```

## Exclusion criteria

```text
E1. truncated tool_call/tool_response markup (any unmatched <tool_call> or <tool_response>)
E2. conversations[0].from != "system"
E3. duplicate id within the inventory
```

## Determinism

```text
seed: 0
sort key: id (UUID, ascii lex)
selection: first 5 rows passing I1-I5 in sorted order, with seed=0 used for
           tie-breaking only when strict sort is insufficient
```

Re-running with the same `seed=0` against the same inventory CSV must
produce a byte-identical pilot CSV.

## What we are NOT balancing on

- `final_success` — does not exist.
- `subcategory` — too sparse at N=5; revisit at HP3 scale-out.
- model config — held fixed at `kimi`. cross-config comparison is a
  later wave.

## Pilot IDs

`hermes_pilot_<n>` where `n` is `01..05`, assigned in id-sorted order.
