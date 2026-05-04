"""Auto-generated caveat blocks for reports.

Every report that consumes retrospective sources must include the
leakage caveat. The set of retrospective sources is derived from the
canonical source registry (`coding_estimator.ingest.sources`), not a
hand-maintained prefix list — adding a new retrospective source there
automatically wires it through here.
"""

from __future__ import annotations

from collections.abc import Iterable

from coding_estimator.ingest.sources import SOURCES, retrospective_source_ids

LIVE_SOURCES: frozenset[str] = frozenset({"tb_live"})

RETROSPECTIVE_CAVEAT = """\
> ⚠️ **Retrospective annotation caveat.** This report draws on
> retrospective sources ({sources}). These ledgers were annotated
> post-hoc with knowledge of the run's outcome. Annotator-outcome
> leakage is unfixable at the estimator layer; treat any reported
> performance as an upper bound on "realistic" performance, not a
> faithful estimate. See `docs/SOURCES.md` for the full statement.
"""

TB_LIVE_FRAMING = """\
> ℹ️ **TB-12 framing.** `tb_live` measures **online realism**, not
> model performance: 12 first-party live runs is too thin a benchmark
> for a headline AUROC. Use this report to confirm the pipeline
> produces honest online checkpoint features; do not optimize against
> tb_live-only metrics.
"""


def _is_retrospective(source_id: str) -> bool:
    """A source is retrospective iff the registry says it is. Unknown
    source ids are treated as non-retrospective; the source must be
    declared in `sources.SOURCES` for the caveat machinery to fire,
    which is exactly the contract we want."""
    return source_id in retrospective_source_ids()


def caveat_block(sources_used: Iterable[str]) -> str:
    """Return the caveat block(s) appropriate to the source set used in
    a report. An empty string is never returned for retrospective sources
    -- the absence of a caveat is a bug."""
    sources = sorted(set(sources_used))
    blocks: list[str] = []
    retro = [s for s in sources if _is_retrospective(s)]
    if retro:
        blocks.append(RETROSPECTIVE_CAVEAT.format(sources=", ".join(f"`{s}`" for s in retro)))
    if any(s in LIVE_SOURCES for s in sources):
        blocks.append(TB_LIVE_FRAMING)
    return "\n".join(blocks)


def assert_caveat_present(report_text: str, sources_used: Iterable[str]) -> None:
    """Hard fail if a report omits the caveat block its source set requires.
    Use this in tests for any function that emits a report consuming
    retrospective or tb_live data."""
    expected = caveat_block(sources_used)
    if not expected:
        return
    # We check on a per-fragment basis -- the report may interleave the
    # caveat with other content as long as each required block is present.
    has_retro = any(_is_retrospective(s) for s in sources_used)
    has_live = any(s in LIVE_SOURCES for s in sources_used)
    for fragment_marker in (
        "Retrospective annotation caveat" if has_retro else "",
        "TB-12 framing" if has_live else "",
    ):
        if fragment_marker and fragment_marker not in report_text:
            raise AssertionError(
                f"report omits required caveat fragment: {fragment_marker!r}"
            )


def assert_registry_consistency() -> None:
    """Cross-check the caveats helper against the source registry.

    Rule: every retrospective source the registry declares must be
    recognized by `_is_retrospective`. Any drift (e.g. a new retrospective
    source added to the registry that this helper doesn't see) is a bug."""
    declared = retrospective_source_ids()
    seen = {sid for sid in SOURCES if _is_retrospective(sid)}
    if declared != seen:
        missing = declared - seen
        extra = seen - declared
        raise AssertionError(
            f"caveats helper out of sync with sources registry "
            f"(missing={sorted(missing)}, extra={sorted(extra)})"
        )
