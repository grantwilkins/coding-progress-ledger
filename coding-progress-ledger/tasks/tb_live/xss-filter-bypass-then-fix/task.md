# xss-filter-bypass-then-fix

Build a safe HTML sanitizer as a Python package.

## What you must produce

A package importable as `xssfilter` exposing exactly one public function:

```python
def sanitize(html: str) -> str
```

`sanitize` takes an HTML string and returns a sanitized HTML string with
dangerous content removed. It must use `html.parser.HTMLParser` from the
Python standard library — do not install or import `beautifulsoup4`, `lxml`,
or any third-party parsing library.

## Required safety rules

Your `sanitize` must enforce all of the following rules:

**Rule 1 — Remove `<script>` tags and their content entirely.**

Any `<script>...</script>` block (including the tags themselves and everything
between them) must be removed from the output. Matching must be
case-insensitive: `<SCRIPT>`, `<Script>`, etc. are all caught.

**Rule 2 — Strip event-handler attributes.**

Any attribute whose name starts with `on` (case-insensitive) must be dropped
from the output. Examples: `onclick`, `onerror`, `ONLOAD`, `onmouseover`.
The tag itself is kept; only the dangerous attribute is removed.

**Rule 3 — Strip `javascript:` URLs from `href` and `src`.**

If an `href` or `src` attribute value, after stripping leading whitespace,
starts with `javascript:` (case-insensitive), that attribute must be dropped
entirely. The tag is kept; the unsafe URL is not. A safe `href` or `src`
(e.g. `https://example.com` or a relative path) must be preserved unchanged.

**Rule 4 — Remove `<iframe>`, `<object>`, and `<embed>` tags and their content.**

These tags and everything between their open and close tags must be stripped
from the output. Matching is case-insensitive.

**Rule 5 — Preserve benign structure.**

The following tags and their content must pass through without modification
(subject to the attribute rules above):

`<p>`, `<a>`, `<b>`, `<i>`, `<em>`, `<strong>`, `<div>`, `<span>`,
`<br>`, `<img>`

Text nodes that are not inside a dangerous tag must pass through unchanged.

## Repository layout

Use the standard `src/` layout. The verifier expects:

```
<agent_repo>/
  src/
    xssfilter/
      __init__.py     # exports `sanitize`
```

The verifier puts `<agent_repo>/src` on `PYTHONPATH` and runs `pytest`
against hidden test files. You may add a `pyproject.toml`, a `tests/`
directory, or anything else you want; only the `src/xssfilter/` contract
is load-bearing.

## Implementation guidance

The standard library's `html.parser.HTMLParser` is the right tool. Subclass
it, override `handle_starttag`, `handle_endtag`, `handle_data`, and
`handle_startendtag` to build up a sanitized output string. Keep a flag or
counter to track when you are inside a tag that should be suppressed
(script, iframe, object, embed) so you can drop everything until the
matching close tag.

For attribute filtering:
- Collect the attributes you want to keep into a list, then re-serialise the
  opening tag from that filtered list.
- An attribute whose name starts with `on` is dropped.
- An attribute named `href` or `src` whose stripped value starts with
  `javascript:` is dropped.

Do not use regular expressions to parse HTML tags — the HTMLParser approach
is more robust and is what the verifier tests for correctness against.

## Checking your work

Write your own tests in a `tests/` directory and run them with
`python -m pytest`. Cover at minimum:

- Script tag removal (lower and upper case)
- Event handler attribute stripping
- `javascript:` URL removal from `href` and `src`
- Preservation of safe `href` values
- Removal of `<iframe>` blocks
- Preservation of `<p>`, `<a>`, `<b>`, `<strong>`, `<img>`
- Mixed input: safe and unsafe content together
- Bare text with no tags

The hidden verifier applies these same categories of checks.

## How to track progress

You are running under the N_TB live ledger harness. After each
meaningful action (subtask added, started, completed, blocked, etc.),
emit one wire-format event with:

```bash
uv run python /Users/grantwilkins/houdini/coding-progress-ledger/scripts/tb_emit.py \
    /Users/grantwilkins/houdini/coding-progress-ledger/runs/tb_live/xss-filter-bypass-then-fix \
    <step_number> \
    '[{"op":"add","id":"s1","description":"...","category":"product"}]'
```

See the project's `docs/AGENT_USAGE.md` and `docs/TB_LIVE_TASK_FORMAT.md`
for the protocol. Use `product` for code-that-ships, `validation` for
tests / asserts / manual checks, `investigation` for reading / search /
trace work. Add subtasks as you discover them, not as a plan up front.
Mark complete only with concrete evidence.

## Done condition

You are done when `verifier.sh` exits 0 against your repo. The
verifier is hidden — you cannot read it. Your fastest path to done is
to write your own tests for each rule in the spec above, run them,
and only declare a leaf complete when the test passes.
