# markdown-to-html-cli

Build a minimal Markdown → HTML converter as a Python package and CLI.

## What you must produce

A package importable as `md2html` exposing exactly one public function:

```python
def convert(md: str) -> str
```

`convert` takes a Markdown string and returns the corresponding HTML.

The package must also be runnable as a module:

```bash
python -m md2html path/to/input.md     # writes HTML to stdout
python -m md2html < input.md            # reads from stdin if no path
```

## Required Markdown features

Your `convert` must handle, at minimum, the following block- and
inline-level constructs. The hidden verifier exercises each.

**Block level**

- ATX headings `#` through `######` → `<h1>` through `<h6>`.
- Paragraphs separated by one or more blank lines → wrapped in `<p>`.
- Unordered lists where every line in a block starts with `- ` →
  `<ul><li>...</li>...</ul>`. Inline markup inside list items must
  still render.
- Fenced code blocks delimited by lines containing exactly ``` →
  `<pre><code>...\n</code></pre>`. The body is preserved verbatim;
  inline markup inside a code block must NOT be interpreted.

**Inline level (everywhere except inside code blocks)**

- `**bold**` → `<strong>bold</strong>`.
- `*italic*` → `<em>italic</em>`.
- `` `code` `` → `<code>code</code>`.
- `[label](url)` → `<a href="url">label</a>`.

## What "blocks" means

Split the input on runs of two or more consecutive newlines. Each
resulting non-empty chunk is one block. A block is a heading iff it
matches `^#{1,6} `. A block is a code block iff its first line is
exactly ``` and its last line is exactly ```. A block is a list iff
every line starts with `- `. Otherwise it is a paragraph.

## Output normalization rules

The verifier compares your output to expected HTML using
`output.strip() == expected.strip()`. So:

- Trailing newlines on the whole document are ignored.
- Internal whitespace inside blocks is significant.
- Adjacent blocks are joined with one blank line between them
  (`\n\n`).
- Inside a `<ul>`, `<li>` items are on their own lines; the opening
  `<ul>` and closing `</ul>` are on their own lines too.
- Inside a `<pre><code>...</code></pre>`, the code body ends with
  `\n` before the closing `</code>` (mirrors how a code block renders
  with a trailing newline preserved).

## What is NOT required

You do not need to support: nested lists, ordered lists, blockquotes,
images, autolinks, HTML passthrough, reference-style links, setext
headings, horizontal rules, tables, inline HTML, or escaped characters.
The verifier will not exercise these.

## Repository layout

Use the standard `src/` layout. The verifier expects:

```
<agent_repo>/
  src/
    md2html/
      __init__.py     # exports `convert`
      __main__.py     # CLI entry point (python -m md2html ...)
```

The verifier puts `<agent_repo>/src` on `PYTHONPATH` and runs `pytest`
against hidden test files. You may add a `pyproject.toml`, a `tests/`
directory, a `README.md`, or anything else you want; only the
`src/md2html/` contract is load-bearing.

## How to track progress

You are running under the N_TB live ledger harness. After each
meaningful action (subtask added, started, completed, blocked, etc.),
emit one wire-format event with:

```bash
uv run python /Users/grantwilkins/houdini/coding-progress-ledger/scripts/tb_emit.py \
    /Users/grantwilkins/houdini/coding-progress-ledger/runs/tb_live/markdown-to-html-cli \
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
to write your own tests for each feature in the spec above, run them,
and only declare a leaf complete when the test passes.
