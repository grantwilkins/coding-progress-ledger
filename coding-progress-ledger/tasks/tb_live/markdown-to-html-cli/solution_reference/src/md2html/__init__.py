import re

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"\*(.+?)\*")
_CODE = re.compile(r"`(.+?)`")
_LINK = re.compile(r"\[(.+?)\]\((.+?)\)")
_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")


def convert(md: str) -> str:
    return "\n\n".join(_render(b) for b in re.split(r"\n{2,}", md.strip()) if b)


def _render(block: str) -> str:
    lines = block.splitlines()
    if lines[0] == "```" and lines[-1] == "```":
        return "<pre><code>" + "\n".join(lines[1:-1]) + "\n</code></pre>"
    if all(line.startswith("- ") for line in lines):
        items = "\n".join(f"<li>{_inline(line[2:])}</li>" for line in lines)
        return f"<ul>\n{items}\n</ul>"
    m = _HEADING.match(block)
    if m and "\n" not in block:
        n = len(m.group(1))
        return f"<h{n}>{_inline(m.group(2))}</h{n}>"
    return f"<p>{_inline(block)}</p>"


def _inline(text: str) -> str:
    text = _CODE.sub(lambda m: f"<code>{m.group(1)}</code>", text)
    text = _LINK.sub(r'<a href="\2">\1</a>', text)
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    text = _ITALIC.sub(r"<em>\1</em>", text)
    return text
