from html.parser import HTMLParser

_BLOCK_TAGS = {"script", "iframe", "object", "embed"}


class _Sanitizer(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _BLOCK_TAGS:
            self._skip += 1
            return
        if self._skip:
            return
        safe_attrs = []
        for name, value in attrs:
            if name.lower().startswith("on"):
                continue
            if name.lower() in ("href", "src") and value and value.lstrip().lower().startswith("javascript:"):
                continue
            safe_attrs.append((name, value))
        attr_str = "".join(
            f' {n}="{v}"' if v is not None else f" {n}" for n, v in safe_attrs
        )
        self._parts.append(f"<{tag}{attr_str}>")

    def handle_endtag(self, tag):
        if tag in _BLOCK_TAGS:
            if self._skip:
                self._skip -= 1
            return
        if self._skip:
            return
        self._parts.append(f"</{tag}>")

    def handle_startendtag(self, tag, attrs):
        if tag in _BLOCK_TAGS:
            return
        if self._skip:
            return
        safe_attrs = []
        for name, value in attrs:
            if name.lower().startswith("on"):
                continue
            if name.lower() in ("href", "src") and value and value.lstrip().lower().startswith("javascript:"):
                continue
            safe_attrs.append((name, value))
        attr_str = "".join(
            f' {n}="{v}"' if v is not None else f" {n}" for n, v in safe_attrs
        )
        self._parts.append(f"<{tag}{attr_str} />")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def result(self):
        return "".join(self._parts)


def sanitize(html: str) -> str:
    p = _Sanitizer()
    p.feed(html)
    return p.result()
