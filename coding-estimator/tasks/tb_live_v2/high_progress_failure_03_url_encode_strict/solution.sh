#!/usr/bin/env bash
set -euo pipefail
cat > url_encode.py <<'PY'
_UNRESERVED = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789-._~"
)


def url_encode(s: str) -> str:
    out = []
    for byte in s.encode("utf-8"):
        ch = chr(byte)
        if ch in _UNRESERVED:
            out.append(ch)
        else:
            out.append(f"%{byte:02X}")
    return "".join(out)
PY
