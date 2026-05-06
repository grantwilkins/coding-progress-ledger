"""Content hashing for prompt segments and state objects.

Real adapters hash text. Synthetic adapters use stable symbolic hashes
(e.g. "hash_shared_repo_v1") so toy traces stay inspectable.
"""
from __future__ import annotations

import hashlib


def segment_hash(text: str) -> str:
    """Stable, short content hash for a text segment. SHA_256 truncated to 16 hex chars."""
    if not isinstance(text, str):
        raise TypeError("segment_hash requires str")
    if not text:
        raise ValueError("segment_hash requires non_empty text")
    return "h_" + hashlib.sha256(text.encode("utf_8")).hexdigest()[:16]
