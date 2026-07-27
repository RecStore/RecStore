"""Lookahead prefetch helpers.

Implementation lives in ``recstore.embedding_read_path``; this module re-exports
for existing model_zoo / bagpipe test imports.
"""

from __future__ import annotations

from recstore.embedding_read_path import LookaheadPrefetcher, PrefetchSlot

__all__ = ["LookaheadPrefetcher", "PrefetchSlot"]
