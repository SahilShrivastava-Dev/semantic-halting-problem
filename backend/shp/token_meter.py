"""
token_meter.py

Process-wide accounting of LLM token usage, so the efficiency experiments can
report *real* token cost (not just round counts).

Why a single global meter instead of threading counts through every call:
every LLM request in this project — Writer, Critic, AND the RAGAS judge — flows
through the chat-model classes built in ``providers.py``. By recording usage at
that one chokepoint (``providers._metered_invoke`` / ``_FixedChatGroq``) we
capture all traffic without modifying RAGAS internals, which never surface token
counts otherwise.

Usage:
    from token_meter import METER

    METER.reset("writer")            # start a fresh scope
    ... run writer ...
    counts = METER.snapshot()        # {"writer": {...}, "critic": {...}, "judge": {...}}

The harness brackets each phase with ``with METER.scope("judge"): ...`` so usage
is attributed to the right role. ``record_usage`` is tolerant of providers that
omit usage metadata (returns silently), so it never crashes a run.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class _Counts:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, inp: int, out: int) -> None:
        self.input_tokens += inp
        self.output_tokens += out
        self.total_tokens += inp + out
        self.calls += 1

    def as_dict(self) -> Dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }


class TokenMeter:
    """
    Thread-safe token accumulator keyed by role ("writer" | "critic" | "judge").

    A single module-level instance (METER) is shared. The *current role* is held
    in a thread-local so nested LLM calls inside a ``scope()`` are attributed
    correctly even when libraries spin up worker threads inheriting the role.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._roles: Dict[str, _Counts] = {}
        self._current_role = "unattributed"

    def reset(self, *roles: str) -> None:
        """Clear counts. With no args clears everything; else clears named roles."""
        with self._lock:
            if not roles:
                self._roles.clear()
            else:
                for r in roles:
                    self._roles.pop(r, None)

    @contextmanager
    def scope(self, role: str):
        """Attribute every metered call inside the block to ``role``."""
        prev = self._current_role
        self._current_role = role
        try:
            yield
        finally:
            self._current_role = prev

    def record_usage(self, usage_metadata: dict | None, role: str | None = None) -> None:
        """
        Record one call's usage. Tolerant: a None/empty metadata dict (provider
        didn't report usage) is silently ignored rather than raising.
        """
        if not usage_metadata:
            return
        inp = int(usage_metadata.get("input_tokens", 0) or 0)
        out = int(usage_metadata.get("output_tokens", 0) or 0)
        if inp == 0 and out == 0:
            # Some providers report only total_tokens.
            total = int(usage_metadata.get("total_tokens", 0) or 0)
            if total == 0:
                return
            inp, out = total, 0
        key = role or self._current_role
        with self._lock:
            self._roles.setdefault(key, _Counts()).add(inp, out)

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        """Return a deep copy of current per-role counts."""
        with self._lock:
            return {k: v.as_dict() for k, v in self._roles.items()}

    def total_tokens(self) -> int:
        with self._lock:
            return sum(c.total_tokens for c in self._roles.values())


# Module-level singleton shared by providers.py and the experiment harness.
METER = TokenMeter()
