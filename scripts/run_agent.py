"""Backward-compatible entry point; use scripts/ask.py for new invocations."""

from __future__ import annotations

from ask import main

if __name__ == "__main__":
    raise SystemExit(main())
