"""Chunking constants (Stage 1, T4).

Shared settings (thresholds, token budgets, model name) live in the top-level
``research_agent.config.IngestSettings``; this module holds chunking-specific
text constants only.
"""

from __future__ import annotations

import re

# Split prose into sentences on whitespace following sentence-final punctuation.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Paragraph-final punctuation that keeps a page break from merging.
PARA_END = tuple(".?!:;)\"'")
# Fence delimiter for code blocks.
CODE_FENCE = "```"
