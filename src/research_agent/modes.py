"""Validated, immutable execution profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from research_agent.config import (
    DEFAULT_GRADER,
    GraderSettings,
    LLMSettings,
    RetrievalSettings,
    Settings,
)

Mode = Literal["fast", "deep"]


class AgentMode:
    FAST = "fast"
    DEEP = "deep"


@dataclass(frozen=True)
class ModeProfile:
    name: Mode
    routing: Settings
    llm: LLMSettings
    retrieval: RetrievalSettings
    grader: GraderSettings
    pre_expand: bool
    n_variants: int
    generate_k: int
    web_slots: int
    web_max_results: int

    def __post_init__(self) -> None:
        if len(set(self.routing.rewrite_order)) < self.routing.max_retries:
            raise ValueError("rewrite_order must contain at least max_retries unique strategies")
        if self.retrieval.final_k > self.retrieval.retrieve_k_each:
            raise ValueError("final_k cannot exceed retrieve_k_each")
        if self.generate_k < 1:
            raise ValueError("generate_k must be >= 1")
        if self.web_slots > self.generate_k:
            raise ValueError("web_slots cannot exceed generate_k")
        if self.web_slots < 0 or self.web_max_results < 0:
            raise ValueError("web counts cannot be negative")


FAST = ModeProfile(
    name="fast",
    routing=Settings(max_retries=1, rewrite_order=("step_back",)),
    llm=LLMSettings(num_ctx=8192),
    retrieval=RetrievalSettings(final_k=6),
    grader=DEFAULT_GRADER,
    pre_expand=False,
    n_variants=0,
    generate_k=6,
    web_slots=2,
    web_max_results=4,
)
DEEP = ModeProfile(
    name="deep",
    routing=Settings(max_retries=3, rewrite_order=("step_back", "specific", "fusion")),
    llm=LLMSettings(num_ctx=12288),
    retrieval=RetrievalSettings(final_k=10),
    grader=DEFAULT_GRADER,
    pre_expand=True,
    n_variants=3,
    generate_k=10,
    web_slots=3,
    web_max_results=5,
)


def get_profile(mode: Mode | str) -> ModeProfile:
    value = str(mode).lower()
    if value == "fast":
        return FAST
    if value == "deep":
        return DEEP
    raise ValueError(f"unknown mode: {mode!r}")


def coerce_mode(mode: Mode | str) -> str:
    return get_profile(mode).name


def run_metadata(profile: ModeProfile, retriever=None) -> dict[str, object]:
    """Build safe tracing metadata; never include questions or credentials."""

    model = getattr(profile.llm, "model", None) or "OLLAMA_MODEL"
    manifest = getattr(retriever, "corpus_version", None)
    return {
        "mode": profile.name,
        "model": model,
        "num_ctx": profile.llm.num_ctx,
        "corpus_version": manifest or "unknown",
        "grader_mode": profile.grader.mode,
        "grader_prompt": profile.grader.prompt_version,
        "rewrite_prompt": "rewrite_v1",
        "generate_prompt": "answer_v1",
        "relevant_threshold": profile.routing.relevant_threshold,
        "ambiguous_threshold": profile.routing.ambiguous_threshold,
        "max_retries": profile.routing.max_retries,
    }
