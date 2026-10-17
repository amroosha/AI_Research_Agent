"""Structured query expansion with retry and deterministic fallback."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field

Strategy = Literal["step_back", "specific", "fusion"]


class SingleQuery(BaseModel):
    query: str = Field(min_length=1)


class QueryList(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=5)


class RewriteResult(TypedDict):
    queries: list[str]
    llm_calls: int
    fallback: bool


def expand_queries(
    llm, question: str, n_variants: int = 3, strategy: Strategy = "fusion"
) -> RewriteResult:
    if not question.strip() or n_variants < 1:
        return {"queries": [question.strip() or "research"], "llm_calls": 0, "fallback": True}
    prompt = f"Create {n_variants} {strategy} search queries for: {question.strip()}"
    calls = 0
    for _ in range(2):
        calls += 1
        try:
            value = llm.with_structured_output(QueryList).invoke(prompt)
            queries = [str(q).strip() for q in value.queries if str(q).strip()]
            if queries:
                return {
                    "queries": list(dict.fromkeys(queries))[:n_variants],
                    "llm_calls": calls,
                    "fallback": False,
                }
        except (AttributeError, TypeError, ValueError):
            continue
    return {"queries": [question.strip()], "llm_calls": calls, "fallback": True}


def rewrite_query(llm, question: str, strategy: Strategy) -> str:
    return expand_queries(llm, question, 1, strategy)["queries"][0]
