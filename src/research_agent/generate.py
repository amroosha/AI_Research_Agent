"""Answer generation contract for the Stage 3 graph."""

from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel, Field


class AnswerResult(BaseModel):
    answer: str = Field(min_length=1)
    cited_chunk_ids: list[str] = Field(default_factory=list)


class GenerateResult(TypedDict):
    answer: str
    citations: list[int]
    status: str
    warnings: list[str]
    llm_calls: int


def generate_answer(
    llm, question: str, context: str, web_tried: bool, relevant_threshold: int
) -> GenerateResult:
    """Generate an answer, requiring numbered citations such as ``[1]``."""

    if not question.strip():
        raise ValueError("question must not be empty")
    prompt = (
        "Answer using only the context. Cite supporting passages with numbered "
        "citations [1], [2]. If evidence is insufficient, state that clearly.\n"
        f"Question: {question.strip()}\nWeb tried: {web_tried}\n"
        f"Relevant threshold: {relevant_threshold}\nContext:\n{context}"
    )
    try:
        result = llm.invoke(prompt)
        text = str(getattr(result, "content", result)).strip()
    except Exception as exc:  # noqa: BLE001
        return {
            "answer": "",
            "citations": [],
            "status": "error",
            "warnings": [str(exc)],
            "llm_calls": 1,
        }
    import re

    citations = [int(value) for value in re.findall(r"\[(\d+)\]", text)]
    return {
        "answer": text,
        "citations": list(dict.fromkeys(citations)),
        "status": "done" if text else "error",
        "warnings": [] if text else ["empty model response"],
        "llm_calls": 1,
    }
