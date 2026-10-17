"""LLM relevance grader (Stage 2, T5).

``grade_chunks`` scores retrieved chunks against a question with the Ollama
model via structured output (JSON schema in Ollama's ``format`` field, the
langchain-ollama ``json_schema`` method verified in T1). Chunks are graded in
batches of up to ``GraderSettings.grader_batch``; a batch whose estimated
prompt tokens would not fit ``num_ctx - num_predict - token_margin`` is split
into smaller batches. Malformed model output is retried once, then graded
chunk-by-chunk, and finally defaulted to score 0 and counted as a parse
failure. No retries/caching/router logic live here beyond that fallback.
"""

from __future__ import annotations

import logging
import math
import time
from importlib import resources

from pydantic import BaseModel

from research_agent.config import DEFAULT_GRADER, DEFAULT_LLM, GraderSettings, LLMSettings
from research_agent.state import Chunk

logger = logging.getLogger(__name__)

# Rough chars-per-token ratio used for the token-budget estimate.
CHARS_PER_TOKEN = 3.5
# Fixed overhead for the prompt template + instructions + question framing.
_PROMPT_OVERHEAD_TOKENS = 200


class GradeBatch(BaseModel):
    """Structured-output schema returned by the model for one batch.

    Deliberately flat (two parallel arrays, not a list of objects): the frozen
    3B model (qwen2.5:3b) reliably fills ``refs``/``scores`` arrays under
    json_schema but returns an empty array for nested object arrays. Refs are
    the 1-based labels shown in the prompt, never long chunk ids.
    """

    refs: list[int]
    scores: list[int]


def estimate_tokens(text: str) -> int:
    """ceil(len(text) / 3.5): the frozen rough token estimate."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def load_prompt(version: str) -> str:
    """Load the versioned prompt template ``prompts/{version}.txt``."""
    return (
        resources.files("research_agent.prompts")
        .joinpath(f"{version}.txt")
        .read_text(encoding="utf-8")
    )


def build_prompt(question: str, chunks: list[Chunk], template: str) -> str:
    """Fill the template: label the chunks ``[1]``..``[n]`` in order."""
    passages = "\n\n".join(
        f"[{index}]\n{chunk.text}" for index, chunk in enumerate(chunks, start=1)
    )
    return template.format(question=question.strip(), passages=passages)


def group_tokens(question: str, chunks: list[Chunk]) -> int:
    """Estimated prompt tokens for grading this group (overhead + question + texts)."""
    return (
        _PROMPT_OVERHEAD_TOKENS
        + estimate_tokens(question)
        + sum(estimate_tokens(chunk.text) for chunk in chunks)
    )


def fits_budget(question: str, chunks: list[Chunk], llm_cfg: LLMSettings, margin: int) -> bool:
    """True when the group's estimated prompt tokens fit the context budget."""
    budget = llm_cfg.num_ctx - llm_cfg.num_predict - margin
    return group_tokens(question, chunks) <= budget


def build_batches(
    question: str,
    chunks: list[Chunk],
    cfg: GraderSettings = DEFAULT_GRADER,
    llm_cfg: LLMSettings = DEFAULT_LLM,
) -> list[list[Chunk]]:
    """Group chunks into batches of <= grader_batch that fit the token budget.

    Starts from consecutive groups of ``grader_batch``; any group that would
    overflow ``num_ctx - num_predict - token_margin`` is split in half
    recursively (a single oversized chunk is still sent — it cannot be split
    further).
    """
    groups = [
        chunks[start : start + cfg.grader_batch]
        for start in range(0, len(chunks), cfg.grader_batch)
    ]
    final: list[list[Chunk]] = []
    for group in groups:
        stack = [group]
        while stack:
            candidate = stack.pop(0)
            if len(candidate) > 1 and not fits_budget(
                question, candidate, llm_cfg, cfg.token_margin
            ):
                middle = len(candidate) // 2
                stack.insert(0, candidate[middle:])
                stack.insert(0, candidate[:middle])
            else:
                final.append(candidate)
    return final


def validate_grades(refs: list[int], scores: list[int], expected_refs: set[int]) -> dict[int, int]:
    """Map ref -> score, rejecting duplicates, missing/extra refs, bad scores.

    ``refs``/``scores`` are the parallel arrays returned by the model (flat
    schema). Raises ``ValueError`` when the refs are not exactly
    ``expected_refs`` (each exactly once) or a score is not an ``int`` in
    0..100 (``bool`` is rejected despite being an ``int`` subclass).
    """
    if len(refs) != len(scores):
        raise ValueError(f"refs ({len(refs)}) and scores ({len(scores)}) have different lengths")
    mapping: dict[int, int] = {}
    for ref, score in zip(refs, scores):
        if type(score) is not int or not (0 <= score <= 100):
            raise ValueError(f"score {score!r} for ref {ref} is not an int in 0..100")
        if ref in mapping:
            raise ValueError(f"ref {ref} appears more than once")
        mapping[ref] = score
    if set(mapping) != expected_refs:
        missing = expected_refs - set(mapping)
        extra = set(mapping) - expected_refs
        raise ValueError(f"refs mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    return mapping


def _grade_batch(
    llm,
    question: str,
    batch: list[Chunk],
    template: str,
    latencies: list[float] | None,
) -> dict[int, int]:
    """One structured call with a single retry; raises on the second failure."""
    expected = set(range(1, len(batch) + 1))
    prompt = build_prompt(question, batch, template)
    structured = llm.with_structured_output(GradeBatch)
    last_error: Exception | None = None
    for attempt in range(2):
        started = time.perf_counter()
        try:
            result = structured.invoke(prompt)
        except Exception as exc:  # noqa: BLE001 - any call failure triggers the fallback
            last_error = exc
            logger.warning("grader batch call failed (attempt %d): %s", attempt + 1, exc)
            continue
        finally:
            if latencies is not None:
                latencies.append(time.perf_counter() - started)
        try:
            return validate_grades(result.refs, result.scores, expected)
        except ValueError as exc:
            last_error = exc
            logger.warning("grader batch output invalid (attempt %d): %s", attempt + 1, exc)
    raise RuntimeError(f"grader batch failed twice: {last_error}")


def grade_chunks(
    llm,
    question: str,
    chunks: list[Chunk],
    cfg: GraderSettings = DEFAULT_GRADER,
    *,
    llm_cfg: LLMSettings = DEFAULT_LLM,
    parse_failures: list[str] | None = None,
    latencies: list[float] | None = None,
) -> list[Chunk]:
    """Score each chunk 0-100 for relevance to ``question``.

    Returns NEW ``Chunk`` objects (``model_copy(update={"score": ...})``) in
    input order; inputs are never mutated. A batch that fails validation twice
    is graded chunk-by-chunk (one retry each); a single chunk that still fails
    gets score 0 and its id is appended to ``parse_failures`` (when given).
    ``latencies`` (when given) receives the wall time of every LLM call.
    """
    if not chunks:
        return []
    template = load_prompt(cfg.prompt_version)
    scores: dict[str, int] = {}
    for batch in build_batches(question, chunks, cfg, llm_cfg):
        try:
            mapping = _grade_batch(llm, question, batch, template, latencies)
        except RuntimeError:
            mapping = {}
            for chunk in batch:
                try:
                    single = _grade_batch(llm, question, [chunk], template, latencies)
                except RuntimeError:
                    logger.warning("grader failed for chunk %s; defaulting to score 0", chunk.id)
                    if parse_failures is not None:
                        parse_failures.append(chunk.id)
                    scores[chunk.id] = 0
                    continue
                scores[chunk.id] = single[1]
            continue
        for index, chunk in enumerate(batch, start=1):
            scores[chunk.id] = mapping[index]
    return [chunk.model_copy(update={"score": scores[chunk.id]}) for chunk in chunks]
