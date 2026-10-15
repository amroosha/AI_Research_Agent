"""LLM capacity probe (Stage 2, T1).

Usage:
    python scripts/probe_llm.py [--ctx 4096 8192 12288 16384] [--structured]

For each context size, sends a needle-in-haystack question over raw HTTP
(/api/chat), reads GPU residency from /api/ps, and derives generation speed
from the response timings. Prints a table, writes data/llm_probe.json, and
recommends the largest qualifying ctx via choose_ctx(). Never edits config.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel, Field

from research_agent.config import DATA_DIR, DEFAULT_LLM, LLMSettings
from research_agent.llm import get_llm, require_model

NEEDLE = "AZURE-FALCON-4417"
NEEDLE_SENTENCE = f"The secret code word is {NEEDLE}."
NEEDLE_QUESTION = "What is the secret code word? Answer with the code word only."

DEFAULT_CTXS = (4096, 8192, 12288, 16384)
TARGET_TOKEN_RATIO = 0.75  # aim for the prompt to fill ~75% of num_ctx
MIN_TOKEN_RATIO = 0.60  # valid band: prompt_eval_count / num_ctx
MAX_TOKEN_RATIO = 0.90
CHARS_PER_TOKEN = 4.0  # first guess; calibrated from prompt_eval_count

# Neutral filler: deterministic, unrelated to the needle, no rare tokens.
_FILLER_SENTENCES: tuple[str, ...] = (
    "The morning light moved slowly across the quiet valley.",
    "Rain fell steadily on the rooftops and stopped by noon.",
    "A plain wooden desk sat beside the window of the small room.",
    "Maps of the region showed rivers, roads, and distant hills.",
    "The train arrived on time and passengers stepped onto the platform.",
    "Fresh bread cooled on the counter of the empty bakery.",
    "Children drew pictures with colored pencils at the long table.",
    "The library stayed open until the sun had set behind the trees.",
    "Garden tools hung in neat rows on the wall of the shed.",
    "A calm harbor held several small boats until the tide turned.",
)


class ProbeAnswer(BaseModel):
    """Structured-output schema for the --structured check: {word, ok}."""

    word: str = Field(description="The single requested word")
    ok: bool = Field(description="Whether the instruction was followed")


STRUCTURED_PROMPTS: tuple[str, ...] = (
    "Set word to 'ping' and ok to true because the instruction was followed.",
    "Set word to 'pong' and ok to true because the instruction was followed.",
    "Set word to 'north' and ok to true because the instruction was followed.",
    "Set word to 'copper' and ok to true because the instruction was followed.",
    "Set word to 'meadow' and ok to true because the instruction was followed.",
)


def build_filler(target_chars: int) -> str:
    """Return deterministic neutral text of at least ``target_chars`` characters."""
    if target_chars <= 0:
        return ""
    parts: list[str] = []
    total = 0
    index = 0
    while total < target_chars:
        sentence = _FILLER_SENTENCES[index % len(_FILLER_SENTENCES)]
        parts.append(sentence)
        total += len(sentence) + 1
        index += 1
    return " ".join(parts)


def build_needle_prompt(target_chars: int, depth: float = 0.5) -> str:
    """Filler with the needle sentence at ``depth`` of the text, plus the question."""
    filler = build_filler(target_chars)
    cut = int(len(filler) * depth)
    while cut < len(filler) and not filler[cut].isspace():
        cut += 1
    prompt = f"{filler[:cut].rstrip()} {NEEDLE_SENTENCE} {filler[cut:].lstrip()}"
    return f"{prompt}\n\n{NEEDLE_QUESTION}"


def parse_ps(payload: dict[str, Any], model: str) -> tuple[int | None, int | None]:
    """Return ``(size, size_vram)`` for ``model`` in a /api/ps payload.

    Returns ``(None, None)`` when the model is not loaded. A payload without
    ``size_vram`` (older Ollama) yields ``size_vram=0`` -> never fully on GPU.
    """
    for entry in payload.get("models", []):
        name = str(entry.get("name", ""))
        if name == model or name.startswith(model):
            size = int(entry.get("size") or 0)
            return size, int(entry.get("size_vram") or 0)
    return None, None


def fully_on_gpu(size: int | None, size_vram: int | None) -> bool:
    """True when at least 99% of the model bytes reside in VRAM."""
    if not size or size_vram is None:
        return False
    return size_vram >= 0.99 * size


def choose_ctx(rows: list[dict[str, Any]]) -> int | None:
    """Largest ctx whose row is valid, passed the needle, and sits on the GPU.

    Invalid, needle-failing, or CPU-offloaded rows are skipped; None when
    nothing qualifies.
    """
    qualifying = [
        row
        for row in rows
        if row.get("valid") and row.get("needle_pass") and row.get("fully_on_gpu")
    ]
    if not qualifying:
        return None
    return max(row["ctx"] for row in qualifying)


def _chat(
    client: httpx.Client, model: str, prompt: str, num_ctx: int, cfg: LLMSettings
) -> dict[str, Any]:
    """One non-streaming /api/chat call with the frozen determinism options."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "keep_alive": cfg.keep_alive,
        "options": {
            "num_ctx": num_ctx,
            "temperature": cfg.temperature,
            "seed": cfg.seed,
        },
    }
    response = client.post("/api/chat", json=payload)
    response.raise_for_status()
    return response.json()


def _ps(client: httpx.Client) -> dict[str, Any]:
    response = client.get("/api/ps")
    response.raise_for_status()
    return response.json()


def probe_ctx(client: httpx.Client, model: str, num_ctx: int, cfg: LLMSettings) -> dict[str, Any]:
    """Run needle + GPU + speed checks for one context size.

    The filler is calibrated against ``prompt_eval_count`` so the prompt lands
    at 60-90% of ``num_ctx`` (rescaled once when it misses); otherwise the row
    is marked invalid.
    """
    row: dict[str, Any] = {
        "ctx": num_ctx,
        "prompt_tokens": None,
        "needle_pass": False,
        "valid": False,
        "fully_on_gpu": False,
        "tokens_per_s": None,
        "load_s": None,
        "total_s": None,
        "answer": None,
        "error": None,
    }

    target_chars = int(TARGET_TOKEN_RATIO * num_ctx * CHARS_PER_TOKEN)
    data: dict[str, Any] | None = None
    for attempt in range(2):  # one calibration rescale, per plan
        prompt = build_needle_prompt(target_chars)
        try:
            data = _chat(client, model, prompt, num_ctx, cfg)
        except httpx.HTTPError as error:
            row["error"] = f"chat request failed: {error}"
            return row
        tokens = int(data.get("prompt_eval_count") or 0)
        row["prompt_tokens"] = tokens
        ratio = tokens / num_ctx if num_ctx else 0.0
        if tokens > 0 and MIN_TOKEN_RATIO <= ratio <= MAX_TOKEN_RATIO:
            row["valid"] = True
            break
        if tokens <= 0:
            row["error"] = "prompt_eval_count is 0; cannot calibrate"
            return row
        if attempt == 0:
            # Rescale the filler so the next attempt hits TARGET_TOKEN_RATIO.
            target_chars = int(target_chars * TARGET_TOKEN_RATIO / ratio)
            continue
        row["error"] = f"prompt tokens outside 60-90% of num_ctx after rescale ({tokens})"
    if data is None:
        return row

    answer = str((data.get("message") or {}).get("content") or "")
    row["answer"] = answer
    row["needle_pass"] = NEEDLE in answer

    eval_count = int(data.get("eval_count") or 0)
    eval_duration = int(data.get("eval_duration") or 0)  # ns
    row["tokens_per_s"] = eval_count / (eval_duration / 1e9) if eval_duration else None
    row["load_s"] = int(data.get("load_duration") or 0) / 1e9
    row["total_s"] = int(data.get("total_duration") or 0) / 1e9

    # GPU residency immediately after the call.
    try:
        size, size_vram = parse_ps(_ps(client), model)
    except httpx.HTTPError as error:
        row["error"] = f"ps request failed: {error}"
        size, size_vram = None, None
    row["size"] = size
    row["size_vram"] = size_vram
    row["fully_on_gpu"] = fully_on_gpu(size, size_vram)
    return row


def run_structured(llm: Any) -> tuple[bool, list[dict[str, Any]]]:
    """Five structured calls; report which parsed into ProbeAnswer."""
    chain = llm.with_structured_output(ProbeAnswer, method="json_schema")
    results: list[dict[str, Any]] = []
    all_ok = True
    for prompt in STRUCTURED_PROMPTS:
        try:
            parsed = chain.invoke(prompt)
            ok = isinstance(parsed, ProbeAnswer)
        except Exception as error:  # noqa: BLE001 - probe reports, never crashes
            ok = False
            results.append({"prompt": prompt, "ok": False, "error": str(error)})
            all_ok = False
            continue
        results.append({"prompt": prompt, "ok": ok, "parsed": parsed if ok else None})
        all_ok = all_ok and ok
    return all_ok, results


def print_table(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'ctx':>6} {'prompt':>8} {'needle':>7} {'gpu':>5} "
        f"{'tok/s':>8} {'load_s':>8} {'total_s':>8}  status"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        tokens = row["prompt_tokens"] if row["prompt_tokens"] is not None else "-"
        needle = "PASS" if row["needle_pass"] else "fail"
        gpu = "yes" if row["fully_on_gpu"] else "no"
        tps = f"{row['tokens_per_s']:.1f}" if row["tokens_per_s"] is not None else "-"
        load = f"{row['load_s']:.2f}" if row["load_s"] is not None else "-"
        total = f"{row['total_s']:.2f}" if row["total_s"] is not None else "-"
        status = "ok" if row["valid"] else f"invalid ({row['error']})"
        print(
            f"{row['ctx']:>6} {tokens:>8} {needle:>7} {gpu:>5} "
            f"{tps:>8} {load:>8} {total:>8}  {status}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe Ollama context sizes for needle/GPU/speed.")
    parser.add_argument(
        "--ctx",
        type=int,
        nargs="+",
        default=list(DEFAULT_CTXS),
        help="Context sizes to probe.",
    )
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Also make 5 structured-output calls and report parse success.",
    )
    args = parser.parse_args(argv)

    cfg = DEFAULT_LLM
    model = require_model()  # raises with the `ollama list` hint
    base_url = os.environ.get("OLLAMA_HOST", "").strip() or cfg.base_url

    rows: list[dict[str, Any]] = []
    with httpx.Client(base_url=base_url, timeout=cfg.request_timeout_s) as client:
        for num_ctx in args.ctx:
            print(f"probing num_ctx={num_ctx} ...", file=sys.stderr)
            rows.append(probe_ctx(client, model, num_ctx, cfg))

    print()
    print_table(rows)

    structured: dict[str, Any] | None = None
    if args.structured:
        structured_ok, structured_rows = run_structured(get_llm())
        structured = {"ok": structured_ok, "rows": structured_rows}
        print(f"\nstructured: {'5/5 parsed' if structured_ok else 'FAILED'}")

    chosen = choose_ctx(rows)
    print(f"\nrecommended ctx: {chosen if chosen is not None else 'NONE'}")

    report = {
        "model": model,
        "base_url": base_url,
        "generated_at": datetime.now(UTC).isoformat(),
        "chosen_ctx": chosen,
        "rows": rows,
        "structured": structured,
    }
    out_path = DATA_DIR / "llm_probe.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")

    if chosen is None:
        print(
            "ERROR: no context size qualifies (needle passed + fully on GPU + valid)",
            file=sys.stderr,
        )
        return 1
    if structured is not None and not structured["ok"]:
        print("ERROR: structured-output check failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
