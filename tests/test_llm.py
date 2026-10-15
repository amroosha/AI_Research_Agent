"""Tests for the Stage 2 T1 Ollama client and capacity probe (no network)."""

from __future__ import annotations

import pytest

from research_agent.config import DEFAULT_LLM, LLMSettings
from research_agent.llm import get_llm, require_model
from scripts.probe_llm import (
    NEEDLE,
    NEEDLE_QUESTION,
    ProbeAnswer,
    build_filler,
    build_needle_prompt,
    choose_ctx,
    fully_on_gpu,
    parse_ps,
)

MODEL_TAG = "qwen2.5:3b-instruct-q4_K_M"


@pytest.fixture(autouse=True)
def _model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", MODEL_TAG)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)


def test_get_llm_passes_expected_params() -> None:
    cfg = LLMSettings(
        base_url="http://localhost:11434",
        num_ctx=4096,
        temperature=0.0,
        seed=7,
        num_predict=512,
        keep_alive="10m",
        request_timeout_s=120.0,
    )
    llm = get_llm(cfg)
    assert llm.model == MODEL_TAG
    assert llm.base_url == "http://localhost:11434"
    assert llm.num_ctx == 4096
    assert llm.temperature == 0.0
    assert llm.seed == 7
    assert llm.num_predict == 512
    assert llm.keep_alive == "10m"
    assert llm.client_kwargs is not None
    assert llm.client_kwargs.get("timeout") == 120.0


def test_get_llm_honors_overrides() -> None:
    llm = get_llm(num_ctx=16384, seed=3)
    assert llm.num_ctx == 16384
    assert llm.seed == 3
    # Untouched fields still come from the frozen defaults.
    assert llm.temperature == DEFAULT_LLM.temperature
    assert llm.model == MODEL_TAG


def test_ollama_host_env_overrides_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:11434")
    assert get_llm().base_url == "http://gpu-box:11434"


def test_missing_model_raises_with_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    with pytest.raises(ValueError, match="ollama list"):
        require_model()
    with pytest.raises(ValueError, match="ollama list"):
        get_llm()


def test_parse_ps_found() -> None:
    payload = {
        "models": [
            {"name": "other:latest", "size": 100, "size_vram": 0},
            {"name": MODEL_TAG, "size": 2_000_000_000, "size_vram": 1_999_000_000},
        ]
    }
    assert parse_ps(payload, MODEL_TAG) == (2_000_000_000, 1_999_000_000)


def test_parse_ps_model_absent() -> None:
    payload = {"models": [{"name": "other:latest", "size": 100, "size_vram": 100}]}
    assert parse_ps(payload, MODEL_TAG) == (None, None)
    assert parse_ps({"models": []}, MODEL_TAG) == (None, None)


def test_parse_ps_missing_vram_field() -> None:
    payload = {"models": [{"name": MODEL_TAG, "size": 42}]}
    assert parse_ps(payload, MODEL_TAG) == (42, 0)


def test_fully_on_gpu_boundaries() -> None:
    assert fully_on_gpu(1000, 991) is True
    assert fully_on_gpu(1000, 989) is False
    assert fully_on_gpu(1000, 99) is False  # only 9.9% resident
    assert fully_on_gpu(1000, 0) is False
    assert fully_on_gpu(0, 0) is False
    assert fully_on_gpu(None, None) is False
    assert fully_on_gpu(99, 99) is True
    assert fully_on_gpu(99, 98) is False


def test_choose_ctx_largest_passing() -> None:
    rows = [
        {"ctx": 4096, "valid": True, "needle_pass": True, "fully_on_gpu": True},
        {"ctx": 8192, "valid": True, "needle_pass": True, "fully_on_gpu": True},
        {"ctx": 12288, "valid": True, "needle_pass": True, "fully_on_gpu": False},
        {"ctx": 16384, "valid": False, "needle_pass": False, "fully_on_gpu": True},
    ]
    assert choose_ctx(rows) == 8192


def test_choose_ctx_skips_invalid_and_cpu_only() -> None:
    rows = [
        {"ctx": 4096, "valid": True, "needle_pass": True, "fully_on_gpu": True},
        {"ctx": 8192, "valid": True, "needle_pass": False, "fully_on_gpu": True},
        {"ctx": 12288, "valid": False, "needle_pass": True, "fully_on_gpu": True},
        {"ctx": 16384, "valid": True, "needle_pass": True, "fully_on_gpu": False},
    ]
    assert choose_ctx(rows) == 4096


def test_choose_ctx_none_when_nothing_qualifies() -> None:
    assert choose_ctx([]) is None
    assert (
        choose_ctx([{"ctx": 4096, "valid": True, "needle_pass": True, "fully_on_gpu": False}])
        is None
    )
    assert (
        choose_ctx([{"ctx": 4096, "valid": False, "needle_pass": False, "fully_on_gpu": False}])
        is None
    )


def test_build_filler_length_and_determinism() -> None:
    filler = build_filler(1000)
    assert len(filler) >= 1000
    assert len(filler) <= 1300  # overshoot bounded by one sentence
    assert build_filler(1000) == filler
    assert NEEDLE not in filler
    assert build_filler(0) == ""
    assert build_filler(-5) == ""


def test_build_needle_prompt_placement() -> None:
    prompt = build_needle_prompt(4000)
    position = prompt.index(NEEDLE) / len(prompt)
    assert 0.35 <= position <= 0.60  # ~50% depth; question adds a tail
    assert prompt.count(NEEDLE) == 1
    assert prompt.endswith(NEEDLE_QUESTION)
    deep = build_needle_prompt(4000, depth=0.75)
    assert deep.index(NEEDLE) > prompt.index(NEEDLE)


@pytest.mark.slow
def test_determinism_and_structured_output() -> None:
    llm = get_llm()
    prompt = "Reply with exactly one word: supported"
    first = llm.invoke(prompt).content
    second = llm.invoke(prompt).content
    assert first == second

    chain = get_llm().with_structured_output(ProbeAnswer, method="json_schema")
    answer = chain.invoke("Set word to 'ready' and ok to true.")
    assert isinstance(answer, ProbeAnswer)
    assert answer.word  # parsed into the model
