"""Ollama client construction (Stage 2, T1).

The model tag comes from the ``OLLAMA_MODEL`` environment variable at call
time, so switching model tags never requires a config edit. No retries,
caching, or prompts live here.
"""

from __future__ import annotations

import os

from langchain_ollama import ChatOllama

from research_agent.config import DEFAULT_LLM, LLMSettings

MODEL_ENV_VAR = "OLLAMA_MODEL"
HOST_ENV_VAR = "OLLAMA_HOST"

_MISSING_MODEL_MESSAGE = (
    f"{MODEL_ENV_VAR} is not set. Run `ollama list` to see your installed model tags "
    'and set it, e.g. PowerShell: $env:OLLAMA_MODEL = "qwen2.5:3b-instruct-q4_K_M"'
)


def require_model() -> str:
    """Return the ``OLLAMA_MODEL`` tag, raising a helpful error when unset."""
    model = os.environ.get(MODEL_ENV_VAR, "").strip()
    if not model:
        raise ValueError(_MISSING_MODEL_MESSAGE)
    return model


def get_llm(cfg: LLMSettings = DEFAULT_LLM, **overrides) -> ChatOllama:
    """Build a ``ChatOllama`` with the frozen determinism defaults.

    ``model`` is read from ``OLLAMA_MODEL`` at call time (missing -> raises
    with an ``ollama list`` hint). ``OLLAMA_HOST`` overrides ``cfg.base_url``.
    ``overrides`` may replace any ChatOllama field; the probe uses it to vary
    ``num_ctx``. The HTTP client timeout comes from ``cfg.request_timeout_s``.
    """
    base_url = os.environ.get(HOST_ENV_VAR, "").strip() or cfg.base_url
    params: dict[str, object] = {
        "model": require_model(),
        "base_url": base_url,
        "num_ctx": cfg.num_ctx,
        "temperature": cfg.temperature,
        "seed": cfg.seed,
        "num_predict": cfg.num_predict,
        "keep_alive": cfg.keep_alive,
        # ChatOllama merges client_kwargs into the ollama Client, which passes
        # them to httpx; `timeout` is the httpx request timeout in seconds.
        "client_kwargs": {"timeout": cfg.request_timeout_s},
    }
    params.update(overrides)
    return ChatOllama(**params)
