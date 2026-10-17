"""Dotenv loading and safe environment diagnostics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from research_agent.config import PROJECT_ROOT

ENV_PATH = PROJECT_ROOT / ".env"
KNOWN_VARS = {
    "OLLAMA_MODEL": {"description": "Ollama model tag", "required": True},
    "OLLAMA_HOST": {"description": "Ollama server URL", "required": False},
    "TAVILY_API_KEY": {"description": "Tavily API key", "required": False},
    "LANGSMITH_TRACING": {"description": "Enable LangSmith tracing", "required": False},
    "LANGSMITH_API_KEY": {"description": "LangSmith API key", "required": False},
    "LANGSMITH_PROJECT": {"description": "LangSmith project name", "required": False},
}


class EnvError(ValueError):
    """Raised when required configuration is unavailable."""


@dataclass(frozen=True)
class EnvSettings:
    ollama_model: str | None = None
    ollama_host: str | None = None
    web_search_url: str = "https://html.duckduckgo.com/html/"
    web_search_enabled: bool = True
    tavily_api_key: str | None = None


def load_env(path: str | Path = ENV_PATH) -> dict[str, str]:
    """Load known settings without overriding existing process variables."""

    load_dotenv(dotenv_path=Path(path), override=False)
    return {name: os.environ[name] for name in KNOWN_VARS if os.environ.get(name)}


def env_status(path: str | Path = ENV_PATH) -> dict[str, bool]:
    values = load_env(path)
    return {name: bool(values.get(name)) for name in KNOWN_VARS}


def require_env(name: str, environ: dict[str, str] | None = None) -> str:
    values = environ if environ is not None else load_env()
    value = values.get(name, "").strip()
    if not value:
        raise EnvError(f"{name} is not set; add it to .env")
    return value
