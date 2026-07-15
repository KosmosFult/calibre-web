# -*- coding: utf-8 -*-
"""Configuration for OpenAI-compatible model backends and novel reasoning."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from ..config_loader import get_yaml_loader


def _yaml(*keys: str, default: Any = None) -> Any:
    return get_yaml_loader().get(*keys, default=default)


def _env(name: str, fallback: Any = None) -> Any:
    value = os.environ.get(name)
    return fallback if value in (None, "") else value


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: Any, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ModelSettings:
    """Provider-neutral settings understood by the OpenAI Python SDK."""

    api_key: str
    base_url: Optional[str]
    chat_model: str
    extraction_model: str
    reasoning_model: str
    embedding_model: str
    api_mode: str = "responses"
    timeout_seconds: int = 120
    max_retries: int = 3


@dataclass(frozen=True)
class NovelSettings:
    chunk_chars: int = 2400
    chunk_overlap_paragraphs: int = 1
    extraction_batch_size: int = 4
    embedding_batch_size: int = 64
    arc_chapters: int = 8
    retrieval_limit: int = 14
    max_reasoning_rounds: int = 3
    graph_hops: int = 2
    keep_index_runs: int = 2
    auto_index_on_question: bool = True


def load_model_settings(require_api_key: bool = True) -> ModelSettings:
    section = ("ai", "openai")
    api_key = str(
        _yaml(*section, "api_key")
        or _env("OPENAI_API_KEY")
        or _env("AI_API_KEY")
        or ""
    ).strip()
    if require_api_key and not api_key:
        raise ValueError("Missing OpenAI-compatible API key (OPENAI_API_KEY or ai.openai.api_key).")

    base_url = _yaml(*section, "base_url") or _env("OPENAI_BASE_URL") or _env("AI_BASE_URL")
    if base_url:
        base_url = str(base_url).strip().rstrip("/")

    chat_model = str(
        _yaml(*section, "chat_model")
        or _yaml(*section, "model")
        or _env("OPENAI_MODEL")
        or "gpt-5.6-terra"
    )
    extraction_model = str(
        _yaml("ai", "novel", "extraction_model")
        or _env("NOVEL_EXTRACTION_MODEL")
        or chat_model
    )
    reasoning_model = str(
        _yaml("ai", "novel", "reasoning_model")
        or _env("NOVEL_REASONING_MODEL")
        or chat_model
    )
    embedding_model = str(
        _yaml("ai", "novel", "embedding_model")
        or _env("OPENAI_EMBEDDING_MODEL")
        or "text-embedding-3-small"
    )
    api_mode = str(
        _yaml(*section, "api_mode") or _env("OPENAI_API_MODE") or "responses"
    ).strip().lower()
    if api_mode not in {"responses", "chat_completions"}:
        raise ValueError("ai.openai.api_mode must be 'responses' or 'chat_completions'.")

    return ModelSettings(
        api_key=api_key,
        base_url=base_url,
        chat_model=chat_model,
        extraction_model=extraction_model,
        reasoning_model=reasoning_model,
        embedding_model=embedding_model,
        api_mode=api_mode,
        timeout_seconds=_as_int(
            _yaml(*section, "timeout_seconds", default=_env("OPENAI_TIMEOUT_SECONDS")),
            120,
        ),
        max_retries=_as_int(
            _yaml(*section, "max_retries", default=_env("OPENAI_MAX_RETRIES")),
            3,
            minimum=0,
        ),
    )


def load_novel_settings() -> NovelSettings:
    prefix = ("ai", "novel")
    return NovelSettings(
        chunk_chars=_as_int(_yaml(*prefix, "chunk_chars", default=_env("NOVEL_CHUNK_CHARS")), 2400, 400),
        chunk_overlap_paragraphs=_as_int(
            _yaml(*prefix, "chunk_overlap_paragraphs", default=_env("NOVEL_CHUNK_OVERLAP_PARAGRAPHS")),
            1,
            0,
        ),
        extraction_batch_size=_as_int(
            _yaml(*prefix, "extraction_batch_size", default=_env("NOVEL_EXTRACTION_BATCH_SIZE")), 4
        ),
        embedding_batch_size=_as_int(
            _yaml(*prefix, "embedding_batch_size", default=_env("NOVEL_EMBEDDING_BATCH_SIZE")), 64
        ),
        arc_chapters=_as_int(_yaml(*prefix, "arc_chapters", default=_env("NOVEL_ARC_CHAPTERS")), 8),
        retrieval_limit=_as_int(
            _yaml(*prefix, "retrieval_limit", default=_env("NOVEL_RETRIEVAL_LIMIT")), 14
        ),
        max_reasoning_rounds=_as_int(
            _yaml(*prefix, "max_reasoning_rounds", default=_env("NOVEL_MAX_REASONING_ROUNDS")), 3
        ),
        graph_hops=_as_int(_yaml(*prefix, "graph_hops", default=_env("NOVEL_GRAPH_HOPS")), 2),
        keep_index_runs=_as_int(
            _yaml(*prefix, "keep_index_runs", default=_env("NOVEL_KEEP_INDEX_RUNS")), 2
        ),
        auto_index_on_question=_as_bool(
            _yaml(*prefix, "auto_index_on_question", default=_env("NOVEL_AUTO_INDEX_ON_QUESTION")),
            True,
        ),
    )


def load_system_prompt() -> Optional[str]:
    return _yaml("ai", "openai", "system_prompt") or _env("OPENAI_SYSTEM_PROMPT")
