# -*- coding: utf-8 -*-
"""Thin OpenAI-compatible model gateway.

Novel reasoning uses a deliberately small interface so another SDK can be
implemented without changing indexing or retrieval.  The default adapter uses
the OpenAI SDK and supports both Responses API and Chat Completions; structured
calls automatically fall back to Chat Completions for compatible vendors that
do not implement Responses yet.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

from openai import OpenAI

from .config import ModelSettings


log = logging.getLogger("calibre-web.ai")


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0


class ModelGateway(Protocol):
    def structured(
        self,
        *,
        system: str,
        prompt: str,
        schema_name: str,
        schema: Mapping[str, Any],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        ...

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        ...


def _json_text(value: str) -> Dict[str, Any]:
    text = (value or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    parsed = json.loads(text.strip())
    if not isinstance(parsed, dict):
        raise ValueError("Structured model response must be a JSON object")
    return parsed


class OpenAIModelGateway:
    def __init__(self, settings: ModelSettings):
        self.settings = settings
        kwargs: Dict[str, Any] = {
            "api_key": settings.api_key,
            "timeout": settings.timeout_seconds,
            "max_retries": settings.max_retries,
        }
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        self.client = OpenAI(**kwargs)
        self.last_usage = ModelUsage()

    def structured(
        self,
        *,
        system: str,
        prompt: str,
        schema_name: str,
        schema: Mapping[str, Any],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        selected_model = model or self.settings.reasoning_model
        if self.settings.api_mode == "responses":
            try:
                return self._responses_structured(
                    system=system,
                    prompt=prompt,
                    schema_name=schema_name,
                    schema=schema,
                    model=selected_model,
                )
            except Exception as exc:  # compatible providers often omit /responses
                if not self._is_capability_error(exc):
                    raise
                log.warning("Responses structured call failed; falling back to Chat Completions: %s", exc)
        return self._chat_structured(
            system=system,
            prompt=prompt,
            schema_name=schema_name,
            schema=schema,
            model=selected_model,
        )

    def _responses_structured(
        self,
        *,
        system: str,
        prompt: str,
        schema_name: str,
        schema: Mapping[str, Any],
        model: str,
    ) -> Dict[str, Any]:
        response = self.client.responses.create(
            model=model,
            instructions=system,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": dict(schema),
                }
            },
        )
        usage = getattr(response, "usage", None)
        self.last_usage = ModelUsage(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )
        return _json_text(getattr(response, "output_text", ""))

    def _chat_structured(
        self,
        *,
        system: str,
        prompt: str,
        schema_name: str,
        schema: Mapping[str, Any],
        model: str,
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        response_format: Dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": dict(schema),
            },
        }
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=response_format,
            )
        except Exception as exc:
            if not self._is_capability_error(exc):
                raise
            log.warning("json_schema unsupported; retrying with JSON mode: %s", exc)
            response = self.client.chat.completions.create(
                model=model,
                messages=messages + [
                    {"role": "user", "content": "Return one JSON object and no markdown."}
                ],
                response_format={"type": "json_object"},
            )
        usage = getattr(response, "usage", None)
        self.last_usage = ModelUsage(
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )
        content = response.choices[0].message.content or ""
        return _json_text(content)

    @staticmethod
    def _is_capability_error(exc: Exception) -> bool:
        if isinstance(exc, (AttributeError, NotImplementedError)):
            return True
        status = getattr(exc, "status_code", None)
        return status in {400, 404, 405, 415, 422}

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        normalized = [text if text.strip() else " " for text in texts]
        response = self.client.embeddings.create(
            model=self.settings.embedding_model,
            input=normalized,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]

    def embed_batches(self, texts: Sequence[str], batch_size: int) -> Iterable[List[List[float]]]:
        for start in range(0, len(texts), batch_size):
            yield self.embed(texts[start : start + batch_size])
