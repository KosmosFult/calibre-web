from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging

from openai import OpenAI

try:
    from google import genai
except ImportError:  # pragma: no cover - optional dependency in some environments
    genai = None

logger = logging.getLogger(__name__)


def _is_gemini_endpoint(base_url: Optional[str], model_name: Optional[str]) -> bool:
    base = (base_url or "").lower()
    model = (model_name or "").lower()
    return ("generativelanguage.googleapis.com" in base) or ("gemini" in model)


def resolve_provider(
    provider: Optional[str],
    base_url: Optional[str],
    model_name: Optional[str],
) -> str:
    if provider:
        normalized = provider.strip().lower()
        if normalized in ("openai_compat", "google_genai"):
            return normalized
    if _is_gemini_endpoint(base_url=base_url, model_name=model_name):
        return "google_genai"
    return "openai_compat"


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: Optional[str] = None


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class ProviderClient:
    """
    Unified provider client for OpenAI-compatible and Google GenAI APIs.
    """

    def __init__(
        self,
        provider: Optional[str],
        api_key: Optional[str],
        base_url: Optional[str],
        model_name: Optional[str] = None,
        http_client: Any = None,
        max_retries: int = 2,
    ) -> None:
        self.provider = resolve_provider(provider, base_url, model_name)
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.http_client = http_client
        self.max_retries = max_retries

        if self.provider == "google_genai":
            if genai is None:
                raise ImportError(
                    "google-genai is required when provider is set to google_genai."
                )
            client_kwargs: Dict[str, Any] = {"api_key": api_key}
            if base_url:
                client_kwargs["client_options"] = {"api_endpoint": base_url.rstrip("/")}
            try:
                self._client = genai.Client(**client_kwargs)
            except TypeError:
                # Older google-genai versions do not support `client_options`.
                # Fall back to default endpoint instead of hard-failing.
                if "client_options" in client_kwargs:
                    logger.warning(
                        "google-genai Client does not support client_options in this environment; "
                        "falling back to default endpoint."
                    )
                    client_kwargs.pop("client_options", None)
                self._client = genai.Client(**client_kwargs)
        else:
            self._client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                http_client=http_client,
                max_retries=max_retries,
            )

    def chat_completion(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Any] = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        n: Optional[int] = None,
    ) -> ChatResult:
        if self.provider == "google_genai":
            return self._chat_completion_google_genai(
                model=model,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                response_format=response_format,
            )
        return self._chat_completion_openai_compat(
            model=model,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            response_format=response_format,
            seed=seed,
            n=n,
        )

    def embed(
        self,
        *,
        model: str,
        texts: List[str],
        output_dimensionality: Optional[int] = None,
    ) -> List[List[float]]:
        if self.provider == "google_genai":
            config = None
            if output_dimensionality is not None:
                config = {"output_dimensionality": int(output_dimensionality)}
            response = self._client.models.embed_content(model=model, contents=texts, config=config)
            embeddings = getattr(response, "embeddings", None) or []
            values: List[List[float]] = []
            for item in embeddings:
                if hasattr(item, "values"):
                    values.append(list(item.values))
                elif isinstance(item, dict):
                    values.append(list(item.get("values", [])))
            return values

        params: Dict[str, Any] = {"input": texts, "model": model}
        if output_dimensionality is not None:
            params["dimensions"] = int(output_dimensionality)
        response = self._client.embeddings.create(**params)
        return [list(item.embedding) for item in response.data]

    def _chat_completion_openai_compat(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_completion_tokens: Optional[int],
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        stop: Optional[Any],
        response_format: Optional[Dict[str, Any]],
        seed: Optional[int],
        n: Optional[int],
    ) -> ChatResult:
        params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if max_completion_tokens is not None:
            params["max_completion_tokens"] = max_completion_tokens
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if stop is not None:
            params["stop"] = stop
        if response_format is not None:
            params["response_format"] = response_format
        if seed is not None:
            params["seed"] = seed
        if n is not None:
            params["n"] = n

        response = self._client.chat.completions.create(**params)
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return ChatResult(
            text=text,
            prompt_tokens=_safe_int(getattr(usage, "prompt_tokens", 0)) if usage else 0,
            completion_tokens=_safe_int(getattr(usage, "completion_tokens", 0)) if usage else 0,
            finish_reason=getattr(response.choices[0], "finish_reason", None),
        )

    def _chat_completion_google_genai(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_completion_tokens: Optional[int],
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        stop: Optional[Any],
        response_format: Optional[Dict[str, Any]],
    ) -> ChatResult:
        contents: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            # GenAI only accepts user/model roles; map system to user.
            if role == "assistant":
                role = "model"
            elif role == "system":
                role = "user"
            content = msg.get("content", "")
            if isinstance(content, str):
                parts = [{"text": content}]
            elif isinstance(content, list):
                parts = content
            else:
                parts = [{"text": str(content)}]
            contents.append({"role": role, "parts": parts})

        config: Dict[str, Any] = {}
        output_tokens = max_completion_tokens or max_tokens
        if output_tokens is not None:
            config["max_output_tokens"] = output_tokens
        if temperature is not None:
            config["temperature"] = temperature
        if top_p is not None:
            config["top_p"] = top_p
        if stop is not None:
            config["stop_sequences"] = stop if isinstance(stop, list) else [stop]
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            config["response_mime_type"] = "application/json"

        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=config if config else None,
        )

        usage = getattr(response, "usage_metadata", None)
        return ChatResult(
            text=getattr(response, "text", "") or "",
            prompt_tokens=_safe_int(getattr(usage, "prompt_token_count", 0)) if usage else 0,
            completion_tokens=_safe_int(getattr(usage, "candidates_token_count", 0)) if usage else 0,
            finish_reason=None,
        )
