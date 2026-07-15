# -*- coding: utf-8 -*-
"""OpenAI-compatible library agent with explicit local tool execution."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

from .config import ModelSettings
from .tools import AgentTool
from .tools import library as _library_tools  # noqa: F401  (register tool decorators)


log = logging.getLogger("calibre-web.ai")


DEFAULT_SYSTEM_PROMPT = """
你是 Calibre-Web 的 AI 图书管家，也是一名严谨的小说研究者。

工作原则：
1. 查询书库或书籍内容时主动使用工具，不得凭模型记忆编造本地书库信息。
2. 对剧情、人物关系、时间线、伏笔、因果链和推理结论，优先调用 ask_novel。它返回可核验的证据与 chunk 引用。
3. 证据不足时继续调用 read_book_window、read_book_segment 或 ask_novel 补查；明确区分原文事实、合理推断和未知信息。
4. 默认避免剧透。若用户没有明确要求详细剧情，在包含重大剧透前先给出提示；用户明确询问剧情谜底时可直接回答。
5. 不向用户暴露内部 book_id、工具协议或数据库结构，除非用户明确要求。
6. 不展示隐藏思维链。给出结论、简洁的证据链、原文位置引用和必要的不确定性即可。
""".strip()


class CalibreAgent:
    MAX_TOOL_TURNS = 10

    def __init__(self, settings: ModelSettings, system_prompt: Optional[str] = None):
        kwargs: Dict[str, Any] = {
            "api_key": settings.api_key,
            "timeout": settings.timeout_seconds,
            "max_retries": settings.max_retries,
        }
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        self.client = OpenAI(**kwargs)
        self.model = settings.chat_model
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.history: List[Dict[str, Any]] = []
        self.last_usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def extend_history(self, messages: Optional[Iterable[Dict[str, Any]]]) -> None:
        for message in messages or []:
            if isinstance(message, dict) and message.get("role"):
                self.history.append(dict(message))

    @staticmethod
    def build_user_message(text: str) -> Dict[str, Any]:
        return {"role": "user", "content": str(text)}

    def append_message(self, message: Dict[str, Any]) -> None:
        self.history.append(dict(message))

    def chat(self):
        """Run one tool-using turn and yield the final user-visible text."""
        for _turn in range(self.MAX_TOOL_TURNS):
            request: Dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "system", "content": self.system_prompt}] + self.history,
            }
            tools = AgentTool.openai_tools()
            if tools:
                request["tools"] = tools
                request["tool_choice"] = "auto"
            response = self.client.chat.completions.create(
                **request
            )
            self._capture_usage(getattr(response, "usage", None))
            message = response.choices[0].message
            serialized = self._serialize_assistant(message)
            self.history.append(serialized)

            tool_calls = serialized.get("tool_calls") or []
            if tool_calls:
                for call in tool_calls:
                    self.history.append(self._execute_tool(call))
                continue

            text = serialized.get("content") or "（模型未返回内容）"
            yield text
            return

        yield "工具调用次数已达上限；现有证据不足以完成请求。"

    def _capture_usage(self, usage: Any) -> None:
        if usage is None:
            return
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        self.last_usage = {
            "input_tokens": self.last_usage["input_tokens"] + input_tokens,
            "output_tokens": self.last_usage["output_tokens"] + output_tokens,
            "total_tokens": self.last_usage["total_tokens"] + input_tokens + output_tokens,
        }

    @staticmethod
    def _serialize_assistant(message: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "role": "assistant",
            "content": getattr(message, "content", None) or "",
        }
        calls = []
        for call in getattr(message, "tool_calls", None) or []:
            calls.append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments or "{}",
                    },
                }
            )
        if calls:
            payload["tool_calls"] = calls
        return payload

    @staticmethod
    def _execute_tool(call: Dict[str, Any]) -> Dict[str, Any]:
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        try:
            arguments = json.loads(function.get("arguments") or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            tool = AgentTool.get_tool_func(name)
            if tool is None:
                raise ValueError("Unknown tool: {}".format(name))
            result = tool(**arguments)
            content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        except Exception as exc:  # keep the model loop alive with a structured error
            log.exception("Agent tool %s failed", name)
            content = json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False)
        return {
            "role": "tool",
            "tool_call_id": call.get("id"),
            "name": name,
            "content": content,
        }
