# -*- coding: utf-8 -*-
import copy
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from .ai_tools import AgentTool

from google import genai
from google.genai import types


log = logging.getLogger("calibre-web.ai")


class CalibreAgent:
    """
    Google GenAI 专用 Agent，直接使用 Gemini API 的对话格式（role + parts）。
    """

    DEFAULT_MODEL = "gemini-2.5-flash"
    MAX_TOOL_TURNS = 8

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        enable_web_search: bool = False,
    ):

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["client_options"] = {"api_endpoint": base_url.rstrip("/")}

        self.client = genai.Client(**client_kwargs)
        self.model = model or self.DEFAULT_MODEL
        self.enable_web_search = enable_web_search
        self.system_prompt = system_prompt or (
            "你是 Calibre-Web 的 AI 图书管家。你能够推荐书籍、概述内容、回答关于书籍的各种问题。"
            "需要查询书库时，请主动调用提供的工具。"
        )
        # 历史记录使用 Gemini 的标准结构：[{role: 'user'|'model', parts: [...]}, ...]
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # History helpers
    # ------------------------------------------------------------------ #
    def extend_history(self, messages: Optional[List[Dict[str, Any]]]):
        if not messages:
            return
        for msg in messages:
            self.history.append(self._normalize_message(msg))

    def build_user_message(self, text: str) -> Dict[str, Any]:
        return {"role": "user", "parts": [{"text": text}]}

    def append_message(self, message: Dict[str, Any]):
        self.history.append(self._normalize_message(message))

    # ------------------------------------------------------------------ #
    # Main chat loop
    # ------------------------------------------------------------------ #
    def chat(self):
        """
        触发 Gemini 一轮对话。调用方应在调用 chat() 之前将用户消息写入 history。
        该方法是生成器，便于前端流式消费。
        """
        turns = 0
        while turns < self.MAX_TOOL_TURNS:
            turns += 1
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=self._history_payload(),
                    config=self._build_generate_config(),
                )
            except Exception as exc:  # pylint: disable=broad-except
                log.error("Gemini 调用失败: %s", exc, exc_info=True)
                yield f"AI 接口调用失败: {exc}"
                return

            candidate = self._get_primary_candidate(response)
            if not candidate:
                yield getattr(response, "text", "AI 没有返回结果")
                return

            parts = self._get_candidate_parts(candidate)
            function_calls = self._extract_function_calls(parts)
            assistant_text = self._collect_text(parts)

            if function_calls:
                assistant_message = {"role": "model", "parts": self._parts_to_dicts(parts)}
                self.history.append(assistant_message)
                if assistant_text:
                    yield assistant_text
                self._execute_function_calls(function_calls)
                continue

            final_text = assistant_text or getattr(candidate, "text", None) or getattr(response, "text", None)
            if not final_text:
                final_text = "（未返回内容）"

            self.history.append({"role": "model", "parts": [{"text": final_text}]})
            yield final_text
            return

        yield "工具调用次数已达上限，无法完成请求。"

    # ------------------------------------------------------------------ #
    # Request construction helpers
    # ------------------------------------------------------------------ #
    def _build_generate_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {"system_instruction": self.system_prompt}
        tools = self._build_tools_payload()
        if tools:
            config["tools"] = tools
        return config

    def _build_tools_payload(self) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        declarations = AgentTool.get_function_declarations()
        if declarations:
            payload.append({"function_declarations": declarations})
        if self.enable_web_search:
            payload.append({"google_search": {}})
        return payload

    def _history_payload(self) -> List[types.Content]:
        # 深拷贝，避免 SDK 修改内部结构
        # old_contents = json.loads(json.dumps(self.history, ensure_ascii=False))
        contents = []
        for message in self.history:
            sanitized_parts = [self._strip_inline_file_path(part) for part in message["parts"]]
            contents.append(types.Content(role=message["role"], parts=sanitized_parts))
        return contents

    # ------------------------------------------------------------------ #
    # Response parsing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_primary_candidate(response: Any) -> Optional[Any]:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return None
        return candidates[0]

    @staticmethod
    def _get_candidate_parts(candidate: Any) -> List[Any]:
        if hasattr(candidate, "content") and getattr(candidate.content, "parts", None):
            return candidate.content.parts
        return getattr(candidate, "parts", []) or []

    def _extract_function_calls(self, parts: List[Any]) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        for part in parts:
            fc = self._function_call_from_part(part)
            if fc:
                calls.append(fc)
        return calls

    def _function_call_from_part(self, part: Any) -> Optional[Dict[str, Any]]:
        function_call = None
        if hasattr(part, "function_call"):
            function_call = getattr(part, "function_call")
        elif isinstance(part, dict):
            function_call = part.get("function_call")

        if not function_call:
            return None

        if hasattr(function_call, "model_dump"):
            data = function_call.model_dump()
        elif isinstance(function_call, dict):
            data = function_call
        else:
            data = {
                "id": getattr(function_call, "id", None),
                "name": getattr(function_call, "name", None),
                "args": getattr(function_call, "args", None),
            }

        if not data.get("name"):
            return None

        args = self._coerce_args_dict(data.get("args"))
        return {
            "id": data.get("id"),
            "name": data.get("name"),
            "args": args,
        }

    @staticmethod
    def _get_text_from_part(part: Any) -> Optional[str]:
        if hasattr(part, "text"):
            return part.text
        if isinstance(part, dict):
            return part.get("text")
        return None

    def _collect_text(self, parts: List[Any]) -> str:
        texts = [text for text in (self._get_text_from_part(p) for p in parts) if text]
        return "".join(texts).strip()

    # ------------------------------------------------------------------ #
    # Tool execution
    # ------------------------------------------------------------------ #
    def _execute_function_calls(self, function_calls: List[Dict[str, Any]]):
        for call in function_calls:
            name = call.get("name")
            args = call.get("args") or {}
            func = AgentTool.get_tool_func(name)

            if not func:
                payload = {"status": "error", "message": f"未找到工具 {name}"}
                image_info = None
            else:
                try:
                    result = func(**args)
                    payload, image_info = self._normalize_tool_output(result)
                except Exception as exc:  # pylint: disable=broad-except
                    log.exception("工具 %s 执行失败", name)
                    payload = {"status": "error", "message": str(exc)}
                    image_info = None

            response_part = {
                "function_response": {
                    "name": name,
                    "response": payload,
                }
            }
            if call.get("id"):
                response_part["function_response"]["id"] = call["id"]

            self.history.append({"role": "user", "parts": [response_part]})

            if image_info:
                self._append_image_message(image_info)

    def _normalize_tool_output(self, result: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if isinstance(result, str):
            try:
                payload = json.loads(result)
            except json.JSONDecodeError:
                payload = {"content": result}
        elif isinstance(result, dict):
            payload = copy.deepcopy(result)
        elif isinstance(result, list):
            payload = {"items": result}
        else:
            payload = {"content": str(result)}

        image_info = None
        if isinstance(payload, dict) and "_image_data" in payload:
            image_info = {
                "image_data" : payload.pop("_image_data"),
                "image_path" : payload.pop("_image_path")
            }

        if isinstance(payload, list):
            payload = {"items": payload}
        elif not isinstance(payload, dict):
            payload = {"result": payload}

        return payload, image_info


    # gemini的图像理解必须追加新的消息，不能直接保存到function response里
    def _append_image_message(self, image_info):
        self.history.append(
            {
                "role": "user",
                "parts": [
                    {"text": "这是获取到的图片。"},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_info["image_data"],
                            "file_path": image_info["image_path"],
                        }
                    },
                ],
            }
        )


    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #
    def _parts_to_dicts(self, parts: List[Any]) -> List[Dict[str, Any]]:
        serialized: List[Dict[str, Any]] = []
        for part in parts:
            if hasattr(part, "text") and part.text:
                serialized.append({"text": part.text})
                continue

            function_call = getattr(part, "function_call", None)
            if function_call:
                args = self._coerce_args_dict(getattr(function_call, "args", None))
                payload = {
                    "name": getattr(function_call, "name", None),
                    "args": args,
                }
                if getattr(function_call, "id", None):
                    payload["id"] = function_call.id
                serialized.append({"function_call": payload})
                continue

            function_response = getattr(part, "function_response", None)
            if function_response:
                response = getattr(function_response, "response", {}) or {}
                if hasattr(response, "model_dump"):
                    response = response.model_dump()
                payload = {
                    "name": getattr(function_response, "name", None),
                    "response": response,
                }
                if getattr(function_response, "id", None):
                    payload["id"] = function_response.id
                serialized.append({"function_response": payload})
                continue

            inline_data = getattr(part, "inline_data", None)
            if inline_data:
                serialized.append(
                    {
                        "inline_data": {
                            "mime_type": getattr(inline_data, "mime_type", "application/octet-stream"),
                            "data": getattr(inline_data, "data", ""),
                        }
                    }
                )
                continue

            file_data = getattr(part, "file_data", None)
            if file_data:
                serialized.append({"file_data": {"file_uri": getattr(file_data, "file_uri", None)}})
                continue

            # Fallback：直接尝试序列化
            try:
                serialized.append(json.loads(json.dumps(part)))
            except TypeError:
                serialized.append({"text": str(part)})

        return serialized

    def _normalize_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        role = (message or {}).get("role", "user")
        parts = message.get("parts") if isinstance(message, dict) else None
        if not isinstance(parts, list):
            parts = []
        normalized_parts = [self._normalize_part(part) for part in parts]
        return {"role": role, "parts": normalized_parts}

    def _normalize_part(self, part: Any) -> Dict[str, Any]:
        if not isinstance(part, dict):
            return {"text": str(part)}
        if "text" in part:
            return {"text": str(part["text"])}
        if "function_call" in part:
            fc = part["function_call"] or {}
            return {
                "function_call": {
                    "name": fc.get("name"),
                    "args": self._coerce_args_dict(fc.get("args")),
                    **({"id": fc["id"]} if fc.get("id") else {}),
                }
            }
        if "function_response" in part:
            fr = part["function_response"] or {}
            response = fr.get("response") or {}
            if isinstance(response, list):
                response = {"items": response}
            elif not isinstance(response, dict):
                response = {"content": response}
            payload = {
                "name": fr.get("name"),
                "response": response,
            }
            if fr.get("id"):
                payload["id"] = fr["id"]
            return {"function_response": payload}
        if "inline_data" in part:
            inline = part["inline_data"] or {}
            return {
                "inline_data": {
                    "mime_type": inline.get("mime_type", "application/octet-stream"),
                    "data": inline.get("data", ""),
                }
            }
        if "file_data" in part:
            file_data = part["file_data"] or {}
            return {"file_data": {"file_uri": file_data.get("file_uri")}}
        return {"text": json.dumps(part, ensure_ascii=False)}

    @staticmethod
    def _coerce_args_dict(args: Any) -> Dict[str, Any]:
        if args is None:
            return {}
        if isinstance(args, dict):
            return args
        if hasattr(args, "model_dump"):
            return args.model_dump()
        if hasattr(args, "to_dict"):
            return args.to_dict()
        if isinstance(args, str):
            try:
                data = json.loads(args)
                if isinstance(data, dict):
                    return data
                return {"value": data}
            except json.JSONDecodeError:
                return {"value": args}
        return {"value": args}

    def _strip_inline_file_path(self, obj: Any) -> Any:
        """
        Remove file_path fields from inline_data before sending to Gemini SDK.
        """
        if isinstance(obj, dict):
            cleaned: Dict[str, Any] = {}
            for key, value in obj.items():
                if key == "inline_data" and isinstance(value, dict):
                    inline_copy = {k: self._strip_inline_file_path(v) for k, v in value.items() if k != "file_path"}
                    cleaned[key] = inline_copy
                else:
                    cleaned[key] = self._strip_inline_file_path(value)
            return cleaned
        if isinstance(obj, list):
            return [self._strip_inline_file_path(item) for item in obj]
        return obj

