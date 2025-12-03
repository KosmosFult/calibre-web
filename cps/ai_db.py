# -*- coding: utf-8 -*-
import base64
import copy
import datetime
import json
import logging
import os
from typing import Any, Dict, List

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, scoped_session, sessionmaker

# AI 专用数据库文件
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "ai.db")
DB_URI = f"sqlite:///{DB_PATH}"

Base = declarative_base()
engine = create_engine(DB_URI, echo=False)
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

log = logging.getLogger("calibre-web.ai")


class AIChat(Base):
    __tablename__ = "ai_chats"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    title = Column(String(120), default="新的对话")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    messages = relationship("AIMessage", back_populates="chat", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "updated_at": self.updated_at.isoformat(),
        }


class AIMessage(Base):
    __tablename__ = "ai_chat_messages"

    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, ForeignKey("ai_chats.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user / model
    parts = Column(Text, nullable=False)  # JSON encoded list of parts
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    chat = relationship("AIChat", back_populates="messages")

    def as_content(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "parts": deserialize_parts(self.parts),
        }

    def visible_text(self) -> str:
        try:
            parts = json.loads(self.parts)
        except json.JSONDecodeError:
            return ""

        for part in parts:
            if isinstance(part, dict) and part.get("text"):
                return part["text"]
        return ""


def init_db():
    Base.metadata.create_all(engine)


def get_session():
    return Session()


def serialize_parts(parts: List[Dict[str, Any]]) -> str:
    safe_parts = copy.deepcopy(parts)
    safe_parts = _strip_inline_data(safe_parts)
    return json.dumps(safe_parts, ensure_ascii=False)


def deserialize_parts(parts_json: str) -> List[Dict[str, Any]]:
    if not parts_json:
        return []
    try:
        data = json.loads(parts_json)
        data = _restore_inline_data(data)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        return []


def _strip_inline_data(obj: Any) -> Any:
    """
    Remove inline image payloads when a file path exists so we only persist paths in DB.
    """
    if isinstance(obj, dict):
        cleaned: Dict[str, Any] = {}
        for key, value in obj.items():
            if key == "inline_data" and isinstance(value, dict):
                inline_copy = _strip_inline_data(value)
                if inline_copy.get("file_path"):
                    inline_copy.pop("data", None)
                cleaned[key] = inline_copy
            else:
                cleaned[key] = _strip_inline_data(value)
        return cleaned
    if isinstance(obj, list):
        return [_strip_inline_data(item) for item in obj]
    return obj


def _restore_inline_data(obj: Any) -> Any:
    """
    On load, hydrate inline images by reading them from disk using stored paths.
    """
    if isinstance(obj, dict):
        restored: Dict[str, Any] = {}
        for key, value in obj.items():
            if key == "inline_data" and isinstance(value, dict):
                restored_inline = dict(value)
                file_path = restored_inline.get("file_path")
                needs_data = file_path and not restored_inline.get("data")
                if needs_data:
                    try:
                        with open(file_path, "rb") as image_file:
                            restored_inline["data"] = base64.b64encode(image_file.read()).decode("utf-8")
                    except OSError as exc:
                        log.warning("无法读取图片文件 %s: %s", file_path, exc)
                restored[key] = restored_inline
            else:
                restored[key] = _restore_inline_data(value)
        return restored
    if isinstance(obj, list):
        return [_restore_inline_data(item) for item in obj]
    return obj


def history_for_chat(db_sess, chat_id: int) -> List[Dict[str, Any]]:
    messages = (
        db_sess.query(AIMessage)
        .filter(AIMessage.chat_id == chat_id)
        .order_by(AIMessage.created_at.asc())
        .all()
    )
    return [m.as_content() for m in messages]


def create_message(chat_id: int, role: str, parts: List[Dict[str, Any]]) -> AIMessage:
    return AIMessage(chat_id=chat_id, role=role, parts=serialize_parts(parts))


def message_to_public_dict(message: AIMessage) -> Dict[str, Any]:
    role = "assistant" if message.role == "model" else message.role
    text = message.visible_text()
    return {
        "role": role,
        "content": text,
        "text": text,
        "created_at": message.created_at.isoformat(),
    }
