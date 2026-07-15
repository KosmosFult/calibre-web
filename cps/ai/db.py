# -*- coding: utf-8 -*-
"""Conversation persistence in the AI-only SQLite database."""

from __future__ import annotations

import datetime
import json
import os
from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, scoped_session, sessionmaker

from .storage import get_ai_db_path


DB_PATH = get_ai_db_path()
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
DB_URI = "sqlite:///{}".format(DB_PATH)

Base = declarative_base()
engine = create_engine(DB_URI, echo=False, connect_args={"check_same_thread": False})
Session = scoped_session(sessionmaker(bind=engine))


class AIChat(Base):
    __tablename__ = "ai_conversations"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    title = Column(String(120), default="新的对话")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    messages = relationship("AIMessage", back_populates="chat", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "updated_at": self.updated_at.isoformat(),
        }


class AIMessage(Base):
    __tablename__ = "ai_conversation_messages"

    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, ForeignKey("ai_conversations.id"), nullable=False, index=True)
    role = Column(String(24), nullable=False)
    content = Column(Text, nullable=False, default="")
    payload = Column(Text, nullable=False)
    visible = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    chat = relationship("AIChat", back_populates="messages")

    def as_openai_message(self) -> Dict[str, Any]:
        try:
            value = json.loads(self.payload)
        except (TypeError, json.JSONDecodeError):
            value = {"role": self.role, "content": self.content}
        return value if isinstance(value, dict) else {"role": self.role, "content": self.content}

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "text": self.content,
            "created_at": self.created_at.isoformat(),
        }


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session():
    return Session()


def history_for_chat(db_sess, chat_id: int) -> List[Dict[str, Any]]:
    rows = (
        db_sess.query(AIMessage)
        .filter(AIMessage.chat_id == int(chat_id))
        .order_by(AIMessage.id.asc())
        .all()
    )
    return [row.as_openai_message() for row in rows]


def create_message(chat_id: int, message: Dict[str, Any], visible: Optional[bool] = None) -> AIMessage:
    role = str(message.get("role") or "user")
    content = message.get("content")
    if isinstance(content, list):
        content_text = "\n".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("text")
        )
    else:
        content_text = str(content or "")
    if visible is None:
        visible = role in {"user", "assistant"} and bool(content_text)
    return AIMessage(
        chat_id=int(chat_id),
        role=role,
        content=content_text,
        payload=json.dumps(message, ensure_ascii=False),
        visible=bool(visible),
    )
