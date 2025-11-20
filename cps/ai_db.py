# -*- coding: utf-8 -*-
import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session

# 定义专门的 AI 数据库路径
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "ai.db")
DB_URI = 'sqlite:///{}'.format(DB_PATH)

Base = declarative_base()
engine = create_engine(DB_URI, echo=False)
# 使用 scoped_session 保证线程安全
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

class AIChatSession(Base):
    __tablename__ = 'ai_chat_sessions'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False) # 关联 Calibre-Web 的 user_id
    title = Column(String(100), default="New Chat")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    messages = relationship("AIChatMessage", back_populates="session", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "updated_at": self.updated_at.isoformat()
        }

class AIChatMessage(Base):
    __tablename__ = 'ai_chat_messages'

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('ai_chat_sessions.id'), nullable=False)
    role = Column(String(20), nullable=False)  # system, user, assistant, tool
    content = Column(Text, nullable=True)
    
    # 新增字段：用于存储完整的工具调用链
    # tool_calls: 存储 assistant 发起的工具调用列表 (JSON string)
    tool_calls = Column(Text, nullable=True) 
    # tool_call_id: 存储 tool 消息对应的调用 ID (String)
    tool_call_id = Column(String(100), nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    session = relationship("AIChatSession", back_populates="messages")

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            # 前端暂时不需要 tool_calls 细节，如果需要展示“正在调用搜索...”，可以加上
            "created_at": self.created_at.isoformat()
        }

def init_db():
    """初始化数据库表"""
    Base.metadata.create_all(engine)

def get_session():
    """获取一个新的数据库会话"""
    return Session()
