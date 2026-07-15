# -*- coding: utf-8 -*-
import os
from typing import Optional

from ..config_loader import get_yaml_loader

def _project_root() -> str:
    # storage.py is at cps/ai/storage.py, so go up two levels to get project root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def _resolve_path(path_value: str) -> str:
    if os.path.isabs(path_value):
        return os.path.abspath(path_value)
    return os.path.abspath(os.path.join(_project_root(), path_value))


def _configured_ai_storage_dir() -> Optional[str]:
    loader = get_yaml_loader()
    configured = loader.get("ai", "ai_storage_dir")
    if configured:
        return str(configured).strip()
    return os.environ.get("AI_STORAGE_DIR")


def get_ai_storage_dir() -> str:
    configured = _configured_ai_storage_dir()
    if configured:
        storage_dir = _resolve_path(configured)
    else:
        storage_dir = os.path.join(_project_root(), "ai_storage")
    os.makedirs(storage_dir, exist_ok=True)
    return storage_dir


def get_ai_db_path() -> str:
    return os.path.join(get_ai_storage_dir(), "ai.db")


def get_lancedb_dir() -> str:
    lancedb_dir = os.path.join(get_ai_storage_dir(), "lancedb")
    os.makedirs(lancedb_dir, exist_ok=True)
    return lancedb_dir
