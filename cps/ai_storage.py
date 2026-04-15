# -*- coding: utf-8 -*-
import os
from typing import Optional

from .config_loader import get_yaml_loader

def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


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


def get_comorag_runtime_dir() -> str:
    # Preferred: unified AI storage root.
    configured_storage = _configured_ai_storage_dir()
    if configured_storage:
        runtime_dir = os.path.join(get_ai_storage_dir(), "runtime")
        os.makedirs(runtime_dir, exist_ok=True)
        return runtime_dir

    # Backward compatibility: allow legacy Comorag runtime override.
    loader = get_yaml_loader()
    legacy_runtime = loader.get("comorag", "runtime_dir") or os.environ.get("COMORAG_RUNTIME_DIR")
    if legacy_runtime:
        runtime_dir = _resolve_path(str(legacy_runtime).strip())
        os.makedirs(runtime_dir, exist_ok=True)
        return runtime_dir

    runtime_dir = os.path.join(get_ai_storage_dir(), "runtime")
    os.makedirs(runtime_dir, exist_ok=True)
    return runtime_dir
