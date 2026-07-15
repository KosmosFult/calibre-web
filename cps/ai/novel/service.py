# -*- coding: utf-8 -*-
"""Application-facing façade for narrative indexing and reasoning."""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import load_model_settings, load_novel_settings
from ..providers import OpenAIModelGateway
from ..storage import get_ai_db_path, get_lancedb_dir
from .indexing import NarrativeIndexer
from .ingestion import resolve_book_path
from .reasoning import NovelReasoner
from .repository import NovelRepository
from .vector_store import NarrativeVectorStore


_repository: Optional[NovelRepository] = None
_repository_lock = threading.Lock()
_book_locks: Dict[int, threading.Lock] = {}


def get_repository() -> NovelRepository:
    global _repository  # pylint: disable=global-statement
    with _repository_lock:
        if _repository is None:
            _repository = NovelRepository(get_ai_db_path())
        return _repository


def _book_lock(book_id: int) -> threading.Lock:
    with _repository_lock:
        return _book_locks.setdefault(int(book_id), threading.Lock())


def get_index_status(book_id: int) -> Dict[str, Any]:
    return get_repository().status(int(book_id))


def trigger_index_build(book_id: int, force: bool = False) -> Tuple[bool, Dict[str, Any]]:
    repository = get_repository()
    with _book_lock(book_id):
        latest = repository.latest_run(book_id)
        if latest and latest["status"] in {"QUEUED", "RUNNING"}:
            return False, {
                "status": "running", "book_id": int(book_id),
                "message": "Narrative index build is already in progress",
                "job": latest,
            }
        try:
            models = load_model_settings(require_api_key=True)
            novel = load_novel_settings()
        except ValueError as exc:
            return False, {"status": "error", "book_id": int(book_id), "message": str(exc)}
        source_path = resolve_book_path(book_id)
        if not source_path:
            return False, {
                "status": "error", "book_id": int(book_id),
                "message": "No readable EPUB, KEPUB, or TXT source found for this book",
            }
        run_id = uuid.uuid4().hex
        repository.create_run(
            run_id=run_id, book_id=int(book_id), status="QUEUED", phase="QUEUED",
            message="Narrative index queued", llm_model=models.extraction_model,
            embedding_model=models.embedding_model, config=asdict(novel),
        )
        repository.append_log(run_id, book_id, "QUEUED", "Narrative index build queued")
        from ...services.worker import WorkerThread
        from ...tasks.novel_index import TaskNovelIndex
        WorkerThread.add(
            user="System",
            task=TaskNovelIndex(book_id=int(book_id), run_id=run_id, force=force, source_path=source_path),
            hidden=False,
        )
        return True, {
            "status": "queued", "book_id": int(book_id), "run_id": run_id,
            "message": "Narrative index build queued",
        }


def _perform_index_build(
    book_id: int,
    *,
    run_id: str,
    force: bool = False,
    source_path: Optional[str] = None,
    progress_callback: Optional[Callable[[str, str, float], None]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    repository = get_repository()
    try:
        models = load_model_settings(require_api_key=True)
        novel = load_novel_settings()
        path = source_path or resolve_book_path(book_id)
        if not path:
            raise ValueError("No readable EPUB, KEPUB, or TXT source found for this book")
        if not repository.get_run(run_id):
            repository.create_run(
                run_id=run_id, book_id=int(book_id), status="RUNNING", phase="START",
                message="Narrative index started", llm_model=models.extraction_model,
                embedding_model=models.embedding_model, config=asdict(novel),
            )
        gateway = OpenAIModelGateway(models)
        indexer = NarrativeIndexer(
            repository=repository,
            vector_store=NarrativeVectorStore(get_lancedb_dir()),
            gateway=gateway,
            model_settings=models,
            novel_settings=novel,
        )
        with _book_lock(book_id):
            result = indexer.build(
                book_id=int(book_id), run_id=run_id, source_path=path, force=force,
                progress_callback=progress_callback,
            )
        return True, result
    except Exception as exc:  # indexer persists phase and error details
        current = repository.get_run(run_id)
        if current and current.get("status") != "FAILED":
            repository.update_run(
                run_id, status="FAILED", phase="FAILED", message="Narrative index failed",
                last_error=str(exc),
            )
            repository.append_log(run_id, book_id, "FAILED", str(exc), "ERROR")
        return False, {"status": "error", "book_id": int(book_id), "message": str(exc)}


def ensure_indexed(book_id: int) -> Tuple[bool, Dict[str, Any]]:
    status = get_index_status(book_id)
    if status.get("indexed"):
        return True, status
    if status.get("status") == "running":
        return False, status
    return trigger_index_build(book_id, force=False)


def ask_book(book_id: int, question: str) -> Dict[str, Any]:
    question = str(question or "").strip()
    if not question:
        return {"status": "error", "message": "question is required", "book_id": int(book_id)}
    repository = get_repository()
    if not repository.active_run(book_id):
        if load_novel_settings().auto_index_on_question:
            ok, payload = trigger_index_build(book_id, force=False)
            if ok:
                payload["message"] = "Narrative index was queued; retry after it becomes ready"
            return payload
        return {"status": "not_indexed", "book_id": int(book_id), "message": "Build the narrative index first"}
    try:
        models = load_model_settings(require_api_key=True)
        reasoner = NovelReasoner(
            repository=repository,
            vector_store=NarrativeVectorStore(get_lancedb_dir()),
            gateway=OpenAIModelGateway(models),
            model_settings=models,
            novel_settings=load_novel_settings(),
        )
        return reasoner.answer(int(book_id), question)
    except Exception as exc:  # pylint: disable=broad-except
        return {"status": "error", "book_id": int(book_id), "message": str(exc)}


def get_chunk_by_order(book_id: int, order_id: int) -> Optional[Dict[str, Any]]:
    return get_repository().get_chunk(int(book_id), int(order_id))


def get_chunks_by_order_range(book_id: int, start_order_id: int, limit: int = 3) -> List[Dict[str, Any]]:
    return get_repository().chunk_range(int(book_id), int(start_order_id), int(limit))


def get_chunk_window(book_id: int, center_order_id: int, before: int = 1, after: int = 1) -> List[Dict[str, Any]]:
    start = max(0, int(center_order_id) - max(0, int(before)))
    limit = max(1, int(before) + int(after) + 1)
    return get_repository().chunk_range(int(book_id), start, limit)


def get_book_outline(book_id: int) -> Dict[str, Any]:
    return get_repository().outline(int(book_id))


def get_chapter_chunks(
    book_id: int,
    chapter_index: int,
    start_chunk_offset: int = 0,
    limit_chunks: int = 5,
) -> Dict[str, Any]:
    outline = get_book_outline(book_id)
    chapter = next(
        (item for item in outline.get("chapters", []) if item["chapter_index"] == int(chapter_index)),
        None,
    )
    if not chapter:
        return {"status": "empty", "book_id": int(book_id), "message": "Chapter not found"}
    start = chapter["start_order_id"] + max(0, int(start_chunk_offset))
    maximum = chapter["end_order_id"] - start + 1
    rows = get_repository().chunk_range(book_id, start, min(maximum, max(1, int(limit_chunks))))
    rows = [row for row in rows if row["chapter_index"] == int(chapter_index)]
    return {"status": "success", "book_id": int(book_id), "chapter": chapter, "chunks": rows}
