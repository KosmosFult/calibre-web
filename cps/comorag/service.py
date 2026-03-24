import logging
import os
import sqlite3
import threading
import datetime
import uuid
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

from flask import has_app_context

from .. import ai_db
from ..ai_search.chunking import BookParser
from ..config_loader import get_yaml_loader
from .ComoRAG import ComoRAG
from .utils.config_utils import BaseConfig


log = logging.getLogger("calibre-web.ai")

_RAG_CACHE: Dict[int, ComoRAG] = {}
_BOOK_LOCKS: Dict[int, threading.Lock] = {}
_CACHE_LOCK = threading.Lock()


def _to_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _yaml_get(*keys, default=None):
    loader = get_yaml_loader()
    return loader.get(*keys, default=default)


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


def _run_with_app_context(func, *args, **kwargs):
    if has_app_context():
        return func(*args, **kwargs)
    from .. import app as flask_app
    with flask_app.app_context():
        return func(*args, **kwargs)


def _notify_progress(
    callback: Optional[Callable[[str, str, float], None]],
    phase: str,
    message: str,
    progress: float,
) -> None:
    if callback is None:
        return
    callback(phase, message, progress)


def _resolve_api_key() -> Optional[str]:
    return (
        _yaml_get("comorag", "api_key")
        or os.environ.get("COMORAG_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_GENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )


def _resolve_base_url() -> Optional[str]:
    return (
        _yaml_get("comorag", "base_url")
        or os.environ.get("COMORAG_BASE_URL")
        or os.environ.get("GENAI_BASE_URL")
        or os.environ.get("GOOGLE_GENAI_BASE_URL")
        or os.environ.get("GOOGLE_API_BASE_URL")
    )


def _build_config() -> BaseConfig:
    api_key = _resolve_api_key()
    base_url = _resolve_base_url()
    if not api_key:
        raise ValueError("Missing API key for ComoRAG. Set COMORAG_API_KEY or GEMINI_API_KEY.")

    llm_model = (
        _yaml_get("comorag", "llm_model")
        or os.environ.get("COMORAG_LLM_MODEL")
        or os.environ.get("GENAI_MODEL")
        or "gemini-3.1-flash-lite-preview"
    )
    embedding_model = (
        _yaml_get("comorag", "embedding_model")
        or os.environ.get("COMORAG_EMBEDDING_MODEL")
        or "gemini-embedding-2-preview"
    )
    need_cluster = _to_bool(
        _yaml_get("comorag", "need_cluster", default=os.environ.get("COMORAG_NEED_CLUSTER")),
        default=True,
    )
    openie_mode = _yaml_get("comorag", "openie_mode") or os.environ.get("COMORAG_OPENIE_MODE") or "online"

    cfg = BaseConfig()
    cfg.llm_name = llm_model
    cfg.llm_api_key = api_key
    cfg.llm_base_url = base_url
    cfg.llm_provider = "google_genai"

    cfg.embedding_model_name = embedding_model
    cfg.embedding_api_key = api_key
    cfg.embedding_base_url = base_url
    cfg.embedding_provider = "google_genai"

    cfg.need_cluster = need_cluster
    cfg.openie_mode = openie_mode
    cfg.save_openie = True
    cfg.is_mc = False
    return cfg


def _book_lock(book_id: int) -> threading.Lock:
    with _CACHE_LOCK:
        lock = _BOOK_LOCKS.get(book_id)
        if lock is None:
            lock = threading.Lock()
            _BOOK_LOCKS[book_id] = lock
        return lock


def _workspace_dir() -> str:
    configured = _yaml_get("comorag", "runtime_dir") or os.environ.get("COMORAG_RUNTIME_DIR")
    if configured:
        if os.path.isabs(configured):
            return configured
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        return os.path.abspath(os.path.join(base_dir, configured))
    return os.path.join(os.path.dirname(__file__), "runtime")


def _connect_db():
    return sqlite3.connect(ai_db.DB_PATH)


def _init_job_tables() -> None:
    with _connect_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_comorag_index_jobs (
                book_id INTEGER PRIMARY KEY,
                run_id TEXT,
                status TEXT NOT NULL,
                phase TEXT,
                message TEXT,
                total_chunks INTEGER NOT NULL DEFAULT 0,
                indexed_chunks INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                started_at TEXT,
                finished_at TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_comorag_index_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER NOT NULL,
                run_id TEXT,
                phase TEXT,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ai_comorag_index_logs_book_id
            ON ai_comorag_index_logs(book_id, id)
            """
        )
        conn.commit()


def _upsert_job(
    book_id: int,
    *,
    run_id: Optional[str] = None,
    status: Optional[str] = None,
    phase: Optional[str] = None,
    message: Optional[str] = None,
    total_chunks: Optional[int] = None,
    indexed_chunks: Optional[int] = None,
    last_error: Optional[str] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
) -> None:
    _init_job_tables()
    with _connect_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO ai_comorag_index_jobs (book_id, status, updated_at)
            VALUES (?, 'IDLE', ?)
            ON CONFLICT(book_id) DO NOTHING
            """,
            (int(book_id), _now_iso()),
        )

        fields = []
        values = []
        if run_id is not None:
            fields.append("run_id = ?")
            values.append(run_id)
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if phase is not None:
            fields.append("phase = ?")
            values.append(phase)
        if message is not None:
            fields.append("message = ?")
            values.append(message)
        if total_chunks is not None:
            fields.append("total_chunks = ?")
            values.append(int(total_chunks))
        if indexed_chunks is not None:
            fields.append("indexed_chunks = ?")
            values.append(int(indexed_chunks))
        if last_error is not None:
            fields.append("last_error = ?")
            values.append(last_error)
        if started_at is not None:
            fields.append("started_at = ?")
            values.append(started_at)
        if finished_at is not None:
            fields.append("finished_at = ?")
            values.append(finished_at)

        fields.append("updated_at = ?")
        values.append(_now_iso())
        values.append(int(book_id))

        cur.execute(
            f"UPDATE ai_comorag_index_jobs SET {', '.join(fields)} WHERE book_id = ?",
            tuple(values),
        )
        conn.commit()


def _append_job_log(book_id: int, run_id: Optional[str], phase: str, level: str, message: str) -> None:
    _init_job_tables()
    with _connect_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO ai_comorag_index_logs (book_id, run_id, phase, level, message, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (int(book_id), run_id, phase, level, message, _now_iso()),
        )
        conn.commit()


def _get_job_row(book_id: int) -> Optional[Dict[str, Any]]:
    _init_job_tables()
    with _connect_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT run_id, status, phase, message, total_chunks, indexed_chunks,
                   last_error, started_at, finished_at, updated_at
            FROM ai_comorag_index_jobs
            WHERE book_id = ?
            LIMIT 1
            """,
            (int(book_id),),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "run_id": row[0],
            "status": row[1],
            "phase": row[2],
            "message": row[3],
            "total_chunks": int(row[4] or 0),
            "indexed_chunks": int(row[5] or 0),
            "last_error": row[6],
            "started_at": row[7],
            "finished_at": row[8],
            "updated_at": row[9],
        }


def _get_recent_logs(book_id: int, limit: int = 30) -> List[Dict[str, Any]]:
    _init_job_tables()
    with _connect_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT phase, level, message, created_at
            FROM ai_comorag_index_logs
            WHERE book_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(book_id), int(limit)),
        )
        rows = cur.fetchall()
    rows.reverse()
    return [
        {
            "phase": r[0],
            "level": r[1],
            "message": r[2],
            "created_at": r[3],
        }
        for r in rows
    ]


def get_rag(book_id: int) -> ComoRAG:
    book_id = int(book_id)
    with _CACHE_LOCK:
        rag = _RAG_CACHE.get(book_id)
        if rag is not None:
            return rag

    os.makedirs(_workspace_dir(), exist_ok=True)
    cfg = _build_config()
    rag = ComoRAG(
        global_config=cfg,
        save_dir=_workspace_dir(),
        book_id=book_id,
    )
    with _CACHE_LOCK:
        _RAG_CACHE[book_id] = rag
    return rag


def _build_docs_for_book(book_id: int) -> List[str]:
    parser = BookParser(
        chunk_size=int(_yaml_get("comorag", "chunk_size") or os.environ.get("COMORAG_CHUNK_SIZE", "800")),
        chunk_overlap=int(_yaml_get("comorag", "chunk_overlap") or os.environ.get("COMORAG_CHUNK_OVERLAP", "200")),
        enable_contextual=_to_bool(
            _yaml_get("comorag", "enable_contextual", default=os.environ.get("COMORAG_ENABLE_CONTEXTUAL")),
            default=False,
        ),
    )
    file_path = parser._resolve_book_path(int(book_id))
    if not file_path:
        return []
    from ebooklib import epub
    book = epub.read_epub(file_path)
    chapter_docs = parser._extract_chapters(book)
    chunks = parser._chunk_chapters(book_id=int(book_id), chapters=chapter_docs)
    return [chunk.text for chunk in chunks]


def _count_ver_indexed(book_id: int) -> int:
    conn = sqlite3.connect(ai_db.DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(1)
            FROM ai_rag_meta_ver
            WHERE book_id = ? AND namespace = 'chunk'
            """,
            (int(book_id),),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def _fetch_ver_row_by_order(book_id: int, order_id: int) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(ai_db.DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT hash_id, content, order_index
            FROM ai_rag_meta_ver
            WHERE book_id = ? AND namespace = 'chunk' AND order_index = ?
            LIMIT 1
            """,
            (int(book_id), int(order_id)),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "hash_id": row[0],
            "content": row[1],
            "order_id": int(row[2]),
        }
    finally:
        conn.close()


def get_chunk_by_order(book_id: int, order_id: int) -> Optional[Dict[str, Any]]:
    return _fetch_ver_row_by_order(book_id=int(book_id), order_id=int(order_id))


def get_index_status(book_id: int) -> Dict[str, Any]:
    indexed_chunks = _count_ver_indexed(int(book_id))
    job = _get_job_row(int(book_id))
    job_status = (job or {}).get("status")
    indexed = indexed_chunks > 0 and job_status not in {"QUEUED", "RUNNING"}
    response = {
        "status": "running" if job_status in {"QUEUED", "RUNNING"} else ("ready" if indexed else "not_indexed"),
        "book_id": int(book_id),
        "total_chunks": (job or {}).get("total_chunks", indexed_chunks),
        "indexed_chunks": indexed_chunks,
        "indexed": indexed,
        "job": job,
        "logs": _get_recent_logs(int(book_id), limit=30),
    }
    return response


def _perform_index_build(
    book_id: int,
    run_id: str,
    force: bool = False,
    progress_callback: Optional[Callable[[str, str, float], None]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    phase = "CHECK_EXISTING"
    try:
        existing = _count_ver_indexed(int(book_id))
        _notify_progress(progress_callback, phase, "Checking existing index", 0.10)
        _upsert_job(
            int(book_id),
            run_id=run_id,
            status="RUNNING",
            phase=phase,
            message="Checking existing index",
            indexed_chunks=existing,
        )
        _append_job_log(int(book_id), run_id, phase, "INFO", f"existing indexed chunks={existing}")

        if existing > 0 and not force:
            _notify_progress(progress_callback, "DONE", "Index already exists", 1.0)
            _upsert_job(
                int(book_id),
                status="SUCCESS",
                phase="DONE",
                message="Index already exists",
                total_chunks=existing,
                indexed_chunks=existing,
                finished_at=_now_iso(),
            )
            _append_job_log(int(book_id), run_id, "DONE", "INFO", "Index already exists, skip rebuild")
            return True, get_index_status(int(book_id))

        if existing > 0 and force:
            _notify_progress(progress_callback, "DONE", "Manual cleanup required before rebuild", 1.0)
            _upsert_job(
                int(book_id),
                status="FAILED",
                phase="DONE",
                message="Manual cleanup required before rebuild",
                last_error="Index already exists; please manually clear comorag tables before force rebuild.",
                finished_at=_now_iso(),
            )
            _append_job_log(
                int(book_id),
                run_id,
                "DONE",
                "ERROR",
                "Force rebuild requested but automatic cleanup is disabled; clear comorag tables manually first.",
            )
            return False, {
                "status": "error",
                "message": "Index already exists; manually clear comorag tables before force rebuild.",
            }

        phase = "LOAD_SOURCE"
        _notify_progress(progress_callback, phase, "Loading source docs", 0.25)
        _upsert_job(int(book_id), status="RUNNING", phase=phase, message="Loading source docs")
        _append_job_log(int(book_id), run_id, phase, "INFO", "Start parsing source book")
        docs = _build_docs_for_book(int(book_id))
        if not docs:
            _notify_progress(progress_callback, "FAILED", "No readable docs available for this book", 1.0)
            _upsert_job(
                int(book_id),
                status="FAILED",
                phase="FAILED",
                message="No readable docs available for this book",
                last_error="No readable docs available for this book",
                finished_at=_now_iso(),
            )
            _append_job_log(int(book_id), run_id, phase, "ERROR", "No readable docs available for this book")
            return False, {"status": "error", "message": "No readable docs available for this book"}

        _notify_progress(progress_callback, phase, f"Loaded {len(docs)} chunks", 0.45)
        _upsert_job(int(book_id), total_chunks=len(docs), message=f"Loaded {len(docs)} chunks")
        _append_job_log(int(book_id), run_id, phase, "INFO", f"Loaded docs count={len(docs)}")

        phase = "INDEXING"
        _notify_progress(progress_callback, phase, "Building index", 0.70)
        _upsert_job(int(book_id), status="RUNNING", phase=phase, message="Building index")
        _append_job_log(int(book_id), run_id, phase, "INFO", "Start ComoRAG.index(docs)")
        rag = get_rag(int(book_id))
        rag.index(docs)

        phase = "FINALIZE"
        _notify_progress(progress_callback, phase, "Finalizing index", 0.95)
        current = _count_ver_indexed(int(book_id))
        _upsert_job(
            int(book_id),
            status="SUCCESS",
            phase="DONE",
            message="Index build completed",
            indexed_chunks=current,
            total_chunks=max(len(docs), current),
            finished_at=_now_iso(),
        )
        _notify_progress(progress_callback, "DONE", "Index build completed", 1.0)
        _append_job_log(int(book_id), run_id, phase, "INFO", f"Index build completed, indexed={current}")
        return True, get_index_status(int(book_id))
    except Exception as exc:  # pylint: disable=broad-except
        err_msg = f"{exc}"
        _notify_progress(progress_callback, "FAILED", "Index build failed", 1.0)
        _upsert_job(
            int(book_id),
            status="FAILED",
            phase="FAILED",
            message="Index build failed",
            last_error=err_msg,
            finished_at=_now_iso(),
        )
        _append_job_log(int(book_id), run_id, phase, "ERROR", err_msg)
        _append_job_log(int(book_id), run_id, "TRACEBACK", "ERROR", traceback.format_exc())
        return False, {"status": "error", "message": err_msg}


def trigger_index_build(book_id: int, force: bool = False) -> Tuple[bool, Dict[str, Any]]:
    lock = _book_lock(int(book_id))
    with lock:
        _init_job_tables()
        current = _get_job_row(int(book_id))
        if current and current.get("status") == "RUNNING":
            return False, {
                "status": "running",
                "message": "Index build already running",
                "book_id": int(book_id),
            }

        run_id = uuid.uuid4().hex
        _upsert_job(
            int(book_id),
            run_id=run_id,
            status="QUEUED",
            phase="QUEUED",
            message="Queued",
            last_error="",
            started_at=_now_iso(),
            finished_at="",
        )
        _append_job_log(int(book_id), run_id, "QUEUED", "INFO", "Index build queued")
        from ..services.worker import WorkerThread
        from ..tasks.comorag_index import TaskComoRAGIndex
        WorkerThread.add(
            user="System",
            task=TaskComoRAGIndex(book_id=int(book_id), run_id=run_id, force=force),
            hidden=False,
        )
        return True, {
            "status": "queued",
            "message": "Index build queued",
            "book_id": int(book_id),
            "run_id": run_id,
        }


def ensure_indexed(book_id: int) -> Tuple[bool, Dict[str, Any]]:
    lock = _book_lock(int(book_id))
    with lock:
        current = _get_job_row(int(book_id))
        if current and current.get("status") in {"QUEUED", "RUNNING"}:
            return False, {
                "status": "running",
                "message": "Index build already in progress",
                "book_id": int(book_id),
                "job": current,
            }
        run_id = uuid.uuid4().hex
        _upsert_job(
            int(book_id),
            run_id=run_id,
            status="RUNNING",
            phase="START",
            message="Synchronous ensure_indexed start",
            started_at=_now_iso(),
            finished_at="",
        )
        _append_job_log(int(book_id), run_id, "START", "INFO", "ensure_indexed called")
        ok, payload = _run_with_app_context(
            _perform_index_build,
            int(book_id),
            run_id=run_id,
            force=False,
        )
        return ok, payload


def ask_book(book_id: int, question: str) -> Dict[str, Any]:
    ok, index_info = ensure_indexed(int(book_id))
    if not ok:
        return index_info

    rag = get_rag(int(book_id))
    results = rag.try_answer([question])
    if not results:
        return {
            "status": "empty",
            "message": "ComoRAG did not return an answer",
            "index": index_info,
        }

    solution = results[0]
    return {
        "status": "success",
        "book_id": int(book_id),
        "question": question,
        "answer": solution.answer,
        "docs": solution.docs,
        "summary": solution.summary,
        "timeline": solution.timeline,
        "index": index_info,
    }
