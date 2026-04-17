import logging
import os
import sqlite3
import threading
import datetime
import uuid
import traceback
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from flask import has_app_context

from .. import ai_db
from ..ai_storage import get_comorag_runtime_dir
from ..ai_search.chunking import BookParser
from ..config_loader import get_yaml_loader
from .ComoRAG import ComoRAG
from .utils.config_utils import BaseConfig


log = logging.getLogger("calibre-web.ai")

_RAG_CACHE: Dict[int, ComoRAG] = {}
_BOOK_LOCKS: Dict[int, threading.Lock] = {}
_CACHE_LOCK = threading.Lock()
ORDER_ID_PATTERN = re.compile(r"\[order_id=([0-9,\s]+)\]")


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


def _normalize_chapter_indices(chapter_indices: Optional[Any]) -> Optional[List[int]]:
    if chapter_indices is None or chapter_indices == "":
        return None
    if isinstance(chapter_indices, str):
        raw = chapter_indices.strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = [part.strip() for part in raw.split(",")]
        chapter_indices = parsed
    if isinstance(chapter_indices, (int, float)):
        chapter_indices = [int(chapter_indices)]
    if not isinstance(chapter_indices, (list, tuple, set)):
        raise ValueError("chapter_indices must be a list of chapter indexes or a comma-separated string")

    normalized = []
    for value in chapter_indices:
        if value is None or value == "":
            continue
        normalized.append(int(value))
    return sorted(set(normalized)) or None


def _chapter_scope_label(chapter_indices: Optional[List[int]]) -> str:
    if not chapter_indices:
        return "full book"
    preview = ",".join(str(v) for v in chapter_indices[:10])
    if len(chapter_indices) > 10:
        preview += ",..."
    return f"chapters[{preview}]"


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
    embedding_normalize = _to_bool(
        _yaml_get("comorag", "embedding_normalize", default=os.environ.get("COMORAG_EMBEDDING_NORMALIZE")),
        default=True,
    )
    embedding_output_dim_raw = (
        _yaml_get("comorag", "embedding_output_dim")
        or os.environ.get("COMORAG_EMBEDDING_OUTPUT_DIM")
    )

    cfg = BaseConfig()
    cfg.llm_name = llm_model
    cfg.llm_api_key = api_key
    cfg.llm_base_url = base_url
    cfg.llm_provider = "google_genai"

    cfg.embedding_model_name = embedding_model
    cfg.embedding_api_key = api_key
    cfg.embedding_base_url = base_url
    cfg.embedding_provider = "google_genai"
    cfg.embedding_return_as_normalized = embedding_normalize
    cfg.embedding_output_dim = int(embedding_output_dim_raw) if embedding_output_dim_raw not in (None, "") else None

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
    return get_comorag_runtime_dir()


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


def _build_chunk_rows_for_book(book_id: int, chapter_indices: Optional[List[int]] = None) -> List[Dict[str, Any]]:
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
    normalized_indices = _normalize_chapter_indices(chapter_indices)
    if normalized_indices is not None:
        allowed = set(normalized_indices)

        # for test
        if book_id == 19:
            allowed = {2, 3, 9}
            target_pos = chapter_docs[9]['text'].find("。……接下来")
            chapter_docs[9]['text'] = chapter_docs[9]['text'][:target_pos]

        chapter_docs = [chapter for chapter in chapter_docs if int(chapter.get("index", -1)) in allowed]

    chunks = parser._chunk_chapters(book_id=int(book_id), chapters=chapter_docs)
    return [chunk.to_rag_row() for chunk in chunks]


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
            SELECT hash_id, content, order_index, chapter_id, chapter_index, chapter_title,
                   chunk_index_in_chapter, word_count, char_count
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
            "chapter_id": row[3],
            "chapter_index": row[4],
            "chapter_title": row[5],
            "chunk_index_in_chapter": row[6],
            "word_count": int(row[7] or 0),
            "char_count": int(row[8] or 0),
        }
    finally:
        conn.close()


def get_chunk_by_order(book_id: int, order_id: int) -> Optional[Dict[str, Any]]:
    return _fetch_ver_row_by_order(book_id=int(book_id), order_id=int(order_id))


def get_chunks_by_order_range(book_id: int, start_order_id: int, limit: int = 3) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(ai_db.DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT hash_id, content, order_index, chapter_id, chapter_index, chapter_title,
                   chunk_index_in_chapter, word_count, char_count
            FROM ai_rag_meta_ver
            WHERE book_id = ? AND namespace = 'chunk' AND order_index >= ?
            ORDER BY order_index ASC
            LIMIT ?
            """,
            (int(book_id), int(start_order_id), max(1, int(limit))),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "hash_id": row[0],
            "content": row[1],
            "order_id": int(row[2]),
            "chapter_id": row[3],
            "chapter_index": row[4],
            "chapter_title": row[5],
            "chunk_index_in_chapter": row[6],
            "word_count": int(row[7] or 0),
            "char_count": int(row[8] or 0),
        }
        for row in rows
    ]


def get_chunk_window(book_id: int, center_order_id: int, before: int = 1, after: int = 1) -> List[Dict[str, Any]]:
    start_order_id = max(0, int(center_order_id) - max(0, int(before)))
    limit = max(1, int(before) + int(after) + 1)
    rows = get_chunks_by_order_range(book_id=int(book_id), start_order_id=start_order_id, limit=limit)
    end_order_id = int(center_order_id) + max(0, int(after))
    return [row for row in rows if row["order_id"] <= end_order_id]


def get_book_outline(book_id: int) -> Dict[str, Any]:
    conn = sqlite3.connect(ai_db.DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT chapter_id, chapter_index, chapter_title, start_order_index, end_order_index,
                   chunk_count, word_count, char_count
            FROM ai_rag_chapters
            WHERE book_id = ?
            ORDER BY chapter_index ASC
            """,
            (int(book_id),),
        )
        chapter_rows = cur.fetchall()
        cur.execute(
            """
            SELECT COUNT(1)
            FROM ai_rag_meta_ver
            WHERE book_id = ? AND namespace = 'chunk'
            """,
            (int(book_id),),
        )
        total_chunk_row = cur.fetchone()
    finally:
        conn.close()

    chapters = [
        {
            "chapter_id": row[0],
            "chapter_index": int(row[1]),
            "chapter_title": row[2],
            "start_order_id": int(row[3]),
            "end_order_id": int(row[4]),
            "chunk_count": int(row[5]),
            "word_count": int(row[6] or 0),
            "char_count": int(row[7] or 0),
        }
        for row in chapter_rows
    ]
    return {
        "book_id": int(book_id),
        "total_chunks": int(total_chunk_row[0] or 0) if total_chunk_row else 0,
        "total_chapters": len(chapters),
        "chapters": chapters,
    }


def get_chapter_chunks(
    book_id: int,
    chapter_index: int,
    start_chunk_offset: int = 0,
    limit_chunks: int = 5,
) -> Dict[str, Any]:
    outline = get_book_outline(book_id=int(book_id))
    target_chapter = next(
        (chapter for chapter in outline["chapters"] if chapter["chapter_index"] == int(chapter_index)),
        None,
    )
    if not target_chapter:
        return {"status": "empty", "message": "Chapter not found", "book_id": int(book_id), "chapter_index": int(chapter_index)}

    start_order_id = target_chapter["start_order_id"] + max(0, int(start_chunk_offset))
    rows = get_chunks_by_order_range(book_id=int(book_id), start_order_id=start_order_id, limit=max(1, int(limit_chunks)))
    chapter_rows = [row for row in rows if row.get("chapter_index") == int(chapter_index)]
    return {
        "status": "success",
        "book_id": int(book_id),
        "chapter": target_chapter,
        "chunks": chapter_rows,
    }


def _extract_order_ids_from_text(*fields: Optional[str]) -> List[int]:
    order_ids = set()
    for field in fields:
        if not field:
            continue
        for segment in ORDER_ID_PATTERN.findall(str(field)):
            for part in segment.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    order_ids.add(int(part))
                except ValueError:
                    continue
    return sorted(order_ids)


def _build_reading_windows(order_ids: List[int], padding: int = 1) -> List[Dict[str, int]]:
    if not order_ids:
        return []
    windows = []
    for order_id in sorted(set(int(v) for v in order_ids)):
        start = max(0, order_id - max(0, int(padding)))
        end = order_id + max(0, int(padding))
        if windows and start <= windows[-1]["end_order_id"] + 1:
            windows[-1]["end_order_id"] = max(windows[-1]["end_order_id"], end)
        else:
            windows.append({"start_order_id": start, "end_order_id": end})
    return windows


def _build_anchor_trace(book_id: int, order_ids: List[int]) -> List[Dict[str, Any]]:
    anchors = []
    for order_id in order_ids:
        row = get_chunk_by_order(book_id=int(book_id), order_id=int(order_id))
        if not row:
            continue
        excerpt = (row.get("content") or "").strip()
        if len(excerpt) > 220:
            excerpt = excerpt[:220].rstrip() + "..."
        anchors.append(
            {
                "order_id": int(order_id),
                "chapter_id": row.get("chapter_id"),
                "chapter_index": row.get("chapter_index"),
                "chapter_title": row.get("chapter_title"),
                "chunk_index_in_chapter": row.get("chunk_index_in_chapter"),
                "excerpt": excerpt,
            }
        )
    return anchors


def _build_reasoning_trace(book_id: int, answer_draft: str, docs: str, summary: str, timeline: str) -> Dict[str, Any]:
    mentioned_order_ids = _extract_order_ids_from_text(answer_draft)
    if not mentioned_order_ids:
        mentioned_order_ids = _extract_order_ids_from_text(docs, summary, timeline)
    return {
        "mentioned_order_ids": mentioned_order_ids,
        "reading_windows": _build_reading_windows(mentioned_order_ids, padding=1),
        "anchors": _build_anchor_trace(book_id=int(book_id), order_ids=mentioned_order_ids),
    }


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
    chapter_indices: Optional[List[int]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    phase = "CHECK_EXISTING"
    normalized_indices = _normalize_chapter_indices(chapter_indices)
    scope_label = _chapter_scope_label(normalized_indices)
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
        _append_job_log(int(book_id), run_id, phase, "INFO", f"existing indexed chunks={existing} scope={scope_label}")

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
                "message": f"Index already exists; manually clear comorag tables before force rebuild ({scope_label}).",
            }

        phase = "LOAD_SOURCE"
        _notify_progress(progress_callback, phase, "Loading source docs", 0.25)
        _upsert_job(int(book_id), status="RUNNING", phase=phase, message=f"Loading source docs ({scope_label})")
        _append_job_log(int(book_id), run_id, phase, "INFO", f"Start parsing source book scope={scope_label}")
        chunk_rows = _build_chunk_rows_for_book(int(book_id), chapter_indices=normalized_indices)
        if not chunk_rows:
            _notify_progress(progress_callback, "FAILED", "No readable docs available for this book", 1.0)
            _upsert_job(
                int(book_id),
                status="FAILED",
                phase="FAILED",
                message=f"No readable docs available for this book ({scope_label})",
                last_error=f"No readable docs available for this book ({scope_label})",
                finished_at=_now_iso(),
            )
            _append_job_log(int(book_id), run_id, phase, "ERROR", f"No readable docs available for this book scope={scope_label}")
            return False, {"status": "error", "message": f"No readable docs available for this book ({scope_label})"}

        _notify_progress(progress_callback, phase, f"Loaded {len(chunk_rows)} chunks", 0.45)
        _upsert_job(int(book_id), total_chunks=len(chunk_rows), message=f"Loaded {len(chunk_rows)} chunks ({scope_label})")
        _append_job_log(int(book_id), run_id, phase, "INFO", f"Loaded docs count={len(chunk_rows)} scope={scope_label}")

        phase = "INDEXING"
        _notify_progress(progress_callback, phase, "Building index", 0.70)
        _upsert_job(int(book_id), status="RUNNING", phase=phase, message=f"Building index ({scope_label})")
        _append_job_log(int(book_id), run_id, phase, "INFO", f"Start ComoRAG.index(chunk_rows) scope={scope_label}")
        rag = get_rag(int(book_id))
        rag.index(chunk_rows)

        phase = "FINALIZE"
        _notify_progress(progress_callback, phase, "Finalizing index", 0.95)
        current = _count_ver_indexed(int(book_id))
        _upsert_job(
            int(book_id),
            status="SUCCESS",
            phase="DONE",
            message=f"Index build completed ({scope_label})",
            indexed_chunks=current,
            total_chunks=max(len(chunk_rows), current),
            finished_at=_now_iso(),
        )
        _notify_progress(progress_callback, "DONE", "Index build completed", 1.0)
        _append_job_log(int(book_id), run_id, phase, "INFO", f"Index build completed, indexed={current} scope={scope_label}")
        payload = get_index_status(int(book_id))
        payload["build_scope"] = {"chapter_indices": normalized_indices, "scope_label": scope_label}
        return True, payload
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
        return False, {"status": "error", "message": err_msg, "build_scope": {"chapter_indices": normalized_indices, "scope_label": scope_label}}


def trigger_index_build(book_id: int, force: bool = False, chapter_indices: Optional[List[int]] = None) -> Tuple[bool, Dict[str, Any]]:
    normalized_indices = _normalize_chapter_indices(chapter_indices)
    scope_label = _chapter_scope_label(normalized_indices)
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
            message=f"Queued ({scope_label})",
            last_error="",
            started_at=_now_iso(),
            finished_at="",
        )
        _append_job_log(int(book_id), run_id, "QUEUED", "INFO", f"Index build queued scope={scope_label}")
        from ..services.worker import WorkerThread
        from ..tasks.comorag_index import TaskComoRAGIndex
        WorkerThread.add(
            user="System",
            task=TaskComoRAGIndex(book_id=int(book_id), run_id=run_id, force=force, chapter_indices=normalized_indices),
            hidden=False,
        )
        return True, {
            "status": "queued",
            "message": "Index build queued",
            "book_id": int(book_id),
            "run_id": run_id,
            "build_scope": {"chapter_indices": normalized_indices, "scope_label": scope_label},
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
    answer_draft = solution.answer
    trace = _build_reasoning_trace(
        book_id=int(book_id),
        answer_draft=answer_draft,
        docs=solution.docs,
        summary=solution.summary,
        timeline=solution.timeline,
    )
    return {
        "status": "success",
        "book_id": int(book_id),
        "question": question,
        "answer": answer_draft,
        "answer_draft": answer_draft,
        "docs": solution.docs,
        "summary": solution.summary,
        "timeline": solution.timeline,
        "trace": trace,
        "index": index_info,
    }
