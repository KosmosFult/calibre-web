# -*- coding: utf-8 -*-
"""SQLite metadata, event graph, lexical search, and reasoning traces."""

from __future__ import annotations

import contextlib
import datetime
import json
import re
import sqlite3
import unicodedata
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .models import NarrativeExtraction, NarrativeMemory, Passage, stable_id


SCHEMA_VERSION = 3


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    return re.sub(r"[^\w\u3400-\u9fff]+", "", value)


class NovelRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.fts_enabled = False
        self.initialize()

    @contextlib.contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS novel_index_runs (
                    run_id TEXT PRIMARY KEY,
                    book_id INTEGER NOT NULL,
                    source_fingerprint TEXT,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    message TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    llm_model TEXT,
                    embedding_model TEXT,
                    vector_table TEXT,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    total_chunks INTEGER NOT NULL DEFAULT 0,
                    indexed_chunks INTEGER NOT NULL DEFAULT 0,
                    active INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_novel_runs_book ON novel_index_runs(book_id, created_at DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_novel_runs_active
                    ON novel_index_runs(book_id) WHERE active = 1;

                CREATE TABLE IF NOT EXISTS novel_index_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    book_id INTEGER NOT NULL,
                    phase TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_novel_logs_book ON novel_index_logs(book_id, id DESC);

                CREATE TABLE IF NOT EXISTS novel_chunks (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    book_id INTEGER NOT NULL,
                    order_id INTEGER NOT NULL,
                    chapter_id TEXT NOT NULL,
                    chapter_index INTEGER NOT NULL,
                    chapter_title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    paragraph_start INTEGER NOT NULL,
                    paragraph_end INTEGER NOT NULL,
                    text_hash TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    UNIQUE(run_id, chunk_id),
                    UNIQUE(run_id, order_id)
                );
                CREATE INDEX IF NOT EXISTS idx_novel_chunks_order ON novel_chunks(run_id, order_id);

                CREATE TABLE IF NOT EXISTS novel_entities (
                    entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    normalized_name TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    aliases_json TEXT NOT NULL,
                    UNIQUE(run_id, normalized_name)
                );
                CREATE INDEX IF NOT EXISTS idx_novel_entity_name ON novel_entities(run_id, normalized_name);

                CREATE TABLE IF NOT EXISTS novel_entity_mentions (
                    run_id TEXT NOT NULL,
                    entity_id INTEGER NOT NULL,
                    chunk_id TEXT NOT NULL,
                    mention TEXT NOT NULL,
                    UNIQUE(run_id, entity_id, chunk_id)
                );

                CREATE TABLE IF NOT EXISTS novel_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    event_key TEXT NOT NULL,
                    sequence_no INTEGER NOT NULL,
                    order_start INTEGER NOT NULL,
                    order_end INTEGER NOT NULL,
                    chapter_title TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    story_time TEXT NOT NULL,
                    location TEXT NOT NULL,
                    epistemic_status TEXT NOT NULL,
                    evidence_chunk_ids_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    UNIQUE(run_id, event_key)
                );
                CREATE INDEX IF NOT EXISTS idx_novel_events_order ON novel_events(run_id, sequence_no, order_start);

                CREATE TABLE IF NOT EXISTS novel_event_entities (
                    run_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    entity_id INTEGER NOT NULL,
                    UNIQUE(run_id, event_id, entity_id)
                );

                CREATE TABLE IF NOT EXISTS novel_event_links (
                    run_id TEXT NOT NULL,
                    source_event_id TEXT NOT NULL,
                    target_event_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    evidence_chunk_id TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    UNIQUE(run_id, source_event_id, target_event_id, relation_type)
                );
                CREATE INDEX IF NOT EXISTS idx_novel_links_source ON novel_event_links(run_id, source_event_id);
                CREATE INDEX IF NOT EXISTS idx_novel_links_target ON novel_event_links(run_id, target_event_id);

                CREATE TABLE IF NOT EXISTS novel_memories (
                    memory_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    level TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    position_start INTEGER NOT NULL,
                    position_end INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    unresolved_json TEXT NOT NULL,
                    source_chunk_ids_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_novel_memories_position
                    ON novel_memories(run_id, level, position_start);

                CREATE TABLE IF NOT EXISTS novel_query_traces (
                    trace_id TEXT PRIMARY KEY,
                    book_id INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    rounds_json TEXT NOT NULL,
                    answer_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            try:
                conn.executescript(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS novel_chunks_fts USING fts5(
                        run_id UNINDEXED, chunk_id UNINDEXED, chapter_title, text,
                        tokenize='unicode61'
                    );
                    CREATE VIRTUAL TABLE IF NOT EXISTS novel_memories_fts USING fts5(
                        run_id UNINDEXED, memory_id UNINDEXED, title, summary,
                        tokenize='unicode61'
                    );
                    """
                )
                self.fts_enabled = True
            except sqlite3.OperationalError:
                self.fts_enabled = False

    def create_run(
        self,
        *,
        run_id: str,
        book_id: int,
        status: str,
        phase: str,
        message: str,
        llm_model: str,
        embedding_model: str,
        config: Mapping[str, Any],
    ) -> None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO novel_index_runs(
                    run_id, book_id, status, phase, message, schema_version,
                    llm_model, embedding_model, config_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, int(book_id), status, phase, message, SCHEMA_VERSION,
                    llm_model, embedding_model, json.dumps(config, ensure_ascii=False), now, now,
                ),
            )

    def update_run(self, run_id: str, **values: Any) -> None:
        allowed = {
            "source_fingerprint", "status", "phase", "message", "vector_table",
            "total_chunks", "indexed_chunks", "last_error", "finished_at",
        }
        filtered = {key: value for key, value in values.items() if key in allowed}
        if not filtered:
            return
        filtered["updated_at"] = utc_now()
        assignments = ", ".join("{} = ?".format(key) for key in filtered)
        with self.connect() as conn:
            conn.execute(
                "UPDATE novel_index_runs SET {} WHERE run_id = ?".format(assignments),
                list(filtered.values()) + [run_id],
            )

    def append_log(self, run_id: str, book_id: int, phase: str, message: str, level: str = "INFO") -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO novel_index_logs(run_id, book_id, phase, level, message, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, int(book_id), phase, level, message, utc_now()),
            )

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM novel_index_runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def active_run(self, book_id: int) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM novel_index_runs WHERE book_id = ? AND active = 1 LIMIT 1",
                (int(book_id),),
            ).fetchone()
        return dict(row) if row else None

    def latest_run(self, book_id: int) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM novel_index_runs WHERE book_id = ? ORDER BY created_at DESC LIMIT 1",
                (int(book_id),),
            ).fetchone()
        return dict(row) if row else None

    def find_matching_active(self, book_id: int, fingerprint: str, embedding_model: str) -> Optional[Dict[str, Any]]:
        row = self.active_run(book_id)
        if row and row.get("source_fingerprint") == fingerprint and row.get("embedding_model") == embedding_model:
            return row
        return None

    def promote_run(self, run_id: str) -> None:
        now = utc_now()
        with self.connect() as conn:
            row = conn.execute("SELECT book_id FROM novel_index_runs WHERE run_id = ?", (run_id,)).fetchone()
            if not row:
                raise ValueError("Unknown index run {}".format(run_id))
            conn.execute("UPDATE novel_index_runs SET active = 0 WHERE book_id = ?", (row["book_id"],))
            conn.execute(
                "UPDATE novel_index_runs SET active = 1, status = 'SUCCESS', phase = 'DONE', message = 'Narrative index ready', finished_at = ?, updated_at = ? WHERE run_id = ?",
                (now, now, run_id),
            )

    def insert_passages(self, run_id: str, passages: Sequence[Passage]) -> None:
        rows = [
            (
                run_id, item.chunk_id, item.book_id, item.order_id, item.chapter_id,
                item.chapter_index, item.chapter_title, item.text, item.paragraph_start,
                item.paragraph_end, item.text_hash, len(item.text),
            )
            for item in passages
        ]
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO novel_chunks(
                    run_id, chunk_id, book_id, order_id, chapter_id, chapter_index,
                    chapter_title, text, paragraph_start, paragraph_end, text_hash, char_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            if self.fts_enabled:
                conn.executemany(
                    "INSERT INTO novel_chunks_fts(run_id, chunk_id, chapter_title, text) VALUES (?, ?, ?, ?)",
                    [(run_id, item.chunk_id, item.chapter_title, item.text) for item in passages],
                )

    def _chunk_positions(self, conn, run_id: str, chunk_ids: Sequence[str]) -> Tuple[int, int, str]:
        ids = [value for value in chunk_ids if value]
        if not ids:
            return 0, 0, ""
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            "SELECT order_id, chapter_title FROM novel_chunks WHERE run_id = ? AND chunk_id IN ({}) ORDER BY order_id".format(placeholders),
            [run_id] + ids,
        ).fetchall()
        if not rows:
            return 0, 0, ""
        return int(rows[0]["order_id"]), int(rows[-1]["order_id"]), str(rows[0]["chapter_title"])

    def insert_extraction(self, run_id: str, extraction: NarrativeExtraction) -> None:
        with self.connect() as conn:
            entity_ids: Dict[str, int] = {}
            for entity in extraction.entities:
                normalized = normalize_name(entity.name)
                if not normalized:
                    continue
                aliases = sorted(set([entity.name] + entity.aliases))
                existing = conn.execute(
                    "SELECT entity_id, aliases_json, description FROM novel_entities WHERE run_id = ? AND normalized_name = ?",
                    (run_id, normalized),
                ).fetchone()
                if existing:
                    merged_aliases = sorted(set(json.loads(existing["aliases_json"]) + aliases))
                    description = existing["description"] or entity.description
                    conn.execute(
                        "UPDATE novel_entities SET aliases_json = ?, description = ? WHERE entity_id = ?",
                        (json.dumps(merged_aliases, ensure_ascii=False), description, existing["entity_id"]),
                    )
                    entity_id = int(existing["entity_id"])
                else:
                    cursor = conn.execute(
                        "INSERT INTO novel_entities(run_id, normalized_name, canonical_name, entity_type, description, aliases_json) VALUES (?, ?, ?, ?, ?, ?)",
                        (run_id, normalized, entity.name, entity.entity_type, entity.description, json.dumps(aliases, ensure_ascii=False)),
                    )
                    entity_id = int(cursor.lastrowid)
                entity_ids[entity.key] = entity_id
                for chunk_id in entity.mention_chunk_ids:
                    conn.execute(
                        "INSERT OR IGNORE INTO novel_entity_mentions(run_id, entity_id, chunk_id, mention) VALUES (?, ?, ?, ?)",
                        (run_id, entity_id, chunk_id, entity.name),
                    )

            event_ids: Dict[str, str] = {}
            for sequence_offset, event in enumerate(extraction.events):
                event_id = stable_id("event", run_id, event.key)
                event_ids[event.key] = event_id
                start, end, chapter_title = self._chunk_positions(conn, run_id, event.evidence_chunk_ids)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO novel_events(
                        event_id, run_id, event_key, sequence_no, order_start, order_end,
                        chapter_title, title, summary, event_type, story_time, location, epistemic_status,
                        evidence_chunk_ids_json, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id, run_id, event.key, start * 100 + sequence_offset, start, end,
                        chapter_title, event.title, event.summary, event.event_type,
                        event.story_time, event.location, event.epistemic_status,
                        json.dumps(event.evidence_chunk_ids, ensure_ascii=False), event.confidence,
                    ),
                )
                for key in event.participant_keys:
                    entity_id = entity_ids.get(key)
                    if entity_id:
                        conn.execute(
                            "INSERT OR IGNORE INTO novel_event_entities(run_id, event_id, entity_id) VALUES (?, ?, ?)",
                            (run_id, event_id, entity_id),
                        )

            for relation in extraction.relations:
                source = event_ids.get(relation.source_event_key)
                target = event_ids.get(relation.target_event_key)
                if source and target:
                    conn.execute(
                        "INSERT OR REPLACE INTO novel_event_links(run_id, source_event_id, target_event_id, relation_type, evidence_chunk_id, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                        (run_id, source, target, relation.relation_type, relation.evidence_chunk_id, relation.confidence),
                    )

    def insert_memory(self, run_id: str, memory: NarrativeMemory) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO novel_memories(
                    memory_id, run_id, level, title, summary, position_start, position_end,
                    state_json, unresolved_json, source_chunk_ids_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.memory_id, run_id, memory.level, memory.title, memory.summary,
                    memory.position_start, memory.position_end,
                    json.dumps(memory.state, ensure_ascii=False),
                    json.dumps(memory.unresolved_threads, ensure_ascii=False),
                    json.dumps(memory.source_chunk_ids, ensure_ascii=False),
                ),
            )
            if self.fts_enabled:
                conn.execute(
                    "INSERT INTO novel_memories_fts(run_id, memory_id, title, summary) VALUES (?, ?, ?, ?)",
                    (run_id, memory.memory_id, memory.title, memory.summary),
                )

    def insert_cross_event_links(self, run_id: str, links: Sequence[Mapping[str, Any]]) -> None:
        with self.connect() as conn:
            for link in links:
                source = str(link.get("source_event_id") or "")
                target = str(link.get("target_event_id") or "")
                if not source or not target or source == target:
                    continue
                valid = conn.execute(
                    "SELECT COUNT(*) FROM novel_events WHERE run_id = ? AND event_id IN (?, ?)",
                    (run_id, source, target),
                ).fetchone()[0]
                if int(valid) != 2:
                    continue
                conn.execute(
                    "INSERT OR REPLACE INTO novel_event_links(run_id, source_event_id, target_event_id, relation_type, evidence_chunk_id, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        run_id, source, target, str(link.get("relation_type") or "related"),
                        str(link.get("evidence_chunk_id") or ""),
                        max(0.0, min(1.0, float(link.get("confidence") or 0.0))),
                    ),
                )

    def chapter_events(self, run_id: str, order_start: int, order_end: int) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM novel_events WHERE run_id = ? AND order_start BETWEEN ? AND ? ORDER BY sequence_no",
                (run_id, int(order_start), int(order_end)),
            ).fetchall()
        return [dict(row) for row in rows]

    def memories(self, run_id: str, level: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM novel_memories WHERE run_id = ?"
        params: List[Any] = [run_id]
        if level:
            query += " AND level = ?"
            params.append(level)
        query += " ORDER BY position_start"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_chunk(self, book_id: int, order_id: int) -> Optional[Dict[str, Any]]:
        active = self.active_run(book_id)
        if not active:
            return None
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM novel_chunks WHERE run_id = ? AND order_id = ?",
                (active["run_id"], int(order_id)),
            ).fetchone()
        return self._public_chunk(row) if row else None

    @staticmethod
    def _public_chunk(row: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "chunk_id": row["chunk_id"], "hash_id": row["chunk_id"],
            "book_id": int(row["book_id"]), "order_id": int(row["order_id"]),
            "chapter_id": row["chapter_id"], "chapter_index": int(row["chapter_index"]),
            "chapter_title": row["chapter_title"], "content": row["text"], "text": row["text"],
            "word_count": len(str(row["text"]).split()), "char_count": int(row["char_count"]),
        }

    def chunk_range(self, book_id: int, start: int, limit: int) -> List[Dict[str, Any]]:
        active = self.active_run(book_id)
        if not active:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM novel_chunks WHERE run_id = ? AND order_id >= ? ORDER BY order_id LIMIT ?",
                (active["run_id"], max(0, int(start)), max(1, min(int(limit), 100))),
            ).fetchall()
        return [self._public_chunk(row) for row in rows]

    def outline(self, book_id: int) -> Dict[str, Any]:
        active = self.active_run(book_id)
        if not active:
            return {"book_id": int(book_id), "total_chunks": 0, "chapters": []}
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT chapter_index, chapter_id, chapter_title, MIN(order_id) AS start_order_id,
                       MAX(order_id) AS end_order_id, COUNT(*) AS chunk_count,
                       SUM(char_count) AS char_count
                FROM novel_chunks WHERE run_id = ?
                GROUP BY chapter_index, chapter_id, chapter_title ORDER BY chapter_index
                """,
                (active["run_id"],),
            ).fetchall()
        chapters = [
            {
                "chapter_index": int(row["chapter_index"]), "chapter_id": row["chapter_id"],
                "chapter_title": row["chapter_title"], "start_order_id": int(row["start_order_id"]),
                "end_order_id": int(row["end_order_id"]), "chunk_count": int(row["chunk_count"]),
                "word_count": int(row["char_count"]),
            }
            for row in rows
        ]
        return {"book_id": int(book_id), "total_chunks": sum(row["chunk_count"] for row in chapters), "chapters": chapters}

    def lexical_search(self, run_id: str, terms: Sequence[str], limit: int) -> List[Dict[str, Any]]:
        cleaned = [str(term).strip().replace('"', "") for term in terms if str(term).strip()]
        if not cleaned:
            return []
        with self.connect() as conn:
            if self.fts_enabled:
                match = " OR ".join('"{}"'.format(term) for term in cleaned[:12])
                try:
                    rows = conn.execute(
                        """
                        SELECT c.*, bm25(novel_chunks_fts) AS lexical_rank
                        FROM novel_chunks_fts
                        JOIN novel_chunks c ON c.run_id = novel_chunks_fts.run_id AND c.chunk_id = novel_chunks_fts.chunk_id
                        WHERE novel_chunks_fts MATCH ? AND novel_chunks_fts.run_id = ?
                        ORDER BY lexical_rank LIMIT ?
                        """,
                        (match, run_id, int(limit)),
                    ).fetchall()
                    return [dict(row) for row in rows]
                except sqlite3.OperationalError:
                    pass
            clauses = " OR ".join("text LIKE ?" for _ in cleaned[:8])
            rows = conn.execute(
                "SELECT *, 0.0 AS lexical_rank FROM novel_chunks WHERE run_id = ? AND ({}) ORDER BY order_id LIMIT ?".format(clauses),
                [run_id] + ["%{}%".format(term) for term in cleaned[:8]] + [int(limit)],
            ).fetchall()
        return [dict(row) for row in rows]

    def graph_search(self, run_id: str, entity_names: Sequence[str], hops: int, limit: int) -> List[Dict[str, Any]]:
        normalized = [normalize_name(name) for name in entity_names if normalize_name(name)]
        if not normalized:
            return []
        with self.connect() as conn:
            clauses = " OR ".join("normalized_name LIKE ? OR aliases_json LIKE ?" for _ in normalized)
            params: List[Any] = [run_id]
            for name in normalized:
                params.extend(["%{}%".format(name), "%{}%".format(name)])
            entities = conn.execute(
                "SELECT entity_id FROM novel_entities WHERE run_id = ? AND ({})".format(clauses), params
            ).fetchall()
            entity_ids = [row["entity_id"] for row in entities]
            if not entity_ids:
                return []
            placeholders = ",".join("?" for _ in entity_ids)
            seeds = conn.execute(
                "SELECT DISTINCT event_id FROM novel_event_entities WHERE run_id = ? AND entity_id IN ({})".format(placeholders),
                [run_id] + entity_ids,
            ).fetchall()
            visited = {row["event_id"] for row in seeds}
            frontier = set(visited)
            for _ in range(max(0, int(hops))):
                if not frontier:
                    break
                marks = ",".join("?" for _ in frontier)
                links = conn.execute(
                    "SELECT source_event_id, target_event_id FROM novel_event_links WHERE run_id = ? AND (source_event_id IN ({0}) OR target_event_id IN ({0}))".format(marks),
                    [run_id] + list(frontier) + list(frontier),
                ).fetchall()
                new_frontier = set()
                for row in links:
                    new_frontier.update([row["source_event_id"], row["target_event_id"]])
                new_frontier -= visited
                visited.update(new_frontier)
                frontier = new_frontier
            if not visited:
                return []
            marks = ",".join("?" for _ in visited)
            rows = conn.execute(
                "SELECT * FROM novel_events WHERE run_id = ? AND event_id IN ({}) ORDER BY sequence_no LIMIT ?".format(marks),
                [run_id] + list(visited) + [int(limit)],
            ).fetchall()
            selected_ids = [row["event_id"] for row in rows]
            links: List[Dict[str, Any]] = []
            if selected_ids:
                selected_marks = ",".join("?" for _ in selected_ids)
                link_rows = conn.execute(
                    """
                    SELECT l.source_event_id, l.target_event_id, l.relation_type,
                           l.evidence_chunk_id, l.confidence,
                           source.title AS source_title, target.title AS target_title
                    FROM novel_event_links l
                    JOIN novel_events source ON source.event_id = l.source_event_id
                    JOIN novel_events target ON target.event_id = l.target_event_id
                    WHERE l.run_id = ? AND (
                        l.source_event_id IN ({0}) OR l.target_event_id IN ({0})
                    )
                    """.format(selected_marks),
                    [run_id] + selected_ids + selected_ids,
                ).fetchall()
                links = [dict(row) for row in link_rows]
        result = [dict(row) for row in rows]
        for event in result:
            event["_graph_links"] = [
                link for link in links
                if link["source_event_id"] == event["event_id"] or link["target_event_id"] == event["event_id"]
            ]
        return result

    def resolve_reference(self, run_id: str, kind: str, ref_id: str) -> Optional[Dict[str, Any]]:
        table = {"chunk": ("novel_chunks", "chunk_id"), "event": ("novel_events", "event_id"), "memory": ("novel_memories", "memory_id")}.get(kind)
        if not table:
            return None
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM {} WHERE run_id = ? AND {} = ?".format(table[0], table[1]),
                (run_id, ref_id),
            ).fetchone()
        return dict(row) if row else None

    def counts(self, run_id: str) -> Dict[str, int]:
        with self.connect() as conn:
            values = {}
            for key, table in (("chunks", "novel_chunks"), ("entities", "novel_entities"), ("events", "novel_events"), ("memories", "novel_memories")):
                values[key] = int(conn.execute("SELECT COUNT(*) FROM {} WHERE run_id = ?".format(table), (run_id,)).fetchone()[0])
        return values

    def status(self, book_id: int, log_limit: int = 30) -> Dict[str, Any]:
        latest = self.latest_run(book_id)
        active = self.active_run(book_id)
        selected = latest or active
        if not selected:
            return {"status": "not_indexed", "book_id": int(book_id), "indexed": False, "total_chunks": 0, "indexed_chunks": 0, "job": None, "logs": [], "counts": {}}
        running = selected["status"] in {"QUEUED", "RUNNING"}
        ready = bool(active) and not running
        metrics = active if active and not running else selected
        with self.connect() as conn:
            logs = conn.execute(
                "SELECT phase, level, message, created_at FROM novel_index_logs WHERE book_id = ? ORDER BY id DESC LIMIT ?",
                (int(book_id), int(log_limit)),
            ).fetchall()
        return {
            "status": "running" if running else ("ready" if ready else "failed" if selected["status"] == "FAILED" else "not_indexed"),
            "book_id": int(book_id), "indexed": ready,
            "total_chunks": int(metrics.get("total_chunks") or 0),
            "indexed_chunks": int(metrics.get("indexed_chunks") or 0),
            "job": selected, "logs": [dict(row) for row in reversed(logs)],
            "counts": self.counts(active["run_id"]) if active else {},
        }

    def save_trace(self, book_id: int, run_id: str, question: str, plan: Mapping[str, Any], rounds: Sequence[Any], answer: Mapping[str, Any]) -> str:
        trace_id = stable_id("trace", run_id, question, utc_now())
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO novel_query_traces(trace_id, book_id, run_id, question, plan_json, rounds_json, answer_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (trace_id, int(book_id), run_id, question, json.dumps(plan, ensure_ascii=False), json.dumps(rounds, ensure_ascii=False), json.dumps(answer, ensure_ascii=False), utc_now()),
            )
        return trace_id

    def obsolete_runs(self, book_id: int, keep: int) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM novel_index_runs WHERE book_id = ? AND active = 0 AND status NOT IN ('QUEUED', 'RUNNING') ORDER BY created_at DESC",
                (int(book_id),),
            ).fetchall()
        return [dict(row) for row in rows[max(0, keep - 1):]]

    def delete_run(self, run_id: str) -> None:
        tables = [
            "novel_chunks", "novel_entities", "novel_entity_mentions", "novel_events",
            "novel_event_entities", "novel_event_links", "novel_memories",
        ]
        with self.connect() as conn:
            if self.fts_enabled:
                conn.execute("DELETE FROM novel_chunks_fts WHERE run_id = ?", (run_id,))
                conn.execute("DELETE FROM novel_memories_fts WHERE run_id = ?", (run_id,))
            for table in tables:
                conn.execute("DELETE FROM {} WHERE run_id = ?".format(table), (run_id,))
            conn.execute("DELETE FROM novel_index_runs WHERE run_id = ? AND active = 0", (run_id,))
