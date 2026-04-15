import numpy as np
import os
from typing import Union, Optional, List, Dict, Set, Any, Tuple, Literal
import logging
from copy import deepcopy
import datetime
import sqlite3
import re
import json

import lancedb
import pyarrow as pa

from .utils.misc_utils import compute_mdhash_id, NerRawOutput, TripleRawOutput
from .. import ai_db
from ..ai_storage import get_lancedb_dir

logger = logging.getLogger(__name__)

CHAPTER_TABLE = "ai_rag_chapters"

class EmbeddingStore:
    TYPE_TO_META_TABLE = {
        "ver": "ai_rag_meta_ver",
        "entity": "ai_rag_meta_entity",
        "fact": "ai_rag_meta_fact",
        "sem": "ai_rag_meta_sem",
        "epi": "ai_rag_meta_epi",
        "misc": "ai_rag_meta_misc",
    }
    TYPE_TO_VECTOR_TABLE = {
        "ver": "emb_ver",
        "entity": "emb_entity",
        "fact": "emb_fact",
        "sem": "emb_sem",
        "epi": "emb_epi",
        "misc": "emb_misc",
    }

    def __init__(self, embedding_model, db_filename, batch_size, namespace, book_id: int):
        """
        Initializes the class with necessary configurations and sets up the working directory.

        Parameters:
        embedding_model: The model used for embeddings.
        db_filename: The directory path where data will be stored or retrieved.
        batch_size: The batch size used for processing.
        namespace: A unique identifier for data segregation.

        Functionality:
        - Assigns the provided parameters to instance variables.
        - Checks if the directory specified by `db_filename` exists.
          - If not, creates the directory and logs the operation.
        - Constructs the filename for storing data in a parquet file format.
        - Calls the method `_load_data()` to initialize the data loading process.
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace
        self.book_id = int(book_id)
        self.embedding_type = self._resolve_embedding_type(namespace)
        self.meta_table = self.TYPE_TO_META_TABLE[self.embedding_type]
        self.vector_table_name = self.TYPE_TO_VECTOR_TABLE[self.embedding_type]

        if db_filename and not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        self.db_path = ai_db.DB_PATH
        # Keep filename for compatibility with downstream code that inspects dirname.
        self.filename = os.path.join(db_filename or ".", f"vdb_{self.namespace}.sqlite")
        self.vector_db_path = get_lancedb_dir()
        self.vector_db = lancedb.connect(self.vector_db_path)
        self.vector_table = None
        self._load_data()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _sanitize_namespace(namespace: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", namespace or "default")
        if sanitized and sanitized[0].isdigit():
            sanitized = f"ns_{sanitized}"
        return sanitized

    @staticmethod
    def _resolve_embedding_type(namespace: str) -> str:
        normalized = (namespace or "").lower()
        if normalized in ("chunk", "emb_ver"):
            return "ver"
        if normalized in ("entity", "emb_entity"):
            return "entity"
        if normalized in ("fact", "emb_fact"):
            return "fact"
        if normalized in ("summary", "emb_sem"):
            return "sem"
        if normalized in ("timeline", "emb_epi") or normalized.startswith("level_"):
            return "epi"
        return "misc"

    def _get_vector_table(self):
        if self.vector_table is not None:
            return self.vector_table
        try:
            self.vector_table = self.vector_db.open_table(self.vector_table_name)
            return self.vector_table
        except Exception:
            return None

    def _init_vector_table(self, records: List[Dict[str, Any]]):
        if not records:
            return
        table_data = self._records_to_arrow_table(records)
        if self._get_vector_table() is None:
            self.vector_table = self.vector_db.create_table(self.vector_table_name, data=table_data, mode="overwrite")
        else:
            self.vector_table.add(table_data)

    def _records_to_arrow_table(self, records: List[Dict[str, Any]]) -> pa.Table:
        normalized = {
            "book_id": [int(record.get("book_id", 0)) for record in records],
            "namespace": [str(record.get("namespace", "")) for record in records],
            "hash_id": [str(record.get("hash_id", "")) for record in records],
            "content": [str(record.get("content", "")) for record in records],
            "order_index": [int(record.get("order_index", -1)) for record in records],
            "vector": [
                [float(v) for v in (record.get("vector") or [])]
                for record in records
            ],
        }
        fields = [
            pa.field("book_id", pa.int64()),
            pa.field("namespace", pa.string()),
            pa.field("hash_id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("order_index", pa.int64()),
            pa.field("vector", pa.list_(pa.float32())),
        ]
        if self.embedding_type != "ver":
            normalized["source_order_ids"] = [
                [int(v) for v in (record.get("source_order_ids") or [])]
                for record in records
            ]
            fields.insert(4, pa.field("source_order_ids", pa.list_(pa.int64())))
        schema = pa.schema(fields)
        return pa.Table.from_pydict(normalized, schema=schema)

    def get_missing_string_hash_ids(self, texts: List[str]):
        nodes_dict = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  {}

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        return {h: {"hash_id": h, "content": t} for h, t in zip(missing_ids, texts_to_encode)}

    def insert_strings(self, texts: List[str], source_order_ids: Optional[List[Optional[List[int]]]] = None):
        nodes_dict = {}
        source_by_hash: Dict[str, List[int]] = {}

        if source_order_ids is not None and len(source_order_ids) != len(texts):
            raise ValueError("source_order_ids length must match texts length")

        for idx, text in enumerate(texts):
            hash_id = compute_mdhash_id(text, prefix=self.namespace + "-")
            nodes_dict[hash_id] = {'content': text}
            if source_order_ids is not None:
                merged_source_orders = set(source_by_hash.get(hash_id, []))
                merged_source_orders.update(source_order_ids[idx] or [])
                source_by_hash[hash_id] = sorted(merged_source_orders)

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  # Nothing to insert.

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

        logger.info(
            f"Inserting {len(missing_ids)} new records, {len(all_hash_ids) - len(missing_ids)} records already exist.")

        if not missing_ids:
            return  {}# All records already exist.

        # Prepare the texts to encode from the "content" field.
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)
        if len(missing_embeddings) != len(missing_ids):
            raise ValueError(
                f"embedding_model.batch_encode returned {len(missing_embeddings)} embeddings "
                f"for {len(missing_ids)} texts in namespace={self.namespace}"
            )

        source_for_missing = {h: source_by_hash.get(h, []) for h in missing_ids}
        self._upsert(missing_ids, texts_to_encode, missing_embeddings, source_order_ids_by_hash=source_for_missing)

    def insert_chunk_rows(self, chunk_rows: List[Dict[str, Any]]):
        if self.embedding_type != "ver":
            raise ValueError("insert_chunk_rows is only supported for ver embedding stores")
        if not chunk_rows:
            return

        normalized_rows: List[Dict[str, Any]] = []
        seen_hashes: Set[str] = set()
        for idx, row in enumerate(chunk_rows):
            text = str(row.get("content") or row.get("text") or "").strip()
            if not text:
                continue
            order_index = int(row.get("order_index", idx))
            hash_id = str(row.get("hash_id") or f"{self.namespace}-{self.book_id}-{order_index}")
            if hash_id in seen_hashes:
                continue
            seen_hashes.add(hash_id)
            normalized_rows.append(
                {
                    "hash_id": hash_id,
                    "content": text,
                    "order_index": int(order_index),
                    "chapter_id": row.get("chapter_id"),
                    "chapter_index": row.get("chapter_index"),
                    "chapter_title": row.get("chapter_title"),
                    "chunk_index_in_chapter": row.get("chunk_index_in_chapter"),
                    "word_count": int(row.get("word_count") or len(text.split())),
                    "char_count": int(row.get("char_count") or len(text)),
                }
            )

        if not normalized_rows:
            return

        texts_to_encode = [row["content"] for row in normalized_rows]
        embeddings = self.embedding_model.batch_encode(texts_to_encode)
        if len(embeddings) != len(normalized_rows):
            raise ValueError(
                f"embedding_model.batch_encode returned {len(embeddings)} embeddings "
                f"for {len(normalized_rows)} ver chunks in namespace={self.namespace}"
            )
        self._replace_ver_rows(normalized_rows, embeddings)

    def _ensure_tables(self):
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.meta_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id INTEGER NOT NULL,
                    namespace TEXT NOT NULL,
                    hash_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    order_index INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(book_id, hash_id)
                )
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.meta_table}_book_namespace_order
                ON {self.meta_table}(book_id, namespace, order_index)
                """
            )
            cur.execute(f"PRAGMA table_info({self.meta_table})")
            existing_cols = {row[1] for row in cur.fetchall()}
            if self.embedding_type != "ver" and "source_order_ids" not in existing_cols:
                cur.execute(f"ALTER TABLE {self.meta_table} ADD COLUMN source_order_ids TEXT")
            if self.embedding_type == "ver":
                ver_columns = {
                    "chapter_id": "TEXT",
                    "chapter_index": "INTEGER",
                    "chapter_title": "TEXT",
                    "chunk_index_in_chapter": "INTEGER",
                    "word_count": "INTEGER",
                    "char_count": "INTEGER",
                }
                for column_name, column_type in ver_columns.items():
                    if column_name not in existing_cols:
                        cur.execute(
                            f"ALTER TABLE {self.meta_table} ADD COLUMN {column_name} {column_type}"
                        )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {CHAPTER_TABLE} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        book_id INTEGER NOT NULL,
                        chapter_id TEXT,
                        chapter_index INTEGER NOT NULL,
                        chapter_title TEXT,
                        start_order_index INTEGER NOT NULL,
                        end_order_index INTEGER NOT NULL,
                        chunk_count INTEGER NOT NULL,
                        word_count INTEGER NOT NULL DEFAULT 0,
                        char_count INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(book_id, chapter_index)
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{CHAPTER_TABLE}_book_chapter
                    ON {CHAPTER_TABLE}(book_id, chapter_index)
                    """
                )
            conn.commit()

    def _load_data(self):
        self._ensure_tables()
        self.hash_ids, self.texts, self.embeddings = [], [], []
        self.hash_id_to_idx, self.hash_id_to_row = {}, {}
        self.hash_id_to_text, self.text_to_hash_id = {}, {}
        with self._connect() as conn:
            cur = conn.cursor()
            if self.embedding_type == "ver":
                cur.execute(
                    f"""
                    SELECT hash_id, content, embedding, order_index, chapter_id, chapter_index, chapter_title, chunk_index_in_chapter,
                           word_count, char_count
                    FROM {self.meta_table}
                    WHERE book_id = ? AND namespace = ?
                    ORDER BY order_index ASC
                    """,
                    (self.book_id, self.namespace),
                )
            else:
                cur.execute(
                    f"""
                    SELECT hash_id, content, source_order_ids, embedding, order_index
                    FROM {self.meta_table}
                    WHERE book_id = ? AND namespace = ?
                    ORDER BY order_index ASC
                    """,
                    (self.book_id, self.namespace),
                )
            rows = cur.fetchall()

        self.hash_id_to_source_orders = {}
        self.hash_id_to_order_index = {}
        for row_tuple in rows:
            hash_id = row_tuple[0]
            content = row_tuple[1]
            if self.embedding_type == "ver":
                embedding_blob = row_tuple[2]
                order_index = row_tuple[3]
                source_orders = []
            else:
                source_order_ids_json = row_tuple[2]
                embedding_blob = row_tuple[3]
                order_index = row_tuple[4]
                source_orders = []
                if source_order_ids_json:
                    try:
                        source_orders = json.loads(source_order_ids_json)
                    except ValueError:
                        source_orders = []
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            self.hash_ids.append(hash_id)
            self.texts.append(content)
            self.embeddings.append(embedding)
            self.hash_id_to_order_index[hash_id] = int(order_index)
            self.hash_id_to_source_orders[hash_id] = source_orders
            row = {
                "hash_id": hash_id,
                "content": content,
                "order_index": int(order_index),
            }
            if self.embedding_type != "ver":
                row["source_order_ids"] = source_orders
            if self.embedding_type == "ver":
                (
                    _hash_id,
                    _content,
                    _embedding_blob,
                    _order_index,
                    chapter_id,
                    chapter_index,
                    chapter_title,
                    chunk_index_in_chapter,
                    word_count,
                    char_count,
                ) = row_tuple
                row.update(
                    {
                        "chapter_id": chapter_id,
                        "chapter_index": chapter_index,
                        "chapter_title": chapter_title,
                        "chunk_index_in_chapter": chunk_index_in_chapter,
                        "word_count": int(word_count or 0),
                        "char_count": int(char_count or 0),
                    }
                )
            self.hash_id_to_row[hash_id] = row

        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        logger.info(
            "Loaded %s records from ai.db table=%s namespace=%s",
            len(self.hash_ids),
            self.meta_table,
            self.namespace,
        )

        # Ensure vector table is open if it exists.
        self._get_vector_table()

    def _save_data(self):
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        logger.info(
            "Saved %s records to ai.db table=%s namespace=%s",
            len(self.hash_ids),
            self.meta_table,
            self.namespace,
        )

    def _upsert(self, hash_ids, texts, embeddings, source_order_ids_by_hash: Optional[Dict[str, List[int]]] = None):
        # 最后TODO: 原子性

        if not hash_ids:
            return
        if not (len(hash_ids) == len(texts) == len(embeddings)):
            raise ValueError(
                "EmbeddingStore._upsert length mismatch: "
                f"hash_ids={len(hash_ids)} texts={len(texts)} embeddings={len(embeddings)} "
                f"namespace={self.namespace}"
            )
        source_order_ids_by_hash = source_order_ids_by_hash or {}
        self._ensure_tables()
        now = datetime.datetime.utcnow().isoformat()
        order_entries: List[Tuple[str, int]] = []
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT COALESCE(MAX(order_index), -1) FROM {self.meta_table} WHERE book_id = ? AND namespace = ?",
                (self.book_id, self.namespace),
            )
            base_order = cur.fetchone()[0] + 1
            for idx, (hash_id, text, embedding) in enumerate(zip(hash_ids, texts, embeddings)):
                emb_arr = np.asarray(embedding, dtype=np.float32)
                order_index = base_order + idx
                cur.execute(
                    f"""
                    INSERT OR REPLACE INTO {self.meta_table}
                    (book_id, namespace, hash_id, content, source_order_ids, embedding, embedding_dim, order_index, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.book_id,
                        self.namespace,
                        hash_id,
                        text,
                        json.dumps(source_order_ids_by_hash.get(hash_id, []), ensure_ascii=False),
                        emb_arr.tobytes(),
                        int(emb_arr.shape[0]),
                        order_index,
                        now,
                        now,
                    ),
                )
                order_entries.append((hash_id, order_index))
            conn.commit()

        self._upsert_lancedb(
            hash_ids=hash_ids,
            texts=texts,
            embeddings=embeddings,
            order_entries=order_entries,
            source_order_ids_by_hash=source_order_ids_by_hash,
        )
        logger.info("Reloading embedding store after upsert.")
        self._load_data()

    def _upsert_lancedb(
        self,
        hash_ids: List[str],
        texts: List[str],
        embeddings: List[np.ndarray],
        order_entries: List[Tuple[str, int]],
        source_order_ids_by_hash: Optional[Dict[str, List[int]]] = None,
    ):
        if not hash_ids:
            return
        source_order_ids_by_hash = source_order_ids_by_hash or {}
        order_map = {h: order for h, order in order_entries}
        records = []
        for hash_id, text, emb in zip(hash_ids, texts, embeddings):
            emb_arr = np.asarray(emb, dtype=np.float32)
            records.append(
                {
                    "book_id": self.book_id,
                    "namespace": self.namespace,
                    "hash_id": hash_id,
                    "content": text,
                    "source_order_ids": source_order_ids_by_hash.get(hash_id, []),
                    "order_index": int(order_map.get(hash_id, -1)),
                    "vector": emb_arr.tolist(),
                }
            )
        self._init_vector_table(records)

    def get_row(self, hash_id):
        return self.hash_id_to_row[hash_id]
    
    def get_rows(self, hash_ids, dtype=np.float32):
        if not hash_ids:
            return {}

        results = {id : self.hash_id_to_row[id] for id in hash_ids}

        return results

    def get_all_ids(self):
        return deepcopy(self.hash_ids)

    def get_text_for_all_rows(self):
        return deepcopy(self.hash_id_to_row)

    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        return self.embeddings[self.hash_id_to_idx[hash_id]].astype(dtype)
    
    def get_embeddings(self, hash_ids, dtype=np.float32) -> list[np.ndarray]:
        if not hash_ids:
            return []

        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        if indices.size and int(indices.max()) >= len(self.embeddings):
            logger.warning(
                "Detected stale in-memory embedding index for namespace=%s; reloading store "
                "(max_idx=%s, embedding_count=%s)",
                self.namespace,
                int(indices.max()),
                len(self.embeddings),
            )
            self._load_data()
            indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings, dtype=dtype)[indices]

        return embeddings

    def get_hash_id_to_order(self) -> Dict[str, int]:
        """
        Returns a dictionary mapping each hash value to its sequential position in the original text list.
        
        Returns:
            Dict[str, int]: Mapping from hash values to sequential positions, e.g., {'hash1': 0, 'hash2': 1, ...}
        """
        # Since the texts list maintains insertion order, we can use it to build the order mapping
        return deepcopy(self.hash_id_to_order_index)

    def _replace_ver_rows(self, chunk_rows: List[Dict[str, Any]], embeddings: List[np.ndarray]):
        self._ensure_tables()
        now = datetime.datetime.utcnow().isoformat()
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                f"DELETE FROM {self.meta_table} WHERE book_id = ? AND namespace = ?",
                (self.book_id, self.namespace),
            )
            cur.execute(
                f"DELETE FROM {CHAPTER_TABLE} WHERE book_id = ?",
                (self.book_id,),
            )
            for row, embedding in zip(chunk_rows, embeddings):
                emb_arr = np.asarray(embedding, dtype=np.float32)
                cur.execute(
                    f"""
                    INSERT INTO {self.meta_table}
                    (book_id, namespace, hash_id, content, embedding, embedding_dim, order_index,
                     chapter_id, chapter_index, chapter_title, chunk_index_in_chapter,
                     word_count, char_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.book_id,
                        self.namespace,
                        row["hash_id"],
                        row["content"],
                        emb_arr.tobytes(),
                        int(emb_arr.shape[0]),
                        int(row["order_index"]),
                        row.get("chapter_id"),
                        row.get("chapter_index"),
                        row.get("chapter_title"),
                        row.get("chunk_index_in_chapter"),
                        int(row.get("word_count") or 0),
                        int(row.get("char_count") or 0),
                        now,
                        now,
                    ),
                )
            self._rebuild_chapters(cur=cur, chunk_rows=chunk_rows, now=now)
            conn.commit()

        self._replace_ver_lancedb(chunk_rows=chunk_rows, embeddings=embeddings)
        self._load_data()

    def _rebuild_chapters(self, cur, chunk_rows: List[Dict[str, Any]], now: str):
        chapters: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for row in chunk_rows:
            chapter_index = int(row.get("chapter_index") or 0)
            chapter_title = str(row.get("chapter_title") or f"Chapter {chapter_index + 1}")
            chapter_id = str(row.get("chapter_id") or f"chapter-{chapter_index}")
            key = (chapter_index, chapter_id)
            if key not in chapters:
                chapters[key] = {
                    "chapter_index": chapter_index,
                    "chapter_id": chapter_id,
                    "chapter_title": chapter_title,
                    "start_order_index": int(row["order_index"]),
                    "end_order_index": int(row["order_index"]),
                    "chunk_count": 0,
                    "word_count": 0,
                    "char_count": 0,
                }
            entry = chapters[key]
            entry["start_order_index"] = min(entry["start_order_index"], int(row["order_index"]))
            entry["end_order_index"] = max(entry["end_order_index"], int(row["order_index"]))
            entry["chunk_count"] += 1
            entry["word_count"] += int(row.get("word_count") or 0)
            entry["char_count"] += int(row.get("char_count") or 0)

        for entry in sorted(chapters.values(), key=lambda item: item["chapter_index"]):
            cur.execute(
                f"""
                INSERT INTO {CHAPTER_TABLE}
                (book_id, chapter_id, chapter_index, chapter_title, start_order_index, end_order_index,
                 chunk_count, word_count, char_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.book_id,
                    entry["chapter_id"],
                    entry["chapter_index"],
                    entry["chapter_title"],
                    entry["start_order_index"],
                    entry["end_order_index"],
                    entry["chunk_count"],
                    entry["word_count"],
                    entry["char_count"],
                    now,
                    now,
                ),
            )

    def _replace_ver_lancedb(self, chunk_rows: List[Dict[str, Any]], embeddings: List[np.ndarray]):
        records = []
        for row, emb in zip(chunk_rows, embeddings):
            emb_arr = np.asarray(emb, dtype=np.float32)
            record = {
                "book_id": self.book_id,
                "namespace": self.namespace,
                "hash_id": row["hash_id"],
                "content": row["content"],
                "order_index": int(row["order_index"]),
                "vector": emb_arr.tolist(),
            }
            records.append(record)
        table = self._get_vector_table()
        if table is not None:
            table.delete(f"book_id = {self.book_id} AND namespace = '{self.namespace}'")
            self.vector_table = table
        self._init_vector_table(records)

    def list_namespaces(self, prefix: str = "") -> List[str]:
        all_tables = list(self.TYPE_TO_META_TABLE.values())
        seen = set()
        with self._connect() as conn:
            cur = conn.cursor()
            for table_name in all_tables:
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,),
                )
                if not cur.fetchone():
                    continue
                if prefix:
                    cur.execute(
                        f"SELECT DISTINCT namespace FROM {table_name} WHERE book_id = ? AND namespace LIKE ?",
                        (self.book_id, f"{prefix}%"),
                    )
                else:
                    cur.execute(f"SELECT DISTINCT namespace FROM {table_name} WHERE book_id = ?", (self.book_id,))
                rows = cur.fetchall()
                for row in rows:
                    seen.add(row[0])
        return sorted(seen)

    def _vector_search_rows(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        table = self._get_vector_table()
        if table is None:
            return []
        query_vec = np.asarray(query_embedding, dtype=np.float32)
        if query_vec.ndim > 1:
            query_vec = np.squeeze(query_vec, axis=0)
        query_list = query_vec.tolist()

        try:
            search = table.search(query_list).metric("cosine")
            try:
                search = search.where(f"book_id = {self.book_id} AND namespace = '{self.namespace}'")
            except Exception:
                pass
            rows = search.limit(int(top_k)).to_list()
        except Exception:
            # Compatibility fallback for LanceDB versions with different search API.
            rows = table.search(query_list).limit(int(top_k)).to_list()
        def same_book(row):
            try:
                return int(row.get("book_id", -1)) == self.book_id
            except (TypeError, ValueError):
                return False
        return [
            row
            for row in rows
            if row.get("namespace") == self.namespace and same_book(row)
        ]

    @staticmethod
    def _row_to_similarity(row: Dict[str, Any]) -> float:
        if "_distance" in row and row["_distance"] is not None:
            return float(1.0 - float(row["_distance"]))
        for key in ("_score", "score"):
            if key in row and row[key] is not None:
                return float(row[key])
        return 0.0

    def search_topk(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        exclude_hash_ids: Optional[Set[str]] = None,
    ) -> Tuple[List[str], np.ndarray]:
        if top_k <= 0:
            return [], np.array([], dtype=np.float32)
        exclude_hash_ids = exclude_hash_ids or set()
        rows = self._vector_search_rows(query_embedding=query_embedding, top_k=max(top_k * 3, top_k))
        hash_ids: List[str] = []
        scores: List[float] = []
        for row in rows:
            hash_id = row.get("hash_id")
            if not hash_id or hash_id in exclude_hash_ids:
                continue
            hash_ids.append(hash_id)
            scores.append(self._row_to_similarity(row))
            if len(hash_ids) >= top_k:
                break
        return hash_ids, np.array(scores, dtype=np.float32)

    def rank_indices_by_vector(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        exclude_hash_ids: Optional[Set[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        hash_ids, scores = self.search_topk(
            query_embedding=query_embedding,
            top_k=top_k,
            exclude_hash_ids=exclude_hash_ids,
        )
        if not hash_ids:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float32)
        indices = np.array(
            [self.hash_id_to_idx[h] for h in hash_ids if h in self.hash_id_to_idx],
            dtype=np.intp,
        )
        kept_scores = np.array(
            [scores[i] for i, h in enumerate(hash_ids) if h in self.hash_id_to_idx],
            dtype=np.float32,
        )
        return indices, kept_scores
