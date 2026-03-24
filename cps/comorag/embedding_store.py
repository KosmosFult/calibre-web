import numpy as np
import os
from typing import Union, Optional, List, Dict, Set, Any, Tuple, Literal
import logging
from copy import deepcopy
import datetime
import sqlite3
import re

import lancedb

from .utils.misc_utils import compute_mdhash_id, NerRawOutput, TripleRawOutput
from .. import ai_db

logger = logging.getLogger(__name__)

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
        self.vector_db_path = os.path.join(os.path.dirname(self.db_path), "lancedb")
        os.makedirs(self.vector_db_path, exist_ok=True)
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
        if self._get_vector_table() is None:
            self.vector_table = self.vector_db.create_table(self.vector_table_name, data=records, mode="overwrite")
        else:
            self.vector_table.add(records)

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

    def insert_strings(self, texts: List[str]):
        nodes_dict = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

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

        self._upsert(missing_ids, texts_to_encode, missing_embeddings)

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
            conn.commit()

    def _load_data(self):
        self._ensure_tables()
        self.hash_ids, self.texts, self.embeddings = [], [], []
        self.hash_id_to_idx, self.hash_id_to_row = {}, {}
        self.hash_id_to_text, self.text_to_hash_id = {}, {}
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT hash_id, content, embedding
                FROM {self.meta_table}
                WHERE book_id = ? AND namespace = ?
                ORDER BY order_index ASC
                """,
                (self.book_id, self.namespace),
            )
            rows = cur.fetchall()

        for hash_id, content, embedding_blob in rows:
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            self.hash_ids.append(hash_id)
            self.texts.append(content)
            self.embeddings.append(embedding)

        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_row = {
            h: {"hash_id": h, "content": t}
            for h, t in zip(self.hash_ids, self.texts)
        }
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
        self.hash_id_to_row = {h: {"hash_id": h, "content": t} for h, t in zip(self.hash_ids, self.texts)}
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        logger.info(
            "Saved %s records to ai.db table=%s namespace=%s",
            len(self.hash_ids),
            self.meta_table,
            self.namespace,
        )

    def _upsert(self, hash_ids, texts, embeddings):
        # 最后TODO: 原子性

        if not hash_ids:
            return
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
                    (book_id, namespace, hash_id, content, embedding, embedding_dim, order_index, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.book_id,
                        self.namespace,
                        hash_id,
                        text,
                        emb_arr.tobytes(),
                        int(emb_arr.shape[0]),
                        order_index,
                        now,
                        now,
                    ),
                )
                order_entries.append((hash_id, order_index))
            conn.commit()

        self._upsert_lancedb(hash_ids=hash_ids, texts=texts, embeddings=embeddings, order_entries=order_entries)
        self.embeddings.extend([np.asarray(e, dtype=np.float32) for e in embeddings])
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)
        logger.info("Saving new records.")
        self._save_data()

    def _upsert_lancedb(
        self,
        hash_ids: List[str],
        texts: List[str],
        embeddings: List[np.ndarray],
        order_entries: List[Tuple[str, int]],
    ):
        if not hash_ids:
            return
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
        embeddings = np.array(self.embeddings, dtype=dtype)[indices]

        return embeddings

    def get_hash_id_to_order(self) -> Dict[str, int]:
        """
        Returns a dictionary mapping each hash value to its sequential position in the original text list.
        
        Returns:
            Dict[str, int]: Mapping from hash values to sequential positions, e.g., {'hash1': 0, 'hash2': 1, ...}
        """
        # Since the texts list maintains insertion order, we can use it to build the order mapping
        return {h: idx for idx, h in enumerate(self.hash_ids)}

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
