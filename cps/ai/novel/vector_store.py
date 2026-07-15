# -*- coding: utf-8 -*-
"""LanceDB adapter for semantic evidence channels."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

import lancedb


_SAFE_TABLE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")


class NarrativeVectorStore:
    def __init__(self, directory: str):
        self.directory = directory
        self.database = lancedb.connect(directory)

    @staticmethod
    def table_name(book_id: int, run_id: str) -> str:
        suffix = re.sub(r"[^A-Za-z0-9]", "", run_id)[:20]
        return "novel_{}_{}".format(int(book_id), suffix)

    @staticmethod
    def _validate(name: str) -> str:
        if not _SAFE_TABLE.match(name):
            raise ValueError("Unsafe LanceDB table name")
        return name

    def create(self, table_name: str, records: Sequence[Mapping[str, Any]]) -> None:
        name = self._validate(table_name)
        if not records:
            raise ValueError("Cannot create an empty vector index")
        self.database.create_table(name, data=[dict(row) for row in records], mode="overwrite")

    def add(self, table_name: str, records: Sequence[Mapping[str, Any]]) -> None:
        if records:
            self.database.open_table(self._validate(table_name)).add([dict(row) for row in records])

    def search(
        self,
        table_name: str,
        query_vector: Sequence[float],
        *,
        limit: int,
        kinds: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        table = self.database.open_table(self._validate(table_name))
        candidate_limit = max(int(limit), int(limit) * (3 if kinds else 1))
        rows = table.search(list(query_vector)).limit(candidate_limit).to_list()
        if kinds:
            allowed = set(kinds)
            rows = [row for row in rows if row.get("kind") in allowed]
        return rows[: int(limit)]

    def drop(self, table_name: str) -> None:
        name = self._validate(table_name)
        try:
            self.database.drop_table(name)
        except (KeyError, ValueError):
            return
