# -*- coding: utf-8 -*-
"""Transactional story-aware indexing pipeline."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..config import ModelSettings, NovelSettings
from ..providers import ModelGateway
from .ingestion import load_chapters, segment_chapters, source_fingerprint
from .models import EXTRACTION_SCHEMA, NarrativeExtraction, NarrativeMemory, Passage, stable_id
from .prompts import EXTRACTION_SYSTEM, MEMORY_SYSTEM
from .repository import NovelRepository, utc_now
from .vector_store import NarrativeVectorStore


ProgressCallback = Optional[Callable[[str, str, float], None]]


MEMORY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "character_states": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "entity": {"type": "string"},
                    "before_state": {"type": "string"},
                    "after_state": {"type": "string"},
                    "change": {"type": "string"},
                    "epistemic_status": {
                        "type": "string",
                        "enum": ["narrator_fact", "character_belief", "inference", "uncertain"],
                    },
                    "evidence_chunk_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["entity", "before_state", "after_state", "change", "epistemic_status", "evidence_chunk_ids"],
            },
        },
        "unresolved_threads": {"type": "array", "items": {"type": "string"}},
        "event_links": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "source_event_id": {"type": "string"},
                    "target_event_id": {"type": "string"},
                    "relation_type": {"type": "string"},
                    "evidence_chunk_id": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["source_event_id", "target_event_id", "relation_type", "evidence_chunk_id", "confidence"],
            },
        },
    },
    "required": ["title", "summary", "character_states", "unresolved_threads", "event_links"],
}


class NarrativeIndexer:
    def __init__(
        self,
        *,
        repository: NovelRepository,
        vector_store: NarrativeVectorStore,
        gateway: ModelGateway,
        model_settings: ModelSettings,
        novel_settings: NovelSettings,
    ):
        self.repository = repository
        self.vector_store = vector_store
        self.gateway = gateway
        self.models = model_settings
        self.settings = novel_settings

    def build(
        self,
        *,
        book_id: int,
        run_id: str,
        source_path: str,
        force: bool = False,
        progress_callback: ProgressCallback = None,
    ) -> Dict[str, Any]:
        try:
            self._phase(run_id, book_id, "LOAD_SOURCE", "Reading book structure", 0.04, progress_callback)
            fingerprint = source_fingerprint(source_path)
            chapters = load_chapters(source_path)
            passages = segment_chapters(
                book_id=int(book_id), chapters=chapters,
                target_chars=self.settings.chunk_chars,
                overlap_paragraphs=self.settings.chunk_overlap_paragraphs,
            )
            if not passages:
                raise ValueError("No readable narrative passages found")
            self.repository.update_run(
                run_id, source_fingerprint=fingerprint, total_chunks=len(passages),
                status="RUNNING", indexed_chunks=0,
            )

            existing = self.repository.find_matching_active(book_id, fingerprint, self.models.embedding_model)
            if existing and not force:
                self.repository.update_run(
                    run_id, status="SUCCESS", phase="DONE", message="Existing index is current",
                    total_chunks=existing["total_chunks"], indexed_chunks=existing["indexed_chunks"],
                    finished_at=utc_now(),
                )
                self.repository.append_log(run_id, book_id, "DONE", "Source and embedding signature unchanged; reused active index")
                return self.repository.status(book_id)

            self._phase(run_id, book_id, "STORE_PASSAGES", "Persisting ordered passages", 0.10, progress_callback)
            self.repository.insert_passages(run_id, passages)

            table_name = self.vector_store.table_name(book_id, run_id)
            self._phase(run_id, book_id, "EMBED_PASSAGES", "Embedding original passages", 0.16, progress_callback)
            chunk_records = self._embed_passages(run_id, passages)
            self.vector_store.create(table_name, chunk_records)
            self.repository.update_run(run_id, vector_table=table_name, indexed_chunks=len(passages))

            self._phase(run_id, book_id, "EXTRACT_WORLD", "Extracting entities, events, and local causal links", 0.32, progress_callback)
            failures = 0
            batches = list(_batched(passages, self.settings.extraction_batch_size))
            for batch_index, batch in enumerate(batches):
                try:
                    extraction = self._extract_batch(batch, "b{:05d}".format(batch_index))
                    self.repository.insert_extraction(run_id, extraction)
                except Exception as exc:  # tolerate isolated malformed chapters/providers
                    failures += 1
                    self.repository.append_log(run_id, book_id, "EXTRACT_WORLD", "batch {} failed: {}".format(batch_index, exc), "WARNING")
                fraction = (batch_index + 1) / max(1, len(batches))
                self._notify(progress_callback, "EXTRACT_WORLD", "Processed {}/{} extraction batches".format(batch_index + 1, len(batches)), 0.32 + fraction * 0.25)
            if failures and failures / max(1, len(batches)) > 0.5:
                raise RuntimeError("More than half of narrative extraction batches failed")

            self._phase(run_id, book_id, "BUILD_MEMORY", "Building chapter and story-arc memory", 0.60, progress_callback)
            memories = self._build_memories(run_id, passages)
            for memory in memories:
                self.repository.insert_memory(run_id, memory)

            self._phase(run_id, book_id, "EMBED_WORLD", "Embedding events and hierarchical memory", 0.82, progress_callback)
            semantic_records = self._embed_world(run_id, passages)
            self.vector_store.add(table_name, semantic_records)

            self._phase(run_id, book_id, "PROMOTE", "Atomically promoting the completed index", 0.95, progress_callback)
            self.repository.promote_run(run_id)
            self._cleanup(book_id)
            self._notify(progress_callback, "DONE", "Narrative index ready", 1.0)
            self.repository.append_log(run_id, book_id, "DONE", "Narrative index promoted successfully")
            return self.repository.status(book_id)
        except Exception as exc:
            self.repository.update_run(
                run_id, status="FAILED", phase="FAILED", message="Narrative index failed",
                last_error=str(exc), finished_at=utc_now(),
            )
            self.repository.append_log(run_id, book_id, "FAILED", str(exc), "ERROR")
            self._notify(progress_callback, "FAILED", str(exc), 1.0)
            raise

    def _extract_batch(self, passages: Sequence[Passage], namespace: str) -> NarrativeExtraction:
        payload = [
            {
                "chunk_id": item.chunk_id,
                "order_id": item.order_id,
                "chapter": item.chapter_title,
                "text": item.text,
            }
            for item in passages
        ]
        data = self.gateway.structured(
            system=EXTRACTION_SYSTEM,
            prompt="Extract the narrative world model from these ordered passages:\n{}".format(json.dumps(payload, ensure_ascii=False)),
            schema_name="narrative_extraction",
            schema=EXTRACTION_SCHEMA,
            model=self.models.extraction_model,
        )
        return NarrativeExtraction.from_dict(data, namespace=namespace)

    def _build_memories(self, run_id: str, passages: Sequence[Passage]) -> List[NarrativeMemory]:
        by_chapter: Dict[int, List[Passage]] = {}
        for passage in passages:
            by_chapter.setdefault(passage.chapter_index, []).append(passage)
        chapter_memories: List[NarrativeMemory] = []
        for chapter_index, chapter_passages in sorted(by_chapter.items()):
            events = self.repository.chapter_events(
                run_id, chapter_passages[0].order_id, chapter_passages[-1].order_id
            )
            evidence = {
                "chapter": chapter_passages[0].chapter_title,
                "passages": [
                    {"chunk_id": item.chunk_id, "order_id": item.order_id, "text": item.text[:800]}
                    for item in chapter_passages
                ],
                "events": self._compact_events(events, 160),
            }
            data = self._memory_call("Build a chapter memory:\n{}".format(json.dumps(evidence, ensure_ascii=False)))
            memory = self._memory_from_data(
                run_id=run_id, level="chapter", data=data,
                position_start=chapter_passages[0].order_id,
                position_end=chapter_passages[-1].order_id,
                source_chunk_ids=[item.chunk_id for item in chapter_passages],
                fallback_title=chapter_passages[0].chapter_title,
            )
            self.repository.insert_cross_event_links(run_id, data.get("event_links", []))
            chapter_memories.append(memory)

        arc_memories: List[NarrativeMemory] = []
        for arc_index, group in enumerate(_batched(chapter_memories, self.settings.arc_chapters)):
            arc_events = self.repository.chapter_events(
                run_id, group[0].position_start, group[-1].position_end
            )
            data = self._memory_call(
                "Build a cross-chapter story-arc memory. Event IDs may be linked across chapters:\n{}".format(
                    json.dumps(
                        {
                            "memories": [self._compact_memory(item) for item in group],
                            "events": self._compact_events(arc_events, 240),
                        },
                        ensure_ascii=False,
                    )
                )
            )
            source_ids = [chunk_id for item in group for chunk_id in item.source_chunk_ids]
            memory = self._memory_from_data(
                run_id=run_id, level="arc", data=data,
                position_start=group[0].position_start, position_end=group[-1].position_end,
                source_chunk_ids=source_ids,
                fallback_title="Story arc {}".format(arc_index + 1),
            )
            self.repository.insert_cross_event_links(run_id, data.get("event_links", []))
            arc_memories.append(memory)

        book_inputs = arc_memories or chapter_memories
        if book_inputs:
            book_events = self.repository.chapter_events(
                run_id, book_inputs[0].position_start, book_inputs[-1].position_end
            )
            data = self._memory_call(
                "Build a whole-book memory that preserves chronology, causal structure, revelations, and unresolved ambiguity:\n{}".format(
                    json.dumps(
                        {
                            "memories": [self._compact_memory(item) for item in book_inputs],
                            "events": self._compact_events(book_events, 320),
                        },
                        ensure_ascii=False,
                    )
                )
            )
            source_ids = [chunk_id for item in book_inputs for chunk_id in item.source_chunk_ids]
            book_memory = self._memory_from_data(
                run_id=run_id, level="book", data=data,
                position_start=book_inputs[0].position_start,
                position_end=book_inputs[-1].position_end,
                source_chunk_ids=source_ids,
                fallback_title="Whole-book narrative memory",
            )
            self.repository.insert_cross_event_links(run_id, data.get("event_links", []))
            return chapter_memories + arc_memories + [book_memory]
        return chapter_memories

    def _memory_call(self, prompt: str) -> Dict[str, Any]:
        return self.gateway.structured(
            system=MEMORY_SYSTEM, prompt=prompt, schema_name="narrative_memory",
            schema=MEMORY_SCHEMA, model=self.models.extraction_model,
        )

    @staticmethod
    def _compact_memory(memory: NarrativeMemory) -> Dict[str, Any]:
        return {
            "memory_id": memory.memory_id,
            "title": memory.title,
            "summary": memory.summary,
            "position_start": memory.position_start,
            "position_end": memory.position_end,
            "state": memory.state,
            "unresolved_threads": memory.unresolved_threads,
        }

    @staticmethod
    def _compact_events(events: Sequence[Mapping[str, Any]], limit: int) -> List[Dict[str, Any]]:
        if len(events) > limit:
            priority_types = {
                "revelation", "discovery", "decision", "death", "conflict",
                "resolution", "betrayal", "clue", "foreshadowing",
            }
            priority = [event for event in events if str(event.get("event_type", "")).lower() in priority_types]
            remainder = [event for event in events if event not in priority]
            available = max(0, limit - min(limit, len(priority)))
            if available and len(remainder) > available:
                step = len(remainder) / float(available)
                remainder = [remainder[min(len(remainder) - 1, int(index * step))] for index in range(available)]
            events = (priority[:limit] + remainder[:available])[:limit]
        return [
            {
                "event_id": event.get("event_id"),
                "title": event.get("title"),
                "summary": str(event.get("summary") or "")[:500],
                "event_type": event.get("event_type"),
                "story_time": event.get("story_time"),
                "epistemic_status": event.get("epistemic_status"),
                "order_start": event.get("order_start"),
                "order_end": event.get("order_end"),
                "evidence_chunk_ids": json.loads(event.get("evidence_chunk_ids_json") or "[]")[:12],
            }
            for event in events
        ]

    @staticmethod
    def _memory_from_data(
        *, run_id: str, level: str, data: Mapping[str, Any], position_start: int,
        position_end: int, source_chunk_ids: Sequence[str], fallback_title: str,
    ) -> NarrativeMemory:
        states = {
            str(item.get("entity") or "unknown"): {
                "before_state": str(item.get("before_state") or "unknown"),
                "after_state": str(item.get("after_state") or item.get("state") or "unknown"),
                "change": str(item.get("change") or ""),
                "epistemic_status": str(item.get("epistemic_status") or "uncertain"),
                "evidence_chunk_ids": list(item.get("evidence_chunk_ids") or []),
            }
            for item in data.get("character_states", [])
            if isinstance(item, Mapping)
        }
        return NarrativeMemory(
            memory_id=stable_id("memory", run_id, level, position_start, position_end),
            level=level, title=str(data.get("title") or fallback_title),
            summary=str(data.get("summary") or ""), position_start=int(position_start),
            position_end=int(position_end), state=states,
            unresolved_threads=[str(value) for value in data.get("unresolved_threads", [])],
            source_chunk_ids=list(dict.fromkeys(source_chunk_ids)),
        )

    def _embed_passages(self, run_id: str, passages: Sequence[Passage]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for batch in _batched(passages, self.settings.embedding_batch_size):
            vectors = self.gateway.embed([item.text for item in batch])
            if len(vectors) != len(batch):
                raise ValueError("Embedding provider returned an unexpected vector count")
            records.extend(
                self._vector_record(
                    run_id, "chunk", item.chunk_id, item.text, item.order_id,
                    item.order_id, item.chapter_title, vector,
                )
                for item, vector in zip(batch, vectors)
            )
        return records

    def _embed_world(self, run_id: str, passages: Sequence[Passage]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        last_order = passages[-1].order_id if passages else 0
        events = self.repository.chapter_events(run_id, 0, last_order)
        memories = self.repository.memories(run_id)
        documents: List[Tuple[str, str, str, int, int, str]] = []
        for event in events:
            text = "{}\n{}\n时间:{} 地点:{} 证据属性:{}".format(
                event["title"], event["summary"], event["story_time"],
                event["location"], event["epistemic_status"],
            )
            documents.append(("event", event["event_id"], text, event["order_start"], event["order_end"], event["chapter_title"]))
        for memory in memories:
            text = "{}\n{}\n人物状态:{}\n未解线索:{}".format(memory["title"], memory["summary"], memory["state_json"], memory["unresolved_json"])
            documents.append(("memory", memory["memory_id"], text, memory["position_start"], memory["position_end"], memory["title"]))
        for batch in _batched(documents, self.settings.embedding_batch_size):
            vectors = self.gateway.embed([item[2] for item in batch])
            records.extend(
                self._vector_record(run_id, item[0], item[1], item[2], item[3], item[4], item[5], vector)
                for item, vector in zip(batch, vectors)
            )
        return records

    @staticmethod
    def _vector_record(run_id: str, kind: str, ref_id: str, text: str, order_start: int, order_end: int, chapter_title: str, vector: Sequence[float]) -> Dict[str, Any]:
        return {
            "run_id": run_id, "kind": kind, "ref_id": ref_id, "text": text,
            "order_start": int(order_start), "order_end": int(order_end),
            "chapter_title": chapter_title or "", "vector": list(vector),
        }

    def _cleanup(self, book_id: int) -> None:
        for run in self.repository.obsolete_runs(book_id, self.settings.keep_index_runs):
            if run.get("vector_table"):
                self.vector_store.drop(run["vector_table"])
            self.repository.delete_run(run["run_id"])

    def _phase(self, run_id: str, book_id: int, phase: str, message: str, progress: float, callback: ProgressCallback) -> None:
        self.repository.update_run(run_id, status="RUNNING", phase=phase, message=message)
        self.repository.append_log(run_id, book_id, phase, message)
        self._notify(callback, phase, message, progress)

    @staticmethod
    def _notify(callback: ProgressCallback, phase: str, message: str, progress: float) -> None:
        if callback:
            callback(phase, message, max(0.0, min(1.0, progress)))


def _batched(values: Sequence[Any], size: int):
    for start in range(0, len(values), max(1, int(size))):
        yield values[start : start + max(1, int(size))]
