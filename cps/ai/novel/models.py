# -*- coding: utf-8 -*-
"""Domain models for the narrative intelligence engine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return "{}-{}".format(prefix, hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24])


@dataclass(frozen=True)
class Chapter:
    chapter_id: str
    index: int
    title: str
    paragraphs: List[str]


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    book_id: int
    order_id: int
    chapter_id: str
    chapter_index: int
    chapter_title: str
    text: str
    paragraph_start: int
    paragraph_end: int
    text_hash: str

    @classmethod
    def create(
        cls,
        *,
        book_id: int,
        order_id: int,
        chapter: Chapter,
        text: str,
        paragraph_start: int,
        paragraph_end: int,
    ) -> "Passage":
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return cls(
            chunk_id=stable_id("chunk", book_id, order_id, text_hash),
            book_id=int(book_id),
            order_id=int(order_id),
            chapter_id=chapter.chapter_id,
            chapter_index=chapter.index,
            chapter_title=chapter.title,
            text=text,
            paragraph_start=paragraph_start,
            paragraph_end=paragraph_end,
            text_hash=text_hash,
        )


@dataclass(frozen=True)
class ExtractedEntity:
    key: str
    name: str
    entity_type: str
    aliases: List[str]
    description: str
    mention_chunk_ids: List[str]


@dataclass(frozen=True)
class ExtractedEvent:
    key: str
    title: str
    summary: str
    event_type: str
    story_time: str
    location: str
    epistemic_status: str
    participant_keys: List[str]
    evidence_chunk_ids: List[str]
    confidence: float


@dataclass(frozen=True)
class EventRelation:
    source_event_key: str
    target_event_key: str
    relation_type: str
    evidence_chunk_id: str
    confidence: float


@dataclass(frozen=True)
class NarrativeExtraction:
    entities: List[ExtractedEntity] = field(default_factory=list)
    events: List[ExtractedEvent] = field(default_factory=list)
    relations: List[EventRelation] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], namespace: str = "") -> "NarrativeExtraction":
        entity_keys = set()
        entities: List[ExtractedEntity] = []
        for item in data.get("entities", []) or []:
            if not isinstance(item, Mapping):
                continue
            key = str(item.get("key") or item.get("name") or "").strip()
            name = str(item.get("name") or "").strip()
            if not key or not name:
                continue
            entity_keys.add(key)
            entities.append(
                ExtractedEntity(
                    key=key,
                    name=name,
                    entity_type=str(item.get("entity_type") or "unknown"),
                    aliases=[str(value) for value in item.get("aliases", []) if str(value).strip()],
                    description=str(item.get("description") or ""),
                    mention_chunk_ids=[str(value) for value in item.get("mention_chunk_ids", [])],
                )
            )

        events: List[ExtractedEvent] = []
        event_keys = set()
        for item in data.get("events", []) or []:
            if not isinstance(item, Mapping):
                continue
            local_key = str(item.get("key") or "").strip()
            if not local_key:
                continue
            key = "{}:{}".format(namespace, local_key) if namespace else local_key
            event_keys.add(local_key)
            events.append(
                ExtractedEvent(
                    key=key,
                    title=str(item.get("title") or local_key),
                    summary=str(item.get("summary") or ""),
                    event_type=str(item.get("event_type") or "event"),
                    story_time=str(item.get("story_time") or "unknown"),
                    location=str(item.get("location") or "unknown"),
                    epistemic_status=str(item.get("epistemic_status") or "uncertain"),
                    participant_keys=[str(value) for value in item.get("participant_keys", []) if str(value) in entity_keys],
                    evidence_chunk_ids=[str(value) for value in item.get("evidence_chunk_ids", [])],
                    confidence=max(0.0, min(1.0, float(item.get("confidence") or 0.0))),
                )
            )

        relations: List[EventRelation] = []
        for item in data.get("relations", []) or []:
            if not isinstance(item, Mapping):
                continue
            source = str(item.get("source_event_key") or "")
            target = str(item.get("target_event_key") or "")
            if source not in event_keys or target not in event_keys:
                continue
            relations.append(
                EventRelation(
                    source_event_key="{}:{}".format(namespace, source) if namespace else source,
                    target_event_key="{}:{}".format(namespace, target) if namespace else target,
                    relation_type=str(item.get("relation_type") or "related"),
                    evidence_chunk_id=str(item.get("evidence_chunk_id") or ""),
                    confidence=max(0.0, min(1.0, float(item.get("confidence") or 0.0))),
                )
            )
        return cls(entities=entities, events=events, relations=relations)


@dataclass(frozen=True)
class NarrativeMemory:
    memory_id: str
    level: str
    title: str
    summary: str
    position_start: int
    position_end: int
    state: Dict[str, Any]
    unresolved_threads: List[str]
    source_chunk_ids: List[str]

    @property
    def embedding_text(self) -> str:
        details = json.dumps(self.state, ensure_ascii=False, sort_keys=True)
        threads = "；".join(self.unresolved_threads)
        return "{}\n{}\n人物状态:{}\n未解线索:{}".format(self.title, self.summary, details, threads)


@dataclass(frozen=True)
class RetrievalPlan:
    answer_type: str
    entities: List[str]
    keywords: List[str]
    temporal_scope: str
    relation_types: List[str]
    channels: List[str]
    probes: List[str]


@dataclass
class Evidence:
    evidence_id: str
    kind: str
    ref_id: str
    text: str
    score: float
    order_start: int
    order_end: int
    chapter_title: str
    source_chunk_ids: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def citation(self) -> str:
        if self.order_start == self.order_end:
            return "[order_id={}]".format(self.order_start)
        return "[order_id={}-{}]".format(self.order_start, self.order_end)


EXTRACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "key": {"type": "string"},
                    "name": {"type": "string"},
                    "entity_type": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                    "mention_chunk_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["key", "name", "entity_type", "aliases", "description", "mention_chunk_ids"],
            },
        },
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "key": {"type": "string"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "event_type": {"type": "string"},
                    "story_time": {"type": "string"},
                    "location": {"type": "string"},
                    "epistemic_status": {
                        "type": "string",
                        "enum": ["narrator_fact", "character_belief", "reported_claim", "inference", "uncertain"],
                    },
                    "participant_keys": {"type": "array", "items": {"type": "string"}},
                    "evidence_chunk_ids": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["key", "title", "summary", "event_type", "story_time", "location", "epistemic_status", "participant_keys", "evidence_chunk_ids", "confidence"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "source_event_key": {"type": "string"},
                    "target_event_key": {"type": "string"},
                    "relation_type": {"type": "string"},
                    "evidence_chunk_id": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["source_event_key", "target_event_key", "relation_type", "evidence_chunk_id", "confidence"],
            },
        },
    },
    "required": ["entities", "events", "relations"],
}


PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer_type": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "temporal_scope": {"type": "string"},
        "relation_types": {"type": "array", "items": {"type": "string"}},
        "channels": {"type": "array", "items": {"type": "string"}},
        "probes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer_type", "entities", "keywords", "temporal_scope", "relation_types", "channels", "probes"],
}


COVERAGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sufficient": {"type": "boolean"},
        "missing": {"type": "array", "items": {"type": "string"}},
        "next_queries": {"type": "array", "items": {"type": "string"}},
        "contradictions": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["sufficient", "missing", "next_queries", "contradictions", "confidence"],
}


ANSWER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string"},
        "evidence_chunk_ids": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
        "spoiler": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["answer", "evidence_chunk_ids", "uncertainties", "spoiler", "confidence"],
}
