# -*- coding: utf-8 -*-
"""Plan–retrieve–critic–probe loop for evidence-grounded novel QA."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Sequence

from ..config import ModelSettings, NovelSettings
from ..providers import ModelGateway
from .models import ANSWER_SCHEMA, COVERAGE_SCHEMA, PLAN_SCHEMA, Evidence, RetrievalPlan
from .prompts import ANSWER_SYSTEM, COVERAGE_SYSTEM, PLANNER_SYSTEM
from .repository import NovelRepository
from .vector_store import NarrativeVectorStore


class NovelReasoner:
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

    def answer(self, book_id: int, question: str) -> Dict[str, Any]:
        active = self.repository.active_run(book_id)
        if not active:
            return {"status": "not_indexed", "book_id": int(book_id), "message": "Narrative index is not ready"}
        plan = self._plan(question)
        evidence_by_key: Dict[str, Evidence] = {}
        round_traces: List[Dict[str, Any]] = []
        next_queries = [question] + plan.probes[:3]
        coverage: Dict[str, Any] = {
            "sufficient": False, "missing": [], "next_queries": [],
            "contradictions": [], "confidence": 0.0,
        }

        for round_index in range(self.settings.max_reasoning_rounds):
            found = self._retrieve(active, plan, next_queries)
            for item in found:
                key = "{}:{}".format(item.kind, item.ref_id)
                existing = evidence_by_key.get(key)
                if existing is None or item.score > existing.score:
                    evidence_by_key[key] = item
            evidence = sorted(evidence_by_key.values(), key=lambda item: item.score, reverse=True)
            evidence = evidence[: self.settings.retrieval_limit * 2]
            coverage = self._judge(question, evidence)
            round_traces.append(
                {
                    "round": round_index + 1,
                    "queries": list(next_queries),
                    "evidence_ids": [item.evidence_id for item in found],
                    "coverage": coverage,
                }
            )
            if coverage.get("sufficient"):
                break
            next_queries = [str(value) for value in coverage.get("next_queries", []) if str(value).strip()][:4]
            if not next_queries:
                break

        final_evidence = sorted(evidence_by_key.values(), key=lambda item: item.score, reverse=True)
        final_evidence = final_evidence[: self.settings.retrieval_limit]
        answer = self._compose(question, final_evidence, coverage)
        trace_id = self.repository.save_trace(
            int(book_id), active["run_id"], question, asdict(plan), round_traces, answer
        )
        mentioned_orders = set()
        for item in final_evidence:
            if item.kind == "memory" or item.order_end - item.order_start > 20:
                mentioned_orders.update([item.order_start, item.order_end])
            else:
                mentioned_orders.update(range(item.order_start, item.order_end + 1))
        return {
            "status": "success",
            "book_id": int(book_id),
            "question": question,
            "answer": answer.get("answer", ""),
            "answer_draft": answer.get("answer", ""),
            "confidence": float(answer.get("confidence") or 0.0),
            "uncertainties": answer.get("uncertainties", []),
            "spoiler": bool(answer.get("spoiler")),
            "evidence": [self._evidence_payload(item) for item in final_evidence],
            "trace": {
                "trace_id": trace_id,
                "rounds": round_traces,
                "mentioned_order_ids": sorted(mentioned_orders),
                "coverage": coverage,
            },
            "index": self.repository.status(book_id),
        }

    def _plan(self, question: str) -> RetrievalPlan:
        try:
            data = self.gateway.structured(
                system=PLANNER_SYSTEM,
                prompt="Plan retrieval for this novel question:\n{}".format(question),
                schema_name="novel_retrieval_plan",
                schema=PLAN_SCHEMA,
                model=self.models.reasoning_model,
            )
            return RetrievalPlan(
                answer_type=str(data.get("answer_type") or "factual"),
                entities=[str(value) for value in data.get("entities", [])],
                keywords=[str(value) for value in data.get("keywords", [])],
                temporal_scope=str(data.get("temporal_scope") or "whole_book"),
                relation_types=[str(value) for value in data.get("relation_types", [])],
                channels=[str(value) for value in data.get("channels", [])] or ["episodic", "semantic", "veridical"],
                probes=[str(value) for value in data.get("probes", [])],
            )
        except Exception:
            terms = [value for value in re.split(r"[\s,，。？！?、]+", question) if len(value) > 1]
            return RetrievalPlan(
                answer_type="factual", entities=[], keywords=terms[:8],
                temporal_scope="whole_book",
                relation_types=["before", "causes", "reveals"],
                channels=["episodic", "semantic", "veridical"], probes=[],
            )

    def _retrieve(self, active: Mapping[str, Any], plan: RetrievalPlan, queries: Sequence[str]) -> List[Evidence]:
        run_id = str(active["run_id"])
        candidates: Dict[str, Evidence] = {}
        vectors = self.gateway.embed(list(queries)) if queries else []
        for query_index, vector in enumerate(vectors):
            rows = self.vector_store.search(
                str(active["vector_table"]), vector,
                limit=self.settings.retrieval_limit,
            )
            for rank, row in enumerate(rows):
                evidence = self._from_vector(run_id, row, rank, query_index)
                if evidence:
                    self._merge(candidates, evidence)

        lexical_terms = list(dict.fromkeys(plan.entities + plan.keywords + list(queries)))
        for rank, row in enumerate(
            self.repository.lexical_search(run_id, lexical_terms, self.settings.retrieval_limit)
        ):
            evidence = self._from_chunk_row(row, score=0.35 + 0.8 / (60 + rank))
            self._merge(candidates, evidence)

        for rank, row in enumerate(
            self.repository.graph_search(
                run_id, plan.entities, self.settings.graph_hops, self.settings.retrieval_limit
            )
        ):
            evidence = self._from_event_row(row, score=0.50 + 1.0 / (60 + rank))
            self._merge(candidates, evidence)

        return sorted(candidates.values(), key=lambda item: item.score, reverse=True)

    def _from_vector(self, run_id: str, hit: Mapping[str, Any], rank: int, query_index: int) -> Optional[Evidence]:
        kind = str(hit.get("kind") or "")
        ref_id = str(hit.get("ref_id") or "")
        row = self.repository.resolve_reference(run_id, kind, ref_id)
        if not row:
            return None
        distance = max(0.0, float(hit.get("_distance") or 0.0))
        score = 1.0 / (60 + rank) + 0.55 / (1.0 + distance)
        if kind == "chunk":
            evidence = self._from_chunk_row(row, score)
        elif kind == "event":
            evidence = self._from_event_row(row, score)
        else:
            evidence = self._from_memory_row(row, score)
        evidence.metadata["semantic_query"] = query_index
        evidence.metadata["distance"] = distance
        return evidence

    @staticmethod
    def _from_chunk_row(row: Mapping[str, Any], score: float) -> Evidence:
        order_id = int(row["order_id"])
        chunk_id = str(row["chunk_id"])
        return Evidence(
            evidence_id="chunk:{}".format(chunk_id), kind="chunk", ref_id=chunk_id,
            text=str(row["text"]), score=score, order_start=order_id, order_end=order_id,
            chapter_title=str(row["chapter_title"]), source_chunk_ids=[chunk_id],
        )

    @staticmethod
    def _from_event_row(row: Mapping[str, Any], score: float) -> Evidence:
        source_ids = json.loads(row.get("evidence_chunk_ids_json") or "[]")
        text = "事件：{}\n{}\n叙事时间：{}；地点：{}；证据属性：{}".format(
            row["title"], row["summary"], row["story_time"], row["location"],
            row.get("epistemic_status", "uncertain"),
        )
        graph_links = row.get("_graph_links") or []
        if graph_links:
            relation_lines = [
                "{} --{}--> {} (confidence={:.2f})".format(
                    link["source_title"], link["relation_type"], link["target_title"],
                    float(link["confidence"]),
                )
                for link in graph_links[:12]
            ]
            text += "\n图关系：\n" + "\n".join(relation_lines)
        return Evidence(
            evidence_id="event:{}".format(row["event_id"]), kind="event",
            ref_id=str(row["event_id"]), text=text, score=score,
            order_start=int(row["order_start"]), order_end=int(row["order_end"]),
            chapter_title=str(row["chapter_title"]), source_chunk_ids=source_ids,
            metadata={
                "event_type": row["event_type"], "confidence": row["confidence"],
                "epistemic_status": row.get("epistemic_status", "uncertain"),
            },
        )

    @staticmethod
    def _from_memory_row(row: Mapping[str, Any], score: float) -> Evidence:
        source_ids = json.loads(row.get("source_chunk_ids_json") or "[]")
        text = "{}\n{}\n人物状态：{}\n未解线索：{}".format(
            row["title"], row["summary"], row["state_json"], row["unresolved_json"]
        )
        return Evidence(
            evidence_id="memory:{}".format(row["memory_id"]), kind="memory",
            ref_id=str(row["memory_id"]), text=text, score=score,
            order_start=int(row["position_start"]), order_end=int(row["position_end"]),
            chapter_title=str(row["title"]), source_chunk_ids=source_ids,
            metadata={"level": row["level"]},
        )

    @staticmethod
    def _merge(candidates: Dict[str, Evidence], evidence: Evidence) -> None:
        key = "{}:{}".format(evidence.kind, evidence.ref_id)
        existing = candidates.get(key)
        if existing is None:
            candidates[key] = evidence
        else:
            existing.score += evidence.score
            existing.metadata.update(evidence.metadata)
            if len(evidence.text) > len(existing.text):
                existing.text = evidence.text

    def _judge(self, question: str, evidence: Sequence[Evidence]) -> Dict[str, Any]:
        if not evidence:
            return {
                "sufficient": False, "missing": ["No evidence retrieved"],
                "next_queries": [question], "contradictions": [], "confidence": 0.0,
            }
        return self.gateway.structured(
            system=COVERAGE_SYSTEM,
            prompt="Question:\n{}\n\nEvidence:\n{}".format(question, self._evidence_text(evidence)),
            schema_name="evidence_coverage",
            schema=COVERAGE_SCHEMA,
            model=self.models.reasoning_model,
        )

    def _compose(self, question: str, evidence: Sequence[Evidence], coverage: Mapping[str, Any]) -> Dict[str, Any]:
        if not evidence:
            return {
                "answer": "当前索引没有检索到足以回答该问题的证据。",
                "evidence_chunk_ids": [], "uncertainties": ["No evidence retrieved"],
                "spoiler": False, "confidence": 0.0,
            }
        result = self.gateway.structured(
            system=ANSWER_SYSTEM,
            prompt="Question:\n{}\n\nCoverage critic:\n{}\n\nEvidence:\n{}".format(
                question, json.dumps(coverage, ensure_ascii=False), self._evidence_text(evidence)
            ),
            schema_name="grounded_novel_answer",
            schema=ANSWER_SCHEMA,
            model=self.models.reasoning_model,
        )
        valid_chunk_ids = {chunk_id for item in evidence for chunk_id in item.source_chunk_ids}
        result["evidence_chunk_ids"] = [
            value for value in result.get("evidence_chunk_ids", []) if value in valid_chunk_ids
        ]
        answer_text = str(result.get("answer") or "")
        if "[order_id=" not in answer_text:
            citations = list(dict.fromkeys(item.citation() for item in evidence[:4]))
            answer_text = "{}\n\n依据：{}".format(answer_text, " ".join(citations)).strip()
        result["answer"] = answer_text
        return result

    @staticmethod
    def _evidence_text(evidence: Sequence[Evidence]) -> str:
        blocks = []
        for index, item in enumerate(evidence, 1):
            text = item.text[:1800]
            blocks.append(
                "E{} | {} | {} | chapter={} | source_chunk_ids={}\n{}".format(
                    index, item.kind, item.citation(), item.chapter_title,
                    ",".join(item.source_chunk_ids[:12]), text,
                )
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _evidence_payload(item: Evidence) -> Dict[str, Any]:
        return {
            "evidence_id": item.evidence_id, "kind": item.kind,
            "order_start": item.order_start, "order_end": item.order_end,
            "chapter_title": item.chapter_title,
            "source_chunk_ids": item.source_chunk_ids[:50],
            "source_chunk_count": len(item.source_chunk_ids),
            "preview": item.text[:600], "score": round(item.score, 6),
            "metadata": item.metadata,
        }
