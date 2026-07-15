from __future__ import annotations

import json
import os
import tempfile
import unittest

from cps.ai.config import ModelSettings, NovelSettings
from cps.ai.novel.indexing import NarrativeIndexer
from cps.ai.novel.ingestion import segment_chapters
from cps.ai.novel.models import Chapter
from cps.ai.novel.reasoning import NovelReasoner
from cps.ai.novel.repository import NovelRepository
from cps.ai.novel.vector_store import NarrativeVectorStore


class FakeGateway:
    def structured(self, *, system, prompt, schema_name, schema, model=None):
        if schema_name == "narrative_extraction":
            passages = json.loads(prompt[prompt.index("["):])
            chunk_ids = [item["chunk_id"] for item in passages]
            events = [
                {
                    "key": "discovery", "title": "Alice finds the key",
                    "summary": "Alice discovers a brass key behind the portrait.",
                    "event_type": "discovery", "story_time": "that night",
                    "location": "study", "epistemic_status": "narrator_fact",
                    "participant_keys": ["alice"],
                    "evidence_chunk_ids": chunk_ids[:1], "confidence": 0.95,
                }
            ]
            relations = []
            if len(chunk_ids) > 1:
                events.append(
                    {
                        "key": "silence", "title": "Alice conceals the discovery",
                        "summary": "Alice tells nobody that she found the key.",
                        "event_type": "decision", "story_time": "at dawn",
                        "location": "house", "epistemic_status": "narrator_fact",
                        "participant_keys": ["alice"],
                        "evidence_chunk_ids": chunk_ids[1:2], "confidence": 0.9,
                    }
                )
                relations.append(
                    {
                        "source_event_key": "discovery", "target_event_key": "silence",
                        "relation_type": "causes", "evidence_chunk_id": chunk_ids[1],
                        "confidence": 0.8,
                    }
                )
            return {
                "entities": [
                    {
                        "key": "alice", "name": "Alice", "entity_type": "person",
                        "aliases": ["A"], "description": "The investigator",
                        "mention_chunk_ids": chunk_ids,
                    }
                ],
                "events": events,
                "relations": relations,
            }
        if schema_name == "narrative_memory":
            return {
                "title": "Key discovery",
                "summary": "Alice finds a key and keeps the discovery secret.",
                "character_states": [
                    {
                        "entity": "Alice", "before_state": "does not have the key",
                        "after_state": "has the brass key", "change": "acquires the key",
                        "epistemic_status": "narrator_fact", "evidence_chunk_ids": [],
                    }
                ],
                "unresolved_threads": ["What does the key open?"],
                "event_links": [],
            }
        if schema_name == "novel_retrieval_plan":
            return {
                "answer_type": "causal", "entities": ["Alice"],
                "keywords": ["key", "portrait"], "temporal_scope": "chapter",
                "relation_types": ["causes", "reveals"],
                "channels": ["episodic", "semantic", "veridical"],
                "probes": ["Where did Alice find the key?"],
            }
        if schema_name == "evidence_coverage":
            return {
                "sufficient": True, "missing": [], "next_queries": [],
                "contradictions": [], "confidence": 0.9,
            }
        if schema_name == "grounded_novel_answer":
            return {
                "answer": "Alice found the brass key behind the portrait. [order_id=0]",
                "evidence_chunk_ids": [], "uncertainties": [],
                "spoiler": True, "confidence": 0.93,
            }
        raise AssertionError("Unexpected schema {}".format(schema_name))

    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([
                float(len(text) % 101) / 101.0,
                1.0 if "key" in lowered or "钥匙" in text else 0.0,
                1.0 if "alice" in lowered else 0.0,
                0.5,
            ])
        return vectors


class SegmentationTests(unittest.TestCase):
    def test_segmentation_never_crosses_chapters_and_is_stable(self):
        chapters = [
            Chapter("c1", 0, "One", ["A" * 45, "B" * 45, "C" * 45]),
            Chapter("c2", 1, "Two", ["D" * 45, "E" * 45]),
        ]
        first = segment_chapters(7, chapters, target_chars=100, overlap_paragraphs=1)
        second = segment_chapters(7, chapters, target_chars=100, overlap_paragraphs=1)
        self.assertEqual([item.chunk_id for item in first], [item.chunk_id for item in second])
        self.assertEqual(list(range(len(first))), [item.order_id for item in first])
        self.assertTrue(all(item.chapter_title in {"One", "Two"} for item in first))
        self.assertFalse(any("A" in item.text and "D" in item.text for item in first))


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp.name, "ai.db")
        self.vector_path = os.path.join(self.temp.name, "lancedb")
        self.book_path = os.path.join(self.temp.name, "novel.txt")
        with open(self.book_path, "w", encoding="utf-8") as stream:
            stream.write(
                "Alice entered the locked study at midnight.\n\n"
                "Behind the portrait, Alice found a brass key and hid it in her coat.\n\n"
                "At dawn she told nobody about the discovery.\n"
            )
        self.repository = NovelRepository(self.db_path)
        self.vectors = NarrativeVectorStore(self.vector_path)
        self.gateway = FakeGateway()
        self.models = ModelSettings(
            api_key="test", base_url=None, chat_model="test-chat",
            extraction_model="test-extract", reasoning_model="test-reason",
            embedding_model="test-embedding",
        )
        self.settings = NovelSettings(
            chunk_chars=100, chunk_overlap_paragraphs=1, extraction_batch_size=2,
            embedding_batch_size=2, arc_chapters=2, retrieval_limit=8,
            max_reasoning_rounds=2, graph_hops=2, keep_index_runs=2,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_index_and_reason_across_all_storage_layers(self):
        run_id = "run00000000000000000001"
        self.repository.create_run(
            run_id=run_id, book_id=42, status="QUEUED", phase="QUEUED",
            message="queued", llm_model=self.models.extraction_model,
            embedding_model=self.models.embedding_model, config={},
        )
        indexer = NarrativeIndexer(
            repository=self.repository, vector_store=self.vectors,
            gateway=self.gateway, model_settings=self.models,
            novel_settings=self.settings,
        )
        status = indexer.build(
            book_id=42, run_id=run_id, source_path=self.book_path, force=True
        )
        self.assertTrue(status["indexed"])
        self.assertGreaterEqual(status["counts"]["chunks"], 2)
        self.assertGreaterEqual(status["counts"]["entities"], 1)
        self.assertGreaterEqual(status["counts"]["events"], 1)
        self.assertGreaterEqual(status["counts"]["memories"], 2)

        graph_rows = self.repository.graph_search(run_id, ["Alice"], hops=1, limit=5)
        self.assertTrue(graph_rows)
        self.assertIn("brass key", graph_rows[0]["summary"])
        self.assertTrue(any(row.get("_graph_links") for row in graph_rows))

        reasoner = NovelReasoner(
            repository=self.repository, vector_store=self.vectors,
            gateway=self.gateway, model_settings=self.models,
            novel_settings=self.settings,
        )
        result = reasoner.answer(42, "Where did Alice find the key?")
        self.assertEqual("success", result["status"])
        self.assertIn("[order_id=0]", result["answer"])
        self.assertTrue(result["evidence"])
        self.assertTrue(result["trace"]["trace_id"].startswith("trace-"))

    def test_failed_run_does_not_replace_active_run(self):
        self.repository.create_run(
            run_id="active", book_id=9, status="RUNNING", phase="TEST",
            message="test", llm_model="m", embedding_model="e", config={},
        )
        self.repository.promote_run("active")
        self.repository.create_run(
            run_id="failed", book_id=9, status="FAILED", phase="FAILED",
            message="failed", llm_model="m", embedding_model="e", config={},
        )
        self.assertEqual("active", self.repository.active_run(9)["run_id"])


if __name__ == "__main__":
    unittest.main()
