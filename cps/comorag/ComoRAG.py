from gc import collect
import json
from multiprocessing import pool
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Union, Optional, List, Set, Dict, Any, Tuple, Literal
from anyio import Semaphore
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import igraph as ig
import re
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken

from .utils import agents
from .utils.timeline_utils import TimelineSummarizer
from .utils.summarization_utils import CommSummarizationModel

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank import DSPyFilter
from .utils.misc_utils import *
from .utils.embed_utils import retrieve_knn
from .utils.typing_utils import Triple
from .utils.config_utils import BaseConfig
from .utils.cluster_utils import ChunkSoftClustering
from .utils.memory_utils import MemoryNode, MemoryPool, NodeType

logger = logging.getLogger(__name__)

class ComoRAG:
    def __init__(self, global_config=None, 
                 save_dir=None, 
                 llm_model_name=None, 
                 llm_base_url=None,
                 llm_api_key=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 embedding_api_key=None,
                 book_id=None
                 ):
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config
        if save_dir is not None:
            self.global_config.save_dir = save_dir
        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name
        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name
        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url
        if llm_api_key is not None:
            self.global_config.llm_api_key = llm_api_key
        if embedding_api_key is not None:
            self.global_config.embedding_api_key = embedding_api_key
        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        if book_id is None:
            raise ValueError("ComoRAG requires a non-null book_id")
        self.book_id = int(book_id)

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"ComoRAG init with config:\n  {_print_config}\n")
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(
            self.global_config.save_dir,
            f"book_{self.book_id}",
            f"{llm_label}_{embedding_label}",
        )
        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        if self.global_config.openie_mode == 'online':
            self.openie = OpenIE(llm_model=self.llm_model)
        elif self.global_config.openie_mode == 'offline':
            self.openie = VLLMOfflineOpenIE(self.global_config)

        if self.global_config.openie_mode == 'offline':
            self.embedding_model = None
        else:
            self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
                embedding_model_name=self.global_config.embedding_model_name)(global_config=self.global_config,
                                                                              embedding_model_name=self.global_config.embedding_model_name)
        self.ver_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "chunk_embeddings"),
                                                    self.global_config.embedding_batch_size, 'chunk', self.book_id)
        self.entity_embedding_store = EmbeddingStore(self.embedding_model,
                                                     os.path.join(self.working_dir, "entity_embeddings"),
                                                     self.global_config.embedding_batch_size, 'entity', self.book_id)
        self.fact_embedding_store = EmbeddingStore(self.embedding_model,
                                                   os.path.join(self.working_dir, "fact_embeddings"),
                                                   self.global_config.embedding_batch_size, 'fact', self.book_id)
        self.ready_to_retrieve = False
        self.flag_cluster = False
        self.sem_embedding_store = None
        self.epi_embedding_store = None
        self.summarization_model = None
        self.timeline_summarizer = None
        self.clustering = None
        self.level_store = None
        self.node_to_node_stats = {}
        self.ent_node_to_num_chunk = {}
        self.graph = self.initialize_graph()
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.openie_results_path = os.path.join(
            self.global_config.save_dir,
            f"openie_results_book_{self.book_id}_ner_{self.global_config.llm_name.replace('/', '_')}.json",
        )
        self.rerank_filter = DSPyFilter(self)

        self.steps = self.global_config.max_iterations
        self.max_tokens_ver = self.global_config.max_tokens_ver
        self.max_tokens_sem = self.global_config.max_tokens_sem
        self.max_tokens_epi = self.global_config.max_tokens_epi

        if self.global_config.need_cluster:
            self._ensure_cluster_components()
            self.level_store = self.timeline_summarizer.get_level_embedding_store(0)

        self.tokenizer = self._build_tokenizer()

    def _build_tokenizer(self):
        model_name = (self.global_config.embedding_model_name or "").lower()
        if "text-embedding" in model_name:
            return tiktoken.get_encoding("cl100k_base")
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(self.global_config.embedding_model_name)
        except Exception:
            logger.warning(
                "Falling back to cl100k_base tokenizer because tokenizer loading failed."
            )
            return tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        if hasattr(self.tokenizer, "encode"):
            return len(self.tokenizer.encode(text))
        return len(text.split())

    @staticmethod
    def _format_context_blocks(items: List[str], block_label: str) -> str:
        """
        Format retrieved evidence as explicitly separated blocks.
        This prevents the model from treating multiple snippets as one
        continuous passage.
        """
        normalized_items = [str(item).strip() for item in items if str(item).strip()]
        if not normalized_items:
            return ""
        blocks = []
        for idx, text in enumerate(normalized_items, start=1):
            marker, body = ComoRAG._extract_reference_marker(text)
            if marker:
                blocks.append(f"[{block_label} #{idx} | {marker}]\n{body}")
            else:
                blocks.append(f"[{block_label} #{idx}]\n{text}")
        return "\n\n-----\n\n".join(blocks)

    @staticmethod
    def _extract_reference_marker(text: str) -> Tuple[str, str]:
        stripped = str(text).strip()
        if not stripped:
            return "", ""
        lines = stripped.splitlines()
        first_line = lines[0].strip()
        if first_line.startswith("[REF ") and first_line.endswith("]"):
            body = "\n".join(lines[1:]).strip()
            return first_line[5:-1].strip(), body
        return "", stripped

    @staticmethod
    def _format_reference_text(
        content: str,
        ref_hash_id: str,
        order_id: Optional[int] = None,
        source_order_ids: Optional[List[int]] = None,
    ) -> str:
        parts = [f"id={ref_hash_id}"]
        if order_id is not None:
            parts.append(f"order_id={order_id}")
        if source_order_ids:
            joined_orders = ",".join(str(v) for v in source_order_ids)
            parts.append(f"source_order_ids={joined_orders}")
        return f"[REF {' | '.join(parts)}]\n{content}"

    @staticmethod
    def _format_fusion_history(temp_nodes, main_nodes) -> str:
        entries = []
        seq = 1
        for node in list(temp_nodes) + list(main_nodes):
            entries.append(
                f"[FUSION #{seq}]\nProbe: {node.probe}\nFinding:\n{node.cue}"
            )
            seq += 1
        return "\n\n-----\n\n".join(entries)

    def _ensure_cluster_components(self):
        if self.sem_embedding_store is None:
            self.sem_embedding_store = EmbeddingStore(
                self.embedding_model,
                os.path.join(self.working_dir, "summary_embeddings"),
                self.global_config.embedding_batch_size,
                'summary',
                self.book_id,
            )
        if self.epi_embedding_store is None:
            self.epi_embedding_store = EmbeddingStore(
                self.embedding_model,
                os.path.join(self.working_dir, "timeline_embeddings"),
                self.global_config.embedding_batch_size,
                'timeline',
                self.book_id,
            )

        self.flag_cluster = bool(self.sem_embedding_store.get_all_ids())

        if self.summarization_model is None:
            self.summarization_model = CommSummarizationModel(
                self.global_config.llm_name,
                self.global_config.llm_base_url,
                self.global_config.llm_api_key,
            )
        if self.timeline_summarizer is None:
            self.timeline_summarizer = TimelineSummarizer(
                chunk_embedding_store=self.ver_embedding_store,
                summary_embedding_store=self.epi_embedding_store,
                summarization_model=self.summarization_model,
                book_id=self.book_id,
            )
        if self.clustering is None and not self.flag_cluster:
            self.clustering = ChunkSoftClustering(
                embedding_store=self.ver_embedding_store,
                reduction_dimension=10,
                threshold=0.01,
                verbose=True,
                db_filename=self.working_dir,
                namespace="rag_chunks",
                summarization_model=self.summarization_model,
                llm_model_name=self.global_config.llm_name,
                llm_base_url=self.global_config.llm_base_url,
                llm_api_key=self.global_config.llm_api_key,
            )
        if self.level_store is None:
            self.timeline_summarizer.load_all_summaries()
            self.level_store = self.timeline_summarizer.get_level_embedding_store(0)
        
    def initialize_graph(self):
        self._graphml_xml_file = os.path.join(
            self.working_dir, f"graph.graphml"
        )

        preloaded_graph = None


        if os.path.exists(self._graphml_xml_file):
            preloaded_graph = ig.Graph.Read_GraphML(self._graphml_xml_file)

        if preloaded_graph is None:
            graph = ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            graph = preloaded_graph

        self.graph = graph
        self._restore_persisted_runtime_state()
        return graph

    def _restore_persisted_runtime_state(self):
        """
        Restore persisted retrieval/runtime structures after a fresh init or
        after index mutations.

        This should be the single place that reconciles persisted stores/graph
        with in-memory runtime state required by retrieval.
        """
        self.node_to_node_stats = {}
        self.ent_node_to_num_chunk = {}
        self.ready_to_retrieve = False
        self.query_to_embedding = {'triple': {}, 'passage': {}}
        if self.sem_embedding_store is not None:
            self.flag_cluster = bool(self.sem_embedding_store.get_all_ids())
        self._restore_graph_runtime_state()

    def _restore_graph_runtime_state(self):
        """
        Recover runtime-only graph statistics from persisted graph/store data.

        `index()` builds `ent_node_to_num_chunk` in memory, but that structure is
        not stored separately. When a fresh ComoRAG instance is created after the
        index already exists, retrieval still needs this count map to normalize
        entity weights during graph search.
        """
        if self.graph is None or self.graph.vcount() == 0:
            return

        graph_names = set(self.graph.vs["name"]) if "name" in self.graph.vs.attribute_names() else set()
        if not graph_names:
            return

        passage_keys = set(self.ver_embedding_store.get_all_ids())
        entity_keys = set(self.entity_embedding_store.get_all_ids())
        if not passage_keys or not entity_keys:
            return

        valid_passage_keys = passage_keys.intersection(graph_names)
        valid_entity_keys = entity_keys.intersection(graph_names)
        if not valid_passage_keys or not valid_entity_keys:
            return

        vertex_index_by_name = {vertex["name"]: vertex.index for vertex in self.graph.vs}
        for entity_key in valid_entity_keys:
            entity_idx = vertex_index_by_name.get(entity_key)
            if entity_idx is None:
                continue
            neighbor_indices = self.graph.neighbors(entity_idx, mode="all")
            linked_passages = 0
            for neighbor_idx in neighbor_indices:
                neighbor_name = self.graph.vs[neighbor_idx]["name"]
                if neighbor_name in valid_passage_keys:
                    linked_passages += 1
            self.ent_node_to_num_chunk[entity_key] = linked_passages

        logger.info(
            "Restored graph runtime state for book_id=%s: %s entity chunk counts",
            self.book_id,
            len(self.ent_node_to_num_chunk),
        )

    def pre_openie(self,  docs: List[str]):
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE Offline")
        
        chunks = self.ver_embedding_store.get_missing_string_hash_ids(docs)

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        assert False, logger.info('Done with OpenIE, run online indexing for future retrieval.')

    def index(self, docs: List[str]):
        logger.info(f"Indexing Documents")

        logger.info(f"Performing OpenIE")
        if self.global_config.openie_mode == 'offline':
            self.pre_openie(docs)

        # 理论上已经构建过的不会重新构建
        self.ver_embedding_store.insert_strings(docs)

        if self.global_config.need_cluster:
            self._ensure_cluster_components()
            timeline_dir = os.path.join(self.working_dir, "timeline_embeddings")
            os.makedirs(timeline_dir, exist_ok=True)
            self.timeline_summarizer.try_load_or_generate_summaries(timeline_dir)
            self.timeline_summarizer.load_all_summaries()
            self.level_store = self.timeline_summarizer.get_level_embedding_store(0)

        if self.global_config.need_cluster and not self.flag_cluster:
            ver_hash_ids = self.ver_embedding_store.get_all_ids()
            ver_texts = [self.ver_embedding_store.get_row(hash_id)['content'] for hash_id in ver_hash_ids]
            ver_order_map = self.ver_embedding_store.get_hash_id_to_order()
            source_orders = [[ver_order_map.get(hash_id, -1)] for hash_id in ver_hash_ids]
            all_summaries, final_summary, summary_source_orders = self._recursive_clustering(
                ver_texts,
                max_iterations=5,  # Set maximum iteration count
                source_order_ids=source_orders,
            )
            self.sem_embedding_store.insert_strings(all_summaries, source_order_ids=summary_source_orders)
            final_summary_path = os.path.join(self.working_dir, "final_summary.txt")
            with open(final_summary_path, 'w', encoding='utf-8') as f:
                f.write(final_summary[0])

        chunks = self.ver_embedding_store.get_text_for_all_rows()
        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}
        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)
        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)
        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        # prepare data_store
        chunk_ids = list(chunks.keys())
        missing_keys = [
            chunk_id for chunk_id in chunk_ids
            if chunk_id not in ner_results_dict or chunk_id not in triple_results_dict
        ]
        if missing_keys:
            logger.warning(
                "OpenIE cache is incomplete for %s chunks, filling empty extraction results.",
                len(missing_keys),
            )
            for chunk_id in missing_keys:
                if chunk_id not in ner_results_dict:
                    ner_results_dict[chunk_id] = NerRawOutput(
                        chunk_id=chunk_id,
                        response="",
                        unique_entities=[],
                        metadata={},
                    )
                if chunk_id not in triple_results_dict:
                    triple_results_dict[chunk_id] = TripleRawOutput(
                        chunk_id=chunk_id,
                        response="",
                        triples=[],
                        metadata={},
                    )
        
        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)
        logger.info(f"Encoding Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"Encoding Facts")
        self.fact_embedding_store.insert_strings([str(fact) for fact in facts])

        logger.info(f"Constructing Graph")
        self.node_to_node_stats = {}
        self.ent_node_to_num_chunk = {}

        self.add_fact_edges(chunk_ids, chunk_triples)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges()
            self.augment_graph()
            self.save_igraph()

        self._restore_persisted_runtime_state()

    def meta_control_loop(self, q_idx, query):
        """process single query"""
        # extract query for retrieval (without options)
        if self.global_config.is_mc:
            retrieve_query = query
        else:
            retrieve_query = query
        pool_agent = agents.PoolAgent(
            model=self.global_config.llm_name,
            llm_base_url=self.global_config.llm_base_url,
            llm_api_key=self.global_config.llm_api_key
        )
        memory_pool = MemoryPool(agent=pool_agent, embedding_model=self.embedding_model)
        probe_agent = agents.ProbeAgent(
            model=self.global_config.llm_name,
            llm_base_url=self.global_config.llm_base_url,
            llm_api_key=self.global_config.llm_api_key
        )

        docs, nodes = self.tri_retrieve(retrieve_query, memory_pool)
        memory_pool = self.mem_encode(query=retrieve_query, docs=docs, memory_pool=memory_pool)
        
        ver_items = [ver for node in memory_pool.get_temp_nodes_by_type(NodeType.VER) for ver in node.original_content]
        sem_items = [sem for node in memory_pool.get_temp_nodes_by_type(NodeType.SEM) for sem in node.original_content]
        epi_items = [epi for node in memory_pool.get_temp_nodes_by_type(NodeType.EPI) for epi in node.original_content]
        ver_context = self._format_context_blocks(ver_items, "VER")
        sem_context = self._format_context_blocks(sem_items, "SEM")
        epi_context = self._format_context_blocks(epi_items, "EPI")
        
        historical_infomation = ""
        all_steps = [] 
        step_answers_local = {}
        
        for i in range(self.global_config.max_meta_loop_max_iterations+1):
            step_info = {
                "step": i + 1,
                "ver_context": ver_context,
                "sem_context": sem_context,
                "epi_context": epi_context,
                "historical_infomation": historical_infomation,
                "total_nodes": len(memory_pool.pool),
                "fusion_nodes": len(memory_pool.get_nodes_by_type(NodeType.FUSION))
            }
            prompt_user = ''
            if self.global_config.use_ver:
                prompt_user += f"### Detail Chunks (independent evidence blocks)\n{ver_context}\n\n"
            if self.global_config.use_sem:
                prompt_user += f"### Semantic Summary (independent evidence blocks)\n{sem_context}\n\n"
            if self.global_config.use_epi:
                prompt_user += f"### Timeline Summary (independent evidence blocks)\n{epi_context}\n\n"
            
            if i != 0:
                prompt_user += f"### Historical Information\n{historical_infomation}\n\n"

            prompt_user += 'Question: ' + query + '\nThought: ' 
            if self.global_config.is_mc:
                if i == 0:
                    qa_message = self.prompt_template_manager.render(name=f'rag_qa_mc', prompt_user=prompt_user)
                else:
                    qa_message = self.prompt_template_manager.render(name=f'rag_qa_mc_memory', prompt_user=prompt_user)
            else:
                qa_message = self.prompt_template_manager.render(name=f'rag_qa_narrativeqa_zh', prompt_user=prompt_user)
                
            result = self.llm_model.infer(qa_message)
            # try:
            if result is None:
                logger.error("LLM returned None response")
                step_info["error"] = "LLM returned None response"
                all_steps.append(step_info)
                continue
                
            response_content = result[0] if isinstance(result, (list, tuple)) else result
            if not response_content:
                logger.error("Empty response content from LLM")
                step_info["error"] = "Empty response content from LLM"
                all_steps.append(step_info)
                continue
                
            try:
                pred_ans = response_content.split('### <Final>')[1].strip()
            except IndexError:
                logger.error("Response does not contain '### Final Answer' section")
                pred_ans = response_content
                step_info["error"] = "Response does not contain '### Final Answer' section"

            step_info["response"] = response_content
            step_info["predicted_answer"] = pred_ans
            step_answers_local[f'step{i}'] = pred_ans
            if pred_ans.strip() == "*" or pred_ans.strip() == "<continue>" or '<continue>' in pred_ans.strip():
                memory_pool.merge_temp_to_main()
                # self-probe
                previous_probes = "\n".join(memory_pool.get_all_probes())
                probes = probe_agent.find_probes(query=retrieve_query, context=prompt_user, previous_probes=previous_probes)
                step_info["probes"] = probes
                for probe in probes:
                    docs,nodes = self.tri_retrieve(query = probe, memory_pool=memory_pool)
                    memory_pool = self.mem_encode(query= retrieve_query+" "+probe, docs=docs, memory_pool=memory_pool, probe=probe)
                # mem-fusion
                historical_infomation = memory_pool.create_fusion_content(probe=retrieve_query,top_k_percent=0.5)
                memory_pool.add_fused_node(probe=retrieve_query, fused_content=historical_infomation, source_nodes=nodes)
                
                sem_context = self._format_context_blocks(
                    [node.cue for node in memory_pool.get_temp_nodes_by_type(NodeType.SEM)],
                    "SEM-CUE",
                )
                epi_context = self._format_context_blocks(
                    [node.cue for node in memory_pool.get_temp_nodes_by_type(NodeType.EPI)],
                    "EPI-CUE",
                )
                ver_context = self._format_context_blocks(
                    [node.cue for node in memory_pool.get_temp_nodes_by_type(NodeType.VER)],
                    "VER-CUE",
                )
                historical_infomation = self._format_fusion_history(
                    memory_pool.get_temp_nodes_by_type(NodeType.FUSION),
                    memory_pool.get_nodes_by_type(NodeType.FUSION),
                )
                all_steps.append(step_info)
            else:
                all_steps.append(step_info)         
                break

            
                

        query_solution = QuerySolution(question=query, docs=ver_context, summary=sem_context, timeline=epi_context)
        query_solution.answer = response_content
        


        pool_info = {
            "total_nodes": len(memory_pool.pool),
            "total_chunks": len(memory_pool.get_nodes_by_type(NodeType.VER)),
            "total_summaries": len(memory_pool.get_nodes_by_type(NodeType.SEM)),
            "total_timelines": len(memory_pool.get_nodes_by_type(NodeType.EPI)),
            "total_probes": len(memory_pool.get_all_probes()),
            "probes": memory_pool.get_all_probes()
        }
        
        output_dir = os.path.join(self.global_config.output_dir, 'details')
        os.makedirs(output_dir, exist_ok=True)
        

        with open(os.path.join(output_dir, f"pool_info_{q_idx}.json"), 'w', encoding='utf-8') as f:
            json.dump(pool_info, f, ensure_ascii=False, indent=4)

        output_dir = os.path.join(self.global_config.output_dir, 'details')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"qa_output_{q_idx}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Problem-Solving Process Overview:\n")
            f.write("="*50 + "\n")
            f.write(f"Query: {query}\n\n")
            f.write("="*50 + "\n")
            for step in all_steps:
                f.write(f"Step {step['step']}:\n")
                f.write("-"*30 + "\n")
                f.write(f"Predicted Answer: {step.get('predicted_answer', 'N/A')}\n")
                f.write("-"*30 + "\n")
                f.write(f"ver_context:\n{step['ver_context']}\n")
                f.write("-"*30 + "\n")
                f.write(f"ver_context:\n{step['ver_context']}\n")
                f.write("-"*30 + "\n")
                f.write(f"epi_context:\n{step['epi_context']}\n")
                f.write("-"*30 + "\n")
                f.write(f"Historical Information:\n{step['historical_infomation']}\n")
                f.write("-"*30 + "\n")
                f.write(f"Response: {step.get('response', 'N/A')}\n")
                if 'probes' in step:
                    f.write("-"*30 + "\n")
                    f.write(f"probes: {', '.join(step['probes'])}\n")
                if 'error' in step:
                    f.write(f"Error: {step['error']}\n")
                f.write("="*50 + "\n\n")

        return q_idx, query_solution, step_answers_local
    def try_answer(self, queries: List[str], num_to_retrieve: int = None) -> List[QuerySolution]:
        queries_solutions = []
        step_answers = {}
        if self.global_config.need_cluster:
            self._ensure_cluster_components()
            self.level_store = self.timeline_summarizer.get_level_embedding_store(0)
        max_workers = min(16, len(queries)) 
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_query = {
                executor.submit(self.meta_control_loop, q_idx, query): q_idx 
                for q_idx, query in enumerate(queries)
            }

            
            queries_solutions = [None] * len(queries)
            step_answers = {}
            for future in tqdm(as_completed(future_to_query), total=len(queries), desc="Processing Queries"):
                q_idx, query_solution, step_answers_local = future.result()
                if query_solution:
                    queries_solutions[q_idx] = query_solution  
                    step_answers[q_idx] = step_answers_local

        queries_solutions = [qs for qs in queries_solutions if qs is not None]
        return queries_solutions
    
    #tri-retrieve
    def tri_retrieve(self, query: str, memory_pool: MemoryPool, ver_top_k: int = None, sem_top_k: int = None, epi_top_k: int = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ver_top_k = self.global_config.qa_ver_top_k if hasattr(self.global_config, 'qa_ver_top_k') else ver_top_k
        sem_top_k = self.global_config.qa_sem_top_k if hasattr(self.global_config, 'qa_sem_top_k') else sem_top_k
        epi_top_k = self.global_config.qa_epi_top_k if hasattr(self.global_config, 'qa_epi_top_k') else epi_top_k

        all_hashes = memory_pool.get_all_hashes()
        ver_hashes = all_hashes.get(NodeType.VER, [])
        sem_hashes = all_hashes.get(NodeType.SEM, [])
        epi_hashes = all_hashes.get(NodeType.EPI, [])

        
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(query)

        # Veridical Index Retrieval
        top_k_fact_indices, top_k_facts, rerank_log, fact_scores_by_idx = self.rerank_facts(query)
        nodes = {"idx" : 0,
                "question" : query,
                "nodes" : None,
                "rerank_log":rerank_log}

        if len(top_k_facts) == 0:
            logger.info('No facts found after reranking, return DPR results')
            query_embedding = self._get_passage_query_embedding(query)
            retrieval_k = max(self.global_config.retrieval_top_k, ver_top_k + len(ver_hashes) + 5)
            sorted_doc_ids, sorted_doc_scores = self.ver_embedding_store.rank_indices_by_vector(
                query_embedding=query_embedding,
                top_k=retrieval_k,
                exclude_hash_ids=set(ver_hashes),
            )

        else:
            self.global_config.passage_node_weight = 0.005
            sorted_doc_ids, sorted_doc_scores,node = self.graph_search_with_fact_entities(query=query,
                                                                                        link_top_k=self.global_config.linking_top_k,
                                                                                        fact_scores_by_idx=fact_scores_by_idx,
                                                                                        top_k_facts=top_k_facts,
                                                                                        top_k_fact_indices=top_k_fact_indices,
                                                                                        passage_node_weight=self.global_config.passage_node_weight)
            nodes["nodes"] = nodes
        top_k_docs: List[str] = []
        selected_ver_rows: List[Tuple[int, str]] = []
        for idx in sorted_doc_ids:
            if idx >= len(self.passage_node_keys):
                continue
            hash_id = self.passage_node_keys[idx]
            if hash_id in ver_hashes:
                continue
            row = self.ver_embedding_store.get_row(hash_id)
            order_id = self.ver_embedding_store.get_hash_id_to_order().get(hash_id)
            top_k_docs.append(
                self._format_reference_text(
                    content=row["content"],
                    ref_hash_id=hash_id,
                    order_id=order_id,
                )
            )
            selected_ver_rows.append((order_id if order_id is not None else 10**9, top_k_docs[-1]))
            if len(top_k_docs) >= ver_top_k:
                break

        top_k_docs = [doc for _, doc in sorted(selected_ver_rows, key=lambda pair: pair[0])]
        

        # Semantic Index Retrieval
        top_k_sem = []
        top_k_epi = []
        if self.global_config.need_cluster:
            query_embedding = self._get_passage_query_embedding(query)
            sem_retrieval_k = max(self.global_config.retrieval_top_k, sem_top_k + len(sem_hashes) + 5)
            sem_indices, _ = self.sem_embedding_store.rank_indices_by_vector(
                query_embedding=query_embedding,
                top_k=sem_retrieval_k,
                exclude_hash_ids=set(sem_hashes),
            )
            selected_sem_hashes: List[str] = []
            for idx in sem_indices:
                if idx >= len(self.summary_node_keys):
                    continue
                hash_id = self.summary_node_keys[idx]
                if hash_id in sem_hashes:
                    continue
                sem_row = self.sem_embedding_store.get_row(hash_id)
                source_orders = sem_row.get("source_order_ids") or []
                top_k_sem.append(
                    self._format_reference_text(
                        content=sem_row["content"],
                        ref_hash_id=hash_id,
                        order_id=self.sem_embedding_store.get_hash_id_to_order().get(hash_id),
                        source_order_ids=source_orders,
                    )
                )
                selected_sem_hashes.append(hash_id)
                if len(top_k_sem) >= sem_top_k:
                    break

            ### Episodic Index Retrieval
            epi_retrieval_k = max(self.global_config.retrieval_top_k, epi_top_k + len(epi_hashes) + 5)
            epi_indices, _ = self.level_store.rank_indices_by_vector(
                query_embedding=query_embedding,
                top_k=epi_retrieval_k,
                exclude_hash_ids=set(epi_hashes),
            )
            selected_epi_rows: List[Tuple[int, str]] = []
            for idx in epi_indices:
                if idx >= len(self.level_store.hash_ids):
                    continue
                hash_id = self.level_store.hash_ids[idx]
                if hash_id in epi_hashes:
                    continue
                epi_row = self.level_store.get_row(hash_id)
                order_id = self.level_store.get_hash_id_to_order().get(hash_id)
                source_orders = epi_row.get("source_order_ids") or []
                top_k_epi.append(
                    self._format_reference_text(
                        content=epi_row["content"],
                        ref_hash_id=hash_id,
                        order_id=order_id,
                        source_order_ids=source_orders,
                    )
                )
                selected_epi_rows.append((order_id if order_id is not None else 10**9, top_k_epi[-1]))
                if len(top_k_epi) >= epi_top_k:
                    break

            top_k_epi = [doc for _, doc in sorted(selected_epi_rows, key=lambda pair: pair[0])]

        docs = {
            "veridical":top_k_docs,
            "semantic":top_k_sem,
            "episodic":top_k_epi,
        }
        return docs, nodes

    #mem-encode
    def mem_encode(self, query: str, docs: Dict, memory_pool: MemoryPool, probe: str = None) -> MemoryNode:
        selected_vers = []
        current_tokens = 0
        for ver in docs["veridical"]:
            ver_tokens = self._count_tokens(ver)
            if current_tokens + ver_tokens > self.max_tokens_ver:
                break
            selected_vers.append(ver)
            current_tokens += ver_tokens
        
        selected_sems = []
        current_tokens = 0
        for sem in docs["semantic"]:
            sem_tokens = self._count_tokens(sem)
            if current_tokens + sem_tokens > self.max_tokens_sem:
                break
            selected_sems.append(sem)
            current_tokens += sem_tokens
        
        selected_epis = []
        current_tokens = 0
        for epi in docs["episodic"]:
            epi_tokens = self._count_tokens(epi)
            if current_tokens + epi_tokens > self.max_tokens_epi:
                break
            selected_epis.append(epi)
            current_tokens += epi_tokens

        
        pool_agent = memory_pool.agent
        ver_cue, sem_cue, epi_cue = pool_agent.fusion(
            query=query, 
            vers="\n".join(selected_vers), 
            sems="\n".join(selected_sems), 
            epis="\n".join(selected_epis), 
        )
        

        
        # Memory Nodes Generation
        ver_node = MemoryNode(
            probe=probe if probe else query,
            node_type=NodeType.VER,
            original_content=selected_vers,
            cue=ver_cue
        )
        ver_node.update_hashes()
        
        sem_node = MemoryNode(
            probe=probe if probe else query,
            node_type=NodeType.SEM,
            original_content=selected_sems,
            cue=sem_cue
        )
        sem_node.update_hashes()
        
        epi_node = MemoryNode(
            probe=probe if probe else query,
            node_type=NodeType.EPI,
            original_content=selected_epis,
            cue=epi_cue 
        )
        epi_node.update_hashes()
        
        memory_pool.add_to_temp_pool(ver_node)
        memory_pool.add_to_temp_pool(sem_node)
        memory_pool.add_to_temp_pool(epi_node)
        
        return memory_pool

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()
        logger.info(f"Adding OpenIE triples to graph.")
        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()
            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)
                    fact_key = compute_mdhash_id(content=str(triple), prefix=("fact-"))
                    node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))
                    self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                        (node_key, node_2_key), 0.0) + 1
                    self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get(
                        (node_2_key, node_key), 0.0) + 1
                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)
                for node in entities_in_chunk:
                    self.ent_node_to_num_chunk[node] = self.ent_node_to_num_chunk.get(node, 0) + 1

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()
        num_new_chunks = 0

        logger.info(f"Connecting passage nodes to phrase nodes.")   
        for idx, chunk_key in tqdm(enumerate(chunk_ids)):
            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self):

        logger.info(f"Expanding graph with synonymy edges")
        self.entity_id_to_row = self.entity_embedding_store.get_text_for_all_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())
        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")
        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)
        
        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)

        num_synonym_triple = 0
        synonym_candidates = []  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1

                        self.node_to_node_stats[sim_edge] = score  # Need to seriously discuss on this
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:

        chunk_keys = list(chunk_keys)
        chunk_key_set = set(chunk_keys)
        chunk_keys_to_save: Set[str] = set()

        if os.path.isfile(self.openie_results_path):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get('docs', [])

            # Normalize old cache records and deduplicate by chunk idx.
            dedup_openie: Dict[str, dict] = {}
            for openie_info in all_openie_info:
                idx = openie_info.get('idx')
                if not idx:
                    passage = openie_info.get('passage', '')
                    idx = compute_mdhash_id(passage, 'chunk-')
                # Keep only chunks in current indexing scope.
                if idx in chunk_key_set:
                    normalized = dict(openie_info)
                    normalized['idx'] = idx
                    dedup_openie[idx] = normalized

            all_openie_info = list(dedup_openie.values())

            existing_openie_keys = set([info['idx'] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput]) -> List[dict]:

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                                 'extracted_triples': triple_results_dict[chunk_key].triples}
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):

        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            avg_ent_chars = round(sum_phrase_chars / num_phrases, 4) if num_phrases > 0 else 0
            avg_ent_words = round(sum_phrase_words / num_phrases, 4) if num_phrases > 0 else 0
            openie_dict = {'docs': all_openie_info, 'avg_ent_chars': avg_ent_chars,
                           'avg_ent_words': avg_ent_words}
            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def augment_graph(self):

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):


        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_nodes = self.entity_embedding_store.get_text_for_all_rows()
        passage_nodes = self.ver_embedding_store.get_text_for_all_rows()
        if self.global_config.need_cluster:
            summary_nodes = self.sem_embedding_store.get_text_for_all_rows()

        nodes = entity_nodes
        nodes.update(passage_nodes)
        if self.global_config.need_cluster:
            nodes.update(summary_nodes)

        new_nodes = {}
        for node_id, node in nodes.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):


        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]: continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({
                "weight": weight
            })

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        self.graph.add_edges(
            valid_edges,
            attributes=valid_weights
        )

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph.write_graphml(self._graphml_xml_file)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:

        graph_info = {}

        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        passage_nodes_keys = self.ver_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        if self.global_config.need_cluster:
            summary_nodes_keys = self.sem_embedding_store.get_all_ids()
            graph_info["num_summary_nodes"] = len(set(summary_nodes_keys))

        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"] + graph_info.get("num_summary_nodes", 0)

        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        num_triples_with_passage_node = 0
        passage_nodes_set = set(passage_nodes_keys)
        num_triples_with_passage_node = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info['num_triples_with_passage_node'] = num_triples_with_passage_node

        graph_info['num_synonymy_triples'] = len(self.node_to_node_stats) - graph_info[
            "num_extracted_triples"] - num_triples_with_passage_node

        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    def prepare_retrieval_objects(self):


        logger.info("Preparing for fast retrieval.")

        logger.info("Loading keys.")
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids()) # a list of phrase node keys
        self.passage_node_keys: List = list(self.ver_embedding_store.get_all_ids()) # a list of passage node keys
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())
        if self.global_config.need_cluster:
            self.summary_node_keys: List = list(self.sem_embedding_store.get_all_ids())
        self.fact_id_to_idx = {node_id: idx for idx, node_id in enumerate(self.fact_node_keys)}
        self.passage_id_to_idx = {node_id: idx for idx, node_id in enumerate(self.passage_node_keys)}
        if self.global_config.need_cluster:
            self.summary_id_to_idx = {node_id: idx for idx, node_id in enumerate(self.summary_node_keys)}

        igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)} # from node key to the index in the backbone graph
        self.node_name_to_vertex_idx = igraph_name_to_idx
        self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys] # a list of backbone graph node index
        self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.passage_node_keys] # a list of backbone passage node index

        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.ver_embedding_store.get_embeddings(self.passage_node_keys))
        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))
        if self.global_config.need_cluster:
            self.summary_embeddings = np.array(self.sem_embedding_store.get_embeddings(self.summary_node_keys))


        logger.info(f"prepare_retrieval_objects: self.entity_embeddings.shape = {self.entity_embeddings.shape if hasattr(self.entity_embeddings, 'shape') else 'N/A'}, dtype = {self.entity_embeddings.dtype if hasattr(self.entity_embeddings, 'dtype') else 'N/A'}")
        logger.info(f"prepare_retrieval_objects: self.passage_embeddings.shape = {self.passage_embeddings.shape if hasattr(self.passage_embeddings, 'shape') else 'N/A'}, dtype = {self.passage_embeddings.dtype if hasattr(self.passage_embeddings, 'dtype') else 'N/A'}")
        logger.info(f"prepare_retrieval_objects: self.fact_embeddings.shape = {self.fact_embeddings.shape if hasattr(self.fact_embeddings, 'shape') else 'N/A'}, dtype = {self.fact_embeddings.dtype if hasattr(self.fact_embeddings, 'dtype') else 'N/A'}")

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: str | List[str] | List[QuerySolution]):
        if isinstance(queries, (str, QuerySolution)):
            queries = [queries]

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # get all query embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction('query_to_fact'),
                                                                            norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def get_top_fact_scores(self, query: str, top_k: Optional[int] = None):
 
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_fact'),
                                                                norm=True)
        if top_k is None:
            top_k = self.global_config.linking_top_k
        fact_ids, fact_scores = self.fact_embedding_store.search_topk(
            query_embedding=query_embedding,
            top_k=max(top_k * 3, top_k),
        )
        if fact_scores.size > 0:
            fact_scores = min_max_normalize(fact_scores)
        return fact_ids, fact_scores

    def dense_passage_retrieval(self, query: str, need_cluster: bool = False, top_k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
      
        if need_cluster:
            if top_k is None:
                top_k = len(self.summary_node_keys) if hasattr(self, "summary_node_keys") else self.global_config.retrieval_top_k
            sorted_doc_ids, sorted_doc_scores = self.sem_embedding_store.rank_indices_by_vector(
                query_embedding=query_embedding,
                top_k=top_k,
            )
        else:
            if top_k is None:
                top_k = self.global_config.retrieval_top_k
            sorted_doc_ids, sorted_doc_scores = self.ver_embedding_store.rank_indices_by_vector(
                query_embedding=query_embedding,
                top_k=top_k,
            )
        sorted_doc_scores = np.asarray(sorted_doc_scores)
        if sorted_doc_scores.size > 0:
            sorted_doc_scores = min_max_normalize(sorted_doc_scores)
        return sorted_doc_ids, sorted_doc_scores

    def _get_passage_query_embedding(self, query: str) -> np.ndarray:
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(
                query,
                instruction=get_query_instruction('query_to_passage'),
                norm=True,
            )
            self.query_to_embedding['passage'][query] = query_embedding
        return query_embedding




    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:

        # Keep top-k candidates first.
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])
        # Drop zero/negative-score entities early; they cannot contribute to reset weights.
        linking_score_map = {
            phrase: score for phrase, score in linking_score_map.items()
            if score is not None and float(score) > 0
        }

        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [compute_mdhash_id(content=top_k_phrase, prefix="entity-") for top_k_phrase in top_k_phrases])
        
        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        # Reconcile mismatch by pruning entities that are not usable at runtime.
        pruned_linking_score_map: Dict[str, float] = {}
        problematic_phrases = []
        for phrase, score in linking_score_map.items():
            phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
            phrase_id = self.node_name_to_vertex_idx.get(phrase_key)
            exists_in_graph = phrase_id is not None
            current_weight = float(all_phrase_weights[phrase_id]) if exists_in_graph else 0.0
            if exists_in_graph and current_weight > 0:
                pruned_linking_score_map[phrase] = score
            else:
                problematic_phrases.append(
                    {
                        "phrase": phrase,
                        "phrase_key": phrase_key,
                        "exists_in_graph": exists_in_graph,
                        "weight": round(current_weight, 6),
                        "score": round(float(score), 6),
                    }
                )

        if problematic_phrases:
            logger.warning(
                "Pruned unusable linking entities: count=%s sample=%s",
                len(problematic_phrases),
                problematic_phrases[:10],
            )
        linking_score_map = pruned_linking_score_map
        return all_phrase_weights, linking_score_map

    def  graph_search_with_fact_entities(self, query: str,
                                        link_top_k: int,
                                        fact_scores_by_idx: Dict[int, float],
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[int],
                                        passage_node_weight: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    
        linking_score_map = {}  # from phrase to the average scores of the facts that contain the phrase 
        phrase_scores = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not 
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))
        used_phrases_with_scores = {}

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = fact_scores_by_idx.get(top_k_fact_indices[rank], 0.0)
            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(
                    content=phrase,
                    prefix="entity-"
                )
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    # Algorithm rule:
                    # For entities appearing in multiple facts, aggregate with MAX contribution
                    # instead of overwrite. This avoids a later low/zero-score fact wiping out
                    # a previously strong entity signal.
                    candidate_weight = fact_score
                    if self.ent_node_to_num_chunk[phrase_key] != 0:
                        candidate_weight /= self.ent_node_to_num_chunk[phrase_key]
                    phrase_weights[phrase_id] = max(phrase_weights[phrase_id], candidate_weight)
                    if phrase_weights[phrase_id] > 0:
                        used_phrases_with_scores[phrase] = phrase_weights[phrase_id]
                if phrase not in phrase_scores:
                    phrase_scores[phrase] = []
                phrase_scores[phrase].append(fact_score)

        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))
        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                           phrase_weights,
                                                                           linking_score_map)
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(
            query,
            top_k=self.global_config.retrieval_top_k,
        )
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)
        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.ver_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        node_weights = phrase_weights + passage_weights
        

        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights)
        assert len(ppr_sorted_doc_ids) == len(
            self.passage_node_idxs), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores,used_phrases_with_scores

    def rerank_facts(self, query: str) -> Tuple[List[int], List[Tuple], dict, Dict[int, float]]:
        """

        Args:

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:


        """
        link_top_k: int = self.global_config.linking_top_k

        candidate_fact_ids, candidate_scores = self.get_top_fact_scores(query, top_k=link_top_k)
        if not candidate_fact_ids:
            return [], [], {"facts_before_rerank": [], "facts_after_rerank": []}, {}
        candidate_fact_indices = [
            self.fact_id_to_idx[fid] for fid in candidate_fact_ids if fid in self.fact_id_to_idx
        ]

        # 这里很迷惑，应该直接就是candidate_fact_ids
        real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
        fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
        candidate_facts = [eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids]
        fact_scores_by_idx = {
            idx: float(candidate_scores[pos])
            for pos, idx in enumerate(candidate_fact_indices)
            if pos < len(candidate_scores)
        }

        top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(query,
                                                                             candidate_facts,
                                                                             candidate_fact_indices,
                                                                             len_after_rerank=link_top_k)

        rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
        return top_k_fact_indices, top_k_facts, rerank_log, fact_scores_by_idx
    
    def run_ppr(self,
                reset_prob: np.ndarray,
                damping: float =0.5) -> Tuple[np.ndarray, np.ndarray]:

        if damping is None: damping = 0.5
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores
    
    def _recursive_clustering(
        self,
        texts,
        max_iterations=5,
        current_iteration=0,
        source_order_ids: Optional[List[List[int]]] = None,
    ):
        # Create temporary folder paths
        temp_embeddings_dir = os.path.join(self.working_dir, "temp_embeddings")
        temp_clusters_dir = os.path.join(self.working_dir, "temp_clusters")
        if source_order_ids is None:
            source_order_ids = [[idx] for idx in range(len(texts))]
        if len(source_order_ids) != len(texts):
            raise ValueError(
                f"_recursive_clustering input mismatch: texts={len(texts)} "
                f"source_order_ids={len(source_order_ids)}"
            )
        
        # Define cleanup function
        def cleanup_temp_folders():
            try:
                import shutil
                if os.path.exists(temp_embeddings_dir):
                    shutil.rmtree(temp_embeddings_dir)
                    print(f"Deleted temporary embeddings folder: {temp_embeddings_dir}")
                if os.path.exists(temp_clusters_dir):
                    shutil.rmtree(temp_clusters_dir)
                    print(f"Deleted temporary clusters folder: {temp_clusters_dir}")
            except Exception as e:
                print(f"Error cleaning up temporary folders: {e}")
        
        # Early return cases
        if len(texts) <= 1:
            cleanup_temp_folders()
            return texts, texts, source_order_ids
            
        if current_iteration >= max_iterations:
            cleanup_temp_folders()
            # `texts` is returned as the accumulated summaries for this level,
            # so source_order_ids must stay aligned 1:1 with `texts`.
            return texts, [texts[0]], source_order_ids
        
        try:
            temp_embedding_store = EmbeddingStore(
                self.embedding_model,
                temp_embeddings_dir,
                self.global_config.embedding_batch_size,
                'temp',
                self.book_id,
            )
            
            temp_embedding_store.insert_strings(texts)
            text_hash_ids = [compute_mdhash_id(text, prefix=temp_embedding_store.namespace + "-") for text in texts]
            hash_to_source_orders = {}
            for idx, hash_id in enumerate(text_hash_ids):
                merged_source_orders = set(hash_to_source_orders.get(hash_id, []))
                merged_source_orders.update(source_order_ids[idx] or [])
                hash_to_source_orders[hash_id] = sorted(merged_source_orders)
            
            clustering = ChunkSoftClustering(
                embedding_store=temp_embedding_store,
                reduction_dimension=10,
                threshold=0.01,
                verbose=True,
                db_filename=temp_clusters_dir,
                namespace="clusters",
                summarization_model=self.summarization_model,
                llm_model_name=self.global_config.llm_name,
                llm_base_url=self.global_config.llm_base_url,
                llm_api_key=self.global_config.llm_api_key
            )
            
            clusters = clustering.perform_clustering()
            
            stats = clustering.get_cluster_stats()
            print(f"Clustering stats: {stats}")
            
            cluster_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(clusters))) as executor:
                future_to_cluster = {
                    executor.submit(clustering.create_cluster_summary, cluster.id): cluster 
                    for cluster in clusters
                }
                
                for future in concurrent.futures.as_completed(future_to_cluster):
                    try:
                        summary = future.result()
                        if summary:  
                            cluster = future_to_cluster[future]
                            cluster_source_orders = set()
                            for member_hash_id in cluster.members.keys():
                                cluster_source_orders.update(hash_to_source_orders.get(member_hash_id, []))
                            cluster_results.append(
                                (cluster.id, summary, sorted(cluster_source_orders))
                            )
                    except Exception as e:
                        logger.error(f"error: {str(e)}")

            cluster_results.sort(key=lambda item: item[0])
            summary_texts = [item[1] for item in cluster_results]
            summary_source_orders = [item[2] for item in cluster_results]
            if len(summary_texts) != len(summary_source_orders):
                raise ValueError(
                    f"_recursive_clustering output mismatch: summaries={len(summary_texts)} "
                    f"source_order_ids={len(summary_source_orders)}"
                )
            
            # Clean up temporary folders for current level
            cleanup_temp_folders()
            
            # Recursively process next level
            if len(summary_texts) == 1:
                return summary_texts, summary_texts, summary_source_orders
            
            next_level_summaries, final_summary, next_level_source_orders = self._recursive_clustering(
                summary_texts, 
                max_iterations=max_iterations,
                current_iteration=current_iteration + 1,
                source_order_ids=summary_source_orders,
            )
            return (
                summary_texts + next_level_summaries,
                final_summary,
                summary_source_orders + next_level_source_orders,
            )
            
        except Exception as e:
            # Ensure temporary folders are cleaned up even in case of exceptions
            print(f"Error during recursive clustering: {e}")
            cleanup_temp_folders()
            raise
