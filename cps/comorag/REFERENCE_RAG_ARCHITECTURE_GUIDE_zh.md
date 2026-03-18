# 参考实现蓝图：三层知识源 + 迭代式检索推理（非源码搬运版）

> 目的：给另一个项目提供可落地的工程方案，复用思路而不是复用具体代码。  
> 适用：长文档问答、多文档推理、需要“检索-推理-再检索”的场景。

---

## 1. 核心目标与设计原则

### 目标
- 建立一个支持长上下文推理的 RAG 系统，不依赖一次性检索。
- 将知识组织成三类来源（细节、语义、情节）并在回答时联合使用。
- 在回答失败或不确定时，触发“自探测”并迭代补充证据。

### 原则
- **分层知识**：不同粒度信息分开构建、分开检索。
- **状态化推理**：多轮过程中保留中间记忆，不每轮从零开始。
- **可替换模块**：LLM、Embedding、重排器、聚类器都通过接口解耦。
- **增量构建**：索引、摘要、图谱都支持缓存和续跑。

---

## 2. 三层知识源（每层的具体做法）

> 这部分按“输入 -> 构建 -> 存储 -> 检索 -> 推理使用 -> 常见坑”写，方便直接实现。

### A. Veridical Layer（细节层）

**1) 输入是什么**
- 文档切分后的 chunk（建议每块 200~800 tokens，保留 `doc_id/chunk_id/order`）。

**2) 怎么构建**
- 对每个 chunk 生成 embedding。
- （可选增强）对 chunk 做 NER + 三元组抽取，得到实体与事实，为图检索准备数据。

**3) 存什么**
- `chunk_id -> {content, embedding, metadata}`。
- 可选图数据：`entity_nodes`、`fact_nodes`、`chunk-entity edges`、`entity-entity edges`。

**4) 怎么检索**
- 基础版：`query_embedding` 与 chunk embedding 做相似度检索（top-k）。
- 增强版：先用事实/实体召回，再结合图扩散分数与向量分数融合排序。

**5) 在推理里怎么用**
- 作为证据原文输入（Detail Chunks），用于“引用事实、精确定位、答案可解释性”。
- 通常给最大 token 预算（因为它是最可信来源）。

**6) 常见坑**
- chunk 太大导致噪声高；太小导致语义断裂。
- 去重没做会造成同一事实反复出现，影响推理稳定性。

### B. Semantic Layer（语义层）

**1) 输入是什么**
- Veridical 层的 chunk 文本与 embedding。

**2) 怎么构建**
- 先对 chunk embedding 聚类（GMM/KMeans/HDBSCAN 均可）。
- 每个簇取成员文本生成簇摘要（cluster summary）。
- 可选递归：对摘要再次聚类再摘要，形成多层抽象。

**3) 存什么**
- `summary_id -> {summary_text, embedding, member_chunk_ids, level}`。

**4) 怎么检索**
- 对 query 做 embedding，与 summary embedding 做相似度检索（top-k）。
- 若有多层摘要，可先检索高层再下钻到低层或 member chunks。

**5) 在推理里怎么用**
- 作为语义压缩上下文（Semantic Summary），用于快速给模型“主题地图”。
- 常与 Veridical 联动：先用语义层定位，再去细节层取证据。

**6) 常见坑**
- 聚类阈值不合理会出现“簇过大/过碎”。
- 摘要提示词过于抽象会丢实体、时间和因果信息。

### C. Episodic Layer（情节层）

**1) 输入是什么**
- 按 `order` 排序后的 chunk 序列（可按文档或章节分别处理）。

**2) 怎么构建**
- 用滑动窗口（如 3~10 个 chunk）做局部时间段摘要。
- 每个窗口产出一条 timeline summary。
- 可选层级化：把 level_0 的 summary 继续汇总成 level_1。

**3) 存什么**
- `episode_id -> {timeline_summary, embedding, covered_chunk_range, level}`。

**4) 怎么检索**
- query embedding 与 episodic summary embedding 相似度检索（top-k）。
- 对候选按原始顺序重排，保证上下文是时间连贯的。

**5) 在推理里怎么用**
- 提供时间线和事件演化（Timeline Summary）。
- 对“先后顺序、因果关系、状态变化”类问题效果最明显。

**6) 常见坑**
- 窗口过大导致时间细节被抹平；过小导致叙事断裂。
- 检索后不按时间重排，会让模型读到“打乱时间线”的上下文。

### D. 三层联合时的推荐策略

- **召回比例**：先给 Veridical 最高预算，再分配 Semantic/Episodic。
- **融合顺序**：`语义定位 -> 细节取证 -> 情节校正`。
- **冲突处理**：三层冲突时优先 Veridical 原文证据，Semantic/Episodic 仅作辅助解释。
- **迭代补证**：若回答不确定，优先生成 probe 去补 Veridical，必要时再补 Episodic。

---

## 3. 端到端流程（建议实现顺序）

1. **文档预处理**
   - 清洗文本、切分 chunk、保留 `doc_id/chunk_id/order`。
2. **建立 Veridical 索引**
   - 为 chunk 建 embedding 向量库。
3. **信息抽取 + 关系图（可选增强）**
   - 对 chunk 做实体/关系抽取，构建实体-事实-段落图。
4. **建立 Semantic 索引**
   - 聚类 chunk，生成 cluster summary，再向量化存储。
5. **建立 Episodic 索引**
   - 按窗口生成 timeline summary，再向量化存储。
6. **三路联合检索**
   - 一次查询并行召回 ver/sem/epi。
7. **记忆编码与多轮推理**
   - 对每层结果压缩成“线索”，进入工作记忆池。
   - LLM 回答若不确定，生成 probe 子问题继续检索。
8. **终止与输出**
   - 达到可回答条件或迭代上限，输出答案和证据轨迹。

---

## 4. 建议模块划分（供其他 AI 按此实现）

### 4.1 Config 层
- `AppConfig`
  - 模型配置（LLM/Embedding）
  - 检索配置（top_k、阈值）
  - 迭代配置（max_steps、token budget）
  - 存储目录与缓存开关

### 4.2 数据与存储层
- `ChunkStore`：原始 chunk 与元数据
- `VectorStore`：统一向量存储抽象（支持 namespace）
- `GraphStore`（可选）：实体与关系图
- `RunCache`：LLM 响应缓存、OpenIE 缓存、摘要缓存

### 4.3 模型适配层
- `LLMClient`（统一 chat 接口）
- `EmbeddingClient`（统一 batch_encode）
- `ExtractorClient`（NER/RE，可LLM或规则）
- `SummarizerClient`

### 4.4 构建层（Indexing）
- `VeridicalIndexer`
- `SemanticIndexer`（聚类 + 摘要）
- `EpisodicIndexer`（窗口摘要 + 层级组织）
- `GraphBuilder`（可选）

### 4.5 检索与推理层
- `TriRetriever`：返回 `{veridical, semantic, episodic}`
- `MemoryPool`：管理临时记忆、历史记忆、融合记忆
- `ReasoningController`
  - 一轮回答
  - 自探测生成（probe）
  - 迭代终止判定

### 4.6 评测与可观测性
- 指标：EM/F1/Recall@k/步骤数/Token 成本
- 输出：`results.json` + 每题过程日志 + 中间检索命中

---

## 5. 推荐数据契约（可直接复用）

### 5.1 输入
- `corpus.jsonl`
  - `id`, `doc_id`, `title`, `contents`
- `qas.jsonl`
  - `id`, `question`, `golden_answers`

### 5.2 中间产物
- `chunk_embeddings.parquet`
- `summary_embeddings.parquet`
- `timeline_level_0.parquet`（可扩展到 level_n）
- `openie_results.json`
- `graph.graphml`（或自定义图格式）

### 5.3 输出
- `results.json`
  - `idx`, `question`, `output`, `golden_answers`
- `details/*.json|txt`
  - 每步上下文、probe、候选证据、最终答案

---

## 6. 推理控制策略（关键）

### 6.1 单轮结构
- 输入：`query + ver_context + sem_context + epi_context + history`
- 输出：`answer` 或 `*`（表示证据不足需继续）

### 6.2 触发继续检索
- 当模型返回“未知/不确定/证据不足”标记时：
  - 生成 1~3 个 probe 子问题
  - 以 probe 再次三路检索
  - 新证据写入临时记忆并融合

### 6.3 终止条件
- 出现明确答案
- 连续多轮无新增有效证据
- 达到 `max_meta_loop_iterations`

### 6.4 Agent 推理 Loop（可直接实现）

**目标**：让 Agent 在“回答 -> 发现证据不足 -> 主动补证 -> 再回答”的闭环中收敛，而不是一次检索后硬答。  

**循环输入状态**
- `query`：当前问题
- `memory_pool`：历史记忆（主池）+ 本轮记忆（临时池）
- `step`：当前迭代轮次
- `docs_last_round`：上一轮检索结果（用于判断是否有新增证据）

**每轮 8 个步骤**
1. **Tri-Retrieve**
   - 用 `query`（或 probe）并行检索三层：`ver/sem/epi`。
2. **Budgeting**
   - 按 token 配额裁剪三层上下文（例如 ver:sem:epi = 6:2:2）。
3. **Memory Encoding**
   - 将三层结果写入临时记忆节点（VER/SEM/EPI），并生成 cue。
4. **Compose Prompt**
   - 拼接 `detail + semantic + episodic + historical_memory`。
5. **Answer Attempt**
   - 调用 LLM 生成答案草稿与置信信号（或约定标记 `*`）。
6. **Decision Gate**
   - 若“可终止”则输出最终答案。
   - 若“证据不足”则进入 probe 阶段。
7. **Probe Expansion**
   - 生成 1~3 个子问题（probe），逐个补检索并写入记忆。
8. **Memory Fusion**
   - 将临时记忆合并到主记忆，形成下一轮历史上下文。

**推荐状态机**
- `INIT -> RETRIEVE -> ENCODE -> ANSWER -> (FINAL | PROBE) -> FUSE -> NEXT_STEP`
- `NEXT_STEP` 若超过上限，转 `FINAL_FALLBACK`（输出“最佳当前答案+不确定性说明”）。

**终止判定（建议并行使用）**
- 规则 1：模型输出了结构化最终答案且非空。
- 规则 2：本轮新增证据比例低于阈值（如 <10%）且连续 2 轮。
- 规则 3：probe 无法产生新证据（top-k 全部命中历史 hash）。
- 规则 4：达到最大轮数。

**失败分支处理**
- LLM 调用异常：本轮跳过，保留上轮最佳答案。
- probe 解析失败：退化为固定模板 probe（who/when/where/why）。
- 检索为空：放宽 top_k 或降低相似度阈值重试一次。

**可观测性（建议最少记录）**
- 每轮记录：`step_id, query, probes, ver_ids, sem_ids, epi_ids, token_usage, decision, answer_draft`
- 便于离线分析“为何循环过多/为何提前终止”。

**Loop 伪代码**
```python
def solve(query):
    mem = MemoryPool()
    best = None
    no_gain_rounds = 0

    for step in range(max_steps):
        docs = tri_retrieve(query, mem)
        cues, gain = memory_encode(docs, mem, budget=token_budget)
        draft = answer_once(query, cues, mem.history())
        best = select_better(best, draft)

        if is_final(draft):
            return finalize(draft, mem)

        if gain < gain_threshold:
            no_gain_rounds += 1
        else:
            no_gain_rounds = 0

        if no_gain_rounds >= 2:
            return finalize(best, mem, uncertain=True)

        probes = generate_probes(query, draft, mem)
        for p in probes:
            p_docs = tri_retrieve(p, mem)
            memory_encode(p_docs, mem, probe=p)

        mem.fuse_temp_into_main()

    return finalize(best, mem, uncertain=True)
```

### 6.5 Memory Fusion 具体做法（实现指南）

**作用**
- 把“本轮新增证据”压缩成可复用的历史记忆，降低下一轮上下文长度。
- 对同类证据做去重与冲突消解，避免循环中信息爆炸。

**输入**
- `temp_nodes`: 本轮临时节点（VER/SEM/EPI）
- `main_nodes`: 历史主池节点
- `query` / `active_probe`

**输出**
- `fused_node`（类型可设为 `FUSION`）
- 更新后的主记忆池（含溯源关系）

**推荐 4 步**
1. **候选筛选**
   - 计算临时节点与 query/probe 的相关性（embedding 相似度或交叉编码打分）。
   - 保留 top-p（如 40%~60%）作为融合输入。
2. **去重与冲突处理**
   - 文本 hash 去重（同内容只保留一份）。
   - 语义近重复去重（余弦相似度 > 0.92 判定重复）。
   - 冲突时保留 Veridical 原文优先，摘要类节点仅保留“解释性语句”。
3. **融合生成**
   - 用 LLM 生成结构化 fused content，建议字段：
     - `finding`（关键发现）
     - `evidence_ids`（证据来源）
     - `confidence`（0-1）
     - `open_questions`（仍未解决点）
4. **写回主池**
   - 把 `fused_node` 写入主池；
   - 临时节点并入主池或按策略丢弃（建议保留 hash 索引便于后续去重）。

**融合 Prompt 建议**
- 输入限制：只给“已筛选 top-p”节点，避免上下文过长。
- 输出格式：JSON，便于程序解析并用于终止判定。

**最小伪代码**
```python
def fuse_temp_into_main(query, temp_nodes, main_nodes):
    ranked = rank_by_relevance(temp_nodes, query)       # 相关性排序
    selected = ranked[:max(1, int(len(ranked)*0.5))]   # top-p
    unique_nodes = dedup(selected, by_hash=True, by_semantic=True)
    resolved = resolve_conflicts(unique_nodes, prefer="VERIDICAL")

    fused = llm_fuse(
        query=query,
        evidence=[n.cue for n in resolved],
        output_schema={"finding": "str", "evidence_ids": "list", "confidence": "float"}
    )
    main_nodes.add(FusionNode.from_payload(fused, sources=[n.id for n in resolved]))
    main_nodes.extend(temp_nodes)  # 或仅写 hash 索引
    return main_nodes
```

### 6.6 Probe Expansion 大致生成方式（实现指南）

> 你消息里写“生存”，这里按“生成”理解。

**作用**
- 当当前答案不充分时，把“大问题”拆成少量“高信息增益子问题”，用于下一轮补证。

**触发时机**
- 回答草稿出现“未知/无法判断/证据不足”。
- 或终止门控判定“不足以最终回答”。

**输入**
- `query`
- `answer_draft`
- `current_context`（本轮三层证据）
- `history_probes`（避免重复）

**输出**
- `probes: List[str]`（建议 1~3 条，最多 5 条）

**生成策略（推荐）**
1. **缺口识别（Gap Detection）**
   - 从草稿中抽取未回答槽位：人名、时间、地点、因果、动机、约束条件。
2. **候选生成**
   - 用模板 + LLM 联合：
     - 模板：`谁/何时/何地/为何/如何` 维度各给 1 条候选；
     - LLM：根据缺口重写成可检索问句。
3. **质量打分**
   - 对候选 probe 按以下指标打分：
     - 与原问题相关性
     - 与历史 probe 的差异度
     - 预估信息增益（是否能显著缩小答案空间）
4. **去重与截断**
   - 语义去重后保留 top-k（一般 2~3 个）。

**Probe 生成 Prompt 约束**
- 必须输出 JSON：`{"probe_1": "...", "probe_2": "..."}`
- 每个 probe 一句话、可直接用于检索，不要解释。
- 不允许与历史 probe 重复或同义改写。

**最小伪代码**
```python
def generate_probes(query, draft, context, history_probes):
    gaps = detect_answer_gaps(query, draft)  # e.g. missing_time, missing_cause
    seed = template_probes(query, gaps)      # who/when/where/why
    llm_candidates = llm_expand_probes(query, draft, context, history_probes, gaps)
    candidates = seed + llm_candidates

    candidates = semantic_dedup(candidates, threshold=0.88)
    candidates = [p for p in candidates if not is_redundant(p, history_probes)]
    scored = score_probe_gain(candidates, query, context)
    return topk(scored, k=3)
```

**常见失败与修正**
- probe 太泛：加“必须包含实体/事件锚点”的约束。
- probe 重复：提高历史去重阈值并加入 n-gram 去重。
- probe 太多：强制 top-k<=3，避免每轮成本失控。

---

## 7. MVP 到增强版的落地路线

### 阶段 1（MVP，1-2 天）
- 仅 Veridical 检索 + 单轮回答
- 输出基础结果文件

### 阶段 2（增强检索）
- 加入 Semantic/Episodic 两层
- 三路检索合并上下文

### 阶段 3（迭代推理）
- 加入 probe 与多轮控制
- 引入 MemoryPool

### 阶段 4（图增强与重排）
- 实体关系图 + 图检索
- rerank 模块

### 阶段 5（工程化）
- 缓存、断点续跑、并行、可观测性

---

## 8. 给“其他 AI”的实施指令模板

你可以把下面这段直接贴给实现 AI：

```text
请基于当前项目实现一个“非源码搬运”的三层知识源 RAG：
1) 设计 Veridical/Semantic/Episodic 三层索引与检索接口；
2) 用统一的 LLM/Embedding 抽象，支持 OpenAI-compatible API；
3) 实现 TriRetriever，返回三层证据；
4) 实现 MemoryPool 与迭代控制器：回答失败时自动生成 probe 并继续检索；
5) 输出 results.json 与每题详细过程日志；
6) 先交付 MVP（仅 Veridical），再增量加入 Semantic/Episodic 和迭代推理；
7) 不复制外部仓库代码，按本项目的数据结构与命名规范实现。
```

---

## 9. 避免“照搬代码”的检查清单

- 不复制原仓库类名和函数体，改成你项目自己的命名。
- Prompt 模板重写，不直接复用原文案。
- 检索融合策略可换（加权、级联、学习排序均可）。
- 日志字段、目录结构可相似但不逐字一致。
- 先保证接口和流程正确，再逐步优化效果。

---

## 10. 最小接口示例（伪代码）

```python
class TriRetriever:
    def retrieve(self, query: str, memory_state) -> dict:
        return {
            "veridical": self.ver_store.search(query, top_k=50),
            "semantic": self.sem_store.search(query, top_k=20),
            "episodic": self.epi_store.search(query, top_k=20),
        }

class ReasoningController:
    def answer(self, query: str):
        mem = MemoryPool()
        for step in range(self.max_steps):
            docs = self.retriever.retrieve(query, mem)
            cues = self.memory_encoder.encode(docs, query)
            result = self.llm.solve(query, cues, mem.history())
            if result.is_final:
                return result
            probes = self.probe_generator.generate(query, result, mem)
            mem.update(probes, cues)
```

---

如果你后面希望，我可以继续补一版：  
**“结合你现有项目目录的定制版任务拆解（到文件级别）”**，让其他 AI 直接按文件清单开工。

---

## 11. 面向 Calibre-Web 的整合方案（LanceDB + 小说理解增强）

> 目标：把“电子书管理系统（Calibre-Web）”与“三层知识源推理能力”结合。  
> 设计原则：不把 ComoRAG 整个搬进来，而是抽取可复用能力，做成可插拔 AI 子系统。

### 11.1 总体架构（你项目里的角色分工）

- **Calibre-Web 主系统**
  - 负责书籍元数据、用户、权限、书架、阅读入口。
  - 提供 AI 入口（问答、角色追踪、剧情回顾）。
- **AI 索引服务（新）**
  - 监听“书籍入库/更新”事件；
  - 负责 chunk、embedding、三层索引构建；
  - 维护 LanceDB + SQLite 元数据。
- **AI 推理服务（新）**
  - 在线处理用户提问；
  - 执行 tri-retrieve + memory loop + probe expansion；
  - 返回答案、证据引用、时间线片段。

### 11.2 存储规划（按你的技术选择）

**LanceDB（向量）**
- 表 `book_chunks`（Veridical）
  - `book_id, chunk_id, chapter, order, text, vector, hash`
- 表 `book_semantic`（Semantic）
  - `book_id, summary_id, level, text, member_chunk_ids, vector`
- 表 `book_episodic`（Episodic）
  - `book_id, episode_id, level, window_range, text, vector`

**SQLite（结构化）**
- `books_ai_state`：每本书索引状态（pending/running/done/failed）
- `chunk_map`：`chunk_id -> chapter/order/hash` 索引
- `memory_sessions`：每个用户会话的记忆状态
- `llm_cache`：请求缓存（可沿用现有 sqlite cache 思路）
- `jobs`：异步任务队列状态

### 11.3 每本书的离线处理流程（入库后自动触发）

1. **抽取文本**
   - 支持 epub/mobi/azw3/pdf（先做统一纯文本抽取）。
   - 按章节保留结构信息（chapter title + order）。
2. **Chunk**
   - 推荐按章节内滑窗分块（token chunk + overlap）。
   - 产出 `chunk_id/hash/order/chapter`。
3. **Veridical 索引**
   - 批量 embedding，写 LanceDB `book_chunks`。
4. **Semantic 索引**
   - 对 chunk embedding 聚类，生成 cluster summaries；
   - 写 LanceDB `book_semantic`。
5. **Episodic 索引**
   - 按章节顺序窗口摘要（timeline），写 `book_episodic`。
6. **状态落库**
   - SQLite 更新该书状态为 `done`，记录耗时、token、失败重试次数。

### 11.4 在线问答流程（用户在 Calibre-Web 页面提问）

1. 根据 `book_id + question` 初始化/恢复 `memory_session`。
2. tri-retrieve 并行查三张 LanceDB 表（限定 `book_id`）。
3. 进入推理 loop：
   - memory encode -> answer attempt -> decision gate；
   - 不足则 probe expansion -> 增量检索 -> memory fusion。
4. 返回结果给前端：
   - 最终答案；
   - 引用片段（chapter + chunk_id）；
   - 可选显示“剧情时间线证据”和“推理步骤摘要”。

### 11.5 如何改造当前项目（最小侵入迁移）

**优先改造 1：存储抽象层**
- 目标：把现有 `EmbeddingStore` 改成接口。
- 做法：
  - 保留 `EmbeddingStore` 方法签名不变；
  - 新增 `LanceEmbeddingStore` 实现；
  - 配置项控制 backend（`parquet` / `lancedb`）。

**优先改造 2：数据域从“单数据集”改为“book_id”**
- 现有逻辑偏 dataset 路径，需引入 `book_id` 作为一级隔离键。
- tri-retrieve 时所有检索必须带 `book_id` filter。

**优先改造 3：把离线索引和在线问答拆成两个入口**
- `index_book(book_id)`：仅做离线构建；
- `ask_book(book_id, question, session_id)`：仅做在线推理。

**优先改造 4：会话记忆持久化（SQLite）**
- 当前 `MemoryPool` 是内存态；
- 在 Calibre-Web 里建议按 `user_id + book_id + session_id` 持久化主记忆摘要。

### 11.6 建议新增的接口契约（给其他 AI 实现）

```python
def index_book(book_id: str, raw_text: str, metadata: dict) -> dict:
    """离线构建三层索引，返回统计信息"""

def ask_book(book_id: str, question: str, session_id: str, user_id: str) -> dict:
    """在线问答，返回 answer + citations + trace"""

def rebuild_book_index(book_id: str, force: bool = False) -> dict:
    """书籍重建索引（文本更新后使用）"""
```

### 11.7 小说场景的专项优化建议

- **章节感知检索**：query 命中后优先返回同章邻近 chunk。
- **角色别名归一化**：维护角色 alias 表（例如昵称、称呼、翻译名）。
- **时间线强化**：Episodic 层保留 `chapter_range`，便于前端展示“剧情进度”。
- **剧透控制**：按用户阅读进度过滤可检索章节（Calibre-Web 特别有用）。
- **答案可追溯**：返回 `chapter + paragraph` 引用，提升用户信任。

### 11.9 交给其他 AI 的定制提示词（可直接用）

```text
请基于我的 Calibre-Web 项目实现一个 AI 子系统，不复制外部仓库代码：
1) 使用 LanceDB 存储三层向量：book_chunks/book_semantic/book_episodic；
2) 使用 SQLite 存储索引状态、会话记忆、任务状态、LLM cache；
3) 实现 index_book(book_id) 与 ask_book(book_id, question, session_id) 两个核心接口；
4) 问答流程采用 tri-retrieve + memory loop + probe expansion + memory fusion；
5) 所有检索必须按 book_id 过滤；
6) 输出答案时附带章节级引用（chapter/chunk_id）；
7) 支持后续扩展为异步任务与可观测性看板。
```

