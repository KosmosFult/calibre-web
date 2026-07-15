# 小说叙事智能 Agent：架构与研究路线

## 1. 问题定义

小说问答不是普通知识库问答。一个问题往往同时依赖人物身份、叙事顺序与故事时间的区别、跨章节状态变化、因果链、角色认知偏差、伏笔与后续揭示。仅靠相似度最高的若干文本块，会漏掉低词面相似但逻辑必要的前提。

本实现把任务定义为：在长篇叙事的多视角证据上，构造一个可更新的世界状态模型，并执行带证据覆盖检查的多跳推理。

## 2. 三个证据通道

1. **Episodic / 原文片段**：保持 EPUB spine、章节和段落顺序的 chunk，是最终事实锚点。
2. **Semantic / 层级记忆**：章节、故事弧、全书三级摘要，重点保存人物状态、目标、秘密和未解线索，而不是只压缩主题。
3. **Veridical / 事件图**：实体、事件和 `before / causes / enables / motivates / foreshadows / reveals / contradicts` 关系，用于补回向量检索不擅长的逻辑邻接。

SQLite 同时提供 FTS 词面检索；LanceDB 在同一个版本化表内索引 chunk、event、memory 三类对象。检索器用 rank fusion 合并语义、词面和图扩展结果。

## 3. 索引流程

```text
EPUB/TXT
  → 按 spine/章节分段
  → chunk embedding
  → 批量抽取实体、事件、局部关系
  → 章节记忆
  → 跨章节故事弧记忆与事件链接
  → 全书记忆
  → event/memory embedding
  → 原子切换 active run
```

每次构建拥有独立 `run_id` 和 LanceDB 表。失败 run 不会污染在线索引；embedding 模型变化也不会造成向量维度冲突。成功后只保留配置数量的最近 run。

## 4. 推理流程

```text
问题
  → Retrieval Planner（实体、时间、关系、通道、probes）
  → 三通道混合检索
  → Evidence Coverage Critic
      ├─ 足够：生成答案
      └─ 不足：产生针对缺口的新 probe，再检索（有轮数上限）
  → 带 order_id 引用、置信度和不确定性的答案
```

数据库保存的是检索计划、命中证据、coverage 判定和最终答案，不保存模型隐藏思维链。这样既便于调试/评测，也避免把不可验证的推理文本当成事实资产。

## 5. 相比原 ComoRAG 雏形的关键变化

- 用显式事件与状态变化代替松散的三元组堆叠。
- 把“叙事顺序”和“故事时间”分开保存，为倒叙、插叙和不可靠叙述者留出表达空间。
- Probe 不再无条件递归；由 evidence coverage critic 针对缺失前提生成。
- 所有答案回到原文 `order_id`，摘要和事件只作为检索桥梁，不作为不可追溯的最终证据。
- 索引使用 copy-on-write run，在线查询永远读完整版本。
- 模型层依赖 OpenAI 协议而非某一家厂商 SDK。

## 6. 可论文验证的实验

建议把数据库中的 `novel_query_traces` 导出为可复现实验记录，并至少做以下消融：

- dense-only vs dense+FTS vs dense+FTS+event graph；
- 无层级记忆 vs chapter memory vs chapter+arc+book memory；
- 单轮检索 vs coverage-guided probing；
- 仅答案正确率 vs evidence recall / citation precision / causal-chain completeness；
- 顺叙小说 vs 倒叙、多视角、不可靠叙述者子集。

可新增“状态一致性”评测：给定两个章节位置，要求系统判断人物所知、所信、所在位置、持有物与目标是否发生变化，并以原文证据验证。该指标比普通 QA 更直接测量叙事世界模型的质量。
