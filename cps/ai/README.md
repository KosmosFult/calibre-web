# Calibre-Web AI

该包是 Calibre-Web 的独立 AI 子系统。它只读取 Calibre 书库元数据与电子书文件；所有会话、小说索引、推理 trace 存入 `ai_storage/ai.db`，向量存入 `ai_storage/lancedb/`，不会修改 Calibre 的 `metadata.db`。

## 结构

```text
cps/ai/
├── agent.py              # OpenAI-compatible 图书管家与工具循环
├── config.py             # YAML/环境变量配置
├── db.py                 # AI 会话数据库
├── providers.py          # Responses + Chat Completions 模型适配层
├── routes.py             # Flask API 与 NDJSON 对话流
├── tools/                # 书库与小说研究工具
└── novel/
    ├── ingestion.py      # 保持 EPUB spine/章节顺序的分段
    ├── repository.py     # SQLite 事件图、FTS、索引 run、trace
    ├── vector_store.py   # LanceDB 语义证据通道
    ├── indexing.py       # 世界模型抽取与层级记忆构建
    └── reasoning.py      # plan → retrieve → critic → probe → answer
```

## 模型后端

默认使用 OpenAI Python SDK。`ai.openai.api_mode: responses` 用于结构化推理；若兼容厂商没有实现 Responses API，会自动退回 Chat Completions。也可以显式配置 `chat_completions`。

```yaml
ai:
  openai:
    api_key: "..."             # 推荐改用 OPENAI_API_KEY
    base_url: "https://.../v1" # 可选
    chat_model: "your-model"
    api_mode: "responses"
  novel:
    extraction_model: "your-model"
    reasoning_model: "your-model"
    embedding_model: "your-embedding-model"
```

更换 embedding 模型会创建新的版本化索引，不会把不同维度的向量混进同一张 LanceDB 表。
