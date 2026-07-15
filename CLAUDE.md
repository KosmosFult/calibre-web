# Repository AI notes

本项目在 Calibre-Web 上增加了“小说叙事智能”子系统，代码集中于 `cps/ai/`。不要恢复已删除的旧 `cps/comorag`、`cps/ai.py`、`cps/agent.py` 或 Google GenAI 专用实现。

## 关键约束

- 不得修改 Calibre 的 `metadata.db`。AI 数据只写 `ai_storage/ai.db` 与 `ai_storage/lancedb/`。
- 模型调用必须经过 `cps.ai.providers` 或 `cps.ai.agent` 的 OpenAI-compatible 接口。
- 索引构建必须先写入新的 run，完整成功后才能原子切换 `active`。
- 回答书内问题必须保存 retrieval trace，并提供可回查的 `order_id` 证据。
- 不把隐藏思维链写入数据库或返回给前端；只保留检索计划、证据覆盖结论和可检查的简要依据。

## 验证

```bash
.venv/bin/python -m unittest discover -s test/ai -v
.venv/bin/python -m compileall -q cps/ai cps/tasks/novel_index.py
```

架构、算法动机和实验建议见 `docs/NOVEL_AGENT_ARCHITECTURE_zh.md`。
