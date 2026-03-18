import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cps.comorag import ComoRAG


def build_smoke_docs():
    # 按“顺序 chunk”组织的最小示例文本
    return [
        "第一天夜里，侦探林泽来到雾港旅店，发现馆主对失踪事件三缄其口。",
        "第二天清晨，旅店后院出现带泥脚印，脚印从仓库通向海边的小码头。",
        "第三天傍晚，林泽在码头仓库里找到带血的袖扣，并确认它属于失踪者周迟。",
    ]


def build_query():
    return "侦探目前掌握了哪些关键线索？"


def run_smoke_test():
    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_GENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY/GOOGLE_API_KEY/GOOGLE_GENAI_API_KEY")

    # base_url = (
    #     os.environ.get("GENAI_BASE_URL")
    #     or os.environ.get("GOOGLE_GENAI_BASE_URL")
    #     or os.environ.get("GOOGLE_API_BASE_URL")
    # )

    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

    llm_model = os.environ.get("COMORAG_LLM_MODEL", "gemini-3.1-flash-lite-preview")
    embedding_model = os.environ.get("COMORAG_EMBEDDING_MODEL", "gemini-embedding-2-preview")

    rag = ComoRAG(
        llm_model_name=llm_model,
        llm_base_url=base_url,
        llm_api_key=api_key,
        embedding_model_name=embedding_model,
        embedding_base_url=base_url,
        embedding_api_key=api_key,
    )
    # 先保持 need_cluster=False 的轻量路径，验证主链路可用
    rag.global_config.need_cluster = False
    rag.global_config.openie_mode = "online"
    rag.global_config.llm_provider = "google_genai"
    rag.global_config.embedding_provider = "google_genai"

    docs = build_smoke_docs()
    rag.index(docs)

    results = rag.try_answer([build_query()])
    if not results:
        raise RuntimeError("ComoRAG returned empty result list")

    first = results[0]
    print("\n=== ComoRAG Smoke Test ===")
    print("Question:", first.question)
    print("Answer:", first.answer)
    print("Retrieved docs:", len(first.docs) if first.docs else 0)


if __name__ == "__main__":
    run_smoke_test()
