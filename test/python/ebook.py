import os
import sys
import time
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cps.comorag import ComoRAG
from cps.book_content_extractor import BookContentExtractor
from ebooklib import epub
from cps.ai_search.chunking import BookParser
from cps.ai_db import get_session, BookChunk


def read_epub(path):
    book = epub.read_epub(path)
    chapters = []

    # 获取书名等元数据
    title = book.get_metadata('DC', 'title')[0][0]

    for item in book.get_items():
        # 只要文档类型的章节
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text()
            # 简单的过滤，去掉太短的章节（如目录页）
            if len(text) > 500:
                chapters.append(text)
    return title, chapters


def build_smoke_docs():
    """
    Use BookParser chunking logic without requiring Flask app context.
    We parse EPUB directly, then call parser internals to split into chunks.
    """
    epub_path = "/Users/kosmosfult/Documents/books/世界树之棺 - 筒城灯士郎.epub"
    # epub_path = "/Users/kosmosfult/Documents/books/永劫馆超连续杀人事件魔女决定与X赴死 ([日]南海游,译者李影恒) (z-library.sk, 1lib.sk, z-lib.sk).epub"

    parser = BookParser(chunk_size=800, chunk_overlap=200, enable_contextual=False)
    book = epub.read_epub(epub_path)
    chapter_docs = parser._extract_chapters(book)
    chunks = parser._chunk_chapters(book_id=13, chapters=chapter_docs)
    return [chunk.text for chunk in chunks]


# def build_smoke_docs_with_app_context(book_id: int = 13):
#     """
#     Alternative path: if you need `chunk_book(book_id)` (Calibre path resolution + DB persist),
#     run it under Flask app context and then read chunk text from ai.db.
#     """
#     from cps import create_app
#
#     app = create_app()
#     parser = BookParser(chunk_size=800, chunk_overlap=200, enable_contextual=False)
#     with app.app_context():
#         parser.chunk_book(book_id)
#
#     sess = get_session()
#     try:
#         rows = (
#             sess.query(BookChunk)
#             .filter(BookChunk.book_id == book_id)
#             .order_by(BookChunk.chunk_index.asc())
#             .all()
#         )
#         return [row.text for row in rows]
#     finally:
#         sess.close()


def build_query():
    return "【恋塚】的帝国女士兵“姐姐大人”真的是世界树洋房里的“姐姐大人”格雷吗"


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

    docs = build_smoke_docs()

    rag = ComoRAG(
        book_id=13,
        llm_model_name=llm_model,
        llm_base_url=base_url,
        llm_api_key=api_key,
        embedding_model_name=embedding_model,
        embedding_base_url=base_url,
        embedding_api_key=api_key,
    )
    # 先保持 need_cluster=False 的轻量路径，验证主链路可用
    rag.global_config.need_cluster = True
    rag.global_config.openie_mode = "online"
    rag.global_config.llm_provider = "google_genai"
    rag.global_config.embedding_provider = "google_genai"
    rag.max_tokens_ver = 6000
    rag.max_tokens_sem = 3000
    rag.max_tokens_epi = 3000

    rag.index(docs)

    try_answer_start = time.perf_counter()
    results = rag.try_answer([build_query()])
    try_answer_elapsed = time.perf_counter() - try_answer_start
    if not results:
        raise RuntimeError("ComoRAG returned empty result list")

    first = results[0]
    print("\n=== ComoRAG Smoke Test ===")
    print("Question:", first.question)
    print("Answer:", first.answer)
    print("Retrieved docs:", len(first.docs) if first.docs else 0)
    return try_answer_elapsed


if __name__ == "__main__":
    try_answer_elapsed = run_smoke_test()
    print(f"try_answer elapsed: {try_answer_elapsed:.3f}s")
