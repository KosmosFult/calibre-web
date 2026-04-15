import base64
import json
import logging
import os
import re

from sqlalchemy.sql.expression import func

from .. import calibre_db, config, db
from ..comorag import service as comorag_service
from .registry import AgentTool

log = logging.getLogger("calibre-web.ai")


def format_books(books):
    results = []
    for book in books:
        authors = [a.name for a in book.authors]
        tags = [t.name for t in book.tags]
        rating = 0
        if book.ratings and len(book.ratings) > 0:
            rating = book.ratings[0].rating

        results.append(
            {
                "id": book.id,
                "title": book.title,
                "authors": authors,
                "tags": tags,
                "rating": rating,
                "year": book.pubdate.year if book.pubdate else "Unknown",
                "description": book.comments[0].text[:200] + "..." if book.comments else "无简介",
            }
        )
    return json.dumps(results, ensure_ascii=False)


def _estimate_text_words(text: str) -> int:
    return len((text or "").split())


def _serialize_chunk_payload(row):
    return {
        "order_id": row.get("order_id"),
        "chunk_id": row.get("hash_id") or row.get("chunk_id"),
        "chapter_id": row.get("chapter_id"),
        "chapter_index": row.get("chapter_index"),
        "chapter_title": row.get("chapter_title"),
        "chunk_index_in_chapter": row.get("chunk_index_in_chapter"),
        "word_count": int(row.get("word_count") or _estimate_text_words(row.get("content") or row.get("text") or "")),
        "char_count": int(row.get("char_count") or len((row.get("content") or row.get("text") or ""))),
        "text": row.get("content") or row.get("text") or "",
    }


@AgentTool(
    name="get_book_cover",
    description="获取书籍的封面图片。当需要向用户介绍书籍外观或封面细节时调用。",
    parameters={
        "type": "object",
        "properties": {
            "book_id": {
                "type": "integer",
                "description": "书籍的 ID",
            }
        },
        "required": ["book_id"],
    },
)
def get_book_cover(book_id: int):
    """
    获取书籍的封面图片。当需要向用户介绍书籍外观或封面细节时调用。

    Args:
        book_id: 书籍的 ID
    """
    session = calibre_db.session
    book = session.query(db.Books).filter(db.Books.id == book_id).first()

    if not book:
        return json.dumps({"status": "error", "message": "Book not found"})

    if not book.has_cover:
        return json.dumps({"status": "error", "message": "Book has no cover"})

    try:
        library_path = config.config_calibre_dir
        book_path = book.path
        cover_path = os.path.join(library_path, book_path, "cover.jpg")

        if os.path.exists(cover_path):
            with open(cover_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            return json.dumps(
                {
                    "status": "success",
                    "message": "Cover loaded successfully",
                    "_image_data": image_data,
                    "_image_path": cover_path,
                }
            )
        else:
            return json.dumps({"status": "error", "message": "Cover file missing from disk"})

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@AgentTool(
    name="find_books",
    description="根据关键词在图书馆本地查找书籍。可以搜索书名、作者或标签。如果用户没有指定搜索字段，默认全搜。",
    parameters={
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "搜索关键词，例如 '科幻', '三体', 'J.K. Rowling'",
            },
            "field": {
                "type": "string",
                "enum": ["title", "author", "tag", "all"],
                "description": "搜索字段：title(书名), author(作者), tag(标签), all(全部)。默认为 all",
            },
        },
        "required": ["keyword"],
    },
)
def search_books(keyword: str, field: str = "all"):
    """
    根据关键词在图书馆本地查找书籍。可以搜索书名、作者或标签。如果用户没有指定搜索字段，默认全搜。

    Args:
        keyword: 搜索关键词，例如 '科幻', '三体', 'J.K. Rowling'
        field: 搜索字段：title(书名), author(作者), tag(标签), all(全部)。默认为 all
    """
    session = calibre_db.session
    query = session.query(db.Books)
    limit = 5

    if field == "title":
        query = query.filter(db.Books.title.ilike(f"%{keyword}%"))
    elif field == "author":
        query = (
            query.join(db.books_authors_link)
            .join(db.Authors)
            .filter(db.Authors.name.ilike(f"%{keyword}%"))
        )
    elif field == "tag":
        query = (
            query.join(db.books_tags_link)
            .join(db.Tags)
            .filter(db.Tags.name.ilike(f"%{keyword}%"))
        )
    elif field == "all":
        query = query.filter(db.Books.title.ilike(f"%{keyword}%"))

    books = query.limit(limit).all()
    if not books:
        return json.dumps({"status": "empty", "message": f"没有找到包含 '{keyword}' 的书籍。"})
    return format_books(books)


@AgentTool(
    name="get_library_stats",
    description="获取图书馆的统计信息，如总书目数、作者数等。",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
)
def get_library_stats():
    """
    获取图书馆的统计信息，如总书目数、作者数等。
    """
    session = calibre_db.session
    book_count = session.query(db.Books).count()
    author_count = session.query(db.Authors).count()

    return json.dumps({"total_books": book_count, "total_authors": author_count}, ensure_ascii=False)


@AgentTool(
    name="get_recent_books",
    description="获取最近入库的新书。",
    parameters={
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "返回数量，默认为 5",
            }
        },
        "required": [],
    },
)
def get_recent_books(limit: int = 5):
    """
    获取最近入库的新书。

    Args:
        limit: 返回数量，默认为 5
    """
    session = calibre_db.session
    books = session.query(db.Books).order_by(db.Books.timestamp.desc()).limit(limit).all()
    if not books:
        return json.dumps({"status": "empty", "message": "书库为空"})
    return format_books(books)


@AgentTool(
    name="get_random_books",
    description="随机推荐几本书籍。",
    parameters={
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "返回数量，默认为 5",
            }
        },
        "required": [],
    },
)
def get_random_books(limit: int = 5):
    """
    随机推荐几本书籍。

    Args:
        limit: 返回数量，默认为 5
    """
    session = calibre_db.session
    books = session.query(db.Books).order_by(func.random()).limit(limit).all()
    if not books:
        return json.dumps({"status": "empty", "message": "书库为空"})
    return format_books(books)


@AgentTool(
    name="get_books_by_rating",
    description="获取评分最高的书籍。",
    parameters={
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "返回数量，默认为 5",
            }
        },
        "required": [],
    },
)
def get_books_by_rating(limit: int = 5):
    """
    获取评分最高的书籍。

    Args:
        limit: 返回数量，默认为 5
    """
    session = calibre_db.session
    books = (
        session.query(db.Books)
        .join(db.books_ratings_link)
        .join(db.Ratings)
        .order_by(db.Ratings.rating.desc(), db.Books.timestamp.desc())
        .limit(limit)
        .all()
    )

    if not books:
        return json.dumps({"status": "empty", "message": "没有评分的书籍"})
    return format_books(books)


@AgentTool(
    name="get_book_outline",
    description="获取书籍的章节目录和每章对应的 chunk order_id 范围。用于理解阅读结构，优先于旧的章节接口。",
    parameters={
        "type": "object",
        "properties": {
            "book_id": {
                "type": "integer",
                "description": "书籍的 ID",
            }
        },
        "required": ["book_id"],
    },
)
def get_book_outline(book_id: int):
    try:
        outline = comorag_service.get_book_outline(book_id=int(book_id))
        if not outline.get("chapters"):
            return json.dumps(
                {
                    "status": "empty",
                    "message": "Book outline not available. Build the ComoRAG index first.",
                    "book_id": int(book_id),
                },
                ensure_ascii=False,
            )
        return json.dumps({"status": "success", **outline}, ensure_ascii=False)
    except Exception as e:  # pylint: disable=broad-except
        log.exception("get_book_outline failed")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@AgentTool(
    name="get_book_chapters",
    description="获取书籍章节列表。兼容旧接口，内部基于 chunk 索引目录实现；建议优先使用 get_book_outline。",
    parameters={
        "type": "object",
        "properties": {
            "book_id": {
                "type": "integer",
                "description": "书籍的 ID",
            }
        },
        "required": ["book_id"],
    },
)
def get_book_chapters(book_id: int):
    try:
        outline = comorag_service.get_book_outline(book_id=int(book_id))
        chapters_info = [
            {
                "index": chapter["chapter_index"],
                "title": chapter["chapter_title"],
                "word_count": chapter["word_count"],
                "chunk_count": chapter["chunk_count"],
                "start_order_id": chapter["start_order_id"],
                "end_order_id": chapter["end_order_id"],
            }
            for chapter in outline.get("chapters", [])
        ]
        return json.dumps(
            {
                "status": "success" if chapters_info else "empty",
                "book_id": int(book_id),
                "total_chapters": len(chapters_info),
                "total_chunks": outline.get("total_chunks", 0),
                "chapters": chapters_info,
            },
            ensure_ascii=False,
        )

    except Exception as e:
        log.error(f"Failed to extract book chapters: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@AgentTool(
    name="read_book_chapter",
    description="读取书籍的指定章节内容。兼容旧接口，内部按 chunk 顺序拼接；建议优先使用 read_chapter_by_chunks 或 read_book_segment。",
    parameters={
        "type": "object",
        "properties": {
            "book_id": {
                "type": "integer",
                "description": "书籍的 ID",
            },
            "chapter_index": {
                "type": "integer",
                "description": "章节索引（从 0 开始）。如果不指定，默认读取第一章",
            },
            "max_words": {
                "type": "integer",
                "description": "最多返回多少字，避免内容过长。默认 3000 字",
            },
        },
        "required": ["book_id"],
    },
)
def read_book_chapter(book_id: int, chapter_index: int = 0, max_words: int = 3000):
    try:
        chapter_payload = comorag_service.get_chapter_chunks(
            book_id=int(book_id),
            chapter_index=int(chapter_index),
            start_chunk_offset=0,
            limit_chunks=1000,
        )
        if chapter_payload.get("status") != "success":
            return json.dumps(chapter_payload, ensure_ascii=False)
        chunks = chapter_payload.get("chunks", [])
        pieces = []
        current_words = 0
        truncated = False
        for row in chunks:
            text = row.get("content") or row.get("text") or ""
            text_words = _estimate_text_words(text)
            if current_words and current_words + text_words > max_words:
                truncated = True
                break
            pieces.append(text)
            current_words += text_words
        content = "\n\n".join(pieces)
        chapter = chapter_payload["chapter"]
        if truncated:
            content += "\n\n[... 内容过长，已按 chunk 截断，可继续调用 read_chapter_by_chunks 读取后续部分]"

        return json.dumps(
            {
                "status": "success",
                "book_id": int(book_id),
                "chapter_index": chapter["chapter_index"],
                "chapter_title": chapter["chapter_title"],
                "start_order_id": chapter["start_order_id"],
                "end_order_id": chapter["end_order_id"],
                "content": content,
                "total_word_count": chapter["word_count"],
                "returned_words": current_words,
            },
            ensure_ascii=False,
        )

    except Exception as e:
        log.error(f"Failed to read book chapter: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@AgentTool(
    name="read_book_segment",
    description="从指定 order_id 开始顺序读取连续多个 chunk。适合从头读、继续往下读，或按顺序核查原文。",
    parameters={
        "type": "object",
        "properties": {
            "book_id": {"type": "integer", "description": "书籍 ID"},
            "start_order_id": {"type": "integer", "description": "起始 chunk 顺序号（order_id）"},
            "limit": {"type": "integer", "description": "连续返回多少个 chunk，默认 3"},
        },
        "required": ["book_id", "start_order_id"],
    },
)
def read_book_segment(book_id: int, start_order_id: int, limit: int = 3):
    try:
        rows = comorag_service.get_chunks_by_order_range(
            book_id=int(book_id),
            start_order_id=int(start_order_id),
            limit=max(1, int(limit)),
        )
        if not rows:
            return json.dumps(
                {
                    "status": "empty",
                    "message": "No chunks found for this range",
                    "book_id": int(book_id),
                    "start_order_id": int(start_order_id),
                },
                ensure_ascii=False,
            )
        items = [_serialize_chunk_payload(row) for row in rows]
        return json.dumps(
            {
                "status": "success",
                "book_id": int(book_id),
                "start_order_id": int(start_order_id),
                "returned_count": len(items),
                "items": items,
            },
            ensure_ascii=False,
        )
    except Exception as e:  # pylint: disable=broad-except
        log.exception("read_book_segment failed")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@AgentTool(
    name="read_book_window",
    description="围绕某个命中的 order_id 读取上下文窗口。适合在检索命中后核查前后文。",
    parameters={
        "type": "object",
        "properties": {
            "book_id": {"type": "integer", "description": "书籍 ID"},
            "center_order_id": {"type": "integer", "description": "中心 chunk 的 order_id"},
            "before": {"type": "integer", "description": "向前读取多少个 chunk，默认 1"},
            "after": {"type": "integer", "description": "向后读取多少个 chunk，默认 1"},
        },
        "required": ["book_id", "center_order_id"],
    },
)
def read_book_window(book_id: int, center_order_id: int, before: int = 1, after: int = 1):
    try:
        rows = comorag_service.get_chunk_window(
            book_id=int(book_id),
            center_order_id=int(center_order_id),
            before=max(0, int(before)),
            after=max(0, int(after)),
        )
        if not rows:
            return json.dumps(
                {
                    "status": "empty",
                    "message": "No chunks found for this window",
                    "book_id": int(book_id),
                    "center_order_id": int(center_order_id),
                },
                ensure_ascii=False,
            )
        items = [_serialize_chunk_payload(row) for row in rows]
        return json.dumps(
            {
                "status": "success",
                "book_id": int(book_id),
                "center_order_id": int(center_order_id),
                "items": items,
            },
            ensure_ascii=False,
        )
    except Exception as e:  # pylint: disable=broad-except
        log.exception("read_book_window failed")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@AgentTool(
    name="read_chapter_by_chunks",
    description="按章节内的 chunk 范围读取内容。适合分段阅读某一章，不会一次返回整章大文本。",
    parameters={
        "type": "object",
        "properties": {
            "book_id": {"type": "integer", "description": "书籍 ID"},
            "chapter_index": {"type": "integer", "description": "章节索引（从 0 开始）"},
            "start_chunk_offset": {"type": "integer", "description": "从该章节内第几个 chunk 开始，默认 0"},
            "limit_chunks": {"type": "integer", "description": "最多返回多少个 chunk，默认 5"},
        },
        "required": ["book_id", "chapter_index"],
    },
)
def read_chapter_by_chunks(book_id: int, chapter_index: int, start_chunk_offset: int = 0, limit_chunks: int = 5):
    try:
        payload = comorag_service.get_chapter_chunks(
            book_id=int(book_id),
            chapter_index=int(chapter_index),
            start_chunk_offset=max(0, int(start_chunk_offset)),
            limit_chunks=max(1, int(limit_chunks)),
        )
        if payload.get("status") != "success":
            return json.dumps(payload, ensure_ascii=False)
        items = [_serialize_chunk_payload(row) for row in payload.get("chunks", [])]
        return json.dumps(
            {
                "status": "success",
                "book_id": int(book_id),
                "chapter": payload["chapter"],
                "returned_count": len(items),
                "items": items,
            },
            ensure_ascii=False,
        )
    except Exception as e:  # pylint: disable=broad-except
        log.exception("read_chapter_by_chunks failed")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


def _extract_order_ids_from_rag_fields(*fields):
    order_ids = set()
    source_pattern = re.compile(r"source_order_ids=([0-9,\s]+)")
    order_pattern = re.compile(r"order_id=(\d+)")

    for field in fields:
        if not field:
            continue
        text = str(field)
        for match in order_pattern.findall(text):
            try:
                order_ids.add(int(match))
            except ValueError:
                continue
        for segment in source_pattern.findall(text):
            for part in segment.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    order_ids.add(int(part))
                except ValueError:
                    continue
    return sorted(order_ids)


@AgentTool(
    name="comorag_ask_book",
    description=(
        "使用 ComoRAG 根据问题对一本书的内容进行检索。"
        "当用户提到剧情细节、角色关系、时间线、证据链、凶手推理等书内问题时优先调用。"
        "它返回的是一份可验证的推理草稿 answer_draft，而不是最终定稿。"
        "草稿中可能包含 [order_id=...] 注释，主 Agent 可以据此自行决定是否继续阅读原文、翻页推理，或再次向 comorag 发起新的问题。"
        "参数必须包含 book_id 与 question。若 return_chunks=true，将额外返回草稿中提到的 chunk order_id 列表。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "book_id": {
                "type": "integer",
                "description": "书籍 ID",
            },
            "question": {
                "type": "string",
                "description": "要问这本书的问题，建议完整自然语言。",
            },
            "return_chunks": {
                "type": "boolean",
                "description": "是否返回命中的 chunk order_id 列表。默认 false。",
            },
        },
        "required": ["book_id", "question"],
    },
)
def comorag_ask_book(book_id: int, question: str, return_chunks: bool = False):
    try:
        result = comorag_service.ask_book(book_id=int(book_id), question=question)
        if result.get("status") != "success":
            return json.dumps(result, ensure_ascii=False)

        # Keep payload concise for tool-call context window.
        condensed = {
            "status": "success",
            "book_id": result.get("book_id"),
            "question": result.get("question"),
            "answer": result.get("answer"),
            "answer_draft": result.get("answer_draft") or result.get("answer"),
            "trace": result.get("trace") or {},
            "evidence_preview": (result.get("docs") or "")[:2400],
            "summary_preview": (result.get("summary") or "")[:1600],
            "timeline_preview": (result.get("timeline") or "")[:1600],
            "index": result.get("index"),
        }
        if return_chunks:
            matched_order_ids = (result.get("trace") or {}).get("mentioned_order_ids") or _extract_order_ids_from_rag_fields(
                result.get("answer_draft") or result.get("answer"),
                result.get("docs"),
                result.get("summary"),
                result.get("timeline"),
            )
            condensed["matched_order_ids"] = matched_order_ids
            condensed["matched_order_count"] = len(matched_order_ids)
        return json.dumps(condensed, ensure_ascii=False)
    except Exception as e:  # pylint: disable=broad-except
        log.exception("comorag_ask_book failed")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@AgentTool(
    name="get_book_chunk_by_order",
    description=(
        "按 book_id + order_id 获取原始 chunk 文本，其中 order_id 对应 chunk_index。相邻chunk文本的order_id是连续的。"
        "你可以通过此工具阅读书籍原文片段"
    ),
    parameters={
        "type": "object",
        "properties": {
            "book_id": {
                "type": "integer",
                "description": "书籍 ID",
            },
            "order_id": {
                "type": "integer",
                "description": "chunk 顺序号（即 chunk_index，从 0 开始）",
            },
        },
        "required": ["book_id", "order_id"],
    },
)
def get_book_chunk_by_order(book_id: int, order_id: int):
    try:
        row = comorag_service.get_chunk_by_order(book_id=int(book_id), order_id=int(order_id))
        if not row:
            return json.dumps(
                {
                    "status": "empty",
                    "message": "Chunk not found for this book/order_id",
                    "book_id": int(book_id),
                    "order_id": int(order_id),
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "status": "success",
                "book_id": int(book_id),
                "order_id": int(order_id),
                "chunk_id": row.get("hash_id"),
                "chapter_id": row.get("chapter_id"),
                "chapter_index": row.get("chapter_index"),
                "chapter_title": row.get("chapter_title"),
                "chunk_index_in_chapter": row.get("chunk_index_in_chapter"),
                "word_count": row.get("word_count"),
                "char_count": row.get("char_count"),
                "text": row.get("content"),
            },
            ensure_ascii=False,
        )
    except Exception as e:  # pylint: disable=broad-except
        log.exception("get_book_chunk_by_order failed")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


__all__ = [
    "get_book_cover",
    "search_books",
    "get_library_stats",
    "get_recent_books",
    "get_random_books",
    "get_books_by_rating",
    "get_book_outline",
    "get_book_chapters",
    "read_book_chapter",
    "read_book_segment",
    "read_book_window",
    "read_chapter_by_chunks",
    "comorag_ask_book",
    "get_book_chunk_by_order",
]
