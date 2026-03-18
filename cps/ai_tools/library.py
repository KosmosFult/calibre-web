import base64
import json
import logging
import os

from sqlalchemy.sql.expression import func

from .. import calibre_db, config, db
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
    name="get_book_chapters",
    description="获取书籍的章节列表（不含内容）。用于了解书籍结构。",
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
    """
    获取书籍的章节列表（不含内容）。用于了解书籍结构。

    Args:
        book_id: 书籍的 ID
    """
    from ..book_content_extractor import BookContentExtractor

    session = calibre_db.session
    book = session.query(db.Books).filter(db.Books.id == book_id).first()

    if not book:
        return json.dumps({"status": "error", "message": "Book not found"})

    readable_formats = ["epub", "kepub", "txt"]
    book_format = None
    book_data = None

    for data in book.data:
        if data.format.lower() in readable_formats:
            book_format = data.format.lower()
            book_data = data
            break

    if not book_format:
        return json.dumps(
            {
                "status": "error",
                "message": f"Book has no readable format. Available: {[d.format for d in book.data]}",
            }
        )

    file_path = os.path.normpath(
        os.path.join(config.config_calibre_dir, book.path, book_data.name + "." + book_format)
    )

    if not os.path.exists(file_path):
        return json.dumps({"status": "error", "message": "Book file not found on disk"})

    try:
        content_data = BookContentExtractor.extract(file_path, book_format)

        chapters_info = [
            {"index": ch["index"], "title": ch["title"], "word_count": ch["word_count"]}
            for ch in content_data["chapters"]
        ]

        return json.dumps(
            {
                "status": "success",
                "book_title": content_data["title"],
                "total_chapters": content_data["total_chapters"],
                "chapters": chapters_info,
            },
            ensure_ascii=False,
        )

    except Exception as e:
        log.error(f"Failed to extract book chapters: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@AgentTool(
    name="read_book_chapter",
    description="读取书籍的指定章节内容。用于回答关于书籍具体内容的问题。",
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
    """
    读取书籍的指定章节内容。用于回答关于书籍具体内容的问题。

    Args:
        book_id: 书籍的 ID
        chapter_index: 章节索引（从 0 开始）。如果不指定，默认读取第一章
        max_words: 最多返回多少字，避免内容过长。默认 3000 字
    """
    from ..book_content_extractor import BookContentExtractor

    session = calibre_db.session
    book = session.query(db.Books).filter(db.Books.id == book_id).first()

    if not book:
        return json.dumps({"status": "error", "message": "Book not found"})

    readable_formats = ["epub", "kepub", "txt"]
    book_format = None
    book_data = None

    for data in book.data:
        if data.format.lower() in readable_formats:
            book_format = data.format.lower()
            book_data = data
            break

    if not book_format:
        return json.dumps(
            {
                "status": "error",
                "message": f"Book has no readable format. Available: {[d.format for d in book.data]}",
            }
        )

    file_path = os.path.normpath(
        os.path.join(config.config_calibre_dir, book.path, book_data.name + "." + book_format)
    )

    if not os.path.exists(file_path):
        return json.dumps({"status": "error", "message": "Book file not found on disk"})

    try:
        content_data = BookContentExtractor.extract(file_path, book_format)

        if chapter_index >= len(content_data["chapters"]):
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Chapter index {chapter_index} out of range. Total chapters: {len(content_data['chapters'])}",
                }
            )

        chapter = content_data["chapters"][chapter_index]
        content = chapter["content"]

        if len(content) > max_words:
            content = content[:max_words] + f"\n\n[... 内容过长，已截断。完整章节共 {chapter['word_count']} 字]"

        return json.dumps(
            {
                "status": "success",
                "book_title": content_data["title"],
                "chapter_index": chapter["index"],
                "chapter_title": chapter["title"],
                "content": content,
                "total_word_count": chapter["word_count"],
                "returned_words": min(len(content), max_words),
            },
            ensure_ascii=False,
        )

    except Exception as e:
        log.error(f"Failed to read book chapter: {e}")
        return json.dumps({"status": "error", "message": str(e)})


__all__ = [
    "get_book_cover",
    "search_books",
    "get_library_stats",
    "get_recent_books",
    "get_random_books",
    "get_books_by_rating",
    "get_book_chapters",
    "read_book_chapter",
]
