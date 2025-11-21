# -*- coding: utf-8 -*-
import json
import logging
import inspect
import base64
import os
from functools import wraps
from sqlalchemy.sql.expression import func

# 尝试导入 openai，如果用户没安装也不要报错导致程序崩溃
try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False

log = logging.getLogger("calibre-web.ai")

class AgentTool:
    """
    装饰器：用于将普通 Python 函数注册为 Agent 可调用的工具
    """
    _registry = {}

    def __init__(self, name, description, parameters):
        self.name = name
        self.description = description
        self.parameters = parameters

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # 注册工具
        AgentTool._registry[self.name] = {
            "function": wrapper,
            "schema": {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.parameters
                }
            }
        }
        return wrapper

    @classmethod
    def get_tools_schema(cls):
        return [item["schema"] for item in cls._registry.values()]

    @classmethod
    def get_tool_func(cls, name):
        if name in cls._registry:
            return cls._registry[name]["function"]
        return None

class CalibreAgent:
    def __init__(self, api_key, base_url, model="gemini-2.5-flash", system_prompt=None, enable_web_search=False):
        if not openai_available:
            raise ImportError("OpenAI module is not installed. Please install it using 'pip install openai'")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.enable_web_search = enable_web_search
        self.system_prompt = system_prompt or (
            "你是 Calibre-Web 的 AI 图书管家。你可以给用户推荐书库里面的书籍，给出书籍介绍，回答关于书籍内容的问题。"
            "你可以通过工具查询书库中的各种书籍信息，请在你需要的时候调用。\n\n"
            "可用工具说明：\n"
            "- search_books: 搜索书籍（按书名、作者、标签）\n"
            "- get_recent_books / get_random_books / get_books_by_rating: 推荐书籍\n"
            "- get_book_cover: 获取书籍封面图片\n"
            "- get_book_chapters: 获取书籍的章节列表（了解书籍结构）\n"
            "- read_book_chapter: 读取书籍的具体章节内容\n\n"
            "使用建议：\n"
            "1. 当用户询问推荐书籍时，使用推荐类工具\n"
            "2. 当用户询问书籍的具体内容、情节、人物时，先用 get_book_chapters 了解结构，再用 read_book_chapter 读取相关章节\n"
            "3. 当需要了解书籍外观时，可以调用 get_book_cover\n"
            "4. 回答时要自然流畅，不要暴露工具调用的细节"
        )
        self.history = []
        # 初始化 System Prompt
        self.history.append({"role": "system", "content": self.system_prompt})

    def chat(self, user_message):
        """
        核心对话循环：User -> LLM -> (Tool Call -> Function -> Tool Output -> LLM) -> Final Response
        这是一个生成器，支持流式输出
        """
        # 1. 添加用户消息
        self.history.append({"role": "user", "content": user_message})
        
        # 准备 Tools
        tools = AgentTool.get_tools_schema()

        # 设置最大循环次数，防止死循环
        max_turns = 5
        current_turn = 0

        while current_turn < max_turns:
            current_turn += 1

            # 2. 调用 LLM (可能会返回 tool_calls)
            try:
                # 构造请求参数
                request_kwargs = {
                    "model": self.model,
                    "messages": self.history,
                    "tools": tools,
                    "tool_choice": "auto",
                    "stream": False
                }
                
                completion = self.client.chat.completions.create(**request_kwargs)
            except Exception as e:
                yield f"AI 接口调用失败: {str(e)}"
                return

            response_message = completion.choices[0].message

            # 3. 检查是否有工具调用请求
            tool_calls = response_message.tool_calls
            
            if tool_calls:
                # 将 AI 的 Tool Call 意图加入历史
                self.history.append(response_message)
                
                # 执行所有请求的工具
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    log.info(f"Agent is calling tool: {function_name} with args: {function_args}")
                    
                    # 查找并执行函数
                    func = AgentTool.get_tool_func(function_name)
                    image_data = None

                    if func:
                        try:
                            function_response = func(**function_args)
                            
                            # check for image response special format
                            try:
                                if isinstance(function_response, str):
                                    resp_json = json.loads(function_response)
                                    if isinstance(resp_json, dict) and "_image_data" in resp_json:
                                        image_data = resp_json["_image_data"]
                                        # remove image data from history to save context
                                        del resp_json["_image_data"]
                                        function_response = json.dumps(resp_json, ensure_ascii=False)
                            except json.JSONDecodeError:
                                pass

                        except Exception as e:
                            function_response = f"Error executing {function_name}: {str(e)}"
                    else:
                        function_response = f"Error: Tool {function_name} not found."

                    # 将工具执行结果加入历史
                    self.history.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(function_response),
                    })
                    
                    # 如果有图片数据，注入一个新的 User 消息
                    if image_data:
                        self.history.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Here is the image requested."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                            ]
                        })
                
                # 关键点：这里 continue，让 while 循环继续，再次调用 LLM
                # LLM 会看到 Tool 的结果，决定是继续调用工具，还是输出最终回答
                continue

            else:
                # 没有工具调用，说明是最终回答，或者是单纯的对话
                content = response_message.content
                self.history.append({"role": "assistant", "content": content})
                yield content
                # 结束循环
                return

# ==============================================================================
# 具体工具实现 (Tools Implementation)
# ==============================================================================

from . import calibre_db, db, config
from sqlalchemy import or_

def format_books(books):
    results = []
    for book in books:
        # 获取作者名
        authors = [a.name for a in book.authors]
        tags = [t.name for t in book.tags]
        
        # 获取评分（兼容性处理）
        # Books.ratings 是一个关系属性，返回 Ratings 对象列表
        # 通常一本书只有一个评分，取第一个
        rating = 0
        if book.ratings and len(book.ratings) > 0:
            # 确保取到的是数值
            rating = book.ratings[0].rating
            
        results.append({
            "id": book.id,
            "title": book.title,
            "authors": authors,
            "tags": tags,
            "rating": rating,
            "year": book.pubdate.year if book.pubdate else "Unknown",
            "description": book.comments[0].text[:200] + "..." if book.comments else "无简介"
        })
    return json.dumps(results, ensure_ascii=False)

@AgentTool(
    name="get_book_cover",
    description="获取书籍的封面图片。当需要向用户介绍书籍外观或封面细节时调用。",
    parameters={
        "type": "object",
        "properties": {
            "book_id": {
                "type": "integer",
                "description": "书籍的 ID"
            }
        },
        "required": ["book_id"]
    }
)
def get_book_cover(book_id):
    session = calibre_db.session
    book = session.query(db.Books).filter(db.Books.id == book_id).first()
    
    if not book:
         return json.dumps({"status": "error", "message": "Book not found"})
    
    if not book.has_cover:
         return json.dumps({"status": "error", "message": "Book has no cover"})

    try:
        library_path = config.config_calibre_dir
        book_path = book.path
        # Use os.path.join to handle separators correctly
        cover_path = os.path.join(library_path, book_path, "cover.jpg")
        
        if os.path.exists(cover_path):
            with open(cover_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            return json.dumps({
                "status": "success",
                "message": "Cover loaded successfully",
                "_image_data": image_data  # special key for chat loop interception
            })
        else:
            return json.dumps({"status": "error", "message": "Cover file missing from disk"})
            
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

@AgentTool(
    name="search_books",
    description="根据关键词搜索书籍。可以搜索书名、作者或标签。如果用户没有指定搜索字段，默认全搜。",
    parameters={
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "搜索关键词，例如 '科幻', '三体', 'J.K. Rowling'"
            },
            "field": {
                "type": "string",
                "enum": ["title", "author", "tag", "all"],
                "description": "搜索字段：title(书名), author(作者), tag(标签), all(全部)。默认为 all"
            }
        },
        "required": ["keyword"]
    }
)
def search_books(keyword, field="all"):
    session = calibre_db.session
    query = session.query(db.Books)
    limit = 5

    if field == "title":
        query = query.filter(db.Books.title.ilike(f"%{keyword}%"))
    elif field == "author":
        query = query.join(db.books_authors_link).join(db.Authors).filter(db.Authors.name.ilike(f"%{keyword}%"))
    elif field == "tag":
        query = query.join(db.books_tags_link).join(db.Tags).filter(db.Tags.name.ilike(f"%{keyword}%"))
    elif field == "all":
        # 简化处理：只搜标题，如果需要全搜比较复杂
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
        "required": []
    }
)
def get_library_stats():
    session = calibre_db.session
    book_count = session.query(db.Books).count()
    author_count = session.query(db.Authors).count()
    
    return json.dumps({
        "total_books": book_count,
        "total_authors": author_count
    }, ensure_ascii=False)

@AgentTool(
    name="get_recent_books",
    description="获取最近入库的新书。",
    parameters={
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "返回数量，默认为 5"
            }
        },
        "required": []
    }
)
def get_recent_books(limit=5):
    session = calibre_db.session
    # timestamp 通常是入库时间，pubdate 是出版日期。新书推荐通常用 timestamp
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
                "description": "返回数量，默认为 5"
            }
        },
        "required": []
    }
)
def get_random_books(limit=5):
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
                "description": "返回数量，默认为 5"
            }
        },
        "required": []
    }
)
def get_books_by_rating(limit=5):
    session = calibre_db.session
    
    # 联表查询：Books -> books_ratings_link -> Ratings
    # 并按 Ratings.rating 排序
    books = session.query(db.Books)\
        .join(db.books_ratings_link)\
        .join(db.Ratings)\
        .order_by(db.Ratings.rating.desc(), db.Books.timestamp.desc())\
        .limit(limit).all()

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
                "description": "书籍的 ID"
            }
        },
        "required": ["book_id"]
    }
)
def get_book_chapters(book_id):
    """获取书籍的章节列表"""
    from .book_content_extractor import BookContentExtractor
    
    session = calibre_db.session
    book = session.query(db.Books).filter(db.Books.id == book_id).first()
    
    if not book:
        return json.dumps({"status": "error", "message": "Book not found"})
    
    # 找到可阅读的格式
    readable_formats = ['epub', 'kepub', 'txt']
    book_format = None
    book_data = None
    
    for data in book.data:
        if data.format.lower() in readable_formats:
            book_format = data.format.lower()
            book_data = data
            break
    
    if not book_format:
        return json.dumps({
            "status": "error", 
            "message": f"Book has no readable format. Available: {[d.format for d in book.data]}"
        })
    
    # 获取文件路径
    file_path = os.path.normpath(
        os.path.join(config.config_calibre_dir, book.path, 
                    book_data.name + "." + book_format)
    )
    
    if not os.path.exists(file_path):
        return json.dumps({"status": "error", "message": "Book file not found on disk"})
    
    try:
        # 提取内容
        content_data = BookContentExtractor.extract(file_path, book_format)
        
        # 只返回章节列表，不包含内容（节省 token）
        chapters_info = [
            {
                "index": ch["index"],
                "title": ch["title"],
                "word_count": ch["word_count"]
            }
            for ch in content_data["chapters"]
        ]
        
        return json.dumps({
            "status": "success",
            "book_title": content_data["title"],
            "total_chapters": content_data["total_chapters"],
            "chapters": chapters_info
        }, ensure_ascii=False)
        
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
                "description": "书籍的 ID"
            },
            "chapter_index": {
                "type": "integer",
                "description": "章节索引（从 0 开始）。如果不指定，默认读取第一章"
            },
            "max_words": {
                "type": "integer",
                "description": "最多返回多少字，避免内容过长。默认 3000 字"
            }
        },
        "required": ["book_id"]
    }
)
def read_book_chapter(book_id, chapter_index=0, max_words=3000):
    """读取书籍的指定章节内容"""
    from .book_content_extractor import BookContentExtractor
    
    session = calibre_db.session
    book = session.query(db.Books).filter(db.Books.id == book_id).first()
    
    if not book:
        return json.dumps({"status": "error", "message": "Book not found"})
    
    # 找到可阅读的格式
    readable_formats = ['epub', 'kepub', 'txt']
    book_format = None
    book_data = None
    
    for data in book.data:
        if data.format.lower() in readable_formats:
            book_format = data.format.lower()
            book_data = data
            break
    
    if not book_format:
        return json.dumps({
            "status": "error", 
            "message": f"Book has no readable format. Available: {[d.format for d in book.data]}"
        })
    
    # 获取文件路径
    file_path = os.path.normpath(
        os.path.join(config.config_calibre_dir, book.path, 
                    book_data.name + "." + book_format)
    )
    
    if not os.path.exists(file_path):
        return json.dumps({"status": "error", "message": "Book file not found on disk"})
    
    try:
        # 提取内容
        content_data = BookContentExtractor.extract(file_path, book_format)
        
        if chapter_index >= len(content_data["chapters"]):
            return json.dumps({
                "status": "error", 
                "message": f"Chapter index {chapter_index} out of range. Total chapters: {len(content_data['chapters'])}"
            })
        
        chapter = content_data["chapters"][chapter_index]
        content = chapter["content"]
        
        # 截断过长的内容
        if len(content) > max_words:
            content = content[:max_words] + f"\n\n[... 内容过长，已截断。完整章节共 {chapter['word_count']} 字]"
        
        return json.dumps({
            "status": "success",
            "book_title": content_data["title"],
            "chapter_index": chapter["index"],
            "chapter_title": chapter["title"],
            "content": content,
            "total_word_count": chapter["word_count"],
            "returned_words": min(len(content), max_words)
        }, ensure_ascii=False)
        
    except Exception as e:
        log.error(f"Failed to read book chapter: {e}")
        return json.dumps({"status": "error", "message": str(e)})
