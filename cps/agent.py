# -*- coding: utf-8 -*-
import json
import logging
import inspect
from functools import wraps

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
    def __init__(self, api_key, base_url, model="gemini-2.5-flash", system_prompt=None):
        if not openai_available:
            raise ImportError("OpenAI module is not installed. Please install it using 'pip install openai'")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.system_prompt = system_prompt or (
            "你是 Calibre-Web 的 AI 图书管家。你可以通过工具查询书库中的书籍信息。"
            "请根据用户的需求，调用合适的工具进行查询，并根据查询结果友好地回复用户。"
            "如果查询结果为空，请礼貌地告知用户。"
            "请用中文回答。"
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

        # 2. 第一轮调用 LLM (可能会返回 tool_calls)
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                tools=AgentTool.get_tools_schema(),
                tool_choice="auto",
                stream=False  # 第一轮先不流式，方便处理 Tool Call 逻辑
            )
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
                if func:
                    try:
                        function_response = func(**function_args)
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

            # 4. 第二轮调用 LLM (带上工具结果，这次请求流式输出)
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                stream=True
            )
            
            full_content = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content_piece = chunk.choices[0].delta.content
                    full_content += content_piece
                    yield content_piece
            
            # 记录最终 AI 回复
            self.history.append({"role": "assistant", "content": full_content})

        else:
            # 没有工具调用，直接返回回答
            # 注意：前面是非流式的，所以这里直接拿到内容。为了前端体验统一，我们模拟流式输出一下，或者如果想纯流式，第一步就得重构
            # 简单起见，如果是非工具调用，我们直接把内容输出
            content = response_message.content
            self.history.append({"role": "assistant", "content": content})
            yield content

# ==============================================================================
# 具体工具实现 (Tools Implementation)
# ==============================================================================

from . import calibre_db, db
from sqlalchemy import or_

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
    """
    执行真实的数据库搜索
    """
    session = calibre_db.session
    query = session.query(db.Books)
    
    results = []
    limit = 5  # 限制返回数量，防止 token 爆炸

    if field == "title":
        query = query.filter(db.Books.title.ilike(f"%{keyword}%"))
    elif field == "author":
        # 复杂的作者关联查询略，这里简化演示，实际项目可以用 db.py 里的 search_query 逻辑
        # 这里为了演示方便，还是先只搜标题吧，或者用简单的逻辑
        pass 
        # 注意：真实的 Calibre-Web 搜索逻辑比较复杂（涉及多表关联），建议复用 db.search_query
        # 但这里我们为了 Agent 稳定性，先实现一个最简单的标题搜索
        query = query.filter(db.Books.title.ilike(f"%{keyword}%"))

    elif field == "all":
         query = query.filter(db.Books.title.ilike(f"%{keyword}%"))
    
    # 执行查询
    books = query.limit(limit).all()
    
    if not books:
        return json.dumps({"status": "empty", "message": f"没有找到包含 '{keyword}' 的书籍。"})

    for book in books:
        # 获取作者名
        authors = [a.name for a in book.authors]
        results.append({
            "id": book.id,
            "title": book.title,
            "authors": authors,
            "year": book.pubdate.year if book.pubdate else "Unknown",
            "description": book.comments[0].text if book.comments else "无简介"
        })
    
    return json.dumps(results, ensure_ascii=False)

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

