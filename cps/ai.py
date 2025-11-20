# -*- coding: utf-8 -*-
from flask import Blueprint, render_template, request, Response, stream_with_context, session
import json
import os
from .agent import CalibreAgent

ai = Blueprint('ai', __name__, url_prefix='/ai')

# TODO: 应该从 config.yaml 中读取
# 为了演示，暂时请您设置环境变量，或者在这里填入您的 Key
# os.environ["OPENAI_API_KEY"] = "sk-..."
# os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"

def get_agent():
    """
    工厂方法：获取或创建当前会话的 Agent 实例
    注意：简单的 Demo 可以把 history 存在 flask session 里，
    但 CalibreAgent 对象本身不好序列化进 session。
    
    这里我们做一个简化：
    每次请求都新建 Agent，但是把前端传来的 history 喂给它（如果前端维护 history），
    或者利用 Flask session 存 history list。
    """
    api_key = os.environ.get("OPENAI_API_KEY", "AIzaSyBaX5Xw-DwsotnMFq_9Q52GT8dYRLQe9iA")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    
    if not api_key:
        return None

    agent = CalibreAgent(api_key=api_key, base_url=base_url)
    
    # 从 Flask Session 恢复历史
    if 'ai_history' in session:
        agent.history = session['ai_history']
    
    return agent

@ai.route('/')
def index():
    # 清空历史，开始新对话
    session.pop('ai_history', None)
    return render_template('ai_chat.html')

@ai.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    agent = get_agent()
    
    if not agent:
        def error_gen():
            yield json.dumps({'text': "系统未配置 OpenAI API Key。请在环境变量中设置 OPENAI_API_KEY。"}) + '\n'
        return Response(stream_with_context(error_gen()), content_type='application/x-ndjson')

    def generate():
        # 调用 Agent 进行对话
        try:
            for chunk in agent.chat(user_message):
                yield json.dumps({'text': chunk}) + '\n'
        except Exception as e:
             yield json.dumps({'text': f"发生错误: {str(e)}"}) + '\n'
        
        # 保存历史回 Session
        # 注意：这里有并发写入风险，且 session 大小有限制 (Cookie based session 只有 4KB)
        # 生产环境应该存数据库或 Redis。这里是 Demo 暂且存内存/Cookie。
        # 由于 Cookie 限制，历史太长会崩。
        # 更好的做法是：Calibre-Web 使用了 server-side session (Flask-Session)? 
        # 看代码 web.py 似乎是基于 Cookie 的 ("session.permanent = True")
        # 所以我们尽量只存最近几轮，或者不要在 session 里存太大的 history。
        session['ai_history'] = agent.history[-10:] # 只保留最近 10 条

    return Response(stream_with_context(generate()), content_type='application/x-ndjson')
