# -*- coding: utf-8 -*-
from flask import Blueprint, render_template, request, Response, stream_with_context, jsonify
from .cw_login import login_required, current_user
import json
import os
from .agent import CalibreAgent
from . import ai_db

ai = Blueprint('ai', __name__, url_prefix='/ai')

# 初始化 AI 数据库
ai_db.init_db()

def get_agent():
    """
    获取 Agent 实例（不带历史，历史由调用方注入）
    """
    api_key = os.environ.get("OPENAI_API_KEY", "AIzaSyBaX5Xw-DwsotnMFq_9Q52GT8dYRLQe9iA")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    
    if not api_key:
        return None

    return CalibreAgent(api_key=api_key, base_url=base_url)

@ai.route('/')
@login_required
def index():
    return render_template('ai_chat.html')

@ai.route('/sessions', methods=['GET'])
@login_required
def list_sessions():
    """获取当前用户的会话列表"""
    db_sess = ai_db.get_session()
    try:
        sessions = db_sess.query(ai_db.AIChatSession)\
            .filter_by(user_id=int(current_user.id))\
            .order_by(ai_db.AIChatSession.updated_at.desc())\
            .all()
        return jsonify([s.to_dict() for s in sessions])
    finally:
        db_sess.close()

@ai.route('/sessions', methods=['POST'])
@login_required
def create_session():
    """创建一个新会话"""
    db_sess = ai_db.get_session()
    try:
        new_session = ai_db.AIChatSession(user_id=int(current_user.id))
        db_sess.add(new_session)
        db_sess.commit()
        return jsonify(new_session.to_dict())
    except Exception as e:
        db_sess.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db_sess.close()

@ai.route('/sessions/<int:session_id>', methods=['DELETE'])
@login_required
def delete_session(session_id):
    """删除会话"""
    db_sess = ai_db.get_session()
    try:
        chat_session = db_sess.query(ai_db.AIChatSession)\
            .filter_by(id=session_id, user_id=int(current_user.id))\
            .first()
        if chat_session:
            db_sess.delete(chat_session)
            db_sess.commit()
            return jsonify({"status": "success"})
        else:
            return jsonify({"error": "Session not found"}), 404
    finally:
        db_sess.close()

@ai.route('/sessions/<int:session_id>/messages', methods=['GET'])
@login_required
def get_messages(session_id):
    """获取指定会话的消息历史"""
    db_sess = ai_db.get_session()
    try:
        # 验证会话归属
        chat_session = db_sess.query(ai_db.AIChatSession)\
            .filter_by(id=session_id, user_id=int(current_user.id))\
            .first()
        
        if not chat_session:
            return jsonify({"error": "Session not found"}), 404

        messages = db_sess.query(ai_db.AIChatMessage)\
            .filter_by(session_id=session_id)\
            .order_by(ai_db.AIChatMessage.created_at.asc())\
            .all()
        
        # 前端只需要展示 user 和 assistant 的文本内容
        # tool 相关的消息可以过滤，或者前端自己处理
        visible_messages = [m.to_dict() for m in messages if m.role in ['user', 'assistant'] and m.content]
        
        return jsonify(visible_messages)
    finally:
        db_sess.close()

@ai.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.json
    user_message = data.get('message', '')
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    agent = get_agent()
    if not agent:
        def error_gen():
            yield json.dumps({'text': "系统未配置 OpenAI API Key。"}) + '\n'
        return Response(stream_with_context(error_gen()), content_type='application/x-ndjson')

    # 准备生成器
    def generate():
        db_sess = ai_db.get_session()
        try:
            # 1. 验证并获取会话
            chat_session = db_sess.query(ai_db.AIChatSession)\
                .filter_by(id=session_id, user_id=int(current_user.id))\
                .first()
            
            if not chat_session:
                yield json.dumps({'text': "Error: Session not found"}) + '\n'
                return

            # 2. 加载历史消息到 Agent
            # 注意：这里必须严格还原 tool_calls 和 tool_call_id，否则会报 400 Invalid Argument
            existing_msgs = db_sess.query(ai_db.AIChatMessage)\
                .filter_by(session_id=session_id)\
                .order_by(ai_db.AIChatMessage.created_at.asc())\
                .all()
            
            # 取最近 20 条，但要注意不要切断了 (Assistant->Tool) 的配对
            # 简单起见，我们取最近的 20 条消息，但如果第一条是 tool，那可能会报错
            # 更稳妥的是取所有消息（如果不多）或者智能截断。这里先简单取最后 20 条。
            subset_msgs = existing_msgs[-30:] 
            
            history_context = []
            for m in subset_msgs:
                msg_obj = {"role": m.role, "content": m.content}
                
                # 还原 tool_calls
                if m.tool_calls:
                    try:
                        msg_obj["tool_calls"] = json.loads(m.tool_calls)
                    except:
                        pass # JSON 解析失败忽略
                
                # 还原 tool_call_id
                if m.tool_call_id:
                    msg_obj["tool_call_id"] = m.tool_call_id
                
                history_context.append(msg_obj)
            
            agent.history.extend(history_context)

            # 3. 保存用户的消息到 DB
            user_msg_db = ai_db.AIChatMessage(
                session_id=session_id, 
                role="user", 
                content=user_message
            )
            db_sess.add(user_msg_db)
            
            # 更新会话时间
            chat_session.updated_at = ai_db.datetime.datetime.utcnow()
            
            # 如果是第一条消息，尝试自动重命名会话
            if len(existing_msgs) == 0:
                chat_session.title = user_message[:30] + "..." if len(user_message) > 30 else user_message

            db_sess.commit()

            # 4. 流式调用 Agent
            # 计算注入历史后的长度，用于后续识别新增消息
            initial_history_len = len(agent.history)
            
            try:
                # user_message 已经在上面手动存库了，agent.chat 内部也会 append user message
                # 但 agent.chat 内部的 append 只是为了 LLM 请求，
                # 我们的 agent.history 其实已经包含了所有上下文。
                # 这里的 user_message 是为了传给 agent.chat 触发新一轮对话
                for chunk in agent.chat(user_message):
                    yield json.dumps({'text': chunk}) + '\n'
            except Exception as e:
                error_msg = f"Error from AI: {str(e)}"
                yield json.dumps({'text': error_msg}) + '\n'
            
            # 5. 保存新增的消息（包括 Tool Calls 和最终 Response）到 DB
            # agent.history 在 chat() 过程中会被追加：User -> [Assistant(ToolCall) -> Tool ->]... -> Assistant(Final)
            # 我们需要把所有新增的都存进去
            
            # 注意：agent.chat 第一步就是 append user message，所以我们要跳过第一条（因为上面第3步已经存了）
            new_messages = agent.history[initial_history_len:]
            
            # 此时 new_messages[0] 应该是 User Message（重复了），跳过它
            if new_messages and new_messages[0].get('role') == 'user':
                new_messages = new_messages[1:]

            for msg in new_messages:
                # msg 可能是 dict 或 ChatCompletionMessage 对象
                if isinstance(msg, dict):
                    role = msg.get('role')
                    content = msg.get('content')
                    tool_calls = msg.get('tool_calls') # dict 里通常没有这个，除非是手动构造的
                    tool_call_id = msg.get('tool_call_id')
                else:
                    # ChatCompletionMessage 对象
                    role = msg.role
                    content = msg.content
                    tool_calls = getattr(msg, 'tool_calls', None)
                    tool_call_id = getattr(msg, 'tool_call_id', None)

                # 序列化 tool_calls
                tool_calls_json = None
                if tool_calls:
                    # tool_calls 是 list[object]，需要转成 dict list
                    tc_list = []
                    for tc in tool_calls:
                        # 兼容 dict 或 object
                        if isinstance(tc, dict):
                            tc_list.append(tc)
                        else:
                            tc_list.append({
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            })
                    tool_calls_json = json.dumps(tc_list)

                ai_msg_db = ai_db.AIChatMessage(
                    session_id=session_id,
                    role=role,
                    content=str(content) if content is not None else None,
                    tool_calls=tool_calls_json,
                    tool_call_id=tool_call_id
                )
                db_sess.add(ai_msg_db)
            
            db_sess.commit()
            
        except Exception as e:
            db_sess.rollback()
            # 记录日志或返回错误
            print(f"Chat Save Error: {e}")
            yield json.dumps({'text': f"System Error: {str(e)}"}) + '\n'
        finally:
            db_sess.close()

    return Response(stream_with_context(generate()), content_type='application/x-ndjson')
