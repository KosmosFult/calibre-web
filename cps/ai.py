# -*- coding: utf-8 -*-
from flask import Blueprint, render_template, request, Response, stream_with_context, jsonify
from .cw_login import login_required, current_user
import datetime
import json
import os
from .agent import CalibreAgent
from . import ai_db
from .config_loader import get_yaml_loader
from .comorag import service as comorag_service

ai = Blueprint('ai', __name__, url_prefix='/ai')

# 初始化 AI 数据库
ai_db.init_db()

def get_agent():
    """
    获取 Agent 实例（不带历史，历史由调用方注入）
    """
    loader = get_yaml_loader()
    yaml_api_key = loader.get("ai", "genai", "api_key")
    yaml_base_url = loader.get("ai", "genai", "base_url")
    yaml_model = loader.get("ai", "genai", "model")
    yaml_system_prompt = loader.get("ai", "genai", "system_prompt")
    yaml_enable_search = loader.get("ai", "genai", "enable_search")

    api_key = (
        yaml_api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_GENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    base_url = (
        yaml_base_url
        or os.environ.get("GENAI_BASE_URL")
        or os.environ.get("GOOGLE_GENAI_BASE_URL")
        or os.environ.get("GOOGLE_API_BASE_URL")
    )
    model = yaml_model or os.environ.get("GENAI_MODEL")
    system_prompt = yaml_system_prompt or os.environ.get("GENAI_SYSTEM_PROMPT")
    if yaml_enable_search is None:
        enable_search = os.environ.get("GENAI_ENABLE_SEARCH", "").lower() in ["1", "true", "yes"]
    else:
        enable_search = bool(yaml_enable_search)

    if not api_key:
        return None

    agent_kwargs = {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "system_prompt": system_prompt,
        "enable_web_search": enable_search,
    }

    # Remove None so we rely on library defaults when optional args are not provided.
    agent_kwargs = {k: v for k, v in agent_kwargs.items() if v is not None}

    return CalibreAgent(**agent_kwargs)

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
        sessions = db_sess.query(ai_db.AIChat)\
            .filter_by(user_id=int(current_user.id))\
            .order_by(ai_db.AIChat.updated_at.desc())\
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
        new_session = ai_db.AIChat(user_id=int(current_user.id))
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
        chat_session = db_sess.query(ai_db.AIChat)\
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
        chat_session = db_sess.query(ai_db.AIChat)\
            .filter_by(id=session_id, user_id=int(current_user.id))\
            .first()
        
        if not chat_session:
            return jsonify({"error": "Session not found"}), 404

        messages = db_sess.query(ai_db.AIMessage)\
            .filter_by(chat_id=session_id)\
            .order_by(ai_db.AIMessage.created_at.asc())\
            .all()

        visible_messages = [
            ai_db.message_to_public_dict(m)
            for m in messages
            if m.role in ["user", "model"] and m.visible_text()
        ]

        return jsonify(visible_messages)
    finally:
        db_sess.close()


@ai.route('/books/<int:book_id>/comorag/index-status', methods=['GET'])
@login_required
def comorag_index_status(book_id):
    try:
        status = comorag_service.get_index_status(book_id=int(book_id))
        return jsonify(status)
    except Exception as e:  # pylint: disable=broad-except
        return jsonify({"status": "error", "message": str(e)}), 500


@ai.route('/books/<int:book_id>/comorag/build-index', methods=['POST'])
@login_required
def comorag_build_index(book_id):
    try:
        force_raw = request.form.get("force") if request.form else None
        force = str(force_raw).lower() in {"1", "true", "yes", "on"}
        ok, payload = comorag_service.trigger_index_build(book_id=int(book_id), force=force)
        if ok:
            code = 202
        else:
            code = 409 if payload.get("status") == "running" else 500
        return jsonify(payload), code
    except Exception as e:  # pylint: disable=broad-except
        return jsonify({"status": "error", "message": str(e)}), 500

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
            yield json.dumps({'text': "系统未配置 Google GenAI API Key。"}) + '\n'
        return Response(stream_with_context(error_gen()), content_type='application/x-ndjson')

    # 准备生成器
    def generate():
        db_sess = ai_db.get_session()
        try:
            # 1. 验证并获取会话
            chat_session = db_sess.query(ai_db.AIChat)\
                .filter_by(id=session_id, user_id=int(current_user.id))\
                .first()
            
            if not chat_session:
                yield json.dumps({'text': "Error: Session not found"}) + '\n'
                return

            # 2. 加载历史为 GenAI 所需格式
            history_messages = ai_db.history_for_chat(db_sess, session_id)
            agent.extend_history(history_messages)

            # 3. 记录用户消息
            user_record = agent.build_user_message(user_message)
            agent.append_message(user_record)
            db_sess.add(
                ai_db.create_message(
                    session_id,
                    user_record["role"],
                    user_record["parts"],
                    user_record.get("thought_signatures"),
                )
            )

            # 更新会话标题和时间
            chat_session.updated_at = datetime.datetime.utcnow()
            if not history_messages:
                chat_session.title = user_message[:30] + "..." if len(user_message) > 30 else user_message

            db_sess.commit()

            history_checkpoint = len(agent.history)

            try:
                for chunk in agent.chat():
                    yield json.dumps({'text': chunk}) + '\n'
            except Exception as e:
                yield json.dumps({'text': f"Error from AI: {str(e)}"}) + '\n'
                return

            new_messages = agent.history[history_checkpoint:]

            for msg in new_messages:
                db_sess.add(
                    ai_db.create_message(
                        session_id,
                        msg["role"],
                        msg["parts"],
                        msg.get("thought_signatures"),
                    )
                )

            chat_session.updated_at = datetime.datetime.utcnow()
            db_sess.commit()
            
        except Exception as e:
            db_sess.rollback()
            # 记录日志或返回错误
            print(f"Chat Save Error: {e}")
            yield json.dumps({'text': f"System Error: {str(e)}"}) + '\n'
        finally:
            db_sess.close()

    return Response(stream_with_context(generate()), content_type='application/x-ndjson')
