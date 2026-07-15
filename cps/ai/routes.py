# -*- coding: utf-8 -*-
"""Flask routes for conversations and narrative index management."""

from __future__ import annotations

import datetime
import json
import queue
import threading
import logging

from flask import Blueprint, Response, current_app, jsonify, render_template, request, stream_with_context

from ..cw_login import current_user, login_required
from . import db as ai_db
from .agent import CalibreAgent
from .config import load_model_settings, load_system_prompt
from .novel import service as novel_service


ai = Blueprint("ai", __name__, url_prefix="/ai")
STREAM_HEARTBEAT_INTERVAL = 10
log = logging.getLogger("calibre-web.ai")
ai_db.init_db()


def get_agent():
    try:
        settings = load_model_settings(require_api_key=True)
    except ValueError:
        return None
    return CalibreAgent(settings=settings, system_prompt=load_system_prompt())


@ai.route("/")
@login_required
def index():
    return render_template("ai_chat.html")


@ai.route("/sessions", methods=["GET"])
@login_required
def list_sessions():
    db_sess = ai_db.get_session()
    try:
        sessions = (
            db_sess.query(ai_db.AIChat)
            .filter_by(user_id=int(current_user.id))
            .order_by(ai_db.AIChat.updated_at.desc())
            .all()
        )
        return jsonify([item.to_dict() for item in sessions])
    finally:
        db_sess.close()


@ai.route("/sessions", methods=["POST"])
@login_required
def create_session():
    db_sess = ai_db.get_session()
    try:
        chat = ai_db.AIChat(user_id=int(current_user.id))
        db_sess.add(chat)
        db_sess.commit()
        return jsonify(chat.to_dict())
    except Exception as exc:  # pylint: disable=broad-except
        db_sess.rollback()
        return jsonify({"error": str(exc)}), 500
    finally:
        db_sess.close()


@ai.route("/sessions/<int:session_id>", methods=["DELETE"])
@login_required
def delete_session(session_id):
    db_sess = ai_db.get_session()
    try:
        chat = db_sess.query(ai_db.AIChat).filter_by(
            id=session_id, user_id=int(current_user.id)
        ).first()
        if not chat:
            return jsonify({"error": "Session not found"}), 404
        db_sess.delete(chat)
        db_sess.commit()
        return jsonify({"status": "success"})
    finally:
        db_sess.close()


@ai.route("/sessions/<int:session_id>/messages", methods=["GET"])
@login_required
def get_messages(session_id):
    db_sess = ai_db.get_session()
    try:
        chat = db_sess.query(ai_db.AIChat).filter_by(
            id=session_id, user_id=int(current_user.id)
        ).first()
        if not chat:
            return jsonify({"error": "Session not found"}), 404
        rows = (
            db_sess.query(ai_db.AIMessage)
            .filter_by(chat_id=session_id, visible=True)
            .order_by(ai_db.AIMessage.id.asc())
            .all()
        )
        return jsonify([row.to_public_dict() for row in rows])
    finally:
        db_sess.close()


@ai.route("/books/<int:book_id>/novel-index/status", methods=["GET"])
@ai.route("/books/<int:book_id>/comorag/index-status", methods=["GET"], endpoint="comorag_index_status")
@login_required
def novel_index_status(book_id):
    try:
        return jsonify(novel_service.get_index_status(int(book_id)))
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"status": "error", "message": str(exc)}), 500


@ai.route("/books/<int:book_id>/novel-index/build", methods=["POST"])
@ai.route("/books/<int:book_id>/comorag/build-index", methods=["POST"], endpoint="comorag_build_index")
@login_required
def novel_build_index(book_id):
    try:
        payload = request.get_json(silent=True) or {}
        force_raw = payload.get("force", request.form.get("force") if request.form else None)
        force = str(force_raw).lower() in {"1", "true", "yes", "on"}
        ok, result = novel_service.trigger_index_build(int(book_id), force=force)
        return jsonify(result), (202 if ok else (409 if result.get("status") == "running" else 500))
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"status": "error", "message": str(exc)}), 500


@ai.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json(silent=True) or {}
    user_message = str(data.get("message") or "").strip()
    session_id = data.get("session_id")
    if not session_id or not user_message:
        return jsonify({"error": "session_id and message are required"}), 400

    agent = get_agent()
    if agent is None:
        def missing_key():
            yield json.dumps({"text": "系统未配置 OpenAI-compatible API Key。"}, ensure_ascii=False) + "\n"
        return Response(stream_with_context(missing_key()), content_type="application/x-ndjson")

    user_id = int(current_user.id)
    flask_app = current_app._get_current_object()

    def generate():
        db_sess = ai_db.get_session()
        try:
            chat_session = db_sess.query(ai_db.AIChat).filter_by(
                id=int(session_id), user_id=user_id
            ).first()
            if not chat_session:
                yield json.dumps({"text": "Error: Session not found"}) + "\n"
                return

            history = ai_db.history_for_chat(db_sess, int(session_id))
            agent.extend_history(history)
            user_record = agent.build_user_message(user_message)
            agent.append_message(user_record)
            db_sess.add(ai_db.create_message(int(session_id), user_record))
            chat_session.updated_at = datetime.datetime.utcnow()
            if not history:
                chat_session.title = user_message[:30] + ("..." if len(user_message) > 30 else "")
            db_sess.commit()
            checkpoint = len(agent.history)

            stream_queue = queue.Queue()
            done = threading.Event()

            def run_agent():
                try:
                    with flask_app.app_context():
                        for chunk in agent.chat():
                            stream_queue.put(("chunk", chunk))
                except Exception as exc:  # pylint: disable=broad-except
                    log.exception("AI agent failed")
                    stream_queue.put(("error", str(exc)))
                finally:
                    done.set()

            threading.Thread(target=run_agent, daemon=True).start()
            while not done.is_set() or not stream_queue.empty():
                try:
                    item_type, value = stream_queue.get(timeout=STREAM_HEARTBEAT_INTERVAL)
                except queue.Empty:
                    yield json.dumps({"heartbeat": True}) + "\n"
                    continue
                if item_type == "error":
                    yield json.dumps({"text": "Error from AI: {}".format(value)}, ensure_ascii=False) + "\n"
                    return
                yield json.dumps({"text": value}, ensure_ascii=False) + "\n"

            for message in agent.history[checkpoint:]:
                db_sess.add(ai_db.create_message(int(session_id), message))
            chat_session.updated_at = datetime.datetime.utcnow()
            db_sess.commit()
            yield json.dumps({"token_stats": agent.last_usage}) + "\n"
        except Exception as exc:  # pylint: disable=broad-except
            db_sess.rollback()
            yield json.dumps({"text": "System Error: {}".format(exc)}, ensure_ascii=False) + "\n"
        finally:
            db_sess.close()

    response = Response(stream_with_context(generate()), content_type="application/x-ndjson")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response
