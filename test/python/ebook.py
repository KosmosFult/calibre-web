"""Manual smoke test for a real Calibre book and configured model backend.

Example:
    OPENAI_API_KEY=... .venv/bin/python test/python/ebook.py 13 \
        "恋塚所说的姐姐大人究竟是谁？"
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cps import app, create_app  # noqa: E402
from cps.ai.config import load_model_settings, load_novel_settings  # noqa: E402
from cps.ai.novel import service  # noqa: E402
from cps.ai.novel.ingestion import resolve_book_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("book_id", type=int)
    parser.add_argument("question")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    create_app()
    with app.app_context():
        status = service.get_index_status(args.book_id)
        if args.rebuild or not status.get("indexed"):
            models = load_model_settings(require_api_key=True)
            novel = load_novel_settings()
            source = resolve_book_path(args.book_id)
            if not source:
                raise RuntimeError("Book has no readable EPUB, KEPUB, or TXT source")
            run_id = uuid.uuid4().hex
            repository = service.get_repository()
            repository.create_run(
                run_id=run_id, book_id=args.book_id, status="RUNNING", phase="START",
                message="manual smoke test", llm_model=models.extraction_model,
                embedding_model=models.embedding_model, config=asdict(novel),
            )
            ok, result = service._perform_index_build(  # pylint: disable=protected-access
                args.book_id, run_id=run_id, force=args.rebuild, source_path=source,
                progress_callback=lambda phase, message, progress: print(
                    "[{:.0%}] {}: {}".format(progress, phase, message), flush=True
                ),
            )
            if not ok:
                raise RuntimeError(result.get("message") or "index build failed")
        answer = service.ask_book(args.book_id, args.question)
        print(json.dumps(answer, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
