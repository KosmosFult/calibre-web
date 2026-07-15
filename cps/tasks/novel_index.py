# -*- coding: utf-8 -*-

from flask_babel import lazy_gettext as N_

from cps import app, logger
from cps.ai.novel import service as novel_service
from cps.services.worker import CalibreTask


class TaskNovelIndex(CalibreTask):
    def __init__(
        self,
        book_id: int,
        run_id: str,
        force: bool = False,
        source_path: str = "",
        task_message=N_("Building narrative intelligence index"),
    ):
        super(TaskNovelIndex, self).__init__(task_message)
        self.log = logger.create()
        self.book_id = int(book_id)
        self.run_id = run_id
        self.force = force
        self.source_path = source_path
        self.message = "Book #{} queued".format(self.book_id)
        self.progress = 0.01

    def _progress_callback(self, phase: str, message: str, progress: float) -> None:
        self.message = "Book #{} [{}] {}".format(self.book_id, phase, message)
        self.progress = progress

    def run(self, worker_thread):
        with app.app_context():
            ok, payload = novel_service._perform_index_build(  # pylint: disable=protected-access
                self.book_id,
                run_id=self.run_id,
                force=self.force,
                source_path=self.source_path,
                progress_callback=self._progress_callback,
            )
        if ok:
            self._handleSuccess()
            self.message = "Book #{} narrative index ready".format(self.book_id)
        else:
            self._handleError(payload.get("message") or "Narrative index build failed")
            self.message = "Book #{} narrative index failed".format(self.book_id)

    @property
    def name(self):
        return "Narrative Index (Book #{})".format(self.book_id)

    @property
    def is_cancellable(self):
        return False
