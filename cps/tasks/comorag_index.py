# -*- coding: utf-8 -*-

from flask_babel import lazy_gettext as N_

from cps import app, logger
from cps.services.worker import CalibreTask
from cps.comorag import service as comorag_service


class TaskComoRAGIndex(CalibreTask):
    def __init__(
        self,
        book_id: int,
        run_id: str,
        force: bool = False,
        chapter_indices=None,
        task_message=N_("Building ComoRAG index"),
    ):
        super(TaskComoRAGIndex, self).__init__(task_message)
        self.log = logger.create()
        self.book_id = int(book_id)
        self.run_id = run_id
        self.force = force
        self.chapter_indices = chapter_indices
        self.message = f"Book #{self.book_id} queued"
        self.progress = 0.02

    def _progress_callback(self, phase: str, message: str, progress: float) -> None:
        self.message = f"Book #{self.book_id} [{phase}] {message}"
        self.progress = progress

    def run(self, worker_thread):
        with app.app_context():
            ok, payload = comorag_service._perform_index_build(  # pylint: disable=protected-access
                self.book_id,
                run_id=self.run_id,
                force=self.force,
                progress_callback=self._progress_callback,
                chapter_indices=self.chapter_indices,
            )
        if ok:
            self._handleSuccess()
            self.message = f"Book #{self.book_id} index ready"
        else:
            self._handleError(payload.get("message") or "ComoRAG index build failed")
            self.message = f"Book #{self.book_id} index failed"

    @property
    def name(self):
        return f"ComoRAG Index (Book #{self.book_id})"

    @property
    def is_cancellable(self):
        return False
