"""The indexing progress bars' final flush."""
from chonks.index.progress import _Bars


class _FakeBar:
    def __init__(self):
        self.n, self.total, self.bar_format = 0, None, None

    def update(self, k):
        self.n += k

    def refresh(self):
        pass

    def set_description(self, d):
        pass

    def set_postfix_str(self, s, refresh=True):
        pass


def test_finish_before_the_flip_keeps_the_scanned_count():
    files, queued, embedded = _FakeBar(), _FakeBar(), _FakeBar()
    bars = _Bars(files, queued, embedded)
    bars.tick(scanned=5, parsed=3, queued=0, embedded=0, cur_file=None, scan_done=False, to_index=None)
    bars.finish(parsed=5, queued=0, embedded=0)
    assert files.n == 5


def test_finish_after_the_flip_adds_the_remaining_parsed_files():
    files, queued, embedded = _FakeBar(), _FakeBar(), _FakeBar()
    bars = _Bars(files, queued, embedded)
    bars.tick(scanned=5, parsed=3, queued=0, embedded=0, cur_file=None, scan_done=True, to_index=5)
    bars.finish(parsed=5, queued=0, embedded=0)
    assert files.n == 5


def test_pass_log_reports_progress_at_most_every_interval(monkeypatch, caplog):
    import logging
    import chonks.index.progress as progress

    ticks = [0.0, 10.0, 31.0, 40.0, 62.0]
    monkeypatch.setattr(progress.time, "monotonic", lambda: ticks.pop(0) if len(ticks) > 1 else ticks[0])
    log = progress.PassLog(logging.getLogger("passlog-test"), "chunk_refs", every=30.0)
    with caplog.at_level(logging.INFO, logger="passlog-test"):
        for i in range(4):
            log.progress("name scan", i, 4)
    assert [r.getMessage() for r in caplog.records] == [
        "chunk_refs: name scan: 1 of 4 (25%), 31s into the pass",
        "chunk_refs: name scan: 3 of 4 (75%), 62s into the pass",
    ]
