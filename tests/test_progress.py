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
