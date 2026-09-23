"""Progress bars of an indexing run."""

from contextlib import contextmanager

from tqdm import tqdm

_FMT_TOTAL    = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
_FMT_NOTOTAL  = "{desc}: {n_fmt} {unit} [{elapsed}, {rate_fmt}]"


class _Bars:
    def __init__(self, pb_files, pb_queued, pb_embedded):
        self.pb_files = pb_files
        self.pb_queued = pb_queued
        self.pb_embedded = pb_embedded
        self.last_parsed = self.last_embedded = 0
        self.files_total_set = False

    def tick(self, scanned, parsed, queued, embedded, cur_file, scan_done, to_index):
        # Two-phase bar: shows discovered-file count pre-scan, then flips to
        # parsed/total once the scan completes, avoiding a misleadingly
        # small bar while the queue is already far ahead.
        if not self.files_total_set:
            self.pb_files.n = scanned
            if scan_done:
                self.pb_files.total       = to_index
                self.pb_files.bar_format  = _FMT_TOTAL
                self.pb_files.set_description("Parsed  ")
                self.pb_files.n           = parsed
                self.last_parsed          = parsed
                self.files_total_set      = True
            self.pb_files.refresh()
        else:
            self.pb_files.update(parsed - self.last_parsed)
            self.last_parsed = parsed

        self.pb_queued.n = queued - embedded  # current backlog, tends to 0
        self.pb_queued.refresh()
        self.pb_embedded.update(embedded - self.last_embedded)

        if cur_file:
            self.pb_files.set_postfix_str(cur_file, refresh=False)

        self.last_embedded = embedded

    def finish(self, parsed, queued, embedded):
        # Before the flip the files bar counts scanned files, not parsed ones.
        if self.files_total_set:
            self.pb_files.update(parsed - self.last_parsed)
        self.pb_queued.n = queued - embedded
        self.pb_queued.refresh()
        self.pb_embedded.update(embedded - self.last_embedded)


@contextmanager
def tqdm_reporter():
    """Files scanned then parsed, chunk backlog, chunks embedded."""
    with \
        tqdm(total=None, desc="Scanning", unit="file",  position=0,
             colour="green",  bar_format=_FMT_NOTOTAL, dynamic_ncols=True) as pb_files, \
        tqdm(total=None,        desc="Queued  ", unit="chunk", position=1,
             colour="cyan",   bar_format=_FMT_NOTOTAL, dynamic_ncols=True) as pb_queued, \
        tqdm(total=None,        desc="Embedded", unit="chunk", position=2,
             colour="yellow", bar_format=_FMT_NOTOTAL, dynamic_ncols=True) as pb_embedded:
        yield _Bars(pb_files, pb_queued, pb_embedded)
