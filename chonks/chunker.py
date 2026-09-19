"""Old import path for chonks.ops.index_cmd and for names of chonks.index."""

from chonks.core.paths import (
    _dir_should_prune,
    _dotdir_prefix,
)
from chonks.index.admission import DEFAULT_FALLBACK_EXTENSIONS, _file_hash
from chonks.index.embed_retry import _embed_isolating, _embed_one_isolating
from chonks.index.pipeline import EmbedderDownError, NoProgressError, index_paths, reembed_all
from chonks.index.rows import _chunk_id
from chonks.ops.index_cmd import main

if __name__ == "__main__":
    main()
