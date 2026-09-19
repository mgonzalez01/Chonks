"""Old import path for chonks.ops.init."""

import sys

from chonks.embed.client import ping_embedder
from chonks.ops.init import (
    JUNK_DIR_NAMES,
    ask,
    ask_yes_no,
    build_config,
    format_size,
    gpu_install_hint,
    render_mcp_add,
    render_mcp_json,
    run_wizard,
    scan_junk_candidates,
    write_config,
)

if __name__ == "__main__":
    sys.exit(run_wizard())
