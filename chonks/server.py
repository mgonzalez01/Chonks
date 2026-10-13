"""Launcher: `python chonks/server.py` and `python -m chonks.server` both start the server."""

import os
import sys

if __name__ == "__main__":
    # `python chonks/server.py` puts chonks/ first on sys.path, where its
    # subpackages would shadow top-level modules of the same name.
    try:
        if os.path.samefile(sys.path[0], os.path.dirname(os.path.realpath(__file__))):
            del sys.path[0]
    except OSError:
        pass

    from chonks.serve.main import main

    main()
