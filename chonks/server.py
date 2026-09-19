"""Old import path for chonks.serve.app, chonks.serve.main, chonks.serve.models and chonks.serve.projects.
Stays runnable: `python chonks/server.py` starts the server."""

import uvicorn
from fastapi import HTTPException

from chonks.serve.app import (
    app,
    find_by_message,
    hubs,
    impact,
    investigate,
    outgoing,
    repomap,
    research,
    root,
    search,
    symbol,
    trace,
    usages,
)
from chonks.serve.main import main
from chonks.serve.models import (
    FindByMessageRequest,
    HubsRequest,
    ImpactRequest,
    InvestigateRequest,
    OutgoingRequest,
    RepomapRequest,
    ResearchRequest,
    SearchRequest,
    SymbolRequest,
    TraceRequest,
    UsagesRequest,
    _QUERY_MAX_LEN,
)
from chonks.serve.projects import (
    DEFAULT_PROJECT,
    _projects,
)


if __name__ == "__main__":
    main()
