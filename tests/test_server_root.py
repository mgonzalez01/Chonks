import json

import chonks.serve.app as serve_app
import chonks.serve.projects as serve_projects


def test_root_endpoint_returns_name_version_and_endpoints():
    body = json.loads(serve_app.root().body)

    assert body["name"] == "chonks"
    assert isinstance(body["version"], str) and body["version"]
    assert body["status_url"] == "/status"
    assert "degraded" not in body


def test_root_endpoint_lists_status_and_search():
    body = json.loads(serve_app.root().body)
    paths = {e["path"] for e in body["endpoints"]}

    assert "/status" in paths
    assert "/search" in paths
    assert "/" in paths


def test_root_endpoint_every_entry_has_method_path_description():
    body = json.loads(serve_app.root().body)

    for entry in body["endpoints"]:
        assert entry["method"] in ("GET", "POST")
        assert entry["path"].startswith("/")
        assert entry["description"]


def test_root_endpoint_does_not_require_a_project():
    # Cold-client entry point, so it can't depend on _get_project().
    serve_projects._projects.clear()
    body = json.loads(serve_app.root().body)
    assert body["name"] == "chonks"
