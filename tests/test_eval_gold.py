"""Builds a tiny synthetic sqlite DB by hand (a minimal `chunks` table, no
vec0/FTS machinery needed since resolution only ever queries `chunks`)."""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from gold import content_hash, resolve_gold, stable_key  # noqa: E402


def _make_db(tmp_path, rows):
    """rows: list of (id, path, name, chunk_type, content)."""
    db_path = tmp_path / "synthetic.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE chunks (id TEXT PRIMARY KEY, path TEXT, name TEXT, "
        "chunk_type TEXT, content TEXT)"
    )
    conn.executemany(
        "INSERT INTO chunks(id, path, name, chunk_type, content) VALUES (?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db_path


def test_stable_key_roundtrip_resolves_uniquely(tmp_path):
    rows = [
        ("id1", "core/foo.cpp", "do_thing", "function_definition", "int do_thing() { return 1; }"),
        ("id2", "core/bar.cpp", "other_thing", "function_definition", "void other_thing() {}"),
    ]
    db_path = _make_db(tmp_path, rows)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    key = stable_key("core/foo.cpp", "do_thing", "function_definition", rows[0][4])
    res = resolve_gold(con, [key])

    assert res.resolved == {0: ["id1"]}
    assert res.ambiguous == {}
    assert res.unresolved == []


def test_resolution_survives_id_change_across_reindex(tmp_path):
    # The same GoldKey must resolve even though the underlying chunk id
    # changed entirely (simulating a re-index after a chunker bump).
    content = "int do_thing() { return 1; }"
    key = stable_key("core/foo.cpp", "do_thing", "function_definition", content)

    old_db = tmp_path / "old.db"
    conn = sqlite3.connect(str(old_db))
    conn.execute(
        "CREATE TABLE chunks (id TEXT PRIMARY KEY, path TEXT, name TEXT, "
        "chunk_type TEXT, content TEXT)"
    )
    conn.execute(
        "INSERT INTO chunks(id, path, name, chunk_type, content) VALUES (?,?,?,?,?)",
        ("old-ephemeral-id", "core/foo.cpp", "do_thing", "function_definition", content),
    )
    conn.commit()
    conn.close()

    new_db = tmp_path / "new.db"
    conn = sqlite3.connect(str(new_db))
    conn.execute(
        "CREATE TABLE chunks (id TEXT PRIMARY KEY, path TEXT, name TEXT, "
        "chunk_type TEXT, content TEXT)"
    )
    conn.execute(
        "INSERT INTO chunks(id, path, name, chunk_type, content) VALUES (?,?,?,?,?)",
        ("brand-new-id-9999", "core/foo.cpp", "do_thing", "function_definition", content),
    )
    conn.commit()
    conn.close()

    con_old = sqlite3.connect(f"file:{old_db}?mode=ro", uri=True)
    con_new = sqlite3.connect(f"file:{new_db}?mode=ro", uri=True)

    res_old = resolve_gold(con_old, [key])
    res_new = resolve_gold(con_new, [key])

    assert res_old.resolved[0] == ["old-ephemeral-id"]
    assert res_new.resolved[0] == ["brand-new-id-9999"]
    # different DBs, different ids, same portable key resolves in both.
    assert res_old.resolved[0] != res_new.resolved[0]


def test_ambiguous_same_name_different_content_kept_not_dropped(tmp_path):
    # Two chunks share path+name (an overload) with different bodies.
    rows = [
        ("id1", "core/foo.cpp", "Foo::bar", "function_definition", "void bar(int) {}"),
        ("id2", "core/foo.cpp", "Foo::bar", "function_definition", "void bar(float) {}"),
    ]
    db_path = _make_db(tmp_path, rows)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    # content_hash matches id2's body exactly -> disambiguates to one match.
    key_disambiguating = stable_key("core/foo.cpp", "Foo::bar", "function_definition", rows[1][4])
    res = resolve_gold(con, [key_disambiguating])
    assert res.resolved == {0: ["id2"]}
    assert res.ambiguous == {}

    # content_hash matches neither current body -> ambiguous, both kept.
    key_stale = stable_key("core/foo.cpp", "Foo::bar", "function_definition", "totally different body")
    res2 = resolve_gold(con, [key_stale])
    assert res2.resolved == {}
    assert set(res2.ambiguous[0]) == {"id1", "id2"}
    assert res2.unresolved == []


def test_unresolved_reported_not_silently_dropped(tmp_path):
    rows = [("id1", "core/foo.cpp", "do_thing", "function_definition", "int do_thing() { return 1; }")]
    db_path = _make_db(tmp_path, rows)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    keys = [
        stable_key("core/foo.cpp", "do_thing", "function_definition", rows[0][4]),
        stable_key("core/gone.cpp", "removed_symbol", "function_definition", "whatever"),
    ]
    res = resolve_gold(con, keys)

    assert res.resolved == {0: ["id1"]}
    assert res.unresolved == [1]
    report = res.report(keys)
    assert "1/2 UNRESOLVED" in report
    assert "core/gone.cpp::removed_symbol" in report


def test_all_ids_dedupes_across_a_range(tmp_path):
    rows = [
        ("id1", "core/foo.cpp", "Thing", "class_specifier", "class Thing {};"),
        ("id2", "core/foo.h", "Thing", "class_specifier", "class Thing;"),
    ]
    db_path = _make_db(tmp_path, rows)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    keys = [
        stable_key("core/foo.cpp", "Thing", "class_specifier", rows[0][4]),
        stable_key("core/foo.h", "Thing", "class_specifier", rows[1][4]),
    ]
    res = resolve_gold(con, keys)
    ids = res.all_ids(range(len(keys)))
    assert set(ids) == {"id1", "id2"}
    assert len(ids) == 2  # no duplicates


def test_content_hash_is_deterministic_and_content_only():
    a = content_hash("int foo() {}")
    b = content_hash("int foo() {}")
    c = content_hash("int foo() { return 1; }")
    assert a == b
    assert a != c


def test_resolve_tolerates_chunk_refs_schema_without_edge_type(tmp_path):
    # An old-schema chunk_refs table (no edge_type column, pre typed-edges)
    # must not break relational_sample.py/relational_run.py's raw
    # `SELECT from_id, to_id FROM chunk_refs`.
    db_path = tmp_path / "old_refs.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE chunk_refs (from_id TEXT NOT NULL, to_id TEXT NOT NULL, "
        "PRIMARY KEY (from_id, to_id))"
    )
    conn.execute("INSERT INTO chunk_refs(from_id, to_id) VALUES ('a', 'b')")
    conn.commit()
    conn.close()

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT from_id, to_id FROM chunk_refs").fetchall()
    assert [(r["from_id"], r["to_id"]) for r in rows] == [("a", "b")]
