"""Schema version and table definitions for the sqlite store."""

SCHEMA_VERSION = 5

SCHEMA_DDL = """
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS files (
                path         TEXT PRIMARY KEY,
                size         INTEGER,
                mtime        REAL,
                content_hash TEXT,
                indexed_at   REAL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id          TEXT PRIMARY KEY,
                path        TEXT NOT NULL,
                language    TEXT,
                chunk_type  TEXT,
                name        TEXT,
                start_line  INTEGER,
                end_line    INTEGER,
                content     TEXT NOT NULL,
                indexed_at  REAL,
                metadata    TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
            CREATE INDEX IF NOT EXISTS idx_files_path  ON files(path);
            -- Backs build_refs' incremental "find chunks named X" lookup;
            -- without it, resolving a few touched names is a full scan.
            CREATE INDEX IF NOT EXISTS idx_chunks_name ON chunks(name);

            CREATE TABLE IF NOT EXISTS chunk_refs (
                from_id   TEXT NOT NULL,
                to_id     TEXT NOT NULL,
                edge_type TEXT NOT NULL DEFAULT 'mentions',
                PRIMARY KEY (from_id, to_id)
            );

            CREATE INDEX IF NOT EXISTS idx_chunk_refs_from ON chunk_refs(from_id);
            CREATE INDEX IF NOT EXISTS idx_chunk_refs_to   ON chunk_refs(to_id);

            -- Persisted PageRank, computed once at index time. Empty on a
            -- DB from before this existed; compute_pagerank_global falls
            -- back to a live compute in that case.
            CREATE TABLE IF NOT EXISTS chunk_pagerank (
                chunk_id TEXT PRIMARY KEY,
                score    REAL NOT NULL
            );

            -- Persisted in-degree, computed at graph-build time. Empty on a
            -- DB from before this existed; get_hubs falls back to a live
            -- GROUP BY, guarded by _HUBS_GLOBAL_MAX_CHUNKS, in that case.
            CREATE TABLE IF NOT EXISTS chunk_indegree (
                chunk_id  TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                n         INTEGER NOT NULL,
                PRIMARY KEY (chunk_id, edge_type)
            );

            -- File hierarchy (dir/file nodes + 'contains' edges) is kept out
            -- of chunk_refs so PageRank doesn't pick up hierarchy nodes and
            -- chunk_pagerank stays chunk-only, comparable to pre-v2 baselines.
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id        TEXT PRIMARY KEY,   -- "dir:<path>" | "file:<path>"
                kind      TEXT NOT NULL,      -- 'dir' | 'file'
                path      TEXT NOT NULL,
                parent_id TEXT                -- containing dir's node id; NULL at root
            );

            CREATE INDEX IF NOT EXISTS idx_graph_nodes_path   ON graph_nodes(path);
            CREATE INDEX IF NOT EXISTS idx_graph_nodes_parent ON graph_nodes(parent_id);

            CREATE TABLE IF NOT EXISTS graph_edges (
                from_id   TEXT NOT NULL,
                to_id     TEXT NOT NULL,      -- node id, or a chunks.id for file->chunk
                edge_type TEXT NOT NULL,      -- 'contains' (Stage 1)
                PRIMARY KEY (from_id, to_id, edge_type)
            );

            CREATE INDEX IF NOT EXISTS idx_graph_edges_from ON graph_edges(from_id);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_to   ON graph_edges(to_id);

            CREATE TABLE IF NOT EXISTS folder_summaries (
                path              TEXT PRIMARY KEY,
                summary           TEXT NOT NULL,
                summary_embedding BLOB NOT NULL,
                content_hash      TEXT NOT NULL,
                generated_at      REAL
            );

            CREATE TABLE IF NOT EXISTS chunk_neighbors (
                chunk_id    TEXT NOT NULL,
                neighbor_id TEXT NOT NULL,
                distance    REAL NOT NULL,
                PRIMARY KEY (chunk_id, neighbor_id)
            );

            CREATE INDEX IF NOT EXISTS idx_chunk_neighbors_chunk ON chunk_neighbors(chunk_id);
            -- Reverse lookup for the incremental k-NN update: finds rows
            -- whose neighbor list contained a chunk that was just deleted,
            -- without a full table scan.
            CREATE INDEX IF NOT EXISTS idx_chunk_neighbors_neighbor ON chunk_neighbors(neighbor_id);

            -- Decoupled symbol index: one row per named boundary, independent
            -- of how the chunker packs content, so folded/merged/split chunks
            -- still have every symbol findable. chunk_id may be NULL.
            CREATE TABLE IF NOT EXISTS symbols (
                id          INTEGER PRIMARY KEY,
                path        TEXT NOT NULL,
                name        TEXT NOT NULL,
                kind        TEXT,
                language    TEXT,
                start_line  INTEGER NOT NULL,
                end_line    INTEGER,
                chunk_id    TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_symbols_name  ON symbols(name);
            CREATE INDEX IF NOT EXISTS idx_symbols_path  ON symbols(path);
            CREATE INDEX IF NOT EXISTS idx_symbols_chunk ON symbols(chunk_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                id UNINDEXED,
                name,
                content,
                content=chunks,
                content_rowid=rowid
            );

            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, id, name, content)
                VALUES (new.rowid, new.id, new.name, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, id, name, content)
                VALUES ('delete', old.rowid, old.id, old.name, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, id, name, content)
                VALUES ('delete', old.rowid, old.id, old.name, old.content);
                INSERT INTO chunks_fts(rowid, id, name, content)
                VALUES (new.rowid, new.id, new.name, new.content);
            END;

            -- CREATE TABLE IF NOT EXISTS alone migrates an old DB: no
            -- SCHEMA_VERSION bump, no forced re-index. Coverage is tracked
            -- by _LITERAL_INDEX_META_KEY, not by this table being empty.
            CREATE TABLE IF NOT EXISTS chunk_literals (
                chunk_id TEXT NOT NULL,
                text     TEXT NOT NULL,
                skeleton TEXT,
                line     INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_chunk_literals_chunk ON chunk_literals(chunk_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS literals_fts USING fts5(
                text,
                content=chunk_literals,
                content_rowid=rowid
            );

            CREATE TRIGGER IF NOT EXISTS chunk_literals_ai AFTER INSERT ON chunk_literals BEGIN
                INSERT INTO literals_fts(rowid, text) VALUES (new.rowid, new.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chunk_literals_ad AFTER DELETE ON chunk_literals BEGIN
                INSERT INTO literals_fts(literals_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chunk_literals_au AFTER UPDATE ON chunk_literals BEGIN
                INSERT INTO literals_fts(literals_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
                INSERT INTO literals_fts(rowid, text) VALUES (new.rowid, new.text);
            END;
            """
