# Chonks — Design notes

This document explains why the system is shaped the way it is. It covers the questions Chonks is meant to answer and the ones it is not, the contract every index-time pass has to satisfy, the relationship to the cAST chunking paper, and the individual design choices that a measurement or a stated trade-off stands behind.

What each component does and how to run it is in [DOCS.md](DOCS.md) and [DEPLOY.md](DEPLOY.md). The published numbers, the corpora they were measured on, and their caveats are in [eval/FREEZES.md](eval/FREEZES.md), which this document cites rather than repeats. The literature this design overlaps with is in [RELATED.md](RELATED.md).

---

## When this helps, and when it doesn't

Chonks is a retrieval layer rather than a substitute for reading code. Its job is to reduce the question of which ten files to look at to a ranked handful of chunks, not to answer questions unread. Grep already handles exact-string lookup well, whereas Chonks costs an index that has to be built and kept fresh and a running embedding server. It repays that cost on the questions grep cannot answer directly, which are what a concept maps to in code, and how a structural change ripples outward. If the question is a known filename or an exact string, grep is faster and needs no index.

Where it helps:

- Structural orientation in an unfamiliar subsystem, where a PageRank-ranked symbol hierarchy replaces reading ten or more files to build a mental model.
- Concept and symbol lookup, since the result is exact chunks with `path:line` citations rather than a file-read-and-grep round trip, which keeps the caller's context window on relevant spans instead of whole files.
- The exact-match and graph-traversal lookups, when a name is already known. These are strictly more precise than semantic search for the questions of where X is defined, who calls X, and whether A reaches B.
- Broad architectural questions through research in `explore` scope, since it uses a typed-edge-weighted profile built for mapping a subsystem.

Where it is limited:

- Cross-subsystem architectural questions return a wide spread of chunks across several areas. Subsystem scoping or several targeted searches are the fallback when a narrower result is needed.
- Embeddings do not fully capture cross-language functional coupling. A GDScript call and the C++ method it binds to through `ClassDB::bind_method` can be functionally paired yet distant in embedding space, and each language side then has to be searched separately.
- The index goes stale the moment a file is edited, and a stale index is worse than no index. Status only reports when the index was last built and does not detect live on-disk drift, so staleness stays a judgment against one's own edit history. The reason for that limit is under Non-goals below.

The headline accuracy figure is Acc@5 64 of 100 on the held-out Loc-Bench instances, with the full set of measurements and their caveats in [eval/FREEZES.md](eval/FREEZES.md). The sample sizes are small and should be read as such.

---

## Non-goals

Each of these was considered and rejected for the trade-off given.

No re-ranker or utility LLM in the pipeline. There is no query expansion, no cross-encoder reranker, and no LLM-generated summaries. The pipeline is embedding, BM25, and graph traversal only, so Chonks retrieves and the outer LLM synthesizes. Folding synthesis in would mean shipping a model dependency inside the indexer and paying inference latency on every query, to duplicate a step the caller already performs with more conversation context than Chonks has.

No LLM-extracted graph. Every edge and every symbol comes from tree-sitter parsing and deterministic name resolution, so the same source produces the same graph on every re-index. That is what makes the graph auditable and cheap to update incrementally, and the incremental paths described below are only provably equivalent to a full rebuild because the underlying algorithm is a pure function of the chunk content. An LLM extraction pass would cost time and money proportional to corpus size and would make the graph nondeterministic, which is a bad property for something PageRank and the incremental updates depend on being stable. The published head-to-head between the two strategies is cited in [RELATED.md](RELATED.md).

No fork of LlamaIndex's RaptorPack. It would inherit framework abstractions, query engines, and an upgrade treadmill in exchange for roughly 300 lines of bespoke code. The idea taken from RAPTOR, which is summary nodes sharing an embedding store with raw chunks, is kept; the clustering plus LLM summarization that builds RAPTOR's tree is replaced with the existing folder hierarchy and structural signals. Clustering is unnecessary here because engineers already organise code into folders on purpose, and that organisation carries more signal than k-means over chunk embeddings would manufacture after the fact. The cost is a summary that describes what a folder contains rather than why.

No folder summaries in the main vector store. Indexing them alongside chunks makes summaries and chunks compete at retrieval time, so overview text outranks implementation chunks on some queries. They live in their own table and are consumed only by the score-modulation re-rank and the subsystem suggestion command.

No auto-applied subsystems. The config file is the user's source of truth, so the suggestion pass prints and the user commits.

No live staleness detection in the status endpoint. Both the stored mtime and the stored index time are frozen at index time, so comparing them can never surface real drift. Detecting real drift would need either an `lstat` sweep over every file on every poll of a cheap and frequently hit health check, or a file watcher, which is a dependency and a background process this tool does not want. Since re-indexing is an explicit step, status reports when the index was last built and leaves the rest to the caller.

---

## Post-index passes: the incremental contract

Every structure derived from the chunk table faces the same question after an incremental index run, which is how to update itself from the set of chunk ids added and the set deleted, without a full rebuild. This applies to the k-NN graph, the cross-reference graph, the folder summaries, and the FTS statistics. A full rebuild is the tempting default because it is simple and always correct, but it scales with the total corpus size, so on a large codebase a one-file change would cost the same as a from-scratch index.

Each pass answers the question independently. The k-NN build has an exact incremental path, with changed and deleted id sets, damaged-row repair, and a full rebuild above a churn threshold of 20% of the corpus. The reference-graph build has an incremental path of the same shape and the same threshold. The folder summaries are incremental through per-folder content hashes. FTS relies on triggers plus a statistics rebuild that only runs under a forced re-index. The changed and deleted delta is collected once during indexing and threaded to the call sites.

Two consequences are worth stating in advance. A parse-time or index-time feature added without an incremental path ships with a hidden cost proportional to the corpus on every future index run, and that cost is usually discovered in production on the largest corpus, so each new derived structure gets its incremental contract designed alongside its initial build. And parse-time features acquire coverage incrementally: anything extracted during parsing exists only for the chunks parsed since the feature shipped, so cold subtrees that never change, such as vendored dependencies, never gain it until a forced re-index. Mixed-coverage indexes should be expected in the wild, and a new parse-time feature should say so in its documentation.

The reference graph is the awkward case, because its schema records that an edge exists but not which symbol name produced it, while two corpus-global rules mean a name's resolved edge set can change even when neither endpoint of an edge was touched. An eighth definition of a shared name added elsewhere pushes that name over the cross-language cap, at which point every one of its existing cross-language edges has to disappear. The incremental path therefore works from the set of touched names rather than from the touched chunks alone, and it fully recomputes each affected chunk's outgoing edges against the current corpus instead of diffing against a previous state it cannot reconstruct.

---

## Relationship to the cAST paper

The chunker follows cAST (Zhang et al., 2025, Findings of EMNLP 2025, cited in [RELATED.md](RELATED.md)) in its core claim, which is that AST boundaries beat fixed-size and line-window splitting because they preserve semantic self-containedness. A fixed window has no idea where a function ends, so it either cuts a method in half, leaving two chunks that are each useless alone, or pads it with unrelated neighbouring code that dilutes the embedding. An AST-boundary chunk is one semantic unit, so a single retrieval hit is a complete answer. The cost is implementation complexity, since a fixed splitter is a one-liner and this is a multi-stage pipeline.

It deviates from cAST in five ways, each chosen for robustness on messy multi-language codebases.

A typed boundary filter instead of full AST traversal. cAST traverses the whole tree with no language-specific assumptions, whereas Chonks pre-selects named node types per language and only surfaces those as chunk candidates. This keeps chunk boundaries predictable per language rather than emergent from the generic tree structure, and the price is a per-language table that has to be filled in before a language is fully supported.

Raw byte spans for chunk sizing instead of non-whitespace characters. The whitespace compression applied before embedding brings the embedded representation in line with cAST's non-whitespace metric, so the metric is paid for once at embed time rather than on every size comparison during segmentation.

A conservative merge instead of a greedy fill. A segment merges forward only when the previous segment is below the minimum and the merged size would not exceed the ceiling. The second condition is what stops a tiny predecessor from absorbing a large successor and bypassing a ceiling that the split step already enforced. The merge carries the first named boundary's name and type onto the merged chunk, so a class of small methods stays in the reference graph, the map, and the breadcrumbs instead of collapsing into anonymous blocks. Several small functions from one file can merge into a single chunk that carries only the first function's name, so the later ones look as though they have no chunk. Every name is still individually addressable through the decoupled symbol index, which is one of the reasons that index exists.

A line-based fallback for leaf nodes. When a large node has no inner boundaries to recurse into, cAST has no equivalent fallback, and without one such a node has no bounded-size answer at all. A final byte-level pass then splits any single line still over the ceiling, such as a one-line data table, and logs the file so it can be excluded. Both the line fallback and the byte split carry the boundary's name and type onto every piece, for the same reason the merge does.

Overlap applied selectively. cAST uses no overlap, arguing that AST-boundary chunks are self-contained and that overlap would introduce duplicate retrieval. That argument holds for AST-produced chunks and not for the line-based fallback, which splits at arbitrary positions, so overlap is applied only to fallback chunks. This keeps cAST's no-duplicate property on the AST path while repairing the boundary artifacts on the other one.

---

## Measured rationale behind individual components

### Exclude paths are the biggest precision lever

Third-party dependencies, build artifacts, and generated code degrade ranking on every query, including queries about one's own code, because they contribute chunks that are semantically similar to the question and irrelevant to it. Excluding them is therefore the single largest quality lever available to a user, and it costs nothing at query time. The reference Godot index excludes `thirdparty/`, `misc/`, `doc/`, `.github/`, `editor/icons/`, and `modules/mono/glue/`, and its measurements in [eval/FREEZES.md](eval/FREEZES.md) are all on that basis.

Because a corpus can be swamped without anyone noticing, indexing reports the chunk share of each top-level path family and warns when the split looks wrong. The warning fires when a single family is at least 80% documentation and at least 40% of the whole corpus, or when documentation chunks are at least 50% of the index regardless of how they spread. Those thresholds are set so that a healthy repository where source is over 90% of the corpus never trips the warning, since normal code dominance fails the documentation leg of the first rule. The case being caught is a repository with a generated-documentation mirror checked in, which can silently become the majority of the index. The warning only names the family and prints a ready-to-paste exclude snippet, and it never changes what is indexed.

### The near-duplicate wall, and how the thresholds were picked

An agent that searches into a wall of mutually near-duplicate chunks, such as a template instantiated across many near-identical binary-format readers, tends to reformulate the query and land back in the same wall instead of obtaining new information. Search results therefore carry an observation of how much of the returned set is one such wall, measured as the share held by the largest connected component in a graph whose edges are pairwise cosine at or above a threshold.

The two constants were calibrated against a confirmed wall episode, which was nine replayed reformulations circling a wall of near-identical ELF, DWARF, COFF, Mach-O, and PDB stream-reader templates in LLVM, and against 22 healthy queries across LLVM and Godot. A high threshold does not work, because code embeddings for structurally identical templates rarely exceed a pairwise cosine of 0.9, so nothing fired at 0.9 or above. A low threshold is worse, because results for one query are correlated by construction, and at 0.75 or below healthy single-topic queries reached a wall share of 1.0. At a cosine threshold of 0.85 the healthy maximum was 0.5 and the wall's strongest reformulation 0.7, so a wall-share threshold of 0.6 splits that gap; one of the nine wall reformulations fires and none of the 22 healthy queries do. A second suspected wall, the AirbrakesV2 Loc-Bench instance, does not fire at the default result size, with a wall share of 0.2, which is reported here rather than fixed by lowering the threshold. The signal is an observation only and never changes the ranking, since a heuristic this weakly separated should not be allowed to reorder results.

The per-file cap on semantic search is the same problem at the level of one hot file. A query against a heavily referenced symbol can return a wall of near-duplicate chunks from one file and crowd out distinct files that are also relevant. The cap admits at most a fixed number of chunks per path while walking the rank-ordered pool, and a skipped chunk frees its slot for the next chunk in rank order. It never reorders within the selection, and if the pool has too few distinct files the result is short rather than padded. It is off by default, so the uncapped behaviour is reproduced exactly unless a caller asks for the cap.

### The interleave selector in research

Research returns a fixed number of chunks, and a plain truncation of the ranked pool buries the thing the caller usually wants, which is the set of distinct files connected to the anchor. Two levers replace the truncation. A per-file cap stops one file's near-duplicate chunks from consuming the window, which is the lever that does the work for file-level localization because it lifts a graph-connected file that sat just past the cutoff into the window. A reserved-slot interleave then rescues the top candidates by structural proximity to the seed anchor that the capped window would still miss. The shipped measurement of the combined effect is the feature-set coverage row in [eval/FREEZES.md](eval/FREEZES.md): on the 20 frozen Godot questions at a 50-chunk budget, research reaches 0.843 of each feature's files against 0.678 for flat search, 12 wins, 8 ties, and no losses.

What keeps the two levers safe is that placement is file-level aware. A file's rank is set by its first returned chunk, so redundant later chunks of an already-seen file can be sacrificed without changing any file's rank, and rescue slots are paid for first from those redundant chunks and only then from the weakest remaining file. A rescue therefore costs the weakest tail slot and never the high-confidence head that carries a query's real localizations. The selector is on by default and the plain truncation is kept as the eval control, so the two can be compared on the panel rather than argued about; the defaults it ships with are recorded in [eval/FREEZES.md](eval/FREEZES.md).

### The mentions layer, and what an ablation says about it

Typed edges require an AST-resolved call, import, or inherit fact. The inferred `mentions` layer adds an edge wherever a chunk's text names an indexed symbol, which is far less precise and far broader. It is most of the graph: on the reference Godot index it is 1.88M of 2.69M edges, or about 70%.

Mentions edges are the noisiest kind: any chunk whose text names an indexed symbol, no verified call, import, or inherit relationship required. Masking them changes nothing on the Loc-Bench repositories (64 of 100 either way), but costs about a quarter of Godot's recovered neighbourhood, research recall at 50 falling from 0.39 to 0.29. The split is because C++ templates and member calls often resolve to no typed edge at all, unlike Python's call edges, so mentions is what bridges that gap. Kept for that reason: it earns its place on C++-heavy corpora and is close to redundant on Python ones.

Because the inferred layer is noisy, a fraction of its pairs are relabelled by pointwise mutual information, which rewards a focused chunk that names a rare name and penalises both ubiquitous names and wall-of-names chunks, a distinction that document frequency alone does not make. The relabelling is inert by default, since the promoted label carries the same weight as the one it was carved from, and setting the fraction to zero reproduces the previous graph bit for bit. It is a label waiting for a weighting experiment rather than a change to ranking.

### Edge-type weighting, and why scope is a per-request choice

Pure query-cosine ranking only surfaces a candidate that echoes the query's own vocabulary, so a component's real neighbours, glue code using different words for the same idea, stay buried even after graph expansion reaches them (measured on Godot, gold neighbours sit at pool rank ~175, cosine ~0.55, beneath generic matches). The structural boost fixes this by scoring a candidate up when it's structurally adjacent to a strong seed, so a wired-in chunk rises while an isolated generic match doesn't. That's what makes research an exploration ranker over an anchor's neighbourhood, not a second relevance ranker duplicating hybrid search.

These weights are shared with index-time PageRank and default to 1.0 (unweighted behaviour, unchanged). They're exposed per request rather than fixed in config because the right weighting depends on intent: "how does X work" and "map this subsystem" want different points on the same trade-off. On Godot, the typed-edge-heavy `explore` profile roughly doubles relational-neighbourhood recall over the uniform `lookup` default, at the cost of single-target precision. Callers pick a named scope rather than tune five floats, on the bet that an LLM gets a name right more often than it tunes a weight map.

### The call-site fingerprint

When a called name has several candidate definers, fanning out to all of them is recall-safe and imprecise. Call sites carry evidence that narrows the set: a receiver token that matches a definer's own qualifier, and an argument count that a definer's signature could not satisfy. Narrowing on that evidence is a precision and recall trade rather than a safe filter, so it was gated against an oracle's typed-only recall, and when neither discriminator finds evidence the full un-narrowed definer set is kept. Absence of evidence is not evidence against a candidate.

The effect shows up on Godot feature-set coverage, where flat search rose from 0.614 to 0.678 while research stayed level at about 0.84, narrowing the gap between the two modes from 0.235 to 0.165. The panel showed no regression. The details of both runs are in [eval/FREEZES.md](eval/FREEZES.md).

### SQLite instead of a vector database

A dedicated vector service such as Pinecone or Qdrant would mean running, and keeping in sync, a second stateful system alongside the chunk metadata store, plus a network hop on every query. sqlite-vec instead keeps the vector index, the FTS5 keyword index, and the relational chunk and graph metadata in one file with one write path and one transaction boundary, so a chunk is inserted once and its vector, its FTS row, and its graph edges are either all consistent or all rolled back together. The trade-off is giving up the scaling ceiling of a purpose-built engine, which a single codebase does not need. Embeddings are stored int8-quantized, which is a quarter of the storage of float32 and, with unit normalization, keeps search quality.

The ceiling is real and worth naming. sqlite-vec does brute-force exact k-NN, so query latency scales linearly with corpus size. For a codebase of hundreds of thousands of files an HNSW-indexed store or a dedicated vector database would be the right answer, and the rest of the stack does not depend on which vector backend is underneath.

The same build-once-read-many trade is made twice more. PageRank is solved at index time and read through a table at query time, because the reference graph of the Godot index alone holds 2.69M edges and the count grows combinatorially with corpus size and language mix, so a live solve is expensive enough to risk exceeding a client timeout on every map call. The semantic k-NN graph is precomputed for the same reason, since computing it live with one vector query per chunk is quadratic in the corpus. Both pay for it in staleness between index runs, which the churn gates bound.

### The embedding model and its prefixes

The pipeline is model-agnostic, in that any OpenAI-compatible embeddings endpoint works. General-purpose models such as qwen3-embedding give good results, and models trained on code understand cross-file references, function signatures, and language-specific patterns better at a fraction of the size. `jina-code-embeddings-0.5b` is what setup writes and what every published number used. Its licence is CC-BY-NC-4.0, which is a real constraint for commercial use, and the Apache-2.0 alternative measured on the 15-instance panel is Qwen3-Embedding-0.6B with its query instruction; that comparison and its caveats are in [eval/FREEZES.md](eval/FREEZES.md). Switching models requires a full re-index, and a new database if the output dimension changes, so the configured model name is recorded in the database and a mismatch raises at open rather than silently mixing two embedding spaces.

Some embedders are trained to take a different instruction prefix on the query side and on the document side of a retrieval pair, and embedding both sides identically measurably degrades them, so Chonks applies the prefixes itself. The prefixes resolve from a preset keyed on the configured model name, and a model with no preset gets no prefix, so its inputs are byte-for-byte what they would have been. This matters for comparisons as much as for quality: a head-to-head against a tool with no prefix support is partly a measurement of prefix handling rather than of retrieval quality, which is why the ChunkHound comparison in [eval/FREEZES.md](eval/FREEZES.md) carries that caveat.

Indexing is gated by the embedding server. On real C++, measured over 500 Godot files, the parse and chunk stage does about 9,000 chunks per second single-threaded, which is roughly a hundred times faster than even a fast embedder consumes them. The single parser thread is therefore never the bottleneck and parallelising it would be wasted effort. It would need processes rather than threads in any case, since tree-sitter with the Python AST walk is GIL-bound, and threads measured about 1x against about 4x for processes. The lever that matters is keeping the GPU fed, and once it is fed the next ceiling is the serial database commit.

---

## Design decisions

The trade-offs below are not covered above, and are recorded here for a reader who wants to know why the obvious alternative was not taken.

| Decision | Why |
|---|---|
| Macro self-heal is C++-only. | Annotation macros are a C++-specific pathology, since other languages' annotations are grammar-legal. The known HLSL breakage, in technique and pass blocks, is a block-construct grammar gap that a blank-and-reparse loop cannot fix; it needs error-node boundary salvage instead. |
| Excluded directories are pruned out of the filesystem walk itself, not filtered per file. | Per-file filtering still walks every file in a huge excluded tree. Pruning the directory list skips it without descending at all. The prune stays include-aware, so a directory an include reaches into is still walked. |
| The FTS mirror is kept in sync by SQLite triggers rather than by manual dual-writes at each call site. | A dual-write desyncs as soon as any code path touches the chunk table without remembering the second write. Triggers make desync structurally impossible regardless of the write path, including the conflict-delete of an insert-or-replace. |
| The schema version is checked on every open and a mismatch raises. | It catches an embedding-model switch or a schema change before an index run starts instead of in the middle of one. |
| The k-NN matmul picks its float32 fast path by embedding dimension rather than by a fixed byte-size threshold. | An int8 dot product is exactly representable in float32 up to a dimension of 1040, so at common dimensions such as 896 the fast path is bit-identical to the exact one. The earlier dimension-blind threshold chose the exact path anyway and cost roughly 50 times the wall clock on a large re-index for no accuracy gain. |
| The embedder's retry loop bails immediately on a connectivity error instead of spending its full per-chunk retry budget. | Against a server that is simply down, the full budget burns minutes on doomed requests before the outage surfaces. Failing fast surfaces it at once. |
| Indexing raises and exits when a worker thread dies, instead of waiting on the queue. | A silently hung run gives the operator no signal at all, whereas a loud error makes an unrecoverable crash visibly different from a slow but progressing run. |
| The server refuses an absolute codebase path from a config file it found by auto-discovery, and requires an explicit config flag to accept one. | Without the guard, a config file dropped into a directory the server is later started from could silently redirect what it reads and re-embeds to an arbitrary filesystem location, with no user action signalling the redirection. |
