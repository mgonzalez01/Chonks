"""eval/git_gold.py: mines commit file-sets as gold, not the retrieval graph, so
scoring the graph-reading arm against graph-derived gold stays non-circular.
Question text is authored later: describe behavior, don't leak identifiers.
"""
import argparse
import json
import subprocess

# Source-file buckets for the --cross-language filter. Extend as corpora need.
LANG_BUCKETS = {
    "cpp": {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".inc"},
    "csharp": {".cs"},
    "python": {".py"},
    "rust": {".rs"},
    "java": {".java"},
    "js": {".js", ".jsx", ".ts", ".tsx"},
    "lua": {".lua"},
    # Shaders co-change with the C++/C# that drives them and are indexed, so
    # they're retrievable gold too.
    "shader": {".hlsl", ".fx", ".fxh", ".glsl", ".vert", ".frag", ".comp"},
}
SOURCE_EXTS = set().union(*LANG_BUCKETS.values())

# Subject markers for changes whose file sets are NOT a coherent feature story.
_SKIP_SUBJECT = ("nfc", "revert", "reformat", "clang-format", "typo",
                 "whitespace", "rename", "bump", "regenerate")


def _lang_of(path: str) -> str | None:
    dot = path.rfind(".")
    if dot < 0:
        return None
    ext = path[dot:].lower()
    for lang, exts in LANG_BUCKETS.items():
        if ext in exts:
            return lang
    return None


def _keep(candidates: list[dict], sha: str, subject: str, files: list[str],
          min_files: int, max_files: int, cross_language: bool) -> None:
    """Shared filter pipeline for the mined (id, subject, files) triples."""
    if any(m in subject.lower() for m in _SKIP_SUBJECT):
        return
    src = [f for f in files if _lang_of(f)]
    if not (min_files <= len(src) <= max_files):
        return
    langs = sorted({_lang_of(f) for f in src})
    if cross_language and len(langs) < 2:
        return
    candidates.append({
        "sha": sha,
        "subject": subject,
        "languages": langs,
        "gold_files": sorted(src),
        "question": None,  # authored later: describe behavior, no identifier leaks
    })


def mine(repo: str, since: str, min_files: int, max_files: int,
         count: int, cross_language: bool) -> list[dict]:
    log = subprocess.run(
        ["git", "-C", repo, "log", "--no-merges", f"--since={since}",
         "--numstat", "--format=@@%H%x00%s"],
        capture_output=True, text=True, check=True,
    ).stdout

    out: list[dict] = []
    sha, subject, files = None, "", []

    def flush():
        if sha is not None:
            _keep(out, sha, subject, files, min_files, max_files, cross_language)

    for line in log.splitlines():
        if line.startswith("@@"):
            flush()
            sha, _, subject = line[2:].partition("\x00")
            files = []
        elif line.strip():
            parts = line.split("\t")
            if len(parts) == 3:
                files.append(parts[2])
    flush()
    return out[:count]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="git checkout to mine")
    ap.add_argument("--out", required=True)
    ap.add_argument("--since", default="6 months ago")
    ap.add_argument("--min-files", type=int, default=3)
    ap.add_argument("--max-files", type=int, default=8)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--cross-language", action="store_true")
    args = ap.parse_args()

    cands = mine(args.repo, args.since, args.min_files, args.max_files,
                 args.count, args.cross_language)
    json.dump(cands, open(args.out, "w"), indent=1)
    print(f"wrote {len(cands)} candidates -> {args.out}")
    for c in cands[:8]:
        print(f"  {c['sha'][:10]} [{','.join(c['languages'])}] "
              f"{len(c['gold_files'])} files  {c['subject'][:70]}")


if __name__ == "__main__":
    main()
