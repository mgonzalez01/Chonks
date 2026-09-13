"""eval/materialize_source.py: rebuilds a source tree from chunk content when the
real checkout is gone. Gaps between chunks come back blank, which handicaps the
baseline arm and flatters Chonks; use a real checkout for a publishable result.
"""
import argparse
import os
import sqlite3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Chonks sqlite DB to reconstruct a source tree from")
    ap.add_argument("--out", required=True, help="directory to write the reconstructed tree to")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    files: dict[str, list] = {}
    for path, s, e, content in con.execute(
            "SELECT path, start_line, end_line, content FROM chunks ORDER BY path, start_line"):
        files.setdefault(path, []).append((s, e, content))

    covered = total = 0
    for path, chunks in files.items():
        n = max(e for _, e, _ in chunks)
        lines = [""] * n
        for s, e, content in chunks:
            body = content.split("\n")
            for j, ln in enumerate(body):
                idx = s - 1 + j
                if 0 <= idx < n:
                    lines[idx] = ln
            covered += e - s + 1
        total += n
        dest = os.path.join(args.out, path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as f:
            f.write("\n".join(lines) + "\n")

    print(f"materialized {len(files)} files -> {args.out}")
    print(f"line coverage: {covered/total:.0%}  (gaps are blank; see fidelity caveat in this file's docstring)")


if __name__ == "__main__":
    main()
