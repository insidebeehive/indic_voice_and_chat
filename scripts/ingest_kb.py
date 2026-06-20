#!/usr/bin/env python3
"""Bulk-ingest a directory of knowledge-base docs into a tenant's KB via the API.

Posts every supported file under ``--dir`` to ``POST /api/v1/knowledge/ingest``
with the tenant's bearer token. Use it to load the global KB (the chatbot then
answers general queries from it via RAG).

Example:
  python scripts/ingest_kb.py \
    --dir /path/to/kb/global \
    --base-url https://voicebot.biznexis.in \
    --token vox_xxxxx

Note: the deployed app must have the Gemini-embeddings build (PR #132) live, or
ingest returns 500 (no embedder).
"""

from __future__ import annotations

import argparse
import mimetypes
import pathlib
import sys

import httpx

_DEFAULT_EXTS = ".md,.txt,.pdf,.docx,.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="directory of docs to ingest (recursive)")
    ap.add_argument("--base-url", required=True, help="e.g. https://voicebot.biznexis.in")
    ap.add_argument("--token", required=True, help="tenant bearer token (vox_...)")
    ap.add_argument("--ext", default=_DEFAULT_EXTS, help="comma-separated extensions")
    args = ap.parse_args()

    exts = {e if e.startswith(".") else f".{e}" for e in args.ext.split(",")}
    base = args.base_url.rstrip("/")
    root = pathlib.Path(args.dir)
    # Skip macOS resource-fork files (._foo) and dotfiles.
    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith(".")
    )
    if not files:
        print(f"no files with {sorted(exts)} under {root}", file=sys.stderr)
        return 1

    ok = 0
    with httpx.Client(timeout=120.0) as client:
        for f in files:
            with open(f, "rb") as fh:
                mime = mimetypes.guess_type(f.name)[0] or "text/plain"
                resp = client.post(
                    f"{base}/api/v1/knowledge/ingest",
                    headers={"Authorization": f"Bearer {args.token}"},
                    files={"file": (f.name, fh, mime)},
                )
            if resp.status_code == 200:
                body = resp.json()
                print(f"OK    {f.name}  ({body.get('chunks_indexed')} chunks, {body.get('language')})")
                ok += 1
            else:
                print(f"FAIL  {f.name}  HTTP {resp.status_code}  {resp.text[:160]}")
    print(f"\ningested {ok}/{len(files)} files")
    return 0 if ok == len(files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
