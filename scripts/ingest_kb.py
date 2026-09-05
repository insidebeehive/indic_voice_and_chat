#!/usr/bin/env python3
"""Bulk-ingest a directory of knowledge-base docs into a tenant's KB via the API.

Posts every supported file under ``--dir`` to ``POST /api/v1/knowledge/ingest``
with the tenant's bearer token. Use it to bulk-load a KB pack (the chatbot
then answers general queries from it via RAG).

Example (bulk):
  python scripts/ingest_kb.py \
    --dir /path/to/kb/packs/betting-default \
    --base-url https://voicebot.biznexis.in \
    --token vox_xxxxx

Example (single file — e.g. onboarding a tenant's layout-specific KB doc,
see data/kb/layouts/operator-to-layout.md):
  python scripts/ingest_kb.py \
    --file data/kb/layouts/layout-1.md \
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


def _collect_files(
    dir_arg: str | None, file_args: list[str] | None, exts: set[str],
) -> list[pathlib.Path]:
    """Resolve the final file list from --dir (recursive, extension-filtered
    sweep) and/or --file (explicit paths — ingested regardless of the
    extension filter, since naming one directly is an intentional choice).
    De-duplicates by resolved path (a --file path may already be covered by
    an overlapping --dir sweep)."""
    files: list[pathlib.Path] = []
    if dir_arg:
        root = pathlib.Path(dir_arg)
        files.extend(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in exts and not p.name.startswith(".")
        )
    if file_args:
        for f in file_args:
            p = pathlib.Path(f)
            if not p.is_file():
                raise FileNotFoundError(f"--file path not found: {f}")
            files.append(p)
    seen: set[pathlib.Path] = set()
    unique: list[pathlib.Path] = []
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return sorted(unique)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", help="directory of docs to ingest (recursive, extension-filtered)")
    ap.add_argument(
        "--file", nargs="+",
        help="one or more explicit file paths to ingest (bypasses the --ext filter)",
    )
    ap.add_argument("--base-url", required=True, help="e.g. https://voicebot.biznexis.in")
    ap.add_argument("--token", required=True, help="tenant bearer token (vox_...)")
    ap.add_argument("--ext", default=_DEFAULT_EXTS, help="comma-separated extensions (applies to --dir only)")
    args = ap.parse_args()

    if not args.dir and not args.file:
        print("error: at least one of --dir or --file is required", file=sys.stderr)
        return 2

    exts = {e if e.startswith(".") else f".{e}" for e in args.ext.split(",")}
    base = args.base_url.rstrip("/")
    try:
        files = _collect_files(args.dir, args.file, exts)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not files:
        print(f"no files with {sorted(exts)} under {args.dir}", file=sys.stderr)
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
