"""Dataset abstractions for benchmark inputs.

JSONL-backed (one record per line) so datasets are easy to diff and to
generate from CRM exports. The four canonical schemas:

STTSample           {id, audio_path | audio_bytes_b64, transcript, language, code_switch?}
TTSSample           {id, text, language, voice_id?}
RAGSample           {id, query, expected_chunks: [id, ...], expected_answer?,
                     expected_spans: [str, ...], expected_files: [str, ...],
                     lang?, intent?, unanswerable?}
TaskScenario        {id, user_turns: [...], expected_disposition, required_slots: {...}}

Each loader can either consume an inlined records list (for tests) or a
``Path`` to a JSONL file.

RAGSample note: ``expected_chunks`` (chunk ids) is legacy and stays for
backward compatibility, but it's fragile — chunk ids are
``f"{doc_id}::chunk-{index}"`` and both halves are unstable (re-chunking
renumbers ``index``; a non-seeded doc gets a random ``doc_id``). New datasets
should label with ``expected_spans`` (verbatim KB text) + ``expected_files``
(source filename) instead, which survive a re-chunk / re-ingest since they're
resolved against whatever chunks are actually in the index at scoring time
(see ``src/benchmarks/rag_benchmark.py``).
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union


# --- Sample types --------------------------------------------------------


@dataclass
class STTSample:
    id: str
    transcript: str
    language: Optional[str] = None
    code_switch: bool = False
    audio_path: Optional[str] = None
    audio_bytes: Optional[bytes] = None

    def resolve_audio(self, base_dir: Optional[Path] = None) -> bytes:
        if self.audio_bytes is not None:
            return self.audio_bytes
        if self.audio_path is None:
            raise ValueError(f"sample {self.id!r} has no audio data")
        p = Path(self.audio_path)
        if not p.is_absolute() and base_dir is not None:
            p = base_dir / p
        return p.read_bytes()


@dataclass
class TTSSample:
    id: str
    text: str
    language: str = "hi-IN"
    voice_id: Optional[str] = None


@dataclass
class RAGSample:
    id: str
    query: str
    expected_chunks: list[str] = field(default_factory=list)
    expected_answer: Optional[str] = None
    # Verbatim substrings of KB documents — a retrieved chunk counts as
    # relevant if it contains at least one of these (whitespace-normalised
    # containment, see score_retrieval_spans). Anchored to text rather than
    # chunk ids so the label survives a re-chunk / re-ingest.
    expected_spans: list[str] = field(default_factory=list)
    # Source filenames (e.g. "04-deposits.md") for the coarser document-level
    # file_hit diagnostic.
    expected_files: list[str] = field(default_factory=list)
    # Script/language of the query: "en" | "hinglish" | "hi".
    lang: Optional[str] = None
    # Short slug grouping queries that mean the same thing across languages
    # (e.g. "deposit_not_credited"), so retrieval quality can be compared
    # across lang for the same underlying question.
    intent: Optional[str] = None
    # True when the KB genuinely has no answer -- retrieval SHOULD come back
    # empty or weak. Scored separately (false_positive_rate), never folded
    # into the normal precision/recall/MRR means.
    unanswerable: bool = False
    # Which corpus tier the expected document belongs to: "pack" (auto-seeded
    # for every tenant), "module" (opt-in KB module), or "layout" (a tenant
    # picks exactly one). Only the "pack" tier is guaranteed present in a
    # default-seeded index -- see run_retrieval_benchmark's unindexed-sample
    # detection and the --tier CLI filter in scripts/run_benchmark.py, both
    # of which exist because scoring a "module"/"layout" sample against an
    # index that never ingested that file isn't a retrieval failure.
    tier: Optional[str] = None


@dataclass
class TaskTurn:
    role: str         # "user" | "system_event"
    content: str = ""
    event: Optional[str] = None


@dataclass
class TaskScenario:
    id: str
    user_turns: list[TaskTurn] = field(default_factory=list)
    expected_disposition: str = ""
    required_slots: dict[str, Any] = field(default_factory=dict)
    language: str = "hi"


# --- Generic JSONL helpers ----------------------------------------------


SourceLike = Union[Path, str, Iterable[dict]]


def _read_records(source: SourceLike) -> list[dict]:
    """Resolve either a JSONL path or an iterable of dict records."""
    if isinstance(source, (str, Path)):
        p = Path(source)
        records: list[dict] = []
        with p.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{p}:{line_num} invalid JSON: {e}") from e
        return records
    return list(source)


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


# --- Concrete loaders ---------------------------------------------------


def load_stt_dataset(source: SourceLike) -> list[STTSample]:
    out: list[STTSample] = []
    for r in _read_records(source):
        audio_bytes = None
        if "audio_bytes_b64" in r and r["audio_bytes_b64"]:
            audio_bytes = base64.b64decode(r["audio_bytes_b64"])
        out.append(STTSample(
            id=str(r["id"]),
            transcript=str(r.get("transcript") or ""),
            language=r.get("language"),
            code_switch=bool(r.get("code_switch", False)),
            audio_path=r.get("audio_path"),
            audio_bytes=audio_bytes,
        ))
    return out


def load_tts_dataset(source: SourceLike) -> list[TTSSample]:
    return [
        TTSSample(
            id=str(r["id"]),
            text=str(r["text"]),
            language=r.get("language", "hi-IN"),
            voice_id=r.get("voice_id"),
        )
        for r in _read_records(source)
    ]


def load_rag_dataset(source: SourceLike) -> list[RAGSample]:
    return [
        RAGSample(
            id=str(r["id"]),
            query=str(r["query"]),
            expected_chunks=list(r.get("expected_chunks") or []),
            expected_answer=r.get("expected_answer"),
            expected_spans=list(r.get("expected_spans") or []),
            expected_files=list(r.get("expected_files") or []),
            lang=r.get("lang"),
            intent=r.get("intent"),
            unanswerable=bool(r.get("unanswerable", False)),
            tier=r.get("tier"),
        )
        for r in _read_records(source)
    ]


def load_task_dataset(source: SourceLike) -> list[TaskScenario]:
    out: list[TaskScenario] = []
    for r in _read_records(source):
        turns = []
        for t in r.get("user_turns") or []:
            turns.append(TaskTurn(
                role=t.get("role", "user"),
                content=t.get("content") or "",
                event=t.get("event"),
            ))
        out.append(TaskScenario(
            id=str(r["id"]),
            user_turns=turns,
            expected_disposition=str(r.get("expected_disposition") or ""),
            required_slots=dict(r.get("required_slots") or {}),
            language=str(r.get("language") or "hi"),
        ))
    return out
