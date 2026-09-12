"""tests/unit/test_build_experiment_index_script.py"""
from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import sys
import types

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "build_experiment_index_script",
    pathlib.Path(__file__).parent.parent.parent / "scripts" / "build_experiment_index.py",
)
build_experiment_index = importlib.util.module_from_spec(_SPEC)
# Must be registered in sys.modules before exec_module: the script defines a
# @dataclass, and dataclasses resolves annotations (deferred by `from
# __future__ import annotations`) by looking up `cls.__module__` in
# sys.modules -- without this it raises AttributeError on a None module.
sys.modules[_SPEC.name] = build_experiment_index
_SPEC.loader.exec_module(build_experiment_index)

from src.interfaces.vector_store import Document  # noqa: E402
from src.rag.ingestion import ChunkConfig  # noqa: E402


# --- guards ---------------------------------------------------------------


def test_guard_refuses_live_crm_id() -> None:
    with pytest.raises(build_experiment_index.GuardError):
        build_experiment_index.guard_target_crm_id("betstudio")


def test_guard_allows_non_live_crm_id() -> None:
    build_experiment_index.guard_target_crm_id("betstudio-exp-small")  # no raise


def test_guard_refuses_nonempty_target_without_overwrite() -> None:
    with pytest.raises(build_experiment_index.GuardError):
        build_experiment_index.guard_overwrite(
            existing_count=5, overwrite=False, crm_id="betstudio-exp-small",
        )


def test_guard_allows_nonempty_target_with_overwrite() -> None:
    build_experiment_index.guard_overwrite(
        existing_count=5, overwrite=True, crm_id="betstudio-exp-small",
    )  # no raise


def test_guard_allows_empty_target_without_overwrite() -> None:
    build_experiment_index.guard_overwrite(
        existing_count=0, overwrite=False, crm_id="betstudio-exp-small",
    )  # no raise


def test_main_refuses_live_crm_id_via_cli(capsys) -> None:
    rc = build_experiment_index.main(["--crm-id", "betstudio"])
    assert rc == 2
    assert "betstudio" in capsys.readouterr().err


def test_guard_no_live_id_collision_blocks_live_prefix() -> None:
    with pytest.raises(build_experiment_index.GuardError):
        build_experiment_index.guard_no_live_id_collision(
            ["crm_kb_betstudio_04-deposits::chunk-0"]
        )


def test_guard_no_live_id_collision_allows_experimental_ids() -> None:
    build_experiment_index.guard_no_live_id_collision(
        ["crm_kb_betstudio-exp-small_04-deposits::chunk-0"]
    )  # no raise -- hyphen, not underscore, after "betstudio"


def test_main_refuses_underscore_crm_id_that_lands_in_live_namespace(
    tmp_path: pathlib.Path, capsys,
) -> None:
    # crm_id "betstudio_extra" -> doc_id "crm_kb_betstudio_extra_x", which
    # starts with the live seeder's exact id prefix "crm_kb_betstudio_" --
    # guard_target_crm_id alone (exact-string match on "betstudio") would
    # miss this; guard_no_live_id_collision is what catches it.
    (tmp_path / "x.md").write_text("some content", encoding="utf-8")

    rc = build_experiment_index.main(["--crm-id", "betstudio_extra", "--kb-dir", str(tmp_path)])

    assert rc == 2
    assert "live" in capsys.readouterr().err.lower()


# --- chunk planning ---------------------------------------------------------


def _write_kb(tmp_path: pathlib.Path, filename: str, text: str) -> pathlib.Path:
    p = tmp_path / filename
    p.write_text(text, encoding="utf-8")
    return p


def test_discover_files_matches_seeder_filters(tmp_path: pathlib.Path) -> None:
    (tmp_path / "sub").mkdir()
    _write_kb(tmp_path, "a.md", "hello")
    _write_kb(tmp_path, "sub/b.txt", "world")
    (tmp_path / ".hidden.md").write_text("x")
    (tmp_path / "ignored.bin").write_text("x")

    files = build_experiment_index.discover_files(tmp_path)

    assert {f.name for f in files} == {"a.md", "b.txt"}


def test_plan_index_chunk_count_at_small_chunk_size(tmp_path: pathlib.Path) -> None:
    # 400 chars of plain text, no natural separators the recursive chunker
    # can use short of a hard word-boundary split -- forces predictable
    # chunking at a small chunk_size.
    text = ("word " * 100).strip()  # 499 chars
    _write_kb(tmp_path, "01-doc.md", text)

    cfg = ChunkConfig(chunk_size=25, chunk_overlap=0)  # ~100 chars/chunk target
    planned = build_experiment_index.plan_index(tmp_path, "betstudio-exp-small", cfg)

    assert len(planned) == 1
    pf = planned[0]
    assert pf.filename == "01-doc.md"
    assert pf.doc_id == "crm_kb_betstudio-exp-small_01-doc"
    # ~499 chars / ~100 chars target -> multiple chunks
    assert len(pf.chunks) > 1


def test_plan_index_ids_and_metadata_match_live_seeder_conventions(tmp_path: pathlib.Path) -> None:
    _write_kb(tmp_path, "04-deposits.md", "Some deposit content that is short.")

    cfg = ChunkConfig()  # defaults, same as src.main._seed_crm_kb
    crm_id = "betstudio-exp-small"
    planned = build_experiment_index.plan_index(tmp_path, crm_id, cfg)

    assert len(planned) == 1
    pf = planned[0]
    assert pf.doc_id == f"crm_kb_{crm_id}_04-deposits"

    docs = build_experiment_index.to_documents(pf)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.id == f"{pf.doc_id}::chunk-0"
    assert doc.metadata["filename"] == "04-deposits.md"  # basename, for eval file_hit
    assert doc.metadata["document_id"] == pf.doc_id
    assert doc.metadata["section"] == 0
    assert doc.metadata["page"] == 0


def test_plan_index_skips_empty_files(tmp_path: pathlib.Path) -> None:
    _write_kb(tmp_path, "empty.md", "   \n  ")
    _write_kb(tmp_path, "real.md", "Actual content here.")

    planned = build_experiment_index.plan_index(tmp_path, "betstudio-exp-small", ChunkConfig())

    assert [pf.filename for pf in planned] == ["real.md"]


# --- dry run never touches network/db ---------------------------------------


def test_dry_run_main_does_not_apply(tmp_path: pathlib.Path, capsys) -> None:
    _write_kb(tmp_path, "doc.md", "Some content for a dry run test.")

    rc = build_experiment_index.main([
        "--crm-id", "betstudio-exp-small",
        "--kb-dir", str(tmp_path),
    ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "Planned chunks total" in out


def test_dry_run_never_reaches_db_or_embedding_helpers(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    """Stronger than asserting on stdout text: proves the dry-run code path
    literally never calls the functions that would open a DB connection or
    hit the embedding API, by making each one blow up if invoked."""

    def _boom(*_a, **_kw):
        raise AssertionError("dry run must not call this")

    monkeypatch.setattr("src.providers.get_vector_store", _boom)
    monkeypatch.setattr("src.rag.embeddings.GeminiEmbedder", _boom)
    monkeypatch.setattr(build_experiment_index, "resolve_database_url", _boom)
    monkeypatch.setattr(build_experiment_index, "resolve_gemini_api_key", _boom)

    _write_kb(tmp_path, "doc.md", "Some content for a dry run test.")
    rc = build_experiment_index.main(["--crm-id", "betstudio-exp-small", "--kb-dir", str(tmp_path)])

    assert rc == 0


# --- resolve_database_url / resolve_gemini_api_key ---------------------------


def test_resolve_database_url_prefers_settings_secrets(monkeypatch) -> None:
    import src.config as config_module

    fake_settings = types.SimpleNamespace(
        secrets=types.SimpleNamespace(DATABASE_URL="postgresql://from-settings"),
    )
    monkeypatch.setattr(config_module, "get_settings", lambda: fake_settings)

    assert build_experiment_index.resolve_database_url() == "postgresql://from-settings"


def test_resolve_database_url_falls_back_to_env_when_settings_unset(monkeypatch) -> None:
    import src.config as config_module

    fake_settings = types.SimpleNamespace(secrets=types.SimpleNamespace(DATABASE_URL=None))
    monkeypatch.setattr(config_module, "get_settings", lambda: fake_settings)
    monkeypatch.setenv("DATABASE_URL", "postgresql://from-env")

    assert build_experiment_index.resolve_database_url() == "postgresql://from-env"


def test_resolve_database_url_falls_back_to_env_when_settings_raise(monkeypatch) -> None:
    import src.config as config_module

    def _raise():
        raise RuntimeError("boom")

    monkeypatch.setattr(config_module, "get_settings", _raise)
    monkeypatch.setenv("DATABASE_URL", "postgresql://from-env-2")

    assert build_experiment_index.resolve_database_url() == "postgresql://from-env-2"


def test_resolve_gemini_api_key_prefers_settings_secrets(monkeypatch) -> None:
    import src.config as config_module

    fake_settings = types.SimpleNamespace(
        secrets=types.SimpleNamespace(GEMINI_API_KEY="key-from-settings"),
    )
    monkeypatch.setattr(config_module, "get_settings", lambda: fake_settings)

    assert build_experiment_index.resolve_gemini_api_key() == "key-from-settings"


def test_resolve_gemini_api_key_falls_back_to_env(monkeypatch) -> None:
    import src.config as config_module

    fake_settings = types.SimpleNamespace(secrets=types.SimpleNamespace(GEMINI_API_KEY=None))
    monkeypatch.setattr(config_module, "get_settings", lambda: fake_settings)
    monkeypatch.setenv("GEMINI_API_KEY", "key-from-env")

    assert build_experiment_index.resolve_gemini_api_key() == "key-from-env"


# --- _apply (fully faked DB + embedder -- no network) ------------------------


class _FakeVectorStore:
    def __init__(self, initial_docs=None):
        self._docs = {d.id: d for d in (initial_docs or [])}
        self.indexed_batches: list[list] = []
        self.deleted_ids: list[str] = []

    async def count(self):
        return len(self._docs)

    async def list_documents(self, limit=2000):
        return list(self._docs.values())[:limit]

    async def delete(self, doc_ids):
        n = 0
        for i in doc_ids:
            if i in self._docs:
                del self._docs[i]
                n += 1
        self.deleted_ids.extend(doc_ids)
        return n

    async def index(self, documents):
        for d in documents:
            self._docs[d.id] = d
        self.indexed_batches.append(list(documents))
        return len(documents)


class _FakeEmbedder:
    calls: list[tuple] = []

    def __init__(self, *, dim, api_key=None, document_task_type=None, query_task_type=None, **_kw):
        self.dim = dim
        self.document_task_type = document_task_type

    def embed_documents(self, texts):
        type(self).calls.append((self.document_task_type, list(texts)))
        return [[0.1] * self.dim for _ in texts]


def _patch_apply_deps(monkeypatch, store) -> None:
    monkeypatch.setattr("src.providers.get_vector_store", lambda cfg: store)
    monkeypatch.setattr("src.rag.embeddings.GeminiEmbedder", _FakeEmbedder)
    monkeypatch.setattr(build_experiment_index, "resolve_database_url", lambda: "postgresql://fake")
    monkeypatch.setattr(build_experiment_index, "resolve_gemini_api_key", lambda: "fake-key")


def test_apply_writes_all_planned_chunks(tmp_path: pathlib.Path, monkeypatch) -> None:
    _write_kb(tmp_path, "doc.md", "Some content long enough to form at least one chunk.")
    planned = build_experiment_index.plan_index(tmp_path, "betstudio-exp-small", ChunkConfig())

    store = _FakeVectorStore()
    _patch_apply_deps(monkeypatch, store)
    args = build_experiment_index.build_arg_parser().parse_args([
        "--crm-id", "betstudio-exp-small", "--kb-dir", str(tmp_path), "--apply",
    ])

    total = asyncio.run(build_experiment_index._apply(args, planned))

    expected = sum(len(pf.chunks) for pf in planned)
    assert total == expected
    assert len(store._docs) == expected


def test_apply_refuses_nonempty_target_without_overwrite(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    _write_kb(tmp_path, "doc.md", "content")
    planned = build_experiment_index.plan_index(tmp_path, "betstudio-exp-small", ChunkConfig())

    store = _FakeVectorStore(initial_docs=[Document(id="existing::chunk-0", content="x")])
    _patch_apply_deps(monkeypatch, store)
    args = build_experiment_index.build_arg_parser().parse_args([
        "--crm-id", "betstudio-exp-small", "--kb-dir", str(tmp_path), "--apply",
    ])

    with pytest.raises(build_experiment_index.GuardError):
        asyncio.run(build_experiment_index._apply(args, planned))

    assert store.indexed_batches == []  # refused before writing anything


def test_apply_overwrite_deletes_stale_chunks_then_writes(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    _write_kb(tmp_path, "doc.md", "content")
    planned = build_experiment_index.plan_index(tmp_path, "betstudio-exp-small", ChunkConfig())

    stale = Document(id="stale-doc::chunk-0", content="leftover from a previous chunk size")
    store = _FakeVectorStore(initial_docs=[stale])
    _patch_apply_deps(monkeypatch, store)
    args = build_experiment_index.build_arg_parser().parse_args([
        "--crm-id", "betstudio-exp-small", "--kb-dir", str(tmp_path), "--apply", "--overwrite",
    ])

    asyncio.run(build_experiment_index._apply(args, planned))

    assert stale.id in store.deleted_ids
    assert stale.id not in store._docs


def test_apply_raises_on_embedding_length_mismatch(tmp_path: pathlib.Path, monkeypatch) -> None:
    _write_kb(tmp_path, "doc.md", "content")
    planned = build_experiment_index.plan_index(tmp_path, "betstudio-exp-small", ChunkConfig())

    class _ShortEmbedder:
        def __init__(self, **_kw) -> None:
            pass

        def embed_documents(self, texts):
            return []  # short response -- must not be silently zipped away

    store = _FakeVectorStore()
    monkeypatch.setattr("src.providers.get_vector_store", lambda cfg: store)
    monkeypatch.setattr("src.rag.embeddings.GeminiEmbedder", _ShortEmbedder)
    monkeypatch.setattr(build_experiment_index, "resolve_database_url", lambda: "postgresql://fake")
    monkeypatch.setattr(build_experiment_index, "resolve_gemini_api_key", lambda: "fake-key")
    args = build_experiment_index.build_arg_parser().parse_args([
        "--crm-id", "betstudio-exp-small", "--kb-dir", str(tmp_path), "--apply",
    ])

    with pytest.raises(RuntimeError, match="embed_documents returned"):
        asyncio.run(build_experiment_index._apply(args, planned))

    assert store.indexed_batches == []


def test_apply_passes_document_task_type_to_embedder(tmp_path: pathlib.Path, monkeypatch) -> None:
    _write_kb(tmp_path, "doc.md", "content long enough to chunk")
    planned = build_experiment_index.plan_index(tmp_path, "betstudio-exp-small", ChunkConfig())

    store = _FakeVectorStore()
    _FakeEmbedder.calls.clear()
    _patch_apply_deps(monkeypatch, store)
    args = build_experiment_index.build_arg_parser().parse_args([
        "--crm-id", "betstudio-exp-small", "--kb-dir", str(tmp_path),
        "--apply", "--document-task-type", "RETRIEVAL_DOCUMENT",
    ])

    asyncio.run(build_experiment_index._apply(args, planned))

    assert _FakeEmbedder.calls
    assert all(task_type == "RETRIEVAL_DOCUMENT" for task_type, _texts in _FakeEmbedder.calls)
