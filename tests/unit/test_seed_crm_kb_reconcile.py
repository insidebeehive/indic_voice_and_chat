"""tests/unit/test_seed_crm_kb_reconcile.py

Covers the self-healing reconcile step added to ``_seed_crm_kb`` in
``src/main.py``: any ``CrmKBDocument`` row previously written by the seeder
(id namespaces ``crm_kb_{crm_id}_*`` or legacy ``global_kb_*``) that no
longer corresponds to a file under ``kb_dir`` gets pruned (DB row + pgvector
chunks), while admin-uploaded docs (``crmdoc_*``) are never touched, and a
second pass over the same ``kb_dir`` is a no-op (idempotent).

The prune half of this step is gated behind ``auto_prune`` (defaults to the
``VOX_KB_AUTO_PRUNE`` env var, off unless set) — all the pruning-behavior
tests below pass ``auto_prune=True`` explicitly to exercise that path
directly, without touching the environment. The gating itself (flag off ->
orphans detected but left alone; flag on -> pruned) is covered by
``test_reconcile_leaves_orphans_when_auto_prune_disabled`` and
``test_reconcile_prunes_when_auto_prune_enabled`` at the bottom of this file.
"""
from __future__ import annotations

from pathlib import Path

import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.main import _seed_crm_kb
from src.models.crm import Crm, CrmKBDocument
from src.models.database import Base
from src.providers.vector_store.faiss_store import FAISSAdapter
from src.rag.embeddings import HashEmbedder, IdentityReranker
from src.rag.retriever import HybridRetriever, RetrievalConfig

CRM_ID = "betstudio"


@pytest_asyncio.fixture
async def sessionmaker_fixture(tmp_faiss_index: str):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async with sm() as s:
        s.add(Crm(id=CRM_ID, name="BetStudio", base_url="https://x"))
        # Pre-migration orphan: a doc the seeder wrote under the legacy
        # `global_kb_*` id namespace whose source file no longer exists.
        s.add(CrmKBDocument(
            id="global_kb_06-casino-games", crm_id=CRM_ID,
            filename="06-casino-games.md", source_type="md",
            language="en", chunk_count=2,
            extra_data={"chunk_ids": [
                "global_kb_06-casino-games::chunk-0",
                "global_kb_06-casino-games::chunk-1",
            ]},
        ))
        # Live admin-uploaded doc — must never be touched by the seeder's
        # auto-reconcile, regardless of what's on disk.
        s.add(CrmKBDocument(
            id="crmdoc_manual_upload_1", crm_id=CRM_ID,
            filename="manual.md", source_type="md",
            language="en", chunk_count=1,
            extra_data={"chunk_ids": ["crmdoc_manual_upload_1::chunk-0"]},
        ))
        await s.commit()

    store = FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index})
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64), vector_store=store,
        reranker=IdentityReranker(),
        config=RetrievalConfig(strategy="hybrid", top_k=3, oversample_k=8),
    )

    deleted_calls: list[list[str]] = []
    orig_delete = retriever.delete

    async def _tracking_delete(chunk_ids):
        deleted_calls.append(list(chunk_ids))
        return await orig_delete(chunk_ids)

    retriever.delete = _tracking_delete  # type: ignore[assignment]

    class _FakeRegistry:
        def get(self, crm_id: str):
            return retriever if crm_id == CRM_ID else None

    yield sm, _FakeRegistry(), deleted_calls
    await engine.dispose()


async def test_reconcile_prunes_stale_docs_and_is_idempotent(
    sessionmaker_fixture, tmp_path: Path
) -> None:
    sm, registry, deleted_calls = sessionmaker_fixture

    kb_dir = tmp_path / "kb_global"
    kb_dir.mkdir()
    (kb_dir / "01-account-registration-login.md").write_text(
        "# Account registration and login\n\nPlaceholder content for the test.\n"
    )

    # --- First pass -------------------------------------------------
    await _seed_crm_kb(registry, sm, kb_dir=kb_dir, auto_prune=True)

    async with sm() as session:
        stale = await session.get(CrmKBDocument, "global_kb_06-casino-games")
        assert stale is None, "legacy orphaned doc should have been pruned"

        manual = await session.get(CrmKBDocument, "crmdoc_manual_upload_1")
        assert manual is not None, "admin-uploaded doc must never be pruned"

        fresh = await session.get(
            CrmKBDocument, f"crm_kb_{CRM_ID}_01-account-registration-login"
        )
        assert fresh is not None, "the file present on disk should have been (re)seeded"

        all_ids = {r.id for r in (await session.execute(select(CrmKBDocument))).scalars().all()}
        assert all_ids == {
            "crmdoc_manual_upload_1",
            f"crm_kb_{CRM_ID}_01-account-registration-login",
        }

    assert deleted_calls == [
        ["global_kb_06-casino-games::chunk-0", "global_kb_06-casino-games::chunk-1"]
    ], "the stale doc's chunk ids should have been resolved from extra_data and deleted"

    # --- Second pass: idempotent, no-op ------------------------------
    deleted_calls.clear()
    await _seed_crm_kb(registry, sm, kb_dir=kb_dir, auto_prune=True)

    async with sm() as session:
        all_ids_second = {
            r.id for r in (await session.execute(select(CrmKBDocument))).scalars().all()
        }
        assert all_ids_second == {
            "crmdoc_manual_upload_1",
            f"crm_kb_{CRM_ID}_01-account-registration-login",
        }

    assert deleted_calls == [], "second pass over the same kb_dir must prune nothing new"


async def test_reconcile_isolates_parse_failure_to_its_own_file(
    sessionmaker_fixture, tmp_path: Path, monkeypatch
) -> None:
    """A file that fails to parse must only protect *its own* derived doc ids
    from pruning. It must NOT blanket-disable reconcile for every other
    file/crm in the same pass — that was the bug in the old
    ``crm_had_failure`` flag, which tainted every crm globally on any single
    file's failure and silently defeated the self-healing prune."""
    sm, registry, deleted_calls = sessionmaker_fixture

    kb_dir = tmp_path / "kb_global"
    kb_dir.mkdir()
    (kb_dir / "01-good.md").write_text(
        "# Good doc\n\nPlaceholder content that parses fine.\n"
    )
    (kb_dir / "02-bad.md").write_text("this file will fail to parse\n")

    # Pre-seed: an orphan totally unrelated to either file above (must still
    # be pruned this pass), and a prior row for the bad file's own doc id
    # (must survive — protected by failed_stems since 02-bad fails to parse
    # again this pass).
    async with sm() as session:
        session.add(CrmKBDocument(
            id="global_kb_99-orphan", crm_id=CRM_ID,
            filename="99-orphan.md", source_type="md",
            language="en", chunk_count=1,
            extra_data={"chunk_ids": ["global_kb_99-orphan::chunk-0"]},
        ))
        session.add(CrmKBDocument(
            id=f"crm_kb_{CRM_ID}_02-bad", crm_id=CRM_ID,
            filename="02-bad.md", source_type="md",
            language="en", chunk_count=1,
            extra_data={"chunk_ids": [f"crm_kb_{CRM_ID}_02-bad::chunk-0"]},
        ))
        await session.commit()

    # _seed_crm_kb does `from src.rag.ingestion import ... parse_document`
    # *inside* the function body on every call, so patching a module-level
    # `src.main.parse_document` name has no effect — there isn't one. Patch
    # it where that import actually resolves from: src.rag.ingestion.
    import src.rag.ingestion as ingestion_module

    orig_parse_document = ingestion_module.parse_document

    def _flaky_parse_document(filename: str, data: bytes) -> str:
        if filename == "02-bad.md":
            raise ValueError("boom: simulated unparseable file")
        return orig_parse_document(filename, data)

    monkeypatch.setattr(ingestion_module, "parse_document", _flaky_parse_document)

    await _seed_crm_kb(registry, sm, kb_dir=kb_dir, auto_prune=True)

    async with sm() as session:
        good = await session.get(CrmKBDocument, f"crm_kb_{CRM_ID}_01-good")
        assert good is not None, "the file that parsed fine should have been (re)seeded"

        bad_prior = await session.get(CrmKBDocument, f"crm_kb_{CRM_ID}_02-bad")
        assert bad_prior is not None, (
            "the failed file's own prior row must be protected from pruning "
            "(failed_stems), not treated as gone"
        )

        orphan = await session.get(CrmKBDocument, "global_kb_99-orphan")
        assert orphan is None, (
            "an unrelated stale orphan must still be pruned even though some "
            "other file failed to parse in the same pass"
        )

        # The legacy orphan pre-seeded by the shared fixture is unrelated to
        # both files here too, and must also still be pruned.
        legacy_orphan = await session.get(CrmKBDocument, "global_kb_06-casino-games")
        assert legacy_orphan is None, (
            "an unrelated legacy-namespace orphan must still be pruned too"
        )

        manual = await session.get(CrmKBDocument, "crmdoc_manual_upload_1")
        assert manual is not None, "admin-uploaded doc must never be pruned"


async def test_reconcile_leaves_orphans_when_auto_prune_disabled(
    sessionmaker_fixture, tmp_path: Path
) -> None:
    """With auto_prune off (the default), orphaned rows found by the
    reconcile step must be detected but left completely alone: no DB row
    deleted, no chunk-delete call issued against the retriever. This is the
    default-off gate for the seeder's unattended destructive prune."""
    sm, registry, deleted_calls = sessionmaker_fixture

    kb_dir = tmp_path / "kb_global"
    kb_dir.mkdir()
    (kb_dir / "01-account-registration-login.md").write_text(
        "# Account registration and login\n\nPlaceholder content for the test.\n"
    )

    await _seed_crm_kb(registry, sm, kb_dir=kb_dir, auto_prune=False)

    async with sm() as session:
        stale = await session.get(CrmKBDocument, "global_kb_06-casino-games")
        assert stale is not None, (
            "orphaned doc must NOT be pruned when auto_prune is disabled"
        )

        manual = await session.get(CrmKBDocument, "crmdoc_manual_upload_1")
        assert manual is not None, "admin-uploaded doc must never be pruned"

        fresh = await session.get(
            CrmKBDocument, f"crm_kb_{CRM_ID}_01-account-registration-login"
        )
        assert fresh is not None, (
            "seeding new/updated content must still happen even when "
            "auto-prune is disabled"
        )

    assert deleted_calls == [], (
        "no chunk-delete call should be issued against the retriever when "
        "auto_prune is disabled"
    )


async def test_reconcile_prunes_when_auto_prune_enabled(
    sessionmaker_fixture, tmp_path: Path
) -> None:
    """Mirror of the disabled case: with auto_prune explicitly on, orphaned
    rows ARE pruned -- i.e. current/existing behavior, unlocked by opt-in."""
    sm, registry, deleted_calls = sessionmaker_fixture

    kb_dir = tmp_path / "kb_global"
    kb_dir.mkdir()
    (kb_dir / "01-account-registration-login.md").write_text(
        "# Account registration and login\n\nPlaceholder content for the test.\n"
    )

    await _seed_crm_kb(registry, sm, kb_dir=kb_dir, auto_prune=True)

    async with sm() as session:
        stale = await session.get(CrmKBDocument, "global_kb_06-casino-games")
        assert stale is None, "orphaned doc must be pruned when auto_prune is enabled"

    assert deleted_calls == [
        ["global_kb_06-casino-games::chunk-0", "global_kb_06-casino-games::chunk-1"]
    ]


async def test_reconcile_default_auto_prune_follows_env_var(
    sessionmaker_fixture, tmp_path: Path, monkeypatch
) -> None:
    """When ``auto_prune`` isn't passed explicitly, it resolves from
    ``VOX_KB_AUTO_PRUNE`` (default off) via ``src.main._kb_auto_prune_enabled``."""
    sm, registry, deleted_calls = sessionmaker_fixture

    kb_dir = tmp_path / "kb_global"
    kb_dir.mkdir()
    (kb_dir / "01-account-registration-login.md").write_text(
        "# Account registration and login\n\nPlaceholder content for the test.\n"
    )

    monkeypatch.delenv("VOX_KB_AUTO_PRUNE", raising=False)
    await _seed_crm_kb(registry, sm, kb_dir=kb_dir)
    async with sm() as session:
        stale = await session.get(CrmKBDocument, "global_kb_06-casino-games")
        assert stale is not None, "unset VOX_KB_AUTO_PRUNE must default to off"

    monkeypatch.setenv("VOX_KB_AUTO_PRUNE", "1")
    await _seed_crm_kb(registry, sm, kb_dir=kb_dir)
    async with sm() as session:
        stale = await session.get(CrmKBDocument, "global_kb_06-casino-games")
        assert stale is None, "VOX_KB_AUTO_PRUNE=1 must enable pruning"
