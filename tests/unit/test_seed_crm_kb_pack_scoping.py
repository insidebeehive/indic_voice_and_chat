"""tests/unit/test_seed_crm_kb_pack_scoping.py

Covers Task 2 of docs/superpowers/plans/2026-09-04-decouple-betting-vertical-from-core.md:
``_seed_crm_kb`` (src/main.py) seeds a bundled KB pack only into CRMs whose
``bundled_kb_pack`` column matches the pack name being seeded — not into
every CRM unconditionally. A CRM with ``bundled_kb_pack`` unset (NULL) or set
to a different pack name gets zero docs from this pack.
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

OPTED_IN_CRM_ID = "betstudio"
NO_PACK_CRM_ID = "no-pack-crm"
OTHER_PACK_CRM_ID = "other-pack-crm"


def _make_retriever(index_path: str) -> HybridRetriever:
    store = FAISSAdapter({"embedding_dim": 64, "index_path": index_path})
    return HybridRetriever(
        embedder=HashEmbedder(dim=64), vector_store=store,
        reranker=IdentityReranker(),
        config=RetrievalConfig(strategy="hybrid", top_k=3, oversample_k=8),
    )


@pytest_asyncio.fixture
async def scoping_fixture(tmp_path: Path):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async with sm() as s:
        # Opted in to the pack being seeded.
        s.add(Crm(
            id=OPTED_IN_CRM_ID, name="BetStudio", base_url="https://x",
            bundled_kb_pack="betting-default",
        ))
        # Never opted in at all (NULL).
        s.add(Crm(id=NO_PACK_CRM_ID, name="No Pack CRM", base_url="https://y"))
        # Opted into a *different* named pack.
        s.add(Crm(
            id=OTHER_PACK_CRM_ID, name="Other Pack CRM", base_url="https://z",
            bundled_kb_pack="some-other-pack",
        ))
        await s.commit()

    retrievers = {
        OPTED_IN_CRM_ID: _make_retriever(str(tmp_path / "idx_opted_in")),
        NO_PACK_CRM_ID: _make_retriever(str(tmp_path / "idx_no_pack")),
        OTHER_PACK_CRM_ID: _make_retriever(str(tmp_path / "idx_other_pack")),
    }

    class _FakeRegistry:
        def get(self, crm_id: str):
            return retrievers.get(crm_id)

    yield sm, _FakeRegistry()
    await engine.dispose()


async def test_only_opted_in_crm_gets_seeded(
    scoping_fixture, tmp_path: Path
) -> None:
    """A CRM whose bundled_kb_pack matches kb_dir's pack gets seeded; CRMs
    with no pack set or a different pack set get nothing from this pass."""
    sm, registry = scoping_fixture

    kb_dir = tmp_path / "betting_default_pack"
    kb_dir.mkdir()
    (kb_dir / "01-account-registration-login.md").write_text(
        "# Account registration and login\n\nPlaceholder content for the test.\n"
    )

    await _seed_crm_kb(
        registry, sm, kb_dir=kb_dir, auto_prune=False,
        bundled_kb_pack="betting-default",
    )

    async with sm() as session:
        opted_in_doc = await session.get(
            CrmKBDocument, f"crm_kb_{OPTED_IN_CRM_ID}_01-account-registration-login"
        )
        assert opted_in_doc is not None, (
            "CRM with bundled_kb_pack='betting-default' must be seeded from "
            "the matching kb_dir"
        )

        no_pack_docs = (
            await session.execute(
                select(CrmKBDocument).where(CrmKBDocument.crm_id == NO_PACK_CRM_ID)
            )
        ).scalars().all()
        assert no_pack_docs == [], (
            "CRM with bundled_kb_pack unset (NULL) must get zero docs seeded"
        )

        other_pack_docs = (
            await session.execute(
                select(CrmKBDocument).where(CrmKBDocument.crm_id == OTHER_PACK_CRM_ID)
            )
        ).scalars().all()
        assert other_pack_docs == [], (
            "CRM opted into a different named pack must get zero docs from "
            "this pack's kb_dir"
        )


async def test_no_opted_in_crms_is_a_clean_no_op(
    tmp_path: Path,
) -> None:
    """When zero CRM rows have opted into the pack being seeded, the whole
    pass is a no-op (returns early) rather than crashing or seeding anyone."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async with sm() as s:
        s.add(Crm(id=NO_PACK_CRM_ID, name="No Pack CRM", base_url="https://y"))
        await s.commit()

    class _EmptyRegistry:
        def get(self, crm_id: str):
            return None

    kb_dir = tmp_path / "betting_default_pack"
    kb_dir.mkdir()
    (kb_dir / "01-account-registration-login.md").write_text(
        "# Account registration and login\n\nPlaceholder content for the test.\n"
    )

    # Must not raise.
    await _seed_crm_kb(
        _EmptyRegistry(), sm, kb_dir=kb_dir, auto_prune=False,
        bundled_kb_pack="betting-default",
    )

    async with sm() as session:
        all_docs = (await session.execute(select(CrmKBDocument))).scalars().all()
        assert all_docs == []

    await engine.dispose()
