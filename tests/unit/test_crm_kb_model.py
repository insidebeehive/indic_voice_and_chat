"""tests/unit/test_crm_kb_model.py"""
from __future__ import annotations

import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.models.crm import Crm, CrmKBDocument
from src.models.database import Base


@pytest.mark.asyncio
async def test_crm_kb_document_requires_crm_id() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Crm(id="betstudio", name="BetStudio", base_url="https://x"))
        await s.commit()
        s.add(CrmKBDocument(id="doc1", crm_id="betstudio", filename="a.md",
                             source_type="md", language="en", chunk_count=2))
        await s.commit()

    inspector_cols = {c.name for c in CrmKBDocument.__table__.columns}
    assert "crm_id" in inspector_cols
    assert CrmKBDocument.__table__.c.crm_id.nullable is False
    await engine.dispose()
