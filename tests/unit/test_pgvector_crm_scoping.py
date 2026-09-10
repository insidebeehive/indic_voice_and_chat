"""tests/unit/test_pgvector_crm_scoping.py"""
from __future__ import annotations

import logging
import os
import uuid

import pytest

log = logging.getLogger(__name__)


def _resolve_database_url() -> str:
    """Resolve DATABASE_URL the same way the application does (via
    src.config, which reads .env through pydantic-settings), falling back to
    the raw process env var.

    A plain ``pytest tests/`` invocation never puts .env into the process
    env, so checking ``os.environ["DATABASE_URL"]`` directly always skips
    this file even when the repo's own .env has a working URL -- meaning the
    multi-tenant isolation boundary this file guards never actually runs by
    default. Going through settings fixes that.

    Any failure while resolving settings (missing/invalid .env, bad YAML,
    etc.) must fall back to the env var rather than turn a clean skip into a
    collection error -- this runs at collection time via the skipif below.
    """
    try:
        from src.config import get_settings
        url = get_settings().secrets.DATABASE_URL
        if url:
            return url
    except Exception:
        pass
    return os.environ.get("DATABASE_URL", "")


_DATABASE_URL = _resolve_database_url()

pytestmark = pytest.mark.skipif(
    not _DATABASE_URL,
    reason=(
        "integration test: requires a live Postgres with the pgvector "
        "extension and a voicebot.knowledge_chunks(vector(384)) table; "
        "set DATABASE_URL (or .env) to run it"
    ),
)

# Real deployments use a 384-dim embedding column (see
# docs/pgvector_setup.sql), so the test exercises the adapter with that same
# dimension rather than a toy dim that would only happen to match whatever
# table shape a given database has.
_DIM = 384


def _embedding(seed: float) -> list[float]:
    return [seed] * _DIM


@pytest.fixture(autouse=True)
async def _reset_pgvector_module_state():
    """pytest-asyncio hands each test its own event loop by default (this
    repo doesn't set loop_scope="session"), but pgvector_store caches its
    asyncpg pools and schema/dim verification results at module level, shared
    across every PGVectorAdapter for the life of the process. With two tests
    in this file now hitting the DB, the second one would otherwise reuse a
    pool created on the first test's (already-closed) event loop and blow up
    with "another operation is in progress". Reset the module's cached state
    before each test so it builds its own pool on its own loop -- production
    code runs a single event loop for the process lifetime and never
    exercises this path."""
    from src.providers.vector_store import pgvector_store as _pgv
    _pgv._pools = {}
    _pgv._schema_ready = set()
    _pgv._verified_dims = set()
    yield


@pytest.mark.asyncio
async def test_crm_scoped_chunk_is_isolated_from_tenant_scoped_chunk() -> None:
    from src.providers.vector_store.pgvector_store import PGVectorAdapter
    from src.interfaces.vector_store import Document

    # Unique per run so concurrent/CI runs never collide and so cleanup only
    # ever touches rows this test itself created -- the live table also
    # holds ~56 production-ish chunks under real crm_id/tenant_id scopes that
    # must never be touched.
    run_id = uuid.uuid4().hex[:12]
    crm_id = f"test-crm-{run_id}"
    tenant_id = f"test-tenant-{run_id}"
    crm_doc_id = f"crm-doc-{run_id}"
    tenant_doc_id = f"tenant-doc-{run_id}"

    crm_store = PGVectorAdapter({"embedding_dim": _DIM, "crm_id": crm_id, "database_url": _DATABASE_URL})
    tenant_store = PGVectorAdapter({"embedding_dim": _DIM, "tenant_id": tenant_id, "database_url": _DATABASE_URL})

    try:
        await crm_store.index([Document(id=crm_doc_id, content="crm content",
                                         metadata={}, embedding=_embedding(0.1))])
        await tenant_store.index([Document(id=tenant_doc_id, content="tenant content",
                                            metadata={}, embedding=_embedding(0.1))])

        crm_results = await crm_store.search(_embedding(0.1), top_k=10)
        tenant_results = await tenant_store.search(_embedding(0.1), top_k=10)

        # The assertion that matters: a CRM-scoped chunk must not be
        # retrievable through a tenant-scoped adapter, and vice versa.
        assert {r.document.id for r in crm_results} == {crm_doc_id}
        assert {r.document.id for r in tenant_results} == {tenant_doc_id}
        assert await crm_store.count() == 1
        assert await tenant_store.count() == 1
    finally:
        # Each cleanup is independently guarded -- crm_store.delete raising
        # must not skip tenant_store.delete (and vice versa), or a failure on
        # one side leaks the other side's row permanently.
        try:
            await crm_store.delete([crm_doc_id])
        except Exception:
            log.exception("cleanup: failed to delete crm-scoped test doc %r", crm_doc_id)
        try:
            await tenant_store.delete([tenant_doc_id])
        except Exception:
            log.exception("cleanup: failed to delete tenant-scoped test doc %r", tenant_doc_id)


@pytest.mark.asyncio
async def test_embedding_dim_mismatch_raises_at_connect_time() -> None:
    """An adapter configured with the wrong embedding_dim must fail loudly
    (naming both dimensions and the config key to change) instead of
    silently proceeding to a confusing DataError on first insert."""
    from src.providers.vector_store.pgvector_store import PGVectorAdapter

    # The live table is vector(384) (see docs/pgvector_setup.sql); 4 is
    # deliberately wrong so the mismatch check must fire. (The
    # _reset_pgvector_module_state fixture above guarantees a fresh
    # _verified_dims cache, so this doesn't depend on test order.)
    bad_dim = 4
    run_id = uuid.uuid4().hex[:12]

    store = PGVectorAdapter({
        "embedding_dim": bad_dim,
        "crm_id": f"test-crm-{run_id}",
        "database_url": _DATABASE_URL,
    })

    with pytest.raises(RuntimeError) as excinfo:
        await store._pool()

    message = str(excinfo.value)
    assert "384" in message
    assert str(bad_dim) in message
    assert "embedding_dim" in message
