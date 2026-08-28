"""Regression tests for chat history hydration on agent (re)connect.

The chatbot agent factory always hands back an AgentSession with turns=[]
(see make_chatbot_factory in src/bootstrap.py). Without _hydrate_agent_history
(src/api/chat.py), a WS reconnect / REST call against an *existing*
session_id makes the bot answer as if the conversation had just begun, even
though the full transcript is sitting in chat_messages. These tests exercise
the four call sites: chat_websocket, chat_message, upload_media and
process_message.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.agents.base import AgentSession
from src.agents.chatbot import ChatBotAgent, MAX_HISTORY_TURNS
from src.api import chat
from src.api.deps import get_db_session
from src.auth import TenantContext, register_tenant_for_test
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import TenantSettings
from src.dialogue.context import SessionStore
from src.interfaces.llm import ILLMProvider, LLMMessage, LLMResult
from src.interfaces.vector_store import Document
from src.models.chat import ChatMessage, ChatSession
from src.models.database import Base
from src.models.tenant import Tenant
from src.providers.vector_store.faiss_store import FAISSAdapter
from src.rag.embeddings import HashEmbedder
from src.rag.retriever import HybridRetriever, RetrievalConfig

HEADERS = {"Authorization": "Bearer test-token"}


class _RecordingLLM(ILLMProvider):
    """Fake LLM that captures every generate() call's message list, so tests
    can assert on exactly what history reached the model."""

    def __init__(self) -> None:
        self.calls: list[list] = []
        self._json = json.dumps({
            "response_text": "Sure.", "language": "en",
            "sources_used": [], "confidence": "high", "action": "none",
        })

    async def generate(self, messages, config) -> LLMResult:
        self.calls.append(list(messages))
        return LLMResult(text=self._json, finish_reason="stop", usage={})

    async def generate_stream(self, messages, config) -> AsyncIterator[str]:
        if False:
            yield  # pragma: no cover


@pytest.fixture
async def app(tmp_faiss_index: str, fake_redis):
    store = FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index})
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(strategy="hybrid", top_k=2, oversample_k=8, reranking=False),
    )
    await retriever.index([
        Document(id="c1", content="Plan B has 500GB unlimited.", metadata={"filename": "plans.md", "page": 0})
    ])

    session_store = SessionStore(fake_redis, ttl_seconds=300, tenant_id="t1")

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Tenant(id="t1", slug="t1", name="Acme"))
        await s.commit()

    async def _session_override():
        async with sm() as session:
            yield session

    llm = _RecordingLLM()

    async def factory(tenant: TenantContext, session_id: str, *, customer_id=None) -> ChatBotAgent:
        return ChatBotAgent(
            session=AgentSession(session_id=session_id),
            llm=llm,
            retriever=retriever,
            company_name=tenant.settings.name,
            store=session_store,
        )

    chat.set_chatbot_factory(factory)
    chat.set_chat_sessionmaker(sm)
    chat.set_chat_handoff_store(session_store)
    tctx = register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="Acme"),
        plaintext_tokens=["test-token"],
    )
    a = FastAPI()
    a.include_router(chat.router)
    a.dependency_overrides[get_db_session] = _session_override
    yield a, sm, llm, tctx
    chat.set_chatbot_factory(None)
    chat.set_chat_sessionmaker(None)
    chat.set_chat_handoff_store(None)
    set_tenant_resolver(None)
    await engine.dispose()


def _create_session(client: TestClient, **body) -> str:
    resp = client.post("/chat/sessions", json=body, headers=HEADERS)
    assert resp.status_code == 201, resp.text
    return resp.json()["session_id"]


async def _seed(sm, sid, pairs, tenant_id="t1"):
    """Persist a fabricated transcript directly (out-of-band of the agent),
    simulating an already-established conversation."""
    async with sm() as s:
        s.add(ChatSession(id=sid, tenant_id=tenant_id, language="en",
                          status="active", mode="ai", extra_data={}))
        for role, content in pairs:
            s.add(ChatMessage(session_id=sid, role=role, type="text", content=content))
        await s.commit()


def _roles_contents(messages) -> list[tuple[str, str]]:
    return [(m.role, m.content) for m in messages]


class _StubAgent:
    """Minimal stand-in exposing only what _hydrate_agent_history touches
    (agent.session), so the helper's per-row logic (merging, trailing-pop,
    empty-skip, idempotency guard) can be tested directly without spinning up
    a full ChatBotAgent + LLM + retriever stack."""

    def __init__(self, session_id: str) -> None:
        self.session = AgentSession(session_id=session_id)


# --- WS reconnect: the core regression ----------------------------------


async def test_ws_reconnect_replays_persisted_history(app) -> None:
    a, sm, llm, _tctx = app
    client = TestClient(a)
    sid = "cs_hydrate1"
    await _seed(sm, sid, [
        ("customer", "I want to deposit 500"),
        ("agent", "Sure, use the deposit tab"),
        ("customer", "it failed"),
        ("agent", "Which method did you use?"),
    ])

    with client.websocket_connect(f"/chat/ws/{sid}") as ws:
        ws.send_text(json.dumps({"type": "message", "text": "Ok sir"}))
        assert json.loads(ws.receive_text())["type"] == "typing"
        assert json.loads(ws.receive_text())["type"] == "message"

    assert llm.calls, "LLM was never called"
    last = llm.calls[-1]
    assert last[0].role == "system"
    assert _roles_contents(last[1:]) == [
        ("user", "I want to deposit 500"),
        ("assistant", "Sure, use the deposit tab"),
        ("user", "it failed"),
        ("assistant", "Which method did you use?"),
        ("user", "Ok sir"),
    ]


def test_first_connect_has_no_history(app) -> None:
    a, sm, llm, _tctx = app
    client = TestClient(a)
    sid = _create_session(client)

    with client.websocket_connect(f"/chat/ws/{sid}") as ws:
        ws.send_text(json.dumps({"type": "message", "text": "Plan B?"}))
        assert json.loads(ws.receive_text())["type"] == "typing"
        assert json.loads(ws.receive_text())["type"] == "message"

    assert llm.calls
    last = llm.calls[-1]
    assert len(last) == 2
    assert last[0].role == "system"
    assert last[1].role == "user"


async def test_hydration_capped_at_window(app) -> None:
    a, sm, llm, _tctx = app
    client = TestClient(a)
    sid = "cs_hydrate_cap"
    pairs = []
    for i in range(30):
        pairs.append(("customer", f"customer message {i}"))
        pairs.append(("agent", f"agent message {i}"))
    await _seed(sm, sid, pairs)

    with client.websocket_connect(f"/chat/ws/{sid}") as ws:
        ws.send_text(json.dumps({"type": "message", "text": "still there?"}))
        assert json.loads(ws.receive_text())["type"] == "typing"
        assert json.loads(ws.receive_text())["type"] == "message"

    last = llm.calls[-1]
    assert len(last) == 1 + 2 * MAX_HISTORY_TURNS + 1
    contents = [m.content for m in last]
    assert "customer message 0" not in contents


async def test_human_agent_messages_replay_as_assistant(app) -> None:
    a, sm, llm, _tctx = app
    client = TestClient(a)
    sid = "cs_hydrate_human"
    await _seed(sm, sid, [
        ("customer", "refund?"),
        ("human_agent", "I've raised ticket 8891 for you"),
        ("customer", "thanks"),
        ("agent", "Anything else?"),
    ])

    with client.websocket_connect(f"/chat/ws/{sid}") as ws:
        ws.send_text(json.dumps({"type": "message", "text": "one more thing"}))
        assert json.loads(ws.receive_text())["type"] == "typing"
        assert json.loads(ws.receive_text())["type"] == "message"

    last = llm.calls[-1]
    ticket_msgs = [m for m in last if "8891" in str(m.content)]
    assert ticket_msgs, "ticket line missing from replayed history"
    assert ticket_msgs[0].role == "assistant"


async def test_system_pushed_messages_replay_as_assistant(app) -> None:
    """push_async_message persists async-pushed messages (deposit verdicts,
    verification-timeout notices) with role="system". _HYDRATE_ROLES now
    includes "system" so these survive a reconnect, replayed as assistant
    turns -- same rationale as human_agent above: from the model's point of
    view, a system-pushed verdict is "what my side already said"."""
    a, sm, llm, _tctx = app
    client = TestClient(a)
    sid = "cs_hydrate_system"
    await _seed(sm, sid, [
        ("customer", "my deposit failed"),
        ("system", "Your deposit dispute was resolved: refund approved."),
        ("customer", "thank you"),
    ])

    with client.websocket_connect(f"/chat/ws/{sid}") as ws:
        ws.send_text(json.dumps({"type": "message", "text": "one more thing"}))
        assert json.loads(ws.receive_text())["type"] == "typing"
        assert json.loads(ws.receive_text())["type"] == "message"

    last = llm.calls[-1]
    verdict_msgs = [m for m in last if "refund approved" in str(m.content)]
    assert verdict_msgs, "system-pushed verdict missing from replayed history"
    assert verdict_msgs[0].role == "assistant"


async def test_window_never_opens_on_assistant_turn(app) -> None:
    a, sm, llm, _tctx = app
    client = TestClient(a)
    sid = "cs_hydrate_window_open"
    # 21 rows, oldest overall is customer, but the fetched window (last 20)
    # starts on an agent row -- forcing the cap to cut mid-pair.
    pairs = [
        ("customer" if i % 2 == 0 else "agent", f"msg {i}")
        for i in range(21)
    ]
    await _seed(sm, sid, pairs)

    with client.websocket_connect(f"/chat/ws/{sid}") as ws:
        ws.send_text(json.dumps({"type": "message", "text": "hi again"}))
        assert json.loads(ws.receive_text())["type"] == "typing"
        assert json.loads(ws.receive_text())["type"] == "message"

    last = llm.calls[-1]
    assert last[0].role == "system"

    # Genuinely verify the alternation invariant the test name promises --
    # not just that the prefix looks right, but that no two adjacent turns
    # anywhere in the sequence share a role (this would catch a regression
    # where the window-open guard or the same-role merge stopped firing).
    for prev, nxt in zip(last, last[1:]):
        assert prev.role != nxt.role, f"adjacent same-role turns: {prev.role!r}"

    replayed = last[1:-1]  # drop the system prefix and the just-sent "hi again"

    # msg 0 (oldest of all 21 seeded rows) is excluded by the LIMIT -- never
    # even fetched from the DB.
    assert "msg 0" not in [m.content for m in replayed]
    # msg 1 (agent) is the oldest row that *was* fetched, but it's an
    # assistant-role row with nothing before it, so the window-open guard
    # must drop it too.
    assert "msg 1" not in [m.content for m in replayed]
    # msg 2 (customer) is therefore the first genuinely replayed turn.
    assert replayed[0].role == "user"
    assert replayed[0].content == "msg 2"


async def test_external_message_carries_history(app) -> None:
    a, sm, llm, tctx = app
    sid = "cs_hydrate_external"
    await _seed(sm, sid, [
        ("customer", "what plans do you have"),
        ("agent", "Plan B has 500GB."),
    ])

    await chat.process_message(tctx, sid, "and plan C?")

    last = llm.calls[-1]
    contents = [m.content for m in last]
    assert "what plans do you have" in contents
    assert "Plan B has 500GB." in contents


def test_hydration_failure_does_not_break_the_socket(app, monkeypatch) -> None:
    a, sm, llm, _tctx = app
    client = TestClient(a)
    sid = _create_session(client)

    # The WS handler itself uses _sm() once up-front (to resolve the owning
    # tenant from the session row) before the hydration call this test is
    # targeting -- let that first call through with the real sessionmaker,
    # and only break the (hydration) call(s) after it.
    calls = {"n": 0}

    def _boom():
        calls["n"] += 1
        if calls["n"] == 1:
            return sm
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(chat, "_sm", _boom)

    with client.websocket_connect(f"/chat/ws/{sid}") as ws:
        ws.send_text(json.dumps({"type": "message", "text": "hello"}))
        assert json.loads(ws.receive_text())["type"] == "typing"
        reply = json.loads(ws.receive_text())
    assert reply["type"] == "message"


# --- Per-row replay logic, exercised directly against the helper ---------


async def test_consecutive_same_role_messages_merge(app) -> None:
    """Two consecutive human_agent rows (an agent sending a burst of lines)
    must merge into a single assistant turn, \\n-joined, not two adjacent
    assistant turns (which every LLM provider here rejects)."""
    a, sm, llm, tctx = app
    sid = "cs_hydrate_merge"
    await _seed(sm, sid, [
        ("customer", "issue A"),
        ("human_agent", "let me check"),
        ("human_agent", "here's your ticket 123"),
    ])

    stub = _StubAgent(sid)
    n = await chat._hydrate_agent_history(stub, sid, tctx.id)

    assert n == 2
    assert stub.session.turns == [
        LLMMessage(role="user", content="issue A"),
        LLMMessage(role="assistant", content="let me check\nhere's your ticket 123"),
    ]


async def test_trailing_unanswered_customer_message_is_dropped(app) -> None:
    """If the replay window's last row is a customer message with no reply
    yet, _hydrate_agent_history must drop it -- _compose appends the new
    incoming message right after, and leaving the stale one in would put two
    user turns back to back."""
    a, sm, llm, tctx = app
    sid = "cs_hydrate_trailing_pop"
    await _seed(sm, sid, [
        ("customer", "hello"),
        ("agent", "hi"),
        ("customer", "new question"),  # unanswered -- must be dropped
    ])

    stub = _StubAgent(sid)
    n = await chat._hydrate_agent_history(stub, sid, tctx.id)

    assert n == 2
    assert stub.session.turns == [
        LLMMessage(role="user", content="hello"),
        LLMMessage(role="assistant", content="hi"),
    ]


async def test_trailing_pop_can_empty_the_window(app) -> None:
    """A single unanswered customer message (nothing before it) pops down to
    zero replayed turns, not an error."""
    a, sm, llm, tctx = app
    sid = "cs_hydrate_trailing_pop_only"
    await _seed(sm, sid, [("customer", "are you there?")])

    stub = _StubAgent(sid)
    n = await chat._hydrate_agent_history(stub, sid, tctx.id)

    assert n == 0
    assert stub.session.turns == []


async def test_empty_or_whitespace_content_is_skipped(app) -> None:
    """Blank/whitespace-only rows (e.g. a placeholder row from a failed
    upload) must be skipped entirely -- not replayed as empty turns, and not
    merged into a neighbouring same-role turn either."""
    a, sm, llm, tctx = app
    sid = "cs_hydrate_blank_skip"
    await _seed(sm, sid, [
        ("customer", "   "),
        ("customer", ""),
        ("customer", "real question"),
        ("agent", "real answer"),
    ])

    stub = _StubAgent(sid)
    n = await chat._hydrate_agent_history(stub, sid, tctx.id)

    assert n == 2
    assert stub.session.turns == [
        LLMMessage(role="user", content="real question"),
        LLMMessage(role="assistant", content="real answer"),
    ]


async def test_idempotency_guard_skips_already_populated_session(app) -> None:
    """If agent.session.turns is already populated (e.g. request_call already
    loaded it on this same agent instance), hydration must be a no-op: return
    0 and leave the existing turns untouched, not duplicate/overwrite them."""
    a, sm, llm, tctx = app
    sid = "cs_hydrate_idempotent"
    await _seed(sm, sid, [
        ("customer", "from the DB, should not be loaded"),
        ("agent", "reply"),
    ])

    stub = _StubAgent(sid)
    existing = [LLMMessage(role="user", content="already here")]
    stub.session.turns.extend(existing)

    n = await chat._hydrate_agent_history(stub, sid, tctx.id)

    assert n == 0
    assert stub.session.turns == existing


# --- Tenant isolation (security regression) -------------------------------


async def test_hydration_is_scoped_to_tenant(app) -> None:
    """Regression test for the cross-tenant transcript leak: tenant A's
    session_id, replayed under tenant B's bearer token via POST
    /chat/message, must not surface any of tenant A's persisted history in
    the LLM call -- the join through ChatSession.tenant_id must degrade this
    to the same "no history" behaviour as a nonexistent session_id."""
    a, sm, llm, tctx_a = app
    client = TestClient(a)
    sid = "cs_tenant_a_secret"
    await _seed(sm, sid, [
        ("customer", "my card number is 4111111111111111"),
        ("agent", "Got it, let me help with that."),
    ], tenant_id="t1")

    register_tenant_for_test(
        TenantSettings(id="t2", slug="t2", name="Widgets"),
        plaintext_tokens=["t2-token"],
    )

    resp = client.post(
        "/chat/message",
        json={"session_id": sid, "message": "hi"},
        headers={"Authorization": "Bearer t2-token"},
    )
    assert resp.status_code == 200, resp.text

    assert llm.calls, "LLM was never called"
    last = llm.calls[-1]
    contents = " ".join(str(m.content) for m in last)
    assert "4111111111111111" not in contents
    assert "Got it, let me help" not in contents
    # Only the system prompt + the brand-new incoming message reached the
    # model -- no history at all was replayed.
    assert len(last) == 2
    assert last[0].role == "system"
    assert last[1].role == "user"
    assert last[1].content == "hi"


async def test_hydration_direct_call_scoped_to_tenant(app) -> None:
    """Same regression as above, exercised directly against the helper (not
    through an HTTP round-trip) so the assertion is pinned precisely to
    _hydrate_agent_history's own contract: wrong tenant_id -> zero turns."""
    a, sm, llm, tctx_a = app
    sid = "cs_tenant_a_direct"
    await _seed(sm, sid, [
        ("customer", "tenant A's private message"),
        ("agent", "tenant A's private reply"),
    ], tenant_id="t1")

    stub = _StubAgent(sid)
    n = await chat._hydrate_agent_history(stub, sid, "t2")  # wrong tenant

    assert n == 0
    assert stub.session.turns == []

    # Sanity check: the same session_id under the correct tenant_id does
    # replay -- proves the zero result above is the tenant filter, not a
    # broken query.
    stub_correct = _StubAgent(sid)
    n2 = await chat._hydrate_agent_history(stub_correct, sid, "t1")
    assert n2 == 2
