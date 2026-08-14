from __future__ import annotations

from src.dialogue.response_parser import ChatBotResponse
from src.interfaces.vector_store import Document
from src.rag.context_builder import (
    GuardConfig,
    apply_hallucination_guard,
    build_rag_context,
    build_voicebot_kb_context,
)
from src.rag.retriever import RetrievedChunk


class _FakeRetriever:
    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    def list_all(self, max_chunks: int = 200) -> list[Document]:
        return self._docs


def _chunk(doc_id: str, content: str, **md) -> RetrievedChunk:
    return RetrievedChunk(
        document=Document(id=doc_id, content=content, metadata=md),
        score=0.9,
    )


# --- Context builder ---------------------------------------------------


def test_build_rag_context_empty_returns_marker() -> None:
    out = build_rag_context([])
    assert "no relevant sources" in out.text
    assert out.source_tags == []
    assert out.chunk_count == 0


def test_build_rag_context_uses_filename_section_tag() -> None:
    out = build_rag_context([
        _chunk("c1", "Plan B has 500GB.", filename="plans.pdf", page=2),
        _chunk("c2", "Plan A has 100GB.", filename="plans.pdf", page=1),
    ])
    assert "plans.pdf:2" in out.text
    assert "plans.pdf:1" in out.text
    assert out.source_tags == ["plans.pdf:2", "plans.pdf:1"]
    assert out.chunk_count == 2


def test_build_rag_context_falls_back_to_id_when_no_metadata() -> None:
    out = build_rag_context([_chunk("doc-42", "content")])
    assert "doc-42" in out.source_tags


# --- Voicebot KB context (static, one-shot, priority-ordered) ----------


def _doc(filename: str, content: str) -> Document:
    return Document(id=filename, content=content, metadata={"filename": filename})


async def test_voicebot_kb_context_prioritizes_product_docs_over_filename_order() -> None:
    # This is the exact bug: casino-games (06) used to lose out to earlier
    # filename-sorted docs (01-05) crowding the char budget before it was
    # ever reached, even though it's core sales-call content.
    docs = [
        _doc("01-account-registration-login.md", "x" * 100),
        _doc("06-casino-games.md", "Casino games include slots, live dealer..."),
    ]
    retriever = _FakeRetriever(docs)
    ctx = await build_voicebot_kb_context([retriever], max_chars=1000)
    assert "casino-games" in ctx
    assert "account-registration-login" in ctx
    # Casino comes first despite registration-login sorting first by filename.
    assert ctx.index("06-casino-games") < ctx.index("01-account-registration-login")


async def test_voicebot_kb_context_excludes_technical_help() -> None:
    docs = [_doc("12-technical-help.md", "Troubleshooting steps...")]
    retriever = _FakeRetriever(docs)
    ctx = await build_voicebot_kb_context([retriever])
    assert ctx == ""


async def test_voicebot_kb_context_includes_unranked_docs_after_priority_list() -> None:
    # A future KB doc not in the curated priority list must still be
    # included (not silently dropped), just ranked after known-priority docs.
    docs = [
        _doc("99-new-feature.md", "Something new."),
        _doc("06-casino-games.md", "Casino content."),
    ]
    retriever = _FakeRetriever(docs)
    ctx = await build_voicebot_kb_context([retriever])
    assert "99-new-feature" in ctx
    assert ctx.index("06-casino-games") < ctx.index("99-new-feature")


async def test_voicebot_kb_context_default_cap_fits_all_tier_one_product_docs() -> None:
    # Regression guard for the actual reported bug: with the real KB doc
    # sizes, all Tier-1 product docs (casino/sports/matka/bonuses) must fit
    # under the default cap, not just the first couple by filename order.
    tier1 = [
        "06-casino-games.md", "07-sports-betting.md",
        "08-matka-lottery-games.md", "09-bonuses-and-promotions.md",
    ]
    docs = [_doc(fn, "y" * 3000) for fn in tier1]  # ~12k, close to real doc sizes
    retriever = _FakeRetriever(docs)
    ctx = await build_voicebot_kb_context([retriever])
    for fn in tier1:
        assert fn in ctx


async def test_voicebot_kb_context_reads_persistent_store_when_bm25_cold(tmp_faiss_index) -> None:
    """Simulates process A (ingest happened here) vs process B (cold BM25, e.g.
    after a restart or on a different worker) sharing the same persistent
    FAISS store. Pins the real bug: HybridRetriever.list_all() only sees
    chunks indexed by THIS process's in-memory BM25, so a freshly-built
    retriever in another process saw nothing — build_voicebot_kb_context must
    fall back to the persistent store via list_all_persistent()."""
    from src.providers.vector_store.faiss_store import FAISSAdapter
    from src.rag.embeddings import HashEmbedder
    from src.rag.retriever import HybridRetriever

    warm = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index}),
    )
    await warm.index([
        Document(
            id="layout_casino::chunk-0",
            content="Casino games include slots and live dealer.",
            metadata={"filename": "06-casino-games.md", "section": 0},
        )
    ])

    cold = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index}),
    )
    assert cold.list_all() == []  # pins the bug's precondition — must stay true
    ctx = await build_voicebot_kb_context([cold])
    assert "Casino games include slots" in ctx


def test_build_rag_context_truncates_at_max_chars() -> None:
    big = "X" * 1000
    chunks = [_chunk(f"c{i}", big, filename=f"f{i}.md") for i in range(10)]
    out = build_rag_context(chunks, max_chars=2500)
    # Should have included at most ~3 chunks before hitting the budget
    assert out.chunk_count <= 4
    assert len(out.text) <= 4500  # block headers add some overhead


# --- Hallucination guard -----------------------------------------------


def test_guard_passes_through_clean_response() -> None:
    rag = build_rag_context([_chunk("c1", "Plan B has 500GB.", filename="plans.pdf", page=2)])
    response = ChatBotResponse(
        response_text="Plan B has 500GB.",
        language="en",
        sources_used=["plans.pdf:2"],
        confidence="high",
        action="none",
    )
    out = apply_hallucination_guard(response, rag)
    assert out.response_text == "Plan B has 500GB."
    assert out.sources_used == ["plans.pdf:2"]
    assert out.confidence == "high"


def test_guard_strips_unsupported_citations_and_downgrades() -> None:
    rag = build_rag_context([_chunk("c1", "Plan B has 500GB.", filename="plans.pdf", page=2)])
    response = ChatBotResponse(
        response_text="Plan B has 500GB and supports 5G.",
        language="en",
        sources_used=["plans.pdf:2", "wireless.pdf:7"],  # second is invented
        confidence="high",
    )
    out = apply_hallucination_guard(response, rag)
    assert out.sources_used == ["plans.pdf:2"]
    assert out.confidence == "low"


def test_guard_no_retrieval_returns_fallback_in_english() -> None:
    rag = build_rag_context([])
    response = ChatBotResponse(
        response_text="Yes, the answer is 42.",
        language="en",
        confidence="high",
    )
    out = apply_hallucination_guard(response, rag)
    assert "not able to find" in out.response_text.lower()
    assert out.confidence == "low"
    assert out.sources_used == []


def test_guard_no_retrieval_returns_fallback_in_hindi() -> None:
    rag = build_rag_context([])
    response = ChatBotResponse(
        response_text="Plan B mein 500GB data hai.",
        language="hi",
        confidence="high",
    )
    out = apply_hallucination_guard(response, rag)
    assert "documentation mein nahi mil raha" in out.response_text.lower()
    assert out.confidence == "low"


def test_guard_no_retrieval_empty_response_unchanged() -> None:
    rag = build_rag_context([])
    response = ChatBotResponse(response_text="", language="en", confidence="medium")
    out = apply_hallucination_guard(response, rag)
    # No fallback substituted because there was nothing to override.
    assert out.response_text == ""
    assert out.confidence == "low"


def test_guard_does_not_mutate_input() -> None:
    rag = build_rag_context([_chunk("c1", "x", filename="f.md")])
    response = ChatBotResponse(
        response_text="x",
        language="en",
        sources_used=["INVALID"],
        confidence="high",
    )
    apply_hallucination_guard(response, rag)
    assert response.sources_used == ["INVALID"]
    assert response.confidence == "high"


def test_guard_low_confidence_with_no_sources_passes_through() -> None:
    rag = build_rag_context([_chunk("c1", "x", filename="f.md")])
    response = ChatBotResponse(
        response_text="I'm not sure",
        language="en",
        sources_used=[],
        confidence="low",
    )
    out = apply_hallucination_guard(response, rag)
    assert out.response_text == "I'm not sure"
    assert out.confidence == "low"
