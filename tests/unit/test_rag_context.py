from __future__ import annotations

from src.dialogue.response_parser import ChatBotResponse
from src.interfaces.vector_store import Document
from src.rag.context_builder import (
    GuardConfig,
    apply_hallucination_guard,
    apply_no_grounding_guard,
    apply_unverified_data_guard,
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


# --- No-grounding guard --------------------------------------------------


def test_no_grounding_guard_fires_on_ungrounded_high_confidence_time_claim() -> None:
    response = ChatBotResponse(
        response_text="It's 11:24 PM right now.",
        language="en",
        confidence="high",
    )
    out = apply_no_grounding_guard(response, retrieved_any=False, tool_calls_made=[])
    assert out.confidence == "low"
    # Text is NOT rewritten — only confidence is downgraded (documented as a
    # deliberate choice: a lexically-detected risk pattern isn't proof of
    # fabrication, just a downgrade-and-log signal).
    assert out.response_text == "It's 11:24 PM right now."


def test_no_grounding_guard_fires_on_ungrounded_currency_figure() -> None:
    response = ChatBotResponse(
        response_text="The minimum bet is ₹10 and the maximum is ₹10,000.",
        language="en",
        confidence="high",
    )
    out = apply_no_grounding_guard(response, retrieved_any=False, tool_calls_made=[])
    assert out.confidence == "low"


def test_no_grounding_guard_skips_when_a_tool_was_called() -> None:
    response = ChatBotResponse(
        response_text="It's 11:24 PM right now.",
        language="en",
        confidence="high",
    )
    out = apply_no_grounding_guard(response, retrieved_any=False, tool_calls_made=["get_game"])
    assert out.confidence == "high"
    assert out is response  # unchanged, passed through as-is


# NOTE: retrieved_any=True is not reachable via the current chatbot.py call site
# (the guard is only invoked when retrieved_all is empty) — this test covers the
# function's own documented contract, not a currently-reachable production path.
def test_no_grounding_guard_skips_when_retrieval_happened() -> None:
    response = ChatBotResponse(
        response_text="It's 11:24 PM right now.",
        language="en",
        confidence="high",
    )
    out = apply_no_grounding_guard(response, retrieved_any=True, tool_calls_made=[])
    assert out.confidence == "high"


def test_no_grounding_guard_skips_low_confidence() -> None:
    response = ChatBotResponse(
        response_text="It's 11:24 PM right now.",
        language="en",
        confidence="low",
    )
    out = apply_no_grounding_guard(response, retrieved_any=False, tool_calls_made=[])
    assert out.confidence == "low"


def test_no_grounding_guard_passes_through_unrelated_text() -> None:
    response = ChatBotResponse(
        response_text="KYC usually takes 24-48 hours to review.",
        language="en",
        confidence="high",
    )
    out = apply_no_grounding_guard(response, retrieved_any=False, tool_calls_made=[])
    assert out.confidence == "high"


# --- Unverified-data guard (Step 5, ticket #1762) -------------------------


def test_unverified_data_guard_regression_ticket_1762() -> None:
    """The literal regression case: the bot claimed 'the only pending
    withdrawal amount showing is ₹8,100' with zero grounding (every
    get_player_transactions call had timed out), while the customer's real,
    documented ₹19,600 withdrawal went unmentioned. This must be replaced,
    and the fallback must name the customer's own real figure."""
    response = ChatBotResponse(
        response_text=(
            "According to our system records, the only pending withdrawal amount "
            "showing on your account is ₹8,100."
        ),
        language="en",
        confidence="high",
    )
    out = apply_unverified_data_guard(
        response,
        grounded_text="",  # every tool call failed; nothing was retrieved
        customer_text="I have a ₹19,600 withdrawal that's not showing.",
    )
    assert "8,100" not in out.response_text and "8100" not in out.response_text
    assert "19,600" in out.response_text
    assert out.confidence == "low"


def test_unverified_data_guard_passes_grounded_figure() -> None:
    """A figure that genuinely appears in grounded text (a real tool result,
    RAG content, or the customer's own words) must survive untouched — false
    positives here are as bad as the original bug."""
    response = ChatBotResponse(
        response_text="I can see your ₹19,600 withdrawal is showing as SUBMITTED.",
        language="en",
        confidence="high",
    )
    out = apply_unverified_data_guard(
        response,
        grounded_text='{"amount": 19600, "status": "SUBMITTED"}',
        customer_text="",
    )
    assert out.response_text == response.response_text
    assert out.confidence == "high"


def test_unverified_data_guard_customer_own_figure_survives_even_unquoted_elsewhere() -> None:
    """The customer's own real number, restated back to them, must remain
    sayable even when no tool succeeded — it's grounded via customer_text/
    grounded_text (their own prior words), not via a tool result."""
    response = ChatBotResponse(
        response_text="I can see you mentioned a ₹19,600 withdrawal.",
        language="en",
        confidence="high",
    )
    out = apply_unverified_data_guard(
        response,
        grounded_text="I have a ₹19,600 withdrawal that's not showing.",
        customer_text="I have a ₹19,600 withdrawal that's not showing.",
    )
    assert out.response_text == response.response_text


def test_unverified_data_guard_no_currency_figure_passes_through() -> None:
    out = apply_unverified_data_guard(
        ChatBotResponse(response_text="Let me check that for you.", confidence="high"),
        grounded_text="",
        customer_text="",
    )
    assert out.response_text == "Let me check that for you."


def test_unverified_data_guard_empty_reply_passes_through() -> None:
    response = ChatBotResponse(response_text="", confidence="low")
    out = apply_unverified_data_guard(response, grounded_text="", customer_text="")
    assert out is response  # short-circuit, no copy needed


def test_unverified_data_guard_generic_fallback_when_no_disputed_figure() -> None:
    """No customer-side figure to name -> generic (not fabricated) fallback."""
    response = ChatBotResponse(response_text="Your balance is ₹500.", confidence="high")
    out = apply_unverified_data_guard(response, grounded_text="", customer_text="")
    assert "500" not in out.response_text
    assert "I'm not able to verify" in out.response_text


def test_unverified_data_guard_partial_number_is_not_a_false_match() -> None:
    """A grounded '19600' must not spuriously 'ground' an unrelated reply
    figure like '600' via naive substring matching -- token comparison must
    be on the FULL numeric token, not a substring."""
    response = ChatBotResponse(response_text="Your fee is ₹600.", confidence="high")
    out = apply_unverified_data_guard(
        response, grounded_text='{"amount": 19600}', customer_text="",
    )
    assert out.response_text != response.response_text
    assert out.confidence == "low"


def test_unverified_data_guard_decimal_figure_from_real_wallet_payload_survives() -> None:
    """Review Fix 1 regression: docs/crm-api-contract.md's real wallet
    payload shape uses decimal amounts (e.g. real_balance: 4250.75,
    bonus_balance: 500.00). A reply that states these exact figures must NOT
    be wrongly blocked just because the reply-side and grounded-text-side
    number-extraction regexes used to normalize decimals differently
    (₹4,250.75 -> '4250' vs a grounded '4250.75' -> no match)."""
    response = ChatBotResponse(
        response_text="Your real balance is ₹4,250.75 and your bonus balance is ₹500.00.",
        language="en",
        confidence="high",
    )
    out = apply_unverified_data_guard(
        response,
        grounded_text='{"real_balance": 4250.75, "bonus_balance": 500.00}',
        customer_text="",
    )
    assert out.response_text == response.response_text
    assert out.confidence == "high"


def test_unverified_data_guard_decimal_and_integer_forms_of_same_value_match() -> None:
    """A grounded whole-number amount (e.g. a real 'amount': 1000) must still
    ground a reply that states it with a trailing '.00' (₹1,000.00), and
    vice versa -- both sides must normalize to the same canonical value."""
    response = ChatBotResponse(response_text="Your deposit of ₹1,000.00 was received.",
                                confidence="high")
    out = apply_unverified_data_guard(
        response, grounded_text='{"amount": 1000}', customer_text="",
    )
    assert out.response_text == response.response_text
    assert out.confidence == "high"


def test_unverified_data_guard_rounded_decimal_survives() -> None:
    """Round-3 review finding: a reply that rounds a real decimal balance to
    the nearest rupee ("about ₹4,250" for a real ₹4,250.75) must NOT be
    blocked -- this is the same false-positive class as Fix 1, one step
    removed. Covers both truncation (4250.75 -> 4250) and round-up
    (4750.75 -> 4751 is NOT what the reply says here, but the whole-number
    truncation 4750 must still ground)."""
    response = ChatBotResponse(
        response_text="You have about ₹4,250 in your wallet, and ₹4,750 available in total.",
        language="en",
        confidence="high",
    )
    out = apply_unverified_data_guard(
        response,
        grounded_text='{"real_balance": 4250.75, "total_available": 4750.75}',
        customer_text="",
    )
    assert out.response_text == response.response_text
    assert out.confidence == "high"


def test_unverified_data_guard_rounding_cannot_ground_an_unrelated_figure() -> None:
    """The rounding widening only ADDS forms derived from an already-grounded
    value -- it must never make an unrelated fabricated figure (e.g. the
    original ₹8,100 incident) pass just because some other grounded decimal
    happens to round near it."""
    response = ChatBotResponse(
        response_text="The only pending withdrawal amount showing is ₹8,100.",
        language="en",
        confidence="high",
    )
    out = apply_unverified_data_guard(
        response,
        grounded_text='{"real_balance": 4250.75, "total_available": 4750.75}',
        customer_text="I reported a ₹19,600 withdrawal",
    )
    assert "8,100" not in out.response_text and "8100" not in out.response_text
    assert out.confidence == "low"


def test_unverified_data_guard_hindi_fallback_used_for_hindi_response() -> None:
    """Review Fix 4 regression: the fallback must be language-selected like
    the sibling guards (apply_hallucination_guard/apply_no_grounding_guard),
    not hardcoded English."""
    response = ChatBotResponse(response_text="Aapka balance ₹8,100 hai.",
                                language="hi", confidence="high")
    out = apply_unverified_data_guard(response, grounded_text="", customer_text="")
    assert out.language == "hi"
    assert "8,100" not in out.response_text and "8100" not in out.response_text
    # Not the English fallback text.
    assert "I'm not able to verify" not in out.response_text


def test_unverified_data_guard_fallback_is_an_offer_not_a_promise() -> None:
    """Review Fix 5 regression: the fallback must OFFER a handoff and require
    confirmation (consistent with the existing ESCALATION prompt pattern),
    not declare one is already happening."""
    response = ChatBotResponse(response_text="Your balance is ₹8,100.", confidence="high")
    out = apply_unverified_data_guard(response, grounded_text="", customer_text="")
    assert "would you like" in out.response_text.lower()
    assert "let me connect you" not in out.response_text.lower()
