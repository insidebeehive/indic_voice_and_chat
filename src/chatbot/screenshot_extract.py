"""Vision extraction for the deposit-verification pre-submission cross-check
(see ``src/chatbot/deposit_verification.py``'s ``_cross_check_screenshot``).

A SEPARATE Gemini Flash vision call reads the customer's deposit screenshot
and extracts a small, structured fact — ``{readable, amount, currency,
date}`` — that the cross-check compares against the tenant's CRM. The chat
model itself is deliberately NOT trusted for this: it is prompted to decide
what to tell the customer, not to be a reliable OCR step, so this call runs
independently of (and before) anything the chat model sees.

``make_gemini_screenshot_extractor`` binds a ``ScreenshotExtractor`` to
whatever LLM provider instance the caller hands it — in practice
``registry.providers.get_platform_llm()`` (see ``src/bootstrap.py``), the
same platform-default client every chat turn runs on, so this never
hardcodes a model name or API key. Tests stub ``ScreenshotExtractor``
directly rather than going through this factory.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Awaitable, Callable, Optional

from src.interfaces.llm import ContentPart, ILLMProvider, LLMConfig, LLMMessage
from src.utils.logging import debug_event

log = logging.getLogger(__name__)


class UnparseableExtractionResponse(Exception):
    """Raised by ``make_gemini_screenshot_extractor``'s wrapper when the
    model's reply can't be parsed as JSON at all -- an empty or
    safety-blocked response, non-JSON text, or a JSON value that isn't an
    object (see ``parse_extraction_response``, which returns ``None`` for
    all of these). This is distinct from a reply that parsed fine but
    reported ``readable: false`` (or an untrustworthy shape) -- that case
    stays a normal return value, not an exception, because it's a real
    judgement about the screenshot, not a failure to get one.

    The pre-submission cross-check (``_cross_check_screenshot`` in
    ``src/chatbot/deposit_verification.py``) catches this the same as any
    other extractor failure and reports ``could_not_check`` -- never
    ``screenshot_unreadable``, since the screenshot was never actually read.
    """

# Approved wording (verbatim) — do not edit without re-confirming with the
# user, same rule as src/dialogue/prompts.py even though this file sits
# outside that package.
EXTRACTION_PROMPT = (
    "You are reading a payment screenshot a customer sent as proof of a deposit.\n"
    "Return JSON only:\n"
    "{\"readable\": bool, \"amount\": number|null, \"currency\": string|null, \"date\": \"YYYY-MM-DD\"|null}\n"
    "- amount: the amount actually paid in this payment. Not a balance, fee, or\n"
    "  cashback. No currency symbols or commas.\n"
    "- date: the date of this payment as shown on the screenshot.\n"
    "- If the amount or the date is not clearly visible, set readable=false.\n"
    "  Never guess.\n"
)

# ScreenshotExtractor(data, mime) -> {"readable": bool, "amount": float | None,
# "currency": str | None, "date": str | None, "usage": dict, "provider": str,
# "model": str}. "usage" mirrors LLMResult.usage's prompt_tokens/
# completion_tokens/cached_tokens shape (src/interfaces/llm.py) so callers can
# feed it straight into src/api/chat_cost.py's compute_chat_turn_cost.
ScreenshotExtractor = Callable[[bytes, str], Awaitable[dict]]

_UNREADABLE: dict = {"readable": False, "amount": None, "currency": None, "date": None}

# Strips a leading/trailing ``` or ```json fence the model sometimes wraps
# its JSON in despite "Return JSON only" — matches at the very start/end of
# the (already-stripped) text only, so a code fence appearing inside a
# string value is left alone.
_FENCE_RE = re.compile(r"\A```(?:json)?\s*|\s*```\Z", re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", (text or "").strip()).strip()


def parse_extraction_response(text: str) -> Optional[dict]:
    """Defensively parse the extraction model's reply.

    Never raises, but distinguishes two different kinds of failure:

    - **Couldn't parse at all** -- an empty/safety-blocked response, non-JSON
      garbage text, or a JSON value that isn't an object -- returns ``None``.
      This is a system/infrastructure failure (the reading was never
      obtained), not a judgement about the screenshot, so callers must NOT
      treat it the same as ``readable: false``. ``make_gemini_screenshot_extractor``
      turns a ``None`` here into ``UnparseableExtractionResponse``.
    - **Parsed, but unreadable or untrustworthy** -- valid JSON object, but
      ``readable`` is false, or ``amount``/``date`` are missing/wrong-typed
      -- returns the same ``_UNREADABLE`` shape regardless of which of those
      applied, since the cross-check can't use a reading it can't trust the
      shape of.
    """
    cleaned = _strip_code_fences(text)
    if not cleaned:
        return None
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    amount = data.get("amount")
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        amount = None
    date = data.get("date")
    if not isinstance(date, str) or not date.strip():
        date = None
    else:
        date = date.strip()
    currency = data.get("currency")
    if not isinstance(currency, str) or not currency.strip():
        currency = None

    readable = bool(data.get("readable")) and amount is not None and date is not None
    return {
        "readable": readable,
        "amount": amount if readable else None,
        "currency": currency,
        "date": date if readable else None,
    }


def make_gemini_screenshot_extractor(
    llm: ILLMProvider, *, provider_name: str, model: str,
) -> ScreenshotExtractor:
    """Build a ``ScreenshotExtractor`` bound to ``llm`` (the platform-default
    LLM client). ``provider_name``/``model`` are stamped onto every result
    for cost-recording — pass the same identity used elsewhere for this
    client (``registry.providers.global_defaults["llm"]`` in
    ``src/bootstrap.py``) so a cost lookup resolves against the right
    ``ProviderCost`` row.
    """

    async def extract(data: bytes, mime: str) -> dict:
        messages = [
            LLMMessage(
                role="user",
                content=[
                    ContentPart(type="text", text=EXTRACTION_PROMPT),
                    ContentPart(type="image", inline_data={"mime_type": mime, "data": data}),
                ],
            )
        ]
        # Low temperature: this is an extraction task, not a creative one —
        # the model should report what it sees, not vary its reading run to
        # run. response_format="json" asks Gemini for a clean JSON body
        # (response_mime_type=application/json); parse_extraction_response
        # still defends against a stray code fence or malformed body, since
        # that request-side hint is not a guarantee.
        config = LLMConfig(temperature=0.0, max_tokens=256, response_format="json")
        result = await llm.generate(messages, config)
        parsed = parse_extraction_response(result.text)
        if parsed is None:
            debug_event(
                log, "screenshot_extract parse_failed", provider=provider_name, model=model,
                raw_text_len=len(result.text or ""),
            )
            raise UnparseableExtractionResponse(
                f"screenshot extraction response could not be parsed as JSON "
                f"(provider={provider_name!r}, model={model!r})"
            )
        parsed["usage"] = result.usage or {}
        parsed["provider"] = provider_name
        parsed["model"] = model
        debug_event(
            log, "screenshot_extract result", provider=provider_name, model=model,
            readable=parsed["readable"], amount=parsed["amount"], date=parsed["date"],
            currency=parsed["currency"], usage=parsed["usage"],
        )
        return parsed

    return extract
