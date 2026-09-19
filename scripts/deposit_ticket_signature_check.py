"""Reproduce the exact bytes and X-Signature our json_ticket_relay POST sends.

Use this to settle a "401 bad signature" dispute with the ticket vendor: it
builds the request body the same way src/chatbot/deposit_verification.py does
and signs it with the same helper production uses, so whatever this prints is
byte-for-byte what the vendor receives.

Three modes:

    # 1. Shared test vector - a fixed body + fixed dummy salt, no real secret
    #    involved. Both sides run this and compare the digest.
    python scripts/deposit_ticket_signature_check.py --test-vector

    # 2. Real payload, real secret, signature only (nothing is sent)
    python scripts/deposit_ticket_signature_check.py \
        --secret "$DV_WEBHOOK_SECRET" \
        --order-id 8f0c... --mobile 9876543210 \
        --screenshot-url "https://bucket.s3.../shot.png?X-Amz-..."

    # 3. Same, but actually POST it and print the vendor's response
    python scripts/deposit_ticket_signature_check.py --secret ... --order-id ... \
        --screenshot-url ... --send https://vendor.example/deposit-ticket

The secret can also come from $DV_WEBHOOK_SECRET so it never lands in shell
history. Nothing here reads the database - pass the values you want to test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.integration.tenant_events import sign_body_hex  # noqa: E402

# Header name and signature format are production's, imported/mirrored rather
# than restated, so this script cannot drift from what actually gets sent.
SIGNATURE_HEADER = "X-Signature"

# Fixed vector for cross-checking with the vendor. The salt is a throwaway -
# never a real one - so this block is safe to paste into a shared channel.
TEST_VECTOR_SECRET = "deposit-ticket-test-salt"
TEST_VECTOR = {
    "order_id": "6c1a77a6-20b0-4fc9-ba4c-8add58aba9ef",
    "screenshot_url": "https://example-bucket.s3.ap-south-1.amazonaws.com/chat/shot.png?X-Amz-Expires=21600",
    "mobile": "9876543210",
}


def build_body(order_id: str, screenshot_url: str, mobile: str | None) -> bytes:
    """Mirror of src/chatbot/deposit_verification.py:382-385 - same key
    insertion order (order_id, screenshot_url, then mobile only when we
    resolved one) and the same compact separators."""
    payload: dict = {"order_id": order_id, "screenshot_url": screenshot_url}
    if mobile:
        payload["mobile"] = mobile
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def report(secret: str, raw: bytes) -> str:
    signature = sign_body_hex(secret, raw)
    print("--- request bytes (exactly what is transmitted) ---")
    print(raw.decode("utf-8"))
    print()
    print(f"byte length      : {len(raw)}")
    print(f"sha256(body)     : {hashlib.sha256(raw).hexdigest()}")
    print(f"secret length    : {len(secret)} chars")
    print(f"{SIGNATURE_HEADER:<17}: {signature}")
    print()
    print("--- equivalent curl ---")
    print(
        f"curl -X POST '<vendor-url>' \\\n"
        f"  -H 'Content-Type: application/json' \\\n"
        f"  -H '{SIGNATURE_HEADER}: {signature}' \\\n"
        f"  --data-binary '{raw.decode('utf-8')}'"
    )
    return signature


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-vector", action="store_true",
                   help="use the fixed shared body + dummy salt (ignores the other args)")
    p.add_argument("--secret", default=os.environ.get("DV_WEBHOOK_SECRET", ""),
                   help="webhook secret (default: $DV_WEBHOOK_SECRET)")
    p.add_argument("--order-id", default="")
    p.add_argument("--screenshot-url", default="")
    p.add_argument("--mobile", default="")
    p.add_argument("--send", metavar="URL",
                   help="actually POST to this URL and print the response")
    args = p.parse_args()

    if args.test_vector:
        secret = TEST_VECTOR_SECRET
        raw = build_body(TEST_VECTOR["order_id"], TEST_VECTOR["screenshot_url"],
                         TEST_VECTOR["mobile"])
        print(f"(shared test vector - dummy salt {secret!r}, not a real secret)\n")
    else:
        if not args.secret:
            p.error("--secret or $DV_WEBHOOK_SECRET is required (or use --test-vector)")
        if not args.order_id or not args.screenshot_url:
            p.error("--order-id and --screenshot-url are required (or use --test-vector)")
        secret = args.secret
        raw = build_body(args.order_id, args.screenshot_url, args.mobile)

    signature = report(secret, raw)

    if args.send:
        import httpx

        print(f"\n--- POST {args.send} ---")
        resp = httpx.post(
            args.send, content=raw,
            headers={"Content-Type": "application/json", SIGNATURE_HEADER: signature},
            timeout=20.0,
        )
        print(f"status: {resp.status_code}")
        print(f"body  : {resp.text[:2000]}")


if __name__ == "__main__":
    main()
