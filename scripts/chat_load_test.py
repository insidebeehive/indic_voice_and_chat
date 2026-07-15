"""Chat load-test harness: N parallel sessions, one message each, staged latencies.

For each of N virtual customers, concurrently:
  1. POST /api/v1/chat/sessions          (tenant bearer token)
  2. WS connect to the returned ws_url
  3. send one text message
  4. wait for the first "message" reply frame

Reports per-stage success counts, error breakdown, and p50/p95 latencies —
the repo previously had no way to reproduce the CRM team's stress test.

Usage:
  python scripts/chat_load_test.py --base-url http://localhost:8000 \
      --token <tenant-bearer-token> [--sessions 50] [--message "whats my balance"] \
      [--reply-timeout 90] [--ramp-seconds 0]

Notes:
  - ws_url in the create response is derived from the request host; --base-url
    is reused for the WS origin (http->ws, https->wss).
  - --ramp-seconds spreads session starts uniformly over that window
    (0 = all at once).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx
import websockets


@dataclass
class Result:
    created_s: float | None = None
    connected_s: float | None = None
    replied_s: float | None = None
    error: str | None = None
    frames: list[str] = field(default_factory=list)


def _pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))
    return values[idx]


async def run_one(
    client: httpx.AsyncClient,
    ws_base: str,
    token: str,
    message: str,
    reply_timeout: float,
    start_delay: float,
) -> Result:
    r = Result()
    if start_delay:
        await asyncio.sleep(start_delay)
    t0 = time.monotonic()
    try:
        resp = await client.post(
            "/api/v1/chat/sessions",
            json={"customer_name": "loadtest"},
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code != 201:
            r.error = f"create:{resp.status_code}"
            return r
        body = resp.json()
        session_id = body["session_id"]
        r.created_s = time.monotonic() - t0
    except Exception as e:  # noqa: BLE001 — report, don't crash the run
        r.error = f"create:{type(e).__name__}"
        return r

    ws_url = f"{ws_base}/api/v1/chat/ws/{session_id}"
    t1 = time.monotonic()
    try:
        async with websockets.connect(ws_url, open_timeout=30, close_timeout=5) as ws:
            r.connected_s = time.monotonic() - t1
            t2 = time.monotonic()
            await ws.send(json.dumps({"type": "message", "text": message}))
            deadline = t2 + reply_timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    r.error = "reply:timeout"
                    return r
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                frame = json.loads(raw)
                r.frames.append(frame.get("type", "?"))
                if frame.get("type") == "message":
                    r.replied_s = time.monotonic() - t2
                    return r
                if frame.get("type") == "error":
                    r.error = f"reply:error:{frame.get('message', '')[:60]}"
                    return r
    except Exception as e:  # noqa: BLE001
        r.error = f"ws:{type(e).__name__}"
        return r


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--token", required=True, help="tenant bearer token")
    ap.add_argument("--sessions", type=int, default=50)
    ap.add_argument("--message", default="what casino games do you have?")
    ap.add_argument("--reply-timeout", type=float, default=90.0)
    ap.add_argument("--ramp-seconds", type=float, default=0.0,
                    help="spread session starts over this window (0 = all at once)")
    args = ap.parse_args()

    parts = urlsplit(args.base_url)
    ws_scheme = "wss" if parts.scheme == "https" else "ws"
    ws_base = f"{ws_scheme}://{parts.netloc}"

    limits = httpx.Limits(max_connections=args.sessions + 10)
    async with httpx.AsyncClient(
        base_url=args.base_url, timeout=httpx.Timeout(60.0, connect=15.0), limits=limits,
    ) as client:
        t_start = time.monotonic()
        results = await asyncio.gather(*[
            run_one(client, ws_base, args.token, args.message, args.reply_timeout,
                    start_delay=random.uniform(0, args.ramp_seconds) if args.ramp_seconds else 0.0)
            for _ in range(args.sessions)
        ])
        wall = time.monotonic() - t_start

    created = [r for r in results if r.created_s is not None]
    connected = [r for r in results if r.connected_s is not None]
    replied = [r for r in results if r.replied_s is not None]
    errors: dict[str, int] = {}
    for r in results:
        if r.error:
            errors[r.error] = errors.get(r.error, 0) + 1

    def stage(name: str, vals: list[float], count: int) -> None:
        print(f"  {name:<10} {count}/{args.sessions}"
              + (f"   p50={_pct(vals, 50):.2f}s  p95={_pct(vals, 95):.2f}s"
                 f"  max={max(vals):.2f}s  mean={statistics.mean(vals):.2f}s" if vals else ""))

    print(f"\n=== chat load test: {args.sessions} sessions, wall {wall:.1f}s ===")
    stage("created", [r.created_s for r in created], len(created))
    stage("connected", [r.connected_s for r in connected], len(connected))
    stage("replied", [r.replied_s for r in replied], len(replied))
    if errors:
        print("  errors:")
        for k, v in sorted(errors.items(), key=lambda kv: -kv[1]):
            print(f"    {v:>4}  {k}")
    ok = len(replied) == args.sessions
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} ({len(replied)}/{args.sessions} full round-trips)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
