# Layout Sports (+ layout-sports-1) — Deltas from Global Baseline

These packages ARE the in-house exchange sportsbook that other layouts embed on their `/sports` page. They are **not operator shells**: no appConfigs, no login/register, no wallet, no KYC, no static pages — all of that belongs to the host operator app. The betting UI itself is documented in `global/06-sports.md`.

## layout-sports (token-based iframe host)

1. Consumed only by `apps/layout-sports-app` — a **single shared deployment serving many operators**. Operator identity, theme, and limits come from a session token minted via `/externalauth`; every route lives under `/t/:tokenId/…` (`/s` landing, `/s/:sportId`, `/e/:sportId/:eventId`, `/mybets`).
2. No app header/footer — it renders inside the host app's iframe. Chrome is the sports bar (home, My Bets, per-sport tabs with live/upcoming counts and an events popover).
3. Invalid/expired token → "403 - Access Denied" page. If a user reports this inside an operator app's sports section, the host session/token is the problem — re-login in the host app.
4. Maintenance page: "This website is temporarily unavailable" / "We are upgrading our site" / "We'll be back shortly."

## layout-sports-1 (embeddable inline fork)

Same betting UI, restructured to mount **inside a host app's own routes** (`/sports`, `/sports/s/:sportId`, `/sports/e/:sportId/:eventId`, `/sports/mybets`) instead of a token URL:

1. Dual render mode per operator: `inline` (native sports UI inside the host chrome — host header/footer stay visible) or `iframe` fallback for operators on external sports providers. Configured per host app (`sportsRenderMode`), not via appConfigs.
2. Per-operator sports theme CSS (scoped `.sp-<operator>` classes) so the sports subtree matches the host brand.
3. Betslip placement (`betPlacementMode`): bottom drawer or inline under the market row — defaults to inline in inline mode.
4. The cashout badge is hidden while any unmatched (pending) bet exists on the market.
5. Event/bet times display in the operator's timezone.
6. Session-restore: refreshing reopens the previously open event.

> Note: layout-sports-1 source currently lives on a worktree branch (`86eyaj34p-architecture-review`), not yet merged to main — treat its behaviors as in-rollout.

## Support triage rule

For sports questions, first determine the surface: money/account questions (deposits, balances, limits shown in the host header) belong to the **host layout's** docs; in-play betting questions (odds, betslip, "Ball Running", "Market Suspended", cashout, My Bets) belong to the shared sports UI (`global/06-sports.md`).
