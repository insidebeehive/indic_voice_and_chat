# Sports Betting — Shared UI Behavior

## Architecture: sports is an embedded app

In every full-shell layout, the `/sports` page embeds a separate sportsbook (the shared `layout-sports` exchange app, or an external provider) — usually in an iframe, or mounted inline for newer operators. **The host layout has no native betslip**; all betting UI lives inside the embedded sports app. A one-time sports rules dialog may appear before first use.

Availability is operator-driven: matka layouts show sports only when `features.showSports` is enabled for that operator (adds a Matka/Sports switch and a Sports statements tab).

## Sports error states (identical copy)

- Maintenance: "Under Maintenance" / "The Sports Section is currently Under Maintenance and will be back soon..."
- "Sports Provider Not Configured" / "No sports provider is configured for this operator. Please contact support."
- "Sports not available"
- Loading: "Loading, Please wait..."

## Inside the exchange sports UI (shared across all operators using the in-house sportsbook)

Navigation: a horizontal sports bar (home, My Bets, one tab per sport with live/upcoming counts and an events popover with search — "🔥 Live Events" / "⏳ Upcoming Events").

### Placing a bet

1. Open a sport → event list shows "In Play"/upcoming events with Back/Lay odds cells; tap an event for its markets.
2. The event page shows a live scoreboard, market tabs (default "All"; fancy markets under "Fancy" with Yes/No runners), Back/Lay price ladders, and limits ("Min. Bet Limit", "Max. Bet Limit", "Max. Payout", "Exposure").
3. Tapping odds opens the **betslip** (bottom drawer, or inline under the market for some operators): BACK/LAY tag, stake amount with live profit/loss, preset stake buttons plus an "Other" keypad.
4. "Place Bet ({amount})" → "Please wait..." while placing. Blocked states: "Ball Running", "Market Suspended" (lock icon on the odds).

### Cashout

Markets with an open position show a "Cashout {amount}" badge → panel "Available cashout {amount}" with a checkbox "Accept at any change" and a "Cashout" button. The badge is hidden while an unmatched (pending) bet exists on the market.

### My Bets

Tabs "Open" / "Settled" / "Cancelled" / "Failed"; expandable bet cards show event/market/runner, Type/Odds/Stake, and the failure reason for failed bets. Empty state: "No Data found!". A desktop "Open Bets" side panel shows current open bets ("Open Bets not available!" when empty). Per-market "Bet History" tables are available from the event page.

### Sports maintenance (in-house sportsbook)

Full-page: "This website is temporarily unavailable" / "We are upgrading our site" / "We'll be back shortly."
