# Bonuses & Promotions — Shared UI Behavior

Bonus UI components differ per layout (tabs vs sections vs filterable grid — see layout docs for where the page lives: `/bonus` and/or `/promotion`), but the bonus lifecycle, dialogs, and copy are shared.

## Bonus lifecycle in the UI

1. **Available bonuses** list claimable offers: "Claim Now" / "Claim {amount}" (deposit bonuses may show "Participate"). Note shown: "* You can claim one bonus at a time!". Confirm dialog: "Claim Bonus!" — "Are you sure you want to claim this bonus?" → toast "Bonus claimed successfully!".
2. **Active bonus** cards show wagering progress: "Deposit: {amount}", WR fields (Total / Fulfilled / Remaining WR), "Free Spins", "Expire At", and statuses like "In Progress" / "Completed" / "Available to Claim".
3. **Forfeit**: "Forfeit Bonus!" — "Are you sure you want to Forfeit your bonus?" / "If you Forfeit, {amount} will be forfeited." → toast "Bonus forfeited successfully!".
4. **Cashout**: "Cashout Bonus" — "Are you sure you want to cashout your bonus?" → toast "Cashed out successfully!". Auto-cashout note where applicable: "Auto cashout will be triggered when balance reaches {threshold}".
5. **Bonus history** tab/section lists past bonuses.

## Shared states & copy

- Empty: "No ACTIVE bonus found!", "No bonus history found!", "No bonuses found!", "No casino bonuses available!".
- Operator-disabled: "The bonuses are temporarily not available to claim." / "Please try again later.".
- Error: "Some error occurred while claiming the bonus, Please try after sometime!".
- While a bonus wallet is active, the header balance shows the bonus label/amount alongside "Real".

## Bonus vs withdrawal

An active deposit-rolling bonus locks part of the balance; attempting to withdraw opens the forfeit dialog ("Cancel bonus and Withdraw") — see `04-withdrawals.md`.

## Rewards points (layout-1 only)

Layout 1 additionally has a Rewards page (redeem reward points into bonus) — see `layouts/layout-1.md`.
