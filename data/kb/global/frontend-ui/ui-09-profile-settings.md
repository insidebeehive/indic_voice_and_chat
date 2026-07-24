# Profile, Account Settings & Statements — Shared UI Behavior

Where these screens live differs per layout (profile hub page, avatar menu, drawer, or sidebar — see layout docs). The behaviors below are shared.

## Profile data

1. First and last name are editable ("Profile Update"; toast "{type} updated successfully!").
2. **Email, mobile number, and username are read-only** — marked "(Read only)". Users cannot change their registered contact identity in the UI; that requires operator support.

## Password change

1. Available post-login (account settings card or a dedicated "Update Password" / "Set New Password" page).
2. Fields: new password + confirm password; requires OTP verification.
3. Validation: "Password should be at least 6 characters long", "Password and Confirm password must be same".

## Bank & address

Bank details (OTP + admin approval — see `04-withdrawals.md`) and, on casino layouts, an address form live in account settings.

## Statements / transaction history

1. Tabs per product: Wallet / Casino / Sports / Matka (or Teer) — the visible set depends on the operator's products.
2. Date-range filtering with quick presets ("Today" … "Last 45 days"). Range limits are enforced in-UI: selections are limited to the last 45 days ("You can only select dates within the last 45 days."); some layouts restrict the span to a two-day window ("You can only select dates within a two-day range from the selected 'From date'.").
3. Rows drill into details ("Round Details", "Bet Details", "Market Bet Details").
4. The Wallet tab supports cancelling pending withdrawals ("Cancel Withdraw").
5. Empty states: "No Transactions", "No WALLET/CASINO/SPORTS/MATKA transactions found for selected date range!".

## Language & theme

1. A language switcher is available in the header/menu/drawer on every layout.
2. Some layouts include a theme switcher (toast "Theme changed"); default theme is operator-configured.

## Logout

A "Logout" action is always present (user menu, sidebar, profile page, or drawer per layout); matka layouts confirm before logging out.
