# Technical Troubleshooting — Shared UI Behavior

These states and messages are identical across layouts (shared app shell in `ui-toolkit`).

## Page not found

Unknown URLs render a 404 page: "Page not found" / "It seems you are lost? Use this path to get home!" with a "Go back home" button.

## Session expiry

1. Toast: "Your session has expired. Please login again."
2. Gated pages (e.g. wallet) without a session show: "Invalid User Session" / "Please login again" / "Please login again after logout. If problem persist, Please connect with support!"
3. Matka apps additionally re-lock with an MPIN dialog after ~2 minutes of inactivity (see `layouts/layout-2.md`).

## Maintenance

1. **Whole site**: a centralized maintenance guard replaces the app with a full-page maintenance screen (operator logo) when the operator is flagged for maintenance.
2. **Casino only**: "Under Maintenance" / "The Casino Section is Under Maintenance and will be back soon..."
3. **Sports only**: "The Sports Section is currently Under Maintenance and will be back soon..."

## Account suspended

"Account Suspended" / "Deposit/Withdraw temporarily not available" / "Please connect with support if you wish to activate your account!" — shown on wallet and game-launch attempts.

## Generic errors

- Error boundary: "Application Error" / "Oops, something unexpected happened!" / "Our team has been notified, and we are working on it!"
- Toasts: "Something went wrong!", "Some error occurred!", "Something went wrong, please try again later!"
- Rate limiting: "You are making too many requests within a short period."
- Toasts appear at the **bottom-center** of the screen and last ~3 seconds — users who miss one can be told where to look.

## App updates (PWA)

An update prompt appears when a new version ships: "Update Available" / "A new version of the app is available. Please update to get the latest features and improvements." with an "Update" button. If users report stale UI or odd behavior after a release, have them accept this prompt or hard-refresh.

## Demo-mode restrictions

Demo users see "You're in Demo Mode." and blocked actions explain themselves: "This action is disabled in Demo Mode. Register now to play for real." Deposits/withdrawals are demo-guarded.

## Support entry points

Every layout exposes a support/chat trigger (header button, bottom-nav item, menu row, or floating button — per layout). The label follows operator config: "Support" / "Help" / "Live Chat" / "WhatsApp" / "WhatsApp Chat".
