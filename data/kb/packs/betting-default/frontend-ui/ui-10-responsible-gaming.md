# Responsible Gaming & Legal Pages — Shared UI Behavior

## Self-exclusion (identical form everywhere)

Available only when the operator is self-exclusion eligible (`operatorDetails.isSelfExclusionEligible`); otherwise the entry card and form do not render.

1. Entry: a "Responsible Gaming" card in profile/account settings — "Take a break by self-excluding from your account for up to 365 days." with a "Set up self-exclusion" link.
2. The form warns: "This action is irreversible." / "Once submitted, you will be logged out immediately and unable to access your account until the selected period has passed. You cannot cancel or shorten an active self-exclusion."
3. Enter "Exclusion duration (days)" — 1 to 365 ("Choose between 1 and 365 days.").
4. Confirm ("Confirm Self-Exclusion") → the session is destroyed and the user lands on the login page.
5. Until the exclusion ends, login shows: "Your account is self-excluded" / "Your account is self-excluded until {date}." / "You will not be able to log in until that date. Access will resume automatically once it has passed." There is no way to lift it from the UI.

## Legal / policy pages

Every layout ships the same ten policy topics (content is operator/i18n text): Privacy Policy, Terms of Service, Registration & KYC, KYC, Fairness, Anti Money Laundering, Dispute Resolution, Payout, Responsible Gaming, Self Exclusion.

Where they live differs per layout:
- **Dedicated routes** (`/termsofservice`, `/privacy`, `/kyc`, …): layouts 1, 7, 9.
- **Single "Legal & Policies" hub** with a page dropdown (`/legal-policy?page=…`): layouts 5, 6, 8.
- **Matka layouts (2, 3, 4)**: a single Terms & Conditions page plus the self-exclusion page; web variants carry About Us / How to Play.

## Age notice

Footer/legal copy includes: "Players need to be 18+ in order to register. Underage gambling is prohibited." Registration requires the 18+ confirmation toggle (see `01-login-register.md`).
