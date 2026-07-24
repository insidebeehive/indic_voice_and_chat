# Login & Registration — Shared UI Behavior

Verified identical across layout packages (all layouts delegate auth to shared `ui-toolkit` components). Screen composition differs per layout — see `layouts/` docs — but the flows and copy below are the same everywhere.

## Two entry models

- **Casino/sports layouts (1, 5, 6, 7, 8, 9):** separate `/login` and `/register` pages with full forms.
- **Matka layouts (2, 3, 4 app variants):** single `/login` phone-first onboarding — the user enters a mobile number, presses "Continue", and the server routes them to a login form (existing account) or an inline registration form (new number).

## OTP verification (all layouts)

1. Auth actions (register, OTP login, withdrawals, bank/password changes) require a 6-digit OTP.
2. OTP screen shows "Enter the OTP*" with a resend option ("Resend OTP" / "Resend OTP in {seconds}s").
3. Confirmation buttons: "Verify OTP & Login" / "Verify OTP & Register".
4. Toasts: "OTP sent on your Phone", "OTP sent on your registered Phone!".
5. Validation: "OTP must be 6 digits".

## Login

1. Password login is the default; a "Login with OTP" link switches to OTP mode (phone-number strategy only in most layouts), with "Login with Password" to switch back.
2. Operators may enable multiple auth strategies (Phone / Email / Username); a toggle switches the input field. Which strategies are enabled is operator config, not layout.
3. Submitting shows "Please wait..." / "Logging in...".
4. Self-excluded accounts see a notice instead of the form: "Your account is self-excluded until {date}." Login is blocked until that date passes (see `10-responsible-gaming.md`).

## No forgot-password page

No layout has a standalone forgot-password flow. Recovery paths are:

1. Use "Login with OTP" instead of the password.
2. After login, change the password from account settings / update-password page (OTP-verified).

Matka layouts additionally have a "Forgot PIN?" flow for the MPIN lock (see `layouts/layout-2.md`).

## Registration

1. Common fields: first name, last name, mobile number, password + confirm password. Email/username appear when the operator's auth strategy uses them. Optional "Referral code" field when the operator enables referrals (`auth.referral.showReferral`); when mandatory, the form notes "Note: Registrations are possible only with Referral Code!".
2. An 18+ confirmation must be accepted: "By signing up, I hereby confirm that I am over 18, I read and accepted the offer agreements with the applicable Terms and Conditions".
3. Flow: fill form → "Send OTP" → enter OTP → "Verify OTP & Register" → success ("Registration Successful!").
4. Validation strings (shared): "First name is required", "The first name should contain only alphabets", "Mobile no is invalid", "Password should be at least 6 characters long", "Password and Confirm password must be same", "Username must be between 5-25 characters long", "Referral code must be between 6 and 10 characters".

## Registration disabled

When an operator turns registration off (`features.registerAvailable: false` or route-level blocking), register CTAs are hidden and `/register` redirects to a disabled-registration page: "To create your account, kindly connect with our support channel, as registration is currently only facilitated through that channel.." / "We appreciate your cooperation!!". Registration may still be possible via an agent link (`/agent/:agentId`); an inactive agent link shows "This agent link is not active now. Kindly contact your agent for new link." (multilingual).

## Demo mode (operator-gated)

Where the operator enables demo mode, the login page shows an "or" divider with a "Try Demo Mode" button. Demo users see a persistent banner "You're in Demo Mode." with a "Register Now" CTA; restricted actions open a modal: "This action is disabled in Demo Mode. Register now to play for real."
