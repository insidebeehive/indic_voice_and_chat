# KYC & Verification — Shared UI Behavior

## No KYC document upload exists in the frontend

Verified across every layout package: there is **no KYC document-upload screen** (no ID/selfie/proof upload) anywhere in the frontend. If a user asks "where do I upload my documents", the answer is that the website/app has no such screen — direct them to the operator's support channel (see the backend KB for the operator's actual KYC process).

## What the UI does have

1. **Static KYC policy pages** — informational text only. Casino layouts expose "KYC" and "Registration & KYC" pages (either as dedicated routes like `/kyc` and `/registration`, or inside a "Legal & Policies" hub page — see the layout doc for the exact location).
2. **Bank-details verification** — the de-facto identity step [INFERRED: operators treat bank-account approval as their verification gate]. The user submits account number, account holder name, and IFSC; the change is OTP-verified and then goes to admin approval:
   - "Bank details submitted. Waiting for admin approval."
   - Approved: "Bank details updated successfully."
   - Rejected: "Admin did not approve your new bank details. Please check and try again."
   - While pending, a "Pending" badge appears next to account-settings entries and withdrawals are blocked ("Bank change request pending. Withdrawals blocked until approved.").
3. **Address details** — casino layouts (1, 5, 8, and layout-9's Address tab) include an address form in account settings (Country, State, City, Pincode, address lines). Some flows require it (e.g. rewards redemption on layout-1 errors with "You have not updated your address").

## Phone verification

The registered mobile number is verified by OTP at registration and re-used as the OTP channel for sensitive actions (withdrawals, bank changes, password changes). The registered mobile/email/username are read-only in profile settings — users cannot change them in the UI.
