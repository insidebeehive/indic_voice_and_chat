# Withdrawals & Bank Details — Shared UI Behavior

Identical flow and copy across layouts (shared `ui-toolkit` wallet domain); only the screen shape differs per layout.

## Prerequisites shown in the UI

1. **Linked bank account required.** Without one, a banner shows "Please update your bank details for withdrawal" and the withdraw action routes to the bank-details form instead.
2. **Withdrawal window.** Withdrawals are disabled outside the operator's configured timing window; the operator's message (or a "Withdraw Timing" row with start–end times) is shown.
3. **Pending bank change blocks withdrawals:** "Bank change request pending. Withdrawals blocked until approved."
4. Operator can disable withdrawals entirely: "Withdraw is temporarily disabled" / "Deposit/Withdraw temporarily not available".

## Withdrawal flow

1. Open the Wallet section and choose Withdraw (button, tab, or drawer per layout).
2. Enter an amount (minimum shown in the field; matka layouts also show "👉 Withdrawal limit is {currency}{min} to {currency}{max}").
3. Submit → an OTP is sent to the registered phone ("We'll send an OTP to your registered phone to confirm.") → enter the 6-digit OTP to confirm.
4. The request is created and goes to admin action (matka layouts toast "Withdrawal request successful, Admin will take an action.").

## Active bonus interaction

If a deposit-rolling bonus is active, a warning dialog intercepts the withdrawal: "As your deposit rolling is active, your … Withdrawable Deposit balance is …" with the option "Cancel bonus and Withdraw" ("Please wait while your bonus is being canceled. First, we cancel your active bonus and then make a withdrawal request.") or "Cancel Withdraw".

## Cancelling a withdrawal

Pending withdrawal requests can be cancelled from the statements/wallet-history list: "Cancel Withdraw" → confirm "Are you sure you want to cancel this withdrawal request?" → toast "Withdraw cancelled successfully" (or "Failed to cancel withdraw").

## Bank account form

1. Fields: account number, account holder name, bank IFSC code. Location differs per layout (account settings section, `/wallet/addbank` page, or a wallet tab).
2. Validation: "Invalid Bank IFSC", "Invalid Account number. The account number must be between 9-18 digits", "The Bank Account name does not contain numbers or special characters!".
3. Submission is OTP-verified, then admin-approved: "Bank details submitted. Waiting for admin approval." / "Bank details updated successfully." / "Bank details are same as current. Please update the bank details.".
4. A **blocked-IFSC dialog** intercepts IFSC codes the operator has blocked.
5. A status banner/badge shows pending or rejected bank-change requests.
