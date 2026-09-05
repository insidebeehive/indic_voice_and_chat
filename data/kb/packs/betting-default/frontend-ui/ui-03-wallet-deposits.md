# Wallet & Deposits — Shared UI Behavior

All layouts use the same payment-gateway deposit model from `ui-toolkit`. The wallet screen's shape (single form, tabs, keypad, drawers) differs per layout — see `layouts/` — but the flow and copy below are identical.

## Deposits are payment-gateway redirects only

**No layout has manual UTR entry, payment-proof/screenshot upload, or a bank-transfer instructions screen.** Verified by the absence of any such component in every layout and in the shared wallet domain. Deposit is always:

1. Open the Wallet section (header wallet link/icon, bottom-nav Wallet item, or user menu "Deposit" — per layout).
2. Enter an amount. The field shows the operator minimum: "Enter amount (min: {currency}{min})" (some layouts also show "Min: {min} · Max: {max}" or quick-amount buttons — operator-configured values).
3. Tap the deposit button. A gateway dialog appears: "INITIATING DEPOSIT REQUEST" / "Please wait, while we initiate your request" / "Don't worry, we will ensure that funds will be transferred to your account after you finish the payment process in another tab."
4. The browser is redirected to the payment provider to complete payment.
5. The provider redirects back to a result page.

## Deposit result pages (identical copy everywhere)

- **Success** (`/wallet/success/:transactionId`): "Payment Request Successful!" / "Thank you! Your payment of {amount} has been received." plus Payment Details (Transaction Id, Amount, Date) and "Please wait for some time for the amount to show up in your account."
- **Failure** (`/wallet/failure/:transactionId`): "Payment Failed!" / "Ops! Your request has not been processed successfully" / "Something's not right. We're unable to complete your request at the moment."
- **Complete/pending** (`/wallet/complete/:transactionId`): "Your payment process was complete!" / "Your account will be credited as soon as we receive confirmation from payment gateway", an auto-redirect countdown, and "Note: If your account is not credited within 15 mins, please refresh page or contact our customer support executive."

If a deposit shows Complete but the balance hasn't updated after ~15 minutes, the in-product guidance is refresh, then contact support.

## Balance display

1. Logged-in users see a balance widget in the header (or a balance card in the wallet) labeled "Real" plus the bonus balance name; layouts with exchange sports also show "Exposure".
2. Tapping the balance (where it's a pill/widget) refreshes it.

## Wallet unavailable states (identical copy)

- Deposits/withdrawals disabled by the operator: banner "Deposit/Withdraw temporarily not available".
- No session: "Invalid User Session" / "Please login again" / "Please login again after logout. If problem persist, Please connect with support!".
- Wallet error: "Error Occured" / "Wallet temporarily not available" / "Please try again after sometime. If problem persist, Please connect with support!".
- Suspended account: "Account Suspended" / "Deposit/Withdraw temporarily not available" / "Please connect with support if you wish to activate your account!".

## Wallet history

Every layout offers a wallet transaction list (a "History" slide-over, a dedicated history page, or the Wallet tab of statements) with per-type empty states like "No wallet transactions found!" / "No DEPOSIT transactions found!" / "No WITHDRAW transactions found!".
