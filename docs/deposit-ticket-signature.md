# Deposit ticket relay — request signing

What our platform sends to the deposit-ticket endpoint, and a shared test
vector both sides can check against.

## Request we send

```
POST <webhook_url>
Content-Type: application/json
X-Signature: <64 lowercase hex chars>

{"order_id":"...","screenshot_url":"...","mobile":"..."}
```

- `X-Signature` is `HMAC_SHA256(secret, raw_request_body).hexdigest()` —
  lowercase hex, no `sha256=` prefix, no other encoding.
- The signed bytes are the exact bytes on the wire. We serialize the body
  once, sign that buffer, and transmit that same buffer — there is no
  re-serialization between signing and sending.
- Separators are compact: `,` and `:` with no spaces.
- Key order is `order_id`, `screenshot_url`, `mobile`.
- `mobile` is **omitted entirely** (not sent as `null` or `""`) when we can't
  resolve a number for the customer. A body with two keys is normal.
- Non-ASCII characters are `\uXXXX`-escaped, so the body is always ASCII.

## Shared test vector

Dummy salt, safe to share — this is not a real secret.

```
secret : deposit-ticket-test-salt
body   : {"order_id":"6c1a77a6-20b0-4fc9-ba4c-8add58aba9ef","screenshot_url":"https://example-bucket.s3.ap-south-1.amazonaws.com/chat/shot.png?X-Amz-Expires=21600","mobile":"9876543210"}

body length  : 177 bytes
sha256(body) : 952291ec7a2e56448519b2dac9dd3e201a3cd9a6e893fd4bfe7b3d27536551be
X-Signature  : 42494d339b7e4e08f84802ed4e4d41acd01e45d2673100e4afe8f38057728ecf
```

If your verifier produces `42494d33…8ecf` for that body and salt, the signing
scheme matches on both sides and any 401 is down to the secret value itself.
If it produces something else, the difference is in how one side assembles the
bytes — compare `sha256(body)` first to see whether the bodies even match
before the HMAC is applied.

## Verify it

Python (standard library only):

```python
import hashlib, hmac

SECRET = "deposit-ticket-test-salt"
BODY = b'{"order_id":"6c1a77a6-20b0-4fc9-ba4c-8add58aba9ef","screenshot_url":"https://example-bucket.s3.ap-south-1.amazonaws.com/chat/shot.png?X-Amz-Expires=21600","mobile":"9876543210"}'

print(hashlib.sha256(BODY).hexdigest())
print(hmac.new(SECRET.encode(), BODY, hashlib.sha256).hexdigest())
```

Node:

```js
const crypto = require("crypto");

const SECRET = "deposit-ticket-test-salt";
const BODY = Buffer.from('{"order_id":"6c1a77a6-20b0-4fc9-ba4c-8add58aba9ef","screenshot_url":"https://example-bucket.s3.ap-south-1.amazonaws.com/chat/shot.png?X-Amz-Expires=21600","mobile":"9876543210"}');

console.log(crypto.createHash("sha256").update(BODY).digest("hex"));
console.log(crypto.createHmac("sha256", SECRET).update(BODY).digest("hex"));
```

## Checking a live request

On the receiving side, HMAC the raw body **as received**, before any JSON
parsing. In Express, `express.json()` discards the raw buffer unless you keep
it:

```js
app.use(express.json({ verify: (req, _res, buf) => { req.rawBody = buf; } }));
// then sign req.rawBody, not JSON.stringify(req.body)
```

`JSON.stringify(req.body)` re-serializes with the parser's own key order and
spacing, which will not reproduce our bytes and will fail every signature
regardless of whether the secret is correct.

## Two things to confirm

1. Does the verifier HMAC the raw received body, or a re-serialized copy of
   the parsed JSON?
2. Is a body without a `mobile` key accepted?
