# Chat Widget — Embed Guide

**Audience:** CRM Frontend team  
**Built and hosted by:** CSS (Chat Support System) team  
**Scope:** How to embed the CSS-owned chat widget on CRM Frontend pages.

---

## Overview

CSS provides a hosted, embeddable chat widget — a JavaScript bundle that CRM Frontend teams add to their pages with one `<script>` tag, exactly like Tawk.to or Intercom.

CSS owns the widget entirely: UI design, WebSocket connection to CS, session lifecycle, escalation transitions, and voice mode handling. CRM Frontend's only responsibility is:

1. Add the script tag (below)
2. Pass the logged-in user's identity
3. Optionally configure appearance and register event callbacks

---

## Quick Start

```html
<!-- Before the widget script, define the config object -->
<script>
  window.SupportChat = {
    tenantId: "acme",
    user: {
      id:       "player-42",
      name:     "Rahul",
      language: "hi"
    }
  };
</script>

<!-- Load the widget bundle — CSS will provide the correct URL during onboarding -->
<script src="https://widget.css.example.com/widget.js" async defer></script>
```

The widget renders a launcher button (bottom-right by default). The customer clicks it to open the chat drawer.

---

## Configuration Reference

Set all options on `window.SupportChat` **before** the script tag.

### Required

| Field | Type | Description |
|---|---|---|
| `tenantId` | string | Identifies the CRM operator. Provided by CSS team at onboarding. |
| `user.id` | string | Logged-in player / user ID. CSS uses this to look up account context. Pass `null` for unauthenticated guests (a guest session is opened instead). |

### Optional user fields

| Field | Type | Description |
|---|---|---|
| `user.name` | string | Display name shown in the widget header. |
| `user.language` | string | BCP-47 language code (e.g. `"hi"`, `"en"`). Defaults to the tenant's configured default. |
| `user.metadata` | object | Arbitrary key-value pairs forwarded to CSS for routing context. Example: `{ "page": "/withdraw", "account_tier": "vip" }`. |

### Appearance (optional)

```js
window.SupportChat = {
  tenantId: "acme",
  user: { id: "player-42", name: "Rahul" },
  appearance: {
    position:     "bottom-right",  // "bottom-right" | "bottom-left"
    primaryColor: "#1a73e8",       // hex — launcher button + chat header
    launcherLabel: "Chat with us"  // omit to show icon only
  }
};
```

---

## JavaScript API

After the widget loads it exposes `window.SupportChat.widget`. Wait for `onReady` before calling any method.

```js
// Open / close the chat drawer
window.SupportChat.widget.open();
window.SupportChat.widget.close();

// Update user context (e.g. after login or page navigation)
window.SupportChat.widget.update({ user: { id: "player-99", name: "Priya" } });

// Tear down the widget entirely (removes DOM + closes socket)
window.SupportChat.widget.destroy();
```

---

## Event Callbacks

Register callbacks on `window.SupportChat` before the script loads. Each is optional.

```js
window.SupportChat = {
  tenantId: "acme",
  user: { id: "player-42", name: "Rahul" },

  onReady:      ()      => { /* widget loaded and ready */ },
  onOpen:       ()      => { /* customer opened the chat drawer */ },
  onClose:      ()      => { /* customer minimized the chat drawer */ },
  onNewMessage: (msg)   => { /* new inbound message — useful for unread badge */ },
  onEscalation: ()      => { /* AI requested a human agent */ },
  onSessionEnd: (data)  => { /* session closed */ }
};
```

| Callback | Payload | Fires when |
|---|---|---|
| `onReady` | — | Widget JS loaded and initialized |
| `onOpen` | — | Customer opens the chat drawer |
| `onClose` | — | Customer minimizes the chat drawer |
| `onNewMessage` | `{ role, text }` | A new message arrives (drawer may be closed) |
| `onEscalation` | — | AI requested a human handover |
| `onSessionEnd` | `{ summary }` | Session ended (by AI, agent, or customer) |

---

## Security

**User identity** — CSS trusts the `user.id` in the config. CRM Frontend must only pass authenticated, server-verified user IDs here. Never pass untrusted input.

**Script integrity** — CSS publishes an SRI hash for each widget bundle release. You may pin it:

```html
<script
  src="https://widget.css.example.com/widget.js"
  integrity="sha256-..."
  crossorigin="anonymous"
  async defer
></script>
```

**Content Security Policy** — the widget makes API calls and opens WebSocket connections to the CSS domain. Add the CSS widget domain to your CSP:

```
Content-Security-Policy: connect-src 'self' https://widget.css.example.com wss://widget.css.example.com;
```

---

## Implementation Checklist

- [ ] Obtain `tenantId` from the CSS team
- [ ] Add `window.SupportChat` config block before the `<script>` tag
- [ ] Pass `user.id` for authenticated users; `null` for guests
- [ ] Add CSS widget domain to CSP `connect-src`
- [ ] Optionally set `appearance.primaryColor` to match CRM brand
- [ ] Register `onEscalation` if CRM Frontend needs to update its own UI when a human agent joins
- [ ] Test: open widget → send a message → verify AI responds → escalation path → session end

---

## How It Works (for your reference)

The widget is a self-contained JS application embedded in your page:

```
Customer browser
  ├── CRM Frontend page (your code)
  └── CSS Widget (embedded, owned by CSS)
            │
            │  POST /api/chat/start
            │  WS   wss://cs.example.com/chat/ws/{session_id}
            ▼
          CSS ──► CS ──► AI Platform
```

1. Widget calls `POST /api/chat/start` on CSS to create a session.
2. CSS decides AI vs. human routing and returns a WS URL.
3. Widget connects to the WS and drives the full conversation: AI responses, typing indicators, escalation, voice mode.

CRM Frontend does not implement any WebSocket protocol. The full protocol reference (for CSS engineers building the widget internals) is in `chat-widget-frontend-integration.md`.
