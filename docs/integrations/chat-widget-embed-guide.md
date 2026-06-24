# Chat Widget — Embed Guide

**Audience:** CRM Frontend team  
**Built and hosted by:** CSS (Chat Support System) team

---

## How to embed

CSS provides a hosted chat widget. Add two things to your page — a config object and the widget script:

```html
<script>
  window.SupportChat = {
    tenantId: "acme",
    user: {
      id:       "player-42",
      name:     "Rahul",
      language: "hi"
      // metadata: { page: "/withdraw", account_tier: "vip" }
    }
    // appearance: { position: "bottom-left", primaryColor: "#1a73e8" }
  };
</script>
<script src="https://widget.css.example.com/widget.js" async defer></script>
```

That's it. The widget renders a launcher button. The customer clicks to open the chat drawer.

---

## Config fields

| Field | Required | Description |
|---|---|---|
| `tenantId` | Yes | Operator identifier — provided by CSS team at onboarding |
| `user.id` | Yes | Logged-in player / user ID. Pass `null` for unauthenticated guests |
| `user.name` | No | Display name shown in the widget header |
| `user.language` | No | BCP-47 code (e.g. `"hi"`, `"en"`). Defaults to tenant's configured default |
| `user.metadata` | No | Extra key-value pairs for routing context, e.g. `{ "page": "/withdraw", "account_tier": "vip" }` |
| `appearance.position` | No | `"bottom-right"` (default) or `"bottom-left"` |
| `appearance.primaryColor` | No | Hex color for the launcher button and chat header |
| `appearance.launcherLabel` | No | Text next to the launcher icon; omit to show icon only |

---

## Notes

- The `tenantId` and widget script URL are provided by the CSS team.
- Pass only authenticated, server-verified `user.id` values — CSS trusts this field.
- The widget handles everything inside the chat drawer: AI conversation, escalation to a human agent, and voice transitions.
