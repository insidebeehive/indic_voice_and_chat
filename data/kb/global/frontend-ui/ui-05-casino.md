# Casino — Shared UI Behavior

## Availability by layout family

- **Casino layouts (1, 5, 6, 9):** full casino browsing (menus, categories, providers, search) — navigation shape differs per layout.
- **Layout 7:** casino secondary to sports (`/live-casino` browsing).
- **Layout 8:** game play works; browsing pages are still stubs.
- **Matka layouts (2, 3, 4):** no casino lobby — only direct game shortcuts (home "Games" panel or a bottom-nav crash-game shortcut) that launch a game directly.

## Game launch flow (identical in every layout that has casino)

1. Tap a game tile → `/casino/play/:gameId/:gameMasterId`.
2. Some games first show a **bet-limit interstitial**: "Bet Limit", "Max Payout", button "Continue to Game Play".
3. Loading state: "Loading Game, Please wait..." — with a regional notice where applicable: "This game may not get launched in Telangana, Andhra Pradesh, Sikkim and Nagaland states!".
4. The game runs in an iframe (desktop layouts add "Fullscreen" and "Close" controls).

## Game launch errors (identical copy)

- "Error Occured" / "Game temporarily not available" / "Please try again after sometime. If problem persist, Please connect with support!"
- "Game Not Available" / "This game is temporarily unavailable"
- Suspended account: "Account Suspended" / "Please connect with support if you wish to activate your account!"

## Casino maintenance (identical copy)

"Under Maintenance" / "The Casino Section is Under Maintenance and will be back soon..." (shown as a 503-style page on casino routes while the rest of the site keeps working).

## Browsing & search (casino layouts)

1. Games are organized by **menu → category**, with provider filters on list pages. Empty states: "No games found in this category." / "No games available.".
2. Search (header icon or search page): placeholder "Search your favorite games"; results "{count} Search result(s) found"; empty `No results for "{query}"`; during maintenance: "Casino is currently under maintenance. Please try again later.".
3. Common personalized rows on casino home pages: "Recently Played Games", "Trending Games", "Top Games", "Favourite Games" (backend-driven).
