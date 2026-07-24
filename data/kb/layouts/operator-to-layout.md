# Operator → Layout Lookup

Mechanical mapping from operator app to layout package, generated from `apps/*/package.json` `@frontend/layout-*` dependencies and each layout's `app-configs/` directory. Use this to resolve which layout doc in `kb/frontend/layouts/` applies to an operator; fall back to `kb/frontend/global/` for anything not listed as a layout delta.

- **App / appConfig name** — directory name under `apps/` and (where applicable) the config file under the layout's `app-configs/`.
- **Operator ID** — from the app's local `.env` `OPERATOR_ID` where present. Blank means the ID is injected via deployment environment and is not readable from this repo.
- Apps named `layout*-app` / `layout*-web` / `layout-app` are internal template/reference apps, not live operators. `jupiter-app` is the internal pilot app for layout-1.

## Layout 1 (casino + sports, single variant)

| App / appConfig | Operator ID |
|---|---|
| 247majestic | |
| abbaby | |
| alphaxch | |
| badibaazi | |
| bigbull | 31c9cdf3-2d23-44e5-b5c7-7612ad449466 |
| cbtfstake | |
| dj247 | |
| dreamcasino9 | 41d72f7c-3ac1-4c6c-aeeb-5dab062bdc0e |
| easygames | |
| init2winit | |
| jeetobig | |
| jupiter-app *(internal pilot)* | ab858a8c-7ad4-47d2-a0b7-05ee93f8f134 |
| karabet | |
| karavip | |
| kbcexchange | b49a01d5-e588-4879-b915-921779747883 |
| khelomama | ab858a8c-7ad4-47d2-a0b7-05ee93f8f134 |
| kingsplay | |
| luckystar123 | |
| mgm91 | |
| mgm91vip | |
| mgmfun | eb93473d-aef9-4a32-91cd-79e1e7760a26 |
| mparadise | |
| playbooth | |
| playpaisa | 50db9e8a-c3c0-49f9-a834-a37ee9d955bd |
| pride95 | |
| redgames1 | |
| ruvibet | |
| shakunimama | |
| shakunivip | |
| spade91 | ddfe84c7-bbab-448d-be3c-d245eb708178 |
| spade91vip | |
| straightcourse | |
| tara888 | |
| vipbets | |
| winking93 | |
| winmax | |
| winpro11 | |
| layout1-app *(template)* | |

## Layout 2 (matka, app + web variants)

| App / appConfig | Variant | Operator ID |
|---|---|---|
| bharat-matka-app | app | |
| bharat-matka-web | web | |
| mangal-matka-web | web | |
| matka-app | app | 6a0f2c47-2da4-4506-a479-99d2ab538cd7 |
| matka-web | web | 6a0f2c47-2da4-4506-a479-99d2ab538cd7 |
| meera567-app | app | |
| meera567-web | web | |
| rama567-app | app | |
| rama567-web | web | |
| ruvi-matka-app | app | |
| ruvi-matka-web | web | |
| shree-matka-app | app | |
| shree-matka-web | web | |
| layout2-app / layout2-web *(templates)* | both | |

## Layout 3 (matka, app + web variants)

| App / appConfig | Variant | Operator ID |
|---|---|---|
| mangal-matka-app | app | |
| layout3-app *(template)* | app | |

> Note: `mangal-matka` splits across layouts — its **app** build runs on layout-3, its **web** build on layout-2. A `meghateer-web.ts` config file exists in layout-3's `web/app-configs/`, but the deployed `meghateer-web` app depends on layout-4; the layout-4 mapping is authoritative.

## Layout 4 (matka, app + web variants)

| App / appConfig | Variant | Operator ID |
|---|---|---|
| meghateer-app | app | |
| meghateer-web | web | |
| layout4-app / layout4-web *(templates)* | both | |

## Layout 5

| App / appConfig | Operator ID |
|---|---|
| spade91-v2 | 6c1a77a6-20b0-4fc9-ba4c-8add58aba9ef |
| layout5-app *(template)* | |

## Layout 6

| App / appConfig | Operator ID |
|---|---|
| aceplay7 | |
| bettingraja247 | |
| pridegame99 | |
| layout6-app *(template)* | |

## Layout 7

| App / appConfig | Operator ID |
|---|---|
| bigbull-v2 | |
| layout7-app *(template)* | |

## Layout 8

| App / appConfig | Operator ID |
|---|---|
| layout8-app *(template — no live operators yet)* | |

## Layout 9

| App / appConfig | Operator ID |
|---|---|
| goaroyals | |
| layout9-app *(template)* | |

## Layout Sports (shared multi-operator host)

| App | Notes |
|---|---|
| layout-sports-app | Single shared deployment serving multiple operators; operator identity arrives via request headers (no per-operator appConfig, no `OPERATOR_ID` env). |
