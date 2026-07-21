# Chat document attachments (PDF/Word/Excel/CSV/txt) — design

**Date:** 2026-07-21
**Status:** Approved approach (A), pending spec review

## Purpose

Customers attach small documents (bills, invoices, statements, short reports —
1–20 pages, under 10 MB) in the ChatBot conversation and the bot answers
questions about them **in that session**. Nothing enters the tenant knowledge
base. The raw file is persisted like other chat media so the transcript and
backoffice retain it.

## Supported types

| Type | Mime | Handling |
|------|------|----------|
| PDF  | `application/pdf` | Sent to Gemini natively as inline data (handles scans/OCR, tables, layout) |
| Word | `application/vnd.openxmlformats-officedocument.wordprocessingml.document` (.docx) | Text extracted via existing `DocxParser` (`src/rag/ingestion.py`) |
| Excel | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` (.xlsx) | New parser (openpyxl): each sheet rendered as CSV-style text, row-capped |
| CSV | `text/csv` | Decoded as text |
| Text/Markdown | `text/plain`, `text/markdown` | Decoded as text |

Legacy `.doc`, `.rtf`, archives, and everything else: rejected with an error
frame naming the supported types. Text decode is UTF-8 with
`errors="replace"` fallback.

## Transport (mirrors image/video/audio)

- **WS frame:** `{"type":"document", "data":<b64> | "media_url":<https URL>,
  "mime":..., "filename":..., "text":<optional caption>}`.
  - `media_url` goes through the existing `_fetch_media_url` (SSRF-guarded,
    https-only, 10 MB cap); its content-type allowlist widens to the document
    mimes above.
  - Mime resolution order: frame `mime` → fetch response content-type →
    `filename` extension. Unresolvable → error frame.
- **REST:** `POST /chat/{session_id}/upload` widened to accept the document
  mimes (same processing path as the WS frame).

## Processing

New module `src/chatbot/documents.py`:

- `prepare_document_content(filename, data, mime, caption) -> list[ContentPart]`
  - PDF → `ContentPart(type="file", inline_data={"mime_type": "application/pdf", "data": ...})`
    plus a text part with the caption/framing.
  - All others → extracted text wrapped as:
    `Customer attached document "<filename>":\n<extracted text>` (+ caption).
  - Extracted text capped at **30,000 chars**; when truncated, append
    `[document truncated — first 30,000 characters shown]` so the model knows.
  - XLSX: max 200 rows per sheet before the char cap applies; sheet names
    included as headers.
- Parse failures on the extraction types (corrupt docx/xlsx, bad zip,
  undecodable text) raise a `DocumentParseError` → WS error frame
  `"Could not read the document — it may be corrupt or password-protected."`;
  socket stays open (existing per-turn error pattern). PDFs are passed to
  Gemini without local validation — a corrupt or encrypted PDF surfaces as
  the model saying it can't read the file, which is acceptable.

`ContentPart` gains the documented type `"file"`; the Gemini adapter's
`_to_parts` check widens from `type == "image"` to any part carrying
`inline_data` (one line). Non-Gemini LLM providers without multimodal support
are out of scope (same status quo as images today).

## Agent

`ChatBotAgent.handle_document(filename, data, mime, caption)` mirroring
`handle_image`: builds the multimodal/text `LLMMessage`, retrieval runs on the
caption text only, tools loop unchanged. The full `user_msg` is appended to
`session.turns` (existing behavior), so follow-up questions within the
`MAX_HISTORY_TURNS` window see the document; after the window slides past it,
the bot may ask the customer to re-attach — accepted limitation, no re-injection
machinery.

## Persistence

Raw file uploaded to S3 via the existing `_media_store`
(`chat/{tenant}/{session}/...` key). `ChatMessage` row: `type="document"`,
`content=caption or "[document: <filename>]"`, `media_mime`, `media_url=object_key`.
Served to BO/transcript via the existing `GET /chat/media/{id}` signed-URL
endpoint. Upload failure degrades exactly like images: the turn still runs,
the file just isn't persisted (logged).

## Errors & limits

- 10 MB raw cap (existing `_MAX_MEDIA_FETCH_BYTES`; enforce the same cap on
  base64 `data` after decode).
- Unsupported mime/extension → error frame listing supported types.
- Extraction dependency missing (openpyxl) → error frame, logged; pypdf and
  python-docx are already dependencies via KB ingest. openpyxl is a **new
  dependency** (pyproject).

## Testing

Unit tests mirroring the existing audio/image WS tests:
- `type:"document"` PDF via base64 → Gemini receives an `inline_data` part
  with `application/pdf`; reply flows; S3 upload happens; `document`-type
  ChatMessage persisted.
- `type:"document"` CSV via `media_url` → fetched, decoded, text injected.
- DOCX and XLSX extraction unit tests on `prepare_document_content` (tiny
  fixture files built in-test).
- Truncation: >30k chars → capped with notice.
- Unsupported type (`.zip`) → error frame, no upload, no agent call.
- `_fetch_media_url` accepts `application/pdf`, rejects `application/zip`.
- REST upload accepts a PDF.

## Docs

`docs/chatbot.md` WS contract line gains `document` in the frame list with the
type table above; upload endpoint line mentions documents.

## Out of scope (explicit)

- Session-scoped RAG/chunking for large documents (seam: the extraction step
  in `documents.py` is where chunking would slot in later).
- KB ingestion from chat uploads.
- Legacy `.doc`/`.rtf` support.
- Re-injecting documents that slid out of the history window.
