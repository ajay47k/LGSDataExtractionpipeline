# Phase 1 — Ingestion & Preprocessing Layer: Complete Production Reference

## LGS Tech Agentic AI Workflow — Service Order Processing

---

## Table of Contents

1. Phase 1 Overview & Architecture
2. Step 1 — Source Intake & Channel Routing
3. Step 2 — Raw Text Extraction
4. Step 3 — Text Cleaning & Normalization
5. Step 4 — PII Detection & Redaction
6. Step 5 — Language Detection & Handling
7. Step 6 — Entity Pre-Extraction (Tier 1 + Tier 2)
8. Step 7 — Segmentation
9. Step 8 — Chunking (Conditional)
10. Step 9 — Canonical Schema Assembly
11. Step 10 — Queue Dispatch to Phase 2
12. Observability & Alerting
13. End-to-End Phase 1 Flow Summary

---

## 1. Phase 1 Overview & Architecture

### What Phase 1 Does

Phase 1 is the ingestion and preprocessing layer. Its job is to take raw, messy, multi-format service order input from various channels and transform it into a clean, normalized, entity-enriched, segmented canonical JSON object that the LangGraph agent (Phase 2) can reliably process.

### Why Phase 1 Exists

Without Phase 1, the LLM in Phase 3 would receive raw email HTML with signatures, encoding artifacts, reply chains, PII, and multiple defects mashed together in one blob. This causes: wasted tokens (paying for noise), hallucination risk (model confused by irrelevant content), PII leakage (sending personal data to model endpoints), missed defects (model focuses on one issue and ignores others), and unreproducible results (same input produces different outputs due to noise variation).

Phase 1 is the quality gate that determines whether the entire downstream pipeline succeeds or fails. Garbage in, garbage out — but in an agentic system, garbage in means the agent takes the wrong branch, retrieves the wrong SOP, and generates the wrong action items, all silently.

### Architecture Diagram

```
[Email/IMAP]  [Web Forms/Webhook]  [ERP/CRM Trigger]
      │              │                     │
      └──────────────┼─────────────────────┘
                     │
              [API Gateway / Load Balancer]
                     │
              [Intake Service]
                     │
              [Message Queue (SQS/Kafka)]
                     │
         [Preprocessing Workers (ECS/K8s)]
          │    │    │    │    │    │    │
          S1   S2   S3   S4   S5   S6   S7→S8→S9
          │
          └──────► [Dispatch Queue → Phase 2 Agent]
```

### Design Decision: Synchronous vs Asynchronous Processing

**Decision:** Asynchronous, queue-based processing.

**Reasoning:** Service order processing is not real-time — a 5-10 second processing delay is acceptable. Asynchronous processing via message queues gives us: decoupled ingestion from processing (intake can spike without overwhelming workers), natural retry semantics (failed messages return to queue), horizontal scalability (add more workers during peak hours), and fault isolation (if preprocessing crashes, orders are not lost — they sit in the queue).

**Tradeoff:** Added infrastructure complexity (queue management, dead letter queues, message ordering). For a low-volume system (<100 orders/day), a synchronous REST API would be simpler and sufficient. The async architecture is justified at LGS Tech's scale of thousands of daily orders.

**Alternative Considered:** Synchronous REST pipeline where each step calls the next directly. Rejected because a failure in Step 5 would require the entire pipeline to retry from Step 1, and there's no natural backpressure mechanism — if orders arrive faster than processing speed, the system starts dropping requests or timing out.

---

## 2. Step 1 — Source Intake & Channel Routing

### What Happens

Service orders arrive from three channels. Each channel delivers data in a different format. Step 1 receives the raw input and routes it to the appropriate extraction handler.

### Input Channels

**Channel A — Email (Primary, ~60% of volume)**

Field technicians and dealers email service requests. These arrive via:
- IMAP listener polling a shared mailbox every 30-60 seconds
- Microsoft Graph API webhook that pushes new emails in real-time
- Gmail API watch notifications (if using Google Workspace)

**Decision:** Microsoft Graph API with webhook push notifications.

**Reasoning:** Polling introduces latency (up to 60 seconds between email arrival and detection) and wastes API calls checking an empty mailbox. Push notifications trigger processing within seconds of email arrival. Graph API is also richer — it gives you parsed MIME parts, attachment metadata, conversation threading IDs, and sender information in structured JSON rather than raw MIME that you'd have to parse yourself with IMAP.

**Tradeoff:** Webhook infrastructure is more complex (you need a publicly reachable endpoint, webhook validation, retry handling). IMAP is simpler to set up but polling-based and gives you raw MIME. For production at scale, the push model wins.

**What the raw email contains:**
- HTML body with inline styles, font tags, div structures
- Plain text body (sometimes both, as multipart/alternative)
- Email headers (From, To, Subject, Date, Message-ID, In-Reply-To)
- Reply chains (quoted previous messages, "On [date], [person] wrote:")
- Email signatures (name, title, phone, company logo)
- Auto-generated footers ("Sent from my iPhone", "Scanned by McAfee")
- Attachments (photos of defects, PDF reports, spreadsheets)
- Tracking pixels (1x1 images from tools like Mailsuite, HubSpot)
- Encoding inconsistencies (UTF-8, ISO-8859-1, Windows-1252)

**Channel B — Web Forms / Service Portals (~30% of volume)**

Technicians submit through a web portal (ServiceNow, custom app). Data arrives as a webhook POST with structured JSON:

```json
{
  "order_id": "SO-28841",
  "submitted_by": "john.smith@dealer.com",
  "submitted_at": "2025-03-04T14:22:00Z",
  "plant_id": "P-007",
  "model_number": "XR-440",
  "serial_number": "SN-99281",
  "priority": "high",
  "description": "Compressor making grinding noise, error E-47 on panel, unit is 18 months old under warranty. Also coolant line near valve has visible leak."
}
```

The structured fields (plant_id, model_number, serial_number) are clean. The "description" field is unstructured free text — this is what needs preprocessing.

**Decision:** Accept webhooks on a dedicated endpoint with signature validation.

**Reasoning:** Webhooks provide real-time delivery without polling overhead. Signature validation (HMAC-SHA256) prevents spoofed submissions. The structured fields bypass most preprocessing steps and go directly into metadata.

**Tradeoff:** Must handle webhook delivery failures (the source system might retry, causing duplicates). Mitigated by idempotency checks using the order_id — if an order_id has already been processed, the duplicate is dropped.

**Channel C — ERP/CRM Triggers (~10% of volume)**

Systems like SAP, Salesforce Service Cloud, or Dynamics 365 fire events when a new service case is created. These arrive via:
- Salesforce Platform Events or Change Data Capture
- SAP IDoc/BAPI integration
- Custom API polling of ERP case tables

The format is similar to web forms — structured metadata plus a free-text case description field.

### Channel Router Logic

```python
def route_incoming(source_type: str, payload: dict) -> ProcessingContext:
    if source_type == "email":
        return ProcessingContext(
            channel="email",
            requires_html_extraction=True,
            requires_signature_removal=True,
            requires_thread_removal=True,
            raw_payload=payload
        )
    elif source_type == "webform":
        return ProcessingContext(
            channel="webform",
            requires_html_extraction=False,  # already structured
            requires_signature_removal=False,
            requires_thread_removal=False,
            structured_metadata=extract_form_fields(payload),
            raw_text=payload["description"],
            raw_payload=payload
        )
    elif source_type == "erp":
        return ProcessingContext(
            channel="erp",
            requires_html_extraction=False,
            requires_signature_removal=False,
            requires_thread_removal=False,
            structured_metadata=extract_erp_fields(payload),
            raw_text=payload["case_description"],
            raw_payload=payload
        )
```

**Decision:** Use a ProcessingContext object that carries channel-specific flags through the pipeline.

**Reasoning:** Each channel needs different preprocessing steps. Email needs HTML extraction, signature removal, and thread removal. Web forms and ERP only need the free-text description processed. Rather than having separate pipelines per channel (code duplication, maintenance burden), a single pipeline with conditional steps based on flags is cleaner and more maintainable.

**Tradeoff:** The conditional logic adds if/else complexity to each step. But this is manageable and far better than maintaining three separate pipeline codebases.

### Idempotency & Deduplication

**Rule:** Every incoming order is checked against a deduplication store before processing.

**Implementation:** A Redis set (or DynamoDB table) stores processed order fingerprints. The fingerprint is a hash of (source_channel + order_id + timestamp). If the fingerprint exists, the message is acknowledged and dropped.

**Why this is necessary:** Webhook retries (source system didn't get a 200 OK, so it resends), email polling overlap (same email picked up twice if polling interval is shorter than processing time), and queue message redelivery (SQS delivers a message twice in rare cases due to at-least-once semantics).

**Decision:** Use a 72-hour TTL on deduplication keys.

**Reasoning:** Long enough to catch any retry scenario (most retries happen within minutes to hours), short enough to not bloat the Redis store indefinitely. If somehow the same order arrives 4 days later, it's likely a genuinely new submission or a resubmission, not a duplicate.

### Observability at Step 1

**Metrics emitted:**
- `intake.orders.received` — counter by channel (email/webform/erp)
- `intake.orders.duplicates_dropped` — counter of deduplication hits
- `intake.latency.channel_to_queue_ms` — time from channel receipt to queue publish
- `intake.webhook.signature_failures` — counter of rejected webhooks (potential security issue)
- `intake.email.parse_failures` — counter of emails that couldn't be parsed

**Alerts:**
- If `intake.orders.received` drops to zero for any channel for >30 minutes during business hours → channel may be down
- If `intake.orders.duplicates_dropped` spikes above 20% of total → source system retry loop or integration misconfigured
- If `intake.webhook.signature_failures` spikes → potential spoofing attempt, investigate immediately
- If `intake.email.parse_failures` exceeds 5% → email format changed or MIME parsing library issue

---

## 3. Step 2 — Raw Text Extraction

### What Happens

This step extracts the actual text content from the raw payload, stripping away all transport-layer artifacts. The output is a raw text string plus any structured metadata that was available.

### Email Text Extraction

**Sub-step 2a — MIME Part Selection**

Emails arrive as MIME multipart messages. A typical email has:

```
multipart/mixed
├── multipart/alternative
│   ├── text/plain (plain text version)
│   └── text/html (HTML version)
├── image/png (attachment: defect photo)
└── application/pdf (attachment: previous report)
```

**Decision:** Prefer HTML body, fall back to plain text.

**Reasoning:** The HTML body is the "intended" version — it's what the technician saw when composing. The plain text version is often auto-generated from HTML and can lose formatting cues (bullet points, numbered lists, paragraph breaks) that are useful for segmentation later. However, the HTML body needs aggressive stripping, so if HTML extraction fails, we fall back to plain text.

**Tradeoff:** Processing HTML adds complexity (need BeautifulSoup or similar parser). Plain text is simpler but loses structural information. The structural information is valuable enough for segmentation to justify the HTML processing overhead.

**Sub-step 2b — HTML Tag Stripping**

Strip all HTML while preserving structural whitespace:
- `<br>`, `<br/>` → newline
- `<p>`, `</p>` → double newline (paragraph break)
- `<li>` → newline + "- " (preserves list structure for segmentation)
- `<td>` → tab (preserves table structure)
- All other tags → removed
- HTML entities decoded: `&amp;` → `&`, `&lt;` → `<`, `&#8217;` → `'`

```python
from bs4 import BeautifulSoup
import html

def extract_text_from_html(html_body: str) -> str:
    soup = BeautifulSoup(html_body, "html.parser")
    
    # Remove script, style, head elements entirely
    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()
    
    # Remove tracking pixels (1x1 images)
    for img in soup.find_all("img"):
        width = img.get("width", "")
        height = img.get("height", "")
        if width in ["1", "0"] or height in ["1", "0"]:
            img.decompose()
    
    # Convert structural HTML to whitespace markers
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for p in soup.find_all("p"):
        p.insert_before("\n\n")
    for li in soup.find_all("li"):
        li.insert_before("\n- ")
    
    text = soup.get_text()
    text = html.unescape(text)  # decode HTML entities
    return text
```

**Decision:** Custom HTML extraction logic rather than a library like `html2text`.

**Reasoning:** Libraries like `html2text` try to preserve Markdown formatting, which adds noise we don't need (header markers, emphasis markers, link formatting). We need plain text with only structural whitespace preserved. Custom logic gives us exact control over what's preserved and what's stripped.

**Tradeoff:** Must maintain the extraction logic ourselves. If email clients change their HTML formatting patterns, we may need to update. Mitigated by logging extraction quality metrics and reviewing failure cases regularly.

**Sub-step 2c — Reply Chain Removal**

Email threads contain quoted replies from previous messages. A technician's latest email might contain 3 previous replies below it. We only want the latest message.

```
Actual content we want:
"The compressor is still making noise after the repair attempt."

Reply chain we don't want:
"On Mar 3, 2025, John Smith wrote:
> Tried replacing the fan belt but noise persists.
> 
> On Mar 2, 2025, Support Team wrote:
>> Please try replacing the fan belt as a first step."
```

**Implementation:**

```python
import re

REPLY_PATTERNS = [
    r"On .{10,60} wrote:\s*$",          # "On [date], [name] wrote:"
    r"^-{3,}\s*Original Message\s*-{3,}",  # "--- Original Message ---"
    r"^From:.*\nSent:.*\nTo:.*\nSubject:", # Outlook-style forwarded header
    r"^>{1,}\s",                            # Quoted lines starting with >
    r"^Le .{10,60} a écrit\s*:",            # French reply pattern
    r"^Am .{10,60} schrieb\s*:",            # German reply pattern
]

def remove_reply_chain(text: str) -> str:
    lines = text.split("\n")
    cutoff_line = len(lines)
    
    for i, line in enumerate(lines):
        for pattern in REPLY_PATTERNS:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                cutoff_line = i
                break
        if cutoff_line < len(lines):
            break
    
    return "\n".join(lines[:cutoff_line]).strip()
```

**Decision:** Pattern-based reply detection with first-match cutoff.

**Reasoning:** Reply chains always appear below the new content. Once we detect the first reply boundary marker, everything below it is previous conversation. We don't need sophisticated parsing — the boundary patterns are well-established across email clients (Outlook, Gmail, Apple Mail all use recognizable patterns).

**Tradeoff:** Occasional false positives — if a technician writes "On Monday, the unit failed" at the start of a line, it might match the reply pattern and truncate the actual content. Mitigated by requiring the pattern to match more specifically (date format + "wrote:" together) and by logging truncation events for manual review.

**Rule:** If reply chain removal removes more than 80% of the text content, flag the order for manual review — this likely indicates a false positive truncation.

**Sub-step 2d — Signature Removal**

Email signatures contain contact information that is not part of the service order:

```
-- 
John Smith | Senior Field Technician
Dealer Services, Western Region
john.smith@dealer.com | (555) 123-4567
www.dealercompany.com
```

**Implementation:** Use the `email-reply-parser` Python library as a first pass (it detects signature blocks reliably for common patterns), then apply regex fallbacks for non-standard signatures:

```python
SIGNATURE_PATTERNS = [
    r"^--\s*$",                           # Standard sig delimiter
    r"^Sent from my (iPhone|iPad|Galaxy)", # Mobile signatures
    r"^Get Outlook for",                   # Outlook mobile
    r"^Scanned by .*$",                    # Antivirus footers
    r"^_{10,}$",                           # Underscore dividers
    r"^This email .*confidential",         # Legal disclaimers
    r"^CONFIDENTIALITY NOTICE",            # Legal headers
]
```

**Decision:** Library-first, regex-fallback approach.

**Reasoning:** `email-reply-parser` handles common cases well and is maintained by the open source community. But it misses domain-specific signatures (company-specific legal disclaimers, custom footers). The regex fallback catches these. Running both and taking the union of detected signature blocks gives us the best coverage.

**Tradeoff:** Over-aggressive signature removal might strip relevant content if a technician puts important info after their signature (rare but possible). Under-aggressive removal lets PII through to the pipeline. We err on the side of over-removal because PII leakage is a harder failure than missing a sentence of content.

**Rule:** Removed signature content is archived (not deleted) in the raw payload store so it can be recovered if needed.

### Web Form / ERP Text Extraction

For web forms and ERP triggers, text extraction is straightforward — pull the description field from the JSON payload. The structured fields (plant_id, model_number, etc.) are extracted directly into metadata.

```python
def extract_from_webform(payload: dict) -> tuple[str, dict]:
    metadata = {
        "order_id": payload.get("order_id"),
        "plant_id": payload.get("plant_id"),
        "model_number": payload.get("model_number"),
        "serial_number": payload.get("serial_number"),
        "priority": payload.get("priority"),
        "submitted_by": payload.get("submitted_by"),
        "submitted_at": payload.get("submitted_at"),
    }
    raw_text = payload.get("description", "")
    return raw_text, metadata
```

### Attachment Handling

**Decision:** Attachments are stored separately and not processed in Phase 1.

**Reasoning:** Attachments (photos, PDFs, spreadsheets) require different processing pipelines (OCR, image analysis, document parsing). Including them in Phase 1 would significantly complicate the preprocessing pipeline and increase processing time. Instead, attachment references are stored in the metadata, and a separate attachment processing pipeline can run in parallel and merge results later.

**Tradeoff:** If a critical defect detail is only in an attachment (e.g., a photo of the error code screen), Phase 1 will miss it. Mitigated by training the action item generation in Phase 5 to flag cases where the text mentions an attachment ("see attached photo") but the attachment hasn't been processed yet.

**Rule:** If the text body contains references to attachments ("see attached", "photo below", "per the attached report"), a flag `has_attachment_reference: true` is added to the metadata. This flag triggers parallel attachment processing and a hold on final action item generation until attachment results are available.

### Observability at Step 2

**Metrics emitted:**
- `extraction.html_strip.duration_ms` — latency of HTML extraction
- `extraction.reply_chain.lines_removed` — how many lines were removed as reply chain
- `extraction.reply_chain.truncation_ratio` — percentage of text removed (alert if >80%)
- `extraction.signature.detected` — boolean, was a signature found
- `extraction.signature.lines_removed` — count of signature lines removed
- `extraction.text_length.raw` — character count before extraction
- `extraction.text_length.clean` — character count after extraction
- `extraction.attachment_references_found` — count of attachment mentions in text

**Alerts:**
- If `extraction.reply_chain.truncation_ratio` exceeds 80% → likely false positive, route to manual review
- If `extraction.text_length.clean` is 0 or under 10 characters → extraction stripped everything, investigate
- If `extraction.html_strip.duration_ms` exceeds 500ms → unusually complex HTML, possible HTML bomb or malformed email

---

## 4. Step 3 — Text Cleaning & Normalization

### What Happens

The extracted raw text is cleaned and normalized into a consistent format. This step handles encoding issues, whitespace normalization, special character handling, and boilerplate removal.

### Sub-step 3a — Encoding Normalization

**Problem:** Text from different sources arrives in different encodings. Emails might be UTF-8, ISO-8859-1, or Windows-1252. ERP systems (especially older SAP installations) might use EBCDIC-derived encodings. Mixed encodings produce garbled characters:

```
Before: "The compressor\x92s valve is broken"  (Windows-1252 apostrophe)
After:  "The compressor's valve is broken"      (Clean UTF-8)
```

**Implementation:**

```python
import chardet

def normalize_encoding(text: bytes) -> str:
    # Detect encoding
    detection = chardet.detect(text)
    detected_encoding = detection["encoding"]
    confidence = detection["confidence"]
    
    # Attempt decode with detected encoding
    if confidence > 0.7:
        try:
            return text.decode(detected_encoding)
        except (UnicodeDecodeError, LookupError):
            pass
    
    # Fallback chain
    for encoding in ["utf-8", "windows-1252", "iso-8859-1", "ascii"]:
        try:
            return text.decode(encoding)
        except UnicodeDecodeError:
            continue
    
    # Last resort: decode with replacement characters
    return text.decode("utf-8", errors="replace")
```

**Decision:** Detect-first, fallback-chain approach with replacement as last resort.

**Reasoning:** `chardet` correctly identifies encoding ~85% of the time. The fallback chain covers the most common encodings in enterprise environments. Using `errors="replace"` as a last resort ensures we never crash on encoding issues — we'd rather have a few `�` replacement characters than a pipeline failure.

**Tradeoff:** `chardet` adds ~5ms per text block. For high-volume processing, this is negligible. The `errors="replace"` fallback can silently corrupt characters, but this is preferable to a processing failure, and the corruption rate is extremely low (<0.1% of orders).

**Rule:** If `errors="replace"` is triggered (replacement characters detected in output), log a warning with the original byte sequence for debugging.

### Sub-step 3b — Unicode Normalization

**Problem:** Unicode has multiple ways to represent the same character. "é" can be a single code point (U+00E9) or a base "e" (U+0065) plus a combining acute accent (U+0301). This inconsistency breaks regex matching and string comparison.

```python
import unicodedata

def normalize_unicode(text: str) -> str:
    # NFC normalization: compose characters into single code points
    return unicodedata.normalize("NFC", text)
```

**Decision:** NFC (Canonical Decomposition followed by Canonical Composition).

**Reasoning:** NFC is the most compact representation and is what most systems expect. It ensures that regex patterns match consistently regardless of how the original text encoded accented characters.

### Sub-step 3c — Smart Quote & Special Character Normalization

**Problem:** Word processors and email clients insert typographic characters that look like standard ASCII but aren't:

```
Smart quotes:  "hello" → "hello"   'it's' → 'it's'
Em-dash:       — → --
En-dash:       – → -
Ellipsis:      … → ...
Non-breaking space: \xa0 → regular space
Zero-width characters: \u200b, \u200c, \ufeff → removed
```

**Implementation:**

```python
CHAR_MAP = {
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote / apostrophe
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u2014": "--",  # em-dash
    "\u2013": "-",   # en-dash
    "\u2026": "...", # ellipsis
    "\xa0": " ",     # non-breaking space
    "\u200b": "",    # zero-width space
    "\u200c": "",    # zero-width non-joiner
    "\u200d": "",    # zero-width joiner
    "\ufeff": "",    # BOM / zero-width no-break space
}

def normalize_special_chars(text: str) -> str:
    for original, replacement in CHAR_MAP.items():
        text = text.replace(original, replacement)
    return text
```

**Decision:** Explicit character mapping rather than aggressive ASCII transliteration.

**Reasoning:** We want to normalize problematic characters while preserving legitimate Unicode (accented names, non-Latin text). A tool like `unidecode` would convert "José" to "Jose" and "München" to "Munchen", losing information. Our explicit map only targets characters that cause processing issues while preserving everything else.

**Tradeoff:** We must maintain the character map manually. New problematic characters may appear over time (email clients occasionally introduce new typography). Mitigated by logging any non-ASCII characters that survive normalization so we can identify new candidates for the map.

### Sub-step 3d — Whitespace Normalization

**Problem:** Raw text contains inconsistent whitespace — multiple spaces, tabs, excessive newlines, mixed line endings (\\r\\n vs \\n vs \\r):

```
Before: "The   XR-440\t\tat  plant   7\r\n\r\n\r\n\r\nis showing    E-47"
After:  "The XR-440 at plant 7\n\nis showing E-47"
```

**Implementation:**

```python
import re

def normalize_whitespace(text: str) -> str:
    # Normalize line endings to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Replace tabs with single space
    text = text.replace("\t", " ")
    
    # Collapse multiple spaces to single space (within lines)
    text = re.sub(r"[^\S\n]+", " ", text)
    
    # Collapse 3+ newlines to double newline (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    
    # Strip leading/trailing whitespace for entire text
    return text.strip()
```

**Decision:** Preserve double newlines (paragraph breaks) but collapse triple+ newlines.

**Reasoning:** Double newlines indicate intentional paragraph breaks, which are valuable signals for segmentation in Step 7. Collapsing them to single newlines would lose this structural information. But triple+ newlines are just sloppy formatting and add no signal.

**Tradeoff:** If a technician intentionally used triple newlines to indicate a stronger separation (e.g., between completely unrelated issues), we lose that emphasis. However, the segmentation algorithm in Step 7 uses multiple signals beyond whitespace, so this loss is compensated.

**Rule:** The regex `[^\S\n]+` collapses all whitespace characters EXCEPT newlines into a single space. This is critical — using `\s+` would also collapse newlines, destroying paragraph structure.

### Sub-step 3e — Boilerplate Removal

**Problem:** Automated footers, legal disclaimers, and system-generated text add noise:

```
Examples of boilerplate:
- "This email and any attachments are confidential..."
- "Please consider the environment before printing"
- "CAUTION: This email originated outside the organization"
- "Classification: Internal"
- "Mailsuite · Email tracked with Mailsuite · Opt out"
```

**Implementation:**

```python
BOILERPLATE_PATTERNS = [
    r"This email .*?confidential.*?(?:\n.*?){0,5}$",
    r"CAUTION:?\s*This email originated.*$",
    r"Please consider the environment.*$",
    r"Classification:\s*(Internal|Public|Confidential).*$",
    r"Email tracked with .*$",
    r"Mailsuite.*(?:Opt out)?.*$",
    r"Unsubscribe.*$",
    r"(?:Powered|Sent) by .*$",
]

BOILERPLATE_STORE = {}  # Configurable, loaded from config service

def remove_boilerplate(text: str) -> str:
    for pattern in BOILERPLATE_PATTERNS + list(BOILERPLATE_STORE.values()):
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()
```

**Decision:** Pattern-based removal with an external configurable store.

**Reasoning:** Boilerplate patterns are predictable and well-suited for regex. But new boilerplate appears regularly (companies change their email disclaimers, new tracking tools are adopted). The external config store allows adding new patterns without code deployment.

**Tradeoff:** Over-aggressive boilerplate removal could strip legitimate content that happens to match a pattern. For example, if a technician writes "This email is about the confidential project" and the boilerplate pattern matches "This email.*confidential", we'd incorrectly strip relevant content. Mitigated by making patterns more specific (requiring "confidential" to be near "attachments" or "intended recipient" rather than matching broadly) and by logging all removed content for audit.

**Rule:** Removed boilerplate content is stored in the raw payload archive. If downstream processing produces low-confidence results, the boilerplate can be reintroduced to check if relevant content was accidentally removed.

### Sub-step 3f — Case Normalization (Selective)

**Decision:** Do NOT lowercase the entire text. Only lowercase specific non-entity text for consistency.

**Reasoning:** Model numbers (XR-440), error codes (E-47), part numbers (P/N 7832-A), and acronyms (SOP, HVAC, OEM) carry semantic meaning in their casing. Lowercasing "XR-440" to "xr-440" would break regex patterns that expect uppercase letters and could confuse the fine-tuned LLM that was trained on properly cased technical data.

**Tradeoff:** Inconsistent casing in non-entity text (a technician writing "COMPRESSOR IS BROKEN" vs "compressor is broken" vs "Compressor Is Broken") means the same semantic concept has multiple surface forms, which could slightly reduce embedding quality in Phase 4. However, modern embedding models (and fine-tuned LLMs) are reasonably case-insensitive for common words, so the risk of corrupting entity casing outweighs the benefit of normalizing non-entity casing.

**Rule:** Case is preserved as-is through Step 3. The fine-tuned LLM in Phase 3 handles case variation naturally.

### Observability at Step 3

**Metrics emitted:**
- `cleaning.encoding.detected` — which encoding was detected (track distribution)
- `cleaning.encoding.fallback_used` — whether the fallback chain was needed
- `cleaning.encoding.replacement_chars` — count of `errors="replace"` characters
- `cleaning.whitespace.newlines_collapsed` — count of excessive newlines removed
- `cleaning.boilerplate.patterns_matched` — which patterns matched (track distribution)
- `cleaning.boilerplate.chars_removed` — total characters removed as boilerplate
- `cleaning.text_length.before` — character count entering Step 3
- `cleaning.text_length.after` — character count exiting Step 3
- `cleaning.reduction_ratio` — (before - after) / before

**Alerts:**
- If `cleaning.encoding.replacement_chars` exceeds 5 per order → encoding detection failing, investigate source
- If `cleaning.reduction_ratio` exceeds 70% → aggressive cleaning may be removing real content
- If `cleaning.encoding.fallback_used` rate exceeds 20% → new encoding pattern from a source system, update detection
- If `cleaning.boilerplate.patterns_matched` shows a single pattern matching >50% of orders → verify it's actually boilerplate, not content

---

## 5. Step 4 — PII Detection & Redaction

### What Happens

Before text reaches any ML model, personally identifiable information must be detected and redacted to comply with data privacy regulations and prevent PII leakage to model inference endpoints.

### Why This Step Exists Here (After Cleaning, Before Extraction)

**After cleaning:** PII detection accuracy depends on clean text. Encoding artifacts and HTML tags interfere with NER models and regex patterns. For example, `john&#46;smith&#64;dealer&#46;com` won't match an email regex, but after HTML entity decoding in Step 3, `john.smith@dealer.com` matches perfectly.

**Before extraction:** Entity extraction in Step 6 uses regex and spaCy NER. If PII is still in the text during extraction, the extraction step might capture a technician's phone number as a "relevant entity," propagating PII into the LangGraph state and downstream systems. Redacting PII before extraction ensures extracted entities are clean.

### PII Categories and Detection Methods

**Category 1 — Structured PII (Regex-Detectable)**

These have predictable formats:

```python
PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "phone_us": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "phone_intl": r"\b\+\d{1,3}[-.\s]?\d{4,14}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "date_of_birth": r"\b(?:DOB|Date of Birth)[:\s]+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
}

REDACTION_MAP = {
    "email": "[EMAIL]",
    "phone_us": "[PHONE]",
    "phone_intl": "[PHONE]",
    "ssn": "[SSN]",
    "credit_card": "[CREDIT_CARD]",
    "ip_address": "[IP_ADDRESS]",
    "date_of_birth": "[DOB]",
}
```

**Category 2 — Semi-Structured PII (NER-Detectable)**

Names, locations, and organizations require NER:

```python
import spacy

nlp = spacy.load("en_core_web_md")  # loaded once at service startup

def detect_ner_pii(text: str) -> list[dict]:
    doc = nlp(text)
    pii_entities = []
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            pii_entities.append({
                "text": ent.text,
                "label": "PERSON",
                "start": ent.start_char,
                "end": ent.end_char,
                "replacement": "[PERSON]"
            })
        elif ent.label_ == "GPE" and is_personal_location(ent, doc):
            # Only redact locations that are personal (home addresses)
            # NOT business locations like plant names
            pii_entities.append({
                "text": ent.text,
                "label": "LOCATION",
                "start": ent.start_char,
                "end": ent.end_char,
                "replacement": "[LOCATION]"
            })
    
    return pii_entities
```

**Decision:** Use spaCy `en_core_web_md` (medium model) rather than `en_core_web_sm` (small).

**Reasoning:** The medium model has significantly better NER accuracy for person names (~92% F1 vs ~85% F1 on standard benchmarks). For PII detection, recall is more important than precision — missing a name is worse than false-flagging a non-name. The medium model adds ~50MB memory and ~10ms latency per text, which is acceptable.

**Tradeoff:** The medium model still misses domain-specific names and non-English names. A perfect solution would use a dedicated PII detection service like Azure Presidio or AWS Comprehend PII, but that adds an external API dependency and latency. For this system, spaCy + regex provides 90%+ coverage, and the remaining edge cases are handled by the redaction-first policy.

### Context-Aware PII Decisions

Not all detected entities should be redacted. This is the tricky part:

**Plant locations are NOT PII.** "Plant 7 in Chicago" refers to a business facility, not a personal address. The segmentation and RAG pipeline need this information. Redacting it would break downstream processing.

**Technician names in context might be operationally relevant.** "Ask for Mike at the site" might be important for the action item. But the technician's personal phone number in the signature is PII.

**Decision rules:**
- Names appearing in email signatures → always redact
- Names appearing in the body text with operational context → redact but store mapping (so downstream can reference "[PERSON_1]" consistently)
- Business locations (plant names, site names, city + "plant/site/facility") → do NOT redact
- Personal addresses (street + city + state pattern) → always redact
- Phone numbers in signatures → always redact
- Phone numbers in body text with operational context ("call [PHONE] for access") → redact but flag

```python
def is_personal_location(entity, doc) -> bool:
    """Check if a location entity is personal (home address) vs business."""
    context_window = doc[max(0, entity.start-5):min(len(doc), entity.end+5)]
    context_text = context_window.text.lower()
    
    business_indicators = ["plant", "site", "facility", "warehouse", 
                           "factory", "office", "campus", "building"]
    
    for indicator in business_indicators:
        if indicator in context_text:
            return False  # Business location, don't redact
    
    # Check for street address pattern
    if re.search(r"\d+\s+\w+\s+(street|st|avenue|ave|road|rd|drive|dr)", 
                 entity.text, re.IGNORECASE):
        return True  # Personal address, redact
    
    return False  # Default: don't redact locations (most are business)
```

**Tradeoff:** This context-aware approach is more complex and has edge cases. A simpler approach would be to redact ALL names and ALL locations. But that would destroy operationally critical information (plant locations, contact persons at sites) that the downstream pipeline needs. The added complexity is justified by the operational need to preserve business context while protecting personal data.

### PII Redaction Index

After redaction, we maintain a PII index that maps redaction tokens to original values:

```python
pii_index = {
    "[PERSON_1]": {"original": "John Smith", "positions": [45, 203]},
    "[PHONE_1]": {"original": "(555) 123-4567", "positions": [67]},
    "[EMAIL_1]": {"original": "john.smith@dealer.com", "positions": [89]},
}
```

**Decision:** Store the PII index in an encrypted, access-controlled store (not in the main processing pipeline).

**Reasoning:** The PII index enables recovery if a human reviewer needs the original names/contacts. But it must not flow through the ML pipeline or be accessible to the LLM. It's stored separately with encryption at rest and access logging, retrievable only by authorized human operators.

**Rule:** The PII index has a retention period aligned with data privacy policy (e.g., 90 days for GDPR, varies by jurisdiction). After the retention period, it's permanently deleted.

### Observability at Step 4

**Metrics emitted:**
- `pii.entities_detected` — count by type (PERSON, PHONE, EMAIL, SSN, etc.)
- `pii.entities_redacted` — count of redactions applied
- `pii.false_positive_rate_estimate` — based on sampling and human review
- `pii.detection.regex_hits` — entities caught by regex
- `pii.detection.ner_hits` — entities caught by spaCy NER
- `pii.context_decisions` — count of "keep vs redact" context decisions
- `pii.processing_time_ms` — latency of PII step

**Alerts:**
- If `pii.entities_detected` for SSN or credit card exceeds 0 → unexpected sensitive data in service orders, investigate source
- If `pii.entities_detected` drops to near-zero for a channel that normally has PII → detection might be broken
- If `pii.processing_time_ms` exceeds 200ms → spaCy model might be processing unusually long text
- Periodic audit: sample 50 redacted orders per week, have a human verify no PII leaked through

---

## 6. Step 5 — Language Detection & Handling

### What Happens

Verify the language of the cleaned, redacted text and route non-English content appropriately.

### Why This Step Exists

LGS Tech operates globally. Service orders from European, Asian, or Latin American regions may arrive in languages other than English. The fine-tuned Llama 3.1 8B in Phase 3 was trained primarily on English technical logs, so non-English input would produce unreliable extractions.

### Implementation

```python
from langdetect import detect, detect_langs

def detect_language(text: str) -> dict:
    try:
        primary = detect(text)
        probabilities = detect_langs(text)  # returns list of Language objects
        return {
            "primary_language": primary,
            "confidence": probabilities[0].prob,
            "all_detected": [(str(l.lang), l.prob) for l in probabilities]
        }
    except Exception:
        return {
            "primary_language": "unknown",
            "confidence": 0.0,
            "all_detected": []
        }
```

### Routing Logic

**Decision tree:**

```
If primary_language == "en" and confidence > 0.8:
    → Continue to Step 6 (normal flow)

If primary_language == "en" and confidence 0.5-0.8:
    → Continue to Step 6 but flag as "mixed_language" for manual review

If primary_language != "en" and confidence > 0.8:
    → Route to translation service, then continue to Step 6

If confidence < 0.5 or primary_language == "unknown":
    → Flag for manual language assessment
```

**Decision:** Route non-English content through translation rather than maintaining separate language-specific pipelines.

**Reasoning:** Building separate fine-tuned LLMs and separate NER models for each language is prohibitively expensive. Translation (using a service like Azure Translator, Google Cloud Translation, or an open-source model like NLLB) converts the content to English where the existing pipeline handles it. Modern translation services are accurate enough for technical content, especially when the source is semi-structured field reports rather than literary text.

**Tradeoff:** Translation adds latency (100-500ms per text block) and cost. Translation errors could corrupt technical terms — a German technician writing "Verdichter" (compressor) might get translated to "condenser" or "compacter" depending on context. Mitigated by maintaining a domain-specific translation glossary that forces correct translation of key technical terms. Also, the original untranslated text is stored alongside the translation so a bilingual reviewer can verify if needed.

**Rule:** Translation glossary entries are maintained as configuration:

```json
{
    "de": {
        "Verdichter": "compressor",
        "Kältemittel": "refrigerant",
        "Störungscode": "error code"
    },
    "fr": {
        "compresseur": "compressor",
        "fluide frigorigène": "refrigerant",
        "code d'erreur": "error code"
    }
}
```

### Observability at Step 5

**Metrics emitted:**
- `language.detected` — distribution of detected languages
- `language.confidence` — average confidence score
- `language.translation_triggered` — count of orders requiring translation
- `language.translation_latency_ms` — translation service latency
- `language.manual_review_flagged` — count of ambiguous language cases

**Alerts:**
- If `language.translation_triggered` suddenly spikes → new regional source may have come online
- If `language.translation_latency_ms` exceeds 1 second → translation service degradation
- If `language.detected` shows a new language not in translation glossary → add glossary entries

---

## 7. Step 6 — Entity Pre-Extraction (Tier 1 + Tier 2)

### What Happens

Before the expensive LLM in Phase 3, we extract entities using fast, deterministic methods. This produces high-confidence anchors for cross-validation and provides structured metadata for segmentation.

### Why Pre-Extraction Before LLM

Three reasons:

1. **Cross-validation.** Pre-extracted entities serve as ground truth anchors. If regex finds model number "XR-440" but the LLM in Phase 3 outputs "XR-400", that mismatch is a validation flag.

2. **Segmentation needs entities.** Step 7 (segmentation) uses entity clusters to identify distinct issues. Without pre-extraction, segmentation would rely solely on linguistic cues, which are less reliable.

3. **Cost reduction.** Every entity that regex captures deterministically is an entity the LLM doesn't need to extract, reducing the extraction burden on the expensive model.

### Tier 1 — Regex Entity Extraction

**Entity types and patterns:**

```python
ENTITY_PATTERNS = {
    "model_number": {
        "patterns": [
            r"\b[A-Z]{2,3}-\d{3,4}[A-Z]?\b",      # XR-440, ABC-1234B
            r"\b[A-Z]{2,3}\d{3,4}[A-Z]?\b",         # XR440, ABC1234B (no dash)
            r"\bModel\s*#?\s*:?\s*([A-Z0-9-]+)\b",  # "Model: XR-440"
        ],
        "confidence": 0.95,
        "validation": lambda m: len(m) >= 4  # minimum length check
    },
    "serial_number": {
        "patterns": [
            r"\bSN[-\s]?\d{5,10}\b",                # SN-99281, SN 99281
            r"\bSerial\s*#?\s*:?\s*([A-Z0-9-]+)\b", # "Serial: 99281"
        ],
        "confidence": 0.95,
        "validation": lambda m: True
    },
    "error_code": {
        "patterns": [
            r"\bE-\d{2,4}\b",                        # E-47, E-1234
            r"\bErr(?:or)?\s*(?:code)?\s*:?\s*(\w+-?\d+)\b",
            r"\bFault\s*(?:code)?\s*:?\s*(\w+-?\d+)\b",
        ],
        "confidence": 0.98,
        "validation": lambda m: True
    },
    "part_number": {
        "patterns": [
            r"\bP/N\s?\d{4,6}-[A-Z]\b",             # P/N 7832-A
            r"\bPart\s*#?\s*:?\s*(\d{4,6}-[A-Z])\b",
        ],
        "confidence": 0.95,
        "validation": lambda m: True
    },
    "warranty_id": {
        "patterns": [
            r"\bWRN-\d{6}\b",                        # WRN-123456
            r"\bWarranty\s*#?\s*:?\s*([\w-]+)\b",
        ],
        "confidence": 0.90,
        "validation": lambda m: True
    },
    "plant_id": {
        "patterns": [
            r"\bPlant\s*(?:#|ID|:)?\s*(\w{1,3}-?\d{1,5})\b",  # Plant 7, Plant P-007
            r"\bP-\d{3}\b",                                      # P-007
        ],
        "confidence": 0.90,
        "validation": lambda m: True
    },
    "date_explicit": {
        "patterns": [
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",             # 03/04/2025
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s*\d{4}\b",
        ],
        "confidence": 0.85,
        "validation": lambda m: True
    },
}
```

**Decision:** Store patterns as configuration, not code.

**Reasoning:** New product lines, error code formats, and part numbering schemes are introduced regularly. Patterns stored in a config file or parameter store can be updated without code deployment. A product manager or support engineer can add a pattern when a new product launches without waiting for a development cycle.

**Implementation details:**

```python
import re
from dataclasses import dataclass

@dataclass
class ExtractedEntity:
    text: str              # The matched text
    entity_type: str       # "model_number", "error_code", etc.
    start: int             # Character offset start
    end: int               # Character offset end
    confidence: float      # Pattern confidence
    source: str            # "regex" or "spacy"
    pattern_id: str        # Which pattern matched (for debugging)

def extract_tier1(text: str) -> list[ExtractedEntity]:
    entities = []
    
    for entity_type, config in ENTITY_PATTERNS.items():
        for i, pattern in enumerate(config["patterns"]):
            for match in re.finditer(pattern, text):
                matched_text = match.group(0)
                
                # Run validation function
                if config["validation"](matched_text):
                    entities.append(ExtractedEntity(
                        text=matched_text,
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=config["confidence"],
                        source="regex",
                        pattern_id=f"{entity_type}_pattern_{i}"
                    ))
    
    # Deduplicate overlapping matches (keep highest confidence)
    entities = resolve_overlaps(entities)
    return entities
```

**Overlap Resolution:**

When multiple patterns match overlapping text spans, keep the most specific (longest) match with the highest confidence. For example, if "Model XR-440" matches both a model_number pattern and a generic alphanumeric pattern, keep the model_number match.

```python
def resolve_overlaps(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    # Sort by start position, then by length (longer first), then by confidence
    entities.sort(key=lambda e: (e.start, -(e.end - e.start), -e.confidence))
    
    resolved = []
    last_end = -1
    
    for entity in entities:
        if entity.start >= last_end:
            resolved.append(entity)
            last_end = entity.end
        # If overlapping, skip (the first one is already kept, 
        # which is longer/higher-confidence due to sorting)
    
    return resolved
```

### Tier 2 — spaCy NER Entity Extraction

**What it catches that regex doesn't:**

- Component names in natural language: "the fan motor assembly", "coolant line near the valve"
- Temporal references: "since last month", "after the repair in January"
- Symptom descriptions: "grinding noise", "visible leak", "intermittent shutdown"
- Informal plant/location references: "the Chicago site", "our west coast facility"

**Implementation:**

```python
# Custom entity types for domain-specific NER
DOMAIN_ENTITY_MAP = {
    # spaCy standard labels → our entity types
    "PERSON": None,       # Already handled in PII step, skip here
    "ORG": "organization",
    "GPE": "location",
    "DATE": "date_natural",
    "TIME": "time_reference",
    "CARDINAL": "quantity",
    "ORDINAL": "ordinal",
}

def extract_tier2(text: str, nlp) -> list[ExtractedEntity]:
    doc = nlp(text)
    entities = []
    
    for ent in doc.ents:
        entity_type = DOMAIN_ENTITY_MAP.get(ent.label_)
        if entity_type:  # Skip unmapped labels
            entities.append(ExtractedEntity(
                text=ent.text,
                entity_type=entity_type,
                start=ent.start_char,
                end=ent.end_char,
                confidence=0.75,  # NER confidence is lower than regex
                source="spacy",
                pattern_id=f"spacy_{ent.label_}"
            ))
    
    return entities
```

**Decision:** Use standard spaCy model, not a custom-trained domain model for Tier 2.

**Reasoning:** A custom domain NER model would be more accurate for component names and symptoms, but requires labeled training data (500+ annotated service orders), training infrastructure, and ongoing retraining as domain vocabulary evolves. At this stage, the standard spaCy model provides adequate NER for dates, locations, and quantities. The fine-tuned LLM in Phase 3 handles the domain-specific semantic extraction that spaCy can't do well. This is the tiered extraction philosophy — each tier handles what it's best at.

**Tradeoff:** Standard spaCy misses domain-specific entities like component names ("fan motor assembly" isn't a standard NER category). These are handled by the LLM in Phase 3. If we later find that having component names before segmentation significantly improves segmentation quality, we'd invest in training a custom spaCy NER model as a Tier 2.5 step.

### Merging Tier 1 and Tier 2 Results

```python
def merge_extractions(tier1: list, tier2: list) -> list[ExtractedEntity]:
    all_entities = tier1 + tier2
    
    # Resolve cross-tier overlaps
    # Priority: regex > spaCy (regex is deterministic and higher confidence)
    all_entities.sort(key=lambda e: (
        e.start,
        0 if e.source == "regex" else 1,  # regex first
        -e.confidence
    ))
    
    merged = []
    last_end = -1
    
    for entity in all_entities:
        if entity.start >= last_end:
            merged.append(entity)
            last_end = entity.end
        elif entity.source != merged[-1].source:
            # Same span, different sources — keep both for cross-validation
            merged.append(entity)
    
    return merged
```

**Rule:** When regex and spaCy both extract an entity at the same position but with different values or types, BOTH are kept with their respective source tags. The conflict is resolved downstream in Phase 3 cross-validation.

### Observability at Step 6

**Metrics emitted:**
- `extraction.tier1.entity_count` — count by entity type
- `extraction.tier1.pattern_hit_distribution` — which patterns are matching most
- `extraction.tier2.entity_count` — count by entity type
- `extraction.tier2.processing_time_ms` — spaCy inference latency
- `extraction.merge.conflicts` — count of cross-tier overlaps
- `extraction.total_entities` — combined entity count per order
- `extraction.entity_density` — entities per 100 tokens (track over time)

**Alerts:**
- If `extraction.tier1.entity_count` for model_number drops below historical mean by 20% → new product naming convention not covered by patterns
- If `extraction.entity_density` drops significantly → extraction degradation, investigate
- If `extraction.tier2.processing_time_ms` exceeds 100ms → text might be unusually long or spaCy model issue
- If a specific `pattern_id` never matches across 1000+ orders → dead pattern, consider removing to reduce processing overhead

---

## 8. Step 7 — Segmentation

### What Happens

Multi-issue service orders are split into separate segments, each representing a single defect or issue. Each segment gets its own independent pass through the LangGraph agent.

### Why Segmentation Matters

If a service order describes three defects and you process it as one blob, the LLM in Phase 3 focuses on the most prominent defect and underweights the others. RAG retrieval in Phase 4 searches for one SOP when three different SOPs are needed. Action item generation in Phase 5 produces one action plan when three are needed. Result: two defects get missed entirely — no action items, no repairs, potentially unsafe conditions.

### Segmentation Strategy — Hybrid Multi-Signal Scoring

**Four signals combined with weighted scoring:**

**Signal 1 — Discourse Markers (Weight: 0.30)**

Linguistic patterns that indicate topic shifts. These are the most reliable signals when present.

```python
SEGMENT_MARKERS = {
    "ordinal": {
        "patterns": [
            r"\b(?:first(?:ly)?|second(?:ly)?|third(?:ly)?|fourth)\b[,:\s]",
            r"\b(?:issue|problem|defect)\s*#?\s*\d+",
            r"^\s*\d+[.)]\s+",
            r"^\s*[-•]\s+",
        ],
        "strength": 0.9,  # high confidence split signal
    },
    "additive": {
        "patterns": [
            r"\b(?:also|additionally|another|separately)\b",
            r"\b(?:on top of that|in addition|besides that)\b",
            r"\b(?:the other (?:issue|problem|defect))\b",
        ],
        "strength": 0.7,  # moderate confidence
    },
    "contrastive": {
        "patterns": [
            r"\b(?:however|but|unlike|whereas)\b.*\b(?:issue|problem|unit)\b",
        ],
        "strength": 0.6,  # lower confidence — "but" doesn't always mean new issue
    },
}
```

**Signal 2 — Entity Discontinuity (Weight: 0.30)**

Uses pre-extracted entities from Step 6 to detect topic shifts based on which entities appear near which sentences.

The algorithm:
1. Split text into sentences
2. For each sentence, identify which extracted entities fall within it (using character offsets)
3. Compute entity overlap between consecutive sentences
4. Low overlap between consecutive sentences = candidate split point

```python
def compute_entity_overlap(entities_a: set, entities_b: set) -> float:
    if not entities_a and not entities_b:
        return 1.0  # No entities in either — assume same topic
    if not entities_a or not entities_b:
        return 0.5  # One has entities, other doesn't — uncertain
    
    # Compare component types specifically (strongest signal)
    components_a = {e.text for e in entities_a if e.entity_type == "component"}
    components_b = {e.text for e in entities_b if e.entity_type == "component"}
    
    if components_a and components_b:
        component_overlap = len(components_a & components_b) / len(components_a | components_b)
    else:
        component_overlap = 0.5  # unknown
    
    # Compare error codes
    errors_a = {e.text for e in entities_a if e.entity_type == "error_code"}
    errors_b = {e.text for e in entities_b if e.entity_type == "error_code"}
    
    if errors_a and errors_b:
        error_overlap = len(errors_a & errors_b) / len(errors_a | errors_b)
    else:
        error_overlap = 0.5
    
    return component_overlap * 0.6 + error_overlap * 0.4
```

**Signal 3 — Semantic Similarity Drop (Weight: 0.25)**

Embedding-based topic change detection between consecutive sentences.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")  # fast, good quality

def compute_similarity_drops(sentences: list[str]) -> list[float]:
    if len(sentences) < 2:
        return []
    
    embeddings = embedder.encode(sentences)
    drops = []
    
    for i in range(1, len(embeddings)):
        similarity = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
        )
        drop = 1.0 - similarity  # Higher = bigger topic shift
        drops.append(drop)
    
    return drops
```

**Decision:** Use `all-MiniLM-L6-v2` for sentence embeddings in segmentation.

**Reasoning:** This model is small (80MB), fast (~5ms per sentence), and produces good-quality 384-dimensional embeddings. We don't need the highest possible embedding quality here — we just need to detect relative similarity drops between consecutive sentences, which this model handles well.

**Tradeoff:** A larger model like `all-mpnet-base-v2` would produce slightly better embeddings but at 3x the latency. Since segmentation runs on every order and the similarity drop detection is relative (we're comparing ratios, not absolute values), the smaller model is sufficient.

**Signal 4 — Structural Cues (Weight: 0.15)**

```python
def check_structural_break(text: str, position: int) -> float:
    # Check for double newline (paragraph break)
    if text[max(0, position-3):position+3].count("\n\n") > 0:
        return 0.8
    
    # Check for line break with indent change
    # (new paragraph starting differently from previous)
    return 0.0
```

**Combined Scoring:**

```python
def find_split_points(text: str, entities: list, threshold: float = 0.50):
    sentences = split_into_sentences(text)
    
    # Compute all signals
    discourse_scores = compute_discourse_signals(sentences)
    entity_scores = compute_entity_discontinuity(sentences, entities)
    semantic_scores = compute_similarity_drops(sentences)
    structural_scores = compute_structural_breaks(text, sentences)
    
    split_points = []
    
    for i in range(1, len(sentences)):
        combined_score = (
            discourse_scores[i]   * 0.30 +
            entity_scores[i]      * 0.30 +
            semantic_scores[i-1]  * 0.25 +
            structural_scores[i]  * 0.15
        )
        
        if combined_score >= threshold:
            split_points.append({
                "position": i,
                "score": combined_score,
                "signals": {
                    "discourse": discourse_scores[i],
                    "entity": entity_scores[i],
                    "semantic": semantic_scores[i-1],
                    "structural": structural_scores[i],
                }
            })
    
    return split_points
```

**Decision:** Threshold of 0.50 for splitting.

**Reasoning:** At 0.50, you need at least two strong signals agreeing (e.g., discourse marker at 0.9 * 0.30 = 0.27 plus entity discontinuity at 0.9 * 0.30 = 0.27 = 0.54, above threshold). A single weak signal alone can't trigger a split. This reduces false positives while catching most true splits.

**Tradeoff:** Some genuine two-issue orders where the technician seamlessly transitions between topics without any linguistic markers AND uses similar terminology for both issues will not be split. These are rare and are caught downstream when the LLM in Phase 3 extracts multiple defect objects from a single segment — Phase 5 validation flags this as a potential under-segmentation.

### Post-Segmentation: Context Inheritance

After splitting, each segment needs shared context from the preamble:

```python
def apply_context_inheritance(text: str, segments: list[dict], entities: list) -> list[dict]:
    if not segments[0].get("split_points"):
        # No splits — single segment with all context
        return [{"text": text, "entities": entities, "inherited_context": {}}]
    
    first_split = segments[0]["split_points"][0]["position"]
    
    # Everything before first split is the preamble
    preamble_sentences = split_into_sentences(text)[:first_split]
    preamble_text = " ".join(preamble_sentences)
    
    # Extract shared entities from preamble
    preamble_entities = [e for e in entities 
                         if e.end <= len(preamble_text)]
    
    shared_metadata = {
        "model_number": next((e.text for e in preamble_entities 
                             if e.entity_type == "model_number"), None),
        "plant_id": next((e.text for e in preamble_entities 
                         if e.entity_type == "plant_id"), None),
        "serial_number": next((e.text for e in preamble_entities 
                              if e.entity_type == "serial_number"), None),
        "preamble_text": preamble_text,
    }
    
    # Each segment inherits shared metadata
    enriched_segments = []
    for segment in segments:
        segment["inherited_context"] = shared_metadata
        enriched_segments.append(segment)
    
    return enriched_segments
```

### Cross-Reference Detection

```python
CROSS_REF_PATTERNS = [
    r"caused by (?:the )?(previous|above|first|other)",
    r"(?:as a )?result of (?:the )?(previous|above|first|other)",
    r"related to (?:the )?(previous|above|first|other)",
    r"same (?:root )?cause",
    r"connected to",
    r"led to",
    r"triggered",
]

def detect_cross_references(segments: list[dict]) -> list[dict]:
    for i, segment in enumerate(segments):
        text = segment["text"].lower()
        for pattern in CROSS_REF_PATTERNS:
            if re.search(pattern, text):
                segment["cross_references"] = {
                    "type": "causal",
                    "related_segments": [j for j in range(len(segments)) if j != i],
                    "pattern_matched": pattern,
                }
                break
    return segments
```

### Observability at Step 7

**Metrics emitted:**
- `segmentation.segments_per_order` — distribution (histogram)
- `segmentation.single_segment_rate` — percentage of orders not split
- `segmentation.multi_segment_rate` — percentage of orders split
- `segmentation.avg_segment_length_tokens` — average segment size
- `segmentation.min_segment_length_tokens` — smallest segment (alert if too small)
- `segmentation.signal_contributions` — which signals triggered splits (distribution)
- `segmentation.cross_references_detected` — count of cross-referenced segments
- `segmentation.processing_time_ms` — total segmentation latency
- `segmentation.threshold_scores` — distribution of combined scores at split points

**Alerts:**
- If `segmentation.segments_per_order` average shifts by >20% from historical baseline → segmentation behavior changed, investigate
- If `segmentation.min_segment_length_tokens` drops below 15 → false splits creating fragments
- If `segmentation.single_segment_rate` drops below 50% (if historically ~70%) → over-splitting
- If `segmentation.cross_references_detected` suddenly spikes → check if pattern matching is too broad
- Weekly evaluation: sample 50 multi-segment orders, have human annotators verify segmentation quality. Compute precision, recall, F1 against human judgment. Alert if F1 drops below 0.85.

---

## 9. Step 8 — Chunking (Conditional)

### What Happens

If any segment exceeds the LLM's practical context limit, it's chunked into smaller pieces. For most service orders, this step is skipped.

### Decision Logic

```python
MAX_SEGMENT_TOKENS = 1500  # Conservative limit for Llama 3.1 8B effective context

def should_chunk(segment: dict, token_counter) -> bool:
    token_count = token_counter(segment["text"])
    return token_count > MAX_SEGMENT_TOKENS
```

**Decision:** Chunk threshold of 1,500 tokens.

**Reasoning:** The fine-tuned Llama 3.1 8B in Phase 3 technically supports 128K context, but extraction quality degrades above 4K tokens for an 8B model. The LLM prompt in Phase 3 uses approximately: 500 tokens for system prompt, 200 tokens for pre-extracted entities, 300 tokens for few-shot examples, and we need 500+ tokens for the model's output. This leaves roughly 2,500 tokens for the actual service order text. Setting the threshold at 1,500 tokens provides comfortable headroom.

**Tradeoff:** A higher threshold (e.g., 3,000 tokens) would avoid chunking more often but risk degraded extraction quality on longer texts. A lower threshold (e.g., 500 tokens) would chunk more aggressively but lose context. At 1,500, we chunk only the longest 2-3% of segments (those where a technician pasted an extensive maintenance history), keeping the common case unchunked.

### Chunking Strategy (When Applied)

**Sentence-boundary chunking with overlap:**

```python
import nltk

def chunk_segment(text: str, max_tokens: int = 1500, 
                  overlap_tokens: int = 200,
                  token_counter=None) -> list[dict]:
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = token_counter(sentence)
        
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            # Finalize current chunk
            chunks.append({
                "text": " ".join(current_chunk),
                "chunk_index": len(chunks),
                "token_count": current_tokens,
            })
            
            # Start new chunk with overlap (last N tokens from previous)
            overlap_sentences = get_overlap_sentences(
                current_chunk, overlap_tokens, token_counter
            )
            current_chunk = overlap_sentences + [sentence]
            current_tokens = sum(token_counter(s) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append({
            "text": " ".join(current_chunk),
            "chunk_index": len(chunks),
            "token_count": current_tokens,
        })
    
    return chunks
```

**Decision:** Sentence-boundary chunking, not recursive or semantic chunking.

**Reasoning:** Service order text is narrative prose from technicians, not structured documents with headings and sections. Recursive chunking (designed for structured documents) has no structural boundaries to leverage. Semantic chunking is overkill — we already did segmentation in Step 7 to split by topic, so within a single segment the content should be about one issue. The only reason we're chunking here is length, not topic mixing. Sentence-boundary chunking is the simplest approach that preserves complete thoughts.

**Tradeoff:** If a single sentence exceeds the max token limit (extremely rare but theoretically possible if a technician writes a 200-word run-on sentence), the chunker would need to fall back to word-level splitting. We handle this as an edge case:

```python
if sentence_tokens > max_tokens:
    # Edge case: single sentence too long
    words = sentence.split()
    mid = len(words) // 2
    # Split at the nearest comma, period, or semicolon near the midpoint
    # If none found, split at midpoint
```

### Observability at Step 8

**Metrics emitted:**
- `chunking.triggered_rate` — percentage of segments that needed chunking
- `chunking.chunks_per_segment` — when chunking occurs, how many chunks
- `chunking.avg_chunk_tokens` — average token count per chunk
- `chunking.overlap_tokens` — actual overlap used
- `chunking.edge_case_splits` — count of single-sentence overflow splits

**Alerts:**
- If `chunking.triggered_rate` exceeds 10% → input patterns may have changed (longer texts), investigate
- If `chunking.edge_case_splits` exceeds 0 → unusually long sentences, review the source data
- If `chunking.chunks_per_segment` exceeds 5 → extremely long segments getting through segmentation, review Step 7

---

## 10. Step 9 — Canonical Schema Assembly

### What Happens

All preprocessed, extracted, segmented, and optionally chunked data is assembled into the final canonical JSON schema that the LangGraph agent (Phase 2) expects.

### The Canonical Schema

```json
{
    "order_id": "SO-28841",
    "source_channel": "email",
    "received_at": "2025-03-04T14:22:00Z",
    "processed_at": "2025-03-04T14:22:03.451Z",
    "processing_metadata": {
        "pipeline_version": "1.4.2",
        "preprocessing_duration_ms": 3451,
        "language_detected": "en",
        "language_confidence": 0.97,
        "was_translated": false,
        "pii_entities_redacted": 3,
        "total_entities_extracted": 8,
        "segments_produced": 2,
        "chunking_applied": false
    },
    "structured_metadata": {
        "plant_id": "P-007",
        "model_number": "XR-440",
        "serial_number": "SN-99281",
        "warranty_id": null,
        "priority": "high",
        "submitted_by": "[EMAIL_1]",
        "customer_id": "C-4421"
    },
    "segments": [
        {
            "segment_id": "SO-28841-S1",
            "segment_index": 0,
            "text": "Compressor making grinding noise, error E-47 on panel, unit is 18 months old under warranty.",
            "token_count": 22,
            "inherited_context": {
                "model_number": "XR-440",
                "plant_id": "P-007",
                "serial_number": "SN-99281",
                "preamble_text": "Two issues with the XR-440 at Plant 7."
            },
            "pre_extracted_entities": [
                {
                    "text": "E-47",
                    "entity_type": "error_code",
                    "start": 45,
                    "end": 49,
                    "confidence": 0.98,
                    "source": "regex"
                },
                {
                    "text": "18 months",
                    "entity_type": "date_natural",
                    "start": 67,
                    "end": 76,
                    "confidence": 0.75,
                    "source": "spacy"
                }
            ],
            "cross_references": null,
            "chunks": null,
            "flags": {
                "has_attachment_reference": false,
                "mixed_language": false,
                "low_confidence_entities": false
            }
        },
        {
            "segment_id": "SO-28841-S2",
            "segment_index": 1,
            "text": "Coolant line near valve assembly P/N 7832-A has visible leak, pressure reading dropping steadily.",
            "token_count": 18,
            "inherited_context": {
                "model_number": "XR-440",
                "plant_id": "P-007",
                "serial_number": "SN-99281",
                "preamble_text": "Two issues with the XR-440 at Plant 7."
            },
            "pre_extracted_entities": [
                {
                    "text": "P/N 7832-A",
                    "entity_type": "part_number",
                    "start": 32,
                    "end": 42,
                    "confidence": 0.95,
                    "source": "regex"
                }
            ],
            "cross_references": null,
            "chunks": null,
            "flags": {
                "has_attachment_reference": false,
                "mixed_language": false,
                "low_confidence_entities": false
            }
        }
    ],
    "raw_archive": {
        "raw_text_hash": "sha256:a1b2c3d4...",
        "storage_location": "s3://raw-orders/2025/03/04/SO-28841.json"
    }
}
```

### Schema Validation

**Decision:** Validate the canonical schema with Pydantic before dispatch.

**Reasoning:** If a malformed object enters the LangGraph agent, it can cause cryptic failures deep in the pipeline that are hard to debug. Validating at the Phase 1 boundary catches issues immediately.

```python
from pydantic import BaseModel, validator
from typing import Optional
from datetime import datetime

class ExtractedEntitySchema(BaseModel):
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float
    source: str

class SegmentSchema(BaseModel):
    segment_id: str
    segment_index: int
    text: str
    token_count: int
    inherited_context: dict
    pre_extracted_entities: list[ExtractedEntitySchema]
    cross_references: Optional[dict] = None
    chunks: Optional[list] = None
    flags: dict

    @validator("text")
    def text_not_empty(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Segment text too short — likely a segmentation error")
        return v

    @validator("token_count")
    def token_count_reasonable(cls, v):
        if v > 5000:
            raise ValueError("Segment token count exceeds limit — chunking should have been applied")
        return v

class CanonicalOrderSchema(BaseModel):
    order_id: str
    source_channel: str
    received_at: datetime
    processed_at: datetime
    processing_metadata: dict
    structured_metadata: dict
    segments: list[SegmentSchema]
    raw_archive: dict

    @validator("segments")
    def at_least_one_segment(cls, v):
        if len(v) == 0:
            raise ValueError("Order must have at least one segment")
        return v
```

**Rule:** If schema validation fails, the order is routed to a dead letter queue for manual investigation. It does NOT silently proceed.

### Observability at Step 9

**Metrics emitted:**
- `schema.validation_pass_rate` — percentage passing Pydantic validation
- `schema.validation_failures` — count and reason distribution
- `schema.avg_segments_per_order` — final segment count
- `schema.avg_entities_per_segment` — final entity density
- `schema.total_processing_time_ms` — end-to-end Phase 1 latency

**Alerts:**
- If `schema.validation_pass_rate` drops below 95% → systematic issue in upstream steps
- If `schema.validation_failures` shows a specific reason spiking → targeted investigation

---

## 11. Step 10 — Queue Dispatch to Phase 2

### What Happens

The validated canonical JSON is published to the dispatch queue, where Phase 2 (LangGraph agent) workers pick it up for processing.

### Implementation

```python
import json
import boto3

sqs = boto3.client("sqs")
DISPATCH_QUEUE_URL = "https://sqs.us-west-2.amazonaws.com/123456/agent-dispatch-queue"

def dispatch_to_phase2(canonical_order: dict):
    # Each segment is dispatched as a separate message
    for segment in canonical_order["segments"]:
        message = {
            "order_id": canonical_order["order_id"],
            "segment_id": segment["segment_id"],
            "segment": segment,
            "structured_metadata": canonical_order["structured_metadata"],
            "processing_metadata": canonical_order["processing_metadata"],
        }
        
        sqs.send_message(
            QueueUrl=DISPATCH_QUEUE_URL,
            MessageBody=json.dumps(message),
            MessageGroupId=canonical_order["order_id"],  # FIFO: segments for same order grouped
            MessageDeduplicationId=segment["segment_id"],
        )
```

**Decision:** Dispatch each segment as a separate message, not the entire order as one message.

**Reasoning:** Each segment needs its own independent pass through the LangGraph agent (its own LLM extraction, its own RAG retrieval, its own action items). By dispatching segments individually, they can be processed in parallel by different agent workers, improving throughput. The `MessageGroupId` ensures that segments from the same order are processed in order within their group (if ordering matters for cross-reference resolution).

**Tradeoff:** More messages to manage, and the downstream system needs to reassemble results per order after all segments are processed. This is handled by an aggregation step after Phase 5 that collects all segment results for an order_id and produces the final unified output. The parallelism benefit outweighs the reassembly complexity.

**Alternative Considered:** Dispatch the entire order as one message and have the agent process segments sequentially. Rejected because it creates a bottleneck — a 5-segment order would take 5x as long as a single-segment order, and the agent worker is blocked for the entire duration.

### Dead Letter Queue

Messages that fail processing after N retries (default: 3) are routed to a dead letter queue (DLQ):

```python
# DLQ configuration on the dispatch queue
{
    "RedrivePolicy": {
        "deadLetterTargetArn": "arn:aws:sqs:us-west-2:123456:agent-dispatch-dlq",
        "maxReceiveCount": 3
    }
}
```

**Rule:** DLQ messages are reviewed daily. Common failure reasons are logged and tracked as a metric. If the same failure pattern appears repeatedly, a fix is prioritized.

### Observability at Step 10

**Metrics emitted:**
- `dispatch.messages_sent` — count of messages published to Phase 2 queue
- `dispatch.messages_per_order` — how many messages per order (should match segment count)
- `dispatch.publish_latency_ms` — SQS publish latency
- `dispatch.dlq_depth` — number of messages in dead letter queue
- `dispatch.dlq_reasons` — distribution of failure reasons in DLQ

**Alerts:**
- If `dispatch.dlq_depth` exceeds 10 → systematic processing failures, investigate immediately
- If `dispatch.publish_latency_ms` exceeds 500ms → SQS performance issue
- If `dispatch.messages_sent` drops to 0 for >15 minutes during business hours → Phase 1 pipeline may be stuck

---

## 12. Observability & Alerting — Complete Phase 1 Dashboard

### Dashboard Layout

**Panel 1 — Intake Health**
- Orders received per minute (by channel)
- Duplicate detection rate
- Channel availability status

**Panel 2 — Processing Pipeline**
- End-to-end Phase 1 latency (P50, P95, P99)
- Processing throughput (orders/minute)
- Step-by-step latency breakdown (waterfall chart)

**Panel 3 — Data Quality**
- Text reduction ratio (cleaning effectiveness)
- PII detection counts (by type)
- Entity extraction counts (by tier, by type)
- Segmentation distribution (segments per order)
- Schema validation pass rate

**Panel 4 — Failure & Recovery**
- Processing error rate
- DLQ depth and age of oldest message
- Manual review queue depth
- Retry rate per step

### Consolidated Alert Rules

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Channel down | Zero orders from a channel for 30min | P1 | Check integration, contact source team |
| High duplicate rate | Duplicates >20% of intake | P2 | Check source system retry config |
| Text over-reduction | Cleaning removes >70% of text | P2 | Review boilerplate/signature patterns |
| PII leak risk | PII detection drops to near-zero | P1 | PII detection might be broken |
| Extraction degradation | Entity density drops >20% from baseline | P2 | Check regex patterns, review new data |
| Over-segmentation | Avg segments/order increases >20% | P2 | Review segmentation threshold |
| Under-segmentation | Downstream duplicate action items spike | P2 | Review segmentation sensitivity |
| Schema validation drop | Validation pass rate below 95% | P1 | Systematic upstream issue |
| DLQ growth | DLQ depth exceeds 10 | P1 | Processing failures, investigate |
| Latency degradation | P95 latency exceeds 10 seconds | P2 | Performance bottleneck, scale workers |

### Logging Strategy

**Decision:** Structured JSON logging at every step.

Every log line includes: `order_id`, `step_name`, `duration_ms`, `input_size`, `output_size`, and step-specific fields. Logs flow to CloudWatch / ELK / Datadog with searchable fields.

```json
{
    "timestamp": "2025-03-04T14:22:01.234Z",
    "order_id": "SO-28841",
    "step": "extraction_tier1",
    "duration_ms": 12,
    "entities_found": 4,
    "entity_types": ["model_number", "error_code", "part_number", "serial_number"],
    "patterns_matched": ["model_number_pattern_0", "error_code_pattern_0"],
    "level": "INFO"
}
```

**Rule:** Never log PII in cleartext. Redacted values are logged as `[REDACTED]`. The PII index is stored in a separate encrypted store, not in application logs.

**Rule:** Log at DEBUG level for step-internal details, INFO level for step completion summaries, WARN level for anomalies (high reduction ratio, encoding fallback, edge cases), ERROR level for processing failures.

---

## 13. End-to-End Phase 1 Flow Summary

```
SERVICE ORDER ARRIVES
        │
Step 1 ─┤  Source Intake & Channel Routing
        │  • Receive from email/webform/ERP
        │  • Route to appropriate handler
        │  • Idempotency check (dedup)
        │  • Publish to intake queue
        │
Step 2 ─┤  Raw Text Extraction
        │  • MIME parsing (email) or field extraction (webform/ERP)
        │  • HTML tag stripping (preserve structural whitespace)
        │  • Reply chain removal (first-match cutoff)
        │  • Signature removal (library + regex fallback)
        │  • Attachment reference detection (flag, don't process)
        │
Step 3 ─┤  Text Cleaning & Normalization
        │  • Encoding normalization (detect → fallback chain → replace)
        │  • Unicode normalization (NFC)
        │  • Smart quote & special character normalization
        │  • Whitespace normalization (preserve paragraph breaks)
        │  • Boilerplate removal (configurable patterns)
        │  • Case preservation (do NOT lowercase)
        │
Step 4 ─┤  PII Detection & Redaction
        │  • Regex PII (email, phone, SSN, credit card)
        │  • NER PII (spaCy: person names)
        │  • Context-aware decisions (business locations preserved)
        │  • Redaction with placeholder tokens ([PERSON], [PHONE])
        │  • PII index stored encrypted, separately
        │
Step 5 ─┤  Language Detection & Handling
        │  • Detect language + confidence
        │  • Route: English → continue, Non-English → translate, Uncertain → manual review
        │  • Domain-specific translation glossary
        │
Step 6 ─┤  Entity Pre-Extraction (Tier 1 + Tier 2)
        │  • Tier 1: Regex (model numbers, serial numbers, error codes, part numbers, dates)
        │  • Tier 2: spaCy NER (locations, dates in natural language, quantities)
        │  • Merge with overlap resolution (regex priority)
        │  • Store with character offsets for cross-validation
        │
Step 7 ─┤  Segmentation
        │  • Hybrid multi-signal scoring:
        │    - Discourse markers (0.30 weight)
        │    - Entity discontinuity (0.30 weight)
        │    - Semantic similarity drop (0.25 weight)
        │    - Structural cues (0.15 weight)
        │  • Threshold-based splitting (score >= 0.50)
        │  • Context inheritance (preamble → all segments)
        │  • Cross-reference detection between segments
        │
Step 8 ─┤  Chunking (Conditional)
        │  • Only if segment exceeds 1,500 tokens (rare: ~2-3% of segments)
        │  • Sentence-boundary chunking with 200-token overlap
        │  • Token-measured (not character-measured)
        │
Step 9 ─┤  Canonical Schema Assembly
        │  • Assemble all data into canonical JSON
        │  • Pydantic schema validation
        │  • Fail → dead letter queue for manual review
        │
Step 10─┤  Queue Dispatch to Phase 2
        │  • Each segment dispatched as separate message
        │  • FIFO ordering within order (MessageGroupId)
        │  • Dead letter queue after 3 failed attempts
        │
        ▼
PHASE 2 — LangGraph Agent receives clean, structured,
          segmented canonical input
```

### Key Design Principles Applied Throughout Phase 1

1. **Fail early, fail loudly.** Schema validation at the boundary catches issues before they silently propagate.
2. **Configuration over code.** Regex patterns, boilerplate patterns, PII patterns, translation glossaries — all stored externally and updatable without deployment.
3. **Tiered processing.** Use the cheapest, fastest, most reliable method first. Only escalate to more expensive methods for what simpler methods can't handle.
4. **Cross-validation.** Multiple extraction methods checking each other's work.
5. **Observability as a first-class concern.** Every step emits metrics, structured logs, and has alert thresholds.
6. **Err toward safety.** Over-redact PII, under-split rather than over-split on uncertain segmentation signals, archive raw data for recovery.
7. **Idempotency everywhere.** Duplicate messages produce the same output without side effects.

---

*Document Version: 1.0 | Last Updated: March 2026 | System: LGS Tech Agentic AI Workflow*
