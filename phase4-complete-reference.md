# Phase 4 — RAG Pipeline (Policy Grounding): Complete Production Reference

## LGS Tech Agentic AI Workflow — Service Order Processing

---

## Table of Contents

1. Phase 4 Overview & Why RAG
2. Why RAG Instead of Fine-Tuning for Policies
3. Document Corpus — What Gets Indexed
4. Offline Indexing Pipeline
5. Chunking Strategy for SOPs and Warranty Documents
6. Embedding Model Selection
7. Pinecone Configuration & Index Design
8. Metadata Schema & Filtering Strategy
9. Query Construction from Extracted Defects
10. Retrieval — Vector Search + Metadata Filtering
11. Reranking with Cross-Encoder
12. Query Reformulation on Low Relevance
13. Context Window Expansion (Neighbor Fetching)
14. Context Assembly for LLM Prompt
15. Document Lifecycle — Updates, Versioning, Re-Indexing
16. Evaluation with Ragas
17. Failure Modes & Graceful Degradation
18. Observability & Alerting
19. End-to-End Phase 4 Flow Summary

---

## 1. Phase 4 Overview & Why RAG

### What Phase 4 Does

Phase 4 is the knowledge grounding layer. After Phase 3 extracts structured defect information (defect type, severity, component, error codes), Phase 4 retrieves the specific SOP procedures, warranty policies, and repair guidelines that apply to that exact defect. The retrieved documents are injected into the LLM prompt in Phase 5 (action item generation) so the model generates action items grounded in actual current policy — not hallucinated from training data.

### The Core Problem RAG Solves

The fine-tuned Llama 3.1 8B knows how to extract defects and how to generate action items. But it doesn't know what the current warranty policy says for model XR-440 compressors, or what the specific repair procedure is for error code E-47, or whether a 18-month-old unit qualifies for warranty coverage under policy version 4.2.

This knowledge is:
- **Frequently updated.** Warranty policies change quarterly. SOPs are revised when new repair techniques are developed. Compliance requirements change with regulations.
- **Precise.** A warranty policy that covers 24 months is different from one that covers 36 months. "Close enough" is not acceptable for warranty determinations.
- **Document-grounded.** Every action item needs to reference a specific document and section. "Covered under warranty" isn't useful — "covered under Warranty Policy XR-Series v4.2, Section 3.2, standard coverage term" is actionable.

Fine-tuning can't solve this because training data becomes stale the moment a policy changes. RAG solves it by retrieving the current version of the relevant document at query time.

---

## 2. Why RAG Instead of Fine-Tuning for Policies

### The Separation Principle

**Fine-tuning teaches the model HOW to reason.** It learns extraction patterns, domain vocabulary, JSON output formatting, severity classification logic. These are stable capabilities that change slowly (if ever).

**RAG provides WHAT to reason about.** It supplies the current warranty terms, the latest SOP procedures, the specific compliance requirements. These change frequently and must be current.

### Why Not Fine-Tune on Policies?

**Problem 1 — Staleness.** If you fine-tune the model on warranty policy v4.1 and then the company releases v4.2, the model still "knows" v4.1 until you retrain. Retraining takes hours and requires evaluation before deployment. During that gap, every order is processed against an outdated policy.

**Problem 2 — Precision loss.** Fine-tuning encodes knowledge as distributed patterns across billions of parameters. The model "knows" that compressors are generally covered for 24 months, but it doesn't reliably distinguish between 24-month standard coverage and 36-month extended coverage, or remember that units manufactured before January 2024 have different terms. This level of precision requires reading the actual document, not recalling from parametric memory.

**Problem 3 — Attribution.** When a fine-tuned model says "covered under warranty," there's no way to verify which document it's referencing. It might be conflating two different policies. RAG provides explicit attribution — you can trace the action item to the specific retrieved chunk and verify correctness.

**Problem 4 — Scale.** LGS Tech might have 500+ SOP documents, 50+ warranty policies, and 200+ compliance guidelines. Fine-tuning on all of these would require a massive training dataset and would degrade the model's extraction capabilities (catastrophic forgetting). RAG stores these documents externally and retrieves only the relevant 3-5 chunks per query — the model never needs to memorize the entire corpus.

### Decision: Strict Separation

Fine-tuning handles reasoning capabilities (Phase 3). RAG handles knowledge retrieval (Phase 4). They are never mixed — the fine-tuned model is explicitly instructed to base all policy decisions on retrieved context, never on its training data.

**Rule:** If the retrieved context doesn't contain information needed for a decision (e.g., warranty eligibility), the model must set `escalation_required: true` rather than guessing from its parametric knowledge. This is enforced in the system prompt and validated in Phase 2's output validation.

---

## 3. Document Corpus — What Gets Indexed

### Document Categories

**Category 1 — Standard Operating Procedures (SOPs)**

These are step-by-step repair and maintenance procedures organized by product line and component. Example: "SOP-XR-COMP-001: Compressor Replacement Procedure for XR Series."

Characteristics: highly structured (numbered steps, part lists, tool requirements), 5-20 pages each, updated 1-2 times per year, organized by product line → component → procedure type.

**Category 2 — Warranty Policies**

Coverage terms, eligibility criteria, exclusions, and claims procedures. Example: "Warranty Policy XR-Series v4.2: Standard and Extended Coverage Terms."

Characteristics: dense legal/technical language, specific date ranges and conditions, version-controlled (v4.1, v4.2), 10-30 pages each, updated quarterly.

**Category 3 — Repair Procedures**

Detailed technical instructions for diagnosing and fixing specific failure modes. Example: "Troubleshooting Guide: Error Code E-47 — Compressor Overcurrent Fault."

Characteristics: diagnostic decision trees, failure mode analysis, specific to error codes and symptoms, 3-10 pages each, updated as new failure modes are discovered.

**Category 4 — Compliance & Safety Documents**

Regulatory requirements, safety protocols, and mandatory reporting guidelines. Example: "Compliance Requirements: Refrigerant Handling and Reporting under EPA Section 608."

Characteristics: regulatory language, mandatory procedures that override standard SOPs, updated when regulations change.

**Category 5 — Parts Catalogs**

Part numbers, specifications, compatibility matrices, and supplier information. Example: "Parts Catalog: XR-Series Compressor Components."

Characteristics: tabular data, cross-reference tables, frequently updated as parts are added/discontinued.

### Corpus Size

Typical corpus for a company like LGS Tech:

```
Document Category          Count       Avg Pages     Total Pages     Est. Tokens
──────────────────────────────────────────────────────────────────────────────────
SOPs                       200         10            2,000           1,000,000
Warranty Policies          50          20            1,000           500,000
Repair Procedures          150         5             750             375,000
Compliance Documents       30          15            450             225,000
Parts Catalogs             40          8             320             160,000
──────────────────────────────────────────────────────────────────────────────────
TOTAL                      470         —             4,520           ~2,260,000
```

At 512 tokens per chunk with 100-token overlap, this produces approximately 5,500-6,500 chunks indexed in Pinecone.

---

## 4. Offline Indexing Pipeline

### What "Offline" Means

Document indexing does NOT happen at query time. It's a batch process that runs when documents are added, updated, or scheduled for re-indexing. The indexed chunks sit in Pinecone ready for retrieval when a service order arrives. This separation means query-time latency is only the vector search + reranking time, not the document processing time.

### Pipeline Architecture

```
[Document Source]
    │
    ├── SharePoint / Confluence / S3 bucket
    │   (where teams maintain SOPs and policies)
    │
    ▼
[Change Detection]
    │
    ├── Webhook on document update
    ├── Or: scheduled scan every 6 hours
    │
    ▼
[Document Processor]
    │
    ├── PDF extraction (PyMuPDF / pdfplumber)
    ├── DOCX extraction (python-docx / pandoc)
    ├── HTML extraction (BeautifulSoup)
    │
    ▼
[Chunking Engine]
    │
    ├── Structure-aware chunking (respects headings, sections)
    ├── 512 tokens per chunk, 100-token overlap
    ├── Metadata extraction from document structure
    │
    ▼
[Embedding Service]
    │
    ├── Batch embedding via embedding model
    ├── One vector per chunk
    │
    ▼
[Pinecone Upsert]
    │
    ├── Upsert chunks with vectors + metadata
    ├── Version-tagged to support rollback
    │
    ▼
[Validation]
    │
    ├── Spot-check: sample queries against new index
    ├── Compare retrieval quality pre/post update
```

### Implementation

```python
import hashlib
from datetime import datetime

class DocumentIndexer:
    def __init__(self, embedding_model, pinecone_index, chunker):
        self.embedding_model = embedding_model
        self.index = pinecone_index
        self.chunker = chunker
    
    def index_document(self, document: dict) -> dict:
        """Index a single document into Pinecone."""
        doc_id = document["id"]
        doc_text = document["text"]
        doc_metadata = document["metadata"]
        
        # Step 1: Chunk the document
        chunks = self.chunker.chunk(
            text=doc_text,
            doc_metadata=doc_metadata,
        )
        
        # Step 2: Generate content hash for change detection
        content_hash = hashlib.sha256(doc_text.encode()).hexdigest()
        
        # Step 3: Check if document has changed since last indexing
        if self.is_already_indexed(doc_id, content_hash):
            return {"status": "skipped", "reason": "no changes detected"}
        
        # Step 4: Delete old chunks for this document
        self.delete_document_chunks(doc_id)
        
        # Step 5: Embed all chunks in batch
        chunk_texts = [c["text"] for c in chunks]
        embeddings = self.embedding_model.embed_batch(chunk_texts)
        
        # Step 6: Prepare vectors for upsert
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}__chunk_{i:04d}"
            
            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    # Document-level metadata
                    "document_id": doc_id,
                    "document_title": doc_metadata["title"],
                    "document_category": doc_metadata["category"],
                    "product_line": doc_metadata.get("product_line", "general"),
                    "region": doc_metadata.get("region", "global"),
                    "version": doc_metadata.get("version", "1.0"),
                    "effective_date": doc_metadata.get("effective_date", ""),
                    "expiry_date": doc_metadata.get("expiry_date", ""),
                    
                    # Chunk-level metadata
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "section_title": chunk["section_title"],
                    "text": chunk["text"],  # Store text for retrieval
                    "token_count": chunk["token_count"],
                    
                    # Navigation metadata
                    "prev_chunk": f"{doc_id}__chunk_{i-1:04d}" if i > 0 else None,
                    "next_chunk": f"{doc_id}__chunk_{i+1:04d}" if i < len(chunks)-1 else None,
                    
                    # Indexing metadata
                    "content_hash": content_hash,
                    "indexed_at": datetime.utcnow().isoformat(),
                    "index_version": "v2",  # Track indexing pipeline version
                }
            })
        
        # Step 7: Upsert to Pinecone in batches of 100
        for batch_start in range(0, len(vectors), 100):
            batch = vectors[batch_start:batch_start + 100]
            self.index.upsert(vectors=batch)
        
        return {
            "status": "indexed",
            "document_id": doc_id,
            "chunks_created": len(chunks),
            "content_hash": content_hash,
        }
```

### Design Decisions

**Decision:** Delete-then-reinsert for document updates (not in-place update).

**Reasoning:** When a document is updated, the number of chunks might change (a section was added or removed). In-place updating would require tracking which chunks are new, which are modified, and which should be deleted — complex logic with edge cases. Deleting all chunks for the document and reinserting the new set is simpler, atomic, and guarantees no stale chunks remain.

**Tradeoff:** There's a brief window (seconds) during delete-and-reinsert where the document has zero chunks in the index. If a query hits during this window, it won't find chunks from this document. For a 6-hour re-indexing schedule, this window is negligible. For real-time critical documents, you could implement blue-green indexing (index new version alongside old, then swap) at the cost of 2x storage.

**Decision:** Content hash for change detection.

**Reasoning:** Without change detection, every scheduled indexing run would re-embed and re-upsert every document — wasteful and expensive (embedding 2 million tokens costs ~$0.20-0.50 per run, and the compute time is 30-60 minutes). Content hashing detects unchanged documents and skips them, reducing typical re-indexing time from 60 minutes to 2-5 minutes (only processing actually changed documents).

---

## 5. Chunking Strategy for SOPs and Warranty Documents

### Structure-Aware Recursive Chunking

SOPs and warranty documents are structured documents with clear hierarchies: document → sections → subsections → paragraphs. The chunking strategy must respect this structure.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

encoder = tiktoken.encoding_for_model("text-embedding-ada-002")

def token_count(text: str) -> int:
    return len(encoder.encode(text))

class StructureAwareChunker:
    def __init__(self, max_tokens=512, overlap_tokens=100):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens,
            chunk_overlap=overlap_tokens,
            length_function=token_count,
            separators=[
                "\n\n\n",    # Triple newline = section break
                "\n\n",      # Double newline = paragraph break
                "\n",        # Single newline = line break
                ". ",        # Sentence boundary
                " ",         # Word boundary (last resort)
            ],
        )
    
    def chunk(self, text: str, doc_metadata: dict) -> list[dict]:
        # Step 1: Detect section structure
        sections = self.detect_sections(text)
        
        # Step 2: Chunk within sections (never across section boundaries)
        all_chunks = []
        for section in sections:
            section_chunks = self.splitter.split_text(section["text"])
            
            for chunk_text in section_chunks:
                all_chunks.append({
                    "text": chunk_text,
                    "section_title": section["title"],
                    "token_count": token_count(chunk_text),
                })
        
        return all_chunks
    
    def detect_sections(self, text: str) -> list[dict]:
        """Split document into sections based on heading patterns."""
        import re
        
        # Common heading patterns in SOPs and policies
        heading_patterns = [
            r"^#{1,3}\s+(.+)$",           # Markdown headings
            r"^(\d+\.)\s+(.+)$",           # Numbered headings: "1. Coverage Terms"
            r"^(\d+\.\d+)\s+(.+)$",        # Sub-headings: "1.2 Exclusions"
            r"^(Section \d+[.:])(.+)$",     # "Section 3: Repair Procedures"
            r"^([A-Z][A-Z\s]{5,})$",        # ALL CAPS headings
        ]
        
        lines = text.split("\n")
        sections = []
        current_section = {"title": "Introduction", "text": ""}
        
        for line in lines:
            is_heading = False
            for pattern in heading_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    # Save current section if it has content
                    if current_section["text"].strip():
                        sections.append(current_section)
                    # Start new section
                    current_section = {
                        "title": line.strip(),
                        "text": "",
                    }
                    is_heading = True
                    break
            
            if not is_heading:
                current_section["text"] += line + "\n"
        
        # Don't forget the last section
        if current_section["text"].strip():
            sections.append(current_section)
        
        return sections
```

### Why 512 Tokens with 100-Token Overlap

**512 tokens** — this matches the natural paragraph length of SOP and warranty documents. A typical policy section paragraph is 300-600 tokens. At 512, each chunk captures roughly one complete topical paragraph, producing a focused embedding vector that represents one coherent concept. Larger chunks (1024) would mix multiple topics, diluting the embedding. Smaller chunks (256) would split complete thoughts, losing context.

**100-token overlap** — approximately 2-3 sentences repeated at chunk boundaries. This ensures that any concept that spans a boundary is fully present in at least one chunk. Without overlap, a critical sentence like "Coverage extends to 24 months from installation date" could be split with "Coverage extends to" in chunk N and "24 months from installation date" in chunk N+1. Neither chunk alone contains the complete policy term.

**Why never chunk across section boundaries** — sections in SOPs and policies are topically distinct. Section 3 (Coverage Terms) has nothing to do with Section 7 (Claims Procedure). A chunk that spans two sections would produce a confused embedding that's partially about coverage and partially about claims, making it a weak match for both topics. By forcing chunk boundaries to respect section boundaries, we guarantee each chunk is topically coherent.

### Special Handling: Tabular Data

Parts catalogs and some SOP appendices contain tables. Tables require special chunking because standard text splitting destroys tabular structure.

```python
def chunk_table(table_text: str, headers: list[str], doc_metadata: dict) -> list[dict]:
    """Chunk a table by groups of rows, preserving headers in each chunk."""
    rows = table_text.strip().split("\n")
    header_row = rows[0]
    data_rows = rows[1:]
    
    chunks = []
    current_rows = []
    current_tokens = token_count(header_row)
    
    for row in data_rows:
        row_tokens = token_count(row)
        
        if current_tokens + row_tokens > 400:  # Slightly smaller than 512 to leave room for header
            chunk_text = header_row + "\n" + "\n".join(current_rows)
            chunks.append({
                "text": chunk_text,
                "section_title": "Parts Table",
                "token_count": token_count(chunk_text),
            })
            current_rows = [row]
            current_tokens = token_count(header_row) + row_tokens
        else:
            current_rows.append(row)
            current_tokens += row_tokens
    
    # Last chunk
    if current_rows:
        chunk_text = header_row + "\n" + "\n".join(current_rows)
        chunks.append({
            "text": chunk_text,
            "section_title": "Parts Table",
            "token_count": token_count(chunk_text),
        })
    
    return chunks
```

**Decision:** Repeat table headers in every chunk.

**Reasoning:** A table row like "P/N 7832-A | Compressor Valve Assembly | $245.00 | In Stock" is meaningless without the headers "Part Number | Description | Price | Availability." If a chunk contains table rows without headers, the embedding won't capture what the columns represent, and the LLM in Phase 5 won't know how to interpret the data.

---

## 6. Embedding Model Selection

### Alternatives Considered

**OpenAI text-embedding-ada-002 / text-embedding-3-small**

Excellent quality, 8191 token limit, 1536 dimensions. API-based — requires sending document text to OpenAI's servers.

**OpenAI text-embedding-3-large**

Higher quality than ada-002, 3072 dimensions. Same API dependency.

**Sentence-Transformers (all-MiniLM-L6-v2)**

Open source, runs locally, 512 token limit, 384 dimensions. Fast (5ms per embedding). Lower quality than OpenAI models on general benchmarks.

**BGE / E5 Models (BAAI/bge-large-en-v1.5)**

Open source, runs locally, 512 token limit, 1024 dimensions. Near OpenAI quality on retrieval benchmarks. Heavier than MiniLM (~1.3GB model).

**Cohere embed-v3**

API-based, 2048 token limit, purpose-built for retrieval. Excellent quality. API dependency.

### Decision: BAAI/bge-large-en-v1.5

**Reasoning:**

Data privacy is the primary driver. Warranty policies and SOPs contain proprietary business information — coverage terms, pricing structures, repair techniques. Sending this to OpenAI's or Cohere's API means proprietary documents leave our infrastructure. Running embeddings locally with an open-source model keeps all data on-premises.

Among open-source models, bge-large-en-v1.5 ranks at the top of the MTEB retrieval benchmarks, matching or exceeding OpenAI ada-002 on most retrieval tasks. Its 1024-dimensional embeddings provide a good balance between quality and storage cost (each vector is ~4KB in Pinecone). The 512 token limit matches our chunk size perfectly.

**Tradeoff:** Local hosting means we need to run the embedding model on a GPU (or fast CPU) for both indexing and query-time embedding. During indexing (batch processing), we can use the same A10G GPU used for training (during off-hours). During query time, we embed a single query (fast — ~10ms on GPU, ~50ms on CPU), so even CPU inference is acceptable.

**Alternative tradeoff considered:** Using OpenAI's API would eliminate the need to host the embedding model, but the data privacy concern is a hard constraint for enterprise customers. If data privacy were not a concern, OpenAI text-embedding-3-small would be the simpler choice.

**Rule:** The same embedding model must be used for both indexing and querying. If you index with bge-large but query with MiniLM, the vector spaces don't align and retrieval quality collapses. This seems obvious but is a common mistake during model upgrades — if you change the embedding model, you must re-index the entire corpus.

---

## 7. Pinecone Configuration & Index Design

### Why Pinecone

**Alternatives Considered:**

- **FAISS (Facebook AI Similarity Search):** Open source, runs locally, excellent performance. But requires self-managed infrastructure — you handle sharding, replication, persistence, and backup. Good for prototyping, risky for production without a dedicated infrastructure team.

- **Weaviate / Milvus / Qdrant:** Open-source vector databases with managed cloud options. Viable alternatives. Weaviate has built-in hybrid search (vector + keyword). Milvus has excellent scalability. But all require more operational overhead than Pinecone's fully managed service.

- **pgvector (PostgreSQL extension):** Adds vector search to an existing PostgreSQL database. Simple if you already use PostgreSQL. But vector search performance degrades significantly beyond 100K vectors, and it lacks advanced features like metadata filtering with vector search.

**Decision:** Pinecone (managed service).

**Reasoning:** Pinecone is fully managed — zero operational overhead for index management, sharding, replication, or backup. At our corpus size (~6,000 chunks), even Pinecone's free tier handles the load. The managed service eliminates an entire category of infrastructure concerns, letting us focus on retrieval quality rather than database operations. Pinecone's native support for metadata filtering combined with vector search is critical for our use case (filter by product line, then rank by semantic similarity).

**Tradeoff:** Vendor lock-in. If Pinecone changes pricing, has an outage, or sunsets a feature, we're dependent. Mitigated by keeping the indexing pipeline abstracted — the indexer writes to a `VectorStore` interface, not directly to Pinecone. Swapping to Weaviate or FAISS would require implementing the interface, not rewriting the pipeline.

**Tradeoff:** Data leaves our infrastructure (Pinecone is a cloud service). The text stored in Pinecone metadata includes SOP and warranty content. Mitigated by Pinecone's SOC 2 Type II compliance and encryption at rest. If this is unacceptable for certain clients, we'd fall back to self-hosted Weaviate or Qdrant.

### Index Configuration

```python
import pinecone

pinecone.init(api_key="...", environment="us-east-1-aws")

pinecone.create_index(
    name="sop-warranty-docs",
    dimension=1024,            # bge-large-en-v1.5 output dimension
    metric="cosine",           # Cosine similarity for normalized embeddings
    pods=1,                    # Single pod for our corpus size
    pod_type="p1.x1",         # Performance-optimized pod
    metadata_config={
        "indexed": [           # Metadata fields available for filtering
            "product_line",
            "document_category",
            "region",
            "version",
        ]
    }
)
```

**Decision:** Cosine similarity metric.

**Reasoning:** bge-large-en-v1.5 produces normalized embeddings (unit length vectors). For normalized vectors, cosine similarity and dot product are mathematically equivalent. We use cosine explicitly for clarity and because it's the standard for text retrieval — it measures the angle between vectors, which corresponds to semantic similarity regardless of vector magnitude.

**Decision:** Index only 4 metadata fields for filtering.

**Reasoning:** Pinecone charges based on metadata index size. Indexing all metadata fields inflates costs unnecessarily. We only need to filter on product_line (narrow to correct product), document_category (sometimes we want only warranty policies, not SOPs), region (different regions have different policies), and version (always retrieve the latest version). Other metadata fields (section_title, chunk_index, prev_chunk, next_chunk) are stored but not indexed — they're retrieved after vector search but can't be used as pre-filters.

---

## 8. Metadata Schema & Filtering Strategy

### The Complete Metadata Schema

Every chunk in Pinecone carries this metadata:

```json
{
    "document_id": "sop-xr-comp-001",
    "document_title": "SOP: Compressor Replacement Procedure for XR Series",
    "document_category": "sop",
    "product_line": "XR",
    "region": "global",
    "version": "3.1",
    "effective_date": "2024-06-15",
    "expiry_date": null,
    
    "chunk_index": 4,
    "total_chunks": 12,
    "section_title": "3.2 Compressor Removal Steps",
    "text": "Step 1: Disconnect power supply and verify lockout-tagout...",
    "token_count": 487,
    
    "prev_chunk": "sop-xr-comp-001__chunk_0003",
    "next_chunk": "sop-xr-comp-001__chunk_0005",
    
    "content_hash": "a1b2c3d4...",
    "indexed_at": "2025-03-01T10:00:00Z",
    "index_version": "v2"
}
```

### Filtering Strategy

Metadata filters are applied BEFORE vector similarity search. This is critical for performance and relevance — filtering narrows the search space from 6,000 chunks to maybe 200-500 chunks for the correct product line, then vector search ranks those 200-500 by semantic relevance.

**Filter 1 — Product Line (always applied when known)**

```python
filter_conditions = {"product_line": {"$eq": "XR"}}
```

This is the most important filter. Without it, a query about XR series compressor failure might retrieve SOP chunks from the YZ series — semantically similar (both about compressors) but wrong product line.

**Filter 2 — Document Category (applied selectively)**

For warranty determination queries, restrict to warranty policies:
```python
filter_conditions = {
    "product_line": {"$eq": "XR"},
    "document_category": {"$eq": "warranty"}
}
```

For repair procedure queries, restrict to SOPs and repair procedures:
```python
filter_conditions = {
    "product_line": {"$eq": "XR"},
    "document_category": {"$in": ["sop", "repair_procedure"]}
}
```

**Decision:** Apply category filters based on the query type, not globally.

**Reasoning:** The action item generation in Phase 5 needs BOTH warranty information (is this covered?) AND repair procedures (how to fix it). If we always filter to one category, we'd miss the other. Instead, the query construction in Section 9 determines which categories to include based on what information the agent needs at that point.

**Filter 3 — Version (always retrieve latest)**

```python
# Handled by only indexing the latest version
# Old versions are deleted during re-indexing
```

**Decision:** Only index the latest version of each document. Don't keep old versions in the index.

**Reasoning:** The agent should always use the current policy. Keeping old versions creates a risk of retrieving an outdated policy that's semantically similar but no longer valid. If historical policy comparison is needed (rare), it's handled by a separate archive system, not the production RAG index.

**Tradeoff:** If a document update introduces errors (a corrupt PDF produces bad chunks), there's no immediate rollback within Pinecone. Mitigated by the validation step in the indexing pipeline (spot-check retrieval quality after re-indexing) and by keeping document version history in the source system (SharePoint/S3), allowing quick re-indexing of the previous version.

### Handling Unknown Product Lines

If the service order doesn't specify a model number (and Phase 1 couldn't extract one), the product line filter can't be applied.

```python
def build_filter(inherited_context: dict, metadata: dict) -> dict:
    filters = {}
    
    model = inherited_context.get("model_number") or metadata.get("model_number")
    if model:
        product_line = extract_product_line(model)  # "XR" from "XR-440"
        if product_line:
            filters["product_line"] = {"$eq": product_line}
    
    # If no product line determined, search the full index (no filter)
    return filters if filters else None
```

**Rule:** Never fabricate a product line filter. If you can't determine the product line from the order, search the full index and rely on semantic similarity to find relevant documents. A broad search with slightly lower precision is better than a wrong filter that excludes the correct documents entirely.

---

## 9. Query Construction from Extracted Defects

### The Query Construction Problem

The extracted defect JSON from Phase 3 looks like:

```json
{
    "defect_type": "mechanical",
    "severity": "high",
    "affected_component": "compressor",
    "error_codes": ["E-47"],
    "symptoms": ["grinding noise"],
    "root_cause_hypothesis": "Possible bearing failure after recent maintenance"
}
```

This needs to be converted into a natural language query that will match semantically with the relevant SOP/warranty chunks in Pinecone. The query must capture the key retrieval dimensions: what component, what's wrong, and what policy context is needed.

### Implementation

```python
class QueryConstructor:
    def construct_queries(self, defects: list, metadata: dict, 
                         inherited_context: dict) -> list[dict]:
        """Generate multiple targeted queries for different retrieval needs."""
        
        model = inherited_context.get("model_number", "")
        queries = []
        
        for defect in defects:
            # Query 1: Repair/SOP query — how to fix this
            repair_query = self.build_repair_query(defect, model)
            queries.append({
                "text": repair_query,
                "purpose": "repair_procedure",
                "category_filter": ["sop", "repair_procedure"],
                "defect_index": defects.index(defect),
            })
            
            # Query 2: Warranty query — is this covered
            warranty_query = self.build_warranty_query(defect, model, inherited_context)
            queries.append({
                "text": warranty_query,
                "purpose": "warranty_determination",
                "category_filter": ["warranty"],
                "defect_index": defects.index(defect),
            })
            
            # Query 3: Error code specific query (if error codes exist)
            if defect.error_codes:
                error_query = self.build_error_code_query(defect, model)
                queries.append({
                    "text": error_query,
                    "purpose": "error_diagnosis",
                    "category_filter": ["repair_procedure", "sop"],
                    "defect_index": defects.index(defect),
                })
        
        return queries
    
    def build_repair_query(self, defect, model: str) -> str:
        """Build a query focused on finding repair procedures."""
        parts = [model] if model else []
        parts.append(defect.affected_component)
        parts.append(defect.defect_type)
        parts.extend(defect.symptoms)
        parts.append("repair procedure maintenance")
        return " ".join(parts)
        # Example: "XR-440 compressor mechanical grinding noise repair procedure maintenance"
    
    def build_warranty_query(self, defect, model: str, context: dict) -> str:
        """Build a query focused on finding warranty coverage terms."""
        parts = [model] if model else []
        parts.append(defect.affected_component)
        parts.append("warranty coverage")
        
        # Add age/duration context if available
        if context.get("unit_age"):
            parts.append(f"{context['unit_age']} months")
        
        parts.append("eligibility terms conditions")
        return " ".join(parts)
        # Example: "XR-440 compressor warranty coverage 18 months eligibility terms conditions"
    
    def build_error_code_query(self, defect, model: str) -> str:
        """Build a query focused on error code diagnosis."""
        parts = [model] if model else []
        parts.extend(defect.error_codes)
        parts.append(defect.affected_component)
        parts.append("troubleshooting diagnosis")
        return " ".join(parts)
        # Example: "XR-440 E-47 compressor troubleshooting diagnosis"
```

### Design Decisions

**Decision:** Generate multiple queries per defect (repair + warranty + error code).

**Reasoning:** A single query trying to find both repair procedures AND warranty coverage would be too broad — the embedding would be a blurred average of both intents, matching neither perfectly. Separate queries with category filters produce sharper retrievals. A repair query with `category_filter: ["sop"]` hits repair documents with high precision. A warranty query with `category_filter: ["warranty"]` hits policy documents with high precision.

**Tradeoff:** Multiple queries per defect means more Pinecone calls. For a single-defect order with 3 queries, that's 3 Pinecone searches. For a 3-defect order, that's 9 searches. Each Pinecone search takes ~20-50ms, so the total retrieval latency is 60-450ms. Acceptable for our latency budget. If latency becomes a concern, we could batch-embed multiple queries and send them as a single Pinecone batch query.

**Decision:** Keep queries short (6-12 words).

**Reasoning:** Embedding models produce the best representations for queries that are concise and focused. A long, verbose query ("Please find the standard operating procedure for repairing a mechanical defect involving grinding noise in the compressor component of the XR-440 unit model") dilutes the embedding with filler words. A short query ("XR-440 compressor grinding noise repair procedure") produces a sharper embedding that matches the relevant document chunks more precisely.

**Tradeoff:** Short queries might miss nuance. The query "XR-440 compressor repair" would match general compressor repair documents but might not surface the specific E-47 error code troubleshooting guide. This is why we generate multiple queries — the error code query specifically targets E-47 documentation.

---

## 10. Retrieval — Vector Search + Metadata Filtering

### Implementation

```python
def retrieve_chunks(self, queries: list[dict], product_line: str = None) -> list[RetrievedChunk]:
    """Execute all queries and merge results."""
    
    all_chunks = {}  # chunk_id → RetrievedChunk (deduplication)
    
    for query in queries:
        # Build metadata filter
        filters = {}
        if product_line:
            filters["product_line"] = {"$eq": product_line}
        if query.get("category_filter"):
            filters["document_category"] = {"$in": query["category_filter"]}
        
        # Embed the query
        query_embedding = self.embedding_model.embed(query["text"])
        
        # Search Pinecone
        results = self.pinecone_index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True,
            filter=filters if filters else None,
        )
        
        # Convert to RetrievedChunk and deduplicate
        for match in results["matches"]:
            chunk_id = match["id"]
            if chunk_id not in all_chunks or match["score"] > all_chunks[chunk_id].relevance_score:
                all_chunks[chunk_id] = RetrievedChunk(
                    chunk_id=chunk_id,
                    document_title=match["metadata"]["document_title"],
                    section=match["metadata"]["section_title"],
                    text=match["metadata"]["text"],
                    relevance_score=match["score"],
                    metadata=match["metadata"],
                    query_purpose=query["purpose"],
                )
    
    # Sort by relevance score and return
    sorted_chunks = sorted(all_chunks.values(), key=lambda c: c.relevance_score, reverse=True)
    return sorted_chunks
```

### Deduplication Across Queries

**Decision:** When multiple queries retrieve the same chunk, keep it once with the highest relevance score.

**Reasoning:** If the repair query and the error code query both retrieve the same chunk about E-47 compressor troubleshooting, storing it twice wastes context window tokens. Keeping the highest score preserves the best relevance signal for reranking.

### Top-K Selection

**Decision:** Retrieve top-10 per query, then deduplicate across queries.

**Reasoning:** For a single-defect order with 3 queries, that's up to 30 chunks before deduplication. After deduplication, typically 15-25 unique chunks remain. This is more than we'll ultimately use (we'll take the top 5 after reranking), but starting with a broad candidate set gives the reranker more options to choose from.

**Tradeoff:** More candidates means more reranking computation (the cross-encoder processes each candidate individually). At 20-25 candidates, reranking takes ~100-150ms. Acceptable. If we reduced to top-3 per query, we'd risk missing relevant chunks that scored 4th or 5th in one query but would have ranked 1st after reranking.

---

## 11. Reranking with Cross-Encoder

### Why Rerank

Vector search (embedding similarity) provides a coarse ranking. The embedding model compresses the entire query and chunk into fixed-size vectors and compares them via cosine similarity. This misses fine-grained interactions between specific query terms and specific document passages.

A cross-encoder sees both the query and the document simultaneously and scores their relevance through deep token-level interactions. It's much more accurate but much slower (can't be pre-computed — must process each query-document pair individually).

### Implementation

```python
from sentence_transformers import CrossEncoder

RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_N = 5

def rerank_chunks(query_text: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Rerank retrieved chunks using cross-encoder."""
    
    if not chunks:
        return []
    
    # Prepare query-chunk pairs
    pairs = [(query_text, chunk.text) for chunk in chunks]
    
    # Score all pairs
    scores = RERANKER.predict(pairs)
    
    # Update relevance scores
    for i, score in enumerate(scores):
        chunks[i].relevance_score = float(score)
    
    # Sort by reranked score
    chunks.sort(key=lambda c: c.relevance_score, reverse=True)
    
    return chunks[:RERANK_TOP_N]
```

### Design Decisions

**Decision:** Use ms-marco-MiniLM-L-6-v2 as the cross-encoder reranker.

**Reasoning:** This model is trained specifically on the MS MARCO passage ranking dataset — it's purpose-built for re-ranking retrieval results. It's small (80MB), fast (~5ms per pair), and highly effective. Larger rerankers (ms-marco-ELECTRA-base) are marginally more accurate but 3-4x slower.

**Tradeoff:** The MiniLM reranker sometimes struggles with very long chunks (400+ tokens) because its context window is limited. Chunks near the 512-token limit might not be fully understood. Mitigated by our chunk size (512 tokens) being within the reranker's effective range, and by the fact that the most important information in a chunk is typically in the first few sentences.

**Decision:** Rerank all queries' results together, using a composite query.

**Reasoning:** Multiple queries (repair, warranty, error code) retrieve different types of chunks. Reranking them with a single composite query would bias toward one query's intent. Instead, we rerank per-purpose:

```python
def rerank_by_purpose(queries: list[dict], chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Rerank separately per query purpose, then merge."""
    
    final_chunks = []
    
    for purpose in ["repair_procedure", "warranty_determination", "error_diagnosis"]:
        purpose_queries = [q for q in queries if q["purpose"] == purpose]
        purpose_chunks = [c for c in chunks if c.query_purpose == purpose]
        
        if purpose_queries and purpose_chunks:
            # Use the first query of this purpose for reranking
            reranked = rerank_chunks(purpose_queries[0]["text"], purpose_chunks)
            # Take top 2 per purpose (total ~6 chunks across 3 purposes)
            final_chunks.extend(reranked[:2])
    
    # Deduplicate again and sort by score
    seen = set()
    unique_chunks = []
    for chunk in sorted(final_chunks, key=lambda c: c.relevance_score, reverse=True):
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            unique_chunks.append(chunk)
    
    return unique_chunks[:RERANK_TOP_N]
```

**Decision:** Take top 2 per purpose, total ~5-6 chunks.

**Reasoning:** The action item generation prompt in Phase 5 needs both repair procedures and warranty context. Taking only the overall top-5 might give you 5 repair chunks and 0 warranty chunks (if repair documents are more semantically similar to the query). By taking top-2 per purpose, we guarantee diversity — at least 2 repair chunks, 2 warranty chunks, and 1-2 error diagnosis chunks. This ensures the LLM has complete context for generating both the repair steps AND the warranty determination.

---

## 12. Query Reformulation on Low Relevance

### When to Reformulate

After reranking, if the top chunk's relevance score is below a threshold (e.g., 0.3 for cross-encoder scores), the initial queries likely used different terminology than the documents. Reformulation generates alternative queries.

```python
MIN_RELEVANCE_THRESHOLD = 0.3

def should_reformulate(reranked_chunks: list[RetrievedChunk]) -> bool:
    if not reranked_chunks:
        return True
    return reranked_chunks[0].relevance_score < MIN_RELEVANCE_THRESHOLD
```

### Reformulation Strategies

**Strategy 1 — Broaden the query**

Remove specific symptoms and focus on component + product:
```python
# Original: "XR-440 compressor mechanical grinding noise repair procedure"
# Broadened: "XR-440 compressor repair maintenance procedure"
```

**Strategy 2 — Use alternative terminology**

Map common technician language to SOP terminology:
```python
SYNONYM_MAP = {
    "grinding noise": "abnormal vibration",
    "leak": "fluid loss seepage",
    "won't start": "failure to operate startup failure",
    "overheating": "thermal overload temperature exceedance",
    "error code": "fault code diagnostic code",
}
```

**Strategy 3 — Component hierarchy expansion**

If searching for "compressor valve assembly" returns nothing, try the parent component "compressor":
```python
COMPONENT_HIERARCHY = {
    "compressor valve assembly": ["compressor valve", "compressor"],
    "control board display": ["control board", "control panel"],
    "fan motor bearing": ["fan motor", "fan assembly"],
}
```

### Implementation

```python
def reformulate_queries(original_queries: list[dict], defects: list) -> list[dict]:
    """Generate reformulated queries using multiple strategies."""
    
    reformulated = []
    
    for query in original_queries:
        # Strategy 1: Broaden
        broad_text = broaden_query(query["text"])
        reformulated.append({
            **query,
            "text": broad_text,
            "is_reformulation": True,
        })
        
        # Strategy 2: Synonyms
        synonym_text = apply_synonyms(query["text"])
        if synonym_text != query["text"]:
            reformulated.append({
                **query,
                "text": synonym_text,
                "is_reformulation": True,
            })
    
    return reformulated
```

**Decision:** Reformulation triggers a second round of retrieval, not a replacement.

**Reasoning:** The reformulated queries search Pinecone again. Results from both the original and reformulated searches are merged, deduplicated, and reranked together. This way, if the original query found some relevant chunks (just below threshold) and the reformulated query finds others, the combined set might have strong results that neither alone produced.

**Tradeoff:** Reformulation roughly doubles retrieval latency (two rounds of vector search + embedding). It only triggers on low-relevance results (~10-15% of queries), so the average latency impact is ~10-15%. The improvement in retrieval quality for these difficult cases justifies the cost.

---

## 13. Context Window Expansion (Neighbor Fetching)

### The Problem

A 512-token chunk captures one section paragraph. But the paragraph might describe the defect symptoms while the resolution steps are in the next paragraph (next chunk). Retrieving only the symptom chunk means the LLM has no repair steps to reference when generating action items.

### Implementation

```python
def expand_context(chunks: list[RetrievedChunk], max_expansion: int = 3) -> list[RetrievedChunk]:
    """Fetch neighboring chunks for the top-N retrieved chunks."""
    
    expanded = []
    fetched_ids = set()
    
    for chunk in chunks[:max_expansion]:  # Only expand top 3
        if needs_expansion(chunk):
            # Fetch previous chunk
            prev_id = chunk.metadata.get("prev_chunk")
            if prev_id and prev_id not in fetched_ids:
                prev_chunk = fetch_from_pinecone(prev_id)
                if prev_chunk:
                    expanded.append(prev_chunk)
                    fetched_ids.add(prev_id)
            
            # Add current chunk
            if chunk.chunk_id not in fetched_ids:
                expanded.append(chunk)
                fetched_ids.add(chunk.chunk_id)
            
            # Fetch next chunk
            next_id = chunk.metadata.get("next_chunk")
            if next_id and next_id not in fetched_ids:
                next_chunk = fetch_from_pinecone(next_id)
                if next_chunk:
                    expanded.append(next_chunk)
                    fetched_ids.add(next_id)
        else:
            if chunk.chunk_id not in fetched_ids:
                expanded.append(chunk)
                fetched_ids.add(chunk.chunk_id)
    
    # Add remaining non-expanded chunks
    for chunk in chunks[max_expansion:]:
        if chunk.chunk_id not in fetched_ids:
            expanded.append(chunk)
            fetched_ids.add(chunk.chunk_id)
    
    return expanded


def needs_expansion(chunk: RetrievedChunk) -> bool:
    """Determine if a chunk needs context expansion."""
    text = chunk.text.strip()
    
    # Starts mid-sentence (no uppercase letter at start)
    if text and not text[0].isupper() and not text[0].isdigit():
        return True
    
    # Ends mid-sentence (no terminal punctuation)
    if text and not text[-1] in '.;:!?':
        return True
    
    # Very short (likely a fragment)
    if len(text.split()) < 40:
        return True
    
    # Contains "continued" or "see below" signals
    continuation_signals = ["continued", "see below", "as follows", "next section"]
    if any(signal in text.lower() for signal in continuation_signals):
        return True
    
    return False
```

### Design Decisions

**Decision:** Only expand the top 3 chunks.

**Reasoning:** Expanding every chunk would roughly triple the number of chunks (each chunk + prev + next), consuming the entire LLM context window with retrieved content. Expanding only the top 3 (most relevant) chunks ensures the most important context is complete without overwhelming the prompt. The remaining chunks are included as-is — they provide supporting context but don't need full surrounding paragraphs.

**Tradeoff:** Chunks ranked 4th or 5th might also need expansion but don't get it. If the resolution steps for a defect happen to be in a chunk that ranked 5th, they won't be expanded. Mitigated by the reranking step typically pushing the most relevant chunks (including those with resolution steps) into the top 3.

---

## 14. Context Assembly for LLM Prompt

### How Retrieved Context Enters the Prompt

The retrieved, reranked, and expanded chunks are assembled into a context block that's injected into the action item generation prompt (Phase 2, Node 4).

```python
def assemble_context_block(chunks: list[RetrievedChunk], max_tokens: int = 3000) -> str:
    """Assemble retrieved chunks into a formatted context block for the LLM prompt."""
    
    context_parts = []
    current_tokens = 0
    
    # Group chunks by document for coherent reading
    chunks_by_doc = {}
    for chunk in chunks:
        doc_id = chunk.metadata["document_id"]
        if doc_id not in chunks_by_doc:
            chunks_by_doc[doc_id] = []
        chunks_by_doc[doc_id].append(chunk)
    
    # Sort chunks within each document by chunk_index (reading order)
    for doc_id in chunks_by_doc:
        chunks_by_doc[doc_id].sort(key=lambda c: c.metadata.get("chunk_index", 0))
    
    # Assemble context, respecting token budget
    for doc_id, doc_chunks in chunks_by_doc.items():
        doc_title = doc_chunks[0].document_title
        doc_section = doc_chunks[0].section
        
        header = f"--- Source: {doc_title} | Section: {doc_section} ---\n"
        header_tokens = token_count(header)
        
        if current_tokens + header_tokens > max_tokens:
            break
        
        context_parts.append(header)
        current_tokens += header_tokens
        
        for chunk in doc_chunks:
            chunk_tokens = chunk.metadata.get("token_count", token_count(chunk.text))
            
            if current_tokens + chunk_tokens > max_tokens:
                # Truncate to fit remaining budget
                remaining = max_tokens - current_tokens
                if remaining > 50:  # Only include if meaningful amount fits
                    truncated = truncate_to_tokens(chunk.text, remaining)
                    context_parts.append(truncated + "\n[...truncated]\n\n")
                break
            
            context_parts.append(chunk.text + "\n\n")
            current_tokens += chunk_tokens
    
    return "".join(context_parts)
```

### Design Decisions

**Decision:** Group chunks by document and sort by chunk_index within each document.

**Reasoning:** If chunks from the same document are interleaved with chunks from other documents, the LLM has to mentally context-switch between documents repeatedly. Grouping by document and presenting in reading order creates a coherent narrative that the LLM can follow more effectively. This is especially important for multi-chunk expansions — prev_chunk → current_chunk → next_chunk should appear sequentially, not scattered.

**Decision:** Hard token budget of 3,000 tokens for retrieved context.

**Reasoning:** The total LLM context budget for the action item generation prompt is approximately 4,000-6,000 usable tokens (for Llama 3.1 8B at high quality). The prompt needs: ~500 tokens for system instruction, ~300 tokens for extracted defects, ~200 tokens for metadata, ~3,000 tokens for retrieved context, and ~1,500-2,000 tokens for the model's output. 3,000 tokens for context is the maximum that leaves enough room for everything else.

**Tradeoff:** 3,000 tokens is roughly 5-6 chunks of 512 tokens each. For orders with multiple defects requiring multiple SOPs and warranty policies, this budget might not capture everything. The ordering by relevance score ensures the most important chunks are included first, and less relevant ones are truncated. If context budget is consistently the bottleneck, upgrading to a model with a larger effective context window (13B or 70B) would be the solution.

**Decision:** Include document title and section name as headers before each chunk group.

**Reasoning:** The LLM needs to know WHERE the information comes from to generate proper SOP references in the action items. Without headers, the model might attribute a warranty term to the wrong document. The headers consume ~30-50 tokens per document group — a small cost for proper attribution.

---

## 15. Document Lifecycle — Updates, Versioning, Re-Indexing

### Document Update Flow

```
[Team updates SOP in SharePoint]
        │
        ▼
[Change detection webhook fires]
        │
        ▼
[Indexing pipeline pulls new version]
        │
        ▼
[Delete old chunks for this document]
        │
        ▼
[Chunk → Embed → Upsert new version]
        │
        ▼
[Spot-check: run 5 known queries, verify retrieval quality]
        │
        ├── Quality OK → Done
        └── Quality degraded → Alert team, investigate
```

### Scheduled Full Re-Index

**Decision:** Full re-index weekly, incremental updates on change detection.

**Reasoning:** Change detection catches individual document updates in near real-time. But it can miss bulk updates (someone uploads 20 revised SOPs), document deletions (a deprecated SOP should be removed from the index), or indexing pipeline changes (new chunking strategy, new embedding model). The weekly full re-index catches anything the incremental updates missed.

**Tradeoff:** Full re-indexing processes all 470 documents even if most haven't changed. With content hashing, unchanged documents are skipped quickly. The actual compute cost is dominated by the changed documents. Total re-index time: ~10-15 minutes (mostly waiting for Pinecone upserts to propagate).

### Embedding Model Upgrade

When upgrading the embedding model (e.g., from bge-large-v1.5 to bge-large-v2.0), a full re-index is mandatory because the vector spaces are incompatible.

```
Step 1: Create new Pinecone index with new dimension (if changed)
Step 2: Index entire corpus with new embedding model
Step 3: Run evaluation queries against new index
Step 4: Compare retrieval quality metrics (new vs old)
Step 5: If improved or equivalent → switch query routing to new index
Step 6: Keep old index for 48 hours as rollback, then delete
```

**Rule:** Never serve queries against an index while it's being re-indexed. Use blue-green deployment — build the new index, validate it, then switch.

---

## 16. Evaluation with Ragas

### What Ragas Measures

Ragas (Retrieval Augmented Generation Assessment) evaluates the RAG pipeline on three axes:

**Faithfulness — Does the generated answer use only the retrieved context?**

If the action item says "covered under 36-month warranty" but the retrieved warranty chunk says "24-month coverage," faithfulness is violated — the model hallucinated or confused the coverage term. Faithfulness measures how closely the output sticks to the retrieved evidence.

```python
from ragas.metrics import faithfulness
from ragas import evaluate

results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness],
)
# Returns a score 0-1 where 1.0 = perfectly faithful
```

**Relevancy — Are the retrieved chunks relevant to the query?**

If the query is about compressor failure but the retrieved chunks discuss electrical wiring, relevancy is low — the retrieval step is returning off-topic documents. This metric isolates retrieval quality from generation quality.

```python
from ragas.metrics import context_relevancy

results = evaluate(
    dataset=eval_dataset,
    metrics=[context_relevancy],
)
```

**Correctness — Does the final output match the expected answer?**

Comparing the generated action items against human-created ground truth. This is the end-to-end metric that captures both retrieval errors and generation errors.

```python
from ragas.metrics import answer_correctness

results = evaluate(
    dataset=eval_dataset,
    metrics=[answer_correctness],
)
```

### Evaluation Dataset

**Decision:** Maintain 50 evaluation examples with known-correct context and action items.

Each evaluation example contains:
- Service order text (input)
- Expected retrieved documents (which SOPs/warranty docs should be found)
- Expected action items (human-created ground truth)

```json
{
    "question": "XR-440 compressor grinding noise E-47 unit 18 months old",
    "ground_truth_context": ["SOP-XR-COMP-001 Section 3.2", "Warranty-XR-v4.2 Section 2.1"],
    "ground_truth_answer": "Replace compressor per SOP-XR-COMP-001. Warranty eligible under v4.2 standard 24-month coverage.",
    "contexts": ["<retrieved chunks will be filled at eval time>"],
    "answer": "<generated answer will be filled at eval time>"
}
```

### Running Evaluation

```python
def run_rag_evaluation():
    """Weekly RAG pipeline evaluation."""
    eval_data = load_evaluation_dataset()
    
    results = {
        "faithfulness": [],
        "relevancy": [],
        "correctness": [],
        "retrieval_recall": [],
    }
    
    for example in eval_data:
        # Run the actual retrieval pipeline
        defects = mock_extraction(example["question"])
        chunks = retrieve_and_rerank(defects)
        
        # Check retrieval recall: did we find the expected documents?
        expected_docs = set(example["ground_truth_context"])
        retrieved_docs = set(f"{c.document_title} {c.section}" for c in chunks)
        recall = len(expected_docs & retrieved_docs) / len(expected_docs)
        results["retrieval_recall"].append(recall)
        
        # Run Ragas metrics
        ragas_result = evaluate_single(
            question=example["question"],
            contexts=[c.text for c in chunks],
            answer=generate_action_items(defects, chunks),
            ground_truth=example["ground_truth_answer"],
        )
        
        results["faithfulness"].append(ragas_result["faithfulness"])
        results["relevancy"].append(ragas_result["context_relevancy"])
        results["correctness"].append(ragas_result["answer_correctness"])
    
    # Compute averages
    summary = {
        metric: sum(values) / len(values) 
        for metric, values in results.items()
    }
    
    return summary
```

### Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Faithfulness | ≥0.90 | Generated output closely follows retrieved context |
| Context Relevancy | ≥0.85 | Retrieved chunks are topically relevant to query |
| Answer Correctness | ≥0.80 | Final output matches ground truth |
| Retrieval Recall | ≥0.85 | Correct source documents found in top-5 |

---

## 17. Failure Modes & Graceful Degradation

### Failure Mode 1 — Pinecone Unreachable

**Cause:** Network issue, Pinecone outage, DNS failure.

**Handling:** Return empty retrieved_chunks. The LangGraph agent proceeds to action item generation (Node 4) without context. The prompt explicitly instructs the LLM to set `escalation_required: true` when context is insufficient. The order gets processed with degraded quality (no SOP grounding) rather than blocked.

**Rule:** A Pinecone outage should never stop order processing. Degraded output with escalation is always better than no output.

### Failure Mode 2 — Zero Relevant Results

**Cause:** Query terminology mismatch, missing documents in index, wrong product line filter.

**Handling:** Trigger query reformulation (Section 12). If reformulation also returns no results, proceed with empty context and escalation flag.

### Failure Mode 3 — Wrong Documents Retrieved

**Cause:** Semantic similarity matches wrong product line, outdated index, embedding model quality issue.

**Handling:** This is the hardest failure to detect automatically because the retrieved chunks look relevant but are actually for a different product or policy version. Caught by the Ragas evaluation pipeline (faithfulness drops when the model uses wrong context) and by human review of escalated orders. When detected, root cause is investigated: usually a missing metadata filter or an outdated document in the index.

### Failure Mode 4 — Index Corruption

**Cause:** Partial re-index failure (some chunks updated, others stale), embedding model mismatch (query embedded with different model than index).

**Handling:** Weekly full re-index catches partial corruption. Embedding model mismatch is prevented by the rule that both indexing and querying must use the same model version. An automated consistency check compares the embedding model version used for indexing against the version used for querying.

---

## 18. Observability & Alerting

### Metrics Emitted

**Indexing Metrics:**

| Metric | Description |
|--------|-------------|
| `indexing.documents_processed` | Count per indexing run |
| `indexing.documents_changed` | How many actually needed re-indexing |
| `indexing.chunks_created` | Total chunks upserted |
| `indexing.embedding_latency_ms` | Batch embedding latency |
| `indexing.pinecone_upsert_latency_ms` | Upsert latency |
| `indexing.total_duration_ms` | End-to-end indexing pipeline time |
| `indexing.errors` | Document processing errors (PDF parse failures, etc.) |

**Retrieval Metrics (per query):**

| Metric | Description |
|--------|-------------|
| `retrieval.query_count` | Number of queries generated per order |
| `retrieval.embedding_latency_ms` | Query embedding latency |
| `retrieval.pinecone_latency_ms` | Vector search latency |
| `retrieval.rerank_latency_ms` | Cross-encoder reranking latency |
| `retrieval.total_latency_ms` | End-to-end retrieval time |
| `retrieval.chunks_retrieved` | Total chunks retrieved before dedup |
| `retrieval.chunks_after_dedup` | Chunks after deduplication |
| `retrieval.chunks_after_rerank` | Final chunk count after reranking |
| `retrieval.top_score` | Highest relevance score |
| `retrieval.avg_score` | Average relevance score of top-5 |
| `retrieval.reformulation_triggered` | Whether query reformulation was needed |
| `retrieval.expansion_triggered` | Whether context expansion was applied |
| `retrieval.zero_results` | Whether any query returned 0 results |
| `retrieval.context_tokens` | Total token count of assembled context |

**Quality Metrics (weekly evaluation):**

| Metric | Description |
|--------|-------------|
| `rag.faithfulness` | Ragas faithfulness score |
| `rag.relevancy` | Ragas context relevancy score |
| `rag.correctness` | Ragas answer correctness score |
| `rag.retrieval_recall` | Percentage of expected documents found |

### Alert Configuration

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Pinecone unreachable | Retrieval error rate >20% in 5min | P1 | Check Pinecone status, network, DNS |
| Zero results spike | >15% of queries return 0 results in 1hr | P2 | New product line not indexed, or metadata filter too restrictive |
| Relevance score drop | Average top-5 score drops below 0.3 | P2 | Embedding drift, index corruption, or query quality issue |
| Reformulation rate spike | >25% of queries trigger reformulation | P2 | Systematic terminology mismatch, update synonym map |
| Retrieval latency spike | P95 > 500ms | P2 | Pinecone performance issue, check index size and pod type |
| Indexing failure | Any document fails to index | P3 | PDF parse error, document format issue, investigate |
| Ragas faithfulness drop | Weekly score below 0.85 | P2 | Model using wrong context, check retrieval and prompt |
| Ragas relevancy drop | Weekly score below 0.80 | P2 | Retrieval quality degrading, check index freshness |
| Stale index | No indexing run in 7+ days | P3 | Scheduled indexing may have failed, investigate |
| Context budget overflow | >20% of orders hitting 3000 token limit | P3 | Chunks too large or too many queries, review chunking strategy |

---

## 19. End-to-End Phase 4 Flow Summary

### Offline: Document Indexing Pipeline

```
DOCUMENT UPDATED IN SHAREPOINT / S3
        │
Step 1 ─┤  CHANGE DETECTION
        │  • Webhook notification or scheduled 6-hour scan
        │  • Content hash comparison against last indexed version
        │  • Skip if unchanged
        │
Step 2 ─┤  DOCUMENT EXTRACTION
        │  • PDF → text (PyMuPDF / pdfplumber)
        │  • DOCX → text (python-docx / pandoc)
        │  • HTML → text (BeautifulSoup)
        │
Step 3 ─┤  STRUCTURE-AWARE CHUNKING
        │  • Detect section boundaries (headings, numbered sections)
        │  • Never chunk across section boundaries
        │  • 512 tokens per chunk, 100-token overlap
        │  • Tables: repeat headers in every chunk
        │  • Token-measured using embedding model's tokenizer
        │
Step 4 ─┤  EMBEDDING
        │  • Model: BAAI/bge-large-en-v1.5 (1024 dimensions)
        │  • Batch embedding for efficiency
        │  • Same model for indexing AND querying (critical rule)
        │
Step 5 ─┤  PINECONE UPSERT
        │  • Delete old chunks for this document
        │  • Upsert new chunks with full metadata
        │  • Metadata: doc title, category, product line, region, version,
        │    section title, chunk index, prev/next pointers, content hash
        │
Step 6 ─┤  VALIDATION
        │  • Spot-check: run 5 known queries, verify retrieval quality
        │  • Compare pre/post update
        │  • Alert if quality degraded
```

### Online: Per-Request Retrieval Pipeline

```
EXTRACTED DEFECTS ARRIVE FROM PHASE 2 (Node 3)
        │
Step 1 ─┤  QUERY CONSTRUCTION
        │  • Generate multiple queries per defect:
        │    - Repair query (component + symptoms + "repair procedure")
        │    - Warranty query (component + "warranty coverage" + age)
        │    - Error code query (error codes + "troubleshooting")
        │  • Short, focused queries (6-12 words each)
        │  • Category filters per query purpose
        │
Step 2 ─┤  METADATA FILTER CONSTRUCTION
        │  • Product line filter from model number (when known)
        │  • Document category filter per query purpose
        │  • No filter when product line unknown (broad search)
        │
Step 3 ─┤  VECTOR SEARCH (Pinecone)
        │  • Embed each query with bge-large-en-v1.5
        │  • Top-10 per query with metadata filter
        │  • Deduplicate across queries (keep highest score)
        │  • Typical: 15-25 unique candidates
        │
Step 4 ─┤  CROSS-ENCODER RERANKING
        │  • Rerank per purpose (repair, warranty, error)
        │  • Top-2 per purpose = ~5-6 diverse chunks
        │  • Model: ms-marco-MiniLM-L-6-v2
        │
Step 5 ─┤  RELEVANCE CHECK
        │  • If top score < 0.3 → trigger query reformulation
        │  • Reformulation strategies: broaden, synonyms, component hierarchy
        │  • Second retrieval round, merge with original results
        │  • Re-rerank combined set
        │
Step 6 ─┤  CONTEXT WINDOW EXPANSION
        │  • Top 3 chunks: check if context is incomplete
        │  • Fetch prev/next neighbors using chunk pointers
        │  • Deduplicate expanded set
        │
Step 7 ─┤  CONTEXT ASSEMBLY
        │  • Group chunks by document (coherent reading order)
        │  • Sort by chunk_index within each document
        │  • Add document title + section headers
        │  • Respect 3,000 token budget
        │  • Truncate lowest-ranked chunks if over budget
        │
        ▼
CONTEXT BLOCK RETURNED TO PHASE 2 (stored in AgentState)
        │
        └──▶ Used by Node 4 (generate_action_items) in the LLM prompt
             alongside extracted defects and metadata
```

### Key Design Principles Applied Throughout Phase 4

1. **Separate what changes from what doesn't.** Fine-tuning teaches reasoning (stable). RAG provides current knowledge (changes frequently). Never mix them.
2. **Filter first, then rank.** Metadata filters narrow the search space to relevant documents before expensive vector similarity and reranking. Wrong product line → wrong SOP, regardless of semantic similarity.
3. **Multiple targeted queries beat one broad query.** Separate queries for repair, warranty, and error diagnosis with category-specific filters produce sharper retrievals than a single query trying to find everything.
4. **Two-stage retrieval (embed → rerank) balances speed and accuracy.** Vector search is fast but coarse. Cross-encoder reranking is slow but precise. Combined, they give accurate results at acceptable latency.
5. **Diverse results over top-ranked results.** Taking top-2 per purpose guarantees the LLM sees both repair procedures and warranty context, not just whichever category happened to score highest.
6. **Graceful degradation over hard failure.** Pinecone down → proceed without context + escalation flag. Zero results → reformulate. Still nothing → escalate. The system never blocks on retrieval failure.
7. **The index must always be current.** Stale SOPs → wrong action items → wrong repairs → equipment damage or warranty disputes. Document freshness is a production safety concern, not an optimization.

---

*Document Version: 1.0 | Last Updated: March 2026 | System: LGS Tech Agentic AI Workflow*
