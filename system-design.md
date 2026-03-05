# Agentic Service Order Processing — System Design

**LGS Tech · LangGraph + Fine-Tuned Llama 3.1 + RAG + Eval Pipeline**

---

## Phase 1 — Ingestion Layer

### Service Order Intake
Incoming service orders arrive as unstructured technical logs — free-text descriptions of defects, warranty claims, and repair requests from field technicians.

`REST API` `JSON / Raw Text`

### Preprocessing
Normalize raw input: strip noise, tokenize, chunk long logs for LLM context windows. Attach metadata (order ID, plant ID, timestamps).

`Python` `Text Chunking`

---

## Phase 2 — Agent Orchestrator (The Brain)

### LangGraph State Machine
Defines the agentic workflow as a directed graph with conditional edges. Each node is a processing step (extract → retrieve → validate → generate). The graph manages state transitions, retries, and branching logic based on intermediate outputs.

`LangGraph` `LangChain` `State Graph`

### Agent State
Typed state object that flows through the graph — holds raw input, extracted defects, retrieved SOPs, validation results, and final action items.

`TypedDict` `Pydantic`

---

## Phase 3 — Fine-Tuned LLM (Core Reasoning)

### Llama 3.1 8B + QLoRA
Base Llama 3.1 8B fine-tuned via QLoRA (4-bit quantized LoRA) on proprietary technical logs. Learns domain-specific language — defect codes, part numbers, failure modes — that a general-purpose LLM wouldn't understand.

`Llama 3.1 8B` `QLoRA / SFT` `HuggingFace`

### Defect Extraction
Takes unstructured log text → outputs structured JSON: defect type, severity, affected component, root cause hypothesis. This is the core reasoning step.

→ *Structured JSON from unstructured logs*

`Structured Output` `JSON Mode`

### Training Pipeline
Curated dataset of labeled technical logs → instruction-response pairs. SFT trains the model to extract defect patterns accurately from domain-specific jargon.

`SFT Dataset` `PyTorch` `PEFT`

---

## Phase 4 — RAG Pipeline (Policy Grounding)

### Query Construction
Takes the extracted defect JSON from Phase 3 and constructs a semantic query — e.g., "warranty policy for compressor failure in model X within 24-month coverage."

`Embedding Model` `Query Builder`

### Pinecone Vector Store
Pre-indexed with SOPs, warranty policies, repair procedures, and compliance documents. Chunked, embedded, and stored with metadata filters (product line, region, policy version).

→ *99% compliance with latest warranty policies*

`Pinecone` `Vector DB` `Embeddings`

### Context Augmentation
Retrieved SOP chunks are appended to the prompt alongside the extracted defects. The LLM now reasons with both the defect data AND the relevant policy context.

`Top-K Retrieval` `Reranking`

---

## Phase 5 — Validation & Guardrails

### Pydantic Schema Validation
Every LLM output is validated against strict Pydantic models — required fields, enum constraints, type checks. If validation fails, the agent retries with corrective prompting (self-healing loop).

→ *95% system reliability via strict validation*

`Pydantic` `JSON Schema`

### Guardrails & Retry Logic
LangGraph conditional edges handle retries: if Pydantic rejects output → re-prompt with error message → retry up to N times → fallback to human review queue if still invalid.

`Conditional Edges` `Retry Policy`

---

## Phase 6 — Output Generation

### Action Item Generator
Takes validated defect data + retrieved SOP context → generates structured action items: repair steps, parts needed, warranty eligibility, escalation flags.

→ *85% reduction in manual processing effort*

`Action Items` `Structured JSON`

### Downstream Integration
Final output pushed to service management systems, ticketing tools, or surfaced in a dashboard for human review and approval.

`API / Webhook` `Dashboard`

---

## Continuous — Evaluation Pipeline

### LangSmith Tracing
Every agent run is traced end-to-end — input, each node's output, LLM calls, retrieval results, validation pass/fail, latency per step. Full observability into the agentic workflow.

`LangSmith` `Tracing`

### Ragas Evaluation
Benchmarks RAG quality: faithfulness (does the answer match retrieved context?), relevancy (are retrieved docs relevant?), and correctness (does output match ground truth?).

`Ragas` `Faithfulness` `Relevancy`

### Regression & Drift Detection
Periodic eval runs against a golden dataset to catch model drift. If accuracy drops below threshold → triggers retraining or SOP re-indexing alerts.

`Golden Dataset` `Threshold Alerts`

---

## End-to-End Request Flow

**1. Service order arrives**
A technician submits a raw service order with unstructured defect logs. The system preprocesses and chunks the text, attaching metadata like order ID and plant info.

**2. LangGraph agent starts execution**
The state graph initializes with the preprocessed input. The first node invokes the fine-tuned Llama 3.1 8B model to extract structured defect patterns from the raw logs.

**3. Defect extraction → structured JSON**
The fine-tuned model reads domain-specific jargon and outputs structured JSON — defect type, severity, component, root cause. This is validated by Pydantic; if invalid, the agent retries with the error appended to the prompt.

**4. RAG retrieval for policy grounding**
The extracted defect JSON is used to construct a semantic query. Pinecone returns the top-K most relevant SOP and warranty policy chunks. These are appended as context for the next LLM call.

**5. Action item generation**
With both the defect data and the retrieved policy context, the LLM generates structured action items — repair steps, parts list, warranty eligibility, and any escalation flags. Output is again Pydantic-validated.

**6. Output delivered + evaluation logged**
The final validated output is pushed downstream. Simultaneously, the entire run is traced in LangSmith and periodically benchmarked by Ragas against golden datasets to catch drift.
