# Continuous — Evaluation Pipeline: Complete Production Reference

## LGS Tech Agentic AI Workflow — Service Order Processing

---

## Table of Contents

1. Evaluation Pipeline Overview & Why It's Continuous
2. The Three Pillars of Evaluation
3. Pillar 1 — LangSmith Tracing (Real-Time Observability)
4. Trace Structure & What Gets Captured
5. LangSmith Dashboard & Debugging Workflows
6. Pillar 2 — Ragas Evaluation (RAG Quality Benchmarking)
7. Ragas Metrics Deep Dive
8. Evaluation Dataset Design & Management
9. Running Ragas Evaluations
10. Pillar 3 — Regression & Drift Detection
11. Golden Dataset Construction & Maintenance
12. Drift Detection Implementation
13. Automated Retraining Triggers
14. SOP Re-Indexing Triggers
15. A/B Testing & Canary Evaluation
16. End-to-End Evaluation Architecture
17. Observability & Alerting for the Evaluation Pipeline Itself
18. End-to-End Evaluation Flow Summary

---

## 1. Evaluation Pipeline Overview & Why It's Continuous

### What the Evaluation Pipeline Does

The evaluation pipeline is not a phase that runs once — it runs continuously alongside production, monitoring every aspect of system quality in real time and performing deeper benchmark evaluations on a scheduled basis. It answers three questions at all times: "Is the system working right now?" (LangSmith tracing), "Is the RAG pipeline retrieving and grounding correctly?" (Ragas evaluation), and "Has anything degraded since we last checked?" (regression and drift detection).

### Why Evaluation Must Be Continuous

LLM-based systems degrade silently. Unlike traditional software where a bug produces an error, an LLM system can produce outputs that look valid but are subtly wrong — a warranty determination based on an outdated policy, a repair procedure for the wrong model, a severity classification that's one level off. These failures don't crash the system or trigger exceptions. They pass validation, flow downstream, and cause real-world harm (wrong repairs, incorrect warranty claims, safety risks) before anyone notices.

Continuous evaluation catches these silent failures through three mechanisms: real-time tracing detects anomalies in individual request behavior (latency spikes, low retrieval scores, high retry rates), periodic benchmarking detects aggregate quality drift (model accuracy declining over weeks), and golden dataset regression testing catches specific capability losses (the model used to handle multi-defect orders well but now misses the second defect).

### What Causes Silent Degradation

**Cause 1 — Data drift.** Technician writing patterns change over time. New product lines introduce unfamiliar terminology. Seasonal patterns shift (HVAC defects spike in summer, heating defects in winter). The model was trained on historical patterns that may not represent current input.

**Cause 2 — Document staleness.** SOPs and warranty policies are updated, but the Pinecone index might not be re-indexed promptly. The RAG pipeline retrieves an outdated SOP version. The action items reference a procedure that's been superseded.

**Cause 3 — Model drift.** Even without retraining, the effective behavior of the system changes as the input distribution shifts. The model might have 95% accuracy on summer defect patterns but only 88% on winter patterns that it saw less frequently during training.

**Cause 4 — Infrastructure drift.** A vLLM configuration change, a Pinecone index migration, an embedding model update — any infrastructure change can subtly affect output quality without producing errors.

---

## 2. The Three Pillars of Evaluation

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE                          │
│                                                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌────────────────────┐  │
│  │  PILLAR 1     │  │  PILLAR 2     │  │  PILLAR 3          │  │
│  │  LangSmith    │  │  Ragas        │  │  Regression &      │  │
│  │  Tracing      │  │  Evaluation   │  │  Drift Detection   │  │
│  │               │  │               │  │                    │  │
│  │  Real-time    │  │  Scheduled    │  │  Scheduled         │  │
│  │  Every request│  │  Weekly       │  │  Weekly + Triggered│  │
│  │               │  │               │  │                    │  │
│  │  "Is this     │  │  "Is RAG      │  │  "Has anything     │  │
│  │   request     │  │   working     │  │   gotten worse     │  │
│  │   healthy?"   │  │   well?"      │  │   since last       │  │
│  │               │  │               │  │   check?"          │  │
│  └───────────────┘  └───────────────┘  └────────────────────┘  │
│                                                                 │
│  SCOPE:     Per-request    Aggregate/Sample    Aggregate/Full   │
│  LATENCY:   Zero overhead  30-60 min eval      15-30 min eval   │
│  FREQUENCY: Continuous     Weekly              Weekly + On-demand│
│  CATCHES:   Anomalies      RAG quality issues  Model degradation │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Pillar 1 — LangSmith Tracing (Real-Time Observability)

### What LangSmith Does

LangSmith is a tracing and observability platform built by LangChain specifically for LLM applications. Every LangGraph execution is automatically traced — every node entry/exit, every LLM call, every retrieval operation, every validation decision. The trace is a hierarchical timeline that shows exactly what happened, in what order, with what inputs and outputs, and how long each step took.

### Why LangSmith Over Generic APM Tools

**Alternative 1 — Datadog / New Relic / Grafana**

General-purpose application performance monitoring tools. They track latency, error rates, and throughput. But they don't understand LLM-specific concepts: token counts, prompt/completion content, retrieval relevance scores, or validation error details. You'd need custom instrumentation for everything LLM-related.

**Alternative 2 — OpenTelemetry Custom Spans**

Open standard for distributed tracing. You could instrument every node with custom spans and send traces to any backend (Jaeger, Zipkin, Grafana Tempo). Maximum flexibility, no vendor lock-in. But you'd have to build all the LLM-specific trace analysis yourself — prompt viewing, token counting, output comparison, evaluation integration.

**Alternative 3 — Weights & Biases (W&B)**

Excellent for training monitoring. Has experiment tracking, dataset management, and model registry. But its production tracing capabilities for LLM inference are less mature than LangSmith's. Better for training-time evaluation than production-time tracing.

**Decision:** LangSmith for production tracing, W&B for training monitoring.

**Reasoning:** LangSmith integrates natively with LangGraph — traces are captured automatically without manual instrumentation. It understands LLM-specific concepts out of the box: prompt/completion viewing, token counting, cost estimation, latency per LLM call. The trace visualization shows the graph execution as a hierarchical timeline, making debugging intuitive. And it integrates directly with Ragas for running evaluations on traced data.

**Tradeoff:** Vendor dependency on LangChain's ecosystem. If we migrate away from LangGraph, LangSmith traces would need to be replaced with custom tracing. Mitigated by also sending key metrics (latency, error rate, throughput) to our primary monitoring system (CloudWatch/Datadog) so we're not completely dependent on LangSmith for operational alerting.

### Integration Implementation

```python
import langsmith
from langsmith import traceable
from langsmith.run_trees import RunTree

# Global configuration
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls_..."
os.environ["LANGCHAIN_PROJECT"] = "lgs-tech-service-orders-prod"

# Automatic tracing for LangGraph — no code changes needed
# LangGraph nodes are automatically wrapped when tracing is enabled

# For custom functions outside LangGraph, use the @traceable decorator
@traceable(name="parts_catalog_lookup", run_type="tool")
def lookup_parts(part_number: str, model: str) -> dict:
    """This function call will appear in the LangSmith trace."""
    result = parts_catalog.lookup(part_number)
    return result

@traceable(name="pinecone_retrieval", run_type="retriever")
def search_pinecone(query: str, filters: dict) -> list:
    """Retrieval operations get their own trace type for filtering."""
    results = pinecone_index.query(
        vector=embed(query),
        top_k=10,
        filter=filters,
    )
    return results

# Adding custom metadata to traces
@traceable(name="extract_defects", metadata={"model_version": "v1.2"})
def extract_defects_node(state: AgentState) -> dict:
    # LangSmith automatically captures:
    # - Input state
    # - Output state  
    # - Duration
    # - Any LLM calls made inside this function
    # - Any nested @traceable calls
    
    # Add custom feedback to the trace
    langsmith.Client().create_feedback(
        run_id=langsmith.get_current_run_tree().id,
        key="extraction_confidence",
        score=avg_confidence,
        comment=f"Extracted {len(defects)} defects",
    )
    
    return result
```

---

## 4. Trace Structure & What Gets Captured

### Hierarchical Trace Structure

Each service order segment produces one trace in LangSmith:

```
📋 Agent Run: SO-28841-S1 [8.45s total]
│
├── 🔧 extract_defects [2.10s]
│   ├── 💬 LLM Call: llama-3.1-8b-defect-extractor [1.95s]
│   │   ├── Input: 847 tokens (system: 312, user: 535)
│   │   ├── Output: 234 tokens
│   │   ├── Model: llama-3.1-8b + defect-extractor-v1.2
│   │   └── Temperature: 0.0
│   └── 📊 Custom: prompt_construction [0.12s]
│
├── ✅ validate_extraction [0.015s]
│   ├── JSON Parse: PASS
│   ├── Pydantic: PASS (2 defects validated)
│   ├── Business Logic: 0 warnings
│   ├── Cross-Validation: 0 warnings
│   └── Route: → retrieve_context
│
├── 🔍 retrieve_context [0.34s]
│   ├── 🔎 Query 1: "XR-440 compressor repair procedure" [0.11s]
│   │   ├── Pinecone: 10 results, top score 0.82
│   │   └── Filter: product_line=XR, category=sop
│   ├── 🔎 Query 2: "XR-440 compressor warranty coverage" [0.09s]
│   │   ├── Pinecone: 8 results, top score 0.76
│   │   └── Filter: product_line=XR, category=warranty
│   ├── 🔎 Query 3: "XR-440 E-47 troubleshooting" [0.08s]
│   │   └── Pinecone: 6 results, top score 0.71
│   └── 📊 Reranking: 5 final chunks, top reranked score 0.89
│
├── 🔧 generate_action_items [3.20s]
│   ├── 💬 LLM Call: llama-3.1-8b-defect-extractor [3.05s]
│   │   ├── Input: 2,341 tokens (system: 428, context: 1,580, defects: 333)
│   │   ├── Output: 612 tokens
│   │   └── Temperature: 0.0
│   └── 📊 Custom: prompt_construction [0.14s]
│
├── ✅ validate_output [0.018s]
│   ├── JSON Parse: PASS
│   ├── Pydantic: PASS (2 action items validated)
│   ├── Business Logic: 0 warnings
│   └── Route: → assemble_output
│
└── 📦 assemble_output [0.045s]
    ├── Parts Enrichment: 2 parts verified
    ├── Warranty Enrichment: 1 item eligible
    ├── Escalation: not required
    └── Dispatch: → ServiceNow (work order created)
```

### What Gets Captured Per Node

**For Every Node:**
- Input state (full AgentState snapshot)
- Output state (the partial state update returned)
- Duration (wall clock time)
- Node name and graph position
- Any errors or exceptions

**For LLM Calls Specifically:**
- Full prompt (system + user message)
- Full completion (raw model output)
- Token counts (input, output, total)
- Model identifier and version
- Temperature and other generation parameters
- Latency (time to first token, total generation time)

**For Retrieval Operations:**
- Query text
- Query embedding (optional, can be disabled for storage)
- Metadata filters applied
- Number of results returned
- Top relevance scores
- Retrieved chunk texts and metadata
- Reranking scores (before and after)

**For Validation Nodes:**
- Validation result (pass/fail)
- Specific errors (with Pydantic error details)
- Business logic warnings
- Cross-validation results
- Routing decision (valid/retry/human_review)

### Custom Metadata Tags

Every trace is tagged with metadata for filtering and analysis:

```python
trace_metadata = {
    "order_id": state["order_id"],
    "segment_id": state["segment_id"],
    "source_channel": "email",
    "product_line": "XR",
    "model_version": "defect-extractor-v1.2",
    "index_version": "v2",
    "environment": "production",
}
```

This enables queries like: "Show me all traces from the XR product line that had extraction retries in the last 24 hours" or "Compare average generation latency between model v1.1 and v1.2."

---

## 5. LangSmith Dashboard & Debugging Workflows

### Standard Debugging Workflow

When a downstream issue is reported (wrong action item, incorrect warranty determination, missed defect), the debugging process follows a consistent path:

**Step 1 — Find the trace.**

Search by order_id or segment_id in LangSmith. The trace shows the complete execution timeline.

**Step 2 — Identify the failure point.**

Walk the trace top-to-bottom: Did extraction miss the defect? Did retrieval return the wrong SOP? Did generation hallucinate the warranty determination? Did validation miss a bad output? The hierarchical trace makes this visual — you can see exactly where the data went wrong.

**Step 3 — Examine the LLM I/O.**

Click into the specific LLM call. Read the exact prompt that was sent. Read the exact response that came back. This is the most critical debugging artifact — it tells you whether the prompt was well-constructed (if the prompt was good but the output was bad, it's a model issue; if the prompt was missing context, it's a retrieval or preprocessing issue).

**Step 4 — Check retrieval quality.**

If the LLM output was bad because the prompt had wrong or missing context, drill into the retrieval node. What queries were constructed? What metadata filters were applied? What chunks were returned? Were the right documents in the index? This narrows the root cause to either query construction (Phase 4), indexing (offline pipeline), or document availability (the SOP hasn't been uploaded yet).

**Step 5 — Root cause and fix.**

Common root causes and their fixes:
- Wrong extraction → add similar examples to training data, retrain
- Wrong retrieval → fix query construction logic or update metadata filters
- Stale SOP retrieved → re-index the updated document
- Hallucinated warranty → strengthen the grounding instruction in the prompt
- Validation missed bad output → add new Pydantic validator or business logic check

### Trace-Based Analytics

LangSmith aggregates trace data for trend analysis:

**Latency Trends:** P50/P95/P99 per node over time. Detect if generation latency is creeping up (GPU contention, longer outputs).

**Token Usage Trends:** Average input/output tokens per LLM call. Detect if prompts are growing (context sections getting larger due to more retrieved chunks).

**Error Patterns:** Most common validation errors over time. Detect if a specific error type is increasing (indicates a systematic issue).

**Retrieval Quality Trends:** Average top-K retrieval score over time. Detect if relevance is declining (index staleness, embedding drift).

---

## 6. Pillar 2 — Ragas Evaluation (RAG Quality Benchmarking)

### What Ragas Is

Ragas (Retrieval Augmented Generation Assessment) is an open-source framework for evaluating RAG pipelines. It provides metrics that measure the quality of both the retrieval step and the generation step, both independently and together. Unlike LangSmith (which observes production traffic), Ragas runs structured evaluations against a curated dataset with known-correct answers.

### Why Ragas in Addition to LangSmith

LangSmith tells you what happened. Ragas tells you whether what happened was correct. LangSmith can show that retrieval returned 5 chunks with a top score of 0.82, but it can't tell you whether those 5 chunks were the right 5 chunks for that query. Ragas compares the retrieved chunks and generated answers against ground truth to measure correctness.

**Decision:** Use LangSmith for real-time production monitoring, Ragas for periodic quality benchmarking.

**Reasoning:** They serve complementary purposes. LangSmith catches anomalies in individual requests (latency spike, retrieval failure, validation loop). Ragas catches aggregate quality issues across the system (faithfulness declining, relevancy dropping). Together, they provide both micro-level (per-request) and macro-level (system-wide) quality visibility.

---

## 7. Ragas Metrics Deep Dive

### Metric 1 — Faithfulness

**What it measures:** Does the generated output use only information from the retrieved context? Or did the LLM hallucinate facts that weren't in the retrieved documents?

**How it works:** Ragas takes the generated answer and the retrieved context. It breaks the answer into individual claims (statements). For each claim, it checks whether the claim is supported by the retrieved context. The faithfulness score is the proportion of claims that are supported.

```
Example:
  Generated: "The compressor is covered under 24-month standard warranty per Policy v4.2"
  Retrieved context contains: "Standard coverage for XR compressors: 24 months from installation"
  
  Claims:
    1. "compressor is covered" → Supported ✓
    2. "24-month coverage" → Supported ✓
    3. "standard warranty" → Supported ✓
    4. "per Policy v4.2" → Need to check if v4.2 is mentioned in context
  
  If the retrieved context doesn't mention "v4.2" but the LLM added it:
    Faithfulness = 3/4 = 0.75
```

**Target:** ≥0.90

**Why this matters for LGS Tech:** A faithfulness violation in warranty determination means the system is making warranty claims based on information the retrieved policy doesn't support. This could lead to approving warranty claims the company shouldn't pay for, or denying claims that are actually covered. Both are costly errors.

### Metric 2 — Context Relevancy

**What it measures:** Are the retrieved documents actually relevant to the query? Or did the retrieval step return off-topic chunks?

**How it works:** Ragas evaluates each retrieved chunk against the query and determines what fraction of the retrieved content is relevant to answering the question. High relevancy means the retrieved context is focused and useful. Low relevancy means the context is padded with irrelevant information.

```
Example:
  Query: "warranty coverage for XR-440 compressor failure"
  
  Retrieved chunks:
    Chunk 1: "Standard coverage for XR compressors: 24 months..." → Relevant ✓
    Chunk 2: "Extended warranty options for premium customers..." → Relevant ✓
    Chunk 3: "Compressor replacement procedure step 1..." → Partially relevant
    Chunk 4: "Electrical wiring diagram for XR-440 control panel..." → Not relevant ✗
    Chunk 5: "Company holiday schedule 2025..." → Not relevant ✗
  
  Context Relevancy = relevant chunks / total chunks ≈ 0.5
```

**Target:** ≥0.85

**Why this matters:** Low relevancy wastes the LLM's context window with irrelevant information. At best, the LLM ignores the irrelevant chunks and produces correct output (wasting tokens). At worst, the irrelevant chunks confuse the LLM, leading to incorrect or unfocused action items. High relevancy means the retrieval pipeline is working efficiently — the LLM sees only what it needs to see.

### Metric 3 — Answer Correctness

**What it measures:** Does the final generated output match the expected ground truth answer? This is the end-to-end quality metric that captures errors from both retrieval and generation.

**How it works:** Ragas compares the generated action items against human-created ground truth action items. It measures both factual correctness (right defect type, right severity, right warranty determination) and semantic similarity (the description conveys the same repair procedure even if worded differently).

```
Example:
  Generated: "Replace compressor, warranty eligible, priority urgent"
  Ground truth: "Replace compressor assembly, covered under standard warranty, urgent priority"
  
  Factual match: action_type ✓, warranty ✓, priority ✓
  Semantic similarity of description: 0.91
  
  Answer Correctness ≈ 0.93
```

**Target:** ≥0.80

**Why this target is lower than faithfulness:** Correctness measures exact match with ground truth, which is inherently harder. Two reasonable humans might produce slightly different (but both correct) action items for the same defect. The model's answer might be correct but phrased differently from the ground truth. A correctness target of 0.80 accounts for this inherent variability.

### Metric 4 — Context Recall (Custom Addition)

**What it measures:** Did the retrieval step find the documents that should have been found?

**How it works:** The evaluation dataset specifies which documents are relevant for each query. Context recall measures what fraction of the expected documents were actually retrieved.

```python
def context_recall(retrieved_docs: list[str], expected_docs: list[str]) -> float:
    """What fraction of expected documents were retrieved?"""
    expected_set = set(expected_docs)
    retrieved_set = set(retrieved_docs)
    
    if not expected_set:
        return 1.0  # No expected docs = nothing to miss
    
    found = expected_set & retrieved_set
    return len(found) / len(expected_set)
```

**Target:** ≥0.85

**Why this matters:** Low context recall means the retrieval pipeline is missing relevant documents — they exist in the index but aren't being found. This might indicate query construction issues (wrong terminology), metadata filter issues (too restrictive), or embedding quality issues (the embedding model doesn't capture the semantic relationship between the query and the document).

### Metric 5 — Answer Relevancy

**What it measures:** Is the generated answer actually addressing the question, or is it generating tangentially related but not directly useful content?

**How it works:** Ragas checks whether the generated output is relevant to the original query/defect. A high answer relevancy means the action items directly address the reported defects. Low answer relevancy means the action items discuss related topics but don't specifically address what was asked.

**Target:** ≥0.85

---

## 8. Evaluation Dataset Design & Management

### Dataset Structure

```python
class EvaluationExample(BaseModel):
    """A single evaluation example with ground truth."""
    
    # Input
    example_id: str
    segment_text: str
    pre_extracted_entities: list[dict]
    inherited_context: dict
    structured_metadata: dict
    
    # Ground truth — retrieval
    expected_documents: list[str]         # Document titles that should be retrieved
    expected_sections: list[str]          # Specific sections within those documents
    
    # Ground truth — extraction
    expected_defects: list[dict]          # Human-annotated defect extractions
    
    # Ground truth — generation
    expected_action_items: list[dict]     # Human-created action items
    expected_warranty_eligible: bool      # Human-determined warranty eligibility
    
    # Metadata
    difficulty: str                       # "easy" / "medium" / "hard"
    category: str                         # "single_defect" / "multi_defect" / "ambiguous" / "safety_critical"
    product_line: str
    created_at: str
    last_verified: str                    # When was this ground truth last checked
```

### Dataset Composition

**Decision:** Maintain 200 evaluation examples with deliberate distribution.

```
Distribution by difficulty:
  Easy (clear, single defect, explicit language):       40% (80 examples)
  Medium (multiple defects, some ambiguity):             35% (70 examples)
  Hard (ambiguous language, implicit defects, edge cases): 25% (50 examples)

Distribution by category:
  Single defect:          40% (80 examples)
  Multi-defect:           25% (50 examples)
  Ambiguous/unclear:      15% (30 examples)
  Safety-critical:        10% (20 examples)
  Cross-referenced defects: 10% (20 examples)

Distribution by product line:
  Proportional to production volume (XR: 45%, YZ: 30%, AB: 15%, other: 10%)
```

**Reasoning:** The distribution is intentionally weighted toward harder cases. If the dataset were proportional to production data (mostly easy single-defect orders), the evaluation would report high accuracy that masks poor performance on the important edge cases. By over-representing hard cases, we get a conservative accuracy estimate that surfaces weaknesses.

**Tradeoff:** The evaluation scores will be lower than production accuracy because the dataset is harder than production data. This can cause confusion if stakeholders expect evaluation scores to match the production automation rate. Mitigated by reporting both the dataset-weighted score (conservative) and an estimated production-weighted score (adjusting for the easier production distribution).

### Dataset Freshness

**Rule:** Refresh 25% of the evaluation dataset every quarter.

**Implementation:**
1. Sample 50 recent production orders (from the last 3 months)
2. Have two annotators independently create ground truth
3. Adjudicate disagreements
4. Replace the oldest 50 examples in the dataset with the new 50
5. Re-run baseline evaluation with the updated dataset
6. Update threshold targets if the new examples change the difficulty distribution

**Why this matters:** If the dataset stays static, it eventually becomes unrepresentative of current production data. New product lines, new defect patterns, and new SOP formats appear over time. A static dataset would show stable evaluation scores while real-world performance degrades on patterns the dataset doesn't cover.

### Dataset Versioning

```
evaluation-datasets/
├── v1.0/                    # Initial 200 examples
│   ├── dataset.json
│   ├── metadata.json        # Creation date, annotator info, distribution stats
│   └── baseline_results.json # Baseline evaluation scores for this version
├── v1.1/                    # Q1 refresh (50 replaced)
│   ├── dataset.json
│   ├── metadata.json
│   ├── changelog.json       # Which examples were added/removed
│   └── baseline_results.json
├── v1.2/                    # Q2 refresh
│   └── ...
└── current -> v1.2/         # Symlink to latest version
```

**Decision:** Version the dataset and keep old versions for comparison.

**Reasoning:** When evaluation scores change, you need to know whether the change is due to model improvement/degradation or dataset change. Keeping old dataset versions lets you run the new model against the old dataset (isolating model change) and the old model against the new dataset (isolating dataset change).

---

## 9. Running Ragas Evaluations

### Implementation

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_relevancy,
    answer_correctness,
    answer_relevancy,
)
from datasets import Dataset
import json
from datetime import datetime

class RagasEvaluator:
    
    def __init__(self, pipeline, dataset_path: str):
        self.pipeline = pipeline  # The full RAG pipeline (retrieve + generate)
        self.dataset = self.load_dataset(dataset_path)
        self.metrics = [
            faithfulness,
            context_relevancy,
            answer_correctness,
            answer_relevancy,
        ]
    
    def run_evaluation(self) -> dict:
        """Run full Ragas evaluation on the dataset."""
        
        eval_start = datetime.utcnow()
        
        # Step 1: Run pipeline on each example to get predictions
        predictions = []
        for example in self.dataset:
            # Run the actual pipeline (same code as production)
            result = self.pipeline.process(
                segment_text=example["segment_text"],
                pre_extracted_entities=example["pre_extracted_entities"],
                inherited_context=example["inherited_context"],
                metadata=example["structured_metadata"],
            )
            
            predictions.append({
                "question": example["segment_text"],
                "answer": self.format_answer(result),
                "contexts": [c.text for c in result.get("retrieved_chunks", [])],
                "ground_truth": self.format_answer({"action_items": example["expected_action_items"]}),
            })
        
        # Step 2: Run Ragas metrics
        ragas_dataset = Dataset.from_list(predictions)
        ragas_results = evaluate(
            dataset=ragas_dataset,
            metrics=self.metrics,
        )
        
        # Step 3: Compute custom metrics
        custom_results = self.compute_custom_metrics(predictions, self.dataset)
        
        # Step 4: Assemble full results
        eval_duration = (datetime.utcnow() - eval_start).total_seconds()
        
        full_results = {
            "eval_timestamp": datetime.utcnow().isoformat(),
            "eval_duration_seconds": eval_duration,
            "dataset_version": self.dataset_version,
            "model_version": self.pipeline.model_version,
            "index_version": self.pipeline.index_version,
            "sample_count": len(predictions),
            
            "ragas_scores": {
                "faithfulness": float(ragas_results["faithfulness"]),
                "context_relevancy": float(ragas_results["context_relevancy"]),
                "answer_correctness": float(ragas_results["answer_correctness"]),
                "answer_relevancy": float(ragas_results["answer_relevancy"]),
            },
            
            "custom_scores": custom_results,
            
            "per_example_scores": self.compute_per_example(ragas_results, predictions),
        }
        
        # Step 5: Store results
        self.store_results(full_results)
        
        # Step 6: Check against thresholds
        self.check_thresholds(full_results)
        
        return full_results
    
    def compute_custom_metrics(self, predictions: list, dataset: list) -> dict:
        """Compute additional metrics beyond standard Ragas."""
        
        # Context recall: did we retrieve the expected documents?
        recall_scores = []
        for pred, example in zip(predictions, dataset):
            expected_docs = set(example.get("expected_documents", []))
            retrieved_docs = set()
            for ctx in pred["contexts"]:
                # Extract document title from context
                retrieved_docs.update(self.extract_doc_titles(ctx))
            
            if expected_docs:
                recall = len(expected_docs & retrieved_docs) / len(expected_docs)
                recall_scores.append(recall)
        
        # Extraction accuracy (if we have extraction ground truth)
        extraction_scores = self.evaluate_extraction_accuracy(predictions, dataset)
        
        # Per-field accuracy
        field_accuracy = self.evaluate_field_accuracy(predictions, dataset)
        
        return {
            "context_recall": sum(recall_scores) / len(recall_scores) if recall_scores else 0,
            "extraction_accuracy": extraction_scores,
            "field_accuracy": field_accuracy,
        }
    
    def check_thresholds(self, results: dict):
        """Check evaluation results against defined thresholds."""
        
        THRESHOLDS = {
            "faithfulness": 0.90,
            "context_relevancy": 0.85,
            "answer_correctness": 0.80,
            "answer_relevancy": 0.85,
            "context_recall": 0.85,
        }
        
        violations = []
        
        for metric, threshold in THRESHOLDS.items():
            score = results["ragas_scores"].get(metric) or results["custom_scores"].get(metric)
            if score is not None and score < threshold:
                violations.append({
                    "metric": metric,
                    "score": score,
                    "threshold": threshold,
                    "gap": threshold - score,
                })
        
        if violations:
            self.alert_threshold_violations(violations, results)
```

### Evaluation Schedule

**Decision:** Weekly automated evaluation + on-demand evaluation after changes.

```
Weekly (Sunday night):
  Full evaluation on complete 200-example dataset
  Results stored in evaluation history
  Threshold violations trigger alerts Monday morning

On-demand triggers:
  After model retraining (before and after promotion)
  After index re-indexing (verify retrieval quality maintained)
  After prompt template changes
  After infrastructure changes (vLLM config, Pinecone migration)
```

**Reasoning:** Weekly evaluation catches slow drift. On-demand evaluation catches immediate impact of changes. Together, they ensure no quality degradation goes undetected for more than 7 days (drift) or a few hours (change-induced).

**Tradeoff:** Full evaluation on 200 examples takes 30-60 minutes (mostly LLM inference time). During evaluation, the pipeline processes eval examples alongside production traffic, consuming GPU capacity. Mitigated by running evaluations during low-traffic periods (Sunday night) and using a separate evaluation endpoint that doesn't affect production latency.

---

## 10. Pillar 3 — Regression & Drift Detection

### Types of Drift

**Data Drift:** The distribution of input data changes over time. New terminology, new product lines, different writing styles, seasonal defect patterns.

**Model Drift:** The model's effective accuracy changes even without retraining, because the input distribution shifted to patterns the model handles less well.

**Concept Drift:** The relationship between inputs and correct outputs changes. A defect that used to be "medium" severity is now "high" because safety standards changed. An SOP that used to recommend repair now recommends replacement.

**Index Drift:** The Pinecone index becomes stale as documents are updated in the source system but not re-indexed.

### Detection Implementation

```python
class DriftDetector:
    
    def __init__(self, eval_history_store):
        self.history = eval_history_store
    
    def detect_drift(self, current_results: dict) -> list[dict]:
        """Compare current evaluation against historical baseline."""
        
        drift_signals = []
        
        # Load historical results
        baseline = self.history.get_baseline()  # Results from model promotion time
        previous = self.history.get_previous()   # Last week's results
        trend = self.history.get_trend(weeks=8)  # Last 8 weeks
        
        # ── Check 1: Absolute threshold violations ──
        THRESHOLDS = {
            "faithfulness": 0.90,
            "context_relevancy": 0.85,
            "answer_correctness": 0.80,
        }
        
        for metric, threshold in THRESHOLDS.items():
            current_score = current_results["ragas_scores"].get(metric, 0)
            if current_score < threshold:
                drift_signals.append({
                    "type": "threshold_violation",
                    "metric": metric,
                    "current": current_score,
                    "threshold": threshold,
                    "severity": "high",
                })
        
        # ── Check 2: Regression from baseline ──
        for metric in ["faithfulness", "context_relevancy", "answer_correctness"]:
            current = current_results["ragas_scores"].get(metric, 0)
            base = baseline["ragas_scores"].get(metric, 0)
            
            if current < base - 0.03:  # More than 3% drop from baseline
                drift_signals.append({
                    "type": "regression_from_baseline",
                    "metric": metric,
                    "current": current,
                    "baseline": base,
                    "drop": base - current,
                    "severity": "medium",
                })
        
        # ── Check 3: Week-over-week decline ──
        for metric in ["faithfulness", "context_relevancy", "answer_correctness"]:
            current = current_results["ragas_scores"].get(metric, 0)
            prev = previous["ragas_scores"].get(metric, 0)
            
            if current < prev - 0.05:  # More than 5% week-over-week drop
                drift_signals.append({
                    "type": "week_over_week_decline",
                    "metric": metric,
                    "current": current,
                    "previous": prev,
                    "drop": prev - current,
                    "severity": "medium",
                })
        
        # ── Check 4: Trend detection (consistent decline over 4+ weeks) ──
        for metric in ["faithfulness", "context_relevancy", "answer_correctness"]:
            scores = [w["ragas_scores"].get(metric, 0) for w in trend]
            
            if len(scores) >= 4:
                # Simple linear regression to detect downward trend
                slope = self.compute_trend_slope(scores)
                if slope < -0.005:  # Declining by 0.5% per week
                    drift_signals.append({
                        "type": "downward_trend",
                        "metric": metric,
                        "slope_per_week": slope,
                        "weeks_analyzed": len(scores),
                        "severity": "high" if slope < -0.01 else "medium",
                    })
        
        # ── Check 5: Per-category regression ──
        for category in ["single_defect", "multi_defect", "ambiguous", "safety_critical"]:
            current_cat = current_results.get("per_category_scores", {}).get(category, {})
            baseline_cat = baseline.get("per_category_scores", {}).get(category, {})
            
            for metric in ["answer_correctness"]:
                current_score = current_cat.get(metric, 0)
                base_score = baseline_cat.get(metric, 0)
                
                if base_score > 0 and current_score < base_score - 0.05:
                    drift_signals.append({
                        "type": "category_regression",
                        "category": category,
                        "metric": metric,
                        "current": current_score,
                        "baseline": base_score,
                        "severity": "high" if category == "safety_critical" else "medium",
                    })
        
        return drift_signals
    
    def compute_trend_slope(self, scores: list[float]) -> float:
        """Simple linear regression slope."""
        n = len(scores)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(scores) / n
        
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, scores))
        denominator = sum((xi - x_mean) ** 2 for xi in x)
        
        return numerator / denominator if denominator != 0 else 0
```

### Design Decisions

**Decision:** Multiple drift detection methods (absolute threshold, baseline regression, week-over-week, trend analysis, per-category).

**Reasoning:** No single detection method catches all types of drift. Absolute thresholds catch severe degradation. Baseline regression catches gradual drift from the model's starting quality. Week-over-week catches sudden drops. Trend analysis catches slow, consistent decline that might be under the week-over-week threshold but is clearly directional over months. Per-category regression catches targeted degradation (model got worse specifically on multi-defect orders) that aggregate scores might mask.

**Decision:** 3% threshold for baseline regression, 5% for week-over-week.

**Reasoning:** Week-to-week variation of ±2-3% is normal (different examples in the evaluation, randomness in retrieval, minor infrastructure differences). A 3% drop from baseline is significant because the baseline is a stable reference. A 5% drop week-over-week is alarming because it exceeds normal variation. Trend analysis uses a lower threshold (0.5% per week) because it compounds — 0.5% per week is 2.5% per month, 10% in 5 months.

**Decision:** Safety-critical category regression is always high severity.

**Reasoning:** A regression on safety-critical examples means the system is handling dangerous situations worse than before. Even a small accuracy drop on safety items could lead to missed safety escalations, incorrect handling of hazardous materials, or delayed emergency responses. This category gets the highest alert priority regardless of the magnitude of the drop.

---

## 11. Golden Dataset Construction & Maintenance

### What Makes a "Golden" Example

A golden example is a service order where we have:

1. The original input text (as received from the technician)
2. Human-verified defect extraction (annotated by domain experts)
3. The correct SOP and warranty documents that should be retrieved
4. Human-created action items (verified by a senior service engineer)
5. Confirmed warranty determination (verified by warranty department)

### Construction Process

```
Step 1: Sample from production
  Select diverse orders: different product lines, defect types,
  severities, single/multi-defect, clear/ambiguous

Step 2: Two-annotator extraction
  Two domain experts independently extract defects
  Cohen's kappa ≥ 0.80 required
  Adjudicator resolves disagreements

Step 3: Document verification
  Service engineer identifies which SOPs and warranty policies
  should be consulted for each order

Step 4: Action item creation
  Senior service engineer creates the "ideal" action items
  Peer reviewed by a second senior engineer

Step 5: Warranty verification
  Warranty department confirms eligibility determination
  Policy version and section documented

Step 6: Final review
  Complete example reviewed for consistency
  Edge cases are explicitly documented (why this is hard, what makes it tricky)
```

### Maintenance Rules

**Rule 1:** Every golden example has an "expires" date — 6 months from creation or the next time the referenced SOP/warranty policy is updated, whichever comes first. Expired examples must be re-verified before use.

**Rule 2:** When a golden example's referenced SOP is updated, the example is flagged for re-verification. The expected action items might change because the new SOP has different procedures.

**Rule 3:** New golden examples must represent current production patterns. If 30% of production orders now come from a new product line not represented in the golden dataset, the dataset is unrepresentative and must be updated.

**Rule 4:** Hard examples are more valuable than easy ones. A golden example that tests an edge case (ambiguous language, implicit defect, cross-referenced issues) provides more evaluation signal than a straightforward single-defect order. The dataset should over-represent hard cases.

---

## 12. Automated Retraining Triggers

### Trigger Conditions

```python
class RetrainingTriggerEvaluator:
    
    def should_retrain(self, drift_signals: list[dict],
                       production_metrics: dict) -> dict:
        """Determine if retraining should be triggered."""
        
        triggers = []
        
        # Trigger 1: Evaluation metric below threshold
        for signal in drift_signals:
            if signal["type"] == "threshold_violation":
                triggers.append({
                    "reason": f"{signal['metric']} below threshold ({signal['current']:.3f} < {signal['threshold']})",
                    "urgency": "high",
                })
        
        # Trigger 2: Consistent downward trend
        for signal in drift_signals:
            if signal["type"] == "downward_trend" and signal["severity"] == "high":
                triggers.append({
                    "reason": f"{signal['metric']} declining at {signal['slope_per_week']:.4f}/week",
                    "urgency": "medium",
                })
        
        # Trigger 3: Production retry rate exceeding threshold
        if production_metrics.get("retry_rate", 0) > 0.15:
            triggers.append({
                "reason": f"Production retry rate {production_metrics['retry_rate']:.1%} exceeds 15%",
                "urgency": "high",
            })
        
        # Trigger 4: Human review correction rate high
        if production_metrics.get("correction_rate", 0) > 0.25:
            triggers.append({
                "reason": f"Human correction rate {production_metrics['correction_rate']:.1%} exceeds 25%",
                "urgency": "medium",
            })
        
        # Trigger 5: Scheduled monthly retraining
        if production_metrics.get("days_since_last_retrain", 0) > 30:
            triggers.append({
                "reason": "Scheduled monthly retraining (30+ days since last)",
                "urgency": "low",
            })
        
        # Trigger 6: New product line with insufficient training coverage
        new_product_lines = production_metrics.get("new_product_lines_seen", [])
        if new_product_lines:
            triggers.append({
                "reason": f"New product line(s) detected in production: {new_product_lines}",
                "urgency": "medium",
            })
        
        should_trigger = len(triggers) > 0
        max_urgency = max([t["urgency"] for t in triggers], key=lambda u: {"high": 3, "medium": 2, "low": 1}[u]) if triggers else "none"
        
        return {
            "should_retrain": should_trigger,
            "triggers": triggers,
            "urgency": max_urgency,
            "recommended_action": self.recommend_action(triggers),
        }
    
    def recommend_action(self, triggers: list) -> str:
        """Recommend specific retraining action based on triggers."""
        
        reasons = [t["reason"] for t in triggers]
        
        if any("retry rate" in r for r in reasons):
            return "Investigate extraction failures. Add recent failure cases to training data. Retrain with augmented dataset."
        
        if any("correction rate" in r for r in reasons):
            return "Incorporate human corrections from last 30 days into training data. Retrain with corrections as high-priority examples."
        
        if any("new product line" in r for r in reasons):
            return "Collect and label 30-50 examples from new product line. Retrain with balanced dataset including new examples."
        
        if any("threshold" in r for r in reasons) or any("trend" in r for r in reasons):
            return "Run full diagnostic on evaluation failures. Identify systematic error patterns. Augment training data for weak areas. Retrain."
        
        return "Scheduled retraining with latest production data incorporated."
```

---

## 13. SOP Re-Indexing Triggers

### When to Re-Index

```python
class ReindexTriggerEvaluator:
    
    def should_reindex(self, drift_signals: list, production_metrics: dict) -> dict:
        """Determine if Pinecone re-indexing should be triggered."""
        
        triggers = []
        
        # Trigger 1: Context relevancy dropped
        for signal in drift_signals:
            if signal["metric"] == "context_relevancy" and signal["type"] in ("threshold_violation", "regression_from_baseline"):
                triggers.append({
                    "reason": f"Context relevancy dropped to {signal['current']:.3f}",
                    "action": "Check for stale documents, re-index changed SOPs",
                })
        
        # Trigger 2: Context recall dropped
        for signal in drift_signals:
            if signal.get("metric") == "context_recall":
                triggers.append({
                    "reason": f"Context recall dropped — expected documents not being found",
                    "action": "Verify all documents are indexed, check metadata filters",
                })
        
        # Trigger 3: Faithfulness dropped but model hasn't changed
        model_unchanged = production_metrics.get("days_since_model_change", 0) > 7
        for signal in drift_signals:
            if signal["metric"] == "faithfulness" and model_unchanged:
                triggers.append({
                    "reason": "Faithfulness dropped without model change — likely context issue",
                    "action": "Documents may have been updated without re-indexing",
                })
        
        # Trigger 4: Document change detection
        stale_docs = production_metrics.get("documents_changed_not_reindexed", 0)
        if stale_docs > 0:
            triggers.append({
                "reason": f"{stale_docs} document(s) changed but not re-indexed",
                "action": "Run incremental re-index for changed documents",
            })
        
        # Trigger 5: Scheduled weekly full re-index
        days_since_full = production_metrics.get("days_since_full_reindex", 0)
        if days_since_full > 7:
            triggers.append({
                "reason": "Scheduled weekly full re-index",
                "action": "Run full re-index pipeline",
            })
        
        return {
            "should_reindex": len(triggers) > 0,
            "triggers": triggers,
        }
```

### Diagnostic: Isolating Retrieval Issues from Model Issues

When evaluation scores drop, you need to determine whether the problem is retrieval (wrong documents found) or generation (right documents found, wrong output generated). This determines whether you re-index or retrain.

```python
def diagnose_quality_drop(current_results: dict) -> str:
    """Determine if quality drop is retrieval-caused or model-caused."""
    
    faithfulness = current_results["ragas_scores"]["faithfulness"]
    relevancy = current_results["ragas_scores"]["context_relevancy"]
    correctness = current_results["ragas_scores"]["answer_correctness"]
    recall = current_results["custom_scores"]["context_recall"]
    
    if relevancy < 0.80 or recall < 0.80:
        # Retrieval is failing — wrong or missing documents
        return "RETRIEVAL_ISSUE: Re-index documents, check query construction, verify metadata filters"
    
    if faithfulness < 0.85 and relevancy >= 0.85:
        # Good retrieval but model not using context faithfully
        return "MODEL_ISSUE: Model hallucinating despite good context. Improve grounding prompt or retrain."
    
    if correctness < 0.75 and faithfulness >= 0.85 and relevancy >= 0.85:
        # Good retrieval, faithful to context, but wrong answer
        return "GROUND_TRUTH_SHIFT: Check if SOPs/policies have changed. Golden dataset may need updating."
    
    return "UNCLEAR: Multiple factors may be contributing. Run detailed per-example analysis."
```

---

## 14. A/B Testing & Canary Evaluation

### Evaluating New Models Before Full Deployment

When a new model version is trained, it goes through a three-stage evaluation:

**Stage 1 — Offline evaluation against golden dataset.**

Run the full Ragas evaluation with the new model. Compare scores against the production model. The new model must match or exceed production scores on all metrics.

**Stage 2 — Canary deployment with shadow evaluation.**

Deploy the new model to serve 10% of production traffic. Both the new model and production model process the same orders (the production model's results are the ones actually used). Compare their outputs side-by-side: does the new model produce different action items? Better or worse? This catches production-specific issues that the golden dataset might not cover.

```python
def canary_evaluation(canary_results: list, production_results: list) -> dict:
    """Compare canary model against production model on same inputs."""
    
    agreement_count = 0
    canary_better = 0
    production_better = 0
    
    for canary, prod in zip(canary_results, production_results):
        if canary["action_items"] == prod["action_items"]:
            agreement_count += 1
        else:
            # Human evaluator rates which is better
            # (or automated comparison against ground truth if available)
            pass
    
    agreement_rate = agreement_count / len(canary_results)
    
    return {
        "agreement_rate": agreement_rate,
        "canary_better_rate": canary_better / len(canary_results),
        "production_better_rate": production_better / len(canary_results),
    }
```

**Stage 3 — Full promotion with monitoring.**

If canary passes (agreement rate >90%, no regressions on key metrics), promote to 100% traffic. Monitor closely for 48 hours. If any metric degrades, immediate rollback to previous model.

---

## 15. End-to-End Evaluation Architecture

```
PRODUCTION PIPELINE
│
├──── [Every request] ──── LangSmith Trace ──── Real-time dashboard
│                                                    │
│                                              Anomaly detection
│                                              (latency, retries,
│                                               low scores)
│
├──── [Weekly: Sunday night] ──── Ragas Evaluation
│                                      │
│                                 Run 200-example dataset
│                                 through full pipeline
│                                      │
│                                 Compute: faithfulness,
│                                 relevancy, correctness,
│                                 recall, per-category
│                                      │
│                                 Compare against:
│                                 ├── Absolute thresholds
│                                 ├── Baseline (promotion time)
│                                 ├── Previous week
│                                 └── 8-week trend
│                                      │
│                                 Drift detected?
│                                 ├── No → log results, done
│                                 └── Yes → diagnose cause
│                                          │
│                                     ┌────┴────┐
│                                     │         │
│                                 Retrieval   Model
│                                 issue       issue
│                                     │         │
│                                 Trigger    Trigger
│                                 re-index   retrain
│
├──── [On model change] ──── Pre-promotion evaluation
│                                │
│                           Offline eval → Canary → Full promotion
│
├──── [On index change] ──── Post-index validation
│                                │
│                           Spot-check queries → Compare pre/post
│
└──── [Monthly] ──── Golden dataset refresh
                          │
                     Replace 25% oldest examples
                     Re-verify remaining 75%
                     Update baselines
```

---

## 16. Observability & Alerting for the Evaluation Pipeline Itself

### Evaluation Pipeline Metrics

| Metric | Description |
|--------|-------------|
| `eval.ragas.faithfulness` | Weekly faithfulness score |
| `eval.ragas.context_relevancy` | Weekly context relevancy score |
| `eval.ragas.answer_correctness` | Weekly answer correctness score |
| `eval.ragas.answer_relevancy` | Weekly answer relevancy score |
| `eval.custom.context_recall` | Weekly context recall score |
| `eval.drift.signals_detected` | Number of drift signals per eval run |
| `eval.drift.severity_distribution` | High/medium/low drift signal counts |
| `eval.dataset.age_days` | Days since last dataset refresh |
| `eval.dataset.expired_examples` | Examples past their expiry date |
| `eval.pipeline.duration_seconds` | How long the eval run took |
| `eval.pipeline.failures` | Examples that failed to process during eval |
| `eval.retraining.triggered` | Whether retraining was triggered |
| `eval.reindex.triggered` | Whether re-indexing was triggered |

### Alert Configuration

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Faithfulness below threshold | <0.90 | P1 | Model hallucinating — investigate context grounding |
| Context relevancy below threshold | <0.85 | P2 | Retrieval returning irrelevant docs — check index and queries |
| Answer correctness below threshold | <0.80 | P2 | End-to-end quality degraded — diagnose retrieval vs model |
| Context recall below threshold | <0.85 | P2 | Missing expected documents — check index completeness |
| Downward trend detected | Any metric declining >0.5%/week over 4 weeks | P2 | Gradual drift — schedule investigation and likely retraining |
| Safety-critical category regression | Any drop on safety_critical examples | P1 | Immediate investigation — safety handling degraded |
| Evaluation pipeline failed | Eval run didn't complete | P2 | Infrastructure issue — eval must run every week |
| Golden dataset stale | >50% of examples past expiry | P3 | Refresh dataset — evaluation may not be representative |
| Multiple drift signals | >3 high-severity signals in one eval | P1 | Systemic quality issue — urgent investigation |
| Evaluation not run | >10 days since last eval | P2 | Scheduled eval may have been skipped or failed |

---

## 17. End-to-End Evaluation Flow Summary

```
CONTINUOUS EVALUATION PIPELINE
│
│  ┌─────────────────────────────────────────────────────────────┐
│  │  PILLAR 1: LangSmith Tracing (Real-Time)                   │
│  │                                                             │
│  │  Every request automatically traced:                        │
│  │  • Per-node I/O, duration, errors                           │
│  │  • LLM prompts, completions, token counts                   │
│  │  • Retrieval queries, scores, chunks                        │
│  │  • Validation decisions, retry counts                       │
│  │  • Custom metadata tags for filtering                       │
│  │                                                             │
│  │  Enables:                                                   │
│  │  • Real-time anomaly detection (latency, errors, retries)   │
│  │  • Per-request debugging (trace walkthrough)                │
│  │  • Aggregate trend analysis (token usage, latency trends)   │
│  │  • Root cause investigation (find exact failing prompt)     │
│  └─────────────────────────────────────────────────────────────┘
│
│  ┌─────────────────────────────────────────────────────────────┐
│  │  PILLAR 2: Ragas Evaluation (Weekly)                        │
│  │                                                             │
│  │  200-example evaluation dataset:                            │
│  │  • 40% easy / 35% medium / 25% hard                        │
│  │  • Covers all product lines and defect categories           │
│  │  • Human-annotated ground truth (2-annotator + adjudicator) │
│  │  • Refreshed 25% quarterly                                  │
│  │                                                             │
│  │  Metrics computed:                                          │
│  │  • Faithfulness ≥0.90 (grounding in retrieved context)      │
│  │  • Context Relevancy ≥0.85 (retrieval precision)            │
│  │  • Answer Correctness ≥0.80 (end-to-end accuracy)           │
│  │  • Answer Relevancy ≥0.85 (output addresses the question)   │
│  │  • Context Recall ≥0.85 (expected docs found)               │
│  │  • Per-category breakdown (single/multi/ambiguous/safety)   │
│  │                                                             │
│  │  Results compared against:                                  │
│  │  • Absolute thresholds (quality floor)                      │
│  │  • Baseline from model promotion (regression detection)     │
│  │  • Previous week (sudden drops)                             │
│  │  • 8-week trend (gradual drift)                             │
│  └─────────────────────────────────────────────────────────────┘
│
│  ┌─────────────────────────────────────────────────────────────┐
│  │  PILLAR 3: Regression & Drift Detection (Weekly + Triggered)│
│  │                                                             │
│  │  Drift types monitored:                                     │
│  │  • Data drift (input patterns changing)                     │
│  │  • Model drift (effective accuracy declining)               │
│  │  • Concept drift (correct answers changing)                 │
│  │  • Index drift (documents stale in Pinecone)                │
│  │                                                             │
│  │  Detection methods:                                         │
│  │  • Absolute threshold violations                            │
│  │  • Baseline regression (>3% drop from promotion time)       │
│  │  • Week-over-week decline (>5% drop)                        │
│  │  • Trend analysis (>0.5%/week over 4+ weeks)                │
│  │  • Per-category regression (especially safety-critical)     │
│  │                                                             │
│  │  Automated responses:                                       │
│  │  • Drift diagnosed → retrieval issue OR model issue         │
│  │  • Retrieval issue → trigger SOP re-indexing                │
│  │  • Model issue → trigger retraining pipeline                │
│  │  • Safety regression → immediate P1 alert                  │
│  └─────────────────────────────────────────────────────────────┘
│
│  ┌─────────────────────────────────────────────────────────────┐
│  │  SUPPORTING PROCESSES                                       │
│  │                                                             │
│  │  Golden Dataset Management:                                 │
│  │  • 200 examples, versioned, refreshed quarterly             │
│  │  • 2-annotator + adjudicator for ground truth               │
│  │  • Expiry dates on all examples (6 months max)              │
│  │  • Distribution balanced by difficulty and category          │
│  │                                                             │
│  │  A/B Testing & Canary Evaluation:                           │
│  │  • Offline eval → canary (10% traffic) → full promotion     │
│  │  • Shadow evaluation: new model runs alongside production   │
│  │  • Agreement rate and side-by-side comparison               │
│  │  • 48-hour monitoring window after full promotion           │
│  │                                                             │
│  │  Feedback Loop:                                             │
│  │  • Human corrections → training data pipeline               │
│  │  • Correction classification and quality gating             │
│  │  • Monthly incorporation into retraining dataset            │
│  │  • Golden dataset enriched with production-verified examples │
│  └─────────────────────────────────────────────────────────────┘
```

### Key Design Principles Applied Throughout the Evaluation Pipeline

1. **Observe everything, alert selectively.** LangSmith traces every request. Ragas evaluates aggregate quality weekly. But alerts fire only when specific thresholds are violated. Observability without alerting is data hoarding. Alerting without observability is blind guessing.

2. **Multiple detection methods catch different failures.** Absolute thresholds catch severe degradation. Baseline regression catches gradual drift. Trend analysis catches slow decline. Per-category analysis catches targeted regression. No single method catches everything.

3. **Diagnose before acting.** When quality drops, the first step is diagnosis (retrieval issue vs model issue vs ground truth shift), not reflexive retraining. Wrong diagnosis → wrong fix → wasted effort and potentially making things worse.

4. **The golden dataset is an asset that requires maintenance.** A stale dataset gives false confidence. Quarterly refreshes, expiry dates, and representation monitoring ensure the dataset stays current and meaningful.

5. **Canary before promote.** New models see real production traffic at 10% before full deployment. This catches production-specific issues that offline evaluation misses.

6. **Safety-critical examples get special treatment.** Any regression on safety-critical examples triggers the highest alert priority. Safety is never traded for convenience or speed.

7. **The evaluation pipeline evaluates itself.** Metrics on evaluation pipeline health (run completion, dataset freshness, example expiry) ensure the evaluation system stays operational. An evaluation system that silently stops running provides zero value.

8. **Continuous improvement is built into the architecture.** Human corrections become training data. Production failures become golden dataset examples. Drift detection triggers retraining. The system doesn't just maintain quality — it actively improves over time.

---

*Document Version: 1.0 | Last Updated: March 2026 | System: LGS Tech Agentic AI Workflow*
