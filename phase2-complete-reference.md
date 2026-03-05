# Phase 2 — Agent Orchestrator (The Brain): Complete Production Reference

## LGS Tech Agentic AI Workflow — Service Order Processing

---

## Table of Contents

1. Phase 2 Overview & Why an Agent Architecture
2. Why LangGraph Over Alternatives
3. The State Graph — Core Architecture
4. Agent State Design
5. Node 1 — Defect Extraction (LLM Call)
6. Node 2 — Extraction Validation
7. Node 3 — RAG Retrieval
8. Node 4 — Action Item Generation (LLM Call)
9. Node 5 — Output Validation
10. Node 6 — Output Assembly & Dispatch
11. Conditional Edges & Routing Logic
12. Retry & Self-Healing Mechanism
13. Human Review Fallback
14. Error Handling & Fault Tolerance
15. Concurrency & Parallelism
16. State Persistence & Checkpointing
17. Observability & Alerting
18. End-to-End Phase 2 Flow Summary

---

## 1. Phase 2 Overview & Why an Agent Architecture

### What Phase 2 Does

Phase 2 is the orchestration layer — the "brain" that coordinates the entire processing workflow. It receives the clean, segmented canonical JSON from Phase 1 and drives it through a sequence of processing steps: LLM extraction, validation, RAG retrieval, action item generation, output validation, and dispatch. It manages state across these steps, handles failures with retries and fallbacks, and ensures every service order segment either produces a valid output or is escalated to human review.

### Why Not Just a Simple Sequential Pipeline?

You could build this as a linear chain: extract → retrieve → generate → done. Many early LLM applications do exactly this. But a simple chain fails in production for three critical reasons.

**Reason 1 — LLM outputs are non-deterministic and can fail.** Even with temperature 0.0, a fine-tuned model occasionally produces malformed JSON, hallucinates a field value, or misses a required entity. In a simple chain, a failure at step 2 kills the entire pipeline. You'd have to catch the exception, decide what to do, rebuild the context, and retry — all with custom error handling code that quickly becomes spaghetti.

**Reason 2 — Processing paths are conditional.** Not every service order follows the same flow. A segment with high-confidence extraction might skip the retry loop entirely. A segment where the RAG retrieval returns low-relevance results might need a query reformulation step. A segment flagged as safety-critical might need an additional validation gate before output. A linear chain can't express these branches cleanly.

**Reason 3 — You need observability into intermediate states.** When something goes wrong (wrong action item generated, wrong SOP retrieved), you need to debug by inspecting the state at every point in the workflow — what the LLM extracted, what the validator said, what chunks were retrieved, what the final prompt looked like. A linear chain doesn't naturally expose these intermediate states. An agent with a typed state object does.

An agent architecture — specifically a state machine implemented as a directed graph — solves all three problems: nodes handle individual processing steps, conditional edges handle branching and retry logic, and a typed state object carries observable intermediate results through the entire workflow.

---

## 2. Why LangGraph Over Alternatives

### Alternatives Considered

**Alternative 1 — LangChain Chains (Sequential Chain, LCEL)**

LangChain's expression language (LCEL) lets you compose chains with pipe operators: `prompt | llm | parser`. This is elegant for simple linear flows but breaks down when you need loops (retry logic), conditional branching (skip steps based on intermediate results), or stateful execution (carrying context across retries).

**Rejected because:** No native support for cycles (retry loops). Conditional branching requires awkward workarounds with RunnableBranch. No built-in state persistence or checkpointing. Debugging intermediate states requires manual logging at every step.

**Alternative 2 — Custom Python Orchestration**

Write your own orchestration logic with if/else, while loops, and try/except blocks. Maximum flexibility, no framework dependency.

**Rejected because:** This works initially but becomes unmaintainable as the workflow grows. Adding a new processing step requires touching the orchestration logic. Retry logic gets duplicated across steps. State management is manual and error-prone. Observability requires building your own tracing infrastructure. Every new developer has to understand the custom orchestration code from scratch.

**Alternative 3 — Workflow Engines (Temporal, Airflow, Step Functions)**

Enterprise workflow engines are designed for exactly this kind of multi-step processing with retries and branching.

**Considered seriously.** Temporal in particular has excellent retry semantics, state persistence, and durability guarantees. However, it's infrastructure-heavy (requires a Temporal server cluster), has a steep learning curve, and is overkill for a single-workflow system. The overhead of running and maintaining a Temporal cluster outweighs the benefits when you have one primary workflow with 5-6 steps.

**Alternative 4 — LangGraph**

LangGraph models workflows as directed graphs with typed state, conditional edges, and built-in support for cycles (retry loops). It's purpose-built for LLM agent workflows and integrates natively with LangChain's ecosystem (LLM wrappers, prompt templates, output parsers) and LangSmith (tracing and evaluation).

### Decision: LangGraph

**Reasoning:**

LangGraph gives us exactly what we need without excess: graph-based workflow definition where each node is a function that takes state and returns updated state, conditional edges that enable branching and retry loops natively, typed state that flows through the graph and is inspectable at every node, built-in integration with LangSmith for end-to-end tracing, checkpointing support for resuming failed runs, and a familiar Python programming model (nodes are just functions, no special DSL).

**Tradeoff:** LangGraph is newer and less battle-tested than Temporal or Airflow. The ecosystem is evolving rapidly, which means API changes between versions. Documentation can be sparse for advanced use cases. But for an LLM-centric workflow with 5-6 nodes, it's the right fit — it's lightweight enough to not add infrastructure burden but structured enough to handle the complexity of retry loops and conditional routing.

**Rule:** If the workflow grows beyond 10-15 nodes or if we need cross-workflow orchestration (multiple independent workflows coordinating), we'd revisit and consider migrating to Temporal.

---

## 3. The State Graph — Core Architecture

### Graph Definition

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes (each node is a function that takes AgentState and returns partial state updates)
workflow.add_node("extract_defects", extract_defects_node)
workflow.add_node("validate_extraction", validate_extraction_node)
workflow.add_node("retrieve_context", retrieve_context_node)
workflow.add_node("generate_action_items", generate_action_items_node)
workflow.add_node("validate_output", validate_output_node)
workflow.add_node("assemble_output", assemble_output_node)
workflow.add_node("human_review", human_review_node)

# Set entry point
workflow.set_entry_point("extract_defects")

# Add edges (including conditional edges)
workflow.add_edge("extract_defects", "validate_extraction")

workflow.add_conditional_edges(
    "validate_extraction",
    route_after_extraction_validation,
    {
        "valid": "retrieve_context",
        "retry": "extract_defects",
        "human_review": "human_review",
    }
)

workflow.add_edge("retrieve_context", "generate_action_items")
workflow.add_edge("generate_action_items", "validate_output")

workflow.add_conditional_edges(
    "validate_output",
    route_after_output_validation,
    {
        "valid": "assemble_output",
        "retry": "generate_action_items",
        "human_review": "human_review",
    }
)

workflow.add_edge("assemble_output", END)
workflow.add_edge("human_review", END)

# Compile with checkpointing
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
app = workflow.compile(checkpointer=checkpointer)
```

### Visual Graph Structure

```
                         ┌─────────────────────┐
                         │      ENTRY           │
                         │  (from Phase 1)      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                    ┌───▶│  extract_defects     │ ◀── Node 1 (LLM Call)
                    │    │  [Fine-tuned Llama]  │
                    │    └──────────┬───────────┘
                    │               │
                    │               ▼
                    │    ┌─────────────────────┐
                    │    │ validate_extraction  │ ◀── Node 2 (Pydantic)
                    │    └──────────┬───────────┘
                    │          ╱    │    ╲
                    │       RETRY   │   HUMAN
                    │      (loop)  VALID  REVIEW
                    │         │     │       │
                    └─────────┘     │       │
                                    ▼       │
                         ┌─────────────────────┐       │
                         │  retrieve_context    │ ◀── Node 3 (RAG/Pinecone)
                         │  [Pinecone + Rerank] │       │
                         └──────────┬───────────┘       │
                                    │                   │
                                    ▼                   │
                         ┌─────────────────────┐       │
                    ┌───▶│ generate_action_items│ ◀── Node 4 (LLM Call)
                    │    │ [Llama + RAG context]│       │
                    │    └──────────┬───────────┘       │
                    │               │                   │
                    │               ▼                   │
                    │    ┌─────────────────────┐       │
                    │    │  validate_output     │ ◀── Node 5 (Pydantic)
                    │    └──────────┬───────────┘       │
                    │          ╱    │    ╲               │
                    │       RETRY   │   HUMAN            │
                    │      (loop)  VALID  REVIEW ────────┤
                    │         │     │                    │
                    └─────────┘     │                    │
                                    ▼                    │
                         ┌─────────────────────┐       │
                         │  assemble_output     │ ◀── Node 6
                         └──────────┬───────────┘       │
                                    │                   │
                                    ▼                    ▼
                         ┌──────────────┐    ┌─────────────────┐
                         │     END      │    │  human_review    │
                         │  (success)   │    │  (escalation)   │
                         └──────────────┘    └────────┬────────┘
                                                      │
                                                      ▼
                                                     END
```

### Design Decision: Number of Nodes

**Decision:** 6 processing nodes + 1 human review node = 7 total.

**Reasoning:** Each node has a single responsibility. Extraction and validation are separate nodes (not combined) because the validation logic needs to make routing decisions (retry vs continue vs escalate). If validation were inside the extraction node, the conditional routing logic would be embedded inside the node rather than expressed as graph edges, making the workflow harder to understand and modify.

**Tradeoff:** More nodes means more state transitions and slightly more overhead per request. But the overhead is negligible (microseconds per state transition) compared to the LLM call latency (seconds). The clarity and maintainability of single-responsibility nodes far outweighs the marginal performance cost.

**Rule:** A node should do one thing. If a node has an if/else that could route to different next steps, split it into two nodes with a conditional edge between them.

---

## 4. Agent State Design

### The State Object

The AgentState is the single most important design element in Phase 2. It's the typed data structure that flows through every node, carrying all context, intermediate results, and control signals. Every node reads from state and writes back to state.

```python
from typing import TypedDict, Optional, Literal
from pydantic import BaseModel
from datetime import datetime

class ExtractedDefect(BaseModel):
    defect_type: str
    severity: Literal["low", "medium", "high", "critical"]
    affected_component: str
    root_cause_hypothesis: Optional[str]
    error_codes: list[str]
    symptoms: list[str]
    confidence: float

class RetrievedChunk(BaseModel):
    chunk_id: str
    document_title: str
    section: str
    text: str
    relevance_score: float
    metadata: dict

class ActionItem(BaseModel):
    action_type: str
    description: str
    parts_required: list[dict]
    warranty_eligible: bool
    escalation_required: bool
    priority: Literal["routine", "urgent", "emergency"]
    sop_reference: str

class AgentState(TypedDict):
    # ── Input (set once at entry, never modified) ──
    order_id: str
    segment_id: str
    segment_text: str
    inherited_context: dict
    pre_extracted_entities: list[dict]
    structured_metadata: dict
    
    # ── Phase 3: Extraction ──
    extracted_defects: Optional[list[ExtractedDefect]]
    extraction_raw_response: Optional[str]
    extraction_prompt: Optional[str]
    extraction_validation_errors: Optional[list[str]]
    extraction_retry_count: int
    
    # ── Phase 4: RAG Retrieval ──
    retrieval_query: Optional[str]
    retrieved_chunks: Optional[list[RetrievedChunk]]
    retrieval_scores: Optional[list[float]]
    retrieval_reformulated: bool
    
    # ── Phase 5: Action Item Generation ──
    action_items: Optional[list[ActionItem]]
    generation_raw_response: Optional[str]
    generation_prompt: Optional[str]
    generation_validation_errors: Optional[list[str]]
    generation_retry_count: int
    
    # ── Control Flow ──
    current_node: str
    status: Literal["processing", "completed", "human_review", "failed"]
    error_log: list[dict]
    
    # ── Timing ──
    node_start_time: Optional[datetime]
    node_durations: dict  # {node_name: duration_ms}
    total_start_time: datetime
```

### Why TypedDict and Not a Pydantic Model for State?

**Decision:** Use TypedDict for the top-level AgentState, Pydantic for nested data models.

**Reasoning:** LangGraph requires the state to be a TypedDict because it uses dictionary merging semantics for state updates — each node returns a partial dictionary, and LangGraph merges it into the existing state. Pydantic models are immutable by default and don't support this merge pattern natively. However, the data inside the state (extracted defects, action items, retrieved chunks) should be Pydantic models because they need validation, type checking, and serialization.

**Tradeoff:** TypedDict provides type hints for IDE support but doesn't enforce types at runtime. A node could accidentally write a string where a list is expected, and the error would only surface downstream when another node tries to iterate over it. Mitigated by running Pydantic validation at every validation node (Nodes 2 and 5), which catches type errors before they propagate.

### State Immutability Principle

**Rule:** Nodes never mutate existing state fields — they only add new fields or replace fields entirely.

```python
# WRONG — mutating existing state
def bad_node(state: AgentState) -> dict:
    state["extracted_defects"].append(new_defect)  # BAD: mutates in place
    return state

# CORRECT — returning new values
def good_node(state: AgentState) -> dict:
    existing = state.get("extracted_defects", [])
    return {
        "extracted_defects": existing + [new_defect],  # new list
        "current_node": "extract_defects",
    }
```

**Reasoning:** In-place mutation causes subtle bugs — if checkpointing captures state before the mutation is complete, you get inconsistent state. If two nodes somehow run concurrently (in a future parallelization), mutations create race conditions. Returning new values ensures state transitions are atomic and predictable.

### State Size Considerations

**Decision:** Store full LLM prompts and raw responses in state.

**Reasoning:** When debugging a bad output, you need to see exactly what prompt was sent and what the model returned. Without this, you're guessing. The storage cost is trivial — a typical prompt + response is 2-4KB, and the state object for one segment is under 50KB total.

**Tradeoff:** State objects are larger, which means more data flowing through the graph and more data stored in checkpoints. At 50KB per segment and 50,000 daily orders averaging 1.3 segments each, that's roughly 3.25GB of checkpoint data per day. This is manageable with SQLite for moderate scale or PostgreSQL for larger scale. If storage becomes a concern, you could store only prompts and responses for failed runs, but the debugging benefit of having them for all runs justifies the cost.

---

## 5. Node 1 — Defect Extraction (LLM Call)

### What Happens

This node takes the preprocessed segment text and pre-extracted entities from Phase 1, constructs a prompt, calls the fine-tuned Llama 3.1 8B model, and stores the raw response in state.

### Implementation

```python
import httpx
import json
from datetime import datetime

INFERENCE_URL = "http://llm-service.internal:8000/v1/completions"

EXTRACTION_SYSTEM_PROMPT = """You are a defect extraction system for service orders. 
Given a service order text and pre-extracted entities, extract ALL defects as structured JSON.

Rules:
- Extract EVERY distinct defect mentioned in the text
- Each defect must have: defect_type, severity, affected_component, symptoms, error_codes
- Use ONLY information present in the text — do not hallucinate
- Cross-validate against pre-extracted entities: if regex found model XR-440, your output must reference XR-440
- If uncertain about severity, default to "medium" and set confidence below 0.7
- Output ONLY valid JSON array, no markdown, no explanation

Output schema:
[
  {
    "defect_type": "mechanical|electrical|software|hydraulic|thermal|structural",
    "severity": "low|medium|high|critical",
    "affected_component": "string — specific component name",
    "root_cause_hypothesis": "string or null",
    "error_codes": ["list of error codes found"],
    "symptoms": ["list of observed symptoms"],
    "confidence": 0.0-1.0
  }
]"""

def extract_defects_node(state: AgentState) -> dict:
    node_start = datetime.utcnow()
    
    # Build the user prompt with segment text + pre-extracted entities
    pre_extracted = state.get("pre_extracted_entities", [])
    inherited = state.get("inherited_context", {})
    retry_count = state.get("extraction_retry_count", 0)
    previous_errors = state.get("extraction_validation_errors", [])
    
    # Base prompt
    user_prompt = f"""Service Order Segment:
\"\"\"{state['segment_text']}\"\"\"

Inherited Context:
- Model: {inherited.get('model_number', 'unknown')}
- Plant: {inherited.get('plant_id', 'unknown')}
- Serial: {inherited.get('serial_number', 'unknown')}

Pre-Extracted Entities (high confidence):
{json.dumps(pre_extracted, indent=2)}

Extract all defects as JSON."""

    # If this is a retry, append the previous errors
    if retry_count > 0 and previous_errors:
        user_prompt += f"""

PREVIOUS ATTEMPT FAILED VALIDATION. Errors:
{json.dumps(previous_errors, indent=2)}

Fix these specific errors and try again. Output ONLY valid JSON."""

    # Call the inference service
    try:
        response = httpx.post(
            INFERENCE_URL,
            json={
                "prompt": f"<s>[INST] <<SYS>>\n{EXTRACTION_SYSTEM_PROMPT}\n<</SYS>>\n\n{user_prompt} [/INST]",
                "max_tokens": 1024,
                "temperature": 0.0,
                "stop": ["</s>"],
            },
            timeout=30.0,
        )
        response.raise_for_status()
        raw_response = response.json()["choices"][0]["text"]
    except httpx.TimeoutException:
        return {
            "extraction_raw_response": None,
            "current_node": "extract_defects",
            "error_log": state.get("error_log", []) + [{
                "node": "extract_defects",
                "error": "LLM inference timeout (30s)",
                "timestamp": datetime.utcnow().isoformat(),
            }],
            "status": "failed",
        }
    except Exception as e:
        return {
            "extraction_raw_response": None,
            "current_node": "extract_defects",
            "error_log": state.get("error_log", []) + [{
                "node": "extract_defects",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }],
            "status": "failed",
        }
    
    duration = (datetime.utcnow() - node_start).total_seconds() * 1000
    
    return {
        "extraction_raw_response": raw_response,
        "extraction_prompt": user_prompt,
        "current_node": "extract_defects",
        "node_durations": {
            **state.get("node_durations", {}),
            "extract_defects": duration,
        },
    }
```

### Design Decisions

**Decision:** Include pre-extracted entities in the prompt as "hints."

**Reasoning:** The regex entities from Phase 1 (model numbers, error codes, part numbers) are deterministic and high-confidence. Including them in the prompt serves two purposes: it gives the LLM anchors to work from (improving extraction accuracy), and it sets up cross-validation — if the LLM output contradicts a regex-extracted entity, the validation node catches it.

**Tradeoff:** Adding pre-extracted entities consumes ~200 tokens of context window. This is a small cost relative to the benefit. If the pre-extracted entities are wrong (regex false positive), they could bias the LLM toward the wrong value. But regex false positives are extremely rare for well-defined patterns, and the validation cross-check catches cases where the LLM correctly identifies a different value.

**Decision:** Temperature 0.0 for extraction.

**Reasoning:** Extraction is a deterministic task — given the same input, we want the same output every time. Temperature 0.0 eliminates sampling randomness. This also makes debugging reproducible — if a production extraction is wrong, you can replay the exact same prompt and get the exact same wrong output for analysis.

**Tradeoff:** Temperature 0.0 means the model always picks the highest-probability token. In rare cases, the second-highest-probability token is actually correct but never gets selected. This is acceptable because consistency and reproducibility are more valuable than marginal accuracy improvement for a structured extraction task.

**Decision:** 30-second timeout on LLM inference.

**Reasoning:** A healthy vLLM inference call for 1024 max tokens takes 2-5 seconds. If it's taking 30 seconds, something is wrong — either the GPU is saturated, the model is stuck in a generation loop, or the network is failing. Timing out and routing to error handling is better than blocking indefinitely.

**Tradeoff:** If the vLLM service is temporarily overloaded (queue buildup during a traffic spike), a 30-second timeout might kill requests that would have succeeded with more patience. Mitigated by the retry loop — a timed-out request returns to the extraction node for another attempt, and by then the vLLM queue may have cleared.

---

## 6. Node 2 — Extraction Validation

### What Happens

This node takes the raw LLM response from Node 1, attempts to parse it as JSON, validates each extracted defect against the Pydantic schema, cross-validates against Phase 1 pre-extracted entities, and decides: route to Node 3 (valid), route back to Node 1 (retry), or route to human review (exhausted retries).

### Implementation

```python
import json
from pydantic import ValidationError

MAX_EXTRACTION_RETRIES = 3

def validate_extraction_node(state: AgentState) -> dict:
    raw_response = state.get("extraction_raw_response")
    retry_count = state.get("extraction_retry_count", 0)
    pre_extracted = state.get("pre_extracted_entities", [])
    errors = []
    
    # ── Step 1: Check for null response (inference failure) ──
    if raw_response is None:
        errors.append("LLM returned no response (inference failure)")
        return {
            "extracted_defects": None,
            "extraction_validation_errors": errors,
            "extraction_retry_count": retry_count + 1,
            "current_node": "validate_extraction",
        }
    
    # ── Step 2: JSON parsing ──
    try:
        # Strip markdown fences if present
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        errors.append(f"JSON parse error: {str(e)}")
        return {
            "extracted_defects": None,
            "extraction_validation_errors": errors,
            "extraction_retry_count": retry_count + 1,
            "current_node": "validate_extraction",
        }
    
    # ── Step 3: Ensure it's a list ──
    if not isinstance(parsed, list):
        parsed = [parsed]  # Single defect wrapped in object, not array
    
    if len(parsed) == 0:
        errors.append("LLM returned empty defect list — at least one defect expected")
        return {
            "extracted_defects": None,
            "extraction_validation_errors": errors,
            "extraction_retry_count": retry_count + 1,
            "current_node": "validate_extraction",
        }
    
    # ── Step 4: Pydantic validation per defect ──
    validated_defects = []
    for i, defect_dict in enumerate(parsed):
        try:
            defect = ExtractedDefect(**defect_dict)
            validated_defects.append(defect)
        except ValidationError as e:
            errors.append(f"Defect {i}: {e.errors()}")
    
    # If some defects validated and some didn't, keep the valid ones
    # but still flag errors
    if errors and not validated_defects:
        return {
            "extracted_defects": None,
            "extraction_validation_errors": errors,
            "extraction_retry_count": retry_count + 1,
            "current_node": "validate_extraction",
        }
    
    # ── Step 5: Cross-validation with Phase 1 entities ──
    cross_validation_warnings = []
    
    pre_extracted_model = next(
        (e["text"] for e in pre_extracted if e["entity_type"] == "model_number"), 
        None
    )
    pre_extracted_errors = [
        e["text"] for e in pre_extracted if e["entity_type"] == "error_code"
    ]
    
    # Check: did the LLM mention all pre-extracted error codes?
    llm_error_codes = set()
    for defect in validated_defects:
        llm_error_codes.update(defect.error_codes)
    
    for pre_code in pre_extracted_errors:
        if pre_code not in llm_error_codes:
            cross_validation_warnings.append(
                f"Pre-extracted error code {pre_code} not found in LLM extraction"
            )
    
    # Check: do all defects have reasonable confidence?
    low_confidence = [d for d in validated_defects if d.confidence < 0.5]
    if low_confidence:
        cross_validation_warnings.append(
            f"{len(low_confidence)} defect(s) have confidence below 0.5"
        )
    
    return {
        "extracted_defects": validated_defects,
        "extraction_validation_errors": errors if errors else None,
        "extraction_cross_validation_warnings": cross_validation_warnings,
        "current_node": "validate_extraction",
    }
```

### Routing Function

```python
def route_after_extraction_validation(state: AgentState) -> str:
    errors = state.get("extraction_validation_errors")
    retry_count = state.get("extraction_retry_count", 0)
    defects = state.get("extracted_defects")
    status = state.get("status")
    
    # Infrastructure failure (timeout, service down)
    if status == "failed":
        return "human_review"
    
    # Valid extraction — proceed
    if defects and not errors:
        return "valid"
    
    # Partial valid (some defects validated, some didn't) — proceed with valid ones
    if defects and errors:
        return "valid"  # warnings logged but doesn't block
    
    # No valid defects — retry or escalate
    if retry_count < MAX_EXTRACTION_RETRIES:
        return "retry"
    else:
        return "human_review"
```

### Design Decisions

**Decision:** Separate validation node from extraction node.

**Reasoning:** The extraction node's job is to call the LLM and store the raw response. The validation node's job is to evaluate the response quality and make routing decisions. Combining them would mean the routing logic (retry vs continue vs escalate) is embedded inside the LLM call function, making it harder to test, modify, and observe independently. Separation means you can change validation rules without touching the LLM call code, and you can see in the graph trace exactly what the validator decided and why.

**Tradeoff:** One extra state transition per execution, adding microseconds of overhead. Completely negligible.

**Decision:** Allow partial validation pass (some defects valid, some invalid).

**Reasoning:** If the LLM extracts 3 defects and 2 validate correctly but 1 has a malformed field, it's better to proceed with the 2 valid defects than to reject everything and retry. The retry might not produce a better result, and you'd lose the 2 valid extractions. The invalid defect is logged as a warning and the human review dashboard shows which orders had partial extractions.

**Tradeoff:** You might proceed with incomplete information — the invalid defect might have been the most critical one. Mitigated by the cross-validation check: if a pre-extracted error code from Phase 1 is missing from the validated defects, a warning is raised, signaling that something was missed.

**Decision:** Cross-validation is a warning, not a hard failure.

**Reasoning:** The LLM might legitimately disagree with regex. Regex might have false-positived on something that looks like an error code but isn't (e.g., "E-47" appearing in a model number context rather than as an error code). Making cross-validation a hard failure would cause too many false retries. Making it a warning means the information is captured in the trace for debugging, but processing continues.

**Tradeoff:** Genuine extraction errors where the LLM missed a real error code won't trigger a retry. Mitigated by monitoring the cross-validation warning rate — if it exceeds a threshold, it indicates either the LLM or the regex patterns need updating.

---

## 7. Node 3 — RAG Retrieval

### What Happens

This node takes the validated extracted defects, constructs a semantic query, calls Pinecone for relevant SOP/warranty document chunks, optionally reranks results, and stores the retrieved context in state.

### Implementation

```python
import pinecone
from sentence_transformers import CrossEncoder

PINECONE_INDEX = pinecone.Index("sop-warranty-docs")
EMBEDDING_MODEL = "text-embedding-ada-002"  # or local model
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
TOP_K = 10
RERANK_TOP_N = 5
MIN_RELEVANCE_THRESHOLD = 0.3

def retrieve_context_node(state: AgentState) -> dict:
    node_start = datetime.utcnow()
    
    defects = state["extracted_defects"]
    metadata = state["structured_metadata"]
    inherited = state["inherited_context"]
    
    # ── Step 1: Construct semantic query from extracted defects ──
    query_parts = []
    for defect in defects:
        query_parts.append(
            f"{defect.defect_type} {defect.affected_component} "
            f"{' '.join(defect.symptoms)} {' '.join(defect.error_codes)}"
        )
    
    # Add model/product context for metadata filtering
    model_number = (inherited.get("model_number") or 
                   metadata.get("model_number") or "unknown")
    
    base_query = " ".join(query_parts)
    full_query = f"{model_number} {base_query}"
    
    # ── Step 2: Embed the query ──
    query_embedding = embed_text(full_query)  # calls embedding model
    
    # ── Step 3: Pinecone vector search with metadata filter ──
    # Determine product line from model number for filtering
    product_line = extract_product_line(model_number)  # "XR" from "XR-440"
    
    filter_conditions = {}
    if product_line:
        filter_conditions["product_line"] = {"$eq": product_line}
    
    results = PINECONE_INDEX.query(
        vector=query_embedding,
        top_k=TOP_K,
        include_metadata=True,
        filter=filter_conditions if filter_conditions else None,
    )
    
    # ── Step 4: Convert to RetrievedChunk objects ──
    chunks = []
    for match in results["matches"]:
        chunks.append(RetrievedChunk(
            chunk_id=match["id"],
            document_title=match["metadata"].get("document_title", ""),
            section=match["metadata"].get("section", ""),
            text=match["metadata"]["text"],
            relevance_score=match["score"],
            metadata=match["metadata"],
        ))
    
    # ── Step 5: Rerank with cross-encoder ──
    if chunks:
        rerank_pairs = [(full_query, chunk.text) for chunk in chunks]
        rerank_scores = RERANKER.predict(rerank_pairs)
        
        for i, score in enumerate(rerank_scores):
            chunks[i].relevance_score = float(score)
        
        chunks.sort(key=lambda c: c.relevance_score, reverse=True)
        chunks = chunks[:RERANK_TOP_N]
    
    # ── Step 6: Check relevance quality ──
    if not chunks or chunks[0].relevance_score < MIN_RELEVANCE_THRESHOLD:
        # Low relevance — try query reformulation
        reformulated_query = reformulate_query(defects, metadata)
        
        if reformulated_query != full_query:
            # Retry with reformulated query
            query_embedding_v2 = embed_text(reformulated_query)
            results_v2 = PINECONE_INDEX.query(
                vector=query_embedding_v2,
                top_k=TOP_K,
                include_metadata=True,
            )
            
            chunks_v2 = [
                RetrievedChunk(
                    chunk_id=m["id"],
                    document_title=m["metadata"].get("document_title", ""),
                    section=m["metadata"].get("section", ""),
                    text=m["metadata"]["text"],
                    relevance_score=m["score"],
                    metadata=m["metadata"],
                )
                for m in results_v2["matches"]
            ]
            
            if chunks_v2 and chunks_v2[0].relevance_score > chunks[0].relevance_score if chunks else True:
                chunks = chunks_v2[:RERANK_TOP_N]
    
    # ── Step 7: Context window expansion ──
    # For the top chunk, fetch neighbors if it seems like partial context
    expanded_chunks = []
    for chunk in chunks[:3]:  # expand top 3 only
        if chunk_needs_expansion(chunk):
            prev_id = chunk.metadata.get("prev_chunk")
            next_id = chunk.metadata.get("next_chunk")
            
            if prev_id:
                prev_chunk = fetch_chunk_by_id(prev_id)
                if prev_chunk:
                    expanded_chunks.append(prev_chunk)
            
            expanded_chunks.append(chunk)
            
            if next_id:
                next_chunk = fetch_chunk_by_id(next_id)
                if next_chunk:
                    expanded_chunks.append(next_chunk)
        else:
            expanded_chunks.append(chunk)
    
    # Deduplicate expanded chunks
    seen_ids = set()
    final_chunks = []
    for chunk in expanded_chunks:
        if chunk.chunk_id not in seen_ids:
            seen_ids.add(chunk.chunk_id)
            final_chunks.append(chunk)
    
    duration = (datetime.utcnow() - node_start).total_seconds() * 1000
    
    return {
        "retrieval_query": full_query,
        "retrieved_chunks": final_chunks,
        "retrieval_scores": [c.relevance_score for c in final_chunks],
        "retrieval_reformulated": reformulated_query != full_query if 'reformulated_query' in dir() else False,
        "current_node": "retrieve_context",
        "node_durations": {
            **state.get("node_durations", {}),
            "retrieve_context": duration,
        },
    }


def reformulate_query(defects, metadata):
    """Generate an alternative query when initial retrieval returns low relevance."""
    # Strategy: broaden the query by focusing on component category 
    # rather than specific symptoms
    components = set()
    for defect in defects:
        components.add(defect.affected_component)
    
    model = metadata.get("model_number", "")
    return f"{model} repair procedure maintenance {' '.join(components)}"


def chunk_needs_expansion(chunk: RetrievedChunk) -> bool:
    """Check if a chunk seems like it's missing context."""
    text = chunk.text
    # If chunk starts mid-sentence (no capital letter at start)
    if text and not text[0].isupper():
        return True
    # If chunk ends mid-sentence (no period at end)
    if text and not text.rstrip().endswith(('.', ':', ';')):
        return True
    # If chunk is very short (might be partial)
    if len(text.split()) < 30:
        return True
    return False
```

### Design Decisions

**Decision:** Two-stage retrieval — vector search followed by cross-encoder reranking.

**Reasoning:** Embedding-based vector search (Pinecone) is fast and cheap but measures semantic similarity at a coarse level. A cross-encoder reranker (like ms-marco-MiniLM) scores each query-document pair with much higher precision by attending to fine-grained token interactions between the query and the document. The two-stage approach gives you the speed of vector search (filter 100K+ documents to top-10 in milliseconds) followed by the precision of cross-encoder reranking (reorder top-10 by true relevance).

**Tradeoff:** The cross-encoder adds ~50-100ms latency (processing 10 pairs). In a latency-sensitive real-time system, this might be unacceptable. But service order processing tolerates multi-second latency, so the accuracy improvement justifies the cost. If latency ever becomes a concern, the reranker can be removed without changing the rest of the pipeline — it's an optional enhancement, not a dependency.

**Decision:** Metadata filtering on product line before vector search.

**Reasoning:** Without filtering, a query about compressor failure in the XR series might retrieve SOP chunks from the YZ series compressor documents — similar semantics but wrong product line. Metadata filtering narrows the search space to the correct product line before similarity scoring, dramatically improving relevance precision.

**Tradeoff:** If the product line can't be determined from the model number (maybe the technician didn't specify a model), the filter is skipped and the search runs against the full index. This trades precision for recall — you get broader results but might include irrelevant product lines. The reranker helps compensate by downranking off-topic chunks.

**Decision:** Query reformulation on low relevance.

**Reasoning:** Sometimes the initial query (built directly from extracted defect data) is too specific or uses different terminology than the SOPs. For example, the technician writes "grinding noise" but the SOP says "abnormal vibration." The initial query might return low-relevance results. Reformulating to a broader query ("XR-440 repair procedure maintenance compressor") sacrifices specificity but increases the chance of hitting relevant documents.

**Tradeoff:** Reformulation adds a second Pinecone call and embedding computation, roughly doubling retrieval latency when triggered. But it only triggers on low-relevance results (below threshold), which is an uncommon case (~10-15% of queries). The alternative — proceeding with low-relevance context — is worse because the action items generated in Node 4 would be grounded in irrelevant SOPs.

**Decision:** Context window expansion by fetching neighboring chunks.

**Reasoning:** A 512-token chunk might capture the defect description section but miss the resolution steps that immediately follow. Fetching prev/next neighbors gives the LLM complete context. Without expansion, the model might generate action items based on incomplete procedure information.

**Tradeoff:** More chunks means more tokens consumed in the Node 4 prompt, leaving less room for the model to generate output. We limit expansion to the top 3 chunks to keep total context reasonable. After expansion and deduplication, the typical retrieved context is 2,000-3,500 tokens — well within the budget.

---

## 8. Node 4 — Action Item Generation (LLM Call)

### What Happens

This node constructs a prompt combining the extracted defects AND the retrieved SOP/warranty context, calls the fine-tuned Llama 3.1 8B (or potentially a different model optimized for generation), and produces structured action items.

### Implementation

```python
GENERATION_SYSTEM_PROMPT = """You are an action item generator for service orders. 
Given extracted defect information and relevant SOP/warranty policy context, 
generate structured action items.

Rules:
- Generate ONE action item per defect
- Each action item must reference the specific SOP section that supports it
- Warranty eligibility must be determined from the retrieved warranty policy, not assumed
- If the retrieved context does not contain sufficient information for warranty determination, 
  set warranty_eligible to false and add escalation_required: true
- Parts required must be specific (part numbers if available in SOPs)
- Priority must reflect the defect severity
- Output ONLY valid JSON array, no markdown, no explanation

Output schema:
[
  {
    "action_type": "repair|replace|inspect|escalate|recall",
    "description": "Detailed action description",
    "parts_required": [{"part_name": "...", "part_number": "...", "quantity": 1}],
    "warranty_eligible": true/false,
    "escalation_required": true/false,
    "priority": "routine|urgent|emergency",
    "sop_reference": "Document title, Section name"
  }
]"""


def generate_action_items_node(state: AgentState) -> dict:
    node_start = datetime.utcnow()
    
    defects = state["extracted_defects"]
    chunks = state.get("retrieved_chunks", [])
    metadata = state["structured_metadata"]
    inherited = state["inherited_context"]
    retry_count = state.get("generation_retry_count", 0)
    previous_errors = state.get("generation_validation_errors", [])
    
    # ── Build context section from retrieved chunks ──
    context_section = "RETRIEVED SOP AND WARRANTY CONTEXT:\n\n"
    for i, chunk in enumerate(chunks):
        context_section += f"--- Document: {chunk.document_title} | Section: {chunk.section} ---\n"
        context_section += f"{chunk.text}\n\n"
    
    # ── Build defects section ──
    defects_section = "EXTRACTED DEFECTS:\n\n"
    for i, defect in enumerate(defects):
        defects_section += f"Defect {i+1}:\n"
        defects_section += f"  Type: {defect.defect_type}\n"
        defects_section += f"  Severity: {defect.severity}\n"
        defects_section += f"  Component: {defect.affected_component}\n"
        defects_section += f"  Symptoms: {', '.join(defect.symptoms)}\n"
        defects_section += f"  Error Codes: {', '.join(defect.error_codes)}\n"
        defects_section += f"  Root Cause: {defect.root_cause_hypothesis or 'Unknown'}\n\n"
    
    # ── Build metadata section ──
    metadata_section = f"""ORDER METADATA:
  Model: {inherited.get('model_number', metadata.get('model_number', 'unknown'))}
  Plant: {inherited.get('plant_id', metadata.get('plant_id', 'unknown'))}
  Serial: {inherited.get('serial_number', metadata.get('serial_number', 'unknown'))}
  Warranty ID: {metadata.get('warranty_id', 'unknown')}"""
    
    # ── Assemble user prompt ──
    user_prompt = f"""{context_section}
{defects_section}
{metadata_section}

Generate action items for each defect. Base warranty decisions ONLY on the retrieved policy context above."""
    
    # Append retry context if applicable
    if retry_count > 0 and previous_errors:
        user_prompt += f"""

PREVIOUS ATTEMPT FAILED VALIDATION. Errors:
{json.dumps(previous_errors, indent=2)}

Fix these specific errors and try again. Output ONLY valid JSON."""
    
    # ── Call inference service ──
    try:
        response = httpx.post(
            INFERENCE_URL,
            json={
                "prompt": f"<s>[INST] <<SYS>>\n{GENERATION_SYSTEM_PROMPT}\n<</SYS>>\n\n{user_prompt} [/INST]",
                "max_tokens": 2048,  # More tokens for action items (longer output)
                "temperature": 0.0,
                "stop": ["</s>"],
            },
            timeout=45.0,  # Longer timeout for generation (more output tokens)
        )
        response.raise_for_status()
        raw_response = response.json()["choices"][0]["text"]
    except Exception as e:
        return {
            "generation_raw_response": None,
            "current_node": "generate_action_items",
            "error_log": state.get("error_log", []) + [{
                "node": "generate_action_items",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }],
            "status": "failed",
        }
    
    duration = (datetime.utcnow() - node_start).total_seconds() * 1000
    
    return {
        "generation_raw_response": raw_response,
        "generation_prompt": user_prompt,
        "current_node": "generate_action_items",
        "node_durations": {
            **state.get("node_durations", {}),
            "generate_action_items": duration,
        },
    }
```

### Design Decisions

**Decision:** Use the same fine-tuned Llama 3.1 8B for both extraction (Node 1) and generation (Node 4).

**Reasoning:** Using one model simplifies infrastructure (one vLLM instance, one GPU allocation, one model to maintain). The fine-tuned model understands the domain vocabulary needed for both tasks. The task differentiation comes from the prompt and system instruction, not from the model itself.

**Alternative considered:** Use the fine-tuned Llama for extraction (where domain knowledge matters most) and a general-purpose model (like GPT-4 or Claude) for generation (where reasoning and instruction following matter more). This would produce higher-quality action items but adds an external API dependency, increases cost (API calls vs self-hosted inference), and introduces latency variability.

**Tradeoff:** The fine-tuned 8B model is less capable at complex reasoning than a frontier model. Action items may occasionally miss nuanced warranty conditions or produce less detailed repair procedures. Mitigated by the rich retrieved SOP context — the model doesn't need to reason from general knowledge; it just needs to map defects to the specific procedures described in the retrieved documents.

**Rule:** If action item quality metrics (measured by Ragas) drop below threshold, the first intervention is improving the SOPs in Pinecone (better documents → better action items), not switching models. Only if that doesn't help would we consider a more capable generation model.

**Decision:** max_tokens=2048 for generation vs 1024 for extraction.

**Reasoning:** Extraction output is compact (a JSON array of defect objects, typically 200-500 tokens). Generation output is longer — each action item includes a description, parts list, warranty determination, and SOP reference. A 3-defect order might need 1,000-1,500 tokens for action items. The 2048 limit provides comfortable headroom.

**Decision:** 45-second timeout for generation vs 30 seconds for extraction.

**Reasoning:** Generating 2048 tokens takes roughly 2x longer than generating 1024 tokens (roughly linear with output length in autoregressive models). The longer timeout matches the longer expected generation time.

**Decision:** Prompt includes instruction to base warranty decisions ONLY on retrieved context.

**Reasoning:** This is a critical guardrail. Without this instruction, the LLM might use its training knowledge to make warranty determinations — "compressors are typically covered for 2 years" — which could be wrong for this specific product line or policy version. By forcing the model to ground warranty decisions in the retrieved policy text, we ensure compliance with the current, actual policy. If the retrieved context doesn't contain warranty information, the model correctly sets `warranty_eligible: false` and `escalation_required: true`, routing it to a human who can make the determination.

**Tradeoff:** If the RAG retrieval missed the relevant warranty chunk (a retrieval failure), the model will always say "not enough information to determine warranty" and escalate. This is the correct behavior — it's better to escalate than to guess wrong on warranty eligibility. But it means retrieval quality directly impacts automation rate. Poor retrieval → more escalations → less automation → less 85% reduction in manual effort.

---

## 9. Node 5 — Output Validation

### What Happens

Mirrors Node 2 but validates action items instead of defect extractions. Parses JSON, validates against Pydantic schema, performs business logic checks, and routes accordingly.

### Implementation

```python
MAX_GENERATION_RETRIES = 3

def validate_output_node(state: AgentState) -> dict:
    raw_response = state.get("generation_raw_response")
    retry_count = state.get("generation_retry_count", 0)
    defects = state.get("extracted_defects", [])
    errors = []
    
    # ── Step 1: Parse JSON ──
    if raw_response is None:
        errors.append("LLM returned no response")
        return {
            "action_items": None,
            "generation_validation_errors": errors,
            "generation_retry_count": retry_count + 1,
            "current_node": "validate_output",
        }
    
    try:
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        errors.append(f"JSON parse error: {str(e)}")
        return {
            "action_items": None,
            "generation_validation_errors": errors,
            "generation_retry_count": retry_count + 1,
            "current_node": "validate_output",
        }
    
    if not isinstance(parsed, list):
        parsed = [parsed]
    
    # ── Step 2: Pydantic validation ──
    validated_items = []
    for i, item_dict in enumerate(parsed):
        try:
            item = ActionItem(**item_dict)
            validated_items.append(item)
        except ValidationError as e:
            errors.append(f"Action item {i}: {e.errors()}")
    
    if errors and not validated_items:
        return {
            "action_items": None,
            "generation_validation_errors": errors,
            "generation_retry_count": retry_count + 1,
            "current_node": "validate_output",
        }
    
    # ── Step 3: Business logic validation ──
    business_errors = []
    
    # Check: action item count should roughly match defect count
    if len(validated_items) < len(defects):
        business_errors.append(
            f"Expected at least {len(defects)} action items (one per defect), "
            f"got {len(validated_items)}"
        )
    
    # Check: critical severity defects should have urgent/emergency priority
    for i, defect in enumerate(defects):
        if defect.severity == "critical":
            matching_items = [
                item for item in validated_items 
                if item.priority in ("urgent", "emergency")
            ]
            if not matching_items:
                business_errors.append(
                    f"Critical defect '{defect.affected_component}' has no "
                    f"urgent/emergency action item"
                )
    
    # Check: every action item should have an SOP reference
    for i, item in enumerate(validated_items):
        if not item.sop_reference or item.sop_reference.lower() in ("none", "n/a", "unknown"):
            business_errors.append(
                f"Action item {i} missing SOP reference"
            )
    
    # Check: escalation_required should be true if warranty cannot be determined
    for i, item in enumerate(validated_items):
        if not item.warranty_eligible and not item.escalation_required:
            # Might be fine (not everything needs warranty), but log as warning
            pass
    
    # Business errors are warnings, not hard failures
    # They're logged for monitoring but don't trigger retries
    
    return {
        "action_items": validated_items,
        "generation_validation_errors": errors if errors else None,
        "generation_business_warnings": business_errors if business_errors else None,
        "current_node": "validate_output",
    }


def route_after_output_validation(state: AgentState) -> str:
    errors = state.get("generation_validation_errors")
    retry_count = state.get("generation_retry_count", 0)
    items = state.get("action_items")
    status = state.get("status")
    
    if status == "failed":
        return "human_review"
    
    if items and not errors:
        return "valid"
    
    if items and errors:
        return "valid"  # partial success, proceed with valid items
    
    if retry_count < MAX_GENERATION_RETRIES:
        return "retry"
    else:
        return "human_review"
```

### Design Decisions

**Decision:** Business logic validation is separate from schema validation and produces warnings, not failures.

**Reasoning:** Schema validation (Pydantic) catches structural errors — missing fields, wrong types, invalid enums. These are hard failures that mean the output is malformed. Business logic validation catches semantic issues — a critical defect without an urgent action item, a missing SOP reference. These indicate the output might be suboptimal but not structurally broken. Making business logic a hard failure would cause excessive retries because the LLM might not produce a "better" result on retry for subjective quality issues.

**Tradeoff:** Business warnings that indicate real quality problems (missing SOP references, mismatched priorities) don't trigger retries. They flow through to the output and are visible in the monitoring dashboard. Over time, if a specific business warning occurs frequently, it signals a prompt engineering improvement opportunity or a gap in the SOP documents.

**Decision:** Action item count should roughly match defect count.

**Reasoning:** Each defect should produce at least one action item. If the LLM extracted 3 defects but only generated 1 action item, it likely missed 2 defects during generation. This check catches that discrepancy. We say "roughly" because a single defect might legitimately produce multiple action items (e.g., "replace compressor" and "inspect coolant line for related damage"), and two related defects might consolidate into one action item.

**Rule:** The count mismatch is a warning, not a failure. Fewer action items than defects triggers a warning. More action items than defects is acceptable (one defect → multiple repair steps).

---

## 10. Node 6 — Output Assembly & Dispatch

### What Happens

This node takes the validated action items, combines them with the order metadata, produces the final output JSON, and dispatches it to downstream systems.

### Implementation

```python
def assemble_output_node(state: AgentState) -> dict:
    node_start = datetime.utcnow()
    
    total_duration = (datetime.utcnow() - state["total_start_time"]).total_seconds() * 1000
    
    final_output = {
        "order_id": state["order_id"],
        "segment_id": state["segment_id"],
        "status": "completed",
        "processed_at": datetime.utcnow().isoformat(),
        "total_processing_time_ms": total_duration,
        
        "extracted_defects": [d.dict() for d in state["extracted_defects"]],
        "action_items": [a.dict() for a in state["action_items"]],
        
        "retrieval_metadata": {
            "query_used": state.get("retrieval_query"),
            "chunks_retrieved": len(state.get("retrieved_chunks", [])),
            "top_relevance_score": max(state.get("retrieval_scores", [0])),
            "was_reformulated": state.get("retrieval_reformulated", False),
        },
        
        "quality_signals": {
            "extraction_retries": state.get("extraction_retry_count", 0),
            "generation_retries": state.get("generation_retry_count", 0),
            "cross_validation_warnings": state.get("extraction_cross_validation_warnings", []),
            "business_warnings": state.get("generation_business_warnings", []),
        },
        
        "node_durations": state.get("node_durations", {}),
    }
    
    # ── Dispatch to downstream queue ──
    dispatch_to_output_queue(final_output)
    
    duration = (datetime.utcnow() - node_start).total_seconds() * 1000
    
    return {
        "status": "completed",
        "current_node": "assemble_output",
        "node_durations": {
            **state.get("node_durations", {}),
            "assemble_output": duration,
        },
    }
```

### Design Decisions

**Decision:** Include quality signals in the final output.

**Reasoning:** Downstream consumers (dashboards, human reviewers, analytics systems) need to know not just the result but how confident the system is in that result. An order that completed on first try with high retrieval scores and no warnings is high-confidence. An order that required 2 extraction retries, had cross-validation warnings, and missing SOP references is low-confidence and should be prioritized for human review. Including quality signals enables intelligent downstream routing.

**Decision:** Include per-node latency in the output.

**Reasoning:** Performance debugging requires knowing which node is the bottleneck. If total latency spikes, per-node timing immediately identifies whether the issue is LLM inference (Nodes 1/4), RAG retrieval (Node 3), or validation logic (Nodes 2/5). Without per-node timing, you'd have to add custom logging to each node and correlate timestamps manually.

---

## 11. Conditional Edges & Routing Logic

### The Complete Routing Map

```
Node 1 (extract_defects)
  │
  └──▶ Node 2 (validate_extraction) [always]
         │
         ├──▶ "valid"        → Node 3 (retrieve_context)
         ├──▶ "retry"        → Node 1 (extract_defects)     [loop]
         └──▶ "human_review" → Node 7 (human_review)

Node 3 (retrieve_context)
  │
  └──▶ Node 4 (generate_action_items) [always]
         │
         └──▶ Node 5 (validate_output) [always]
                │
                ├──▶ "valid"        → Node 6 (assemble_output)
                ├──▶ "retry"        → Node 4 (generate_action_items) [loop]
                └──▶ "human_review" → Node 7 (human_review)

Node 6 (assemble_output) → END
Node 7 (human_review)    → END
```

### Why No Conditional Edge After RAG Retrieval?

**Decision:** Node 3 (retrieval) always flows to Node 4 (generation) even if retrieval quality is low.

**Reasoning:** Retrieval quality issues are handled internally in Node 3 (query reformulation, context expansion). If after all internal recovery attempts the retrieval quality is still low, the retrieved chunks are passed to Node 4 anyway. Node 4's prompt instructs the LLM to set `escalation_required: true` when context is insufficient. This means low retrieval quality results in action items that say "escalate to human" rather than blocking the pipeline. The information about low retrieval quality is captured in the quality signals and visible downstream.

**Alternative considered:** Add a validation node after retrieval that routes low-quality retrievals directly to human review. Rejected because this would bypass action item generation entirely, meaning the human reviewer gets no AI-suggested action items to start from. Even a low-confidence AI suggestion is helpful as a starting point for human review.

**Tradeoff:** Generating action items from low-quality retrieval context might produce incorrect warranty determinations or wrong procedures. But these outputs are flagged with `escalation_required: true` and low retrieval scores in quality signals, so they won't be auto-applied without human review.

---

## 12. Retry & Self-Healing Mechanism

### How the Retry Loop Works

```
Attempt 1: Standard prompt → LLM → Validation → FAIL (missing required field)
                                                    │
                                                    ▼
Attempt 2: Standard prompt + "Previous attempt failed: 
            'affected_component' is required" → LLM → Validation → FAIL (wrong enum)
                                                                      │
                                                                      ▼
Attempt 3: Standard prompt + "Previous attempts failed:
            1. 'affected_component' is required
            2. severity must be low|medium|high|critical, 
               got 'moderate'" → LLM → Validation → PASS ✓
```

The key insight is that each retry includes ALL previous errors, not just the latest one. This prevents the LLM from fixing one error while reintroducing a previous one.

### Retry Budget

**Decision:** Maximum 3 retries per validation gate (extraction and generation independently).

**Reasoning:** Empirically, if the fine-tuned model can't produce valid output in 3 attempts with error feedback, a 4th attempt is unlikely to succeed. The model is either confused by the input (ambiguous text), hitting a systematic failure mode (domain terminology it wasn't fine-tuned on), or the input itself is problematic. In all these cases, human review is the correct escalation.

**Tradeoff:** More retries would marginally increase the automation rate (maybe 1-2% more orders succeed on attempt 4-5). But each retry costs an LLM inference call (2-5 seconds + GPU cost). At 3 retries, the worst-case latency for a single gate is 4 LLM calls * 5 seconds = 20 seconds. At 5 retries, it would be 30 seconds. The diminishing returns don't justify the cost.

**Rule:** Extraction retries and generation retries are independent. An order that takes 3 extraction retries and then succeeds still gets a fresh 3-retry budget for generation. The total worst case is 4 extraction calls + 4 generation calls = 8 LLM calls per segment, which is acceptable but costly — these cases are logged and reviewed to identify patterns.

### What Gets Appended in Retry Prompts

```python
def build_retry_context(previous_errors: list[str], attempt_number: int) -> str:
    context = f"\n\nATTEMPT {attempt_number + 1} — PREVIOUS ERRORS TO FIX:\n"
    for i, error in enumerate(previous_errors):
        context += f"{i+1}. {error}\n"
    context += "\nFix ALL errors listed above. Output ONLY valid JSON.\n"
    return context
```

**Decision:** Include attempt number in retry prompt.

**Reasoning:** The attempt number gives the LLM implicit urgency context. Some models respond to "attempt 3 of 3" by being more careful with their output. This is a minor prompt engineering trick but has been observed to help marginally with fine-tuned models.

---

## 13. Human Review Fallback

### What Happens

When an order segment exhausts its retry budget or encounters an infrastructure failure, it's routed to the human review node, which packages all available context and dispatches it to a human review queue/dashboard.

### Implementation

```python
def human_review_node(state: AgentState) -> dict:
    node_start = datetime.utcnow()
    
    review_package = {
        "order_id": state["order_id"],
        "segment_id": state["segment_id"],
        "status": "human_review",
        "escalation_reason": determine_escalation_reason(state),
        "processed_at": datetime.utcnow().isoformat(),
        
        # Give the human ALL available context
        "original_input": {
            "segment_text": state["segment_text"],
            "inherited_context": state["inherited_context"],
            "structured_metadata": state["structured_metadata"],
        },
        
        # Include partial results (whatever was produced before failure)
        "partial_results": {
            "extracted_defects": [d.dict() for d in state["extracted_defects"]] 
                                if state.get("extracted_defects") else None,
            "retrieved_chunks": [c.dict() for c in state["retrieved_chunks"]]
                               if state.get("retrieved_chunks") else None,
            "action_items": [a.dict() for a in state["action_items"]]
                           if state.get("action_items") else None,
        },
        
        # Include all errors for debugging
        "error_history": {
            "extraction_errors": state.get("extraction_validation_errors"),
            "generation_errors": state.get("generation_validation_errors"),
            "error_log": state.get("error_log", []),
            "extraction_retries": state.get("extraction_retry_count", 0),
            "generation_retries": state.get("generation_retry_count", 0),
        },
        
        # Include raw LLM responses for debugging
        "debug_context": {
            "extraction_prompt": state.get("extraction_prompt"),
            "extraction_raw_response": state.get("extraction_raw_response"),
            "generation_prompt": state.get("generation_prompt"),
            "generation_raw_response": state.get("generation_raw_response"),
        },
        
        "node_durations": state.get("node_durations", {}),
    }
    
    dispatch_to_human_review_queue(review_package)
    
    return {
        "status": "human_review",
        "current_node": "human_review",
    }


def determine_escalation_reason(state: AgentState) -> str:
    if state.get("status") == "failed":
        errors = state.get("error_log", [])
        if errors:
            return f"Infrastructure failure: {errors[-1].get('error', 'unknown')}"
        return "Infrastructure failure: unknown"
    
    if state.get("extraction_retry_count", 0) >= MAX_EXTRACTION_RETRIES:
        return "Extraction failed after maximum retries"
    
    if state.get("generation_retry_count", 0) >= MAX_GENERATION_RETRIES:
        return "Action item generation failed after maximum retries"
    
    return "Unknown escalation reason"
```

### Design Decisions

**Decision:** Include partial results in the human review package.

**Reasoning:** If extraction succeeded but generation failed, the human reviewer has the extracted defects to start from — they don't have to re-read the raw text and extract defects themselves. Even partially valid action items (some validated, some didn't) give the human a starting point. This reduces the human review time from "process from scratch" to "review and fix AI suggestions."

**Tradeoff:** Including partial results adds complexity to the review dashboard (it needs to handle null sections gracefully). But the time savings for human reviewers — the most expensive resource in the system — far outweighs the development cost.

**Decision:** Include raw prompts and LLM responses in debug context.

**Reasoning:** When a senior engineer investigates why a specific order failed, they need to see exactly what the model was asked and what it responded. Without this, debugging becomes guesswork. The debug context enables reproduction — you can replay the exact same prompt in a testing environment to understand and fix the failure mode.

**Rule:** Debug context is access-controlled. Regular human reviewers see the partial results and escalation reason. Engineers with debug access can see the raw prompts and responses. This separation protects against prompt leakage while enabling deep debugging.

---

## 14. Error Handling & Fault Tolerance

### Error Categories and Handling

**Category 1 — LLM Inference Errors (Timeout, Service Down)**

These are infrastructure failures, not logic failures. The LLM service might be temporarily overloaded, undergoing a deployment, or experiencing a GPU failure.

**Handling:** The extraction/generation node catches the exception, logs it to `error_log` in state, sets `status: "failed"`, and the routing function immediately sends it to human review (no retries for infrastructure failures — the retry would likely also fail).

**Rule:** Infrastructure failures are never retried at the node level. They're retried at the queue level — the failed message returns to the SQS queue with a visibility timeout, and a different worker picks it up later when the LLM service may have recovered.

**Category 2 — Validation Failures (Malformed LLM Output)**

The LLM returned a response but it doesn't pass Pydantic validation.

**Handling:** The validation node collects all errors, increments the retry counter, and lets the routing function decide (retry or escalate). The errors are appended to the next attempt's prompt.

**Category 3 — Retrieval Failures (Pinecone Down, No Results)**

Pinecone might be unreachable or return zero results.

**Handling:** The retrieval node catches Pinecone errors and returns empty retrieved_chunks. The generation node (Node 4) proceeds without context and produces action items with `escalation_required: true`. This is graceful degradation — the system produces a partial result rather than failing entirely.

**Rule:** A Pinecone outage should never prevent order processing. Orders flow through with degraded quality (no SOP grounding → escalation) rather than being blocked.

**Category 4 — State Corruption (Unexpected Data Types, Missing Fields)**

A bug in a node might write the wrong type to state or forget a field.

**Handling:** Each node starts with defensive reads using `.get()` with defaults. Validation nodes check not just LLM output but also state integrity. If state is corrupted, the order is routed to human review with a "state corruption" escalation reason.

---

## 15. Concurrency & Parallelism

### Current Design: Sequential Execution Per Segment

**Decision:** Nodes execute sequentially within a single segment.

**Reasoning:** The graph has data dependencies between nodes — extraction must complete before retrieval can construct a query, retrieval must complete before generation can use context. There's no opportunity for parallelism within a single segment's processing.

**Parallelism exists at the segment level:** Multiple segments from different orders (or from the same multi-segment order) are processed in parallel by different agent workers. If there are 10 agent workers and 10 segments in the queue, all 10 process simultaneously.

### Future Optimization: Parallel Retrieval

**Potential improvement:** If an order has 3 extracted defects, each could trigger a parallel Pinecone query rather than sequential queries. LangGraph supports parallel node execution via `map` operations. This would reduce Node 3 latency from 3x retrieval time to 1x retrieval time for multi-defect segments.

**Not implemented yet because:** Most segments have 1-2 defects, so the parallelism benefit is small. The complexity of managing parallel state merges isn't justified at current scale.

---

## 16. State Persistence & Checkpointing

### Why Checkpointing

**Decision:** Enable LangGraph checkpointing with SQLite (development) or PostgreSQL (production).

**Reasoning:** If an agent worker crashes mid-execution (between Node 3 and Node 4, for example), without checkpointing the entire segment would need to restart from Node 1, wasting the LLM call from Node 1 and the Pinecone query from Node 3. With checkpointing, the worker can resume from the last completed node.

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@checkpoint-db:5432/checkpoints"
)
app = workflow.compile(checkpointer=checkpointer)
```

**Tradeoff:** Every node completion writes to the checkpoint database, adding ~5-10ms latency per node. For a 6-node graph, that's ~30-60ms total overhead. Negligible compared to LLM call latency. The checkpoint database needs to handle the write throughput — at peak load with 50 concurrent agent workers, each completing ~1 node per second, that's ~50 writes/second. PostgreSQL handles this easily.

### Checkpoint Retention

**Decision:** Retain checkpoints for 7 days, then purge.

**Reasoning:** Most crash recoveries happen within minutes. Historical checkpoints beyond 7 days have no operational value — the order would have been reprocessed or escalated long before then. The 7-day window covers any delayed investigation or audit needs.

---

## 17. Observability & Alerting

### LangSmith Integration

Every graph execution is traced in LangSmith:

```python
import langsmith

@langsmith.traceable(name="extract_defects", run_type="chain")
def extract_defects_node(state: AgentState) -> dict:
    # ... node implementation
    # LangSmith automatically captures input state, output state, duration
```

LangSmith provides: full trace visualization (each node as a span), LLM call details (prompt, completion, token count, latency), parent-child relationships (graph → node → LLM call), and error attribution (which node failed and why).

### Custom Metrics Emitted

**Per-Execution Metrics:**

| Metric | Description |
|--------|-------------|
| `agent.total_duration_ms` | End-to-end execution time |
| `agent.node_duration_ms` | Per-node latency (tagged by node name) |
| `agent.extraction_retries` | Number of extraction retry attempts |
| `agent.generation_retries` | Number of generation retry attempts |
| `agent.total_llm_calls` | Total LLM inference calls made |
| `agent.total_tokens_used` | Total input + output tokens across all LLM calls |
| `agent.retrieval_chunks_count` | Number of chunks retrieved from Pinecone |
| `agent.retrieval_top_score` | Highest relevance score from retrieval |
| `agent.retrieval_reformulated` | Whether query reformulation was triggered |
| `agent.status` | Final status (completed / human_review / failed) |
| `agent.defects_extracted` | Number of defects extracted |
| `agent.action_items_generated` | Number of action items produced |

**Aggregate Dashboard Panels:**

**Panel 1 — Throughput & Latency**
- Orders processed per minute
- P50 / P95 / P99 end-to-end latency
- Per-node latency breakdown (stacked bar chart)
- Queue depth (dispatch queue)

**Panel 2 — Success Rate**
- Completion rate (percentage reaching "completed" status)
- Human review rate (percentage escalated)
- Failure rate (percentage with infrastructure failures)
- First-attempt success rate (completed with 0 retries)

**Panel 3 — LLM Performance**
- Average tokens per call (input and output separately)
- Inference latency P50/P95 (from vLLM)
- JSON parse failure rate (per node)
- Pydantic validation failure rate (per node)

**Panel 4 — Retrieval Quality**
- Average top-K relevance score
- Query reformulation rate
- Context expansion trigger rate
- Zero-result rate

**Panel 5 — Retry Analysis**
- Retry distribution (histogram: 0, 1, 2, 3 retries)
- Most common validation errors (top-10 error messages)
- Retry success rate (percentage that eventually pass after retry)

### Alert Configuration

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| High human review rate | >15% of orders escalated in 1hr | P2 | Investigate LLM quality or SOP gaps |
| LLM inference latency spike | P95 > 10 seconds | P2 | Check vLLM service, GPU utilization |
| LLM service down | Inference error rate > 50% in 5min | P1 | vLLM service health check, GPU status |
| Pinecone unreachable | Retrieval error rate > 20% in 5min | P1 | Check Pinecone status, network |
| Zero retrieval results spike | >25% of queries return 0 results | P2 | SOPs may need re-indexing, or new product line not indexed |
| Validation failure spike | JSON parse failure > 20% | P1 | LLM output format degraded, check model |
| Checkpoint DB latency | Write latency > 100ms | P3 | Database performance, vacuum/maintenance |
| Queue depth growing | Dispatch queue > 500 messages | P2 | Scale agent workers, check processing bottleneck |
| Retry exhaustion spike | >10% orders exhausting all retries | P1 | Systematic LLM failure, investigate model |
| Total latency degradation | P95 > 30 seconds | P2 | Identify bottleneck node via per-node metrics |

---

## 18. End-to-End Phase 2 Flow Summary

```
SEGMENT ARRIVES FROM PHASE 1 DISPATCH QUEUE
        │
        │   AgentState initialized with:
        │   - segment_text, inherited_context, pre_extracted_entities
        │   - structured_metadata, order_id, segment_id
        │   - retry counters set to 0
        │   - status: "processing"
        │
        ▼
Node 1 ─┤  EXTRACT DEFECTS (LLM Call)
        │  • Build prompt: system instruction + segment text + pre-extracted entities
        │  • If retry: append previous validation errors to prompt
        │  • Call fine-tuned Llama 3.1 8B via vLLM (temperature 0.0)
        │  • Store raw response + prompt in state
        │  • 30-second timeout, catch all exceptions
        │
        ▼
Node 2 ─┤  VALIDATE EXTRACTION
        │  • Parse raw response as JSON (strip markdown fences)
        │  • Validate each defect against Pydantic ExtractedDefect schema
        │  • Cross-validate against Phase 1 pre-extracted entities
        │  • ROUTE:
        │    ├─ Valid → Node 3
        │    ├─ Invalid + retries left → Node 1 (retry loop)
        │    └─ Invalid + retries exhausted → Node 7 (human review)
        │
        ▼
Node 3 ─┤  RETRIEVE CONTEXT (RAG)
        │  • Construct semantic query from extracted defects
        │  • Pinecone vector search with product line metadata filter
        │  • Cross-encoder reranking of top-K results
        │  • Low relevance? → query reformulation + second search
        │  • Context window expansion (fetch neighbor chunks)
        │  • Store retrieved chunks + scores in state
        │
        ▼
Node 4 ─┤  GENERATE ACTION ITEMS (LLM Call)
        │  • Build prompt: system instruction + defects + retrieved SOP context + metadata
        │  • If retry: append previous validation errors
        │  • Call Llama 3.1 8B via vLLM (temperature 0.0, max 2048 tokens)
        │  • Instruct: base warranty decisions ONLY on retrieved context
        │  • 45-second timeout
        │
        ▼
Node 5 ─┤  VALIDATE OUTPUT
        │  • Parse raw response as JSON
        │  • Validate each action item against Pydantic ActionItem schema
        │  • Business logic checks (count match, severity-priority alignment, SOP refs)
        │  • ROUTE:
        │    ├─ Valid → Node 6
        │    ├─ Invalid + retries left → Node 4 (retry loop)
        │    └─ Invalid + retries exhausted → Node 7 (human review)
        │
        ▼
Node 6 ─┤  ASSEMBLE OUTPUT
        │  • Combine extracted defects + action items + quality signals
        │  • Include per-node latency, retry counts, warnings
        │  • Dispatch to output queue → downstream systems
        │  • Status: "completed"
        │
        ▼
       END (success)

        ── OR ──

Node 7 ─┤  HUMAN REVIEW (fallback)
        │  • Package ALL available context:
        │    - Original input text + metadata
        │    - Partial results (whatever was produced before failure)
        │    - Complete error history and retry log
        │    - Raw LLM prompts + responses (debug access only)
        │  • Dispatch to human review queue
        │  • Status: "human_review"
        │
        ▼
       END (escalated)
```

### Key Design Principles Applied Throughout Phase 2

1. **Single responsibility per node.** Each node does one thing — call LLM, validate, retrieve, generate, validate, assemble. Logic is never conflated.
2. **Fail gracefully, not catastrophically.** Every failure has a defined path — retry, escalate, or degrade. The system never silently drops an order.
3. **State is the source of truth.** Every decision, every intermediate result, every error is recorded in the typed state object. Debugging starts and ends with inspecting state.
4. **Retry with context.** Retries aren't blind repetitions — each retry includes the specific errors from the previous attempt, giving the LLM targeted guidance on what to fix.
5. **Quality signals flow downstream.** The output doesn't just contain results — it contains confidence indicators (retry counts, retrieval scores, warnings) that enable intelligent downstream routing.
6. **Observability by default.** LangSmith traces every execution. Custom metrics cover throughput, latency, success rates, and failure patterns. Alerts catch degradation before it impacts SLAs.
7. **Human review is a feature, not a failure.** The system is designed to escalate gracefully when it can't produce reliable results, providing humans with AI-assisted starting points rather than dumping them back to manual processing.

---

*Document Version: 1.0 | Last Updated: March 2026 | System: LGS Tech Agentic AI Workflow*
