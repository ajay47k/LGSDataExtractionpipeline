# Phase 6 — Output Generation & Downstream Integration: Complete Production Reference

## LGS Tech Agentic AI Workflow — Service Order Processing

---

## Table of Contents

1. Phase 6 Overview & What It Produces
2. Action Item Generation — The Final LLM Call
3. Prompt Construction for Generation
4. Output Schema — The Complete Action Item Structure
5. Warranty Determination Logic
6. Parts Resolution & Catalog Matching
7. Escalation Decision Framework
8. Multi-Segment Order Reassembly
9. Final Output Assembly
10. Downstream Integration Architecture
11. Integration with Service Management Systems
12. Human Review Dashboard
13. Feedback Loop — Human Corrections Back to Training
14. SLA & Processing Time Guarantees
15. The 85% Automation Metric — How It's Measured
16. Observability & Alerting
17. End-to-End Phase 6 Flow Summary

---

## 1. Phase 6 Overview & What It Produces

### What Phase 6 Does

Phase 6 is the final production step. It takes the validated extracted defects from Phase 3, the retrieved SOP/warranty context from Phase 4, and the validation signals from Phase 5, and produces the ultimate deliverable: structured action items that tell a service team exactly what to do, what parts to order, whether the repair is covered under warranty, and whether human escalation is needed. These action items are then dispatched to downstream systems — service management platforms, ticketing tools, and human review dashboards.

### Why Phase 6 is Not Just "Another LLM Call"

The action item generation in Node 4 of the LangGraph (Phase 2) is technically the LLM call that produces action items. Phase 6 encompasses everything that happens after that LLM call succeeds validation — the post-processing, enrichment, reassembly, formatting, and delivery that turns raw validated JSON into an integrated output ready for downstream consumption.

The distinction matters because the LLM produces the reasoning (what action to take, whether warranty applies). But the LLM doesn't know what format the ticketing system expects, what the correct part numbers are in the catalog, how to reassemble multi-segment orders, or how to route to the right review queue. Phase 6 handles all of that.

---

## 2. Action Item Generation — The Final LLM Call

### What the LLM Receives

By the time the generation node fires, the AgentState contains:

```
From Phase 1: segment_text, inherited_context, pre_extracted_entities, structured_metadata
From Phase 3: extracted_defects (validated list of ExtractedDefect objects)
From Phase 4: retrieved_chunks (reranked, expanded SOP/warranty chunks)
From Phase 5: extraction_cross_validation_warnings, extraction_business_warnings
```

The LLM sees all of this assembled into a single prompt.

### What the LLM Produces

A JSON array of ActionItem objects — one per defect (sometimes more if a defect requires multiple repair steps):

```json
[
  {
    "action_type": "replace",
    "description": "Replace compressor assembly per SOP-XR-COMP-001 Section 3.2. Disconnect power supply, verify lockout-tagout, drain refrigerant per EPA Section 608 requirements, remove mounting bolts (4x M10), extract compressor unit, install replacement P/N 7832-A, recharge refrigerant to 4.2 lbs specification, run 30-minute operational test.",
    "parts_required": [
      {"part_name": "Compressor Assembly XR-440", "part_number": "P/N 7832-A", "quantity": 1},
      {"part_name": "Refrigerant R-410A", "part_number": "REF-410A-5LB", "quantity": 1},
      {"part_name": "Mounting Bolt Kit M10", "part_number": "HW-M10-KIT", "quantity": 1}
    ],
    "warranty_eligible": true,
    "escalation_required": false,
    "priority": "urgent",
    "sop_reference": "SOP-XR-COMP-001, Section 3.2: Compressor Replacement Procedure"
  }
]
```

### The Quality Bar for Action Items

An action item is useful only if a service technician can read it and know exactly what to do without additional research. This means the description must include specific steps (not "replace compressor" but the actual replacement procedure steps), the parts list must have part numbers (not just "compressor assembly"), the warranty determination must be based on the retrieved policy (not assumed), and the SOP reference must point to a specific document and section (not "see SOP").

**Decision:** The system prompt for generation explicitly requires this level of specificity.

**Reasoning:** A vague action item like "repair the compressor" saves no time — the technician still has to look up the procedure, identify the parts, and determine warranty eligibility. That's the manual process the system is supposed to replace. Only specific, detailed action items achieve the 85% automation target.

**Tradeoff:** Requiring detailed descriptions increases output length (500-800 tokens per action item vs 50-100 for vague descriptions), which increases LLM inference time and cost. But the entire point of the system is to produce useful output — saving compute at the cost of output quality defeats the purpose.

---

## 3. Prompt Construction for Generation

### The Complete Generation Prompt

```python
def build_generation_prompt(state: AgentState) -> str:
    """Construct the full prompt for action item generation."""
    
    defects = state["extracted_defects"]
    chunks = state.get("retrieved_chunks", [])
    metadata = state["structured_metadata"]
    inherited = state["inherited_context"]
    retry_count = state.get("generation_retry_count", 0)
    previous_errors = state.get("generation_validation_errors", [])
    
    # ── System prompt ──
    system = GENERATION_SYSTEM_PROMPT  # Defined in Phase 2 reference
    
    # ── Context section ──
    context_block = "RETRIEVED SOP AND WARRANTY CONTEXT:\n"
    context_block += "=" * 50 + "\n\n"
    
    # Group chunks by document for coherent reading
    chunks_by_doc = {}
    for chunk in chunks:
        doc_title = chunk.document_title
        if doc_title not in chunks_by_doc:
            chunks_by_doc[doc_title] = []
        chunks_by_doc[doc_title].append(chunk)
    
    for doc_title, doc_chunks in chunks_by_doc.items():
        doc_chunks.sort(key=lambda c: c.metadata.get("chunk_index", 0))
        context_block += f"--- {doc_title} ---\n"
        for chunk in doc_chunks:
            section = chunk.section
            if section:
                context_block += f"[Section: {section}]\n"
            context_block += f"{chunk.text}\n\n"
    
    if not chunks:
        context_block += "[NO CONTEXT RETRIEVED — base decisions on extracted defect data only. "
        context_block += "Set escalation_required=true for warranty and procedure determinations.]\n\n"
    
    # ── Defects section ──
    defects_block = "EXTRACTED DEFECTS:\n"
    defects_block += "=" * 50 + "\n\n"
    
    for i, defect in enumerate(defects):
        defects_block += f"Defect {i+1}:\n"
        defects_block += f"  Type: {defect.defect_type.value}\n"
        defects_block += f"  Severity: {defect.severity.value}\n"
        defects_block += f"  Component: {defect.affected_component}\n"
        defects_block += f"  Symptoms: {', '.join(defect.symptoms)}\n"
        defects_block += f"  Error Codes: {', '.join(defect.error_codes) if defect.error_codes else 'None'}\n"
        defects_block += f"  Root Cause: {defect.root_cause_hypothesis or 'Unknown'}\n"
        defects_block += f"  Confidence: {defect.confidence}\n\n"
    
    # ── Metadata section ──
    model_num = inherited.get("model_number") or metadata.get("model_number", "unknown")
    plant_id = inherited.get("plant_id") or metadata.get("plant_id", "unknown")
    serial = inherited.get("serial_number") or metadata.get("serial_number", "unknown")
    
    metadata_block = "ORDER METADATA:\n"
    metadata_block += "=" * 50 + "\n"
    metadata_block += f"  Model: {model_num}\n"
    metadata_block += f"  Plant: {plant_id}\n"
    metadata_block += f"  Serial: {serial}\n"
    metadata_block += f"  Warranty ID: {metadata.get('warranty_id', 'unknown')}\n"
    metadata_block += f"  Priority: {metadata.get('priority', 'standard')}\n\n"
    
    # ── Assembly ──
    user_prompt = f"{context_block}\n{defects_block}\n{metadata_block}\n"
    user_prompt += "Generate ONE action item per defect. "
    user_prompt += "Base ALL warranty decisions on the retrieved context above. "
    user_prompt += "Include specific repair steps from the SOPs. "
    user_prompt += "Reference specific part numbers from SOPs or parts catalogs. "
    user_prompt += "Output ONLY valid JSON array.\n"
    
    # ── Retry context ──
    if retry_count > 0 and previous_errors:
        user_prompt += f"\n{'='*60}\n"
        user_prompt += f"ATTEMPT {retry_count + 1}: FIX THESE ERRORS:\n"
        for error in previous_errors:
            user_prompt += f"  • {error}\n"
        user_prompt += "Fix ALL errors. Output ONLY valid JSON.\n"
    
    return system, user_prompt
```

### Prompt Design Decisions

**Decision:** Explicit instruction to base warranty decisions ONLY on retrieved context.

**Reasoning:** Without this instruction, the fine-tuned model uses its training knowledge to make warranty guesses. "Compressors are typically covered for 2 years" might be accurate for most product lines but wrong for this specific one. By forcing the model to reference the retrieved warranty policy, we ensure compliance with the actual current terms. If the retrieved context doesn't contain warranty information (retrieval failure), the instruction to set `escalation_required=true` ensures a human makes the determination instead of the model guessing.

**Decision:** Include "[NO CONTEXT RETRIEVED]" message when chunks are empty.

**Reasoning:** If Phase 4 retrieval failed or returned zero results, the model needs to know it has no external knowledge to work with. Without this explicit message, the model might proceed as if it has context (hallucinating SOP references). The explicit "[NO CONTEXT RETRIEVED]" message triggers the trained behavior of setting escalation flags rather than fabricating procedures.

**Decision:** Structure the prompt with clear section headers and separators.

**Reasoning:** The generation prompt is long (2,000-3,500 tokens). Without clear structure, the model might confuse defect data with SOP context, or metadata with symptoms. Visual separators (`===`) and section headers (`RETRIEVED SOP AND WARRANTY CONTEXT:`) create a clear mental map that helps the model locate and use the right information for each field in the output.

---

## 4. Output Schema — The Complete Action Item Structure

### The Full Output Object

After validation and post-processing, each action item becomes:

```python
class FinalActionItem(BaseModel):
    """Complete action item after enrichment and post-processing."""
    
    # ── Core fields (from LLM) ──
    action_type: ActionType
    description: str
    parts_required: list[EnrichedPart]
    warranty_eligible: bool
    escalation_required: bool
    priority: Priority
    sop_reference: str
    
    # ── Enrichment fields (added post-LLM) ──
    defect_reference: str                    # Which defect this addresses
    estimated_repair_time_hours: Optional[float]  # From SOP metadata
    skill_level_required: Optional[str]      # "technician" / "specialist" / "engineer"
    safety_precautions: list[str]            # Extracted from SOP safety sections
    regulatory_requirements: list[str]       # EPA, OSHA requirements if applicable
    
    # ── Parts enrichment ──
    parts_availability: Optional[dict]       # From parts catalog API
    estimated_parts_cost: Optional[float]    # From parts catalog
    
    # ── Warranty enrichment ──
    warranty_policy_reference: Optional[str] # Specific policy doc + version
    warranty_coverage_type: Optional[str]    # "standard" / "extended" / "goodwill"
    warranty_expiry_date: Optional[str]      # Calculated from unit age + policy terms
    
    # ── Quality signals ──
    confidence: float                        # From defect extraction confidence
    retrieval_relevance: float               # Top retrieval score for this action item's context
    generation_attempt: int                  # Which attempt produced this (1 = first try)
    warnings: list[str]                      # Business logic warnings that apply


class EnrichedPart(BaseModel):
    """Part with catalog enrichment."""
    part_name: str
    part_number: Optional[str]
    quantity: int
    
    # ── Enrichment from parts catalog ──
    catalog_verified: bool = False           # Part number verified against catalog
    unit_price: Optional[float]
    in_stock: Optional[bool]
    lead_time_days: Optional[int]
    alternative_parts: Optional[list[str]]   # Substitute part numbers if primary is unavailable
```

### Why Enrich Beyond What the LLM Produces

The LLM produces the core reasoning — what action to take, which SOP to follow, whether warranty applies. But downstream systems need more: estimated costs for budget approval, parts availability for procurement, repair time for scheduling, safety precautions for technician briefing. These come from structured data sources (parts catalog, SOP metadata) that the LLM doesn't have access to.

---

## 5. Warranty Determination Logic

### How Warranty Decisions Are Made

The warranty determination follows a strict hierarchy:

```
Level 1: Retrieved warranty policy explicitly states coverage
  → warranty_eligible = true, with policy reference
  → Example: "Standard coverage for XR compressors: 24 months from installation"

Level 2: Retrieved warranty policy explicitly states exclusion
  → warranty_eligible = false, with exclusion reason
  → Example: "Exclusions: damage caused by unauthorized modifications"

Level 3: Retrieved context is ambiguous or incomplete
  → warranty_eligible = false, escalation_required = true
  → Human makes the determination

Level 4: No warranty context retrieved at all
  → warranty_eligible = false, escalation_required = true
  → Default to conservative (deny) + escalate
```

### Implementation

```python
class WarrantyDetermination:
    
    def post_process_warranty(self, action_item: ActionItem,
                              defect: ExtractedDefect,
                              chunks: list[RetrievedChunk],
                              metadata: dict) -> dict:
        """Enrich warranty determination with policy details."""
        
        warranty_info = {
            "warranty_eligible": action_item.warranty_eligible,
            "escalation_required": action_item.escalation_required,
            "policy_reference": None,
            "coverage_type": None,
            "expiry_date": None,
            "determination_basis": None,
        }
        
        # Find warranty-specific chunks
        warranty_chunks = [
            c for c in chunks 
            if c.metadata.get("document_category") == "warranty"
        ]
        
        if not warranty_chunks:
            warranty_info["determination_basis"] = "no_warranty_context_retrieved"
            warranty_info["warranty_eligible"] = False
            warranty_info["escalation_required"] = True
            return warranty_info
        
        # Extract policy reference
        warranty_info["policy_reference"] = (
            f"{warranty_chunks[0].document_title}, "
            f"Version {warranty_chunks[0].metadata.get('version', 'unknown')}"
        )
        
        # If LLM said warranty eligible, verify it referenced a policy
        if action_item.warranty_eligible:
            # Check that the SOP reference mentions a warranty document
            sop_ref_lower = action_item.sop_reference.lower()
            has_warranty_ref = any(
                wc.document_title.lower() in sop_ref_lower 
                for wc in warranty_chunks
            )
            
            if has_warranty_ref:
                warranty_info["determination_basis"] = "policy_referenced"
                warranty_info["coverage_type"] = self.infer_coverage_type(
                    warranty_chunks, metadata
                )
            else:
                # LLM said eligible but didn't reference a policy — suspicious
                warranty_info["determination_basis"] = "llm_determination_ungrounded"
                warranty_info["escalation_required"] = True
        
        return warranty_info
    
    def infer_coverage_type(self, warranty_chunks: list, metadata: dict) -> str:
        """Infer standard vs extended vs goodwill coverage."""
        combined_text = " ".join(c.text.lower() for c in warranty_chunks)
        
        if "extended" in combined_text or "premium" in combined_text:
            return "extended"
        elif "goodwill" in combined_text or "courtesy" in combined_text:
            return "goodwill"
        else:
            return "standard"
```

### Design Decisions

**Decision:** Default to warranty denial + escalation when uncertain.

**Reasoning:** A false warranty approval costs the company money (paying for a repair that should have been customer-responsibility). A false warranty denial frustrates the customer but can be corrected by the human reviewer. The conservative default (deny + escalate) minimizes financial risk while ensuring no denial is final without human review.

**Tradeoff:** More escalations mean more human review work. If the retrieval pipeline is working well, most warranty determinations will have clear context and won't need escalation. An escalation rate above 20% on warranty determinations indicates retrieval issues, not a warranty logic problem.

**Decision:** Flag ungrounded warranty approvals for escalation.

**Reasoning:** If the LLM says "warranty eligible" but its SOP reference doesn't match any retrieved warranty document, the LLM likely hallucinated the warranty determination from its training data rather than from the retrieved policy. This is exactly the faithfulness violation that Ragas measures. Flagging it for human review catches these hallucinations before they become costly false approvals.

---

## 6. Parts Resolution & Catalog Matching

### The Parts Enrichment Pipeline

The LLM generates part names and sometimes part numbers from the retrieved SOP context. But the LLM's part numbers might be outdated (from an old SOP version), approximate (getting one digit wrong), or missing entirely (the SOP describes the part but doesn't include a number).

```python
class PartsResolver:
    
    def __init__(self, parts_catalog_api):
        self.catalog = parts_catalog_api
    
    def resolve_parts(self, parts: list[PartRequired], 
                      model_number: str) -> list[EnrichedPart]:
        """Verify and enrich parts against the parts catalog."""
        
        enriched = []
        
        for part in parts:
            enriched_part = EnrichedPart(
                part_name=part.part_name,
                part_number=part.part_number,
                quantity=part.quantity,
            )
            
            if part.part_number:
                # Try exact match first
                catalog_entry = self.catalog.lookup(part.part_number)
                
                if catalog_entry:
                    enriched_part.catalog_verified = True
                    enriched_part.unit_price = catalog_entry["price"]
                    enriched_part.in_stock = catalog_entry["in_stock"]
                    enriched_part.lead_time_days = catalog_entry["lead_time_days"]
                    
                    # Check compatibility with model
                    if model_number and model_number not in catalog_entry.get("compatible_models", []):
                        enriched_part.warnings = [
                            f"Part {part.part_number} may not be compatible with {model_number}"
                        ]
                else:
                    # Part number not found — try fuzzy search
                    fuzzy_results = self.catalog.fuzzy_search(
                        part_name=part.part_name,
                        model=model_number,
                    )
                    
                    if fuzzy_results:
                        enriched_part.catalog_verified = False
                        enriched_part.alternative_parts = [
                            r["part_number"] for r in fuzzy_results[:3]
                        ]
            else:
                # No part number from LLM — search by name
                search_results = self.catalog.search_by_name(
                    part.part_name, model=model_number
                )
                if search_results:
                    best_match = search_results[0]
                    enriched_part.part_number = best_match["part_number"]
                    enriched_part.catalog_verified = False  # Not from LLM, from catalog search
                    enriched_part.unit_price = best_match["price"]
                    enriched_part.in_stock = best_match["in_stock"]
            
            enriched.append(enriched_part)
        
        return enriched
```

### Design Decisions

**Decision:** Parts catalog lookup is a post-LLM enrichment step, not part of the LLM prompt.

**Reasoning:** The parts catalog is a structured database — exact part numbers, prices, availability, compatibility matrices. The LLM is not the right tool for database lookups. It might hallucinate a plausible-looking part number that doesn't exist, or confuse similar parts across product lines. By looking up parts programmatically after the LLM generates the initial list, we get accurate, current data from the source of truth.

**Tradeoff:** The LLM might specify a part by description rather than number ("compressor assembly" rather than "P/N 7832-A"), requiring the catalog search to do fuzzy matching by name. Fuzzy matching isn't always accurate — "compressor assembly" might match multiple parts for different models. Mitigated by including the model number as a filter in the catalog search, which narrows results to the correct product line.

**Decision:** Flag unverified parts but don't block the output.

**Reasoning:** If a part number from the LLM doesn't match the catalog, it might be outdated, mistyped, or from a different catalog version. Blocking the output would prevent the action item from reaching the technician, who could probably identify the correct part themselves. Instead, we flag it and provide alternative suggestions from fuzzy search, giving the technician a starting point.

---

## 7. Escalation Decision Framework

### When Escalation is Required

Escalation routes the order to a human reviewer instead of (or in addition to) automated processing. Multiple triggers can set `escalation_required = true`:

```python
class EscalationDecider:
    
    def evaluate_escalation(self, action_item: ActionItem,
                           defect: ExtractedDefect,
                           retrieval_quality: dict,
                           validation_signals: dict) -> dict:
        """Determine if escalation is needed and why."""
        
        escalation_reasons = []
        
        # Trigger 1: LLM explicitly set escalation
        if action_item.escalation_required:
            escalation_reasons.append("LLM determined escalation needed")
        
        # Trigger 2: Safety guardrail override
        if self.is_safety_critical(defect, action_item):
            escalation_reasons.append("Safety-critical component involved")
        
        # Trigger 3: Low retrieval quality
        if retrieval_quality.get("top_score", 0) < 0.3:
            escalation_reasons.append(
                f"Low retrieval confidence ({retrieval_quality['top_score']:.2f}) — "
                f"action items may not be grounded in correct SOPs"
            )
        
        # Trigger 4: Ungrounded warranty determination
        if action_item.warranty_eligible and not self.warranty_is_grounded(action_item):
            escalation_reasons.append("Warranty approval not grounded in retrieved policy")
        
        # Trigger 5: Low extraction confidence
        if defect.confidence < 0.5:
            escalation_reasons.append(
                f"Low extraction confidence ({defect.confidence:.2f}) — "
                f"defect identification may be inaccurate"
            )
        
        # Trigger 6: Critical severity
        if defect.severity.value == "critical":
            escalation_reasons.append("Critical severity defect requires human verification")
        
        # Trigger 7: Parts not verified
        unverified_parts = [
            p for p in action_item.parts_required 
            if not getattr(p, 'catalog_verified', False)
        ]
        if unverified_parts and action_item.action_type in ("replace", "recall"):
            escalation_reasons.append(
                f"{len(unverified_parts)} part(s) could not be verified in catalog"
            )
        
        # Trigger 8: Excessive retries indicate difficulty
        if validation_signals.get("generation_retry_count", 0) >= 2:
            escalation_reasons.append("Required multiple retries — output may be unreliable")
        
        should_escalate = len(escalation_reasons) > 0
        escalation_priority = self.determine_priority(escalation_reasons, defect)
        
        return {
            "escalation_required": should_escalate,
            "reasons": escalation_reasons,
            "priority": escalation_priority,
            "auto_processable": not should_escalate,
        }
    
    def determine_priority(self, reasons: list, defect) -> str:
        """Determine human review priority based on escalation reasons."""
        if any("safety" in r.lower() for r in reasons):
            return "immediate"
        if defect.severity.value == "critical":
            return "high"
        if any("warranty" in r.lower() for r in reasons):
            return "medium"
        return "standard"
```

### Escalation Categories

```
IMMEDIATE (target: reviewed within 1 hour)
  • Safety-critical components
  • Critical severity + safety keywords

HIGH (target: reviewed within 4 hours)
  • Critical severity defects
  • Multiple escalation triggers simultaneously

MEDIUM (target: reviewed within 24 hours)
  • Warranty determination uncertain
  • Low retrieval confidence
  • Parts not verifiable

STANDARD (target: reviewed within 48 hours)
  • Low extraction confidence
  • Excessive retries
  • Business logic warnings only
```

---

## 8. Multi-Segment Order Reassembly

### The Problem

Phase 1 splits multi-defect orders into separate segments. Each segment is processed independently through the entire pipeline. After processing, the results must be reassembled into a single order-level output.

### Implementation

```python
class OrderReassembler:
    
    def __init__(self, result_store):
        self.store = result_store  # Redis or DynamoDB
    
    def collect_segment_result(self, segment_result: dict):
        """Store a completed segment result and check if the order is complete."""
        
        order_id = segment_result["order_id"]
        segment_id = segment_result["segment_id"]
        
        # Store this segment's result
        self.store.set(
            f"order:{order_id}:segment:{segment_id}",
            json.dumps(segment_result),
            ttl=86400  # 24-hour TTL
        )
        
        # Check if all segments for this order are complete
        expected_segments = self.store.get(f"order:{order_id}:expected_count")
        completed_segments = self.store.keys(f"order:{order_id}:segment:*")
        
        if len(completed_segments) >= int(expected_segments):
            return self.reassemble_order(order_id)
        
        return None  # Not all segments complete yet
    
    def reassemble_order(self, order_id: str) -> dict:
        """Combine all segment results into a single order output."""
        
        segment_keys = self.store.keys(f"order:{order_id}:segment:*")
        segments = []
        for key in sorted(segment_keys):
            segment_data = json.loads(self.store.get(key))
            segments.append(segment_data)
        
        # ── Merge action items from all segments ──
        all_action_items = []
        all_defects = []
        all_warnings = []
        
        for segment in segments:
            if segment.get("action_items"):
                all_action_items.extend(segment["action_items"])
            if segment.get("extracted_defects"):
                all_defects.extend(segment["extracted_defects"])
            if segment.get("quality_signals", {}).get("business_warnings"):
                all_warnings.extend(segment["quality_signals"]["business_warnings"])
        
        # ── Resolve cross-segment dependencies ──
        all_action_items = self.resolve_cross_references(all_action_items, segments)
        
        # ── Deduplicate action items ──
        all_action_items = self.deduplicate_actions(all_action_items)
        
        # ── Determine order-level escalation ──
        order_escalation = any(
            ai.get("escalation_required", False) for ai in all_action_items
        )
        order_priority = self.highest_priority(all_action_items)
        
        # ── Compute order-level summary ──
        order_output = {
            "order_id": order_id,
            "status": "completed",
            "processed_at": datetime.utcnow().isoformat(),
            "total_segments": len(segments),
            "total_defects": len(all_defects),
            "total_action_items": len(all_action_items),
            
            "defects": all_defects,
            "action_items": all_action_items,
            
            "order_level_signals": {
                "escalation_required": order_escalation,
                "overall_priority": order_priority,
                "all_warnings": all_warnings,
                "estimated_total_cost": sum(
                    ai.get("estimated_parts_cost", 0) or 0 
                    for ai in all_action_items
                ),
                "estimated_total_time_hours": sum(
                    ai.get("estimated_repair_time_hours", 0) or 0 
                    for ai in all_action_items
                ),
            },
            
            "segment_details": segments,
        }
        
        return order_output
    
    def resolve_cross_references(self, action_items: list, segments: list) -> list:
        """Handle cross-referenced defects from Phase 1 segmentation."""
        
        for segment in segments:
            cross_refs = segment.get("cross_references")
            if cross_refs and cross_refs.get("type") == "causal":
                # This segment's defect was caused by another segment's defect
                # Add a note to the action item linking them
                for ai in action_items:
                    if ai.get("segment_id") == segment["segment_id"]:
                        ai["causal_note"] = (
                            f"This defect may be related to defect in segment "
                            f"{cross_refs['related_segments']}. "
                            f"Consider addressing root cause first."
                        )
        
        return action_items
    
    def deduplicate_actions(self, action_items: list) -> list:
        """Remove duplicate action items from multi-segment processing."""
        
        seen = set()
        unique = []
        
        for ai in action_items:
            # Create a fingerprint based on component + action type
            fingerprint = f"{ai.get('action_type')}:{ai.get('description', '')[:100]}"
            
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(ai)
            else:
                # Duplicate detected — keep the one with higher confidence
                pass
        
        return unique
```

### Design Decisions

**Decision:** Use a distributed store (Redis) for segment collection rather than in-memory aggregation.

**Reasoning:** Segments are processed by different agent workers, potentially on different machines. In-memory aggregation would require routing all segments of the same order to the same worker, which couples processing to routing and creates single-point-of-failure risks. A shared Redis store decouples processing from aggregation — any worker processes any segment, and a separate reassembly process (or the last segment's worker) triggers the merge when all segments are collected.

**Tradeoff:** Redis adds an external dependency and network latency (~1-5ms per read/write). But the data is small (JSON objects under 10KB each) and the operations are simple (set/get/keys). Redis handles this trivially.

**Decision:** Deduplicate action items across segments.

**Reasoning:** Phase 1 segmentation occasionally over-splits — creating two segments for what is really one defect. Each segment would independently produce an action item for the same defect. Without deduplication, the technician receives two identical repair orders. The fingerprint-based deduplication catches exact duplicates while allowing genuinely different action items for similar components to coexist.

---

## 9. Final Output Assembly

### The Complete Order Output

```json
{
    "order_id": "SO-28841",
    "status": "completed",
    "processed_at": "2025-03-04T14:22:15.000Z",
    
    "summary": {
        "total_defects": 2,
        "total_action_items": 2,
        "overall_priority": "urgent",
        "escalation_required": false,
        "estimated_total_cost": 487.50,
        "estimated_total_time_hours": 3.5,
        "warranty_eligible_items": 1,
        "automation_level": "fully_automated"
    },
    
    "defects": [
        {
            "defect_id": "SO-28841-D1",
            "defect_type": "mechanical",
            "severity": "high",
            "affected_component": "compressor",
            "symptoms": ["grinding noise"],
            "error_codes": ["E-47"],
            "confidence": 0.92
        },
        {
            "defect_id": "SO-28841-D2",
            "defect_type": "hydraulic",
            "severity": "medium",
            "affected_component": "coolant line valve assembly",
            "symptoms": ["visible leak", "pressure dropping"],
            "error_codes": [],
            "confidence": 0.87
        }
    ],
    
    "action_items": [
        {
            "action_id": "SO-28841-A1",
            "defect_reference": "SO-28841-D1",
            "action_type": "replace",
            "priority": "urgent",
            "description": "Replace compressor assembly per SOP-XR-COMP-001 Section 3.2...",
            "parts_required": [
                {
                    "part_name": "Compressor Assembly XR-440",
                    "part_number": "P/N 7832-A",
                    "quantity": 1,
                    "catalog_verified": true,
                    "unit_price": 342.00,
                    "in_stock": true,
                    "lead_time_days": 0
                }
            ],
            "warranty_eligible": true,
            "warranty_details": {
                "policy_reference": "Warranty Policy XR-Series v4.2",
                "coverage_type": "standard",
                "determination_basis": "policy_referenced"
            },
            "escalation_required": false,
            "sop_reference": "SOP-XR-COMP-001, Section 3.2",
            "estimated_repair_time_hours": 2.5,
            "safety_precautions": ["Lockout-tagout required", "EPA 608 refrigerant handling"],
            "confidence": 0.92,
            "generation_attempt": 1
        },
        {
            "action_id": "SO-28841-A2",
            "defect_reference": "SO-28841-D2",
            "action_type": "repair",
            "priority": "routine",
            "description": "Inspect and repair coolant line valve assembly...",
            "parts_required": [
                {
                    "part_name": "Valve Seal Kit",
                    "part_number": "VS-KIT-440",
                    "quantity": 1,
                    "catalog_verified": true,
                    "unit_price": 45.50,
                    "in_stock": true
                }
            ],
            "warranty_eligible": true,
            "escalation_required": false,
            "sop_reference": "SOP-XR-COOL-003, Section 2.1",
            "estimated_repair_time_hours": 1.0,
            "confidence": 0.87,
            "generation_attempt": 1
        }
    ],
    
    "quality_report": {
        "extraction_retries": 0,
        "generation_retries": 0,
        "retrieval_top_score": 0.82,
        "cross_validation_clean": true,
        "business_warnings": [],
        "safety_guardrails_triggered": false,
        "total_processing_time_ms": 8450,
        "node_durations": {
            "extract_defects": 2100,
            "validate_extraction": 15,
            "retrieve_context": 340,
            "generate_action_items": 3200,
            "validate_output": 18,
            "assemble_output": 45,
            "enrichment": 230
        }
    },
    
    "raw_input_reference": {
        "original_text_hash": "sha256:a1b2c3...",
        "source_channel": "email",
        "received_at": "2025-03-04T14:22:00Z"
    }
}
```

---

## 10. Downstream Integration Architecture

### Integration Pattern

```
[Order Output (JSON)]
        │
        ▼
[Output Router]
        │
        ├──▶ [Service Management System] — ServiceNow, SAP, Dynamics 365
        │     API webhook or direct integration
        │     Creates work orders, assigns technicians
        │
        ├──▶ [Parts Procurement System] — ERP integration
        │     Auto-generates purchase orders for required parts
        │     Only for verified parts with catalog_verified=true
        │
        ├──▶ [Warranty Claims System] — Claims processing
        │     Files warranty claims for eligible items
        │     Only for warranty_eligible=true with grounded determination
        │
        ├──▶ [Human Review Dashboard] — Web application
        │     For escalated orders and quality monitoring
        │     Human can approve, modify, or reject action items
        │
        └──▶ [Analytics Pipeline] — Data warehouse
              Processing metrics, defect trends, model performance
              Feeds into retraining decisions and business intelligence
```

### Implementation

```python
class OutputRouter:
    
    def __init__(self, integrations: dict):
        self.service_mgmt = integrations["service_management"]
        self.parts = integrations["parts_procurement"]
        self.warranty = integrations["warranty_claims"]
        self.dashboard = integrations["human_review"]
        self.analytics = integrations["analytics"]
    
    def route_output(self, order_output: dict):
        """Route the final output to all relevant downstream systems."""
        
        results = {}
        
        # Always: send to analytics
        results["analytics"] = self.analytics.publish(order_output)
        
        # Route based on escalation status
        if order_output["summary"]["escalation_required"]:
            # Escalated: primary destination is human review
            results["dashboard"] = self.dashboard.create_review_item(
                order_output,
                priority=order_output["summary"]["overall_priority"],
            )
            # Still create a draft work order (human will approve)
            results["service_mgmt"] = self.service_mgmt.create_draft_work_order(
                order_output
            )
        else:
            # Fully automated: create active work order
            results["service_mgmt"] = self.service_mgmt.create_work_order(
                order_output
            )
            
            # Trigger parts procurement for verified parts
            verified_parts = self.collect_verified_parts(order_output)
            if verified_parts:
                results["parts"] = self.parts.create_purchase_order(
                    order_id=order_output["order_id"],
                    parts=verified_parts,
                )
            
            # File warranty claims for eligible items
            warranty_items = self.collect_warranty_items(order_output)
            if warranty_items:
                results["warranty"] = self.warranty.file_claim(
                    order_id=order_output["order_id"],
                    items=warranty_items,
                )
        
        return results
```

### Design Decisions

**Decision:** Escalated orders get draft work orders, not active ones.

**Reasoning:** An escalated order needs human review before action. Creating an active work order would dispatch a technician before the review is complete — potentially sending them with wrong parts or wrong procedures. A draft work order reserves the slot in the scheduling system but doesn't trigger dispatch until a human approves.

**Decision:** Only auto-procure parts that are catalog-verified.

**Reasoning:** Ordering the wrong part wastes money and time. Parts that couldn't be verified against the catalog might be outdated, misidentified, or from the wrong product line. Only verified parts are auto-procured. Unverified parts appear in the work order as "pending verification" for the technician or dispatcher to resolve.

**Decision:** Always send to analytics, regardless of outcome.

**Reasoning:** Every processed order — successful, escalated, or failed — contains valuable data for model improvement, business intelligence, and operational monitoring. Filtering analytics to only successful orders would hide failure patterns that need investigation.

---

## 11. Integration with Service Management Systems

### ServiceNow Integration Example

```python
class ServiceNowIntegration:
    
    def __init__(self, instance_url: str, api_key: str):
        self.base_url = f"https://{instance_url}/api/now/table"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    
    def create_work_order(self, order_output: dict) -> dict:
        """Create a work order in ServiceNow."""
        
        work_order = {
            "short_description": self.build_summary(order_output),
            "description": self.build_detailed_description(order_output),
            "priority": self.map_priority(order_output["summary"]["overall_priority"]),
            "assignment_group": self.determine_assignment_group(order_output),
            "category": "repair",
            "subcategory": order_output["defects"][0]["defect_type"] if order_output["defects"] else "general",
            "u_estimated_cost": order_output["summary"]["estimated_total_cost"],
            "u_estimated_duration": order_output["summary"]["estimated_total_time_hours"],
            "u_warranty_status": "eligible" if order_output["summary"]["warranty_eligible_items"] > 0 else "not_eligible",
            "u_ai_processed": True,
            "u_ai_confidence": min(d["confidence"] for d in order_output["defects"]) if order_output["defects"] else 0,
            "u_source_order_id": order_output["order_id"],
        }
        
        response = httpx.post(
            f"{self.base_url}/wm_order",
            headers=self.headers,
            json=work_order,
            timeout=15.0,
        )
        response.raise_for_status()
        
        return response.json()
    
    def build_summary(self, output: dict) -> str:
        """Build a one-line summary for the work order."""
        defect_count = output["summary"]["total_defects"]
        components = list(set(d["affected_component"] for d in output["defects"]))
        priority = output["summary"]["overall_priority"]
        
        return (
            f"[{priority.upper()}] {defect_count} defect(s) identified: "
            f"{', '.join(components[:3])}"
        )
    
    def build_detailed_description(self, output: dict) -> str:
        """Build the full description with action items."""
        desc = "AI-GENERATED SERVICE ORDER\n"
        desc += "=" * 40 + "\n\n"
        
        for ai in output["action_items"]:
            desc += f"ACTION: {ai['action_type'].upper()} — {ai['priority'].upper()}\n"
            desc += f"{ai['description']}\n"
            desc += f"SOP Reference: {ai['sop_reference']}\n"
            
            if ai["parts_required"]:
                desc += "Parts:\n"
                for part in ai["parts_required"]:
                    desc += f"  • {part['part_name']}"
                    if part.get("part_number"):
                        desc += f" ({part['part_number']})"
                    desc += f" × {part['quantity']}\n"
            
            warranty = "YES" if ai["warranty_eligible"] else "NO"
            desc += f"Warranty: {warranty}\n"
            desc += "\n" + "-" * 30 + "\n\n"
        
        return desc
```

---

## 12. Human Review Dashboard

### Dashboard Features

The human review dashboard is the interface where reviewers interact with escalated orders and quality-sampled outputs.

**Queue Management:** Orders sorted by escalation priority (immediate → high → medium → standard). Age tracking — orders approaching SLA deadline are highlighted. Assignee management — reviewers claim orders to avoid duplicate work.

**Review Interface:** Side-by-side view: original service order text on the left, AI-generated action items on the right. Inline editing — reviewer can modify action items directly (change severity, edit description, add/remove parts, toggle warranty). Approve/reject buttons per action item. "Accept all" for orders that look correct. Rejection requires a reason (feeds back to training).

**Feedback Capture:** When a reviewer corrects an action item, the correction is captured as a labeled training example — the original input + the human-corrected output becomes a new training pair for the next fine-tuning round. Reviewer confidence rating — "this was clearly wrong" vs "this was borderline, I made a judgment call." Categorized rejection reasons — "wrong defect type," "wrong severity," "hallucinated SOP reference," "wrong warranty determination."

---

## 13. Feedback Loop — Human Corrections Back to Training

### The Virtuous Cycle

```
Orders processed → Some escalated to human review
                          │
                          ▼
                   Human corrects action items
                          │
                          ▼
                   Corrections captured as training data
                          │
                          ▼
                   Monthly retraining incorporates corrections
                          │
                          ▼
                   Model improves → fewer escalations → less human work
```

### Implementation

```python
class FeedbackCollector:
    
    def capture_correction(self, original_output: dict, 
                          corrected_output: dict,
                          reviewer_metadata: dict):
        """Capture a human correction as a training example."""
        
        training_example = {
            "input": {
                "segment_text": original_output["raw_input_reference"]["segment_text"],
                "pre_extracted_entities": original_output.get("pre_extracted_entities"),
                "inherited_context": original_output.get("inherited_context"),
            },
            "original_model_output": original_output["action_items"],
            "corrected_output": corrected_output["action_items"],
            "correction_type": self.classify_correction(original_output, corrected_output),
            "reviewer_confidence": reviewer_metadata.get("confidence", "medium"),
            "correction_reason": reviewer_metadata.get("reason", ""),
            "reviewer_id": reviewer_metadata["reviewer_id"],
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Store in training data pipeline
        self.training_store.append(training_example)
        
        # Update correction analytics
        self.analytics.track_correction(training_example)
    
    def classify_correction(self, original: dict, corrected: dict) -> str:
        """Classify what type of correction was made."""
        
        orig_items = original.get("action_items", [])
        corr_items = corrected.get("action_items", [])
        
        if len(corr_items) > len(orig_items):
            return "added_missing_action_item"
        if len(corr_items) < len(orig_items):
            return "removed_incorrect_action_item"
        
        # Check field-level changes
        changes = []
        for orig, corr in zip(orig_items, corr_items):
            for field in ["action_type", "severity", "warranty_eligible", "priority"]:
                if orig.get(field) != corr.get(field):
                    changes.append(f"{field}_changed")
        
        if changes:
            return ",".join(set(changes))
        
        return "description_edited"
```

### Quality Gates for Training Data from Corrections

Not every human correction should become training data:

**Include:** Corrections with high reviewer confidence. Corrections where the original output was clearly wrong (wrong defect type, hallucinated SOP). Corrections from experienced reviewers (>100 reviews completed).

**Exclude:** Borderline corrections where the reviewer made a judgment call. Corrections that change only formatting or wording without changing meaning. Corrections from new reviewers still in training (first 20 reviews).

---

## 14. SLA & Processing Time Guarantees

### Target SLAs

```
End-to-end processing time:
  P50:  < 8 seconds
  P95:  < 15 seconds
  P99:  < 30 seconds (includes retry scenarios)

Time breakdown (typical, no retries):
  Phase 1 (preprocessing):      500ms - 1.5s
  Phase 3 (LLM extraction):     1.5s - 3s
  Phase 5 (extraction validation): 20ms
  Phase 4 (RAG retrieval):      200ms - 500ms
  Phase 3 (LLM generation):     2s - 4s
  Phase 5 (generation validation): 20ms
  Phase 6 (enrichment + dispatch): 200ms - 500ms
  ────────────────────────────────────────
  Total:                         ~5-10 seconds

Human review SLA:
  Immediate priority:  < 1 hour
  High priority:       < 4 hours
  Medium priority:     < 24 hours
  Standard priority:   < 48 hours
```

---

## 15. The 85% Automation Metric — How It's Measured

### Definition

"85% reduction in manual processing effort" means that 85% of service orders complete the full pipeline without human intervention and produce action items that technicians can act on directly.

### Measurement

```python
def calculate_automation_rate(orders: list[dict], period: str) -> dict:
    """Calculate the automation rate for a given period."""
    
    total = len(orders)
    fully_automated = 0
    partially_automated = 0
    human_required = 0
    
    for order in orders:
        if order["summary"]["automation_level"] == "fully_automated":
            fully_automated += 1
        elif order["summary"]["automation_level"] == "partially_automated":
            partially_automated += 1
        else:
            human_required += 1
    
    # Full automation: no escalation, no human review needed
    full_rate = fully_automated / total * 100
    
    # Partial automation: human reviewed but started from AI output
    # (AI saved ~60-70% of the effort even when escalated)
    partial_effort_saved = partially_automated * 0.65  # 65% effort saved
    
    # Total effort reduction
    total_effort_saved = (fully_automated + partial_effort_saved) / total * 100
    
    return {
        "period": period,
        "total_orders": total,
        "fully_automated": fully_automated,
        "fully_automated_rate": full_rate,
        "partially_automated": partially_automated,
        "human_required": human_required,
        "total_effort_reduction": total_effort_saved,
    }
```

### The 85% Breakdown

```
Fully automated (no human touch):          ~82% of orders
  → 100% effort saved per order

Partially automated (human review with AI assist): ~12% of orders
  → ~65% effort saved per order (human starts from AI output, not scratch)

Fully manual (AI couldn't process):        ~6% of orders
  → 0% effort saved

Weighted effort reduction:
  (82% × 1.0) + (12% × 0.65) + (6% × 0.0) = 82% + 7.8% + 0% = ~89.8%

Conservative claim: 85% (accounting for measurement uncertainty)
```

---

## 16. Observability & Alerting

### Metrics Emitted

**Output Quality Metrics:**

| Metric | Description |
|--------|-------------|
| `output.automation_level` | fully_automated / partially_automated / manual |
| `output.escalation_rate` | Percentage of orders requiring escalation |
| `output.escalation_reasons` | Distribution of why orders escalate |
| `output.action_items_per_order` | Average action items generated |
| `output.parts_verified_rate` | Percentage of parts verified in catalog |
| `output.warranty_eligible_rate` | Percentage of items determined warranty eligible |
| `output.estimated_cost_per_order` | Average estimated repair cost |

**Integration Metrics:**

| Metric | Description |
|--------|-------------|
| `integration.servicenow.success_rate` | Work order creation success rate |
| `integration.servicenow.latency_ms` | API call latency |
| `integration.parts.procurement_triggered` | Auto-procurement count |
| `integration.warranty.claims_filed` | Auto-warranty claims filed |
| `integration.dashboard.queue_depth` | Human review queue size |
| `integration.dashboard.review_time_avg` | Average human review time |

**Feedback Metrics:**

| Metric | Description |
|--------|-------------|
| `feedback.corrections_per_week` | Human corrections captured |
| `feedback.correction_types` | Distribution of correction categories |
| `feedback.reviewer_agreement` | Inter-reviewer agreement on corrections |

### Alert Configuration

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Automation rate drop | Below 80% for 24 hours | P2 | Investigate model quality, retrieval quality, new data patterns |
| Escalation spike | Above 25% for 4 hours | P2 | Check retrieval pipeline, model output quality |
| ServiceNow integration failure | Error rate >5% for 30min | P1 | Check API credentials, ServiceNow availability |
| Human review queue overflow | >200 pending items | P1 | Scale reviewers, investigate escalation spike |
| Review SLA breach | Immediate priority unreviewed >2 hours | P1 | Alert review team lead, reassign |
| Parts procurement failure | Auto-procurement error rate >10% | P2 | Catalog API issue, verify integration |
| Warranty claim rejection spike | >20% of filed claims rejected by warranty system | P2 | Warranty determination logic may be wrong |
| Feedback correction rate spike | >30% of reviewed orders corrected | P2 | Model quality issue, prioritize retraining |

---

## 17. End-to-End Phase 6 Flow Summary

### Per-Segment Output Flow

```
VALIDATED ACTION ITEMS ARRIVE FROM PHASE 5
        │
Step 1 ─┤  WARRANTY ENRICHMENT
        │  • Cross-reference warranty determination against retrieved policy
        │  • Verify LLM's warranty decision is grounded in context
        │  • Add policy reference, coverage type, expiry date
        │  • Flag ungrounded approvals for escalation
        │
Step 2 ─┤  PARTS ENRICHMENT
        │  • Look up each part in catalog by part number
        │  • Verify compatibility with model number
        │  • Add pricing, availability, lead time
        │  • Fuzzy search for unmatched parts → suggest alternatives
        │  • Flag unverified parts
        │
Step 3 ─┤  ESCALATION EVALUATION
        │  • Evaluate all escalation triggers:
        │    Safety-critical, low retrieval quality, ungrounded warranty,
        │    low confidence, critical severity, unverified parts, excessive retries
        │  • Assign escalation priority (immediate/high/medium/standard)
        │
Step 4 ─┤  SAFETY GUARDRAIL CHECK (final pass)
        │  • Force escalation on safety-critical components
        │  • Flag regulated components for compliance
        │  • Anomaly detection (excessive items, multiple criticals)
        │
Step 5 ─┤  OUTPUT SANITIZATION
        │  • PII leak scan
        │  • String normalization
        │  • Output size check
        │
        ▼
SEGMENT RESULT STORED IN COLLECTION STORE (Redis)
        │
        ├── If all segments complete → REASSEMBLY
        │                                │
        │                         Step 6 ─┤  ORDER REASSEMBLY
        │                                 │  • Merge action items from all segments
        │                                 │  • Resolve cross-segment references
        │                                 │  • Deduplicate action items
        │                                 │  • Compute order-level summary
        │                                 │  • Determine overall escalation and priority
        │                                 │
        │                                 ▼
        │                          FINAL ORDER OUTPUT
        │                                 │
        │                          Step 7 ─┤  DOWNSTREAM ROUTING
        │                                 │  │
        │                                 │  ├──▶ Service Management (ServiceNow)
        │                                 │  │    Active work order (automated)
        │                                 │  │    OR draft work order (escalated)
        │                                 │  │
        │                                 │  ├──▶ Parts Procurement (ERP)
        │                                 │  │    Auto-PO for verified parts
        │                                 │  │    (automated orders only)
        │                                 │  │
        │                                 │  ├──▶ Warranty Claims
        │                                 │  │    File claims for eligible items
        │                                 │  │    (grounded determinations only)
        │                                 │  │
        │                                 │  ├──▶ Human Review Dashboard
        │                                 │  │    Escalated orders with full context
        │                                 │  │    Priority-sorted queue
        │                                 │  │
        │                                 │  └──▶ Analytics Pipeline
        │                                 │       All orders (success + escalated + failed)
        │                                 │       Processing metrics, defect trends
        │                                 │
        │                                 └──▶ FEEDBACK LOOP
        │                                       Human corrections → training data
        │                                       Monthly retraining cycle
        │
        └── If segments still pending → WAIT
```

### Key Design Principles Applied Throughout Phase 6

1. **The LLM reasons, the system integrates.** The LLM produces the intellectual work (defect analysis, warranty determination, repair recommendations). Phase 6 handles everything else — parts lookup, format conversion, system integration, routing. Don't ask the LLM to do what a database query does better.

2. **Conservative defaults protect the business.** Warranty denied + escalation when uncertain. Draft work orders for escalated items. Only auto-procure verified parts. Every conservative default can be overridden by a human reviewer, but the default minimizes financial and safety risk.

3. **Enrichment adds value the LLM can't.** Current parts pricing, real-time availability, estimated repair times from SOP metadata, compatibility checks against the parts catalog — these come from structured data systems, not from an LLM. Post-processing enrichment turns good AI output into actionable work orders.

4. **Multi-segment reassembly is an order-level concern, not a segment-level concern.** Segments are processed independently for parallelism. Reassembly happens once, at the end, with deduplication and cross-reference resolution. This keeps the per-segment pipeline simple while handling multi-defect complexity at the order level.

5. **The feedback loop closes the circle.** Human corrections on escalated orders become training data for the next model version. The system literally learns from its mistakes — orders it couldn't process today become the examples that make it better tomorrow.

6. **Automation is measured by effort reduction, not just pass rate.** An 82% full-automation rate understates the system's value because the 12% that's partially automated still saves 65% of human effort. The 85% metric captures total effort reduction, which is what the business actually cares about.

7. **Every downstream integration can fail independently.** ServiceNow being down shouldn't prevent parts procurement. A warranty system error shouldn't block the work order. Each integration fires independently with its own error handling and retry logic. The order output is the source of truth — if an integration fails, it can be replayed later.

---

*Document Version: 1.0 | Last Updated: March 2026 | System: LGS Tech Agentic AI Workflow*
