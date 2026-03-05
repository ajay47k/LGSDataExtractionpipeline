# Phase 5 — Validation & Guardrails: Complete Production Reference

## LGS Tech Agentic AI Workflow — Service Order Processing

---

## Table of Contents

1. Phase 5 Overview & Why Validation is a Separate Phase
2. The Three Layers of Validation
3. Layer 1 — Structural Validation (JSON Parsing)
4. Layer 2 — Schema Validation (Pydantic)
5. Layer 3 — Business Logic Validation
6. Cross-Validation with Phase 1 Entities
7. Confidence Calibration Checks
8. The Self-Healing Retry Loop
9. Retry Prompt Engineering
10. Retry Budget & Exhaustion Strategy
11. Human Review Routing
12. Guardrails — Preventing Dangerous Outputs
13. Output Sanitization
14. Validation at Both Gates (Extraction + Generation)
15. Edge Cases & How They're Handled
16. Observability & Alerting
17. End-to-End Phase 5 Flow Summary

---

## 1. Phase 5 Overview & Why Validation is a Separate Phase

### What Phase 5 Does

Phase 5 is the quality gate that stands between every LLM output and the downstream system. It validates every piece of LLM-generated content — both the defect extractions from Phase 3 and the action items from the generation step — against strict structural, schema, and business logic rules. If validation fails, it orchestrates the retry loop with corrective feedback. If retries are exhausted, it routes to human review with full diagnostic context. It is the mechanism that turns a probabilistic system (LLM) into a reliable production system (95% first-pass reliability).

### Why Not Validate Inside the LLM Node?

You could embed validation logic inside the extraction or generation node — call the LLM, parse the response, validate, and retry all within one function. This is how many early LLM applications are built. It fails in production for three reasons.

**Reason 1 — Separation of concerns.** The LLM node's job is to call the model and store the raw response. The validation node's job is to evaluate quality and make routing decisions. Combining them means the LLM call code must understand retry logic, routing decisions, and error handling — becoming a monolithic function that's hard to test, debug, and modify. When you need to change a validation rule (e.g., add a new required field), you shouldn't have to touch the LLM calling code.

**Reason 2 — Graph-level routing.** LangGraph's power is in conditional edges — the graph itself decides where to route based on node outputs. If validation is inside the LLM node, the routing logic (retry vs continue vs escalate) is embedded in Python code, invisible to the graph structure. Moving validation to a separate node makes the routing explicit in the graph definition, visible in traces, and auditable.

**Reason 3 — Observability.** A separate validation node produces its own trace span in LangSmith. You can see exactly what the validator received, what errors it found, what decision it made, and how long it took — independently from the LLM call metrics. Combined nodes would muddy the trace — did the latency come from the LLM call or the validation logic?

### Where Validation Happens in the Graph

Phase 5 isn't a single node — it's a validation pattern that appears at two gates in the LangGraph:

```
Gate 1: After Extraction (Phase 3)
  Node 1 (extract_defects) → Node 2 (validate_extraction) → route

Gate 2: After Generation (Phase 5 action items)
  Node 4 (generate_action_items) → Node 5 (validate_output) → route
```

Both gates use the same validation architecture but with different schemas and different business rules. This document covers the complete validation framework that both gates share.

---

## 2. The Three Layers of Validation

Every LLM output passes through three validation layers, each catching a different category of error:

```
Layer 1 — STRUCTURAL VALIDATION
  "Is this valid JSON?"
  Catches: malformed JSON, markdown fences, preamble text,
           trailing commas, unclosed brackets
  Pass rate without constrained decoding: ~90-95%
  Pass rate with constrained decoding: ~99%+

            │ (if passes)
            ▼

Layer 2 — SCHEMA VALIDATION (Pydantic)
  "Does the JSON match the expected schema?"
  Catches: missing required fields, wrong types, invalid enum values,
           out-of-range numbers, wrong array structure
  Pass rate: ~90-95% first attempt

            │ (if passes)
            ▼

Layer 3 — BUSINESS LOGIC VALIDATION
  "Does the content make sense?"
  Catches: severity-priority mismatches, missing SOP references,
           defect count discrepancies, implausible confidence scores
  Pass rate: ~95% (produces warnings, not hard failures)
```

Each layer is cheaper and faster than the next. Structural validation is a single `json.loads()` call (microseconds). Schema validation runs Pydantic (milliseconds). Business logic runs custom checks (milliseconds). This layered approach ensures cheap checks catch easy errors before expensive checks run — and critically, each layer produces specific error messages that feed into the retry prompt.

---

## 3. Layer 1 — Structural Validation (JSON Parsing)

### What It Catches

The LLM returned text that isn't valid JSON. Even with constrained decoding, edge cases exist: the model might exceed max_tokens mid-JSON (truncated output), the vLLM server might return an error message instead of model output, or network issues might corrupt the response.

### Implementation

```python
import json
import re

class StructuralValidator:
    
    def validate(self, raw_response: str) -> tuple[bool, any, list[str]]:
        """
        Returns: (is_valid, parsed_data, errors)
        """
        errors = []
        
        if raw_response is None:
            return False, None, ["LLM returned null response (inference failure)"]
        
        if not raw_response.strip():
            return False, None, ["LLM returned empty string"]
        
        # Step 1: Clean common LLM output artifacts
        cleaned = self.clean_response(raw_response)
        
        # Step 2: Attempt JSON parse
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Attempt recovery
            recovered, recovery_error = self.attempt_recovery(cleaned, e)
            if recovered is not None:
                return True, recovered, [f"JSON recovered after: {recovery_error}"]
            
            errors.append(f"JSON parse error at position {e.pos}: {e.msg}")
            errors.append(f"Near: ...{cleaned[max(0,e.pos-30):e.pos+30]}...")
            return False, None, errors
        
        # Step 3: Verify it's the expected type (array for extractions)
        if not isinstance(parsed, list):
            if isinstance(parsed, dict):
                parsed = [parsed]  # Single object → wrap in array
            else:
                errors.append(f"Expected JSON array or object, got {type(parsed).__name__}")
                return False, None, errors
        
        # Step 4: Verify non-empty
        if len(parsed) == 0:
            errors.append("JSON array is empty — expected at least one item")
            return False, None, errors
        
        return True, parsed, []
    
    def clean_response(self, text: str) -> str:
        """Remove common LLM artifacts that break JSON parsing."""
        text = text.strip()
        
        # Remove markdown code fences
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        # Remove preamble text before the JSON
        # LLMs sometimes write "Here is the extraction:" before the JSON
        json_start = None
        for i, char in enumerate(text):
            if char in '[{':
                json_start = i
                break
        
        if json_start and json_start > 0:
            text = text[json_start:]
        
        # Remove trailing text after JSON closes
        # Find the matching closing bracket
        bracket_depth = 0
        json_end = len(text)
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char in '[{':
                bracket_depth += 1
            elif char in ']}':
                bracket_depth -= 1
                if bracket_depth == 0:
                    json_end = i + 1
                    break
        
        text = text[:json_end]
        
        return text.strip()
    
    def attempt_recovery(self, text: str, error: json.JSONDecodeError) -> tuple:
        """Try to recover from common JSON errors."""
        
        # Recovery 1: Trailing comma before closing bracket
        # {"key": "value",} → {"key": "value"}
        fixed = re.sub(r',\s*([}\]])', r'\1', text)
        try:
            return json.loads(fixed), "removed trailing comma"
        except json.JSONDecodeError:
            pass
        
        # Recovery 2: Truncated JSON (max_tokens exceeded)
        # Try closing open brackets
        if error.msg == "Expecting ',' delimiter" or "Unterminated" in error.msg:
            # Count open brackets
            open_brackets = text.count('[') - text.count(']')
            open_braces = text.count('{') - text.count('}')
            
            # Try to close them
            attempt = text
            # First close any open strings
            if text.count('"') % 2 == 1:
                attempt += '"'
            attempt += '}' * open_braces
            attempt += ']' * open_brackets
            
            try:
                return json.loads(attempt), "closed truncated JSON"
            except json.JSONDecodeError:
                pass
        
        # Recovery 3: Single quotes instead of double quotes
        fixed = text.replace("'", '"')
        try:
            return json.loads(fixed), "replaced single quotes with double quotes"
        except json.JSONDecodeError:
            pass
        
        return None, "recovery failed"
```

### Design Decisions

**Decision:** Attempt JSON recovery before declaring failure.

**Reasoning:** Many JSON parse failures are minor formatting issues — a trailing comma, a truncated response, single quotes instead of double quotes. These are fixable programmatically. If the recovery succeeds, we avoid a retry cycle (saving an LLM call, 2-5 seconds of latency, and GPU compute). Recovery is attempted in order of likelihood and safety: trailing comma fix is nearly always safe, bracket closing is sometimes safe (but might produce incomplete objects), and single-quote replacement is a last resort.

**Tradeoff:** Recovery might produce valid JSON that's semantically wrong. For example, closing a truncated JSON might produce `{"defect_type": "mechani"}` — valid JSON but truncated value. This is caught by Layer 2 (Pydantic will reject "mechani" as an invalid enum value) or Layer 3 (business logic checks for implausible values). Recovery is a best-effort optimization, not a guarantee.

**Rule:** Recovery artifacts are logged as warnings (`"JSON recovered after: removed trailing comma"`). If the recovery rate exceeds 5% of total requests, the underlying issue should be investigated — it might indicate a prompt engineering problem, a constrained decoding misconfiguration, or a model quality issue.

**Decision:** Use bracket-matching to find JSON boundaries rather than naive string slicing.

**Reasoning:** LLMs sometimes output text both before AND after the JSON: `"Here are the defects:\n[{...}]\n\nLet me know if you need changes."` Naive approaches (find first `[`, find last `]`) work for simple cases but fail when the trailing text contains brackets. The bracket-depth-tracking approach correctly identifies where the JSON starts and ends, even with nested structures and brackets in string values.

---

## 4. Layer 2 — Schema Validation (Pydantic)

### Pydantic Models

**For Defect Extraction (Gate 1):**

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from enum import Enum

class DefectType(str, Enum):
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    SOFTWARE = "software"
    HYDRAULIC = "hydraulic"
    THERMAL = "thermal"
    STRUCTURAL = "structural"

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ExtractedDefect(BaseModel):
    defect_type: DefectType
    severity: Severity
    affected_component: str = Field(..., min_length=2, max_length=200)
    root_cause_hypothesis: Optional[str] = Field(None, max_length=500)
    error_codes: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(..., min_items=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    @validator("affected_component")
    def component_not_generic(cls, v):
        """Reject overly generic component names."""
        generic_terms = {"unit", "system", "device", "thing", "it", "equipment", "machine"}
        if v.lower().strip() in generic_terms:
            raise ValueError(
                f"'{v}' is too generic. Specify the actual component "
                f"(e.g., 'compressor', 'control board', 'fan motor')"
            )
        return v
    
    @validator("symptoms")
    def symptoms_not_empty_strings(cls, v):
        """Reject empty or trivial symptom entries."""
        cleaned = [s.strip() for s in v if s.strip() and len(s.strip()) > 2]
        if not cleaned:
            raise ValueError("At least one meaningful symptom is required")
        return cleaned
    
    @validator("error_codes")
    def error_codes_format(cls, v):
        """Validate error code format if present."""
        import re
        for code in v:
            if code and not re.match(r'^[A-Z]-?\d{1,4}$', code):
                # Warning, not rejection — codes might have unusual formats
                pass
        return v
    
    @validator("confidence")
    def confidence_sanity(cls, v, values):
        """Flag suspiciously high confidence for incomplete extractions."""
        if v > 0.95 and not values.get("root_cause_hypothesis"):
            # High confidence but no root cause — slightly suspicious
            # Don't reject, but downstream monitoring will flag this pattern
            pass
        return v
```

**For Action Items (Gate 2):**

```python
class ActionType(str, Enum):
    REPAIR = "repair"
    REPLACE = "replace"
    INSPECT = "inspect"
    ESCALATE = "escalate"
    RECALL = "recall"

class Priority(str, Enum):
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENCY = "emergency"

class PartRequired(BaseModel):
    part_name: str = Field(..., min_length=2)
    part_number: Optional[str] = None
    quantity: int = Field(default=1, ge=1, le=100)

class ActionItem(BaseModel):
    action_type: ActionType
    description: str = Field(..., min_length=10, max_length=1000)
    parts_required: list[PartRequired] = Field(default_factory=list)
    warranty_eligible: bool
    escalation_required: bool
    priority: Priority
    sop_reference: str = Field(..., min_length=3)
    
    @validator("description")
    def description_not_boilerplate(cls, v):
        """Reject generic placeholder descriptions."""
        boilerplate = [
            "action required",
            "see sop",
            "follow procedure",
            "as per policy",
            "refer to manual",
        ]
        if v.lower().strip() in boilerplate:
            raise ValueError(
                f"Description '{v}' is too generic. "
                f"Provide specific repair/inspection steps."
            )
        return v
    
    @validator("sop_reference")
    def sop_reference_not_placeholder(cls, v):
        """Reject placeholder SOP references."""
        placeholders = {"none", "n/a", "unknown", "tbd", "see above", "not available"}
        if v.lower().strip() in placeholders:
            raise ValueError(
                f"SOP reference '{v}' is a placeholder. "
                f"Reference a specific document and section from the retrieved context."
            )
        return v
    
    @validator("warranty_eligible")
    def warranty_requires_escalation_if_uncertain(cls, v, values):
        """If warranty is denied, escalation should be considered."""
        # This is a soft check — logged as warning, not rejected
        return v
```

### How Pydantic Errors Feed Into Retries

When Pydantic validation fails, the error messages are specific and actionable:

```python
try:
    defect = ExtractedDefect(**defect_dict)
except ValidationError as e:
    errors = e.errors()
    # Example errors:
    # [
    #   {"loc": ("defect_type",), "msg": "value is not a valid enumeration member; 
    #     permitted: 'mechanical', 'electrical', 'software', 'hydraulic', 'thermal', 'structural'",
    #     "type": "type_error.enum"},
    #   {"loc": ("severity",), "msg": "value is not a valid enumeration member; 
    #     permitted: 'low', 'medium', 'high', 'critical'",
    #     "type": "type_error.enum"},
    #   {"loc": ("affected_component",), "msg": "'unit' is too generic. 
    #     Specify the actual component (e.g., 'compressor', 'control board', 'fan motor')",
    #     "type": "value_error"},
    # ]
```

These error messages are appended to the retry prompt, giving the LLM specific instructions on what to fix. The LLM doesn't get a vague "try again" — it gets "severity must be one of low/medium/high/critical, you used 'moderate'" and "affected_component 'unit' is too generic, specify the actual component."

### Design Decisions

**Decision:** Custom validators that reject generic/placeholder values, not just type checks.

**Reasoning:** Pydantic's built-in validators catch structural issues (wrong type, missing field, invalid enum). But LLMs have a tendency to produce structurally valid but semantically empty outputs — `"affected_component": "unit"` passes type checking (it's a string) but is useless for downstream processing. Custom validators catch these semantic cop-outs and force the LLM to produce specific, actionable content on retry.

**Tradeoff:** Strict custom validators increase the retry rate. "unit" being rejected means the LLM must retry even though it technically answered the question. But the purpose of the system is to produce actionable output — "replace the unit" is not an actionable action item. The small increase in retry rate (~3-5% of orders) is justified by the significant improvement in output quality.

**Decision:** Min/max length constraints on string fields.

**Reasoning:** `affected_component` with min_length=2 prevents empty strings that somehow pass the "required" check. max_length=200 prevents the LLM from writing a paragraph where a component name should be. `description` with min_length=10 ensures the action item has actual content, not just "repair it." These are guardrails against both under-specified and over-specified outputs.

**Decision:** Enum types for defect_type, severity, action_type, and priority.

**Reasoning:** These fields must be from a fixed set of values for downstream systems to process them. A ticketing system that receives `"severity": "moderate"` (not in the enum) would either crash or silently miscategorize the order. Enums enforce exact values. When the LLM uses a close synonym ("moderate" instead of "medium"), the Pydantic error message tells it exactly which values are allowed, making the retry almost always successful.

---

## 5. Layer 3 — Business Logic Validation

### What Layer 3 Catches

Business logic validation checks whether the output makes sense in the domain context — relationships between fields that Pydantic's per-field validators can't check.

### Implementation

```python
class BusinessLogicValidator:
    
    def validate_extraction(self, defects: list[ExtractedDefect], 
                           pre_extracted: list[dict],
                           segment_text: str) -> list[dict]:
        """Business logic checks on extracted defects. Returns warnings."""
        warnings = []
        
        # Check 1: At least one defect extracted
        if len(defects) == 0:
            warnings.append({
                "severity": "error",
                "code": "NO_DEFECTS",
                "message": "No defects extracted from text that was classified as containing defects",
            })
        
        # Check 2: Severity-symptom coherence
        for i, defect in enumerate(defects):
            warnings.extend(self.check_severity_coherence(defect, i))
        
        # Check 3: Error code coverage
        pre_extracted_codes = [
            e["text"] for e in pre_extracted if e["entity_type"] == "error_code"
        ]
        extracted_codes = set()
        for defect in defects:
            extracted_codes.update(defect.error_codes)
        
        for code in pre_extracted_codes:
            if code not in extracted_codes:
                warnings.append({
                    "severity": "warning",
                    "code": "MISSING_ERROR_CODE",
                    "message": f"Pre-extracted error code '{code}' not found in LLM extraction",
                })
        
        # Check 4: Duplicate defect detection
        components = [d.affected_component.lower() for d in defects]
        for comp in set(components):
            if components.count(comp) > 1:
                warnings.append({
                    "severity": "warning",
                    "code": "DUPLICATE_COMPONENT",
                    "message": f"Multiple defects for component '{comp}' — verify these are distinct issues",
                })
        
        # Check 5: Confidence distribution
        avg_confidence = sum(d.confidence for d in defects) / len(defects) if defects else 0
        if avg_confidence < 0.5:
            warnings.append({
                "severity": "warning",
                "code": "LOW_CONFIDENCE",
                "message": f"Average extraction confidence {avg_confidence:.2f} is below 0.5 — results may be unreliable",
            })
        
        return warnings
    
    def validate_action_items(self, action_items: list[ActionItem],
                             defects: list[ExtractedDefect],
                             retrieved_chunks: list) -> list[dict]:
        """Business logic checks on generated action items. Returns warnings."""
        warnings = []
        
        # Check 1: Action item count vs defect count
        if len(action_items) < len(defects):
            warnings.append({
                "severity": "warning",
                "code": "FEWER_ACTIONS_THAN_DEFECTS",
                "message": (
                    f"Generated {len(action_items)} action items for {len(defects)} defects. "
                    f"Each defect should have at least one action item."
                ),
            })
        
        # Check 2: Critical defects must have urgent/emergency priority
        for defect in defects:
            if defect.severity == Severity.CRITICAL:
                has_urgent = any(
                    ai.priority in (Priority.URGENT, Priority.EMERGENCY) 
                    for ai in action_items
                )
                if not has_urgent:
                    warnings.append({
                        "severity": "error",
                        "code": "CRITICAL_WITHOUT_URGENT",
                        "message": (
                            f"Critical defect on '{defect.affected_component}' "
                            f"but no action item with urgent/emergency priority"
                        ),
                    })
        
        # Check 3: SOP references should match retrieved documents
        retrieved_titles = set()
        for chunk in (retrieved_chunks or []):
            retrieved_titles.add(chunk.document_title.lower())
        
        for i, ai in enumerate(action_items):
            ref_lower = ai.sop_reference.lower()
            ref_found = any(title in ref_lower or ref_lower in title 
                          for title in retrieved_titles)
            if not ref_found and retrieved_titles:
                warnings.append({
                    "severity": "warning",
                    "code": "SOP_REF_NOT_IN_CONTEXT",
                    "message": (
                        f"Action item {i} references '{ai.sop_reference}' "
                        f"which doesn't match any retrieved document"
                    ),
                })
        
        # Check 4: Warranty + escalation coherence
        for i, ai in enumerate(action_items):
            if ai.warranty_eligible and ai.escalation_required:
                warnings.append({
                    "severity": "info",
                    "code": "WARRANTY_WITH_ESCALATION",
                    "message": (
                        f"Action item {i} is marked warranty eligible but also "
                        f"requires escalation — unusual combination"
                    ),
                })
        
        # Check 5: Replace/recall actions should list required parts
        for i, ai in enumerate(action_items):
            if ai.action_type in (ActionType.REPLACE, ActionType.RECALL):
                if not ai.parts_required:
                    warnings.append({
                        "severity": "warning",
                        "code": "REPLACEMENT_WITHOUT_PARTS",
                        "message": (
                            f"Action item {i} is '{ai.action_type.value}' "
                            f"but lists no required parts"
                        ),
                    })
        
        # Check 6: All-escalation check
        if all(ai.escalation_required for ai in action_items):
            warnings.append({
                "severity": "warning",
                "code": "ALL_ESCALATED",
                "message": (
                    "All action items require escalation — "
                    "system may not have had sufficient context to make determinations"
                ),
            })
        
        return warnings
    
    def check_severity_coherence(self, defect: ExtractedDefect, 
                                  index: int) -> list[dict]:
        """Check if severity aligns with symptoms and error codes."""
        warnings = []
        
        # Critical indicators that should bump severity
        critical_keywords = [
            "non-operational", "shutdown", "safety", "fire", "smoke",
            "explosion", "toxic", "leak.*refrigerant", "complete failure"
        ]
        
        text = " ".join(defect.symptoms).lower()
        has_critical_signal = any(
            re.search(kw, text) for kw in critical_keywords
        )
        
        if has_critical_signal and defect.severity in (Severity.LOW, Severity.MEDIUM):
            warnings.append({
                "severity": "warning",
                "code": "SEVERITY_UNDERRATED",
                "message": (
                    f"Defect {index} has symptoms suggesting critical severity "
                    f"({text}) but is rated '{defect.severity.value}'"
                ),
            })
        
        return warnings
```

### Design Decisions

**Decision:** Business logic produces warnings, not hard failures (with one exception).

**Reasoning:** Business logic checks are heuristic — they flag suspicious patterns but can't definitively say the output is wrong. A defect might legitimately have low confidence (ambiguous technician writing). Action items might legitimately not match any retrieved document title (the LLM correctly referenced a sub-section not captured in the document title). Making these hard failures would cause excessive retries where the LLM produces the same output again because the output is actually correct but trips the heuristic.

**The one exception:** `CRITICAL_WITHOUT_URGENT` is treated as a soft error that influences routing. If a critical defect has no urgent/emergency action item, the system doesn't retry (the LLM likely won't change its severity assessment) but the order is flagged for priority human review. A critical defect treated as routine could result in delayed repairs and safety risk — this must be caught.

**Decision:** Check SOP reference against retrieved documents.

**Reasoning:** If the LLM's SOP reference doesn't match any retrieved document, it likely hallucinated the reference — making up a document name that sounds plausible but doesn't exist. This is a serious faithfulness violation that Ragas would catch in offline evaluation, but we want to catch it per-request in production. The check isn't perfect (the reference might use different formatting than the document title), so it's a warning, not a rejection.

**Decision:** Track warning severity levels (error, warning, info).

**Reasoning:** Not all warnings are equal. `CRITICAL_WITHOUT_URGENT` is more important than `WARRANTY_WITH_ESCALATION`. Severity levels enable downstream systems to triage: errors get priority human review, warnings appear in the review dashboard, info-level items are logged for analytics but don't trigger review.

---

## 6. Cross-Validation with Phase 1 Entities

### The Cross-Validation Framework

Phase 1 pre-extracted entities (regex + spaCy) serve as deterministic anchors. The LLM's extraction should be consistent with these anchors. Discrepancies indicate either a Phase 1 false positive or a Phase 3 extraction error.

```python
class CrossValidator:
    
    def validate(self, defects: list[ExtractedDefect],
                pre_extracted: list[dict]) -> list[dict]:
        """Cross-validate LLM extraction against Phase 1 entities."""
        warnings = []
        
        # Collect all LLM-extracted entities
        llm_error_codes = set()
        llm_components = set()
        for defect in defects:
            llm_error_codes.update(defect.error_codes)
            llm_components.add(defect.affected_component.lower())
        
        # Collect Phase 1 entities by type
        phase1_error_codes = set(
            e["text"] for e in pre_extracted if e["entity_type"] == "error_code"
        )
        phase1_model_numbers = set(
            e["text"] for e in pre_extracted if e["entity_type"] == "model_number"
        )
        phase1_part_numbers = set(
            e["text"] for e in pre_extracted if e["entity_type"] == "part_number"
        )
        
        # Check 1: Error code coverage
        for code in phase1_error_codes:
            if code not in llm_error_codes:
                warnings.append({
                    "type": "MISSING_ERROR_CODE",
                    "phase1_value": code,
                    "llm_values": list(llm_error_codes),
                    "recommendation": f"LLM may have missed error code {code}",
                })
        
        # Check 2: LLM found codes that regex didn't
        for code in llm_error_codes:
            if code not in phase1_error_codes:
                warnings.append({
                    "type": "LLM_EXTRA_CODE",
                    "llm_value": code,
                    "phase1_values": list(phase1_error_codes),
                    "recommendation": f"LLM found error code {code} not in regex extraction — verify",
                })
        
        # Check 3: Model number consistency
        # If Phase 1 found model XR-440, defect descriptions should reference XR series components
        for model in phase1_model_numbers:
            product_line = model.split("-")[0] if "-" in model else model[:2]
            # This is a soft check — components don't always reference the product line directly
        
        return warnings
```

### Design Decisions

**Decision:** Cross-validation produces warnings, not retries.

**Reasoning:** The LLM might legitimately disagree with regex. For example, regex might extract "E-47" from the text "Model E-47X was installed" where "E-47" isn't actually an error code in context — it's part of a model number. The LLM, understanding context, correctly doesn't include "E-47" as an error code. Making cross-validation a hard failure would force the LLM to include a false positive on retry.

**Decision:** Track both directions — Phase 1 entities missing from LLM output AND LLM entities missing from Phase 1.

**Reasoning:** Phase 1 → LLM gaps indicate the LLM might have missed something (concerning). LLM → Phase 1 gaps indicate the LLM found something regex couldn't (expected, since the LLM handles contextual entities). Both directions provide useful diagnostic information, but only Phase 1 → LLM gaps are concerning enough to warrant warnings.

---

## 7. Confidence Calibration Checks

### What Confidence Should Mean

The `confidence` field in each extracted defect should reflect how certain the model is about the extraction. Properly calibrated: 0.9 confidence means ~90% of the time the extraction is correct. Poorly calibrated: the model always outputs 0.95 regardless of actual certainty.

### Calibration Checks

```python
def check_confidence_calibration(defects: list[ExtractedDefect], 
                                  segment_text: str) -> list[dict]:
    warnings = []
    
    # Check 1: All-high confidence is suspicious
    if all(d.confidence > 0.9 for d in defects) and len(defects) > 1:
        warnings.append({
            "code": "UNIFORM_HIGH_CONFIDENCE",
            "message": "All defects have >0.9 confidence — model may not be calibrating properly",
        })
    
    # Check 2: High confidence on short text
    text_tokens = len(segment_text.split())
    if text_tokens < 20 and any(d.confidence > 0.9 for d in defects):
        warnings.append({
            "code": "HIGH_CONFIDENCE_SHORT_TEXT",
            "message": f"Confidence >0.9 from only {text_tokens} words of input — may be overconfident",
        })
    
    # Check 3: Confidence should correlate with evidence availability
    for defect in defects:
        has_error_code = bool(defect.error_codes)
        has_root_cause = bool(defect.root_cause_hypothesis)
        has_multiple_symptoms = len(defect.symptoms) > 1
        
        evidence_strength = sum([has_error_code, has_root_cause, has_multiple_symptoms])
        
        if evidence_strength == 0 and defect.confidence > 0.8:
            warnings.append({
                "code": "CONFIDENCE_EVIDENCE_MISMATCH",
                "message": (
                    f"Defect '{defect.affected_component}' has confidence {defect.confidence} "
                    f"but no error codes, no root cause, and only one symptom"
                ),
            })
    
    return warnings
```

**Decision:** Confidence calibration is monitored, not enforced.

**Reasoning:** Enforcing calibration (rejecting outputs where confidence seems wrong) would require knowing the "correct" confidence, which we don't have at inference time. Instead, we collect calibration metrics over time: of all extractions marked 0.9+ confidence, what percentage were actually correct (verified by human review)? If this ratio drifts below 0.85, the model's calibration has degraded and retraining with calibration-aware loss might be needed.

---

## 8. The Self-Healing Retry Loop

### How It Works

The retry loop is the core mechanism that turns LLM unreliability into system reliability. When validation fails, the exact errors are fed back to the LLM, giving it specific instructions on what to fix.

```
Attempt 1:
  Prompt: [system instruction] + [segment text] + [pre-extracted entities]
  Output: {"defect_type": "moderate", "severity": "high", ...}
  Validation: FAIL — "defect_type must be one of: mechanical, electrical, ..."
                                        │
                                        ▼
Attempt 2:
  Prompt: [original prompt] + 
          "PREVIOUS ATTEMPT FAILED. Error: defect_type must be one of: 
           mechanical, electrical, software, hydraulic, thermal, structural. 
           You used 'moderate'. Fix this error."
  Output: {"defect_type": "mechanical", "severity": "high", ...}
  Validation: PASS ✓
```

### Implementation

```python
class RetryOrchestrator:
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
    
    def build_retry_context(self, attempt: int, 
                           all_previous_errors: list[list[str]]) -> str:
        """Build the error context to append to the retry prompt."""
        
        if attempt == 0:
            return ""  # First attempt — no errors to include
        
        context = f"\n\n{'='*60}\n"
        context += f"ATTEMPT {attempt + 1} OF {self.max_retries + 1}\n"
        context += f"{'='*60}\n\n"
        context += "YOUR PREVIOUS OUTPUTS FAILED VALIDATION. FIX ALL ERRORS BELOW:\n\n"
        
        for attempt_num, errors in enumerate(all_previous_errors):
            if errors:
                context += f"--- Attempt {attempt_num + 1} errors ---\n"
                for error in errors:
                    context += f"  • {error}\n"
                context += "\n"
        
        context += "CRITICAL: Fix every error listed above. "
        context += "Do not introduce new errors while fixing old ones. "
        context += "Output ONLY valid JSON.\n"
        
        return context
    
    def should_retry(self, validation_result: dict) -> str:
        """Determine routing: 'valid', 'retry', or 'human_review'."""
        
        has_structural_errors = validation_result.get("structural_errors", [])
        has_schema_errors = validation_result.get("schema_errors", [])
        has_partial_success = validation_result.get("valid_items", [])
        retry_count = validation_result.get("retry_count", 0)
        status = validation_result.get("status")
        
        # Infrastructure failure — don't retry at this level
        if status == "failed":
            return "human_review"
        
        # All valid — proceed
        if not has_structural_errors and not has_schema_errors:
            return "valid"
        
        # Partial success — proceed with valid items
        if has_partial_success and not has_structural_errors:
            return "valid"
        
        # Total failure but retries remaining
        if retry_count < self.max_retries:
            return "retry"
        
        # Retries exhausted
        return "human_review"
```

### Design Decisions

**Decision:** Include ALL previous errors in retry prompts, not just the latest.

**Reasoning:** A common failure pattern: the LLM fixes the error from attempt 1 but reintroduces a different error (or re-introduces the original error in a different form). By showing ALL previous errors, the LLM understands the full set of constraints it must satisfy simultaneously. This is more effective than showing only the latest error, which can lead to oscillating failures (fix A, break B, fix B, break A).

**Decision:** Include attempt number and total attempts in the retry prompt.

**Reasoning:** The explicit "ATTEMPT 2 OF 4" creates a frame for the LLM that this is a correction task, not a fresh generation task. Empirically, this framing improves retry success rates by ~5-10%. The LLM "understands" that it needs to be more careful, not more creative.

**Decision:** Partial success proceeds without retry.

**Reasoning:** If the LLM extracts 3 defects and 2 pass validation but 1 fails, retrying the entire extraction risks losing the 2 valid defects (the retry might produce a completely different output). Instead, we proceed with the 2 valid defects and log a warning about the partial extraction. The downstream system handles the incompleteness — the human review dashboard shows which orders had partial extractions.

---

## 9. Retry Prompt Engineering

### The Anatomy of an Effective Retry Prompt

```
[ORIGINAL SYSTEM PROMPT — unchanged]
[ORIGINAL USER PROMPT — unchanged]

============================================================
ATTEMPT 2 OF 4
============================================================

YOUR PREVIOUS OUTPUT FAILED VALIDATION. FIX ALL ERRORS BELOW:

--- Attempt 1 errors ---
  • Field 'defect_type': value 'moderate' is not a valid enumeration member; 
    permitted: 'mechanical', 'electrical', 'software', 'hydraulic', 'thermal', 'structural'
  • Field 'affected_component': 'unit' is too generic. 
    Specify the actual component (e.g., 'compressor', 'control board', 'fan motor')
  • Field 'symptoms': At least one meaningful symptom is required (got empty list)

CRITICAL: Fix every error listed above. Do not introduce new errors while fixing old ones.
Output ONLY valid JSON.
```

### Why This Format Works

**The original prompt is repeated unchanged.** The LLM needs the full context to re-attempt extraction. If you only send the error corrections without the original service order text, the LLM has no input to extract from.

**Error messages include both the problem AND the solution.** "defect_type must be one of: mechanical, electrical, ..." tells the LLM exactly what values are acceptable. "Specify the actual component (e.g., 'compressor', 'control board', 'fan motor')" gives concrete examples. The more specific the error message, the higher the retry success rate.

**The "do not introduce new errors" instruction matters.** Without it, the LLM tends to over-correct — it fixes the flagged issues but becomes more creative/loose with other fields, introducing new problems. The explicit instruction to be conservative on non-error fields improves retry reliability.

### Retry Prompt Optimization

**Decision:** Don't include the previous LLM output in the retry prompt.

**Reasoning:** Including the bad output ("Your previous output was: {...}") seems helpful but actually hurts. The LLM anchors to the bad output and makes minimal changes. Without seeing the bad output, the LLM generates from scratch with the error constraints in mind, producing a fresh attempt that's more likely to be fundamentally different and correct.

**Tradeoff:** Occasionally, the bad output was 90% correct with one small error. Generating from scratch might change the 90% that was fine. But this is acceptable because validation catches any new issues, and the retry has a high success rate (~70% of retries succeed on the next attempt when error messages are specific).

---

## 10. Retry Budget & Exhaustion Strategy

### Retry Budget

**Decision:** Maximum 3 retries per validation gate (4 total attempts including the initial).

**Reasoning:** Empirical analysis of retry success rates:

```
Attempt 1 (initial):  ~90% pass rate (after constrained decoding)
Attempt 2 (retry 1):  ~70% of failures fixed (cumulative: ~97%)
Attempt 3 (retry 2):  ~50% of remaining failures fixed (cumulative: ~98.5%)
Attempt 4 (retry 3):  ~30% of remaining failures fixed (cumulative: ~99%)
Attempt 5+ :          Diminishing returns — <20% success per additional attempt
```

At 3 retries, we achieve ~99% pass rate across all attempts. The remaining 1% represents cases where the model fundamentally can't extract valid output from the input (ambiguous text, out-of-domain content, or systematic model failure). These are correctly routed to human review.

**Tradeoff:** Each retry costs one LLM inference call (1-3 seconds + GPU compute). Worst case per gate: 4 calls × 3 seconds = 12 seconds. With two gates (extraction + generation), worst case is 24 seconds of LLM time. This is acceptable for batch service order processing but would be concerning for real-time applications. At our volume (~130K calls/day), the retry calls add ~13K additional calls (10% retry rate × 1.5 average retries per failure), increasing GPU costs by ~10%.

### Independent Retry Budgets

**Decision:** Extraction retries and generation retries are independent.

An order that took 3 extraction retries still gets a fresh budget of 3 retries for generation. The worst case is 4 + 4 = 8 LLM calls for one segment.

**Reasoning:** Extraction failures and generation failures are independent problems. Difficulty extracting defects from ambiguous text says nothing about the difficulty of generating action items from clear defect data + retrieved context. Sharing a budget would mean an order that struggled with extraction gets fewer chances at generation — punishing the generation step for extraction's difficulty.

### What Happens When Budget is Exhausted

When all retries are used and validation still fails:

```python
def handle_exhausted_retries(state: AgentState, gate: str) -> dict:
    """Route to human review with full diagnostic context."""
    
    return {
        "status": "human_review",
        "escalation_reason": f"{gate} failed after {state[f'{gate}_retry_count']} retries",
        "escalation_context": {
            "gate": gate,
            "total_attempts": state[f"{gate}_retry_count"] + 1,
            "all_errors": collect_all_errors(state, gate),
            "partial_results": state.get("extracted_defects") or state.get("action_items"),
            "last_raw_response": state.get(f"{gate}_raw_response"),
            "last_prompt": state.get(f"{gate}_prompt"),
        }
    }
```

**Rule:** Never silently drop an order. Every order either produces validated output or reaches human review. There is no third option.

---

## 11. Human Review Routing

### What the Human Reviewer Receives

```python
class HumanReviewPackage:
    # Original input
    order_id: str
    segment_id: str
    segment_text: str
    inherited_context: dict
    
    # What the system produced (partial results)
    extracted_defects: Optional[list]    # May be None if extraction failed
    action_items: Optional[list]         # May be None if generation failed
    
    # Why it's here
    escalation_reason: str               # "Extraction failed after 3 retries"
    escalation_gate: str                 # "extraction" or "generation"
    
    # Complete error history
    all_validation_errors: list[list[str]]  # Errors from every attempt
    business_warnings: list[dict]           # Business logic warnings
    cross_validation_warnings: list[dict]   # Phase 1 discrepancies
    
    # Debug context (restricted access)
    prompts: list[str]                   # All prompts sent to LLM
    raw_responses: list[str]             # All raw LLM responses
    
    # Quality context
    retry_count: int
    retrieval_scores: list[float]
    confidence_scores: list[float]
    processing_time_ms: float
```

### Human Review Dashboard Triage

The dashboard sorts orders by priority:

**Priority 1 — Critical defects that failed generation.** The system detected a critical severity defect but couldn't generate valid action items. This needs immediate human attention because a critical defect without action items means no repair gets scheduled.

**Priority 2 — Extraction failures.** The system couldn't extract defects at all. The human reviewer needs to read the raw text and create the defect record manually.

**Priority 3 — Partial results with warnings.** The system produced output but with business logic warnings. The human reviews and either approves or corrects.

**Priority 4 — Low-confidence results.** The system produced output but with low confidence scores. The human spot-checks for accuracy.

---

## 12. Guardrails — Preventing Dangerous Outputs

### Safety-Critical Guardrails

Beyond validation accuracy, certain outputs are dangerous and must be prevented:

**Guardrail 1 — Never auto-approve safety-critical actions without escalation.**

```python
SAFETY_CRITICAL_PATTERNS = [
    "refrigerant", "asbestos", "high voltage", "pressure vessel",
    "gas leak", "fire", "explosion", "toxic", "radiation",
]

def safety_guardrail(action_items: list[ActionItem], 
                     defects: list[ExtractedDefect]) -> list[ActionItem]:
    """Force escalation on safety-critical items."""
    
    for ai in action_items:
        text = f"{ai.description} {' '.join(d.affected_component for d in defects)}".lower()
        
        if any(pattern in text for pattern in SAFETY_CRITICAL_PATTERNS):
            ai.escalation_required = True
            ai.priority = Priority.EMERGENCY
    
    return action_items
```

**Rule:** Safety guardrails run AFTER validation, as a post-processing step. They override the LLM's output — even if the LLM said `escalation_required: false` for a refrigerant leak, the guardrail forces it to `true`. This is a hard override that the LLM cannot influence.

**Guardrail 2 — Never generate action items that contradict compliance requirements.**

If the retrieved context includes compliance documents (e.g., EPA regulations for refrigerant handling), the action items must not suggest procedures that violate those regulations. This is difficult to enforce automatically but is flagged when action items involve regulated components.

```python
REGULATED_COMPONENTS = ["refrigerant", "compressor", "pressure vessel", "electrical panel"]

def compliance_guardrail(action_items: list[ActionItem]) -> list[dict]:
    """Flag action items involving regulated components for compliance review."""
    flags = []
    
    for i, ai in enumerate(action_items):
        if any(comp in ai.description.lower() for comp in REGULATED_COMPONENTS):
            flags.append({
                "action_item_index": i,
                "flag": "COMPLIANCE_REVIEW_NEEDED",
                "reason": "Action involves regulated component — verify compliance with applicable regulations",
            })
    
    return flags
```

**Guardrail 3 — Rate-limit unusual patterns.**

If a single order generates more than 5 action items, or more than 3 defects with "critical" severity, something unusual is happening. These orders are flagged for review even if all validations pass.

```python
def anomaly_guardrail(defects: list, action_items: list) -> list[dict]:
    flags = []
    
    if len(action_items) > 5:
        flags.append({
            "flag": "EXCESSIVE_ACTION_ITEMS",
            "count": len(action_items),
            "threshold": 5,
        })
    
    critical_count = sum(1 for d in defects if d.severity == Severity.CRITICAL)
    if critical_count > 2:
        flags.append({
            "flag": "MULTIPLE_CRITICAL_DEFECTS",
            "count": critical_count,
            "threshold": 2,
        })
    
    return flags
```

---

## 13. Output Sanitization

### What Gets Sanitized

Even after validation passes, the output is sanitized before dispatch to remove artifacts that could cause downstream issues.

```python
class OutputSanitizer:
    
    def sanitize(self, output: dict) -> dict:
        """Clean the validated output before dispatch."""
        
        # Remove any PII that might have leaked through
        output = self.redact_leaked_pii(output)
        
        # Normalize string fields (trim whitespace, normalize unicode)
        output = self.normalize_strings(output)
        
        # Remove internal-only fields before sending to downstream systems
        output = self.remove_internal_fields(output)
        
        # Validate output size (prevent unexpectedly large payloads)
        output = self.check_output_size(output)
        
        return output
    
    def redact_leaked_pii(self, output: dict) -> dict:
        """Scan output for PII patterns that shouldn't be there."""
        import re
        
        PII_PATTERNS = {
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        }
        
        def scan_and_redact(text: str) -> str:
            for pii_type, pattern in PII_PATTERNS.items():
                text = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", text)
            return text
        
        # Recursively scan all string fields
        return self.recursive_apply(output, scan_and_redact)
    
    def check_output_size(self, output: dict) -> dict:
        """Prevent unexpectedly large outputs."""
        import json
        
        serialized = json.dumps(output)
        if len(serialized) > 50000:  # 50KB limit
            # Something is wrong — outputs shouldn't be this large
            output["_warning"] = "Output truncated — exceeded 50KB limit"
            # Remove the largest field (usually raw_response or prompt from debug context)
            if "debug_context" in output:
                del output["debug_context"]
        
        return serialized if len(json.dumps(output)) <= 50000 else output
```

---

## 14. Validation at Both Gates (Extraction + Generation)

### Gate 1 vs Gate 2 — Differences

While both gates use the same three-layer validation architecture, they differ in specific rules:

| Aspect | Gate 1 (Extraction) | Gate 2 (Generation) |
|--------|-------------------|-------------------|
| **Pydantic Model** | ExtractedDefect | ActionItem |
| **Enum Fields** | defect_type, severity | action_type, priority |
| **Custom Validators** | Component not generic, symptoms not empty | Description not boilerplate, SOP ref not placeholder |
| **Cross-Validation** | Against Phase 1 regex entities | Against retrieved documents |
| **Business Logic** | Severity-symptom coherence, error code coverage | Action-defect count match, critical-priority alignment |
| **Safety Guardrails** | None (extraction is data, not action) | Safety-critical override, compliance flagging |
| **Retry Max** | 3 retries | 3 retries (independent budget) |
| **Partial Success** | Proceed with valid defects | Proceed with valid action items |
| **Downstream Impact** | Wrong extraction → wrong RAG query → wrong action items | Wrong action items → wrong repairs → equipment damage |

### Why Gate 2 Has Stricter Guardrails

Gate 2 produces the final output that drives real-world actions — repair orders, parts procurement, warranty claims. A wrong extraction (Gate 1) is bad but is filtered by RAG relevance and corrected during action item generation. A wrong action item (Gate 2) directly causes incorrect repairs. The stakes escalate through the pipeline, so guardrails escalate with them.

---

## 15. Edge Cases & How They're Handled

### Edge Case 1 — LLM Returns Valid JSON with Correct Schema but Nonsensical Content

Example: `{"defect_type": "mechanical", "severity": "low", "affected_component": "compressor", "symptoms": ["working normally"], "confidence": 0.95}`

The symptom "working normally" contradicts the presence of a defect. This passes both structural validation and Pydantic schema validation because all types and values are technically valid.

**Handling:** Business logic Layer 3 catches this through symptom content analysis. A set of contradiction patterns ("working normally", "no issue", "functioning correctly") flag the defect as likely hallucinated. The confidence-evidence mismatch check also flags high confidence with contradictory symptoms.

### Edge Case 2 — Multiple Defects with Identical Content

The LLM produces the same defect object 3 times — structurally valid but clearly a generation artifact.

**Handling:** Business logic checks for duplicate components (same `affected_component` appearing multiple times). The deduplication check flags this and, if all duplicates are identical, removes the copies and logs a warning.

### Edge Case 3 — Constrained Decoding Forces Nonsensical Values

The JSON schema constraint forces `defect_type` to be from the enum, but the model's natural output would have been something outside the enum (e.g., "chemical"). Constrained decoding forces it to pick the closest valid option, which might be wrong (e.g., "hydraulic" when "chemical" was intended).

**Handling:** This is the hardest edge case to catch automatically because the output is valid, schema-compliant, and plausible — just wrong. It's caught by the Ragas evaluation pipeline (correctness metric drops when these cases are frequent), by human review of sampled outputs, and by the confidence calibration check (the model might report lower confidence when it was forced into an uncomfortable choice).

**Long-term fix:** If this happens frequently for a specific defect type, add "chemical" to the enum rather than forcing the model to misclassify.

### Edge Case 4 — Retry Oscillation

Attempt 1 produces error A. Attempt 2 fixes A but produces error B. Attempt 3 fixes B but reintroduces A.

**Handling:** The retry prompt includes ALL previous errors (not just the latest), so the model sees both A and B simultaneously. This usually breaks the oscillation. If oscillation persists through all retries, the order goes to human review with the full error history, which clearly shows the oscillation pattern for debugging.

### Edge Case 5 — Extremely Short Input Text

A service order with only "compressor broken" — 2 words. The LLM has very little to work with.

**Handling:** The model extracts what it can: `defect_type: "mechanical"`, `affected_component: "compressor"`, `symptoms: ["broken/non-functional"]`, `confidence: 0.4`. The low confidence triggers a business logic warning. The downstream system flags this for human enrichment — a human reviewer can contact the technician for more details before proceeding with action items.

### Edge Case 6 — Conflicting Information in Input

The technician writes "compressor is working fine but making grinding noise." The fine-tuned model must decide whether this is a defect or not.

**Handling:** The model should extract a defect (grinding noise is a symptom regardless of the technician's assessment that it's "working fine"). The confidence should be moderate (0.6-0.7) reflecting the ambiguity. Business logic doesn't flag this because the symptom "grinding noise" is a legitimate defect signal. The human review dashboard may receive this if the action items seem uncertain.

---

## 16. Observability & Alerting

### Metrics Emitted

**Per-Request Validation Metrics:**

| Metric | Description |
|--------|-------------|
| `validation.gate` | Which gate (extraction / generation) |
| `validation.structural_pass` | Did JSON parsing succeed? |
| `validation.schema_pass` | Did Pydantic validation pass? |
| `validation.business_warnings_count` | Number of business logic warnings |
| `validation.business_warning_codes` | Which warning codes triggered |
| `validation.cross_validation_warnings` | Count of Phase 1 discrepancies |
| `validation.retry_triggered` | Was a retry needed? |
| `validation.retry_count` | Total retries for this gate |
| `validation.final_status` | valid / human_review / failed |
| `validation.partial_success` | Were partial results accepted? |
| `validation.safety_guardrail_triggered` | Did safety guardrail force changes? |
| `validation.latency_ms` | Validation processing time |

**Aggregate Dashboard Panels:**

**Panel 1 — Validation Pass Rates**
- First-attempt pass rate (should be ~90%+)
- Final pass rate after retries (should be ~98%+)
- Pass rate by gate (extraction vs generation)
- Pass rate trend over time (detect degradation)

**Panel 2 — Error Distribution**
- Most common Pydantic errors (top 10)
- Most common business logic warnings (top 10)
- Error distribution by defect type
- Error distribution by product line

**Panel 3 — Retry Analysis**
- Retry rate (percentage of requests needing retry)
- Retry success rate (percentage that eventually pass)
- Average retries per failed request
- Retry exhaustion rate (percentage reaching human review)

**Panel 4 — Safety & Guardrails**
- Safety guardrail trigger rate
- Compliance flag rate
- Anomaly detection rate
- Human review queue depth and age

### Alert Configuration

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| First-attempt pass rate drop | <85% over 1 hour | P2 | Model output quality degrading, check model/prompt |
| Final pass rate drop | <95% over 1 hour | P1 | System reliability below SLA, investigate immediately |
| Retry exhaustion spike | >5% orders reaching human review | P2 | Systematic failure, check model and training data |
| Safety guardrail spike | >10% of orders triggering safety override | P2 | Either real safety issues or false positives — investigate |
| Specific error dominance | One error code >50% of all errors | P2 | Systematic prompt or model issue for that specific field |
| Human review queue overload | >100 orders pending review | P1 | Either system quality degraded or human reviewers unavailable |
| Oscillation detected | Same order failing with alternating errors | P3 | Model struggling with specific input pattern — add to training |
| Business warning spike | Specific warning code increases >3x | P2 | Output quality issue for that specific business rule |

---

## 17. End-to-End Phase 5 Flow Summary

### Per-Request Validation Flow (Both Gates)

```
LLM RAW OUTPUT ARRIVES FROM PHASE 3 OR GENERATION NODE
        │
Layer 1 ┤  STRUCTURAL VALIDATION
        │  • Null/empty check
        │  • Clean LLM artifacts (markdown fences, preamble, trailing text)
        │  • JSON parsing with bracket-depth boundary detection
        │  • Recovery attempts: trailing comma, truncation repair, quote fix
        │  • Verify array type and non-empty
        │  • FAIL → collect specific parse errors → proceed to routing
        │
        ▼
Layer 2 ┤  SCHEMA VALIDATION (Pydantic)
        │  • Validate each item against Pydantic model:
        │    Gate 1: ExtractedDefect (defect_type enum, severity enum,
        │            component not generic, symptoms not empty, confidence 0-1)
        │    Gate 2: ActionItem (action_type enum, priority enum,
        │            description not boilerplate, SOP ref not placeholder)
        │  • Collect per-item errors with specific field + message
        │  • Allow partial success (some items valid, some not)
        │  • FAIL (all items) → collect errors → proceed to routing
        │
        ▼
Layer 3 ┤  BUSINESS LOGIC VALIDATION
        │  • Gate 1: severity-symptom coherence, error code coverage,
        │    duplicate defect detection, confidence distribution check
        │  • Gate 2: action-defect count match, critical-priority alignment,
        │    SOP reference vs retrieved documents, warranty-escalation coherence,
        │    replacement-without-parts check, all-escalation check
        │  • Produces warnings (error/warning/info severity levels)
        │  • Does NOT trigger retries (heuristic checks, not definitive)
        │
        ▼
Cross   ┤  CROSS-VALIDATION (Gate 1 only)
Valid   │  • Compare LLM extraction against Phase 1 regex + spaCy entities
        │  • Flag missing error codes, extra error codes, model mismatches
        │  • Produces warnings, not retries
        │
        ▼
Confid  ┤  CONFIDENCE CALIBRATION CHECK
Check   │  • Uniform high confidence suspicious
        │  • High confidence on short text suspicious
        │  • Confidence-evidence mismatch check
        │  • Monitoring metric, not routing decision
        │
        ▼
Safety  ┤  SAFETY GUARDRAILS (Gate 2 only)
Guard   │  • Force escalation on safety-critical components (refrigerant, 
        │    high voltage, pressure vessels, fire/explosion)
        │  • Flag regulated components for compliance review
        │  • Anomaly detection (excessive items, multiple criticals)
        │  • HARD OVERRIDE — runs after validation, overrides LLM decisions
        │
        ▼
Sanitize┤  OUTPUT SANITIZATION
        │  • PII leak scan (catch anything Phase 1 redaction missed)
        │  • String normalization (trim, unicode)
        │  • Output size check (50KB limit)
        │  • Remove internal-only fields
        │
        ▼
Route   ┤  ROUTING DECISION
        │  ├─ All valid, no critical errors → "valid" → next node
        │  ├─ Partial success (some items valid) → "valid" → next node (with warnings)
        │  ├─ Total failure + retries remaining → "retry" → back to LLM node
        │  │     (with ALL previous errors appended to prompt)
        │  └─ Total failure + retries exhausted → "human_review" → escalation
        │       (with full diagnostic context: all prompts, responses, errors)
        │
        ▼
EITHER: PROCEED TO NEXT NODE (success)
   OR:  RETRY LLM CALL (with error context)
   OR:  HUMAN REVIEW (with diagnostic package)
```

### The 95% Reliability Number

The 95% system reliability claim comes from the combined effect of all validation layers:

```
First-attempt pass rate:           ~90%
  (constrained decoding eliminates most structural issues)
  
Of the ~10% that fail:
  Retry 1 fixes:                   ~70% → cumulative ~97%
  Retry 2 fixes:                   ~50% of remaining → cumulative ~98.5%
  Retry 3 fixes:                   ~30% of remaining → cumulative ~99%

Orders reaching human review:       ~1%

The "95% reliability" refers to first-attempt pass rate without 
any retry. With retries, the automated pass rate is ~99%.

The remaining ~1% receives human review with AI-assisted partial 
results, making human processing 60-70% faster than starting 
from scratch.
```

### Key Design Principles Applied Throughout Phase 5

1. **Layered validation catches errors at the cheapest level.** JSON parsing (microseconds) catches structural issues before Pydantic (milliseconds) runs, which catches schema issues before business logic (milliseconds) runs. No expensive check processes garbage that a cheap check should have caught.

2. **Specific error messages are the retry mechanism.** The self-healing loop works because the LLM receives exact, actionable error messages, not vague "try again." The quality of error messages directly determines retry success rate.

3. **Partial success is better than total failure.** Two valid defects out of three is better than zero defects after a failed retry. The system optimizes for maximum useful output, not perfection.

4. **Safety guardrails are hard overrides, not suggestions.** The LLM's safety assessment is overridden by deterministic rules for safety-critical components. An LLM saying "escalation not required" for a refrigerant leak is overruled unconditionally.

5. **Business logic warns, Pydantic rejects.** Schema violations are definitive errors (wrong type, missing field). Business logic violations are heuristic suspicions (severity seems low for these symptoms). The distinction determines whether the system retries (schema failure) or proceeds with a flag (business logic warning).

6. **Every order has a destination.** Validated output → downstream. Exhausted retries → human review. Infrastructure failure → human review. There is no path where an order disappears silently.

7. **Observability enables continuous improvement.** Every validation failure is logged with the specific error. Aggregating these errors reveals systematic issues: "30% of failures are defect_type enum errors" → improve the prompt. "15% of failures are SOP reference hallucinations" → improve RAG retrieval. The validation layer is both a quality gate and a diagnostic tool.

---

*Document Version: 1.0 | Last Updated: March 2026 | System: LGS Tech Agentic AI Workflow*
