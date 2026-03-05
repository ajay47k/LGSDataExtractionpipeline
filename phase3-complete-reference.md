# Phase 3 — Fine-Tuned LLM (Core Reasoning): Complete Production Reference

## LGS Tech Agentic AI Workflow — Service Order Processing

---

## Table of Contents

1. Phase 3 Overview & Why Fine-Tuning
2. Why Llama 3.1 8B as the Base Model
3. Why QLoRA Over Other Fine-Tuning Methods
4. Training Data Preparation
5. Data Labeling & Annotation Pipeline
6. Training Configuration & Hyperparameters
7. Training Infrastructure & Execution
8. Model Evaluation & Benchmarking
9. Inference Architecture (vLLM)
10. Prompt Engineering for Extraction
11. Structured Output & JSON Mode
12. Cross-Validation with Phase 1 Entities
13. Model Versioning & Registry
14. Retraining Strategy & Drift Detection
15. Observability & Alerting
16. End-to-End Phase 3 Flow Summary

---

## 1. Phase 3 Overview & Why Fine-Tuning

### What Phase 3 Does

Phase 3 is the core reasoning engine. It takes preprocessed, segmented service order text from Phase 1 (via the LangGraph agent in Phase 2) and extracts structured defect information as JSON — defect type, severity, affected component, root cause hypothesis, error codes, and symptoms. This is the step that transforms "compressor making grinding noise, noticed after last maintenance, E-47 showing on panel" into a structured, machine-actionable defect record that can drive RAG retrieval and action item generation downstream.

### Why Fine-Tuning is Necessary

A general-purpose LLM (GPT-4, Claude, base Llama) can do basic information extraction. But it fails in this domain for three specific reasons.

**Reason 1 — Domain-specific vocabulary.**

LGS Tech's service orders contain proprietary terminology that doesn't appear in public training data. Defect codes like "E-47" have company-specific meanings. Part numbers like "P/N 7832-A" reference internal catalogs. Failure mode descriptions like "stage-3 condenser fouling with sub-critical refrigerant bypass" use technical jargon that a general model either misinterprets or ignores. A general model might extract "condenser fouling" as the defect but miss that "stage-3" and "sub-critical refrigerant bypass" are critical qualifiers that determine the severity and repair procedure.

**Reason 2 — Extraction schema adherence.**

General models are good at following instructions but inconsistent at producing structured output that exactly matches a required schema. You might get `"severity": "moderate"` when the schema requires `"severity": "medium"`, or the model might output a narrative paragraph instead of JSON. Fine-tuning on hundreds of examples where the correct output is always the exact schema format trains the model to reliably produce schema-compliant output, reducing the retry rate in Phase 2's validation loop.

**Reason 3 — Implicit domain reasoning.**

Some extraction tasks require domain knowledge that isn't stated in the text. If a technician writes "unit is 18 months old, E-47 on panel," a domain-trained model knows that E-47 is a compressor-related error code and that 18 months is within the standard warranty period — information that affects both the defect classification and the downstream warranty determination. A general model treats E-47 as an opaque string. A fine-tuned model has learned the relationships between error codes, components, and failure modes from hundreds of labeled examples.

### What Fine-Tuning Does NOT Solve

Fine-tuning teaches the model domain vocabulary, extraction patterns, and implicit reasoning. It does NOT teach the model current warranty policies or current SOPs — these change frequently and must come from RAG retrieval in Phase 4. Fine-tuning provides the reasoning capability; RAG provides the current knowledge. This separation is intentional and critical.

---

## 2. Why Llama 3.1 8B as the Base Model

### Alternatives Considered

**GPT-4 / GPT-4o (OpenAI API)**

Highest capability, excellent instruction following, native JSON mode. But cannot be fine-tuned with QLoRA (it's a closed model). OpenAI does offer fine-tuning, but you're sending proprietary technical logs to a third-party API — a significant data privacy and compliance concern for enterprise customers. Also, API dependency means you're subject to rate limits, pricing changes, and service outages you can't control.

**Rejected because:** Data privacy concerns (proprietary logs leave your infrastructure), no QLoRA/LoRA support, API dependency, unpredictable costs at scale.

**Claude / Anthropic API**

Similar capability to GPT-4, strong at structured extraction. Same fundamental issues: closed model, data privacy, API dependency. Anthropic doesn't offer fine-tuning.

**Rejected because:** Same reasons as GPT-4, plus no fine-tuning option at all.

**Llama 3.1 70B**

Significantly more capable than 8B. Better reasoning, better instruction following, fewer extraction errors. But 70B requires 4x A100 80GB GPUs (even quantized) for inference, costs roughly $15-20/hour in GPU compute, and has 3-4x higher latency per request than 8B.

**Rejected because:** Infrastructure cost and latency at scale. At 50,000 daily orders * 1.3 segments * 2 LLM calls (extraction + generation) = ~130,000 LLM calls/day. At 70B latency (~3-5 seconds per call), you'd need multiple GPU instances running in parallel, pushing GPU costs to $400-600/day. At 8B (~1-2 seconds per call on a single A10G), one or two instances handle the load at ~$50-100/day.

**Llama 3.1 8B**

Good baseline reasoning capability, excellent fine-tuning support through the HuggingFace ecosystem, runs on a single A10G GPU (24GB VRAM) with QLoRA quantization, fast inference with vLLM, and entirely self-hosted — proprietary data never leaves your infrastructure.

### Decision: Llama 3.1 8B

**Reasoning:** The 8B model provides sufficient reasoning capability for structured extraction when fine-tuned on domain data. The key insight is that fine-tuning on 500+ domain-specific examples closes most of the capability gap between 8B and 70B for this specific task. General benchmarks show 70B far ahead of 8B, but on a narrow domain task with targeted fine-tuning, 8B reaches 90-95% of 70B's accuracy at a fraction of the cost.

**Tradeoff:** The 8B model struggles with complex multi-defect extractions where defects interact (e.g., "the compressor failure caused a coolant leak which then damaged the control board"). It tends to extract each defect independently without capturing causal relationships. This is acceptable because causal relationships are captured in the cross-reference detection during Phase 1 segmentation, and the LLM doesn't need to reason about causality — it just needs to extract each defect's attributes accurately.

**Rule:** If extraction accuracy on the evaluation dataset drops below 90% F1, the first intervention is improving training data quality and quantity. If that doesn't help, the second intervention is upgrading to Llama 3.1 13B or 70B with quantization. Model size is the last lever, not the first.

---

## 3. Why QLoRA Over Other Fine-Tuning Methods

### Fine-Tuning Methods Compared

**Full Fine-Tuning**

Update all model parameters. Maximum flexibility, maximum accuracy potential. But requires enormous GPU memory (8B model with fp16 needs ~32GB just for parameters + optimizer states + gradients, meaning you need at least an A100 80GB). Training is slow (hours to days). And you create a complete copy of the model weights, making storage and versioning expensive.

**Rejected because:** GPU memory requirements exceed single A10G (24GB). Cost of A100 infrastructure is not justified for this task. Risk of catastrophic forgetting — updating all parameters can degrade the model's general capabilities.

**Standard LoRA (Low-Rank Adaptation)**

Freeze the base model weights entirely. Add small trainable low-rank matrices (adapters) to specific layers (typically attention projections). Only the adapter weights are updated during training. The base model stays intact, preserving general capabilities. Adapter size is typically 0.1-1% of total parameters.

**Viable option.** Much lower memory requirement than full fine-tuning. But the base model still needs to be loaded in fp16 (16GB for 8B model), plus optimizer states for the adapters. Total memory: ~20-22GB. Fits on an A10G but leaves little headroom.

**QLoRA (Quantized LoRA)**

Same concept as LoRA, but the base model is loaded in 4-bit quantization (NF4 — NormalFloat4 data type) instead of fp16. This reduces the base model's memory footprint from ~16GB to ~4GB. The LoRA adapters are still trained in fp16/bf16 for training stability. Total memory during training: ~6-8GB. Inference memory: ~5-6GB.

### Decision: QLoRA

**Reasoning:**

QLoRA gives us the benefits of LoRA (parameter-efficient, preserves base model capabilities, small adapter size) with dramatically reduced memory requirements. The 4-bit quantized base model fits comfortably on a single A10G GPU with plenty of headroom for batch processing during training and inference. This means we can train on a single $1/hour GPU instance instead of needing multi-GPU infrastructure.

**The critical insight about QLoRA is that 4-bit quantization during training barely impacts fine-tuning quality.** The original QLoRA paper (Dettmers et al., 2023) showed that QLoRA matches full 16-bit fine-tuning performance on most benchmarks. The reason is that the base model weights are frozen anyway — they're used for forward passes but not updated. Whether those frozen weights are in fp16 or NF4 matters much less than you'd expect, because the gradient signal that updates the LoRA adapters still captures the necessary information.

**Tradeoff:** 4-bit quantization does introduce a small accuracy penalty compared to fp16 LoRA. On general benchmarks, this is typically 0.5-1% accuracy loss. For our specific task (structured extraction from technical logs), we validated through evaluation that the QLoRA-trained model matches our accuracy targets. If it hadn't, we'd have fallen back to standard LoRA on a larger GPU.

**Rule:** Always validate QLoRA performance against a held-out evaluation set before deploying. Do not assume the general benchmark findings transfer to your specific domain — always measure.

### QLoRA Configuration

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 — best for QLoRA
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in bf16 for stability
    bnb_4bit_use_double_quant=True,        # Double quantization — quantize the quantization constants
)

# LoRA adapter config
lora_config = LoraConfig(
    r=64,                    # Rank of the low-rank matrices
    lora_alpha=16,           # Scaling factor
    target_modules=[         # Which layers to adapt
        "q_proj",            # Query projection in self-attention
        "k_proj",            # Key projection
        "v_proj",            # Value projection
        "o_proj",            # Output projection
        "gate_proj",         # FFN gate projection (Llama-specific)
        "up_proj",           # FFN up projection
        "down_proj",         # FFN down projection
    ],
    lora_dropout=0.05,       # Light dropout for regularization
    bias="none",             # Don't train bias terms
    task_type=TaskType.CAUSAL_LM,
)
```

### Design Decisions in QLoRA Config

**Decision:** LoRA rank r=64.

**Reasoning:** Rank determines the expressiveness of the LoRA adapters. Higher rank = more trainable parameters = more capacity to learn domain-specific patterns. The original LoRA paper found r=8-16 sufficient for general tasks. But for domain-specific fine-tuning where the model needs to learn entirely new vocabulary and extraction patterns, r=64 provides significantly better performance. At r=64, the total trainable parameters are approximately 40-50 million out of 8 billion (~0.6% of total), which is still very parameter-efficient.

**Tradeoff:** Higher rank increases training time (~20% longer than r=16) and adapter file size (~200MB vs ~50MB). Both are acceptable — training runs once and the adapter is small relative to the base model.

**Decision:** Target all attention AND FFN layers (7 module types).

**Reasoning:** Earlier LoRA approaches only targeted attention layers (q_proj, v_proj). But for domain-specific tasks, the FFN layers (gate_proj, up_proj, down_proj) store factual knowledge and vocabulary relationships. Adapting these layers allows the model to learn domain-specific vocabulary mappings (e.g., "E-47" → compressor-related error) that live in the FFN layers, not just the attention patterns.

**Tradeoff:** More target modules means more trainable parameters and slightly longer training time. But the improvement in domain-specific extraction accuracy (measured ~5-8% F1 improvement in our evaluation) justifies the cost.

**Decision:** lora_alpha=16 with r=64 (scaling ratio of 16/64 = 0.25).

**Reasoning:** lora_alpha controls how much the LoRA adapters influence the base model's output. The effective scaling is alpha/r. A ratio of 0.25 means the adapter's contribution is moderate — enough to shift the model's behavior toward domain-specific extraction without overwhelming the base model's general capabilities. If alpha were too high (e.g., 128 with r=64, ratio = 2.0), the adapter could dominate and cause erratic behavior on edge cases the training data didn't cover.

**Decision:** bnb_4bit_use_double_quant=True.

**Reasoning:** Double quantization quantizes the quantization constants themselves, saving an additional ~0.4GB of memory. No measurable accuracy impact. Free memory savings.

---

## 4. Training Data Preparation

### Data Source

The training data comes from LGS Tech's historical service orders — orders that were previously processed manually by human operators. Each historical order has the original unstructured text AND the structured defect record that a human created from it. This human-created structured record becomes the ground truth label.

### Data Collection Strategy

**Decision:** Start with 500-800 labeled examples for the initial fine-tuning run.

**Reasoning:** For a narrow, well-defined extraction task with a consistent schema, 500-800 examples is the sweet spot for QLoRA fine-tuning. Below 300, the model doesn't learn the extraction patterns reliably. Above 1000, you hit diminishing returns — the marginal accuracy improvement per additional example drops sharply. The initial training focuses on getting a working model quickly; subsequent retraining rounds add examples for specific failure cases.

**Tradeoff:** 500-800 examples won't cover every edge case (rare defect types, unusual language patterns, multi-language orders). The model will make mistakes on these edges. But the Pydantic validation in Phase 2 catches structural errors, and the human review fallback handles cases the model can't process. As edge cases are identified in production, they're added to the training set for the next fine-tuning round.

### Data Cleaning for Training

Not all historical service orders are suitable for training. You need to clean the dataset:

**Remove duplicates.** The same service order submitted through multiple channels (email + web form) creates duplicate training examples. Deduplicate by order_id.

**Remove low-quality labels.** Some historical orders were processed hastily — incomplete defect records, missing severity, vague component names like "unit" instead of "compressor." These teach the model bad habits. Filter out orders where the human-created defect record has more than 1 missing required field.

**Remove extremely long orders.** Orders with 2000+ tokens of text are outliers that don't fit the typical processing pattern. They're kept for evaluation but excluded from training to avoid skewing the model toward handling unusually long context.

**Remove non-English orders.** If the training data includes orders from non-English regions, remove them unless you're specifically training a multilingual model. Mixing languages in a small dataset can confuse the model.

**Balance defect type distribution.** If 80% of training data is "mechanical" defects and only 2% is "software" defects, the model will be biased toward predicting "mechanical." Oversample rare defect types or undersample common types to achieve a more balanced distribution. Don't aim for perfectly uniform — just reduce extreme imbalance.

```python
from collections import Counter

# Check distribution
defect_types = [example["label"]["defect_type"] for example in training_data]
distribution = Counter(defect_types)
print(distribution)
# Counter({'mechanical': 340, 'electrical': 120, 'thermal': 45, 
#          'hydraulic': 30, 'software': 12, 'structural': 8})

# Oversample rare types to minimum 30 examples each
min_count = 30
for defect_type, count in distribution.items():
    if count < min_count:
        examples_of_type = [e for e in training_data if e["label"]["defect_type"] == defect_type]
        # Duplicate examples (with slight text augmentation) to reach minimum
        while len(examples_of_type) < min_count:
            examples_of_type.append(augment_example(random.choice(examples_of_type)))
        training_data.extend(examples_of_type[count:])  # add the new copies
```

### Data Augmentation

For rare defect types, simple duplication isn't enough — the model memorizes the exact text. Light augmentation creates training variety:

**Synonym replacement.** Replace "grinding noise" with "rattling sound" or "abnormal vibration." These are semantically equivalent symptoms.

**Paraphrase variation.** "The compressor is making grinding noise" → "Grinding noise observed from compressor unit" → "Compressor grinding, abnormal."

**Metadata shuffling.** Change the model number, plant ID, and serial number to prevent the model from associating specific entities with specific defect types.

**Decision:** Use augmentation sparingly — maximum 2x oversampling per rare category.

**Reasoning:** Heavy augmentation (5x-10x) creates synthetic patterns that don't match real data. The model learns to extract from augmented text that's subtly different from real technician writing. Light augmentation (2x) provides just enough variety to prevent memorization without introducing artificial patterns.

**Tradeoff:** Even light augmentation means ~20-30% of your training data for rare categories is synthetic. If the augmentation introduces unrealistic patterns (e.g., a synonym that no technician would actually use), the model might struggle with real examples of that type. Mitigated by having a domain expert review augmented examples before training.

---

## 5. Data Labeling & Annotation Pipeline

### Instruction-Response Format

Fine-tuning requires instruction-response pairs — the model learns to produce the response given the instruction. For our extraction task:

```json
{
  "instruction": "Extract all defects from the following service order as structured JSON.\n\nService Order:\n\"Compressor making grinding noise at Plant 7, unit XR-440. Error E-47 on control panel. Unit is 18 months old, under standard warranty. Noticed after last scheduled maintenance 2 weeks ago.\"\n\nPre-extracted entities:\n- Model: XR-440\n- Error Code: E-47\n- Plant: Plant 7\n\nOutput only valid JSON array.",
  
  "response": "[\n  {\n    \"defect_type\": \"mechanical\",\n    \"severity\": \"high\",\n    \"affected_component\": \"compressor\",\n    \"root_cause_hypothesis\": \"Possible bearing failure or internal component wear, onset correlates with recent maintenance\",\n    \"error_codes\": [\"E-47\"],\n    \"symptoms\": [\"grinding noise\"],\n    \"confidence\": 0.9\n  }\n]"
}
```

### Annotation Guidelines

Consistent labeling is critical for training quality. The annotation guidelines define how annotators should handle every decision:

**Defect Type Assignment Rules:**
- "mechanical" — physical movement components: compressors, motors, fans, bearings, belts, valves
- "electrical" — wiring, circuits, control boards, sensors, power supply, fuses
- "software" — firmware errors, control logic failures, calibration issues, display errors
- "hydraulic" — coolant lines, refrigerant systems, pressure issues, leaks in fluid systems
- "thermal" — overheating, freezing, temperature regulation failures, heat exchanger issues
- "structural" — housing damage, mounting failures, corrosion, physical frame issues

**Severity Assignment Rules:**
- "low" — cosmetic issues, intermittent minor symptoms, no functional impact
- "medium" — functional degradation but unit still operates, performance reduced
- "high" — unit functionality significantly impaired, requires prompt attention
- "critical" — unit non-operational, safety risk, or risk of cascading damage to other components

**Root Cause Hypothesis Rules:**
- Include if there are contextual clues in the text ("after maintenance," "during storm," "since installation")
- Set to null if no contextual clues — do NOT fabricate hypotheses
- Frame as hypothesis, not conclusion: "Possible bearing failure" not "Bearing has failed"

**Confidence Assignment Rules:**
- 0.9-1.0: Clear, unambiguous defect description with specific symptoms and error codes
- 0.7-0.8: Defect is described but some details are vague or missing
- 0.5-0.6: Defect is implied but not explicitly stated, requires interpretation
- Below 0.5: Do not include — too speculative

**Multi-Defect Rules:**
- Each distinct defect is a separate JSON object in the array
- If defects are causally related, each still gets its own object (causality is handled in segmentation)
- A single component can have multiple defects (e.g., compressor has both noise AND error code = 1 defect with multiple symptoms, NOT 2 separate defects)

### Annotation Quality Control

**Decision:** Use a two-annotator + adjudicator model.

**Reasoning:** A single annotator introduces personal bias — they might consistently rate severity one level higher than appropriate, or always classify ambiguous cases as "mechanical." Two independent annotators labeling the same examples, with an expert adjudicating disagreements, produces much more consistent training data.

**Implementation:**
1. Each service order is labeled independently by 2 annotators
2. Inter-annotator agreement is measured (Cohen's kappa)
3. Orders with perfect agreement go directly to training data
4. Orders with disagreement go to an expert adjudicator who makes the final decision
5. Systematic disagreement patterns (one annotator always harsher on severity) are identified and corrected through calibration sessions

**Target:** Cohen's kappa ≥ 0.80 (strong agreement). If kappa drops below 0.80, hold a calibration session to realign annotators on the guidelines.

**Tradeoff:** The two-annotator model doubles annotation cost and time. For 500 examples, this means 1000 annotation tasks plus adjudication time. But the training data quality improvement directly translates to model accuracy. Noisy labels cause noisy models — this is one area where cutting corners has outsized negative consequences.

---

## 6. Training Configuration & Hyperparameters

### Full Training Script

```python
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import torch

# ── Load base model with 4-bit quantization ──
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # Flash Attention for efficiency
)

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Prepare model for QLoRA training ──
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
# Output: Trainable: 41,943,040 / 8,030,261,248 (0.52%)

# ── Training arguments ──
training_args = TrainingArguments(
    output_dir="./llama-3.1-8b-defect-extractor",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,      # Effective batch size = 4 * 4 = 16
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,                          # bf16 training for stability
    optim="paged_adamw_32bit",          # Paged optimizer for memory efficiency
    gradient_checkpointing=True,        # Trade compute for memory
    max_grad_norm=0.3,                  # Gradient clipping
    report_to="wandb",                  # Log to Weights & Biases
    seed=42,
)

# ── Initialize trainer ──
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_seq_length=2048,
    dataset_text_field="formatted_text",  # The formatted instruction-response text
    packing=True,                         # Pack multiple short examples into one sequence
)

# ── Train ──
trainer.train()

# ── Save adapter weights ──
trainer.model.save_pretrained("./llama-3.1-8b-defect-extractor/final_adapter")
tokenizer.save_pretrained("./llama-3.1-8b-defect-extractor/final_adapter")
```

### Hyperparameter Decisions

**Decision:** 3 training epochs.

**Reasoning:** For SFT on a small dataset (500-800 examples), 3 epochs is the sweet spot. At 1 epoch, the model hasn't seen each example enough to learn the extraction pattern reliably. At 5+ epochs, the model starts memorizing specific examples rather than learning general patterns (overfitting). 3 epochs gives sufficient exposure without overfitting, validated by watching the eval loss — it should decrease through epochs 1-2, flatten in epoch 3, and start increasing in epoch 4+ if you were to continue.

**Decision:** Effective batch size of 16 (per_device=4 * gradient_accumulation=4).

**Reasoning:** Batch size affects training stability. Too small (1-2) creates noisy gradient updates and unstable training. Too large (64+) can smooth out gradients too much and converge to suboptimal solutions. 16 is a well-established sweet spot for fine-tuning language models. We achieve this through gradient accumulation because a per-device batch of 16 wouldn't fit in the A10G's 24GB VRAM alongside the quantized model.

**Decision:** Learning rate 2e-4 with cosine schedule and 3% warmup.

**Reasoning:** 2e-4 is the standard learning rate for QLoRA fine-tuning — established by the original QLoRA paper and validated across many subsequent works. Too high (1e-3) causes training instability and adapter weights oscillating. Too low (1e-5) means the adapters barely change from their random initialization in 3 epochs. The cosine schedule gradually reduces the learning rate as training progresses, allowing fine-grained optimization toward the end. The 3% warmup prevents early instability when the adapter weights are still random.

**Decision:** gradient_checkpointing=True.

**Reasoning:** Gradient checkpointing trades compute for memory — instead of storing all intermediate activations during the forward pass (needed for backpropagation), it recomputes them during the backward pass. This roughly halves the memory required for training at the cost of ~20% slower training. On a memory-constrained A10G, this is essential to fit the training process.

**Tradeoff:** Training takes ~20% longer. For a 3-epoch training run that takes 2-3 hours, this adds ~30-40 minutes. Acceptable for the memory savings.

**Decision:** packing=True in SFTTrainer.

**Reasoning:** Service order texts are short — most are under 500 tokens. Without packing, each training example is padded to max_seq_length (2048), wasting ~75% of each training batch on padding tokens. Packing concatenates multiple short examples into a single 2048-token sequence (separated by EOS tokens), dramatically increasing training efficiency — you're learning from real tokens, not padding.

**Tradeoff:** Packing introduces minor artifacts where attention can leak across example boundaries within a packed sequence. In practice, this doesn't measurably affect quality for instruction fine-tuning because the model quickly learns to treat EOS tokens as hard boundaries.

---

## 7. Training Infrastructure & Execution

### Hardware

**Decision:** Single AWS g5.xlarge instance (NVIDIA A10G, 24GB VRAM).

**Cost:** ~$1.00-1.20/hour on-demand, ~$0.40-0.60/hour spot.

**Training time:** ~2-3 hours for 3 epochs on 600 examples with QLoRA.

**Total training cost:** ~$3-5 per training run (spot pricing).

This is remarkably cheap. You can afford to retrain frequently — weekly or even daily if needed.

### Training Pipeline Automation

```python
# training_pipeline.py — Run end-to-end training

import subprocess
import json
from datetime import datetime

def run_training_pipeline():
    run_id = f"train_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    # Step 1: Pull latest labeled data from annotation store
    training_data = pull_labeled_data()
    
    # Step 2: Split into train/eval/test (80/10/10)
    train, eval_set, test = split_data(training_data, ratios=[0.8, 0.1, 0.1])
    
    # Step 3: Format into instruction-response pairs
    train_formatted = format_for_sft(train)
    eval_formatted = format_for_sft(eval_set)
    
    # Step 4: Run training
    train_model(train_formatted, eval_formatted, run_id)
    
    # Step 5: Evaluate on held-out test set
    test_results = evaluate_model(test, run_id)
    
    # Step 6: Compare against production model
    production_results = evaluate_model(test, "production_current")
    
    # Step 7: Promote if improvement meets threshold
    if test_results["f1"] >= production_results["f1"] - 0.02:
        # New model is at least within 2% of production
        if test_results["f1"] > production_results["f1"]:
            promote_model(run_id, auto=True)
            print(f"Model {run_id} auto-promoted (F1: {test_results['f1']:.3f} > {production_results['f1']:.3f})")
        else:
            print(f"Model {run_id} within threshold but not better. Manual review recommended.")
    else:
        print(f"Model {run_id} WORSE than production. Not promoted.")
        alert_team(run_id, test_results, production_results)
```

**Decision:** Automated training pipeline with automated promotion only if the new model is strictly better.

**Reasoning:** Manual model promotion creates a bottleneck — a trained model sits in staging for days waiting for someone to evaluate and promote it. Automated evaluation against the production baseline with automated promotion for improvements removes this bottleneck. But we only auto-promote if the new model is strictly better (higher F1), not if it's merely equivalent (within threshold). This prevents unnecessary model switches that could introduce instability.

**Tradeoff:** Automated promotion assumes the held-out test set is representative of real production data. If the test set has a different distribution than production (e.g., it was collected during a different season when different defect types are common), the new model might perform well on the test set but worse in production. Mitigated by periodically refreshing the test set from recent production data.

---

## 8. Model Evaluation & Benchmarking

### Evaluation Metrics

**Metric 1 — JSON Validity Rate**

What percentage of model outputs are valid JSON that can be parsed without errors? This is the most basic quality bar.

```python
def json_validity_rate(predictions: list[str]) -> float:
    valid = 0
    for pred in predictions:
        try:
            parsed = json.loads(pred.strip())
            if isinstance(parsed, list):
                valid += 1
        except json.JSONDecodeError:
            pass
    return valid / len(predictions)
```

**Target:** ≥95% (the remaining 5% is caught by the retry loop in Phase 2).

**Metric 2 — Schema Compliance Rate**

Of the valid JSON outputs, what percentage passes full Pydantic validation?

```python
def schema_compliance_rate(predictions: list[str]) -> float:
    compliant = 0
    for pred in predictions:
        try:
            parsed = json.loads(pred.strip())
            for defect in parsed:
                ExtractedDefect(**defect)  # Pydantic validation
            compliant += 1
        except (json.JSONDecodeError, ValidationError):
            pass
    return compliant / len(predictions)
```

**Target:** ≥90% first-attempt compliance. With retries, ≥98%.

**Metric 3 — Entity-Level F1**

For each field in the defect schema, measure precision, recall, and F1 against ground truth:

```python
def entity_f1(predictions: list, ground_truth: list) -> dict:
    metrics = {}
    
    for field in ["defect_type", "severity", "affected_component"]:
        true_values = [gt[field] for gt in ground_truth]
        pred_values = [pred.get(field, None) for pred in predictions]
        
        correct = sum(1 for t, p in zip(true_values, pred_values) if t == p)
        precision = correct / len(pred_values) if pred_values else 0
        recall = correct / len(true_values) if true_values else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[field] = {"precision": precision, "recall": recall, "f1": f1}
    
    return metrics
```

**Targets:**
- defect_type F1: ≥0.92
- severity F1: ≥0.85 (severity is more subjective, so lower threshold)
- affected_component F1: ≥0.90
- error_codes recall: ≥0.95 (missing an error code is worse than hallucinating one)

**Metric 4 — Defect Count Accuracy**

Does the model extract the correct number of defects? Under-extraction (missing a defect) is worse than over-extraction (hallucinating a defect).

```python
def defect_count_accuracy(predictions, ground_truth) -> dict:
    exact_match = 0
    under_extract = 0
    over_extract = 0
    
    for pred, gt in zip(predictions, ground_truth):
        if len(pred) == len(gt):
            exact_match += 1
        elif len(pred) < len(gt):
            under_extract += 1
        else:
            over_extract += 1
    
    return {
        "exact_match_rate": exact_match / len(predictions),
        "under_extraction_rate": under_extract / len(predictions),
        "over_extraction_rate": over_extract / len(predictions),
    }
```

**Targets:**
- Exact match rate: ≥80%
- Under-extraction rate: ≤10% (this is the dangerous failure mode)
- Over-extraction rate: ≤15% (annoying but not dangerous — duplicates caught downstream)

**Metric 5 — Latency**

Inference latency per request. Measured at the vLLM service level.

**Targets:**
- P50: ≤1.5 seconds
- P95: ≤3.0 seconds
- P99: ≤5.0 seconds

### Evaluation Dataset

**Decision:** Maintain a held-out golden test set of 100 examples, never used in training.

**Reasoning:** The test set must be completely independent of training data to give an unbiased performance estimate. If even one test example leaks into training, the evaluation becomes unreliable.

**Rule:** The test set is refreshed quarterly by replacing 25% of examples with recent production data (orders that were processed by the system and then verified by human reviewers). This ensures the test set stays representative of current data patterns.

---

## 9. Inference Architecture (vLLM)

### Why vLLM

**Decision:** Use vLLM for production inference.

**Alternatives considered:**

- **HuggingFace Transformers generate():** Simple but slow — no batching optimization, sequential decoding, no PagedAttention. Latency 3-5x worse than vLLM.
- **TGI (Text Generation Inference by HuggingFace):** Good option, supports continuous batching and quantization. But vLLM has better throughput benchmarks for Llama models specifically.
- **Triton Inference Server (NVIDIA):** Enterprise-grade, supports model ensembles and complex serving pipelines. Overkill for serving a single model. More operational complexity than vLLM.

**Reasoning:** vLLM's key innovation is PagedAttention — it manages the KV cache (the key-value attention cache that grows during autoregressive generation) using virtual memory paging, similar to how operating systems manage RAM. This eliminates memory fragmentation, allowing more concurrent requests per GPU. Combined with continuous batching (new requests can join a batch mid-generation rather than waiting for the current batch to complete), vLLM achieves 2-4x higher throughput than naive implementations.

### vLLM Deployment

```bash
# Start vLLM server with the QLoRA adapter
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-modules defect-extractor=./adapters/defect-extractor-v1.2 \
    --quantization awq \
    --dtype half \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 32 \
    --port 8000
```

### Configuration Decisions

**Decision:** --quantization awq for inference (not the NF4 from training).

**Reasoning:** QLoRA uses NF4 quantization during training, but NF4 is not optimal for inference. AWQ (Activation-aware Weight Quantization) is specifically designed for inference — it identifies which weights are most important (based on activation patterns) and preserves their precision while aggressively quantizing less important weights. AWQ typically provides better inference accuracy than NF4 at the same 4-bit size. The workflow is: train with QLoRA/NF4, then convert the merged model (base + adapter) to AWQ for serving.

**Tradeoff:** Converting from QLoRA to AWQ requires a one-time conversion step that takes 30-60 minutes. Also, AWQ quantization is slightly lossy — there's a tiny accuracy difference between the QLoRA-trained model and the AWQ-served model. This difference is validated during model evaluation (we evaluate the AWQ model, not the QLoRA model) to ensure it meets accuracy targets.

**Decision:** --max-num-seqs 32 (maximum concurrent sequences).

**Reasoning:** With an A10G (24GB VRAM), the quantized 8B model uses ~5GB. Each concurrent request's KV cache uses ~100-200MB depending on sequence length. At 32 concurrent sequences, KV cache uses ~3-6GB, total GPU memory ~8-11GB. The remaining 13-16GB provides comfortable headroom for PagedAttention's virtual memory management.

**Tradeoff:** More concurrent sequences mean higher throughput but longer per-request latency (each request gets a smaller share of GPU compute). At 32 sequences, per-request latency might increase by 20-30% compared to sequential processing. But throughput (requests per second) increases by 10-15x. Since we have thousands of daily orders and acceptable latency is seconds (not milliseconds), throughput optimization is the right choice.

**Decision:** --gpu-memory-utilization 0.90.

**Reasoning:** vLLM pre-allocates GPU memory for the KV cache. Setting utilization to 0.90 means vLLM uses 90% of available VRAM for the model and KV cache, leaving 10% as buffer for CUDA operations, memory fragmentation, and unexpected spikes. Setting it higher (0.95) risks CUDA out-of-memory errors under peak load. Setting it lower (0.80) wastes available GPU memory that could serve more concurrent requests.

### LoRA Adapter Hot-Swapping

**Decision:** Use vLLM's --enable-lora with named adapters for zero-downtime model updates.

**Reasoning:** When a new adapter version is trained and evaluated, you can add it as a new named module (`defect-extractor-v1.3`) without removing the current one (`defect-extractor-v1.2`). Then you update the routing to point to the new adapter. If the new adapter performs poorly in production, you instantly roll back by pointing routing back to the previous adapter. This is much faster than restarting the vLLM server with a new model.

**Tradeoff:** Multiple LoRA adapters consume additional GPU memory (~200MB per adapter). With 2-3 active adapters (current + previous + canary), that's 400-600MB — trivial compared to the base model's memory footprint.

---

## 10. Prompt Engineering for Extraction

### System Prompt Design

```
You are a defect extraction system for LGS Tech service orders.
Given a service order text and pre-extracted entities, extract ALL defects as structured JSON.

RULES:
1. Extract EVERY distinct defect mentioned in the text
2. Each defect must include: defect_type, severity, affected_component, 
   root_cause_hypothesis, error_codes, symptoms, confidence
3. Use ONLY information present in the text — do not hallucinate or infer beyond what is stated
4. Cross-validate against pre-extracted entities provided
5. defect_type must be one of: mechanical, electrical, software, hydraulic, thermal, structural
6. severity must be one of: low, medium, high, critical
7. confidence must reflect certainty: 0.9+ for clear descriptions, 0.5-0.8 for ambiguous
8. If multiple symptoms relate to one component, group them as ONE defect with multiple symptoms
9. Output ONLY a valid JSON array — no markdown, no explanation, no preamble
```

### Why Include Rules in the Prompt Even After Fine-Tuning?

**Decision:** Include extraction rules in the system prompt even though the model is fine-tuned.

**Reasoning:** Fine-tuning teaches the model the general pattern of "input text → structured JSON." But it doesn't guarantee perfect adherence to every rule in every edge case. The system prompt acts as a runtime reminder that reinforces the fine-tuned behavior. Think of it like a checklist — a pilot is trained to fly the plane, but they still use the pre-flight checklist every time.

**Tradeoff:** The system prompt consumes ~200 tokens of context window. This is a small cost. Removing the system prompt would save tokens but increase the rate of rule violations (wrong enum values, hallucinated fields), which increases the retry rate in Phase 2.

### Few-Shot Examples in Prompt

**Decision:** Do NOT include few-shot examples in the prompt.

**Reasoning:** The fine-tuned model has already seen 500+ examples during training. Adding 2-3 few-shot examples in the prompt would consume 400-600 tokens with minimal benefit. The model already knows the pattern. Few-shot examples are valuable for general models that haven't been fine-tuned — they're redundant for a fine-tuned model.

**Tradeoff:** If the model encounters a truly novel input pattern (something very different from training data), few-shot examples might help ground it. But these edge cases are rare and are handled by the retry loop. Saving 400-600 tokens per request (multiplied by 130,000 daily requests) is a significant cost savings in GPU compute.

**Exception:** During the retry loop (attempt 2+), we DO include the failed output as a negative example ("this was wrong because..."). This is a targeted few-shot correction, not a general few-shot pattern.

---

## 11. Structured Output & JSON Mode

### Forcing JSON Output

**Decision:** Use constrained decoding in vLLM to force valid JSON output.

**Implementation:** vLLM supports guided/constrained generation via grammars or JSON schemas. When enabled, the model can only generate tokens that are valid according to the JSON grammar at each step — it literally cannot produce invalid JSON.

```python
# vLLM API call with JSON schema enforcement
response = httpx.post(
    "http://llm-service.internal:8000/v1/completions",
    json={
        "prompt": formatted_prompt,
        "max_tokens": 1024,
        "temperature": 0.0,
        "guided_json": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "defect_type": {"type": "string", "enum": ["mechanical", "electrical", "software", "hydraulic", "thermal", "structural"]},
                    "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "affected_component": {"type": "string"},
                    "root_cause_hypothesis": {"type": ["string", "null"]},
                    "error_codes": {"type": "array", "items": {"type": "string"}},
                    "symptoms": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["defect_type", "severity", "affected_component", "error_codes", "symptoms", "confidence"]
            }
        }
    }
)
```

**Reasoning:** Constrained decoding eliminates an entire category of failures — JSON parse errors. Without it, the model might output a perfectly good extraction with a trailing comma that breaks JSON parsing, or wrap the output in markdown fences, or add a "Here is the extraction:" preamble. These are formatting failures, not reasoning failures, and constrained decoding prevents them entirely.

**Tradeoff:** Constrained decoding adds ~10-15% latency overhead because the model's logits must be masked at each generation step to enforce the grammar. At our scale, this is ~0.1-0.2 seconds extra per request — acceptable. Also, extremely aggressive schema constraints can force the model into generating nonsensical values to satisfy the schema (e.g., writing "unknown" for affected_component because the schema requires a string). Mitigated by using the schema for structure enforcement only and relying on Pydantic validation in Phase 2 for semantic validation.

---

## 12. Cross-Validation with Phase 1 Entities

### How Cross-Validation Works

Phase 1 pre-extracts entities using regex (high confidence, deterministic) and spaCy NER. These are passed as "hints" in the extraction prompt. After the LLM generates its extraction, the validation node in Phase 2 cross-checks:

```
Phase 1 Regex found:         LLM extracted:           Cross-validation result:
─────────────────────────────────────────────────────────────────────────────────
model: "XR-440"              component mentions XR-440  ✓ Match
error_code: "E-47"           error_codes: ["E-47"]      ✓ Match
error_code: "E-12"           error_codes: ["E-47"]      ⚠ Warning: E-12 missing
part: "P/N 7832-A"           not mentioned               ⚠ Warning: part not in extraction
(nothing)                    error_codes: ["E-99"]       ⚠ Warning: LLM found code regex missed
```

### Resolution Rules

**Rule 1:** If regex found an entity and the LLM agrees → high confidence, no action needed.

**Rule 2:** If regex found an entity and the LLM disagrees → warning logged, LLM extraction preserved (the LLM may have correctly interpreted context that regex couldn't).

**Rule 3:** If regex found an entity and the LLM missed it entirely → warning logged, the missed entity is added to the extraction as a low-confidence annotation for the human reviewer to verify.

**Rule 4:** If the LLM found an entity that regex didn't → accepted (the LLM can identify entities in context that regex patterns don't cover, like informal component references).

---

## 13. Model Versioning & Registry

### Version Management

```
model-registry/
├── base-models/
│   └── meta-llama-3.1-8b-instruct/     # Downloaded once
├── adapters/
│   ├── defect-extractor-v1.0/           # Initial training
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── training_metadata.json       # Hyperparams, dataset hash, eval results
│   ├── defect-extractor-v1.1/           # Added 100 new examples
│   ├── defect-extractor-v1.2/           # Current production
│   └── defect-extractor-v1.3/           # Canary testing
├── merged-models/
│   ├── defect-extractor-v1.2-awq/       # Production: merged + AWQ quantized
│   └── defect-extractor-v1.3-awq/       # Canary: merged + AWQ quantized
└── evaluation-results/
    ├── v1.0-eval.json
    ├── v1.1-eval.json
    ├── v1.2-eval.json
    └── v1.3-eval.json
```

**Decision:** Store only adapter weights, not full merged models (except for AWQ-converted serving models).

**Reasoning:** Each adapter is ~200MB. A full merged model is ~16GB. Storing only adapters reduces storage by 80x while preserving the ability to reproduce any model version (base model + adapter = complete model).

### Promotion Workflow

```
Training → Evaluation on test set → Compare to production baseline
                                          │
                                    Better? ──── Yes ──── Canary deploy (10% traffic)
                                          │                       │
                                          No                Monitor for 24 hours
                                          │                       │
                                     Do not promote         Metrics stable?
                                                                  │
                                                            Yes ── Full promotion (100%)
                                                                  │
                                                            No ─── Rollback to previous
```

**Decision:** Canary deployment before full promotion.

**Reasoning:** Even if the new model performs well on the held-out test set, production data might have patterns the test set doesn't cover. A canary deployment (routing 10% of traffic to the new model while 90% stays on the current model) reveals production-specific issues before they affect all orders. If the canary shows regression (higher retry rate, lower schema compliance, more human escalations), it's rolled back without affecting the majority of orders.

---

## 14. Retraining Strategy & Drift Detection

### When to Retrain

**Trigger 1 — Scheduled retraining (monthly).**

Every month, incorporate new labeled examples from production (orders that were processed and human-verified) into the training set and retrain. This keeps the model current with evolving vocabulary and defect patterns.

**Trigger 2 — Accuracy drift detected.**

Weekly evaluation against the golden test set. If any key metric drops below threshold (e.g., defect_type F1 drops below 0.90), trigger an immediate retraining cycle.

**Trigger 3 — New product line or defect category introduced.**

When LGS Tech launches a new product line or encounters a new defect category not in the training data, proactively collect and label 30-50 examples of the new category and retrain.

**Trigger 4 — High retry rate in production.**

If the Phase 2 extraction retry rate exceeds 15% over a 24-hour period, something is wrong — either the model is degrading or the input data distribution has shifted. Investigate and retrain if needed.

### Drift Detection Implementation

```python
# Run weekly evaluation
def weekly_drift_check():
    golden_test_set = load_golden_dataset()
    current_model = load_production_model()
    
    results = evaluate_model(current_model, golden_test_set)
    
    # Compare against baseline (results from when model was promoted)
    baseline = load_baseline_results(current_model.version)
    
    drift_signals = {
        "defect_type_f1_drop": baseline["defect_type_f1"] - results["defect_type_f1"],
        "severity_f1_drop": baseline["severity_f1"] - results["severity_f1"],
        "json_validity_drop": baseline["json_validity_rate"] - results["json_validity_rate"],
        "schema_compliance_drop": baseline["schema_compliance_rate"] - results["schema_compliance_rate"],
    }
    
    # Alert if any metric dropped by more than 3%
    for metric, drop in drift_signals.items():
        if drop > 0.03:
            alert_team(
                severity="P2",
                message=f"Model drift detected: {metric} dropped by {drop:.1%}",
                recommended_action="Investigate and retrain"
            )
```

### Continual Learning Anti-Patterns to Avoid

**Anti-pattern 1 — Training on ALL production data without curation.**

Not every production order is a good training example. Orders that required 3 retries before succeeding had noisy LLM outputs that shouldn't become training data. Only use orders that passed validation on the first attempt AND were verified by human review.

**Anti-pattern 2 — Never refreshing the test set.**

If the test set stays static while production data evolves, the test set becomes unrepresentative. High test scores would mask real production degradation. Refresh 25% of the test set quarterly.

**Anti-pattern 3 — Retraining from scratch every time.**

Each retraining should start from the current production adapter, not from the base model. This preserves learned patterns and only requires the model to learn incremental changes. Training from scratch risks losing previously learned edge cases.

---

## 15. Observability & Alerting

### Metrics Emitted

**Training Metrics (logged to Weights & Biases):**

| Metric | Description |
|--------|-------------|
| `train/loss` | Training loss per step |
| `eval/loss` | Evaluation loss per step |
| `train/learning_rate` | Current learning rate |
| `train/epoch` | Current epoch |
| `train/grad_norm` | Gradient norm (detect exploding gradients) |
| `eval/json_validity_rate` | JSON validity on eval set per checkpoint |
| `eval/schema_compliance_rate` | Schema compliance on eval set per checkpoint |
| `eval/defect_type_f1` | Per-field F1 on eval set per checkpoint |

**Inference Metrics (logged to CloudWatch/Datadog):**

| Metric | Description |
|--------|-------------|
| `inference.latency_ms` | Per-request inference latency |
| `inference.tokens_input` | Input token count per request |
| `inference.tokens_output` | Output token count per request |
| `inference.batch_size` | Current continuous batch size in vLLM |
| `inference.gpu_utilization` | GPU compute utilization percentage |
| `inference.gpu_memory_used` | GPU memory utilization |
| `inference.requests_per_second` | Throughput |
| `inference.queue_depth` | Number of requests waiting in vLLM queue |
| `inference.timeout_rate` | Percentage of requests that timed out |

**Quality Metrics (logged per-request):**

| Metric | Description |
|--------|-------------|
| `extraction.json_valid` | Did the output parse as JSON? |
| `extraction.schema_compliant` | Did the output pass Pydantic? |
| `extraction.defect_count` | Number of defects extracted |
| `extraction.avg_confidence` | Average confidence across defects |
| `extraction.retry_triggered` | Did this request trigger a retry? |
| `extraction.cross_validation_match` | Did LLM match Phase 1 entities? |

### Alert Configuration

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| GPU utilization sustained high | >95% for 15 minutes | P2 | Scale to second GPU instance or optimize batch config |
| GPU out of memory | CUDA OOM error | P1 | Reduce max_num_seqs, check for memory leak |
| Inference latency spike | P95 > 5 seconds | P2 | Check GPU load, batch queue depth, model serving health |
| vLLM service down | Health check fails for 60 seconds | P1 | Restart service, check GPU health, failover if available |
| JSON validity drop | <90% over 1 hour | P1 | Model generating malformed output, check prompt template |
| Schema compliance drop | <85% over 1 hour | P2 | Model producing wrong field types/values, investigate |
| Retry rate spike | >20% of requests trigger retry | P2 | Model quality degrading, consider retraining |
| Extraction confidence drop | Average confidence drops below 0.6 | P2 | Model uncertain on new data patterns, add to training set |
| Timeout rate spike | >5% of requests timing out | P2 | GPU overloaded or model stuck in generation loop |
| Training loss divergence | Training loss increases for 100+ steps | P3 | Hyperparameter issue, bad training data, restart training |
| Drift detected | Weekly eval metric drops >3% | P2 | Investigate data shift, schedule retraining |

---

## 16. End-to-End Phase 3 Flow Summary

### Training Pipeline Flow

```
HISTORICAL SERVICE ORDERS (manually processed)
        │
Step 1 ─┤  DATA COLLECTION
        │  • Pull orders with human-created defect records
        │  • Filter: remove duplicates, low-quality labels, outliers
        │  • Target: 500-800 high-quality examples
        │
Step 2 ─┤  DATA LABELING & QUALITY
        │  • Two-annotator + adjudicator model
        │  • Annotation guidelines: defect type, severity, component rules
        │  • Inter-annotator agreement target: Cohen's kappa ≥ 0.80
        │  • Balance defect type distribution (oversample rare types)
        │  • Light augmentation for rare categories (max 2x)
        │
Step 3 ─┤  DATA FORMATTING
        │  • Convert to instruction-response pairs
        │  • Format: system prompt + service order text + pre-extracted entities → JSON
        │  • Split: 80% train / 10% eval / 10% test
        │  • Test set held out permanently — never used in training
        │
Step 4 ─┤  MODEL TRAINING (QLoRA)
        │  • Base: Llama 3.1 8B Instruct, loaded in 4-bit NF4
        │  • LoRA: r=64, alpha=16, all attention + FFN layers
        │  • Training: 3 epochs, LR 2e-4, cosine schedule, batch 16
        │  • Hardware: Single A10G GPU (~2-3 hours, ~$3-5)
        │  • Logged to Weights & Biases
        │
Step 5 ─┤  MODEL EVALUATION
        │  • Evaluate on held-out test set (100 examples)
        │  • Metrics: JSON validity (≥95%), schema compliance (≥90%),
        │    defect_type F1 (≥0.92), severity F1 (≥0.85),
        │    component F1 (≥0.90), error_codes recall (≥0.95)
        │  • Compare against current production model
        │
Step 6 ─┤  MODEL CONVERSION
        │  • Merge LoRA adapter into base model
        │  • Convert merged model to AWQ quantization for inference
        │  • Validate AWQ model accuracy matches QLoRA model
        │
Step 7 ─┤  MODEL DEPLOYMENT
        │  • Upload AWQ model to model registry
        │  • Canary deploy: 10% traffic for 24 hours
        │  • Monitor: retry rate, schema compliance, human escalation rate
        │  • If stable: promote to 100% traffic
        │  • If regression: rollback to previous version
```

### Inference Flow (Per Request)

```
SEGMENT ARRIVES FROM PHASE 2 (LangGraph Node 1)
        │
        │  Input: segment_text + pre_extracted_entities + inherited_context
        │
Step 1 ─┤  PROMPT CONSTRUCTION
        │  • System prompt (extraction rules, schema definition)
        │  • User prompt: segment text + pre-extracted entity hints + metadata
        │  • If retry: append previous validation errors
        │  • Total prompt: ~500-800 tokens
        │
Step 2 ─┤  INFERENCE (vLLM)
        │  • Model: Llama 3.1 8B + QLoRA adapter (AWQ quantized)
        │  • Temperature: 0.0 (deterministic)
        │  • Max output: 1024 tokens
        │  • Constrained decoding: JSON schema enforced
        │  • Continuous batching with PagedAttention
        │  • Timeout: 30 seconds
        │
Step 3 ─┤  RAW OUTPUT
        │  • Valid JSON array of defect objects (enforced by constrained decoding)
        │  • Stored in AgentState.extraction_raw_response
        │  • Passed to Phase 2 Node 2 (validation)
        │
        ▼
RETURNS TO PHASE 2 FOR VALIDATION
        │
        ├─ Valid → Phase 4 (RAG Retrieval)
        ├─ Invalid + retries left → Back to Step 1 (retry with error context)
        └─ Invalid + retries exhausted → Human review
```

### Key Design Principles Applied Throughout Phase 3

1. **Fine-tune for domain, not for knowledge.** The model learns extraction patterns and domain vocabulary through fine-tuning. Current policies and procedures come from RAG. This separation means the model doesn't need retraining when policies change — only when extraction patterns change.

2. **Small model, big data.** An 8B model fine-tuned on 500+ domain examples outperforms a 70B general model on this specific task at a fraction of the cost. Right-size the model for the task complexity.

3. **QLoRA makes iteration cheap.** At $3-5 per training run, you can afford to experiment aggressively — try different hyperparameters, add augmented data, test new training examples. Fast iteration beats careful planning.

4. **Constrained decoding eliminates format failures.** Don't waste retry budget on formatting issues (broken JSON, wrong structure). Constrained decoding forces valid output at the token level, preserving retries for genuine reasoning failures.

5. **Cross-validation catches what the model misses.** Phase 1's deterministic regex extractions serve as anchors. The LLM's semantic extractions add depth. Cross-validating between them catches both regex false positives and LLM hallucinations.

6. **Canary before promote.** Never deploy a new model to 100% traffic without validating on real production data first. The held-out test set is a necessary but insufficient quality gate — production data is the ground truth.

7. **Observability at every layer.** Training metrics (W&B), inference metrics (GPU utilization, latency, throughput), and quality metrics (JSON validity, schema compliance, F1) together give complete visibility into model health from training through production.

---

*Document Version: 1.0 | Last Updated: March 2026 | System: LGS Tech Agentic AI Workflow*
