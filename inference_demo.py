#!/usr/bin/env python3
"""
CreditMaster - Inference and Testing Script (v2 RAG)

Loads the fine-tuned LoRA adapters, retrieves the relevant credit-policy sections for each
applicant (same retriever used to build the training set), and prints the parsed JSON opinion
next to the rule-engine gold answer.

Env:
  OUTPUT_DIR  base artifacts dir (default /media/serotini/SSD-Sata/Output_TCC)
  RUN_NAME    training run (default rag_v2) -> $OUTPUT_DIR/$RUN_NAME/lora_model
  LORA_DIR    explicit adapters path (overrides the two above)
  MODE        rag (default) | v1 (legacy model trained without context)
"""

import json
import os

import torch
from unsloth import FastLanguageModel

from creditmaster.prompts import build_messages
from creditmaster.retriever import PolicyRetriever
from creditmaster.rules import assess
from creditmaster.schema import parse_json_output

# ==========================================
# 1. MODEL LOADING & INFERENCE OPTIMIZATION
# ==========================================
max_seq_length = int(os.getenv("MAX_SEQ_LENGTH", "2560"))
dtype = None
load_in_4bit = True  # Must match training (bnb-4bit base) to fit in 8GB VRAM
mode = os.getenv("MODE", "rag")

base_output_dir = os.getenv("OUTPUT_DIR", "/media/serotini/SSD-Sata/Output_TCC")
run_dir = os.path.join(base_output_dir, os.getenv("RUN_NAME", "rag_v2"))
lora_dir = os.getenv("LORA_DIR") or (
    os.path.join(run_dir, "lora_model") if os.path.exists(os.path.join(run_dir, "lora_model")) else "lora_model"
)

print(f"Loading fine-tuned CreditMaster model from '{lora_dir}' (mode={mode})...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = lora_dir,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model)

retriever = PolicyRetriever()  # BM25, same defaults as build_dataset.py


def analyze_credit_profile(profile: dict) -> dict:
    """Run retrieval + generation for a raw profile dict (LendingClub column names)."""
    chunks, retrieved_ids = retriever.retrieve(profile) if mode == "rag" else (None, [])
    messages = build_messages(profile, mode=mode, chunks=chunks)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 1024,
            temperature = 0.2,
            top_p = 0.9,
            use_cache = True,
            pad_token_id = tokenizer.eos_token_id,
        )

    response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    parsed = parse_json_output(response_text)
    return {"retrieved_sections": retrieved_ids, "output": parsed if parsed is not None else {"raw_output": response_text}}


# ==========================================
# 2. DEMO SAMPLE TEST CASES
# ==========================================
TEST_CASES = {
    "Case 1: Prime Borrower (Strong Profile)": dict(
        loan_amnt="12500", term=" 36 months", emp_length="6 years", home_ownership="OWN",
        annual_inc="85000", purpose="home_improvement", dti="12.30", fico_range_low="760",
        delinq_2yrs="0", inq_last_6mths="0", revol_util="22.40", open_acc="8", pub_rec="0"),
    "Case 2: High Risk Borrower (Overleveraged & Delinquent)": dict(
        loan_amnt="28000", term=" 60 months", emp_length="1 year", home_ownership="RENT",
        annual_inc="42000", purpose="debt_consolidation", dti="38.90", fico_range_low="615",
        delinq_2yrs="2", inq_last_6mths="4", revol_util="86.50", open_acc="14", pub_rec="1"),
    "Case 3: Good Score but DTI above 30% (policy §4)": dict(
        loan_amnt="35000", term=" 36 months", emp_length="6 years", home_ownership="MORTGAGE",
        annual_inc="105000", purpose="credit_card", dti="31.50", fico_range_low="725",
        delinq_2yrs="0", inq_last_6mths="3", revol_util="31.00", open_acc="33", pub_rec="0"),
    "Case 4: Missing DTI (policy §11 -> Suspend)": dict(
        loan_amnt="10000", term=" 36 months", emp_length="3 years", home_ownership="RENT",
        annual_inc="60000", purpose="car", dti="", fico_range_low="700",
        delinq_2yrs="0", inq_last_6mths="1", revol_util="50.00", open_acc="5", pub_rec="0"),
}

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING CREDITMASTER INFERENCE EVALUATION")
    print("=" * 70)

    for name, profile in TEST_CASES.items():
        gold, _ = assess(profile)
        print(f"\n▶ Testing: {name}")
        print(f"  Expected (rule engine): {gold['recommendation']} | {gold['risk_level']} | {gold['authority_level']}")
        print("-" * 50)
        print(json.dumps(analyze_credit_profile(profile), indent=2, ensure_ascii=False))
        print("=" * 70)
