#!/usr/bin/env python3
"""
CreditMaster — query one applicant profile (RAG + model served by Ollama).

Does exactly what the model expects: retrieves the policy sections for the profile,
builds the training-time prompt, calls Ollama and prints the opinion next to the
rule-engine gold answer.

Usage (Windows, from D:\\TCC, with Ollama running):
  python query.py                                    # asks for each field
  python query.py --profile examples/sample_profile.json
  python query.py --profile my.json --json           # model JSON only (for integrations)
  python query.py --model llama3.1                   # any other Ollama model (e.g. baseline)

Standard library only, plus the local `creditmaster` package.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

from creditmaster.prompts import build_messages
from creditmaster.retriever import PolicyRetriever
from creditmaster.rules import PROFILE_FIELDS, assess
from creditmaster.schema import parse_json_output, schema_errors

# (field, question, hint)
QUESTIONS = [
    ("loan_amnt", "Requested amount (US$)", "e.g. 12000"),
    ("term", "Term", "36 or 60 (months)"),
    ("annual_inc", "Annual income (US$)", "e.g. 65000"),
    ("purpose", "Purpose", "debt_consolidation, credit_card, home_improvement, car, small_business, medical, other..."),
    ("home_ownership", "Housing status", "RENT, OWN, MORTGAGE, OTHER"),
    ("emp_length", "Employment length", "e.g. '< 1 year', '3 years', '10+ years'"),
    ("dti", "DTI (%)", "debt-to-income ratio, e.g. 18.5"),
    ("fico_range_low", "Applicant FICO score (lower bound of the bureau range)", "e.g. 700"),
    ("delinq_2yrs", "Delinquencies in the last 2 years", "e.g. 0"),
    ("inq_last_6mths", "Credit inquiries in the last 6 months", "e.g. 1"),
    ("revol_util", "Revolving utilization (%)", "e.g. 35"),
    ("open_acc", "Open accounts", "e.g. 8"),
    ("pub_rec", "Derogatory public records", "e.g. 0"),
]


def ask_profile():
    print("Enter the applicant profile (empty = 'not provided'; missing data counts against the applicant).\n")
    profile = {}
    for field, question, hint in QUESTIONS:
        value = input(f"{question} [{hint}]: ").strip()
        if field == "term" and value in ("36", "60"):
            value = f"{value} months"
        profile[field] = value
    return profile


def load_profile(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    unknown = set(data) - set(PROFILE_FIELDS)
    if unknown:
        sys.exit(f"Unknown fields in {path}: {', '.join(sorted(unknown))}. Valid fields: {', '.join(PROFILE_FIELDS)}")
    return {k: str(data.get(k, "")) for k in PROFILE_FIELDS}


def call_ollama(model, host, messages, timeout):
    payload = {
        "model": model,
        "system": messages[0]["content"],
        "prompt": messages[1]["content"],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 1024},
    }
    req = urllib.request.Request(f"{host}/api/generate", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"]


def show(label, model_value, gold_value):
    mark = "✔" if model_value == gold_value else "✘"
    print(f"  {mark} {label:<18} model: {model_value}")
    if model_value != gold_value:
        print(f"    {'':<18} rules: {gold_value}")


def main():
    parser = argparse.ArgumentParser(description="Query CreditMaster (RAG + Ollama) for one applicant profile.")
    parser.add_argument("--profile", help="JSON file with the profile fields (see examples/sample_profile.json)")
    parser.add_argument("--model", default="creditmaster-rag-v2", help="Ollama model name")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--json", action="store_true", help="Print only the JSON returned by the model")
    args = parser.parse_args()

    profile = load_profile(args.profile) if args.profile else ask_profile()

    chunks, retrieved = PolicyRetriever().retrieve(profile)
    messages = build_messages(profile, mode="rag", chunks=chunks)

    start = time.time()
    try:
        raw = call_ollama(args.model, args.host, messages, args.timeout)
    except urllib.error.URLError as exc:
        sys.exit(f"Could not reach Ollama at {args.host} ({exc}). Is it running? "
                 f"Does '{args.model}' appear in `ollama list`?")
    elapsed = time.time() - start
    opinion = parse_json_output(raw)

    if args.json:
        print(json.dumps(opinion if opinion is not None else {"raw_output": raw}, indent=2, ensure_ascii=False))
        return

    gold, _ = assess(profile)
    sections = ", ".join(f"§{i}" for i in retrieved) + " + §11, §12, §13 (always)"
    print("\n" + "=" * 78)
    print(f" Policy sections sent to the model: {sections}")
    print(f" Model: {args.model}  |  time: {elapsed:.1f}s")
    print("=" * 78)

    if opinion is None:
        print("The model did not return valid JSON. Raw response:\n" + raw)
        return
    errors = schema_errors(opinion)
    if errors:
        print("⚠ Unexpected format: " + "; ".join(errors))

    print("\nComparison with the rule-engine gold answer (creditmaster/rules.py):")
    for key in ("recommendation", "risk_level", "authority_level"):
        show(key, opinion.get(key), gold[key])
    show("policy_references", opinion.get("policy_references"), gold["policy_references"])
    show("mitigants", opinion.get("mitigants"), gold["mitigants"])

    print("\nStrengths:    " + ("; ".join(opinion.get("strengths") or []) or "-"))
    print("Risk factors: " + ("; ".join(opinion.get("risk_factors") or []) or "-"))
    print("\nOpinion:\n" + str(opinion.get("opinion", "")))


if __name__ == "__main__":
    main()
