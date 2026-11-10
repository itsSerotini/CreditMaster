#!/usr/bin/env python3
"""
CreditMaster — evaluation harness for the 4-configuration comparison.

Configurations (examples):
  1. Base, no RAG       python evaluate.py --backend ollama --model llama3.1 --mode norag --name base_norag
  2. Base + RAG         python evaluate.py --backend ollama --model llama3.1 --mode rag   --name base_rag
  3. Fine-tuned v1      python evaluate.py --backend unsloth --lora legacy_v1/lora_model --mode v1 --name ft_v1  (WSL)
                        (or --backend ollama --model creditmaster --mode v1 once the v1 GGUF exists)
  4. Fine-tuned v2+RAG  python evaluate.py --backend ollama --model creditmaster-rag-v2 --mode rag --name ft_rag

Sanity / retrieval-only:
  python evaluate.py --backend rules --name oracle        # gold vs gold -> must be 100%
  python evaluate.py --retrieval-only                     # retriever recall/precision, no LLM

Metrics (written to outputs/eval/<name>/report.json + predictions.jsonl):
  json_valid, schema_valid, recommendation accuracy / macro-F1 / confusion matrix,
  risk_level & authority_level accuracy, policy_references precision/recall (by section number),
  mitigants Jaccard, numeric fidelity of the opinion (numbers present in the prompt), latency.

Mode 'v1' evaluates the legacy model: its labels are mapped to v2 (Conditional Approval ->
Approve with Conditions, Very High -> High) and fields it never learned are reported as N/A.
"""

import argparse
import json
import re
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from creditmaster.prompts import build_messages
from creditmaster.retriever import PolicyRetriever
from creditmaster.rules import RECOMMENDATIONS
from creditmaster.schema import (OUTPUT_KEYS, V1_LABEL_MAP, parse_json_output,
                                 schema_errors, section_numbers)

ROOT = Path(__file__).resolve().parent
V1_KEYS = ["recommendation", "risk_level", "strengths", "risk_factors", "opinion"]
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class OllamaBackend:
    def __init__(self, model, host, timeout, num_ctx):
        self.model, self.host, self.timeout, self.num_ctx = model, host, timeout, num_ctx

    def generate(self, messages):
        payload = {
            "model": self.model,
            "system": messages[0]["content"],
            "prompt": messages[1]["content"],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": 1024},
        }
        req = urllib.request.Request(f"{self.host}/api/generate", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())["response"]


class UnslothBackend:
    def __init__(self, lora_dir, max_seq_length):
        from unsloth import FastLanguageModel  # lazy: only available on the GPU training machine
        import torch
        self.torch = torch
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_dir, max_seq_length=max_seq_length, dtype=None, load_in_4bit=True)
        FastLanguageModel.for_inference(self.model)

    def generate(self, messages):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True,
                                      pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class RulesBackend:
    """Oracle: returns the gold answer. Validates the harness itself (expects 100%)."""
    def __init__(self):
        self.gold = None

    def generate(self, messages):
        return json.dumps(self.gold, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def numbers_in(text):
    vals = set()
    for raw in _NUMBER.findall(text or ""):
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        vals.add(round(v, 1))
    return vals


def numeric_fidelity(opinion, prompt_text, gold_opinion):
    """Share of numbers in the opinion that also appear in the prompt (or in the gold derivations)."""
    found = numbers_in(opinion)
    if not found:
        return None
    allowed = numbers_in(prompt_text) | numbers_in(gold_opinion)
    return sum(1 for n in found if n in allowed) / len(found)


def jaccard(a, b):
    a, b = {x.strip().lower() for x in a or []}, {x.strip().lower() for x in b or []}
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def prf(pred, gold):
    if not pred and not gold:
        return 1.0, 1.0
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 1.0
    return precision, recall


def macro_f1(pairs, labels):
    f1s = []
    for label in labels:
        tp = sum(1 for p, g in pairs if p == label and g == label)
        fp = sum(1 for p, g in pairs if p == label and g != label)
        fn = sum(1 for p, g in pairs if p != label and g == label)
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def load_meta(path, limit):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return rows[:limit] if limit else rows


def retrieval_only(meta_rows, retriever):
    precisions, recalls, sizes = [], [], []
    per_section = defaultdict(lambda: [0, 0])
    for row in meta_rows:
        _, ids = retriever.retrieve(row["profile"])
        gold = set(row["diagnostics"]["risk_sections"])
        p, r = prf(set(ids), gold)
        precisions.append(p)
        recalls.append(r)
        sizes.append(len(ids))
        for s in gold:
            per_section[s][1] += 1
            per_section[s][0] += s in ids
    return {
        "n": len(meta_rows),
        "retriever": retriever.config(),
        "precision_mean": mean(precisions),
        "recall_mean": mean(recalls),
        "full_recall_rate": mean([1.0 if r == 1.0 else 0.0 for r in recalls]),
        "avg_sections_retrieved": mean(sizes),
        "recall_by_section": {f"§{s}": round(h / t, 4) for s, (h, t) in sorted(per_section.items())},
    }


def evaluate(meta_rows, backend, mode, retriever, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = open(out_dir / "predictions.jsonl", "w", encoding="utf-8")
    required = V1_KEYS if mode == "v1" else OUTPUT_KEYS
    acc = defaultdict(list)
    rec_pairs = []
    errors = Counter()

    for i, row in enumerate(meta_rows, 1):
        profile, gold = row["profile"], row["gold"]
        chunks = retriever.retrieve(profile)[0] if mode == "rag" else None
        messages = build_messages(profile, mode=mode, chunks=chunks)
        if isinstance(backend, RulesBackend):
            backend.gold = gold

        start = time.time()
        try:
            raw = backend.generate(messages)
            failure = None
        except Exception as exc:  # network/timeout/model errors are recorded, not fatal
            raw, failure = "", f"{type(exc).__name__}: {exc}"
        latency = time.time() - start

        pred = parse_json_output(raw)
        if pred and mode == "v1":
            pred = {k: V1_LABEL_MAP.get(v, v) if isinstance(v, str) else v for k, v in pred.items()}
        errs = schema_errors(pred, required) if pred is not None else ["invalid JSON"]
        for e in errs:
            errors[e.split(":")[0]] += 1

        acc["json_valid"].append(1.0 if pred is not None else 0.0)
        acc["schema_valid"].append(1.0 if not errs else 0.0)
        acc["latency_s"].append(latency)
        p = pred or {}
        rec_pairs.append((p.get("recommendation"), gold["recommendation"]))
        acc["recommendation_acc"].append(1.0 if p.get("recommendation") == gold["recommendation"] else 0.0)
        acc["risk_level_acc"].append(1.0 if p.get("risk_level") == gold["risk_level"] else 0.0)
        acc["numeric_fidelity"].append(
            numeric_fidelity(p.get("opinion", ""), messages[1]["content"], gold["opinion"]) if pred else None)
        if mode != "v1":
            acc["authority_acc"].append(1.0 if p.get("authority_level") == gold["authority_level"] else 0.0)
            refs_p, refs_r = prf(section_numbers(p.get("policy_references")),
                                 section_numbers(gold["policy_references"]))
            acc["policy_ref_precision"].append(refs_p)
            acc["policy_ref_recall"].append(refs_r)
            acc["mitigants_jaccard"].append(jaccard(p.get("mitigants"), gold["mitigants"]))

        pred_file.write(json.dumps({"id": row["id"], "gold": gold["recommendation"],
                                    "pred": p.get("recommendation"), "schema_errors": errs,
                                    "latency_s": round(latency, 2), "failure": failure, "raw": raw},
                                   ensure_ascii=False) + "\n")
        pred_file.flush()
        print(f"[{i}/{len(meta_rows)}] gold={gold['recommendation']:<24} pred={str(p.get('recommendation')):<24} "
              f"{'OK ' if not errs else 'ERR'} {latency:5.1f}s")
    pred_file.close()

    labels = RECOMMENDATIONS
    confusion = {g: {pl: 0 for pl in labels + ["<invalid>"]} for g in labels}
    for pr, g in rec_pairs:
        confusion[g][pr if pr in labels else "<invalid>"] += 1

    report = {"n": len(meta_rows), "mode": mode}
    report.update({k: mean(v) for k, v in acc.items()})
    report["recommendation_macro_f1"] = round(macro_f1(rec_pairs, labels), 4)
    report["confusion_matrix"] = confusion
    report["schema_error_counts"] = dict(errors)
    if mode == "v1":
        report["note"] = "v1 model: authority/mitigants/policy_references not in its schema; Suspend unreachable."
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate CreditMaster configurations.")
    parser.add_argument("--meta", default=str(ROOT / "Dataset" / "rag_v2" / "test_meta.jsonl"))
    parser.add_argument("--backend", choices=["ollama", "unsloth", "rules"], default="rules")
    parser.add_argument("--model", default="llama3.1", help="Ollama model name")
    parser.add_argument("--lora", help="LoRA directory (unsloth backend)")
    parser.add_argument("--mode", choices=["rag", "norag", "v1"], default="rag")
    parser.add_argument("--name", help="Run name (output folder under outputs/eval/)")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N samples")
    parser.add_argument("--retriever", choices=["bm25", "ollama"], default="bm25",
                        help="Must match the retriever used to build the training set")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--max-seq-length", type=int, default=2560)
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    meta_rows = load_meta(args.meta, args.limit)
    retriever = PolicyRetriever(backend=args.retriever, min_k=args.min_k, max_k=args.max_k,
                                embed_model=args.embed_model, ollama_host=args.host)

    if args.retrieval_only:
        report = retrieval_only(meta_rows, retriever)
        name = args.name or f"retrieval_{args.retriever}"
    else:
        if args.backend == "ollama":
            backend = OllamaBackend(args.model, args.host, args.timeout, args.num_ctx)
        elif args.backend == "unsloth":
            if not args.lora:
                parser.error("--lora is required for the unsloth backend")
            backend = UnslothBackend(args.lora, args.max_seq_length)
        else:
            backend = RulesBackend()
        name = args.name or f"{args.backend}_{args.mode}"
        report = evaluate(meta_rows, backend, args.mode, retriever, ROOT / "outputs" / "eval" / name)
        report.update({"backend": args.backend, "model": args.model if args.backend == "ollama" else args.lora,
                       "retriever": retriever.config() if args.mode == "rag" else None})

    out_dir = ROOT / "outputs" / "eval" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "confusion_matrix"}, indent=2, ensure_ascii=False))
    print(f"Report: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
