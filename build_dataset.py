#!/usr/bin/env python3
"""
CreditMaster v2 — RAG-aware dataset generator (RAFT-style).

For each sampled LendingClub application:
  1. Gold label from the v2 rule engine (creditmaster/rules.py, aligned with the policy).
  2. Policy context from the retriever (creditmaster/retriever.py) — same one used at inference.
  3. Training samples may get one irrelevant "distractor" section so the model learns to
     ignore context that does not apply (RAFT). The test split never has distractors.

Outputs (default Dataset/rag_v2/):
  train.jsonl       chat messages (system, user with context, assistant JSON) for SFT
  test.jsonl        same format, held-out stratified split
  test_meta.jsonl   one line per test sample: raw profile, gold output, retrieved ids, loan_status
  audit.csv         every sample with decision, flags and retrieved sections
  manifest.json     configuration, class counts, policy hash (reproducibility)

Usage:
  python build_dataset.py --input Dataset/accepted_2007_to_2018Q4.csv
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from creditmaster.policy import DEFAULT_POLICY_PATH, PINNED_SECTIONS, policy_sha256
from creditmaster.prompts import build_messages
from creditmaster.retriever import PolicyRetriever
from creditmaster.rules import ESSENTIAL_FIELDS, PROFILE_FIELDS, RECOMMENDATIONS, assess, to_float

ROOT = Path(__file__).resolve().parent
EXTRA_FIELDS = ["id", "issue_d", "loan_status"]
MASK_WEIGHTS = {"dti": 0.4, "fico_range_low": 0.3, "annual_inc": 0.3}


def read_pool(path, scan_rows):
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        missing = [c for c in PROFILE_FIELDS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing columns in CSV: {', '.join(missing)}")
        for i, row in enumerate(reader):
            if i >= scan_rows:
                break
            if to_float(row.get("loan_amnt")) is None:
                continue
            rows.append({k: row.get(k, "") for k in PROFILE_FIELDS + EXTRA_FIELDS})
    return rows


def has_essentials(row):
    return all(to_float(row.get(f)) is not None for f in ESSENTIAL_FIELDS)


def mask_essentials(row, rng):
    """Create a 'Suspend' example by blanking 1 (80%) or 2 (20%) essential fields."""
    masked = dict(row)
    fields = list(MASK_WEIGHTS)
    n = 2 if rng.random() < 0.2 else 1
    chosen = set()
    while len(chosen) < n:
        chosen.add(rng.choices(fields, weights=[MASK_WEIGHTS[f] for f in fields])[0])
    for field in chosen:
        masked[field] = ""
    masked["_masked"] = ",".join(sorted(chosen))
    return masked


def sample(pool, total, suspend_ratio, rng):
    rng.shuffle(pool)
    n_suspend = round(total * suspend_ratio)
    per_class = (total - n_suspend) // 3
    remainder = (total - n_suspend) - per_class * 3
    quotas = {"Approve": per_class, "Approve with Conditions": per_class, "Deny": per_class, "Suspend": n_suspend}
    for label in ["Approve", "Approve with Conditions", "Deny"][:remainder]:
        quotas[label] += 1

    buckets = defaultdict(list)
    leftovers = []
    for row in pool:
        if not has_essentials(row):
            if len(buckets["Suspend"]) < quotas["Suspend"]:   # naturally incomplete rows first
                buckets["Suspend"].append(row)
            continue
        label = assess(row)[0]["recommendation"]
        if len(buckets[label]) < quotas[label]:
            buckets[label].append(row)
        else:
            leftovers.append(row)
        if all(len(buckets[k]) >= q for k, q in quotas.items() if k != "Suspend"):
            break

    while len(buckets["Suspend"]) < quotas["Suspend"] and leftovers:
        buckets["Suspend"].append(mask_essentials(leftovers.pop(), rng))

    for label, quota in quotas.items():
        if len(buckets[label]) < quota:
            print(f"WARNING: only {len(buckets[label])}/{quota} samples for {label!r}; increase --scan-rows.",
                  file=sys.stderr)
    return buckets, quotas


def stratified_split(buckets, test_size, rng):
    train, test = [], []
    for label in RECOMMENDATIONS:
        rows = list(buckets.get(label, []))
        rng.shuffle(rows)
        n_test = round(len(rows) * test_size)
        test.extend(rows[:n_test])
        train.extend(rows[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def build_sample(row, retriever, rng, distractor_prob):
    profile = {k: row.get(k, "") for k in PROFILE_FIELDS}
    gold, diag = assess(profile)
    chunks, retrieved_ids = retriever.retrieve(profile)
    distractor_id = None
    if distractor_prob > 0 and rng.random() < distractor_prob:
        options = [c for c in retriever.candidates if c["id"] not in retrieved_ids]
        if options:
            distractor = rng.choice(options)
            distractor_id = distractor["id"]
            indicator = chunks[: len(retrieved_ids)]
            indicator.insert(rng.randint(0, len(indicator)), distractor)
            chunks = indicator + chunks[len(retrieved_ids):]
    messages = build_messages(profile, mode="rag", chunks=chunks)
    messages.append({"role": "assistant", "content": json.dumps(gold, ensure_ascii=False)})
    meta = {
        "id": row.get("id"),
        "issue_d": row.get("issue_d"),
        "loan_status": row.get("loan_status"),
        "masked_fields": row.get("_masked", ""),
        "profile": profile,
        "gold": gold,
        "diagnostics": diag,
        "retrieved_ids": retrieved_ids,
        "distractor_id": distractor_id,
    }
    return {"messages": messages}, meta


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8", newline="") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate the CreditMaster v2 RAG fine-tuning dataset.")
    parser.add_argument("--input", default=str(ROOT / "Dataset" / "accepted_2007_to_2018Q4.csv"))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--out-dir", default=str(ROOT / "Dataset" / "rag_v2"))
    parser.add_argument("--total", type=int, default=2000)
    parser.add_argument("--suspend-ratio", type=float, default=0.10)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--scan-rows", type=int, default=300_000,
                        help="Rows read from the CSV before sampling (v1 read only total*5).")
    parser.add_argument("--retriever", choices=["bm25", "ollama"], default="bm25")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--distractor-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading up to {args.scan_rows:,} rows from {args.input} ...")
    pool = read_pool(args.input, args.scan_rows)
    print(f"Pool: {len(pool):,} rows")

    buckets, quotas = sample(pool, args.total, args.suspend_ratio, rng)
    train_rows, test_rows = stratified_split(buckets, args.test_size, rng)

    retriever = PolicyRetriever(args.policy, backend=args.retriever, min_k=args.min_k,
                                max_k=args.max_k, embed_model=args.embed_model)

    train, test, test_meta, audit = [], [], [], []
    for split, rows, distractor_prob in (("train", train_rows, args.distractor_prob), ("test", test_rows, 0.0)):
        for row in rows:
            record, meta = build_sample(row, retriever, rng, distractor_prob)
            (train if split == "train" else test).append(record)
            if split == "test":
                test_meta.append(meta)
            audit.append({
                "split": split, "id": meta["id"], "issue_d": meta["issue_d"], "loan_status": meta["loan_status"],
                "recommendation": meta["gold"]["recommendation"], "risk_level": meta["gold"]["risk_level"],
                "authority_level": meta["gold"]["authority_level"], "score": meta["diagnostics"]["score"],
                "flags": "|".join(meta["diagnostics"]["flags"]),
                "gold_sections": "|".join(map(str, meta["diagnostics"]["risk_sections"])),
                "retrieved_sections": "|".join(map(str, meta["retrieved_ids"])),
                "distractor": meta["distractor_id"] or "", "masked_fields": meta["masked_fields"],
                **meta["profile"],
            })

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "test.jsonl", test)
    write_jsonl(out_dir / "test_meta.jsonl", test_meta)
    with open(out_dir / "audit.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit[0].keys()))
        writer.writeheader()
        writer.writerows(audit)

    counts = {split: dict(Counter(a["recommendation"] for a in audit if a["split"] == split))
              for split in ("train", "test")}
    manifest = {
        "generator": "build_dataset.py",
        "args": vars(args),
        "retriever": retriever.config(),
        "pinned_sections": list(PINNED_SECTIONS),
        "policy_sha256": policy_sha256(args.policy),
        "quotas": quotas,
        "counts": counts,
        "n_train": len(train),
        "n_test": len(test),
        "n_masked_suspend": sum(1 for a in audit if a["masked_fields"]),
        "n_train_with_distractor": sum(1 for a in audit if a["split"] == "train" and a["distractor"]),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("counts", "n_train", "n_test", "n_masked_suspend",
                                                "n_train_with_distractor")}, indent=2))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
