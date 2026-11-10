# CreditMaster — Technical Documentation (continuity reference)

> **Audience:** AI models and developers who will continue the project. To *operate* the system, use [RUNBOOK.md](RUNBOOK.md).
> **State as of:** 2026-11-09. Branch `main`; last commit `6833896` (2026-08-19). The 2026-11-08/09 work (v2 pipeline, reorganisation, English renaming) is **not committed**.
> **Convention:** `[FACT]` verified in code/files/runs; `[INFERENCE]` deduction; `[PENDING]` known gap.

---

## 0. Summary (quick ingestion)

```yaml
project: CreditMaster — undergraduate thesis (TCC); the monograph is LaTeX in a separate repo (/home/serotini/LaTeX-TCC), in Portuguese
goal: local LLM that writes credit opinions in JSON, grounded in the institutional credit policy
current_model: creditmaster-rag-v2 (Ollama, GGUF Q4_K_M, 4.9 GB)
  base: unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
  technique: QLoRA r=16 + RAG (BM25 over the policy) + training with retrieved context (RAFT-style)
  trained: 2026-11-09, WSL2, 426 steps, 132.9 min, eval_loss 0.0034, peak VRAM 6.39/7.93 GB
labels: deterministic rule engine (creditmaster/rules.py), aligned with Dataset/credit_policy_en.md
classes: [Approve, Approve with Conditions, Suspend, Deny]
output: {recommendation, risk_level, authority_level, strengths, risk_factors, mitigants, policy_references, opinion}
data: LendingClub accepted_2007_to_2018Q4.csv → Dataset/rag_v2 (1700 train / 300 test)
entry_point: query.py (Windows + Ollama)
training_env: WSL2 Ubuntu-24.04, root, /root/tcc_env (torch 2.10+cu128, Unsloth 2026.8.10)
old_baseline: legacy_v1/ (no-RAG model whose labels had 135 contradictions with the policy)
language: English for all code, file names, flags, logs and docs; only text inside thesis charts/LaTeX table is Portuguese
main_pending: full evaluation of the 4 configurations with evaluate.py (n=300)
```

---

## 1. Context and goals

- Headless generative-AI service for credit analysis, **100% local** (LGPD/GDPR, banking secrecy).
- Thesis hypothesis: **fine-tuning** teaches format and jargon; **RAG** supplies the policy rules at decision time, reducing hallucination and making the opinion auditable (cited sections, authority level, mitigants).
- The dataset is synthetic over real LendingClub applications: labels come from the policy rules applied to indicators available at application time (t0). The outcome (`loan_status`) is **not** used for labelling, avoiding leakage.
- Model language: English.

---

## 2. Repository layout

```
D:\TCC
├── README.md                Public overview                                          [git]
├── RUNBOOK.md               Step-by-step operations guide                            [git]
├── documentation.md         This document                                            [git]
├── .gitignore                                                                        [git]
│
├── query.py                 Query 1 profile: RAG + Ollama + comparison with rules    [git] Windows
├── evaluate.py              Evaluate configurations on the test set                  [git] Windows/WSL
├── build_dataset.py         Build Dataset/rag_v2 (labels + RAG context)              [git] Windows/WSL
├── train.py                 QLoRA fine-tuning + GGUF export + Modelfile              [git] WSL
├── inference_demo.py        4-case inference demo via Unsloth                        [git] WSL
├── plot_metrics.py          Loss/LR curves + LaTeX table                             [git] WSL
│
├── creditmaster/            Core library (imported by the scripts)                   [git]
│   ├── rules.py             v2 rule engine = gold answers
│   ├── policy.py            Loads the policy and splits it into sections
│   ├── retriever.py         BM25 / Ollama-embedding retriever + query building
│   ├── prompts.py           System prompts and message building
│   └── schema.py            Output schema, parsing and validation
│
├── Dataset/
│   ├── credit_policy_en.md  Credit policy (RAG corpus; source of the rules)          [git]
│   ├── credit_policy_pt.md  Portuguese translation (not used by code)                [ignored]
│   ├── accepted_2007_to_2018Q4.csv  LendingClub, 1.6 GB, 151 columns (Kaggle)        [ignored]
│   └── rag_v2/              Output of build_dataset.py
│       ├── train.jsonl / test.jsonl   chat messages for SFT                          [ignored]
│       ├── test_meta.jsonl  raw profile + gold + retrieved sections (for evaluate.py) [ignored]
│       ├── audit.csv        audit of every sample                                    [ignored]
│       └── manifest.json    configuration, counts, policy hash                       [git]
│
├── examples/sample_profile.json  Sample profile for query.py                         [git]
├── setup/
│   ├── setup_wsl.sh         Installs the training environment in WSL                 [git]
│   ├── requirements-train.txt   Packages (training versions)                         [git]
│   ├── constraints-train.txt    Mandatory pins (torch 2.10, unsloth_zoo 2026.8.x)    [git]
│   └── wslconfig.example    .wslconfig template (11 GB RAM, 24 GB swap)              [git]
├── ollama/Modelfile.rag_v2  Generated by training; FROM must point to the .gguf      [ignored]
│
├── legacy_v1/               First model (no RAG) — thesis baseline
│   ├── generate_opinions_v1.py  v1 generator (old rules)                             [git]
│   ├── INSTRUCTIONS.md      Original v1 guide (Linux Mint paths)                     [git]
│   ├── dataset/credit_opinions_2000.jsonl  v1 dataset (1998 samples)                 [git]
│   ├── dataset/credit_opinions_audit.csv   v1 audit                                  [ignored]
│   ├── lora_model/          v1 LoRA adapters (160 MB)                                [ignored]
│   └── outputs/             v1 training log (Portuguese, historical), metrics, summary [git]
│
├── outputs/eval/<name>/     evaluate.py reports                                      [ignored]
├── charts/<run>/            plot_metrics.py output                                   [ignored]
└── models/                  (temporary) .gguf copy for the Ollama import             [ignored]
```

**Outside the repository** `[FACT]`:

| Location | Content |
|---|---|
| WSL `/root/tcc_env` | training venv (Python 3.12.3) |
| WSL `/root/.cache/huggingface` | 4-bit base model (5.5 GB) |
| WSL `/root/Output_TCC/rag_v2/outputs/` | `treinamento.log` (run made before the renaming; Portuguese messages), `training_metrics.csv`, `training_summary.json`, `checkpoint-425/426` |
| WSL `/root/Output_TCC/rag_v2/lora_model/` | v2 LoRA adapters (160 MB) |
| WSL `/root/Output_TCC/rag_v2/model_gguf/` | v2 bf16 merge (~15 GB; disposable, regenerable) |
| WSL `/root/Output_TCC/rag_v2/model_gguf_gguf/` | `Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf` (4.9 GB) |
| WSL `/root/.unsloth/llama.cpp` | llama.cpp built by Unsloth (GGUF export) |
| Ollama (Windows) | `creditmaster-rag-v2:latest`, `llama3.1:latest` (baseline) |
| `C:\Users\dud11\.wslconfig` | `memory=11GB`, `swap=24GB` |

---

## 3. Architecture and data flow

```
                 ┌──────────────────────────┐
 LendingClub CSV │  build_dataset.py        │  rules.assess()  → gold answer (8 fields)
 ───────────────▶│  quota sampling          │  retriever       → relevant sections (+ distractor in train)
 credit_policy   │                          │  prompts         → chat messages
 _en.md ────────▶└────────────┬─────────────┘
                              ▼
            Dataset/rag_v2/{train,test,test_meta}.jsonl
                              │
                              ▼  (WSL)
                 ┌──────────────────────────┐
                 │  train.py                │ → lora_model/ → GGUF Q4_K_M → ollama/Modelfile
                 │  QLoRA + resp-only loss  │ → outputs/ (log, metrics)
                 └────────────┬─────────────┘
                              ▼  ollama create
                   Ollama: creditmaster-rag-v2
                              ▲
   profile ─▶ retriever ─▶ prompts ─┘─▶ JSON ─▶ schema (validation) ◀── rules (gold)
              (query.py / evaluate.py)
```

**Inference = the same pipeline as training.** The model only sees context if the Python code builds it. Called directly (`ollama run`), it keeps the format but invents citations (observed on 2026-11-09: it cited "§5 Credit Score and Derogatory Marks", which does not exist).

---

## 4. `creditmaster/` package (API)

### 4.1 `rules.py` — v2 rule engine (gold answers)

Constants: `PROFILE_FIELDS` (13 LendingClub fields), `ESSENTIAL_FIELDS` (`fico_range_low`, `annual_inc`, `dti`), `RECOMMENDATIONS`, `RISK_LEVELS`, `AUTHORITY_LEVELS`, `POLICY_SECTIONS` (§3–§13 → title), `MITIGANT_*` (exact §12 wording), `PURPOSES`, `HOME_OWNERSHIP`.
Helpers: `to_float`, `clean_text`, `money`, `percent`, `integer`, `translate_purpose/home/term`, `parse_emp_years`, `section_ref(n)`.

`assess(profile: dict) -> (result: dict, diagnostics: dict)`
- `result`: the 8 schema keys (§7.3).
- `diagnostics`: `score`, `flags`, `risk_sections` (retriever gold), `missing_essential`.

**Scoring** (summed into `score`; factors go to `strengths`/`risk_factors`):

| Indicator | Condition → points (flag) | Section |
|---|---|---|
| FICO | missing: risk · ≥760 +3 · 720–759 +2 · 680–719 +1 · 660–679 −1 (`fico_moderate`) · 640–659 −2 (`fico_below_660`) · <640 −3 (`fico_weak`,`fico_below_660`) | §3 |
| DTI | missing: risk · ≤15 +2 · ≤30 +1 · ≤40 −2 (`dti_high`) · >40 −4 (`dti_high`,`dti_severe`) | §4 |
| Delinquencies 2y | missing −1 · >0 −(3+n) (`delinq`) · 0 +2 | §5 |
| Inquiries 6m | missing −1 · ≥3 −2 (`inq`) | §6 |
| Revolving | missing −1 · ≥80% −3 (`revol_high`) · ≤35% +2 | §7 |
| Amount/income | >45% −3 (`lti_high`) · ≤20% +2 | §8 |
| Term | 60 months −1 (`long_term`) | §9 |
| Employment | missing: risk (0) · ≥8 years +1 | §10 |
| Purpose | consolidation/credit card + (`revol_high`/`inq`/`dti_high`) −1 | §10 |
| Public records | >0 −3 (`pub_rec`) — **no section in the policy** | — |

**Decision:**
1. Missing FICO, income or DTI → `Suspend`, risk `High`.
2. `Approve` if FICO ≥ 680, no `dti_high`, delinquencies == 0, no `lti_high`, no `pub_rec` and score ≥ 4 → risk `Low` (score ≥ 7) or `Low to Moderate`.
3. `Deny` (risk `High`) if score < 0, or recent delinquency combined with FICO < 660, DTI > 30 or a 60-month term (§5).
4. Otherwise `Approve with Conditions`, risk `Moderate`.

**Authority (§13):** analyst → coordination (`dti_high`, `long_term`, `inq`) → management (`fico_below_660`, `delinq`, `lti_high`) → committee (≥2 severe flags among {`fico_below_660`, `delinq`, `lti_high`, `dti_severe`, `pub_rec`} or missing essential data).

**Mitigants (§12)**, only for *Approve with Conditions* and *Deny*: `lti_high` → amount reduction; `long_term` → term reduction; `dti_high` → income evidence; `revol_high`/`inq` → settle short-term debts; `fico_below_660`/`delinq`/`pub_rec` → co-obligor; authority ≥ management → additional collateral; none of the above → income evidence.

**`policy_references`:** sections of the risk factors + §11 + §12 (if mitigants) + §13 (if authority > analyst).
**`opinion`:** sentences built per rule, with citations ("Per §4, DTI above 30% precludes automatic approval.").

**On `fico_range_low`:** this is the applicant's own bureau score, not a policy minimum. LendingClub reports the score as a 4-point range (`fico_range_low`/`fico_range_high`); the lower bound is used as the conservative choice. The policy owns the *cut-offs* (§3); the score itself is an input, bought from the bureau (it cannot be computed locally). The prompt label "Minimum FICO" is misleading but is baked into the trained prompt; rename it only together with a retrain (`[PENDING]`, §11).

### 4.2 `policy.py`
- `DEFAULT_POLICY_PATH` = `Dataset/credit_policy_en.md`; `PINNED_SECTIONS = (11, 12, 13)`.
- `load_policy_chunks(path)` → list of `{id, title, ref, text}`, one per `## N. Title` heading. **Fails** if titles diverge from `rules.POLICY_SECTIONS` (guards against policy edits).
- `policy_sha256(path)`, `format_context(chunks)` → `### Retrieved credit policy context` block with `[§N Title]\n<text>`.

### 4.3 `retriever.py`
- `build_descriptors(profile)` → one sentence per detected signal (e.g. DTI > 30 → "high DTI above 30% debt burden..."). Needed because every profile has the same 13 fields: retrieving on the raw text would return the same sections every time.
- `PolicyRetriever(policy_path=None, backend="bm25"|"ollama", min_k=2, max_k=8, embed_model="nomic-embed-text", ollama_host=...)`
  - `retrieve(profile, top_k=None) -> (chunks, retrieved_ids)`: top-k candidate sections (k = number of descriptors clamped to [2, 8]) followed by the 3 pinned ones.
  - `rank(query)`, `config()`.
  - Candidates: the 13 non-pinned sections (§1–§10, §14–§16).
- BM25 in pure Python (`_BM25`); the `ollama` backend uses `/api/embed` (needs `ollama pull nomic-embed-text`; untested).

### 4.4 `prompts.py`
- `SYSTEM_PROMPT_V1` (exact v1), `SYSTEM_PROMPT_V2_RAG`, `SYSTEM_PROMPT_V2_NORAG` (both list the valid enum values of each field).
- `build_profile_text(profile)` → 14 lines "Prepare an objective credit opinion... / Requested amount: US$ ... / ... / Public records: N". Missing → `not provided`.
- `build_messages(profile, mode, chunks=None)`; `mode`: `rag` (context + `### Applicant profile` + profile), `norag`, `v1`.
- **The only place** prompts are built. Changing anything here requires rebuilding the dataset and retraining.

### 4.5 `schema.py`
- `OUTPUT_KEYS`, `LIST_KEYS`, `V1_LABEL_MAP` (`Conditional Approval` → `Approve with Conditions`, `Very High` → `High`).
- `parse_json_output(text)` (tolerates text around the JSON), `schema_errors(obj, required_keys)`, `section_numbers(refs)` (`"§4 ..."` → `{4}`).

---

## 5. Scripts (usage reference)

### 5.1 `query.py` — query one profile `[Windows]`
```
python query.py [--profile file.json] [--model creditmaster-rag-v2] [--host http://localhost:11434] [--timeout 300] [--json]
```
Without `--profile` it asks for the 13 fields (empty = missing; `36`/`60` become `"36 months"`). Pipeline: retrieval → `rag` prompt → `/api/generate` (`format=json`, `temperature=0`, `num_ctx=4096`) → schema validation → prints the sections sent, a ✔/✘ comparison with `rules.assess` (recommendation, risk, authority, references, mitigants) and the opinion. `--json` prints only the JSON. Profile file keys: `PROFILE_FIELDS` (unknown keys are rejected).
`[FACT]` 2026-11-09: with the sample profile, 5/5 compared fields correct (12.9 s); a profile without DTI correctly returned `Suspend`.

### 5.2 `evaluate.py` — evaluation `[Windows; WSL for the unsloth backend]`
```
python evaluate.py --backend {ollama,unsloth,rules} [--model NAME] [--lora DIR] --mode {rag,norag,v1}
                   [--name OUT] [--limit N] [--meta Dataset/rag_v2/test_meta.jsonl]
                   [--retriever bm25|ollama] [--min-k 2] [--max-k 8] [--num-ctx 4096] [--timeout 300]
python evaluate.py --retrieval-only
```
- Backends: `ollama` (`/api/generate`, `temperature=0`); `unsloth` (loads LoRA, greedy decoding); `rules` (oracle returning the gold answer; must score 100%).
- `--mode v1` uses the v1 prompt and maps v1 labels → v2; authority, mitigants and references do not apply.
- Metrics: `json_valid`, `schema_valid`, `recommendation_acc`, `recommendation_macro_f1`, `confusion_matrix`, `risk_level_acc`, `authority_acc`, `policy_ref_precision/recall` (by section number), `mitigants_jaccard`, `numeric_fidelity` (numbers in the opinion that appear in the prompt/gold), `latency_s`, `schema_error_counts`.
- Output: `outputs/eval/<name>/report.json` + `predictions.jsonl`.

### 5.3 `build_dataset.py` — v2 dataset
```
python build_dataset.py [--input CSV] [--policy MD] [--out-dir Dataset/rag_v2] [--total 2000]
       [--suspend-ratio 0.10] [--test-size 0.15] [--scan-rows 300000] [--retriever bm25|ollama]
       [--min-k 2] [--max-k 8] [--distractor-prob 0.5] [--seed 42]
```
Reads `scan-rows` rows, shuffles, fills quotas (600/600/600 + 200 Suspend). *Suspend* samples are synthetic: 1 essential field blanked (2 fields in 20% of cases; weights DTI 0.4, FICO 0.3, income 0.3). Stratified 85/15 split. ~50% of training samples get 1 distractor section at a random position; the test split never does. Writes `train.jsonl`, `test.jsonl`, `test_meta.jsonl`, `audit.csv`, `manifest.json`. Deterministic.

### 5.4 `train.py` — fine-tuning `[WSL]`
Configured through environment variables (table in the [RUNBOOK](RUNBOOK.md#quick-reference--trainpy-environment-variables)). Steps:
1. Load the 4-bit base model; LoRA r=16, α=16, dropout 0 on q,k,v,o,gate,up,down; "unsloth" gradient checkpointing.
2. llama-3.1 template; measure tokens and truncation; `train_on_responses_only` (loss on the assistant answer only).
3. SFT: lr 2e-4 cosine, 5% warmup, weight decay 0.01, `adamw_8bit`, bf16, seed 42, `prediction_loss_only=True`.
4. **Low-VRAM evaluation:** custom `prediction_step` with `UNSLOTH_RETURN_LOGITS=0` (§10, incident 3), `empty_cache()` before each evaluation, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
5. Checkpoint every `SAVE_STEPS`; `RESUME=1` resumes from the latest (log opened in append mode).
6. Writes `outputs/training.log`, `training_metrics.csv`, `training_summary.json`, `lora_model/`, GGUF Q4_K_M (non-fatal on failure; looks for the `.gguf` in `model_gguf/` and `model_gguf_gguf/`) and `ollama/Modelfile.<RUN_NAME>` (system prompt copied from the dataset; `num_ctx 4096`, `temperature 0.2`).

### 5.5 `inference_demo.py` — Unsloth inference `[WSL]`
Env: `OUTPUT_DIR`, `RUN_NAME`, `LORA_DIR`, `MODE` (`rag`|`v1`), `MAX_SEQ_LENGTH`. Runs 4 cases (prime, high risk, DTI > 30%, missing DTI) and prints JSON + retrieved sections + gold.

### 5.6 `plot_metrics.py` `[WSL]`
`OUTPUT_DIR=<run folder> python plot_metrics.py`. Reads `$OUTPUT_DIR/outputs/training_metrics.csv` and `training_summary.json`; writes `charts/<basename of OUTPUT_DIR>/loss_curve.{png,pdf}`, `learning_rate.png` and `metrics_table.tex` (GPU, VRAM, losses and time read from the summary). Env: `CHARTS_DIR`, `LATEX_IMG_DIR`. **Figure and table text is Portuguese by design** (embedded in the Portuguese thesis); copies into `LATEX_IMG_DIR` keep the names the thesis references (`curva_perda_treinamento.{png,pdf}`).

### 5.7 `setup/setup_wsl.sh`
apt packages (build-essential, cmake, libcurl, python3.12-venv) → venv → `torch==2.10.0 torchvision==0.25.0` (cu128 index) → `requirements-train.txt` with `-c constraints-train.txt` → remove `xformers` → verification. Run: `wsl -d Ubuntu-24.04 -u root -- bash /mnt/d/TCC/setup/setup_wsl.sh`.

### 5.8 `legacy_v1/generate_opinions_v1.py`
v1 generator, kept unchanged for reproducibility. Default paths are Linux Mint ones (`/home/serotini/TCC/...`): pass `--input/--jsonl/--csv`.

---

## 6. Credit policy (`Dataset/credit_policy_en.md`)

16 sections: §1 Objective, §2 Decision Principles, §3 FICO, §4 DTI, §5 Delinquency, §6 Inquiries, §7 Revolving, §8 Amount/Income, §9 Term, §10 Purpose/Housing/Employment, §11 Recommendation Matrix, §12 Mitigants, §13 Authority Levels, §14 Opinion Guidelines, §15 Governance, §16 Executive Positioning.
The §3–§13 titles are a **contract** with `rules.POLICY_SECTIONS` (checked in `policy.load_policy_chunks`). Editing the policy requires rebuilding the dataset and retraining.

---

## 7. Data

### 7.1 v2 dataset (`Dataset/rag_v2/manifest.json`)
1700 train (510/510/510/170) / 300 test (90/90/90/30). 885 training samples with a distractor. Tokens with template: mean 1477, max 2245. **0 policy violations** in the labels (verified 2026-11-08).

### 7.2 Format (train/test.jsonl)
```json
{"messages": [
  {"role": "system", "content": "<SYSTEM_PROMPT_V2_RAG>"},
  {"role": "user", "content": "### Retrieved credit policy context\n\n[§4 Repayment Capacity and DTI]\n...\n\n### Applicant profile\nPrepare an objective credit opinion...\nRequested amount: US$ 35,000.00\n..."},
  {"role": "assistant", "content": "{\"recommendation\": ..., \"opinion\": ...}"}
]}
```
`test_meta.jsonl`: `id`, `issue_d`, `loan_status`, `masked_fields`, `profile`, `gold`, `diagnostics`, `retrieved_ids`, `distractor_id`.

### 7.3 Output schema
```json
{"recommendation": "Approve | Approve with Conditions | Suspend | Deny",
 "risk_level": "Low | Low to Moderate | Moderate | High",
 "authority_level": "Credit analyst | Credit coordination | Credit management | Credit committee",
 "strengths": ["..."], "risk_factors": ["..."], "mitigants": ["<§12 wording>"],
 "policy_references": ["§4 Repayment Capacity and DTI", "§11 Recommendation Matrix"],
 "opinion": "Recommendation: ... Per §4, ..."}
```

---

## 8. Training runs and results

| Metric | v1 (2026-08-17, Linux Mint) | v2 RAG (2026-11-09, WSL2) |
|---|---|---|
| Train / validation samples | 1698 / 300 (random split) | 1700 / 300 (stratified) |
| Tokens per sample | ~500 | 1477 (max 2245) |
| Effective batch / steps / epochs | 8 (2×4) / 426 / 2 | 8 (1×8) / 426 / 2 |
| Duration | 75.6 min | 132.9 min (+ ~16 min GGUF export) |
| Peak VRAM | 6.44 / 7.52 GB | 6.39 / 7.93 GB |
| Train loss (step 5 → 425) | 1.327 → 0.0045 | 0.761 → 0.0032 |
| Final eval loss | 0.0045 | 0.0034 |
| GGUF | failed | ok (4.9 GB) |

`[INFERENCE]` v1 and v2 losses are **not comparable**: targets differ, and in v2 part of the answer can be copied from the context. The valid comparison is `evaluate.py`.

### 8.1 Evaluations run

| Run | Result |
|---|---|
| `--backend rules` (oracle, n=300) | 100% on everything; evaluator validated |
| `--retrieval-only` (n=300) | recall 0.999; full recall on 99.7% of samples; precision 0.54; 2.77 sections retrieved + 3 pinned |
| base `llama3.1` + RAG, n=6 (smoke test, 2026-11-08) | 50% schema valid, 33% recommendation accuracy |
| `creditmaster-rag-v2` via `query.py`, 3 manual profiles | correct decisions on all 3; on a FICO-600 profile it got authority and mitigants wrong |
| **4 configurations, n=300** | `[PENDING]` |

---

## 9. Training environment (WSL2)

- `Ubuntu-24.04` installed with `--no-launch` and used as **root** (no account created). If Ubuntu is opened from the Start menu, it creates/uses a regular user without access to `/root`: use `sudo -i` or `wsl -d Ubuntu-24.04 -u root`.
- The GPU reaches WSL through the Windows driver (617.14); 7.93 GB visible.
- Stack: Python 3.12.3, torch 2.10.0+cu128, Unsloth 2026.8.10, unsloth_zoo 2026.8.x, transformers 5.5.0, trl 0.24.0, peft 0.19.1, datasets 4.3.0, bitsandbytes 0.50.x.
- `.wslconfig`: 11 GB RAM + 24 GB swap (the default 7.5 + 2 GB was risky for the 16-bit merge during export).

---

## 10. Incidents and recorded decisions

| # | Date | Problem | Cause | Resolution |
|---|---|---|---|---|
| 1 | 2026-11-08 | v1 labels contradicted the policy (26 Approve with DTI > 30, 6 with amount > 45% of income, 103 missing-data cases without Suspend) | v1 rules written before the policy | v2 engine (`rules.py`); v1 isolated in `legacy_v1/` |
| 2 | 2026-11-08 | pip installed torch 2.11/cu130 and unsloth_zoo 2026.11 ("source anchor not found" warning) | unpinned transitive deps | `setup/constraints-train.txt` |
| 3 | 2026-11-08 | Training crashed at step 50, first evaluation: `CUDA driver error: device not ready` | Unsloth's `prediction_step` forces `UNSLOTH_RETURN_LOGITS=1` and materialises the logits (seq × 128256, fp32): ~1.7 GB extra | loss-only `prediction_step`; identical eval loss (0.2569 vs 0.2570), peak VRAM 7.26 → 6.33 GB |
| 4 | 2026-11-08 | Checkpoint lost in the crash | `save_steps` = `eval_steps` (saved after eval) | `SAVE_STEPS=25`, `RESUME=1` |
| 5 | 2026-11-09 | "No .gguf file found" | Unsloth 2026.8 writes to `<dir>_gguf/` | search both folders |
| 6 | 2026-11-09 | `ollama create` very slow | reading 4.9 GB through `\\wsl$` | copy the `.gguf` to D: first |
| 7 | 2026-11-09 | `ollama run` with free text cited non-existent sections and approved FICO 600 | model called without RAG and outside the training format | `query.py`; recorded as thesis evidence |
| 8 | 2026-11-08 | `NameError` in the demo script (`load_in_4bit`); LaTeX table `ValueError` and invented VRAM (7.2/8.0 GB) | v1 bugs | fixed |
| 9 | 2026-11-09 | Clean-up: removed (to the Recycle Bin) the v1 `model_gguf/` (15 GB), v1 checkpoints, `rejected_2007_to_2018Q4.csv`, the legacy Portuguese dataset `pareceres_credito_2000.*`, caches | approved by the author | — |
| 10 | 2026-11-09 | Project standardised to English: files renamed (`consultar.py`→`query.py`, `avaliar.py`→`evaluate.py`, `gerar_dataset_rag.py`→`build_dataset.py`, `treinamento.py`→`train.py`, `testar.py`→`inference_demo.py`, `gerar_graficos_metricas.py`→`plot_metrics.py`, `GUIA_EXECUCAO.md`→`RUNBOOK.md`, `exemplos/`→`examples/`, `graficos/`→`charts/`), flags (`--perfil`→`--profile`), logs and docs | author preference | thesis figure/table text kept in Portuguese |

---

## 11. Pending items and next steps

1. `[PENDING]` **Run the 4 full evaluations** (RUNBOOK §B) and bring the numbers into Chapter 4.
2. `[PENDING]` Decide the `pub_rec` rule: add a section to the policy or drop the rule (currently −3 with no citation).
3. `[PENDING]` Commit the 2026-11-08/09 work (nothing committed).
4. `[PENDING]` Next retrain: rename the prompt label "Minimum FICO" to something like "FICO score (lower bound of bureau range)" (requires rebuilding the dataset).
5. Optional: delete `/root/Output_TCC/rag_v2/model_gguf/` (15 GB) and checkpoints 425/426 in WSL if no longer needed.
6. Optional: compare `--retriever ollama` (embeddings) with BM25; if switching, rebuild the dataset and retrain.
7. Optional (thesis differentiator): RAG over similar historical cases with their real `loan_status`.
8. Optional: RAFT ablation (`--distractor-prob 0`) to measure robustness to irrelevant context.
9. Optional: REST API (FastAPI) on top of `query.py` (mentioned by the old README; does not exist).
10. Thesis discussion points: double counting (delinquencies, inquiries and revolving utilisation already feed the FICO score and are scored again); §3 cut-offs are on the FICO scale (300–850) and would need recalibration for Brazilian bureau scores (0–1000).

---

## 12. Legacy v1 (`legacy_v1/`)

First model, trained on 2026-08-17 on Linux Mint, **without RAG**. 5-field schema (`recommendation` ∈ Approve / Conditional Approval / Deny; `risk_level` including "Very High"; `strengths`, `risk_factors`, `opinion`, up to 3 items per list). Sampled from the first 10k CSV rows (temporal bias: Dec-2015), 1998 samples (666 per class). Can be evaluated with `evaluate.py --backend unsloth --lora legacy_v1/lora_model --mode v1`. Its GGUF export failed at the time and the 16-bit merge was deleted in the clean-up; to serve it in Ollama, re-merge from `legacy_v1/lora_model` + the base model.

---

## 13. History

| Date | Event |
|---|---|
| 2026-06-26 | Repository created; README translated to English |
| 2026-07-15 | Dataset migrated from Portuguese to English (`aec7b8b`, `e1a93b2`) |
| 2026-08-17 | v1 training (Linux Mint) |
| 2026-08-19 | Last commit: training script, charts, v1 dataset (`6833896`) |
| 2026-11-08 | Documentation; bug fixes; v2 pipeline (rules, RAG, dataset, evaluation); WSL2 environment; VRAM incident fixed |
| 2026-11-09 | v2 training completed; GGUF imported into Ollama; `query.py`; repository reorganisation and clean-up; English standardisation |
