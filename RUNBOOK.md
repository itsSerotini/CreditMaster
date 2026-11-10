# 🏦 CreditMaster — Runbook

Practical guide to use, evaluate and retrain CreditMaster on this machine (Windows 11 + WSL2 + RTX 5060 Ti 8 GB).
Technical details for every file: [documentation.md](documentation.md).

---

## What do you want to do?

| Goal | Runs on | Time | Section |
|---|---|---|---|
| **Query the model** with an applicant profile | Windows | ~10 s | [A](#a--query-the-model) |
| **Evaluate** and compare models (thesis results) | Windows (+ WSL for v1) | 30–60 min per configuration | [B](#b--evaluate-and-compare-models) |
| **Build charts and the LaTeX table** for a training run | WSL | ~10 s | [C](#c--charts-and-latex-table) |
| **Retrain** the model | WSL | ~2 h 30 min | [D](#d--retrain-the-model) |
| Set up a **new computer** | Windows + WSL | ~30 min | [E](#e--set-up-a-new-computer) |
| Fix an **error** | — | — | [F](#f--troubleshooting) |

### Current state (2026-11-09)
- ✅ `creditmaster-rag-v2` trained and imported into Ollama.
- ✅ Training environment ready in WSL (`Ubuntu-24.04`, root user, venv `/root/tcc_env`).
- ⏳ Full evaluation of the 4 configurations not run yet.

### Two important rules
1. **The model does not fetch the policy by itself.** RAG is done by the Python code (`creditmaster/` folder). Always use `query.py`. With `ollama run` the model still answers in JSON, but it **invents policy sections and decisions**.
2. **Training and Unsloth only run in WSL**, always as **root**. Querying and evaluation through Ollama run on Windows.

---

## A — Query the model

**Where:** PowerShell on Windows, with Ollama running.

```powershell
cd D:\TCC
python query.py
```
The script asks for the 13 profile fields (empty = not provided). Or use a file:
```powershell
python query.py --profile examples/sample_profile.json
```

**Output:**
1. The policy sections sent to the model (e.g. `§6, §10, §4, §3 + §11, §12, §13`).
2. Comparison with the rule-engine gold answer: ✔ when the model agrees, ✘ showing the expected value.
3. Strengths, risk factors and the full opinion.

**Useful options:**

| Option | Effect |
|---|---|
| `--json` | print only the model JSON (for integrations) |
| `--model llama3.1` | query another Ollama model (e.g. the base Llama, for comparison) |

**Profile format** (see [examples/sample_profile.json](examples/sample_profile.json)):

| Field | Meaning | Example |
|---|---|---|
| `loan_amnt` | requested amount (US$) | `35000` |
| `term` | loan term | `36 months` or `60 months` |
| `annual_inc` | annual income (US$) | `105000` |
| `purpose` | purpose (LendingClub code) | `credit_card`, `debt_consolidation`, `car`, `home_improvement`, `small_business`, `medical`, `other` |
| `home_ownership` | housing status | `RENT`, `OWN`, `MORTGAGE`, `OTHER` |
| `emp_length` | employment length | `< 1 year`, `6 years`, `10+ years` |
| `dti` | debt-to-income (%) | `31.5` |
| `fico_range_low` | applicant FICO score (lower bound of the bureau range) | `725` |
| `delinq_2yrs` | delinquencies in 2 years | `0` |
| `inq_last_6mths` | credit inquiries in 6 months | `3` |
| `revol_util` | revolving utilization (%) | `31` |
| `open_acc` | open accounts | `33` |
| `pub_rec` | derogatory public records | `0` |

Without FICO, income or DTI the correct answer is **Suspend** (policy §11).

---

## B — Evaluate and compare models

Measures how often each configuration is right on the same 300 test cases. These numbers go into the thesis (Chapter 4).

**On Windows** (PowerShell, `D:\TCC`, Ollama running):
```powershell
python evaluate.py --backend ollama --model llama3.1 --mode norag --name base_norag
python evaluate.py --backend ollama --model llama3.1 --mode rag --name base_rag
python evaluate.py --backend ollama --model creditmaster-rag-v2 --mode rag --name ft_rag
```

**In WSL** (v1 model, which has no GGUF and uses its LoRA adapters):
```bash
python evaluate.py --backend unsloth --lora legacy_v1/lora_model --mode v1 --name ft_v1
```

| Configuration | What it shows |
|---|---|
| `base_norag` | Llama without fine-tuning and without policy (baseline) |
| `base_rag` | effect of RAG alone |
| `ft_v1` | effect of fine-tuning alone (old model) |
| `ft_rag` | final proposal: fine-tuning + RAG |

**Tips:**
- Try `--limit 20` first (~3 min) before the full run.
- Results go to `outputs/eval/<name>/report.json` (metrics and confusion matrix) and `predictions.jsonl` (one answer per case).
- Quick checks without an LLM: `python evaluate.py --backend rules --name oracle` (must be 100%) and `python evaluate.py --retrieval-only` (RAG quality).

---

## C — Charts and LaTeX table

**In WSL** (the v2 training metrics live on the WSL disk):
```bash
OUTPUT_DIR=/root/Output_TCC/rag_v2 python plot_metrics.py
OUTPUT_DIR=/mnt/d/TCC/legacy_v1 python plot_metrics.py
```
Each command writes to `charts/<rag_v2 | legacy_v1>/`:
- `loss_curve.png` and `.pdf`: training vs validation loss
- `learning_rate.png`: learning-rate schedule
- `metrics_table.tex`: table ready for `cap4.tex`

The text inside the figures and the table is in Portuguese on purpose: they go into the Portuguese thesis. To copy the figures straight into the thesis repository, add `LATEX_IMG_DIR=/path/to/imagens` (the copy keeps the file names the thesis already references).

---

## D — Retrain the model

Only needed if you change the rules (`creditmaster/rules.py`), the policy (`Dataset/credit_policy_en.md`) or the retriever.

### D1. Enter WSL as root
```powershell
wsl -d Ubuntu-24.04 -u root
```
The prompt must end with `#`. If it shows `your_user@...$`, run `sudo -i`.

```bash
source /root/tcc_env/bin/activate
cd /mnt/d/TCC
```
The prompt now starts with `(tcc_env)`.

### D2. Rebuild the dataset (only if rules, policy or retriever changed)
```bash
python build_dataset.py
```
Expected: 1700 train / 300 test in `Dataset/rag_v2/` (takes seconds and is always identical, seed 42).

### D3. Train (~2 h 15 min)
Before starting: close games, videos and Ollama (they share the VRAM) and disable Windows sleep.

```bash
tmux new -s train
```
```bash
OUTPUT_DIR=/root/Output_TCC RUN_NAME=rag_v2 python train.py 2>&1 | tee /root/train_rag_v2.log
```
- `tmux` keeps training alive if the terminal closes. To come back: `wsl -d Ubuntu-24.04 -u root`, then `tmux attach -t train`.
- Healthy signs: `truncated (> 2560): 0`, `Step 5/426 ... VRAM: ~6.3GB`, and every 50 steps `[EVAL] Eval Loss` with VRAM ~6.4 GB.
- If it crashes: a checkpoint is saved every 25 steps. Resume by prefixing the same command with `RESUME=1` (and use `tee -a`).
- To stop: `Ctrl+C`.
- At the end: `TRAINING COMPLETED SUCCESSFULLY` and, ~16 min later, `GGUF export completed`.

### D4. Import into Ollama
**In WSL**, copy the GGUF to D: (reading it from WSL through Windows is very slow):
```bash
mkdir -p /mnt/d/TCC/models && cp /root/Output_TCC/rag_v2/model_gguf_gguf/*.Q4_K_M.gguf /mnt/d/TCC/models/creditmaster-rag-v2.Q4_K_M.gguf
```
Edit the first line of `D:\TCC\ollama\Modelfile.rag_v2`:
```
FROM D:\TCC\models\creditmaster-rag-v2.Q4_K_M.gguf
```
**In PowerShell:**
```powershell
ollama create creditmaster-rag-v2 -f D:\TCC\ollama\Modelfile.rag_v2
ollama list
```
After `create`, Ollama keeps its own copy and the `models/` folder can be deleted.

### D5. Check
```powershell
python query.py --profile examples/sample_profile.json
```

---

## E — Set up a new computer

> Already done on this machine. Requirements: NVIDIA GPU with the Windows driver (do not install a driver inside Ubuntu), Ollama for Windows, ~30 GB free on C: and the LendingClub CSV at `Dataset/accepted_2007_to_2018Q4.csv` (Kaggle).

1. Install Ubuntu without creating a user:
   ```powershell
   wsl --install -d Ubuntu-24.04 --no-launch
   ```
2. Give WSL more memory: copy [setup/wslconfig.example](setup/wslconfig.example) to `C:\Users\<you>\.wslconfig` and run:
   ```powershell
   wsl --shutdown
   ```
3. Install the training environment (pinned versions):
   ```powershell
   wsl -d Ubuntu-24.04 -u root -- bash /mnt/d/TCC/setup/setup_wsl.sh
   ```
   It must end with `torch 2.10.0+cu128 | CUDA 12.8 | available True`.
4. Build the dataset, train and import: section [D](#d--retrain-the-model).

⚠️ Never install packages into the venv without `-c setup/constraints-train.txt`: pip would upgrade torch and `unsloth_zoo` and break Unsloth.

---

## F — Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Model cites non-existent sections or makes absurd decisions | called without RAG (`ollama run`) | use `query.py` |
| `Could not reach Ollama` | Ollama closed or model missing | start Ollama; check `ollama list` |
| `/root/tcc_env/bin/activate: Permission denied` | entered WSL as a regular user | `sudo -i` or `wsl -d Ubuntu-24.04 -u root` |
| `sudo: source: command not found` | `source` is a shell builtin | run `sudo -i` first |
| `python: command not found` (WSL) | venv not activated | `source /root/tcc_env/bin/activate` |
| `CUDA driver error: device not ready` | VRAM exhausted | close GPU apps; use the current `train.py` (fixed) |
| Warning `source anchor not found ... sft_prepare_dataset` | `unsloth_zoo` too new | reinstall with `setup/setup_wsl.sh` |
| `ollama create` very slow | reading the GGUF through `\\wsl$` | copy it to D: first (D4) |
| Training dies during GGUF export | WSL out of RAM | check `.wslconfig` (E2); LoRA adapters are already saved |
| Training stopped midway | any interruption | `RESUME=1` (D3) |

---

## Quick reference — `train.py` environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OUTPUT_DIR` | `/media/serotini/SSD-Sata/Output_TCC` | artifacts root (in WSL use `/root/Output_TCC`) |
| `RUN_NAME` | `rag_v2` | run sub-folder and `ollama/Modelfile.<RUN_NAME>` name |
| `DATASET_TRAIN` / `DATASET_TEST` | `Dataset/rag_v2/train.jsonl` / `test.jsonl` | without `DATASET_TEST`, an 85/15 split is used |
| `MAX_SEQ_LENGTH` | 2560 | tokens per sample |
| `BATCH_SIZE` / `GRAD_ACCUM` | 1 / 8 | effective batch 8 |
| `EPOCHS` | 2 | |
| `EVAL_STEPS` / `SAVE_STEPS` | 50 / 25 | evaluation / checkpoint interval |
| `RESUME` | 0 | `1` resumes from the latest checkpoint |
| `MAX_STEPS` | -1 | `>0` for a quick test |
| `SKIP_GGUF` | 0 | `1` skips the GGUF export |

A run's artifacts live in `/root/Output_TCC/<RUN_NAME>/` (from Windows: `\\wsl$\Ubuntu-24.04\root\Output_TCC\<RUN_NAME>`): `outputs/` (log, metrics, checkpoints), `lora_model/` (adapters), `model_gguf/` (16-bit merge, ~15 GB, can be deleted) and `model_gguf_gguf/` (GGUF Q4_K_M).

> The v2 run of 2026-11-09 was made before the English renaming, so its log in WSL is named `outputs/treinamento.log` and its messages are in Portuguese. Future runs write `outputs/training.log` in English.
