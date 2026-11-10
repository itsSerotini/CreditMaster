# 🏦 CreditMaster v1 — Original Execution Guide (historical)

> **Historical document.** This is the guide written for the first model (v1, trained on 2026-08-17 on Linux Mint), translated to English. Paths (`/home/serotini/...`, `/media/serotini/SSD-Sata/...`) and file names refer to that setup and are **obsolete**. Scripts were renamed and moved: `Dataset/gerar_pareceres.py` → `legacy_v1/generate_opinions_v1.py`; `treinamento.py` → `train.py` (now v2); `gerar_graficos_metricas.py` → `plot_metrics.py`; `testar.py` → `inference_demo.py`. For the current pipeline, see [../RUNBOOK.md](../RUNBOOK.md).

---

## 1. Architecture overview

CreditMaster v1 was designed around these goals:
1. **Main language: English** technical finance wording (aligned with international credit-risk and lending-policy standards).
2. **No data leakage:** credit decisions in the dataset are derived **strictly from financial indicators at application time (t0)** (FICO score, DTI, delinquencies in the last 2 years, revolving utilisation, income commitment), with no ex-post use of the historical outcome.
3. **Standardised output:** the model answers in **structured JSON**, easy to integrate into banking APIs (`recommendation`, `risk_level`, `strengths`, `risk_factors`, `opinion`).
4. **Hardware optimisation:** 4-bit QLoRA training with **Unsloth**, tuned for the **NVIDIA GeForce RTX 5060 Ti (8 GB VRAM)**.
5. **Training mask (`train_on_responses_only`):** loss and gradients cover **only** the assistant's answer, so the model does not spend capacity learning to reproduce the user prompt.

---

## 2. Execution environment

The Python environment with all dependencies (`torch 2.10`, `CUDA 12.8`, `unsloth`, `trl`, `transformers`, `matplotlib`, `seaborn`) lived in:
```bash
/home/serotini/tcc_env
```
To activate:
```bash
source /home/serotini/tcc_env/bin/activate
cd /home/serotini/TCC
```

---

## 3. Step by step

### Step 1: Generate the opinion dataset (`gerar_pareceres.py`, now `generate_opinions_v1.py`)
Generates 2,000 objective, stratified opinions (approvals, conditional approvals and denials) from the credit-policy rules.
```bash
/home/serotini/tcc_env/bin/python Dataset/gerar_pareceres.py --total 2000
```
Outputs:
- `Dataset/credit_opinions_2000.jsonl`: dataset in the native Llama 3.1 message format (`system`, `user`, `assistant`).
- `Dataset/credit_opinions_audit.csv`: audit spreadsheet of the generated decisions.

### Step 2: QLoRA training (`treinamento.py`)
Fine-tunes **Llama 3.1 8B Instruct (4-bit)**:
- Splits the dataset into **85% train** (1,698 samples) and **15% validation** (300 samples).
- **Output directory:** heavy artifacts go to `/media/serotini/SSD-Sata/Output_TCC/` (configurable via `OUTPUT_DIR`).
- **Detailed real-time logging** to the terminal and `/media/serotini/SSD-Sata/Output_TCC/outputs/treinamento.log`: timestamps, elapsed time, **ETA**, peak **VRAM** (GB/%), learning rate and **eval loss**.
- Saves the LoRA adapters to `/media/serotini/SSD-Sata/Output_TCC/lora_model/`.
- Exports the quantised GGUF model to `/media/serotini/SSD-Sata/Output_TCC/model_gguf/`.
- Writes the Ollama `Modelfile` (in `/media/serotini/SSD-Sata/Output_TCC/Modelfile` and a copy in the root folder).
- Saves training metrics to `outputs/training_metrics.csv` and the summary to `outputs/training_summary.json`.
```bash
/home/serotini/tcc_env/bin/python treinamento.py
```
> Note added later: the GGUF export of this run did not complete; no `.gguf` or `Modelfile` was produced.

### Step 3: Charts and tables for LaTeX (`gerar_graficos_metricas.py`)
Generates high-resolution (300 DPI) charts and the LaTeX table for **Chapter 4**:
```bash
/home/serotini/tcc_env/bin/python gerar_graficos_metricas.py
```
Files generated in `graficos/` and copied to `/home/serotini/LaTeX-TCC/imagens/`:
- `curva_perda_treinamento.png` / `.pdf` (training vs validation loss).
- `taxa_aprendizado.png` (cosine decay with warmup).
- `outputs/tabela_metricas_latex.tex` (table ready for the thesis).

### Step 4: Python inference test (`testar.py`)
```bash
/home/serotini/tcc_env/bin/python testar.py
```

---

## 4. Using the model in Ollama (local chatbot)

`treinamento.py` was meant to export the merged model as **GGUF (Q4_K_M)** and create the `Modelfile`.

1. Create the model:
   ```bash
   ollama create creditmaster -f /home/serotini/TCC/Modelfile
   ```
2. Run the interactive chatbot:
   ```bash
   ollama run creditmaster
   ```
3. Paste a financial profile:
   ```text
   Prepare an objective credit opinion based on the financial and risk profile below.
   Requested amount: US$ 10,000.00
   Term: 36 months
   Annual income: US$ 75,000.00
   Purpose: debt consolidation
   Housing status: owned home
   Employment length: 5 years
   DTI: 14.50%
   Minimum FICO: 750
   Delinquencies in 2 years: 0
   Inquiries in 6 months: 0
   Revolving utilization: 20.00%
   Open accounts: 8
   Public records: 0
   ```
4. Or call the Ollama REST API:
   ```python
   import requests

   response = requests.post(
       "http://localhost:11434/api/generate",
       json={
           "model": "creditmaster",
           "prompt": "Prepare an objective credit opinion based on...\n...",
           "stream": False,
           "format": "json",
       },
   )
   print(response.json()["response"])
   ```

---

## 5. Adding the results to the thesis (`LaTeX-TCC`)

In `/home/serotini/LaTeX-TCC/cap4.tex`:

### Loss-curve figure
```latex
\begin{figure}[htbp]
    \centering
    \caption{Curva de convergência e perda durante o ajuste fino (QLoRA) do Llama 3.1 8B.}
    \includegraphics[width=0.85\textwidth]{imagens/curva_perda_treinamento.pdf}
    \fonte{Elaborado pelo autor (2026).}
    \label{fig:curva_loss}
\end{figure}
```
(The caption is in Portuguese because the thesis is written in Portuguese.)

### Metrics table
Open the generated `outputs/tabela_metricas_latex.tex` and paste it into `cap4.tex`, in the *Computational Performance and Training* section.

---

## 6. File layout (at the time)

```
/home/serotini/TCC/
├── Dataset/
│   ├── accepted_2007_to_2018Q4.csv     # LendingClub source data
│   ├── credit_policy_en.md             # Credit policy (English)
│   ├── credit_policy_pt.md             # Credit policy (Portuguese)
│   ├── credit_opinions_2000.jsonl      # Fine-tuning dataset
│   └── gerar_pareceres.py              # Extraction and labelling script
├── graficos/                           # High-resolution charts for the thesis
├── Modelfile                           # Ollama agent configuration (points to the SSD)
├── treinamento.py                      # Main fine-tuning script
├── testar.py                           # Local inference test
├── gerar_graficos_metricas.py          # Thesis asset generator
├── INSTRUCOES.md                       # This guide
└── README.md                           # Repository overview

/media/serotini/SSD-Sata/Output_TCC/    # High-capacity external storage
├── outputs/                            # Checkpoints, logs and tables
│   ├── treinamento.log
│   ├── training_metrics.csv
│   ├── training_summary.json
│   └── tabela_metricas_latex.tex
├── lora_model/                         # LoRA adapters
├── model_gguf/                         # Merged, quantised model
└── Modelfile                           # Ollama Modelfile copy
```
