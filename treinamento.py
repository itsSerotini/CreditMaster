#!/usr/bin/env python3
"""
CreditMaster - QLoRA Fine-Tuning Script with Detailed Logging & Metrics
Base Model: Llama 3.1 8B Instruct (4-bit quant) via Unsloth
Target Hardware: NVIDIA RTX 5060 Ti (8GB VRAM)

Features:
 - Comprehensive Logging Callback (Step time, ETA, GPU VRAM peak, train/eval loss)
 - Logs saved to outputs/treinamento.log & outputs/training_summary.json
 - Train/Validation Split (85% / 15%)
 - train_on_responses_only masking (loss computed solely on assistant's response)
 - LoRA adapters + GGUF Quantization (q4_k_m) + Ollama Modelfile creation
"""

import os
import sys
import json
import time
import datetime
import pandas as pd
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

# ==========================================
# 0. Diretórios e Logs
# ==========================================
BASE_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/media/serotini/SSD-Sata/Output_TCC")
output_dir = os.path.join(BASE_OUTPUT_DIR, "outputs")
lora_dir = os.path.join(BASE_OUTPUT_DIR, "lora_model")
gguf_dir = os.path.join(BASE_OUTPUT_DIR, "model_gguf")

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(lora_dir, exist_ok=True)
os.makedirs(gguf_dir, exist_ok=True)

log_file_path = os.path.join(output_dir, "treinamento.log")

def log_msg(msg: str, to_console: bool = True):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    formatted_msg = f"{timestamp} {msg}"
    if to_console:
        print(formatted_msg)
        sys.stdout.flush()
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(formatted_msg + "\n")

# Criação e inicialização do arquivo de log
with open(log_file_path, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("      🏦 CREDITMASTER - LOG DE TREINAMENTO DETALHADO (FINE-TUNING)      \n")
    f.write("=" * 80 + "\n\n")

# Inspeção do hardware
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected"
total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0

log_msg(f"Iniciando ambiente de treinamento...")
log_msg(f"Hardware Detectado: {gpu_name}")
log_msg(f"VRAM Total Disponível: {total_vram_gb:.2f} GB")
log_msg(f"PyTorch Version: {torch.__version__} | CUDA Version: {torch.version.cuda}")

# ==========================================
# 1. Configuração do Modelo e Hardware
# ==========================================
max_seq_length = 2048
dtype = None
load_in_4bit = True # Para caber em 8GB de VRAM

log_msg("Carregando o modelo base 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit'...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
log_msg("Modelo base carregado com sucesso.")

# ==========================================
# 2. Injetando Adaptadores LoRA (QLoRA)
# ==========================================
lora_r = 16
lora_alpha = 16
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

log_msg(f"Injetando adaptadores LoRA (Rank r={lora_r}, Alpha={lora_alpha}, Módulos={len(target_modules)})...")
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_r,
    target_modules = target_modules,
    lora_alpha = lora_alpha,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 42,
)
log_msg("Adaptadores QLoRA configurados com sucesso.")

# ==========================================
# 3. Preparando Dataset e Aplicando Chat Template
# ==========================================
log_msg("Aplicando Llama 3.1 Chat Template e carregando dataset...")
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

def format_chat_messages(examples):
    conversations = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        for conv in conversations
    ]
    return {"text": texts}

dataset_file = "/home/serotini/TCC/Dataset/credit_opinions_2000.jsonl"
raw_dataset = load_dataset("json", data_files=dataset_file, split="train")

split_dataset = raw_dataset.train_test_split(test_size=0.15, seed=42)
train_dataset = split_dataset["train"].map(format_chat_messages, batched=True)
eval_dataset = split_dataset["test"].map(format_chat_messages, batched=True)

num_train = len(train_dataset)
num_eval = len(eval_dataset)
log_msg(f"Dataset processado: {num_train} amostras de treino (85%) | {num_eval} amostras de validação (15%).")

# ==========================================
# 4. Log detalhado de treinamento
# ==========================================
class DetailedTrainingLogger(TrainerCallback):
    def __init__(self, log_path, total_steps=None):
        self.log_path = log_path
        self.total_steps = total_steps
        self.start_time = None
        self.last_step_time = None
        self.step_history = []
        self.initial_loss = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_step_time = self.start_time
        torch.cuda.reset_peak_memory_stats()
        log_msg("=" * 80)
        log_msg("INÍCIO DO TREINAMENTO QLoRA (SFT)")
        log_msg(f"Épocas Totais: {args.num_train_epochs} | Max Steps Estimados: {state.max_steps}")
        log_msg(f"Batch Size Efetivo: {args.per_device_train_batch_size * args.gradient_accumulation_steps} (Batch {args.per_device_train_batch_size} x Accum {args.gradient_accumulation_steps})")
        log_msg(f"Learning Rate Inicial: {args.learning_rate} | Scheduler: {args.lr_scheduler_type}")
        log_msg("=" * 80)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        now = time.time()
        elapsed_sec = now - self.start_time
        step = state.global_step
        max_steps = state.max_steps if state.max_steps > 0 else 1
        epoch = logs.get("epoch", state.epoch)
        
        # ETA
        steps_done = max(step, 1)
        time_per_step = elapsed_sec / steps_done
        remaining_steps = max(0, max_steps - step)
        eta_sec = remaining_steps * time_per_step

        # VRAM
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        vram_pct = (peak_vram_gb / total_vram_gb * 100) if total_vram_gb > 0 else 0

        # Loss & LR
        train_loss = logs.get("loss")
        eval_loss = logs.get("eval_loss")
        lr = logs.get("learning_rate", 0)

        record = {
            "step": step,
            "epoch": round(epoch, 3) if epoch else 0,
            "elapsed_seconds": round(elapsed_sec, 1),
            "peak_vram_gb": round(peak_vram_gb, 2),
            "learning_rate": lr,
        }

        if train_loss is not None:
            record["train_loss"] = round(train_loss, 4)
            if self.initial_loss is None:
                self.initial_loss = train_loss
            loss_info = f"Train Loss: {train_loss:.4f}"
        else:
            loss_info = "                  "

        if eval_loss is not None:
            record["eval_loss"] = round(eval_loss, 4)
            eval_info = f" | [EVAL] Eval Loss: {eval_loss:.4f} ⭐"
        else:
            eval_info = ""

        self.step_history.append(record)

        log_str = (
            f"Step {step:4d}/{max_steps} | Epoch {epoch:4.2f} | "
            f"{loss_info}{eval_info} | "
            f"LR: {lr:.2e} | "
            f"VRAM: {peak_vram_gb:.2f}GB ({vram_pct:.0f}%) | "
            f"Tempo: {elapsed_sec/60:4.1f}m | ETA: {eta_sec/60:4.1f}m"
        )
        log_msg(log_str)

    def on_train_end(self, args, state, control, **kwargs):
        total_time_sec = time.time() - self.start_time
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        
        log_msg("=" * 80)
        log_msg("🏁 TREINAMENTO CONCLUÍDO COM SUCESSO!")
        log_msg(f"Tempo Total de Execução: {total_time_sec/60:.2f} minutos ({total_time_sec:.1f} segundos)")
        log_msg(f"Total de Passos Executados: {state.global_step}")
        log_msg(f"Pico Máximo de VRAM Alocada: {peak_vram_gb:.2f} GB / {total_vram_gb:.2f} GB")
        if self.initial_loss and self.step_history:
            last_loss = next((r["train_loss"] for r in reversed(self.step_history) if "train_loss" in r), None)
            if last_loss:
                diff = ((self.initial_loss - last_loss) / self.initial_loss) * 100
                log_msg(f"Convergência de Perda: {self.initial_loss:.4f} -> {last_loss:.4f} (Redução de {diff:.1f}%)")
        log_msg("=" * 80)

# ==========================================
# 5. Configuração do SFT
# ==========================================
sft_config = SFTConfig(
    output_dir = output_dir,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    per_device_eval_batch_size = 2,
    eval_accumulation_steps = 4,
    
    num_train_epochs = 2,
    learning_rate = 2e-4,
    lr_scheduler_type = "cosine",
    warmup_ratio = 0.05,
    weight_decay = 0.01,
    
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    optim = "adamw_8bit",
    
    eval_strategy = "steps",
    eval_steps = 10,
    logging_steps = 5,
    save_strategy = "steps",
    save_steps = 50,
    save_total_limit = 2,
    
    seed = 42,
    report_to = "none",
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    args = sft_config,
)

log_msg("Aplicando 'train_on_responses_only' para isolar a perda apenas na resposta do assistente...")
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

detailed_logger = DetailedTrainingLogger(log_file_path)
trainer.add_callback(detailed_logger)

# ==========================================
# 6. Execução do Treinamento
# ==========================================
log_msg("Disparando rotina de otimização dos pesos...")
train_results = trainer.train()

# ==========================================
# 7. Export de Logs e Métricas
# ==========================================
log_msg("Exportando logs e métricas estruturadas para relatórios e gráficos do LaTeX...")

log_history = trainer.state.log_history
step_data = {}
for entry in log_history:
    s = entry.get("step")
    if s is None:
        continue
    if s not in step_data:
        step_data[s] = {"step": s, "epoch": entry.get("epoch", 0)}
    if "loss" in entry:
        step_data[s]["train_loss"] = entry["loss"]
        step_data[s]["learning_rate"] = entry.get("learning_rate")
    if "eval_loss" in entry:
        step_data[s]["eval_loss"] = entry["eval_loss"]

metrics_df = pd.DataFrame(list(step_data.values())).sort_values("step")
metrics_csv_path = os.path.join(output_dir, "training_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)

summary_meta = {
    "gpu_name": gpu_name,
    "total_vram_gb": total_vram_gb,
    "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0,
    "train_samples": num_train,
    "eval_samples": num_eval,
    "epochs": 2,
    "total_duration_seconds": train_results.metrics.get("train_runtime", 0),
    "total_duration_minutes": (train_results.metrics.get("train_runtime", 0)) / 60,
    "train_samples_per_second": train_results.metrics.get("train_samples_per_second", 0),
    "train_steps_per_second": train_results.metrics.get("train_steps_per_second", 0),
    "train_loss_final": train_results.metrics.get("train_loss", 0),
    "timestamp_completed": datetime.datetime.now().isoformat(),
}
with open(os.path.join(output_dir, "training_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary_meta, f, indent=2)

log_msg(f"Métricas salvas com sucesso em: {metrics_csv_path}")

# ==========================================
# 8. Exportação para testes sem dificuldades no terminal
# ==========================================
log_msg(f"Salvando adaptadores LoRA treinados em '{lora_dir}'...")
model.save_pretrained(lora_dir)
tokenizer.save_pretrained(lora_dir)
log_msg(f"Pesos LoRA salvos na pasta '{lora_dir}'")

log_msg(f"Exportando modelo quantizado em formato GGUF (Q4_K_M) para '{gguf_dir}'...")
model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method = "q4_k_m")
log_msg("Exportação GGUF concluída.")

# Create Ollama Modelfile
gguf_file_path = os.path.join(gguf_dir, "unsloth.Q4_K_M.gguf")
modelfile_content = f"""FROM {gguf_file_path}

TEMPLATE \"\"\"<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{{{ .Response }}}}<|eot_id|>\"\"\"

SYSTEM \"\"\"You are CreditMaster, an expert automated credit risk analyst. Analyze the applicant's financial and risk profile according to institutional credit policies. Output an objective, structured credit opinion in JSON format containing recommendation, risk_level, strengths, risk_factors, and executive opinion.\"\"\"

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
"""

modelfile_dest = os.path.join(BASE_OUTPUT_DIR, "Modelfile")
with open(modelfile_dest, "w", encoding="utf-8") as f:
    f.write(modelfile_content)

with open("Modelfile", "w", encoding="utf-8") as f:
    f.write(modelfile_content)

log_msg(f"Arquivo 'Modelfile' do Ollama gerado com sucesso em '{modelfile_dest}' e './Modelfile'.")
log_msg("Para executar: ollama create creditmaster -f Modelfile")
