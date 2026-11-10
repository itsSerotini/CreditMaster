"""
Build the thesis (Chapter 4) figures and LaTeX table from a training run's metrics.

Usage (WSL, from /mnt/d/TCC):
  OUTPUT_DIR=/root/Output_TCC/rag_v2 python plot_metrics.py   # v2 run
  OUTPUT_DIR=/mnt/d/TCC/legacy_v1    python plot_metrics.py   # v1 run

Reads   $OUTPUT_DIR/outputs/training_metrics.csv and training_summary.json
Writes  charts/<basename of OUTPUT_DIR>/ (e.g. charts/rag_v2/):
        loss_curve.{png,pdf}, learning_rate.png, metrics_table.tex
Optional env: CHARTS_DIR (output folder), LATEX_IMG_DIR (copy figures into the LaTeX repository).

Note: the text rendered inside the figures and the LaTeX table is intentionally in Portuguese,
because they are embedded in the Portuguese thesis (cap4.tex).
"""
import os
import json
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Plot style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
})

BASE_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/media/serotini/SSD-Sata/Output_TCC")

def resolve_input_path(ssd_subpath: str, local_path: str) -> str:
    ssd_full_path = os.path.join(BASE_OUTPUT_DIR, ssd_subpath)
    if os.path.exists(ssd_full_path):
        return ssd_full_path
    return local_path

RUN_LABEL = os.path.basename(os.path.normpath(BASE_OUTPUT_DIR))

def generate_visualizations(
    metrics_csv=None,
    metrics_json=None,
    output_dir=os.getenv("CHARTS_DIR", os.path.join("charts", RUN_LABEL)),
    latex_img_dir=os.getenv("LATEX_IMG_DIR", "/home/serotini/LaTeX-TCC/imagens")
):
    if metrics_csv is None:
        metrics_csv = resolve_input_path("outputs/training_metrics.csv", "outputs/training_metrics.csv")
    if metrics_json is None:
        metrics_json = resolve_input_path("outputs/training_summary.json", "outputs/training_summary.json")

    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(latex_img_dir):
        os.makedirs(latex_img_dir, exist_ok=True)

    if not os.path.exists(metrics_csv):
        print(f"WARNING: {metrics_csv} not found. Run train.py first.")
        return

    df = pd.read_csv(metrics_csv)
    
    # Training vs validation loss curve (labels in Portuguese for the thesis)
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    
    train_points = df.dropna(subset=["train_loss"])
    eval_points = df.dropna(subset=["eval_loss"])

    if not train_points.empty:
        ax.plot(
            train_points["step"],
            train_points["train_loss"],
            label="Training Loss ($L_{train}$)",
            color="#1f77b4",
            linewidth=2,
            alpha=0.85
        )

    if not eval_points.empty:
        ax.plot(
            eval_points["step"],
            eval_points["eval_loss"],
            label="Validation Loss ($L_{val}$)",
            color="#d62728",
            linewidth=2.2,
            marker="o",
            markersize=5,
            linestyle="--"
        )

    ax.set_title("Curva de Aprendizado - Fine-Tuning QLoRA (Llama 3.1 8B)", pad=12, fontweight="bold")
    ax.set_xlabel("Passos de Treinamento (Optimization Steps)", labelpad=8)
    ax.set_ylabel("Perda (Cross-Entropy Loss)", labelpad=8)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    loss_png = os.path.join(output_dir, "loss_curve.png")
    loss_pdf = os.path.join(output_dir, "loss_curve.pdf")
    fig.savefig(loss_png, dpi=300)
    fig.savefig(loss_pdf)
    plt.close(fig)

    print(f" -> Loss plot saved: {loss_png} and {loss_pdf}")

    # Copy into the LaTeX repository, keeping the file names the thesis already references
    if os.path.exists(latex_img_dir):
        shutil.copy(loss_png, os.path.join(latex_img_dir, "curva_perda_treinamento.png"))
        shutil.copy(loss_pdf, os.path.join(latex_img_dir, "curva_perda_treinamento.pdf"))
        print(f" -> Copied to LaTeX repository: {latex_img_dir}/curva_perda_treinamento.pdf")

    if "learning_rate" in df.columns:
        lr_points = df.dropna(subset=["learning_rate"])
        if not lr_points.empty:
            fig_lr, ax_lr = plt.subplots(figsize=(8.5, 4), dpi=300)
            ax_lr.plot(
                lr_points["step"],
                lr_points["learning_rate"],
                label="Learning Rate (Cosine Decay with Warmup)",
                color="#2ca02c",
                linewidth=2
            )
            ax_lr.set_title("Evolução da Taxa de Aprendizado (LR Schedule)", pad=12, fontweight="bold")
            ax_lr.set_xlabel("Passos de Treinamento (Steps)", labelpad=8)
            ax_lr.set_ylabel("Taxa de Aprendizado (Learning Rate)", labelpad=8)
            ax_lr.grid(True, linestyle=":", alpha=0.6)
            ax_lr.legend(loc="upper right")
            plt.tight_layout()
            
            lr_png = os.path.join(output_dir, "learning_rate.png")
            fig_lr.savefig(lr_png, dpi=300)
            plt.close(fig_lr)
            print(f" -> Learning rate plot saved: {lr_png}")

    # LaTeX summary table (content in Portuguese for the thesis)
    meta = {}
    if os.path.exists(metrics_json):
        with open(metrics_json, "r") as f:
            meta = json.load(f)

    initial_loss = train_points["train_loss"].iloc[0] if not train_points.empty else "N/A"
    final_train_loss = train_points["train_loss"].iloc[-1] if not train_points.empty else "N/A"
    final_eval_loss = eval_points["eval_loss"].iloc[-1] if not eval_points.empty else "N/A"
    duration_min = meta.get("total_duration_minutes", 0)
    train_samples = meta.get("train_samples", 1700)
    eval_samples = meta.get("eval_samples", 300)
    epochs = meta.get("epochs", 2)
    gpu_name = meta.get("gpu_name", "NVIDIA GeForce RTX 5060 Ti")
    total_vram = meta.get("total_vram_gb")
    peak_vram = meta.get("peak_vram_gb")

    def fmt_loss(value):
        return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)

    hardware_str = f"{gpu_name} ({total_vram:.2f} GB VRAM)" if total_vram else gpu_name
    if peak_vram and total_vram:
        vram_str = f"{peak_vram:.2f} GB / {total_vram:.2f} GB ({peak_vram / total_vram * 100:.0f}\\%)"
    else:
        vram_str = "N/A"

    latex_table = f"""% Tabela de Desempenho e Metricas de Treinamento gerada automaticamente
\\begin{{table}}[htbp]
\\centering
\\caption{{Métricas e Resultados Computacionais do Ajuste Fino (QLoRA)}}
\\label{{tab:metricas_treinamento}}
\\begin{{tabular}}{{ll}}
\\hline
\\textbf{{Parâmetro / Métrica}} & \\textbf{{Valor Obtido}} \\\\ \\hline
Modelo Base & Meta Llama 3.1 8B Instruct (4-bit NF4) \\\\
Hardware Utilizado & {hardware_str} \\\\
Amostras de Treino (85\\%) & {train_samples} \\\\
Amostras de Validação (15\\%) & {eval_samples} \\\\
Épocas de Treinamento & {epochs} \\\\
Perda Inicial de Treino ($L_{{inicial}}$) & {fmt_loss(initial_loss)} \\\\
Perda Final de Treino ($L_{{train}}$) & {fmt_loss(final_train_loss)} \\\\
Perda Final de Validação ($L_{{val}}$) & {fmt_loss(final_eval_loss)} \\\\
Tempo Total de Ajuste Fino & {duration_min:.2f} minutos \\\\
Pico de VRAM Alocada & {vram_str} \\\\ \\hline
\\end{{tabular}}
\\end{{table}}
"""
    table_path = os.path.join(output_dir, "metrics_table.tex")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"\n -> LaTeX Table exported to: {table_path}")

if __name__ == "__main__":
    generate_visualizations()
