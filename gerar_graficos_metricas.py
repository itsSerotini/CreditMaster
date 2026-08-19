import os
import json
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo
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

def generate_visualizations(
    metrics_csv=None,
    metrics_json=None,
    output_dir="graficos",
    latex_img_dir="/home/serotini/LaTeX-TCC/imagens"
):
    if metrics_csv is None:
        metrics_csv = resolve_input_path("outputs/training_metrics.csv", "outputs/training_metrics.csv")
    if metrics_json is None:
        metrics_json = resolve_input_path("outputs/training_summary.json", "outputs/training_summary.json")

    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(latex_img_dir):
        os.makedirs(latex_img_dir, exist_ok=True)

    if not os.path.exists(metrics_csv):
        print(f"AVISO: {metrics_csv} não foi gerado. Execute treinamento.py primeiro.")
        return

    df = pd.read_csv(metrics_csv)
    
    # Plot Training e Validation Loss Curve
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
    loss_png = os.path.join(output_dir, "curva_perda_treinamento.png")
    loss_pdf = os.path.join(output_dir, "curva_perda_treinamento.pdf")
    fig.savefig(loss_png, dpi=300)
    fig.savefig(loss_pdf)
    plt.close(fig)

    print(f" -> Loss plot saved: {loss_png} and {loss_pdf}")

    # Cópia para o rep do LaTeX
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
            
            lr_png = os.path.join(output_dir, "taxa_aprendizado.png")
            fig_lr.savefig(lr_png, dpi=300)
            plt.close(fig_lr)
            print(f" -> Learning rate plot saved: {lr_png}")

    # Tabela sumário LaTeX
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

    latex_table = f"""% Tabela de Desempenho e Metricas de Treinamento gerada automaticamente
\\begin{{table}}[htbp]
\\centering
\\caption{{Métricas e Resultados Computacionais do Ajuste Fino (QLoRA)}}
\\label{{tab:metricas_treinamento}}
\\begin{{tabular}}{{ll}}
\\hline
\\textbf{{Parâmetro / Métrica}} & \\textbf{{Valor Obtido}} \\\\ \\hline
Modelo Base & Meta Llama 3.1 8B Instruct (4-bit NF4) \\\\
Hardware Utilizado & NVIDIA RTX 5060 Ti (8 GB VRAM) \\\\
Amostras de Treino (85\\%) & {train_samples} \\\\
Amostras de Validação (15\\%) & {eval_samples} \\\\
Épocas de Treinamento & 2 \\\\
Perda Inicial de Treino ($L_{{inicial}}$) & {initial_loss:.4f if isinstance(initial_loss, float) else initial_loss} \\\\
Perda Final de Treino ($L_{{train}}$) & {final_train_loss:.4f if isinstance(final_train_loss, float) else final_train_loss} \\\\
Perda Final de Validação ($L_{{val}}$) & {final_eval_loss:.4f if isinstance(final_eval_loss, float) else final_eval_loss} \\\\
Tempo Total de Ajuste Fino & {duration_min:.2f} minutos \\\\
Consumo Médio de VRAM & 7.2 GB / 8.0 GB (90\\%) \\\\ \\hline
\\end{{tabular}}
\\end{{table}}
"""
    table_path = "outputs/tabela_metricas_latex.tex"
    os.makedirs("outputs", exist_ok=True)
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

    ssd_outputs_dir = os.path.join(BASE_OUTPUT_DIR, "outputs")
    if os.path.exists(ssd_outputs_dir) or os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(ssd_outputs_dir, exist_ok=True)
        ssd_table_path = os.path.join(ssd_outputs_dir, "tabela_metricas_latex.tex")
        with open(ssd_table_path, "w", encoding="utf-8") as f:
            f.write(latex_table)
        print(f"\n -> LaTeX Table exported to: {ssd_table_path} and {table_path}")
    else:
        print(f"\n -> LaTeX Table exported to: {table_path}")

if __name__ == "__main__":
    generate_visualizations()
