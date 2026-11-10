#!/usr/bin/env bash
# CreditMaster — training environment setup for WSL2 (Ubuntu 24.04) or native Linux.
#
# From Windows PowerShell:
#   wsl --install -d Ubuntu-24.04 --no-launch
#   wsl -d Ubuntu-24.04 -u root -- bash /mnt/d/TCC/setup/setup_wsl.sh
#
# The GPU is exposed to WSL2 by the Windows NVIDIA driver; do NOT install a Linux driver.
set -euo pipefail

VENV="${VENV:-$HOME/tcc_env}"
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # holds requirements/constraints

echo "==> System packages (build tools are needed by Unsloth's llama.cpp GGUF export)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3.12-venv python3-dev build-essential cmake git curl libcurl4-openssl-dev

echo "==> Virtualenv at $VENV"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip

echo "==> PyTorch 2.10.0 + CUDA 12.8"
"$VENV/bin/pip" install -q torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

echo "==> Training stack (constrained so torch/unsloth_zoo are not upgraded)"
"$VENV/bin/pip" install -q -r "$SETUP_DIR/requirements-train.txt" -c "$SETUP_DIR/constraints-train.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu128
# xformers wheels are built per torch version; a mismatched one breaks imports. Unsloth uses SDPA without it.
"$VENV/bin/pip" uninstall -y -q xformers 2>/dev/null || true

echo "==> Verification"
"$VENV/bin/python" - <<'EOF'
import torch
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| available", torch.cuda.is_available())
print("GPU", torch.cuda.get_device_name(0), "| bf16", torch.cuda.is_bf16_supported())
import unsloth, transformers, trl, peft, datasets
print("unsloth", unsloth.__version__, "| transformers", transformers.__version__, "| trl", trl.__version__,
      "| peft", peft.__version__, "| datasets", datasets.__version__)
EOF
echo "==> Done. Activate with: source $VENV/bin/activate"
