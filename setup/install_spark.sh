#!/usr/bin/env bash
# 在 spark（aarch64 + CUDA 13）上建 conda 环境 dpt。
#   bash setup/install_spark.sh            # 推理与时延测试
#   bash setup/install_spark.sh --train    # 再装训练依赖
# 环境变量：ENV_NAME（默认 dpt）、PYTHON_VERSION（默认 3.10）、CONDA_ROOT（默认 ~/miniconda3）
set -euo pipefail

ENV_NAME="${ENV_NAME:-dpt}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
REQ=requirements-infer.txt
[[ "${1:-}" == "--train" ]] && REQ=requirements-train.txt

cd "$(dirname "$0")"   # setup/，使 requirements 里的 -r 相对路径生效

if ! command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "环境 $ENV_NAME 已存在，直接在其中补装依赖"
else
    # 只用 conda-forge，避免 defaults 频道的服务条款确认
    conda create -y -n "$ENV_NAME" -c conda-forge --override-channels "python=$PYTHON_VERSION" pip
fi

# 用环境里的 pip，不依赖 shell 是否 activate
PY="$(conda run -n "$ENV_NAME" which python)"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$REQ"

echo
echo "== 自检 =="
"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device="cuda")
    print("matmul ok", float((x @ x).sum().isfinite()))
EOF
echo
echo "完成。使用前：conda activate $ENV_NAME；接着 bash setup/fetch_weights.sh"
