#!/usr/bin/env bash
# 下载推理所需权重并校验 SHA-256，哈希不符就删除并报错（checkpoint 是 pickle，加载时用了 weights_only=False，
# 所以只加载校验通过的文件）。在仓库根目录或任意位置运行均可。
#   bash setup/fetch_weights.sh
# HF_ENDPOINT 可改镜像：HF_ENDPOINT=https://hf-mirror.com bash setup/fetch_weights.sh
# 官方 Google Drive 里还有不预测头动的第 2 套权重，没有公开镜像，不在这里下载。
set -euo pipefail

cd "$(dirname "$0")/.."
HF="${HF_ENDPOINT:-https://huggingface.co}"

# 目标路径  下载地址  SHA-256
fetch() {
    local dest="$1" url="$2" sha="$3"
    if [[ -f "$dest" ]] && [[ "$(sha256sum "$dest" | cut -d' ' -f1)" == "$sha" ]]; then
        echo "已存在且校验通过：$dest"
        return
    fi
    mkdir -p "$(dirname "$dest")"
    echo "下载 $dest"
    curl -fL --retry 5 --retry-delay 3 -C - -o "$dest" "$url"
    local got
    got="$(sha256sum "$dest" | cut -d' ' -f1)"
    if [[ "$got" != "$sha" ]]; then
        echo "校验失败：$dest" >&2
        echo "  期望 $sha" >&2
        echo "  实际 $got" >&2
        rm -f "$dest"
        exit 1
    fi
    echo "校验通过：$dest"
}

# 去噪网络（第 1 套，预测头动）@110k，518 MB
fetch experiments/DPT/head-SA-hubert-WM/checkpoints/iter_0110000.pt \
    "$HF/multimodalart/diffposetalk/resolve/main/iter_0110000.pt" \
    3a46c89297530b2a5864823bad2df611d031c1b17c6aaf0799f75115b12cc794

# 系数统计量（反归一化用）
fetch datasets/HDTF_TFHP/lmdb/stats_train.npz \
    "$HF/multimodalart/diffposetalk/resolve/main/stats_train.npz" \
    266f0aa8b883e82997c0b47deac392e6abcbd2eb5b88ff66fe65eebb3c387761

# 风格编码器 @26k，3 MB（只有提取新风格向量时才用；demo/input/style 里已有现成的风格向量）
fetch experiments/SE/head-L4H4-T0.1-BS32/checkpoints/iter_0026000.pt \
    "$HF/DoozyWo/Diffposetalk/resolve/main/iter_0026000.pt" \
    47cad62e34ae5f6456d3d2946ded2dd8721c4d6f1ec349c2e9f76b34b2a963d4

# HuBERT 音频编码器：缓存到 ~/.cache/huggingface，需要先 conda activate dpt
if command -v python >/dev/null 2>&1 && python -c "import huggingface_hub" 2>/dev/null; then
    echo "缓存 facebook/hubert-base-ls960"
    HF_ENDPOINT="$HF" python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('facebook/hubert-base-ls960', allow_patterns=['*.json', '*.bin', '*.safetensors', '*.txt']))
"
else
    echo "未检测到装有 huggingface_hub 的 python：先 conda activate dpt 再重跑本脚本，以缓存 HuBERT" >&2
fi
