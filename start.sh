#!/bin/bash
set -e

WORKSPACE="/workspace"
COMFY_DIR="$WORKSPACE/ComfyUI"
CUSTOM_NODES_DIR="$COMFY_DIR/custom_nodes"
MODELS_DIR="$COMFY_DIR/models"
WORKFLOW_JSON="$WORKSPACE/workflow.json"

# ---- Step 0: Python & Torch install ----
echo "Installing PyTorch with CUDA 12.8..."
pip install --upgrade pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128

# ---- Step 0.5: Install Nunchaku prebuilt wheel ----
echo "Installing Nunchaku prebuilt wheel for CUDA 12.8..."
pip install https://github.com/nathannicho/Nunchaku/releases/download/prebuilt-cu128/nunchaku-0.1.0+cu128-cp310-cp310-linux_x86_64.whl

# ---- Helper: install custom node from GitHub ----
install_node() {
    repo_url="$1"
    repo_name=$(basename "$repo_url" .git)
    target_dir="$CUSTOM_NODES_DIR/$repo_name"

    if [ ! -d "$target_dir" ]; then
        echo "Installing custom node: $repo_url"
        git clone "$repo_url" "$target_dir"
    else
        echo "Custom node already installed: $repo_name"
    fi
}

# ---- Helper: detect model type based on filename ----
get_target_dir() {
    filename=$(basename "$1")
    ext="${filename##*.}"
    name_lower=$(echo "$filename" | tr '[:upper:]' '[:lower:]')

    case "$ext" in
        safetensors|ckpt) 
            echo "$MODELS_DIR/checkpoints" ;;
        pt)
            echo "$MODELS_DIR/loras" ;;
        *)
            if [[ "$name_lower" == *"vae"* ]]; then
                echo "$MODELS_DIR/vae"
            elif [[ "$name_lower" == *"clip"* || "$name_lower" == *"t5"* ]]; then
                echo "$MODELS_DIR/clip"
            elif [[ "$name_lower" == *"emb"* || "$ext" == "bin" ]]; then
                echo "$MODELS_DIR/embeddings"
            else
                echo "$MODELS_DIR/others"
            fi
            ;;
    esac
}

# ---- Helper: download a model from a URL ----
download_model() {
    url="$1"
    dest_dir=$(get_target_dir "$url")
    mkdir -p "$dest_dir"

    echo "Downloading model: $url -> $dest_dir"
    if [[ $url == *"civitai.com"* ]]; then
        wget --content-disposition "$url" -P "$dest_dir"
    elif [[ $url == *"huggingface.co"* ]]; then
        huggingface-cli download "$url" --local-dir "$dest_dir" --resume-download
    else
        wget -O "$dest_dir/$(basename "$url")" "$url"
    fi
}

# ---- Step 1: Install ComfyUI + Manager ----
if [ ! -d "$COMFY_DIR" ]; then
    echo "Installing ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
fi
install_node "https://github.com/ltdrdata/ComfyUI-Manager"

# ---- Step 2: Install workflow-required custom nodes ----
install_node "https://github.com/nathannicho/ComfyUI-Nunchaku"
install_node "https://github.com/kijai/ComfyUI-KJNodes"
install_node "https://github.com/rgthree/rgthree-comfy"

# ---- Step 3: Parse workflow.json for model names ----
if [ -f "$WORKFLOW_JSON" ]; then
    echo "Parsing workflow.json for model names..."
    grep -oE '"[a-zA-Z0-9_\-/.]+\.safetensors"|"[a-zA-Z0-9_\-/.]+\.ckpt"|"[a-zA-Z0-9_\-/.]+\.pt"|"[a-zA-Z0-9_\-/.]+\.bin"' "$WORKFLOW_JSON" \
    | tr -d '"' \
    | sort -u \
    | while read model_file; do
        if [[ "$model_file" == http* ]]; then
            download_model "$model_file"
        else
            echo "⚠ Model '$model_file' not a URL. Please include in EXTRA_MODEL_URLS if you want auto-download."
        fi
    done
else
    echo "⚠ No workflow.json found in $WORKSPACE"
fi

# ---- Step 4: Download core FLUX models ----
echo "Downloading core FLUX models..."
mkdir -p "$MODELS_DIR/checkpoints"
wget -nc -O "$MODELS_DIR/checkpoints/flux_dev.safetensors" "https://huggingface.co/mit-han-lab/nunchaku-flux.1-dev/resolve/main/svdq-int4_r32-flux.1-dev.safetensors?download=true"
wget -nc -O "$MODELS_DIR/checkpoints/flux_fill.safetensors" "https://huggingface.co/mit-han-lab/nunchaku-flux.1-fill-dev/resolve/main/svdq-int4_r32-flux.1-fill-dev.safetensors?download=true"
wget -nc -O "$MODELS_DIR/checkpoints/flux_context.safetensors" "https://huggingface.co/mit-han-lab/nunchaku-flux.1-kontext-dev/resolve/main/svdq-int4_r32-flux.1-kontext-dev.safetensors?download=true"

# ---- Step 5: Handle EXTRA_MODEL_URLS ----
if [ ! -z "$EXTRA_MODEL_URLS" ]; then
    echo "Downloading extra models from EXTRA_MODEL_URLS..."
    IFS=',' read -ra URLS <<< "$EXTRA_MODEL_URLS"
    for url in "${URLS[@]}"; do
        download_model "$url"
    done
fi

# ---- Step 6: Install Python deps for nodes ----
echo "Installing remaining Python dependencies..."
pip install xformers transformers safetensors huggingface_hub requests

# ---- Step 7: Start ComfyUI ----
cd "$COMFY_DIR"
echo "Starting ComfyUI..."
python3 main.py --listen 0.0.0.0 --port 8188
