#!/bin/bash
# CHORD 4DGS: Cactus swaying animation with Wan 2.2 14B SDS
# Usage: bash run_cactus_sway.sh

set -e

cd "$(dirname "$0")"

# Step 1: Install dependencies if needed
echo "=== Checking dependencies ==="
python3 -c "import diffusers" 2>/dev/null || {
    echo "Installing diffusers..."
    pip install -q diffusers transformers accelerate sentencepiece
}

# Step 2: Run training
echo "=== Starting CHORD 4DGS training ==="
echo "Model: Wan 2.2 14B"
echo "Prompt: A cactus in a pot swaying left and right gently"

CUDA_VISIBLE_DEVICES=0,1 python3 train.py \
    --ply_path ../data/1.ply \
    --text_prompt "A cactus in a pot swaying left and right gently" \
    --sds_model_name "Wan-AI/Wan2.2-T2V-A14B-Diffusers" \
    --total_iterations 20000 \
    --num_frames 16 \
    --render_width 512 \
    --render_height 288 \
    --save_every 500 \
    --log_every 100 \
    --num_coarse_cp 10000 \
    --num_fine_cp 10000 \
    --camera_radius 3.0 \
    --device cuda:1 \
    --output_dir outputs/wan2.2-14b_cactus

echo "=== Training complete ==="
echo "Output GIFs in outputs/cactus_sway/"
