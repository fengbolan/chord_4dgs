#!/bin/bash
# CHORD 4DGS: Cactus flower blooming animation with Wan 2.2 14B SDS
# Usage: bash run_cactus_flower_wan2.2-14b.sh

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
echo "Prompt: A small bud slowly emerges from the top and gradually bursts into a stunning, vibrant pink and yellow flower"

CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --ply_path ../data/1.ply \
    --text_prompt "A small bud slowly emerges from the top and gradually bursts into a stunning, vibrant pink and yellow flower" \
    --sds_model_name "Wan-AI/Wan2.2-T2V-A14B-Diffusers" \
    --total_iterations 20000 \
    --num_frames 16 \
    --render_width 512 \
    --render_height 288 \
    --save_every 500 \
    --log_every 100 \
    --num_coarse_cp 5000 \
    --num_fine_cp 5000 \
    --camera_radius 3.0 \
    --device cuda:0 \
    --scene_name cactus_flower \
    --output_dir outputs/wan2.2-14b_cactus_flower

echo "=== Training complete ==="
echo "Output GIFs in outputs/wan2.2-14b_cactus_flower/"
