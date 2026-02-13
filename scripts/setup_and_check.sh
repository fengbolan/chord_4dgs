#!/bin/bash
echo "=== Step 1: Check diffusers ==="
python3 -c "import diffusers; print('diffusers:', diffusers.__version__)" 2>/dev/null || {
    echo "Installing diffusers..."
    pip install -q diffusers transformers accelerate sentencepiece 2>&1 | tail -3
}

echo "=== Step 2: Check Wan model ==="
ls /mnt/cpfs/Wan2.2-T2V-A14B/ 2>/dev/null || echo "Wan model NOT at /mnt/cpfs/"

echo "=== Step 3: Check config ==="
cat /mnt/workspace/fblan/trajectory-generation/video_sds_eval/config/wan2.2_14b.yaml 2>/dev/null || echo "Config not found"

echo "=== Step 4: Test WanPipeline import ==="
python3 -c "from diffusers import WanPipeline; print('WanPipeline OK')" 2>&1

echo "=== Done ==="
