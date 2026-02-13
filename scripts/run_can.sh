#!/bin/bash
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0 /mnt/workspace/fblan/miniforge3/envs/gsplat/bin/python train.py \
  --scene can \
  --sds_model_name Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --total_iterations 8000 \
  --batch_size 4 \
  --num_frames 41 \
  --render_width 512 \
  --render_height 288 \
  --save_every 200 \
  --log_every 20 \
  --num_coarse_cp 10 \
  --num_fine_cp 100 \
  --temp_weight_start 9.6 \
  --temp_weight_end 1.6 \
  --spatial_weight_start 12000.0 \
  --spatial_weight_end 600.0 \
  --scene_name can_squeezed_low_cp \
  --device cuda:0
