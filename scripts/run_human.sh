#!/bin/bash
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=1 /mnt/workspace/fblan/miniforge3/envs/gsplat/bin/python train.py \
  --ply_path /mnt/workspace/fblan/trajectory-generation/4dgs_generation/sds_data/characters/ply/tony.ply \
  --text_prompt "A person slowly brings both arms together in front of their chest, wrapping them inward as if giving a warm hug" \
  --sds_model_name Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --camera_radius 3.5 \
  --scene_rotate_x 90 \
  --color_white_point 0.2 \
  --total_iterations 8000 \
  --batch_size 4 \
  --num_frames 41 \
  --render_width 512 \
  --render_height 288 \
  --save_every 200 \
  --log_every 20 \
  --num_coarse_cp 50 \
  --num_fine_cp 500 \
  --temp_weight_start 9.6 \
  --temp_weight_end 1.6 \
  --spatial_weight_start 6000.0 \
  --spatial_weight_end 300.0 \
  --scene_name tony_hug \
  --device cuda:0
# Note: CUDA_VISIBLE_DEVICES=1 maps physical GPU 1 to cuda:0 inside the process
