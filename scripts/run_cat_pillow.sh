#!/bin/bash
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_multiobj.py \
  --scene cat_pillow \
  --sds_model_name Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --total_iterations 8000 \
  --batch_size 4 \
  --num_frames 41 \
  --render_width 512 \
  --render_height 288 \
  --save_every 200 \
  --log_every 20 \
  --temp_weight_start 9.6 \
  --temp_weight_end 1.6 \
  --accel_weight_start 0.5 \
  --accel_weight_end 0.5 \
  --spatial_weight_start 3000.0 \
  --spatial_weight_end 300.0 \
  --displacement_weight_start 500.0 \
  --displacement_weight_end 50.0 \
  --scene_name cat_pillow_41f_accel_disp \
  --device cuda:0
