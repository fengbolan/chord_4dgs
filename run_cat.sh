#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
  --scene cat \
  --sds_model_name Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --total_iterations 8000 \
  --batch_size 4 \
  --num_frames 41 \
  --render_width 512 \
  --render_height 288 \
  --save_every 200 \
  --log_every 20 \
  --num_coarse_cp 50 \
  --num_fine_cp 500 \
  --temp_weight_start 19.2 \
  --temp_weight_end 1.6 \
  --spatial_weight_start 6000.0 \
  --spatial_weight_end 300.0 \
  --scene_name cat_41f_strong_reg \
  --device cuda:0
