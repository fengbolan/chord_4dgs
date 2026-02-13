# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CHORD 4DGS: Text-guided 4D Gaussian Splatting animation via Score Distillation Sampling (SDS). Takes a static 3DGS PLY file + text prompt, optimizes per-frame deformations using a video diffusion model (Wan 2.2) to produce animated 3D scenes.

## Common Commands

```bash
# Training (always run via shell scripts, which set CUDA_VISIBLE_DEVICES and full params)
bash scripts/run_cat.sh    # or run_can.sh, run_human.sh, etc.

# Inference (default mode: orbit = 360° camera sweep + 4 fixed-view GIFs)
python inference.py --checkpoint outputs/<exp>/final_checkpoint.pt
python inference.py --checkpoint outputs/<exp>/checkpoint_step_04000.pt --output outputs/<exp>/inference_step4000

# Specific inference modes
python inference.py --checkpoint outputs/<exp>/final_checkpoint.pt --mode single --elevation 15 --azimuth 45
python inference.py --checkpoint outputs/<exp>/final_checkpoint.pt --mode multiview --num_azimuths 8
```

The Python interpreter is at `/mnt/workspace/fblan/miniforge3/envs/gsplat/bin/python`. There are no tests, linter, or build system — this is a research codebase.

## Directory Structure

```
├── core/                  # Renderer, SDS loss, regularization
├── models/                # GaussianModel, Deformation4D, Fenwick tree, control points
├── utils/                 # Utility helpers
├── scripts/               # Shell run scripts (run_cat.sh, run_can.sh, etc.)
├── visualization/         # Experiment visualization scripts
├── debug/                 # Pipeline test scripts and debug output images
├── train.py               # Main training entry point
├── inference.py            # Inference / rendering entry point
├── config.py              # TrainConfig dataclass
└── scene_configs.py       # Per-scene presets
```

## Architecture

**Data flow:** Static 3DGS PLY → `GaussianModel` → `Deformation4D` (LBS with control points) → `render_video` (gsplat) → SDS loss (Wan 2.2 diffusion) + regularization → optimize control point translations & rotations.

### Key modules

- **`models/deformation.py`** — Core deformation model. Two-level hierarchy: coarse control points (global motion) + fine control points (local detail). Uses Linear Blend Skinning (LBS) with RBF blending weights. `deform_all_frames()` computes live differentiable weights for training; `deform_frame()` uses cached detached weights for inference.
- **`models/fenwick_tree.py`** — Binary Indexed Tree storing cumulative per-frame delta transformations. Enables efficient prefix-sum queries and CHORD-style reinit (zeroing incremental deltas to copy a reference frame's state forward).
- **`models/control_points.py`** — Farthest Point Sampling initialization, RBF blending weight computation with learnable `log_sigma` parameters.
- **`models/gaussian_model.py`** — Loads PLY, stores means/quats/scales/opacities/SH. Static (not optimized) — only deformation parameters are trained.
- **`core/sds_loss.py`** — Wraps Wan 2.2 video diffusion model for SDS gradient computation. Handles noise scheduling, CFG, and tau annealing.
- **`core/renderer.py`** — gsplat-based differentiable rasterization. `render_video()` accepts per-frame viewmats `[T,4,4]` for camera-follow mode or shared `[4,4]`.
- **`core/regularization.py`** — Temporal smoothness (`L_temp = ||mu_t - mu_{t+1}||^2`) and ARAP spatial rigidity (SVD-based local rotation fitting).

### Configuration

`config.py` defines `TrainConfig` dataclass (~50 params). `scene_configs.py` provides per-scene presets (cat, dog, can, cactus) with PLY paths, text prompts, camera params, and control point counts. CLI args override preset values.

### Training design decisions

- **Coarse-to-fine**: Only coarse CPs optimized in first half; fine CPs added at `fine_start_ratio` (50%).
- **Multi-view SDS**: Each step renders `batch_size` (default 4) random camera views, each with independent forward/backward to avoid OOM.
- **Reinit**: At step 100, reference frame's deformation is copied to all later frames (CHORD paper strategy).
- **Camera follow**: For walking animations (cat, dog), camera tracks per-frame object centroid.
- **Color white point**: Dark PLY scenes (cat, dog, can) use `color_white_point=0.2` to remap `[0, 0.2] → [0, 1]`.

### Output structure

```
outputs/{timestamp}_{model}_{scene}_{iterations}/
  train.log, train_log.jsonl, tb/, wandb/
  checkpoint_step_*.pt, final_checkpoint.pt
  step_*_ele*_azi*.gif
```

## Key Conventions

- Checkpoints save `deformation_state_dict` + `config` dict. The static 3DGS is reloaded from `ply_path` at inference time.
- Scene rotations (`scene_rotate_x/y/z`) are applied at load time to orient PLY models upright. Most dark scenes need `color_white_point=0.2`.
- Regularization weights decay linearly: `spatial_weight_start → spatial_weight_end`, `temp_weight_start → temp_weight_end`.
- Learning rates decay log-linearly (exponential): `lr_deformation` 0.006 → 0.00006.
