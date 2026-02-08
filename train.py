import os
import sys
import math
import argparse
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from config import TrainConfig
from models.gaussian_model import GaussianModel
from models.deformation import Deformation4D
from core.renderer import render_gaussians, render_video
from core.regularization import temporal_regularization
from utils.camera_utils import orbit_camera, random_camera, get_intrinsics


def log_linear_decay(start, end, step, total):
    if total <= 1:
        return start
    ratio = step / (total - 1)
    return math.exp(math.log(start) * (1 - ratio) + math.log(end) * ratio)


def linear_decay(start, end, step, total):
    if total <= 1:
        return start
    ratio = step / (total - 1)
    return start + (end - start) * ratio


def save_gif(video_tensor, path, duration=200):
    frames = []
    for t in range(video_tensor.shape[0]):
        frame = video_tensor[t].detach().cpu().clamp(0, 1).numpy()
        frames.append(Image.fromarray((frame * 255).astype(np.uint8)))
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0)


def train(config: TrainConfig):
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device(config.device)

    # Load 3DGS
    print("Loading 3DGS model...")
    gs = GaussianModel()
    gs.load_ply(config.ply_path)
    gs.to(device)
    print(f"Loaded {gs.num_gaussians} gaussians")

    activated_opacities = gs.get_activated_opacities()
    activated_scales = gs.get_activated_scales()
    sh_coeffs = gs.get_sh_coeffs()
    sh_degree = int(math.sqrt(sh_coeffs.shape[1])) - 1

    # Initialize deformation
    print("Initializing 4D deformation...")
    deform = Deformation4D(
        gs,
        num_coarse=config.num_coarse_cp,
        num_fine=config.num_fine_cp,
        num_frames=config.num_frames
    ).to(device)
    print(f"Coarse CPs: {config.num_coarse_cp}, Fine CPs: {config.num_fine_cp}")

    # Load SDS
    use_real_sds = config.sds_model_name is not None
    sds = None
    if use_real_sds:
        from core.sds_loss import SDSLossWrapper
        sds = SDSLossWrapper(
            model_name=config.sds_model_name,
            device=config.device,
            guidance_scale=config.cfg_scale_start,
            min_tau=0.02, max_tau=0.98,
            total_iterations=config.total_iterations,
            target_h=480, target_w=832,
        )
        sds.load_model()
    else:
        print("Using fake SDS gradient (no video model loaded)")

    # Camera setup
    scene_center = gs.means.mean(dim=0).detach().cpu()
    fx, fy, cx, cy = get_intrinsics(config.fovy_deg, config.render_width, config.render_height)
    K = (fx, fy, cx, cy)
    vis_viewmat = orbit_camera(15, 45, config.camera_radius, target=scene_center).to(device)

    # Training loop
    print(f"\nStarting training for {config.total_iterations} iterations...")
    print(f"Text prompt: {config.text_prompt}")
    print(f"Render: {config.render_width}x{config.render_height}, {config.num_frames} frames")

    for step in range(config.total_iterations):
        use_fine = step >= config.total_iterations * config.fine_start_ratio

        # LR decay
        lr_def = log_linear_decay(config.lr_deformation, config.lr_deformation_end,
                                  step, config.total_iterations)
        lr_sc = log_linear_decay(config.lr_scale, config.lr_scale_end,
                                 step, config.total_iterations)

        param_groups = deform.get_optimizable_params(use_fine)
        for pg in param_groups:
            if 'sigma' in pg['name']:
                pg['lr'] = lr_sc
            elif 'rots' in pg['name']:
                pg['lr'] = lr_def * 0.5
            else:
                pg['lr'] = lr_def

        optimizer = torch.optim.Adam(param_groups)
        optimizer.zero_grad()

        # Random camera
        viewmat = random_camera(
            elevation_range=config.elevation_range,
            azimuth_range=(0, 360),
            radius_range=(config.camera_radius * 0.8, config.camera_radius * 1.2),
            target=scene_center
        ).to(device)

        # Deform all frames
        all_means, all_quats = deform.deform_all_frames(gs, use_fine=use_fine)

        # Render video
        video = render_video(
            all_means, all_quats,
            activated_scales, sh_coeffs, activated_opacities,
            viewmat, K, config.render_width, config.render_height,
            config.num_frames, sh_degree=sh_degree
        )

        # Regularization first (retain graph for SDS backward)
        temp_w = linear_decay(config.temp_weight_start, config.temp_weight_end,
                              step, config.total_iterations)
        L_temp = temporal_regularization(all_means)
        (temp_w * L_temp).backward(retain_graph=True)

        # SDS loss
        if use_real_sds and sds is not None:
            # Update guidance scale via annealing
            sds.guidance_scale = linear_decay(config.cfg_scale_start, config.cfg_scale_end,
                                              step, config.total_iterations)
            sds_loss, tau = sds.compute_sds_loss(
                video, config.text_prompt,
                negative_prompt="static, still, no motion, blurry, ugly",
                iteration=step
            )
            sds_loss.backward()
        else:
            sds_grad = torch.randn_like(video) * 0.01
            video.backward(sds_grad)

        optimizer.step()

        # Reinit: reset late frames at reinit_step
        if step == config.reinit_step:
            print(f"Step {step}: Reinitializing late frame deformations")

        # Logging
        if step % config.log_every == 0:
            sds_val = sds_loss.item() if (use_real_sds and sds is not None) else 0.0
            print(f"Step {step}/{config.total_iterations} | "
                  f"L_temp: {L_temp.item():.6f} | "
                  f"SDS: {sds_val:.6f} | "
                  f"LR: {lr_def:.6f} | Fine: {use_fine}")

        # Visualization
        if step % config.save_every == 0 or step == config.total_iterations - 1:
            with torch.no_grad():
                vis_means, vis_quats = deform.deform_all_frames(gs, use_fine=use_fine)
                vis_video = render_video(
                    vis_means, vis_quats,
                    activated_scales, sh_coeffs, activated_opacities,
                    vis_viewmat, K, config.render_width, config.render_height,
                    config.num_frames, sh_degree=sh_degree
                )
                gif_path = os.path.join(config.output_dir, f'step_{step:05d}.gif')
                save_gif(vis_video, gif_path)
                print(f"  Saved {gif_path}")

    print("\nTraining complete!")
    ckpt_path = os.path.join(config.output_dir, 'final_checkpoint.pt')
    torch.save({
        'deformation_state_dict': deform.state_dict(),
        'config': config.__dict__,
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply_path', type=str, default='../data/1.ply')
    parser.add_argument('--text_prompt', type=str, default='A cactus in a pot swaying left and right')
    parser.add_argument('--total_iterations', type=int, default=500)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--render_width', type=int, default=512)
    parser.add_argument('--render_height', type=int, default=288)
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--num_coarse_cp', type=int, default=80)
    parser.add_argument('--num_fine_cp', type=int, default=300)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--sds_model_name', type=str, default=None)
    parser.add_argument('--camera_radius', type=float, default=3.0)
    parser.add_argument('--fovy_deg', type=float, default=49.1)
    args = parser.parse_args()

    config = TrainConfig(**{k: v for k, v in vars(args).items() if v is not None})
    train(config)
