"""
Multi-object 4DGS training script.

Trains multiple objects with independent deformations in a shared scene,
using composite SDS guidance and contact/proximity losses.

Usage:
    python train_multiobj.py --scene cat_pillow
    python train_multiobj.py --scene cat_pillow --total_iterations 8000 --sds_model_name Wan-AI/Wan2.2-T2V-A14B-Diffusers
"""

import os
import sys
import math
import time
from datetime import datetime
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import TrainConfig, ObjectConfig
from scene_configs import get_scene_preset
from models.multi_object_scene import MultiObjectScene
from core.renderer import render_video
from core.regularization import contact_proximity_loss, impact_deformation_loss
from utils.camera_utils import orbit_camera, random_camera, get_intrinsics
from train import TrainLogger, log_linear_decay, linear_decay, save_gif, _make_model_short_name, _ts


def train_multiobj(config: TrainConfig):
    # Validate multi-object config
    if not config.objects:
        raise ValueError("No objects defined. Use --scene with a multi-object preset (e.g. cat_pillow).")

    # Build ObjectConfig list
    obj_configs = []
    for obj_dict in config.objects:
        # Convert list position_offset to tuple if needed
        d = dict(obj_dict)
        if 'position_offset' in d and isinstance(d['position_offset'], list):
            d['position_offset'] = tuple(d['position_offset'])
        obj_configs.append(ObjectConfig(**d))

    # Auto-generate output_dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_short = _make_model_short_name(config.sds_model_name) if config.sds_model_name else "nosds"
    scene = getattr(config, 'scene_name', 'default')
    run_name = f"{timestamp}_{model_short}_{scene}_{config.total_iterations}"
    config.output_dir = os.path.join(config.output_dir, run_name)
    os.makedirs(config.output_dir, exist_ok=True)
    if config.wandb_run_name is None:
        config.wandb_run_name = run_name

    # Tee stdout/stderr to both terminal and train.log
    log_path = os.path.join(config.output_dir, 'train.log')
    log_file = open(log_path, 'a', buffering=1)
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr

    class _TeeWriter:
        """Write to both terminal and log file simultaneously."""
        def __init__(self, terminal, logfile):
            self.terminal = terminal
            self.logfile = logfile
        def write(self, msg):
            self.terminal.write(msg)
            self.logfile.write(msg)
        def flush(self):
            self.terminal.flush()
            self.logfile.flush()
        def isatty(self):
            return self.terminal.isatty()
        def fileno(self):
            return self.terminal.fileno()

    sys.stdout = _TeeWriter(_orig_stdout, log_file)
    sys.stderr = _TeeWriter(_orig_stderr, log_file)

    device = torch.device(config.device)

    # Initialize logger
    config_dict = {
        k: str(v) if isinstance(v, (tuple, list)) else v
        for k, v in config.__dict__.items()
    }
    logger = TrainLogger(
        config.output_dir,
        config_dict=config_dict,
        use_wandb=config.use_wandb,
        wandb_project=config.wandb_project,
        wandb_run_name=config.wandb_run_name,
    )

    # Build multi-object scene
    print(f"[{_ts()}] Building multi-object scene with {len(obj_configs)} objects...")
    scene = MultiObjectScene(
        obj_configs,
        num_frames=config.num_frames,
        global_num_coarse=config.num_coarse_cp,
        global_num_fine=config.num_fine_cp,
        K_neighbors=config.K_neighbors,
        device=config.device,
    )

    for i, oc in enumerate(obj_configs):
        gs = scene.gaussian_models[i]
        print(f"[{_ts()}]   Object '{oc.name}': {gs.num_gaussians} gaussians, "
              f"offset={oc.position_offset}, rotate_x={oc.scene_rotate_x}")
    print(f"[{_ts()}] Total gaussians: {scene.total_gaussians}")

    # Get static rendering properties (concatenated)
    activated_scales, sh_coeffs, activated_opacities, sh_degree = scene.get_static_properties()

    # Global color white point (min across objects)
    color_white_point = scene.get_global_color_white_point()
    # Override with config-level if set
    if config.color_white_point != 1.0:
        color_white_point = config.color_white_point

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
            target_h=config.render_height, target_w=config.render_width,
        )
        sds.load_model()
    else:
        print(f"[{_ts()}] Using fake SDS gradient (no video model loaded)")

    # Camera setup
    scene_center = scene.scene_center().detach().cpu()
    fx, fy, cx, cy = get_intrinsics(config.fovy_deg, config.render_width, config.render_height)
    K = (fx, fy, cx, cy)
    vis_views = [
        ('ele15_azi045', 15, 45),
        ('ele15_azi180', 15, 180),
        ('ele15_azi270', 15, 270),
    ]
    vis_viewmats = {
        name: orbit_camera(ele, azi, config.camera_radius, target=scene_center).to(device)
        for name, ele, azi in vis_views
    }

    # Training loop
    print(f"\n[{_ts()}] Starting multi-object training for {config.total_iterations} iterations...")
    print(f"[{_ts()}] Text prompt: {config.text_prompt}")
    print(f"[{_ts()}] Render: {config.render_width}x{config.render_height}, {config.num_frames} frames")
    print(f"[{_ts()}] Objects: {[oc.name for oc in obj_configs]}")

    # Create optimizer
    use_fine = False
    param_groups = scene.get_optimizable_params(use_fine)
    for pg in param_groups:
        if 'sigma' in pg['name']:
            pg['lr'] = config.lr_scale
        elif 'rots' in pg['name']:
            pg['lr'] = config.lr_deformation * 0.5
        else:
            pg['lr'] = config.lr_deformation
    optimizer = torch.optim.Adam(param_groups)

    t_start = time.time()

    for step in range(config.total_iterations):
        step_t0 = time.time()
        new_use_fine = step >= config.total_iterations * config.fine_start_ratio

        # Rebuild optimizer when fine stage starts
        if new_use_fine and not use_fine:
            use_fine = True
            param_groups = scene.get_optimizable_params(use_fine)
            for pg in param_groups:
                if 'sigma' in pg['name']:
                    pg['lr'] = config.lr_scale
                elif 'rots' in pg['name']:
                    pg['lr'] = config.lr_deformation * 0.5
                else:
                    pg['lr'] = config.lr_deformation
            optimizer = torch.optim.Adam(param_groups)
            print(f"[{_ts()}] Step {step}: Fine stage started, optimizer rebuilt with fine params")

        # LR decay
        lr_def = log_linear_decay(config.lr_deformation, config.lr_deformation_end,
                                  step, config.total_iterations)
        lr_sc = log_linear_decay(config.lr_scale, config.lr_scale_end,
                                 step, config.total_iterations)
        for pg in optimizer.param_groups:
            if 'sigma' in pg['name']:
                pg['lr'] = lr_sc
            elif 'rots' in pg['name']:
                pg['lr'] = lr_def * 0.5
            else:
                pg['lr'] = lr_def

        optimizer.zero_grad()

        # Reg weight decay
        temp_w = linear_decay(config.temp_weight_start, config.temp_weight_end,
                              step, config.total_iterations)
        spatial_w = linear_decay(config.spatial_weight_start, config.spatial_weight_end,
                                 step, config.total_iterations)
        accel_w = linear_decay(config.accel_weight_start, config.accel_weight_end,
                               step, config.total_iterations)
        contact_w = linear_decay(config.contact_weight_start, config.contact_weight_end,
                                 step, config.total_iterations)
        disp_w = linear_decay(config.displacement_weight_start, config.displacement_weight_end,
                              step, config.total_iterations)

        sds_val = 0.0
        tau_val = 0.0
        cfg_val = 0.0
        sds_metrics = {}
        reg_metrics = {}
        L_contact_val = 0.0
        L_impact_val = 0.0

        for view_i in range(config.batch_size):
            # Fresh deform per view
            all_means, all_quats = scene.deform_all_frames(
                use_fine=use_fine, K_neighbors=config.K_neighbors)

            # Per-object regularization on first view only
            if view_i == 0:
                reg_loss, reg_metrics = scene.compute_per_object_regularization(
                    all_means, temp_w, spatial_w, accel_w, disp_w,
                    num_arap_points=getattr(config, 'num_arap_points', 5000))
                reg_loss.backward(retain_graph=True)

                # Contact loss between object pairs (every 4th frame for efficiency)
                if contact_w > 0 and scene.num_objects >= 2:
                    L_contact = torch.tensor(0.0, device=device)
                    L_impact = torch.tensor(0.0, device=device)
                    frame_step = 4
                    num_sampled = 0
                    for t in range(0, config.num_frames, frame_step):
                        # Contact between first two objects (generalize later if needed)
                        means_a = all_means[t, scene.get_object_slice(0), :]
                        means_b = all_means[t, scene.get_object_slice(1), :]
                        L_contact = L_contact + contact_proximity_loss(
                            means_a, means_b,
                            target_distance=config.contact_target_distance)

                        # Impact deformation loss (optional)
                        if config.impact_weight > 0:
                            # Object 1 (pillow) should deform near object 0 (cat)
                            gs_b = scene.gaussian_models[1]
                            # Get original positions with offset applied
                            orig_b = gs_b.means.detach()
                            L_impact = L_impact + impact_deformation_loss(
                                means_b, orig_b, means_a,
                                influence_radius=config.impact_influence_radius)

                        num_sampled += 1

                    if num_sampled > 0:
                        L_contact = L_contact / num_sampled
                        contact_loss = contact_w * L_contact
                        if config.impact_weight > 0:
                            L_impact = L_impact / num_sampled
                            contact_loss = contact_loss + config.impact_weight * L_impact
                        contact_loss.backward(retain_graph=True)
                        L_contact_val = L_contact.item()
                        L_impact_val = L_impact.item()

            # Camera
            jitter = getattr(config, 'camera_radius_jitter', 0.1)
            if getattr(config, 'camera_follow', False):
                frame_centers = all_means.detach().mean(dim=1).cpu()
                ele = np.random.uniform(*config.elevation_range)
                azi = np.random.uniform(0, 360)
                rad = np.random.uniform(
                    config.camera_radius * (1 - jitter),
                    config.camera_radius * (1 + jitter))
                viewmat = torch.stack([
                    orbit_camera(ele, azi, rad, target=frame_centers[t])
                    for t in range(config.num_frames)
                ]).to(device)
            else:
                viewmat = random_camera(
                    elevation_range=config.elevation_range,
                    azimuth_range=(0, 360),
                    radius_range=(config.camera_radius * (1 - jitter),
                                  config.camera_radius * (1 + jitter)),
                    target=scene_center
                ).to(device)

            video = render_video(
                all_means, all_quats,
                activated_scales, sh_coeffs, activated_opacities,
                viewmat, K, config.render_width, config.render_height,
                config.num_frames, sh_degree=sh_degree
            )

            # Color levels adjustment
            if color_white_point != 1.0:
                video = (video / color_white_point).clamp(0, 1)

            if use_real_sds and sds is not None:
                cfg_val = linear_decay(config.cfg_scale_start, config.cfg_scale_end,
                                       step, config.total_iterations)
                sds.guidance_scale = cfg_val
                video.retain_grad()
                sds_loss, tau_val, view_metrics = sds.compute_sds_loss(
                    video, config.text_prompt,
                    negative_prompt="static, still, no motion, blurry, ugly",
                    iteration=step
                )
                (sds_loss / config.batch_size).backward()
                sds_val += sds_loss.item() / config.batch_size
                if video.grad is not None:
                    view_metrics['sds/video_grad_norm'] = video.grad.norm().item()
                    view_metrics['sds/video_grad_mean'] = video.grad.mean().item()
                sds_metrics = view_metrics
            else:
                sds_grad = torch.randn_like(video) * 0.01
                video.backward(sds_grad / config.batch_size)

        optimizer.step()

        step_time = time.time() - step_t0

        # Refresh weights periodically
        refresh_every = getattr(config, 'weight_refresh_every', 50)
        if refresh_every > 0 and (step + 1) % refresh_every == 0:
            with torch.no_grad():
                scene.refresh_weights(config.K_neighbors)

        # Deformation diagnostics
        with torch.no_grad():
            deform_mag = 0.0
            deform_std = 0.0
            for i, gs in enumerate(scene.gaussian_models):
                slc = scene.get_object_slice(i)
                obj_means = all_means[:, slc, :]
                per_frame_mag = (obj_means - gs.means.unsqueeze(0)).norm(dim=-1).mean(dim=1)  # [T]
                deform_mag += per_frame_mag.mean().item()
                deform_std += per_frame_mag.std().item()
            deform_mag /= scene.num_objects
            deform_std /= scene.num_objects
            # Video temporal smoothness: mean pixel difference between adjacent frames
            video_smoothness = (video[:-1] - video[1:]).abs().mean().item()

        # Reinit
        if step == config.reinit_step:
            t_ref = int(config.num_frames * getattr(config, 'reinit_ref_ratio', 0.75))
            t_ref = max(1, min(t_ref, config.num_frames - 1))
            print(f"[{_ts()}] Step {step}: Reinit - copying frame {t_ref} deformation to all objects")
            scene.reinit_later_frames(t_ref)

        # Logging
        if step % config.log_every == 0:
            metrics = {
                'sds_loss': sds_val,
                'L_contact': L_contact_val,
                'L_impact': L_impact_val,
                'tau': tau_val,
                'cfg_scale': cfg_val,
                'lr_def': lr_def,
                'lr_scale': lr_sc,
                'temp_weight': temp_w,
                'spatial_weight': spatial_w,
                'accel_weight': accel_w,
                'contact_weight': contact_w,
                'displacement_weight': disp_w,
                'deform_mag': deform_mag,
                'deform_std': deform_std,
                'video_smoothness': video_smoothness,
                'fine': int(use_fine),
                'step_time': step_time,
            }
            metrics.update(reg_metrics)
            metrics.update(sds_metrics)
            logger.log_step(step, metrics)

            elapsed = time.time() - t_start
            eta = elapsed / (step + 1) * (config.total_iterations - step - 1)
            print(f"[{_ts()}] Step {step}/{config.total_iterations} | "
                  f"SDS: {sds_val:.4f} | "
                  f"L_contact: {L_contact_val:.6f} | "
                  f"grad_norm: {sds_metrics.get('sds/video_grad_norm', 0.0):.4f} | "
                  f"tau: {tau_val:.3f} | "
                  f"cfg: {cfg_val:.1f} | "
                  f"deform: {deform_mag:.4f} (std {deform_std:.4f}) | "
                  f"v_smooth: {video_smoothness:.4f} | "
                  f"LR: {lr_def:.6f} | Fine: {use_fine} | "
                  f"ETA: {eta/60:.1f}min")

        # Visualization & checkpointing
        if step % config.save_every == 0 or step == config.total_iterations - 1:
            with torch.no_grad():
                vis_means, vis_quats = scene.deform_all_frames(
                    use_fine=use_fine, K_neighbors=config.K_neighbors)

                if getattr(config, 'camera_follow', False):
                    vis_frame_centers = vis_means.mean(dim=1).cpu()
                    active_vis_viewmats = {}
                    for vname, ele, azi in vis_views:
                        vms = torch.stack([
                            orbit_camera(ele, azi, config.camera_radius,
                                         target=vis_frame_centers[t])
                            for t in range(config.num_frames)
                        ]).to(device)
                        active_vis_viewmats[vname] = vms
                else:
                    active_vis_viewmats = vis_viewmats

                first_gif_path = None
                for vname, vmat in active_vis_viewmats.items():
                    vis_video = render_video(
                        vis_means, vis_quats,
                        activated_scales, sh_coeffs, activated_opacities,
                        vmat, K, config.render_width, config.render_height,
                        config.num_frames, sh_degree=sh_degree
                    )
                    if color_white_point != 1.0:
                        vis_video = (vis_video / color_white_point).clamp(0, 1)
                    gif_path = os.path.join(config.output_dir, f'step_{step:05d}_{vname}.gif')
                    save_gif(vis_video, gif_path)
                    if first_gif_path is None:
                        first_gif_path = gif_path

                print(f"[{_ts()}]   Saved {len(active_vis_viewmats)} views for step {step}")
                logger.log_image(step, 'render/frame0', first_gif_path)
                logger.log_video(step, 'render/video', first_gif_path)

                # Save multi-object checkpoint
                ckpt_path = os.path.join(config.output_dir, f'checkpoint_step_{step:05d}.pt')
                torch.save({
                    'step': step,
                    'multi_object': True,
                    'deformation_state_dicts': scene.state_dict_multi(),
                    'config': config.__dict__,
                }, ckpt_path)
                print(f"[{_ts()}]   Saved {ckpt_path}")

    total_time = time.time() - t_start
    print(f"\n[{_ts()}] Training complete! Total time: {total_time/60:.1f} minutes")
    logger.log_step(config.total_iterations, {'total_time_min': total_time / 60})
    logger.close()

    ckpt_path = os.path.join(config.output_dir, 'final_checkpoint.pt')
    torch.save({
        'multi_object': True,
        'deformation_state_dicts': scene.state_dict_multi(),
        'config': config.__dict__,
    }, ckpt_path)
    print(f"[{_ts()}] Saved checkpoint to {ckpt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-object 4DGS Training')
    parser.add_argument('--scene', type=str, required=True,
                        help='Scene preset name (must have objects list, e.g. cat_pillow)')
    parser.add_argument('--total_iterations', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_frames', type=int, default=None)
    parser.add_argument('--render_width', type=int, default=None)
    parser.add_argument('--render_height', type=int, default=None)
    parser.add_argument('--save_every', type=int, default=None)
    parser.add_argument('--log_every', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--scene_name', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--sds_model_name', type=str, default=None)
    parser.add_argument('--camera_radius', type=float, default=None)
    parser.add_argument('--camera_follow', action='store_true', default=None)
    parser.add_argument('--fovy_deg', type=float, default=None)
    parser.add_argument('--color_white_point', type=float, default=None)
    parser.add_argument('--contact_weight_start', type=float, default=None)
    parser.add_argument('--contact_weight_end', type=float, default=None)
    parser.add_argument('--contact_target_distance', type=float, default=None)
    parser.add_argument('--impact_weight', type=float, default=None)
    parser.add_argument('--temp_weight_start', type=float, default=None)
    parser.add_argument('--temp_weight_end', type=float, default=None)
    parser.add_argument('--accel_weight_start', type=float, default=None)
    parser.add_argument('--accel_weight_end', type=float, default=None)
    parser.add_argument('--spatial_weight_start', type=float, default=None)
    parser.add_argument('--spatial_weight_end', type=float, default=None)
    parser.add_argument('--displacement_weight_start', type=float, default=None)
    parser.add_argument('--displacement_weight_end', type=float, default=None)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    args = parser.parse_args()

    # Build config: scene preset -> CLI overrides
    config_dict = {}
    config_dict.update(get_scene_preset(args.scene))
    cli_overrides = {k: v for k, v in vars(args).items()
                     if v is not None and k not in ('scene', 'no_wandb')}
    config_dict.update(cli_overrides)
    if args.no_wandb:
        config_dict['use_wandb'] = False
    config = TrainConfig(**config_dict)
    train_multiobj(config)
