import os
import sys
import math
import json
import time
from datetime import datetime
import argparse
import torch
import numpy as np
from PIL import Image

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

sys.path.insert(0, os.path.dirname(__file__))

from config import TrainConfig
from scene_configs import get_scene_preset
from models.gaussian_model import GaussianModel
from models.deformation import Deformation4D
from core.renderer import render_gaussians, render_video
from core.regularization import temporal_regularization, arap_regularization
from utils.camera_utils import orbit_camera, random_camera, get_intrinsics


def _ts():
    """Return current timestamp string for log prefixes."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class TrainLogger:
    """Logs training metrics to JSON lines file, tensorboard, and wandb."""

    def __init__(self, output_dir, config_dict=None, use_wandb=False,
                 wandb_project=None, wandb_run_name=None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.jsonl_path = os.path.join(output_dir, 'train_log.jsonl')
        self.writer = None
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Try to use tensorboard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(output_dir, 'tb'))
        except ImportError:
            pass

        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project=wandb_project or "chord-4dgs",
                name=wandb_run_name,
                config=config_dict or {},
                dir=output_dir,
            )
            print(f"[{_ts()}] wandb initialized: {wandb.run.url}")

        # Write config as first line
        if config_dict:
            self._write_json({'type': 'config', **config_dict})

    def _write_json(self, data):
        with open(self.jsonl_path, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def log_step(self, step, metrics: dict):
        """Log metrics for a training step."""
        record = {'type': 'step', 'step': step, **metrics}
        self._write_json(record)

        if self.writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(k, v, step)

        if self.use_wandb:
            wandb.log(metrics, step=step)

    def log_image(self, step, tag, image_path):
        """Log a reference to a saved image."""
        if self.writer:
            try:
                img = np.array(Image.open(image_path).convert('RGB'))
                self.writer.add_image(tag, img, step, dataformats='HWC')
            except Exception:
                pass

    def log_video(self, step, tag, gif_path):
        """Log a GIF video to wandb."""
        if self.use_wandb:
            try:
                wandb.log({tag: wandb.Video(gif_path, format="gif")}, step=step)
            except Exception:
                pass

    def close(self):
        if self.writer:
            self.writer.close()
        if self.use_wandb:
            wandb.finish()


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


def _make_model_short_name(sds_model_name: str) -> str:
    """Extract short model name, e.g. 'Wan-AI/Wan2.2-T2V-A14B-Diffusers' -> 'wan2.2-14b'."""
    name = sds_model_name.split('/')[-1].lower()
    # Extract version like wan2.2 or wan2.1
    import re
    ver_match = re.search(r'wan(\d+\.\d+)', name)
    ver = f"wan{ver_match.group(1)}" if ver_match else "unknown"
    # Extract param size like 14b, 1.3b
    size_match = re.search(r'[\-_](?:a)?(\d+(?:\.\d+)?b)', name)
    size = size_match.group(1) if size_match else ""
    return f"{ver}-{size}" if size else ver


def train(config: TrainConfig):
    # Auto-generate output_dir: {timestamp}_{model}_{scene}_{iterations}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_short = _make_model_short_name(config.sds_model_name) if config.sds_model_name else "nosds"
    scene = getattr(config, 'scene_name', 'default')
    run_name = f"{timestamp}_{model_short}_{scene}_{config.total_iterations}"
    config.output_dir = os.path.join(config.output_dir, run_name)
    os.makedirs(config.output_dir, exist_ok=True)
    if config.wandb_run_name is None:
        config.wandb_run_name = run_name

    # Redirect stdout/stderr to train.log inside the experiment directory
    import sys
    log_path = os.path.join(config.output_dir, 'train.log')
    log_file = open(log_path, 'a', buffering=1)  # line-buffered
    sys.stdout = sys.stderr = log_file

    device = torch.device(config.device)

    # Initialize logger
    config_dict = {
        k: str(v) if isinstance(v, tuple) else v
        for k, v in config.__dict__.items()
    }
    logger = TrainLogger(
        config.output_dir,
        config_dict=config_dict,
        use_wandb=config.use_wandb,
        wandb_project=config.wandb_project,
        wandb_run_name=config.wandb_run_name,
    )

    # Load 3DGS
    print(f"[{_ts()}] Loading 3DGS model...")
    gs = GaussianModel()
    gs.load_ply(config.ply_path)
    gs.apply_rotation(config.scene_rotate_x, config.scene_rotate_y, config.scene_rotate_z)
    if getattr(config, 'center_to_origin', True):
        offset = gs.center_to_origin()
        print(f"[{_ts()}] Centered to origin (offset: {offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f})")
    gs.to(device)
    print(f"[{_ts()}] Loaded {gs.num_gaussians} gaussians")
    if config.scene_rotate_x or config.scene_rotate_y or config.scene_rotate_z:
        print(f"[{_ts()}] Applied scene rotation: x={config.scene_rotate_x}, y={config.scene_rotate_y}, z={config.scene_rotate_z}")

    activated_opacities = gs.get_activated_opacities()
    activated_scales = gs.get_activated_scales()
    sh_coeffs = gs.get_sh_coeffs()
    sh_degree = int(math.sqrt(sh_coeffs.shape[1])) - 1

    # Initialize deformation
    print(f"[{_ts()}] Initializing 4D deformation...")
    deform = Deformation4D(
        gs,
        num_coarse=config.num_coarse_cp,
        num_fine=config.num_fine_cp,
        num_frames=config.num_frames
    ).to(device)
    print(f"[{_ts()}] Coarse CPs: {config.num_coarse_cp}, Fine CPs: {config.num_fine_cp}")

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
    scene_center = gs.means.mean(dim=0).detach().cpu()
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
    print(f"\n[{_ts()}] Starting training for {config.total_iterations} iterations...")
    print(f"[{_ts()}] Text prompt: {config.text_prompt}")
    print(f"[{_ts()}] Render: {config.render_width}x{config.render_height}, {config.num_frames} frames")

    # Create optimizer ONCE (persist Adam momentum across steps)
    use_fine = False
    param_groups = deform.get_optimizable_params(use_fine)
    for pg in param_groups:
        if 'sigma' in pg['name']:
            pg['lr'] = config.lr_scale
        elif 'rots' in pg['name']:
            pg['lr'] = config.lr_deformation * 0.5
        else:
            pg['lr'] = config.lr_deformation
    optimizer = torch.optim.Adam(param_groups)

    # Pre-sample ARAP point indices (re-sampled periodically)
    num_arap_pts = getattr(config, 'num_arap_points', 5000)
    arap_idx = torch.randint(0, gs.num_gaussians, (num_arap_pts,), device=device)
    arap_orig_pts = gs.means[arap_idx].detach()

    t_start = time.time()

    for step in range(config.total_iterations):
        step_t0 = time.time()
        new_use_fine = step >= config.total_iterations * config.fine_start_ratio

        # Rebuild optimizer when fine stage starts (preserves coarse momentum impossible
        # across param group change, but at least we don't recreate every step)
        if new_use_fine and not use_fine:
            use_fine = True
            param_groups = deform.get_optimizable_params(use_fine)
            for pg in param_groups:
                if 'sigma' in pg['name']:
                    pg['lr'] = config.lr_scale
                elif 'rots' in pg['name']:
                    pg['lr'] = config.lr_deformation * 0.5
                else:
                    pg['lr'] = config.lr_deformation
            optimizer = torch.optim.Adam(param_groups)
            print(f"[{_ts()}] Step {step}: Fine stage started, optimizer rebuilt with fine params")

        # LR decay (update in-place on existing optimizer)
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

        # Multi-view SDS: each view gets its own deform→render→SDS→backward
        # so the computation graph is freed after each view (no OOM).
        sds_val = 0.0
        tau_val = 0.0
        cfg_val = 0.0
        sds_metrics = {}
        L_temp = torch.tensor(0.0, device=device)
        L_arap = torch.tensor(0.0, device=device)
        temp_w = linear_decay(config.temp_weight_start, config.temp_weight_end,
                              step, config.total_iterations)
        spatial_w = linear_decay(config.spatial_weight_start, config.spatial_weight_end,
                                 step, config.total_iterations)

        for view_i in range(config.batch_size):
            # Fresh deform per view (independent graph, frees after backward)
            all_means, all_quats = deform.deform_all_frames(gs, use_fine=use_fine)

            # Regularization on first view only
            if view_i == 0:
                L_temp = temporal_regularization(all_means)
                L_arap = arap_regularization(
                    arap_orig_pts, all_means[:, arap_idx, :])
                reg_loss = temp_w * L_temp + spatial_w * L_arap
                reg_loss.backward(retain_graph=True)

            jitter = getattr(config, 'camera_radius_jitter', 0.1)
            if getattr(config, 'camera_follow', False):
                # Camera follow: per-frame viewmat tracking object center
                frame_centers = all_means.detach().mean(dim=1).cpu()  # [T, 3]
                ele = np.random.uniform(*config.elevation_range)
                azi = np.random.uniform(0, 360)
                rad = np.random.uniform(
                    config.camera_radius * (1 - jitter),
                    config.camera_radius * (1 + jitter))
                viewmat = torch.stack([
                    orbit_camera(ele, azi, rad, target=frame_centers[t])
                    for t in range(config.num_frames)
                ]).to(device)  # [T, 4, 4]
            else:
                viewmat = random_camera(
                    elevation_range=config.elevation_range,
                    azimuth_range=(0, 360),
                    radius_range=(config.camera_radius * (1 - jitter), config.camera_radius * (1 + jitter)),
                    target=scene_center
                ).to(device)

            video = render_video(
                all_means, all_quats,
                activated_scales, sh_coeffs, activated_opacities,
                viewmat, K, config.render_width, config.render_height,
                config.num_frames, sh_degree=sh_degree
            )

            # Color levels adjustment: remap [0, white_point] -> [0, 1]
            if config.color_white_point != 1.0:
                video = (video / config.color_white_point).clamp(0, 1)

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
                sds_metrics = view_metrics  # keep last view metrics
            else:
                sds_grad = torch.randn_like(video) * 0.01
                video.backward(sds_grad / config.batch_size)

        optimizer.step()

        step_time = time.time() - step_t0

        # Re-sample ARAP points periodically for diversity
        refresh_every = getattr(config, 'weight_refresh_every', 50)
        if refresh_every > 0 and (step + 1) % refresh_every == 0:
            arap_idx = torch.randint(0, gs.num_gaussians, (num_arap_pts,), device=device)
            arap_orig_pts = gs.means[arap_idx].detach()
            # Also update cached weights for deform_frame (used in visualization)
            with torch.no_grad():
                deform.refresh_weights(gs.means, config.K_neighbors)

        # Compute deformation magnitude for diagnostics (use last view's all_means)
        with torch.no_grad():
            deform_mag = (all_means - gs.means.unsqueeze(0)).norm(dim=-1).mean().item()

        # Reinit: copy ref frame deformation to later frames (CHORD-style)
        if step == config.reinit_step:
            t_ref = int(config.num_frames * getattr(config, 'reinit_ref_ratio', 0.75))
            t_ref = max(1, min(t_ref, config.num_frames - 1))
            print(f"[{_ts()}] Step {step}: Reinit — copying frame {t_ref} deformation to frames {t_ref+1}..{config.num_frames-1}")
            deform.reinit_later_frames(t_ref)

        # Logging
        if step % config.log_every == 0:
            metrics = {
                'L_temp': L_temp.item(),
                'L_arap': L_arap.item(),
                'sds_loss': sds_val,
                'tau': tau_val,
                'cfg_scale': cfg_val,
                'lr_def': lr_def,
                'lr_scale': lr_sc,
                'temp_weight': temp_w,
                'spatial_weight': spatial_w,
                'deform_mag': deform_mag,
                'fine': int(use_fine),
                'step_time': step_time,
            }
            metrics.update(sds_metrics)
            logger.log_step(step, metrics)

            elapsed = time.time() - t_start
            eta = elapsed / (step + 1) * (config.total_iterations - step - 1)
            print(f"[{_ts()}] Step {step}/{config.total_iterations} | "
                  f"L_temp: {L_temp.item():.6f} | "
                  f"L_arap: {L_arap.item():.6f} | "
                  f"SDS: {sds_val:.4f} | "
                  f"grad_norm: {sds_metrics.get('sds/video_grad_norm', 0.0):.4f} | "
                  f"tau: {tau_val:.3f} | "
                  f"cfg: {cfg_val:.1f} | "
                  f"deform: {deform_mag:.5f} | "
                  f"LR: {lr_def:.6f} | Fine: {use_fine} | "
                  f"ETA: {eta/60:.1f}min")

        # Visualization
        if step % config.save_every == 0 or step == config.total_iterations - 1:
            with torch.no_grad():
                vis_means, vis_quats = deform.deform_all_frames(gs, use_fine=use_fine)
                # Build per-frame viewmats for camera follow mode
                if getattr(config, 'camera_follow', False):
                    vis_frame_centers = vis_means.mean(dim=1).cpu()  # [T, 3]
                    vis_viewmats_follow = {}
                    for vname, ele, azi in vis_views:
                        vms = torch.stack([
                            orbit_camera(ele, azi, config.camera_radius,
                                         target=vis_frame_centers[t])
                            for t in range(config.num_frames)
                        ]).to(device)  # [T, 4, 4]
                        vis_viewmats_follow[vname] = vms
                    active_vis_viewmats = vis_viewmats_follow
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
                    if config.color_white_point != 1.0:
                        vis_video = (vis_video / config.color_white_point).clamp(0, 1)
                    gif_path = os.path.join(config.output_dir, f'step_{step:05d}_{vname}.gif')
                    save_gif(vis_video, gif_path)
                    if first_gif_path is None:
                        first_gif_path = gif_path
                print(f"[{_ts()}]   Saved {len(active_vis_viewmats)} views for step {step}")
                # Log first view to tensorboard/wandb
                logger.log_image(step, 'render/frame0', first_gif_path)
                logger.log_video(step, 'render/video', first_gif_path)
                # Save checkpoint
                ckpt_path = os.path.join(config.output_dir, f'checkpoint_step_{step:05d}.pt')
                torch.save({
                    'step': step,
                    'deformation_state_dict': deform.state_dict(),
                    'config': config.__dict__,
                }, ckpt_path)
                print(f"[{_ts()}]   Saved {ckpt_path}")

    total_time = time.time() - t_start
    print(f"\n[{_ts()}] Training complete! Total time: {total_time/60:.1f} minutes")
    logger.log_step(config.total_iterations, {'total_time_min': total_time / 60})
    logger.close()

    ckpt_path = os.path.join(config.output_dir, 'final_checkpoint.pt')
    torch.save({
        'deformation_state_dict': deform.state_dict(),
        'config': config.__dict__,
    }, ckpt_path)
    print(f"[{_ts()}] Saved checkpoint to {ckpt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Scene preset (loads defaults from scene_configs.py, CLI args override)
    parser.add_argument('--scene', type=str, default=None,
                        help='Scene preset name (e.g. cat, cactus). Loads per-scene defaults.')
    parser.add_argument('--ply_path', type=str, default=None)
    parser.add_argument('--text_prompt', type=str, default=None)
    parser.add_argument('--total_iterations', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Number of camera views per step')
    parser.add_argument('--num_frames', type=int, default=None)
    parser.add_argument('--render_width', type=int, default=None)
    parser.add_argument('--render_height', type=int, default=None)
    parser.add_argument('--save_every', type=int, default=None)
    parser.add_argument('--log_every', type=int, default=None)
    parser.add_argument('--num_coarse_cp', type=int, default=None)
    parser.add_argument('--num_fine_cp', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Base output directory (run subfolder auto-generated)')
    parser.add_argument('--scene_name', type=str, default=None,
                        help='Scene name for experiment naming (auto-set by --scene)')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--sds_model_name', type=str, default=None)
    parser.add_argument('--camera_radius', type=float, default=None)
    parser.add_argument('--camera_follow', action='store_true', default=None,
                        help='Camera follows object center per frame (for walking animals)')
    parser.add_argument('--temp_weight_start', type=float, default=None)
    parser.add_argument('--temp_weight_end', type=float, default=None)
    parser.add_argument('--spatial_weight_start', type=float, default=None)
    parser.add_argument('--spatial_weight_end', type=float, default=None)
    parser.add_argument('--fovy_deg', type=float, default=None)
    parser.add_argument('--color_white_point', type=float, default=None)
    parser.add_argument('--scene_rotate_x', type=float, default=None)
    parser.add_argument('--scene_rotate_y', type=float, default=None)
    parser.add_argument('--scene_rotate_z', type=float, default=None)
    # wandb (default on; use --no_wandb to disable)
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    args = parser.parse_args()

    # Build config: scene preset (if any) -> then CLI overrides
    config_dict = {}
    if args.scene:
        config_dict.update(get_scene_preset(args.scene))
    # CLI args override preset (only non-None values)
    cli_overrides = {k: v for k, v in vars(args).items()
                     if v is not None and k not in ('scene', 'no_wandb')}
    config_dict.update(cli_overrides)
    if args.no_wandb:
        config_dict['use_wandb'] = False
    config = TrainConfig(**config_dict)
    train(config)
