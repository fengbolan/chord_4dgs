"""
CHORD 4DGS Inference Script

Usage:
  # Render from a single viewpoint
  python inference.py --checkpoint outputs/xxx/final_checkpoint.pt

  # Orbit: camera rotates 360° around the scene, for each frame
  python inference.py --checkpoint outputs/xxx/final_checkpoint.pt --mode orbit

  # Multi-view grid: N elevations x M azimuths, one GIF per view
  python inference.py --checkpoint outputs/xxx/final_checkpoint.pt --mode multiview

  # Custom single view
  python inference.py --checkpoint outputs/xxx/final_checkpoint.pt --elevation 15 --azimuth 45
"""

import argparse
import os
import math
import torch
import numpy as np
from PIL import Image

from config import TrainConfig
from models.gaussian_model import GaussianModel
from models.deformation import Deformation4D
from core.renderer import render_video
from utils.camera_utils import orbit_camera, get_intrinsics


def save_gif(video_tensor, path, duration=200):
    frames = []
    for t in range(video_tensor.shape[0]):
        frame = video_tensor[t].detach().cpu().clamp(0, 1).numpy()
        frames.append(Image.fromarray((frame * 255).astype(np.uint8)))
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0)


def save_orbit_video(frames_list, path, fps=24):
    """Save a list of [H,W,3] numpy frames as mp4 or gif."""
    if path.endswith('.gif'):
        imgs = [Image.fromarray(f) for f in frames_list]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0)
    else:
        import imageio
        imageio.mimwrite(path, frames_list, fps=fps)


def load_model(checkpoint_path, device='cuda:0'):
    """Load trained deformation model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config_dict = ckpt['config']
    config = TrainConfig(**{k: v for k, v in config_dict.items()
                            if hasattr(TrainConfig, k)})

    # Load 3DGS (apply same rotation as training)
    gs = GaussianModel()
    gs.load_ply(config.ply_path)
    gs.apply_rotation(
        getattr(config, 'scene_rotate_x', 0),
        getattr(config, 'scene_rotate_y', 0),
        getattr(config, 'scene_rotate_z', 0),
    )
    gs.to(device)

    # Initialize deformation with same architecture
    deform = Deformation4D(
        gs,
        num_coarse=config.num_coarse_cp,
        num_fine=config.num_fine_cp,
        num_frames=config.num_frames,
    ).to(device)
    deform.load_state_dict(ckpt['deformation_state_dict'])
    deform.eval()

    return gs, deform, config


@torch.no_grad()
def render_from_view(gs, deform, config, elevation, azimuth, radius,
                     width, height, device='cuda:0'):
    """Render 4DGS video from a specific viewpoint."""
    scene_center = gs.means.mean(dim=0).cpu()
    viewmat = orbit_camera(elevation, azimuth, radius, target=scene_center).to(device)

    K = get_intrinsics(config.fovy_deg, width, height)

    activated_opacities = gs.get_activated_opacities()
    activated_scales = gs.get_activated_scales()
    sh_coeffs = gs.get_sh_coeffs()
    sh_degree = int(math.sqrt(sh_coeffs.shape[1])) - 1

    all_means, all_quats = deform.deform_all_frames(gs, use_fine=True)

    video = render_video(
        all_means, all_quats,
        activated_scales, sh_coeffs, activated_opacities,
        viewmat, K, width, height,
        config.num_frames, sh_degree=sh_degree
    )
    # Color levels adjustment
    if getattr(config, 'color_white_point', 1.0) != 1.0:
        video = (video / config.color_white_point).clamp(0, 1)
    return video


@torch.no_grad()
def mode_single(gs, deform, config, args):
    """Render a single viewpoint GIF."""
    print(f"Rendering: elevation={args.elevation}, azimuth={args.azimuth}, radius={args.radius}")
    video = render_from_view(
        gs, deform, config,
        args.elevation, args.azimuth, args.radius,
        args.width, args.height, args.device
    )
    out_path = os.path.join(args.output, 'render.gif')
    save_gif(video, out_path, duration=args.frame_duration)
    print(f"Saved {out_path}")


@torch.no_grad()
def mode_multiview(gs, deform, config, args):
    """Render from a grid of viewpoints."""
    elevations = [float(e) for e in args.elevations.split(',')]
    azimuths = np.linspace(0, 360, args.num_azimuths, endpoint=False)

    for ele in elevations:
        for azi in azimuths:
            print(f"Rendering: ele={ele:.0f}, azi={azi:.0f}")
            video = render_from_view(
                gs, deform, config,
                ele, azi, args.radius,
                args.width, args.height, args.device
            )
            name = f'ele{ele:.0f}_azi{azi:.0f}.gif'
            out_path = os.path.join(args.output, name)
            save_gif(video, out_path, duration=args.frame_duration)
    print(f"Saved {len(elevations) * len(azimuths)} views to {args.output}/")


@torch.no_grad()
def mode_orbit(gs, deform, config, args):
    """Camera orbits 360° while the 4DGS animation plays.

    Output: a single video where each output frame = one (time, azimuth) pair.
    The camera sweeps through `num_orbits` full rotations over the animation.
    """
    scene_center = gs.means.mean(dim=0).cpu()
    K = get_intrinsics(config.fovy_deg, args.width, args.height)

    activated_opacities = gs.get_activated_opacities()
    activated_scales = gs.get_activated_scales()
    sh_coeffs = gs.get_sh_coeffs()
    sh_degree = int(math.sqrt(sh_coeffs.shape[1])) - 1

    all_means, all_quats = deform.deform_all_frames(gs, use_fine=True)

    T = config.num_frames
    total_out_frames = args.orbit_frames
    orbit_frames_list = []

    for i in range(total_out_frames):
        # Time index: loop through animation
        t_float = (i / total_out_frames) * T
        t = int(t_float) % T

        # Azimuth: sweep full circle(s)
        azi = (i / total_out_frames) * 360.0 * args.num_orbits
        ele = args.elevation

        viewmat = orbit_camera(ele, azi, args.radius, target=scene_center).to(args.device)

        from core.renderer import render_gaussians
        img, _ = render_gaussians(
            all_means[t], all_quats[t],
            activated_scales, sh_coeffs, activated_opacities,
            viewmat, K, args.width, args.height, sh_degree=sh_degree
        )
        # Color levels adjustment
        if getattr(config, 'color_white_point', 1.0) != 1.0:
            img = (img / config.color_white_point).clamp(0, 1)
        frame_np = (img.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
        orbit_frames_list.append(frame_np)

        if (i + 1) % 20 == 0:
            print(f"  Frame {i+1}/{total_out_frames} (t={t}, azi={azi:.1f}°)")

    out_path = os.path.join(args.output, 'orbit.gif')
    save_orbit_video(orbit_frames_list, out_path, fps=args.orbit_fps)
    print(f"Saved orbit video ({total_out_frames} frames) to {out_path}")

    # Also save individual time-step GIFs from a few fixed views
    for azi in [0, 90, 180, 270]:
        video = render_from_view(
            gs, deform, config,
            args.elevation, azi, args.radius,
            args.width, args.height, args.device
        )
        name = f'view_azi{azi:03d}.gif'
        save_gif(video, os.path.join(args.output, name), duration=args.frame_duration)
    print(f"Saved 4 fixed-view GIFs (azi=0,90,180,270)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CHORD 4DGS Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint .pt file')
    parser.add_argument('--mode', type=str, default='orbit',
                        choices=['single', 'multiview', 'orbit'],
                        help='Rendering mode')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: same as checkpoint dir)')
    parser.add_argument('--device', type=str, default='cuda:0')
    # Resolution
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=288)
    # Camera
    parser.add_argument('--elevation', type=float, default=15.0)
    parser.add_argument('--azimuth', type=float, default=45.0)
    parser.add_argument('--radius', type=float, default=3.0)
    # Multiview options
    parser.add_argument('--elevations', type=str, default='-15,0,15,30',
                        help='Comma-separated elevation angles')
    parser.add_argument('--num_azimuths', type=int, default=8)
    # Orbit options
    parser.add_argument('--orbit_frames', type=int, default=120,
                        help='Total frames in orbit video')
    parser.add_argument('--orbit_fps', type=int, default=24)
    parser.add_argument('--num_orbits', type=int, default=1,
                        help='Number of full 360° orbits')
    # GIF timing
    parser.add_argument('--frame_duration', type=int, default=200,
                        help='Per-frame duration in ms for animation GIFs')

    args = parser.parse_args()

    # Default output dir = checkpoint's parent directory + /inference/
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.checkpoint), 'inference')
    os.makedirs(args.output, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    gs, deform, config = load_model(args.checkpoint, args.device)
    print(f"Loaded: {gs.num_gaussians} gaussians, {config.num_frames} frames")
    # Use checkpoint's radius if not explicitly set
    if args.radius == 3.0 and hasattr(config, 'camera_radius'):
        args.radius = config.camera_radius

    if args.mode == 'single':
        mode_single(gs, deform, config, args)
    elif args.mode == 'multiview':
        mode_multiview(gs, deform, config, args)
    elif args.mode == 'orbit':
        mode_orbit(gs, deform, config, args)
