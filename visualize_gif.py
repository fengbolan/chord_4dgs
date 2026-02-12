"""
Generate GIF animations of 3D point cloud motion for each experiment.

1. Per-experiment: animated point cloud with trailing trajectories (rainbow)
2. Multi-experiment: 4 experiments side-by-side, synchronized animation
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image
import io

sys.path.insert(0, os.path.dirname(__file__))
from inference import load_model

# ── Config ──────────────────────────────────────────────────────────
EXPERIMENTS = {
    "follow":        "outputs/20260211_024110_wan2.2-14b_cat_41f_follow_8000",
    "strong_reg":    "outputs/20260210_231645_wan2.2-14b_cat_41f_strong_reg_8000",
    "strong_spatial":"outputs/20260210_223229_wan2.2-14b_cat_41f_strong_spaital_8000",
    "strong_temp":   "outputs/20260210_223157_wan2.2-14b_cat_41f_strong_temp_8000",
}

OUT_DIR = "outputs/visualization_comparison"
DEVICE = "cuda:3"
N_PTS = 500          # points per experiment
TRAIL_LEN = 8        # number of past frames to show as trail
PT_SIZE = 4
TRAIL_LW = 1.0
GIF_DURATION = 120   # ms per frame
DPI = 150

# ── Helpers ─────────────────────────────────────────────────────────

def fps_sample(points, n):
    N = points.shape[0]
    if n >= N:
        return np.arange(N)
    selected = [np.random.randint(N)]
    min_dists = np.full(N, np.inf)
    for _ in range(n - 1):
        d = np.sum((points - points[selected[-1]]) ** 2, axis=1)
        min_dists = np.minimum(min_dists, d)
        selected.append(np.argmax(min_dists))
    return np.array(selected)


def load_all_means(exp_path, device):
    ckpt_path = os.path.join(exp_path, "final_checkpoint.pt")
    gs, deform, config = load_model(ckpt_path, device)
    with torch.no_grad():
        all_means, _ = deform.deform_all_frames(gs, use_fine=True)
    return all_means.cpu().numpy(), config


def fig_to_image(fig):
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


def compute_bounds(all_data, idx):
    """Compute shared 3D bounds across all experiments."""
    all_pts = np.concatenate([v[:, idx, :].reshape(-1, 3) for v in all_data.values()])
    margin = 0.15
    return {
        'xlim': (all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin),
        'ylim': (all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin),
        'zlim': (all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin),
    }


# ── GIF 1: Per-experiment trajectory animation ─────────────────────

def make_single_experiment_gif(name, all_means, idx, out_dir, elev=25, azim_start=30):
    """Animated GIF: point cloud moves with rainbow trailing trajectories."""
    T = all_means.shape[0]
    pts = all_means[:, idx, :]  # [T, n, 3]
    time_colors = cm.rainbow(np.linspace(0, 1, T))

    # Compute bounds for this experiment
    flat = pts.reshape(-1, 3)
    margin = 0.1
    xlim = (flat[:, 0].min() - margin, flat[:, 0].max() + margin)
    ylim = (flat[:, 1].min() - margin, flat[:, 1].max() + margin)
    zlim = (flat[:, 2].min() - margin, flat[:, 2].max() + margin)

    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(7, 6), facecolor='black')
        ax = fig.add_subplot(111, projection='3d', facecolor='black')

        # Draw trailing trajectories
        trail_start = max(0, t - TRAIL_LEN)
        for pi in range(len(idx)):
            if t > 0:
                traj = pts[trail_start:t+1, pi, :]
                n_seg = len(traj) - 1
                if n_seg > 0:
                    segments = np.array([[traj[s], traj[s+1]] for s in range(n_seg)])
                    # Color by global time, fade alpha for older segments
                    seg_colors = []
                    for s in range(n_seg):
                        global_t = trail_start + s
                        c = list(time_colors[global_t])
                        alpha = 0.3 + 0.7 * (s / max(n_seg, 1))
                        c[3] = alpha
                        seg_colors.append(c)
                    lc = Line3DCollection(segments, colors=seg_colors, linewidths=TRAIL_LW)
                    ax.add_collection3d(lc)

        # Draw current points colored by time
        cur_pts = pts[t]
        ax.scatter(cur_pts[:, 0], cur_pts[:, 1], cur_pts[:, 2],
                   c=[time_colors[t]], s=PT_SIZE, alpha=0.9, edgecolors='none')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        # Slow rotation
        azim = azim_start + t * (120.0 / T)
        ax.view_init(elev=elev, azim=azim)

        # Style
        ax.set_xlabel('X', color='white', fontsize=8)
        ax.set_ylabel('Y', color='white', fontsize=8)
        ax.set_zlabel('Z', color='white', fontsize=8)
        ax.tick_params(colors='white', labelsize=6)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.grid(True, alpha=0.2)

        ax.set_title(f"{name}  —  frame {t}/{T-1}", color='white', fontsize=12, pad=10)

        frames.append(fig_to_image(fig))
        plt.close(fig)

    path = os.path.join(out_dir, f"anim_{name}.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=GIF_DURATION, loop=0)
    print(f"  Saved {path} ({len(frames)} frames)")


# ── GIF 2: Multi-experiment side-by-side ───────────────────────────

def make_comparison_gif(all_data, idx, out_dir, elev=25, azim_start=30):
    """4 experiments side-by-side animated GIF."""
    exp_names = list(all_data.keys())
    exp_colors = {
        "follow":        "#e74c3c",
        "strong_reg":    "#3498db",
        "strong_spatial":"#2ecc71",
        "strong_temp":   "#f39c12",
    }
    T = list(all_data.values())[0].shape[0]
    bounds = compute_bounds(all_data, idx)
    time_colors = cm.rainbow(np.linspace(0, 1, T))

    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(20, 5), facecolor='black')

        for ei, name in enumerate(exp_names):
            ax = fig.add_subplot(1, 4, ei + 1, projection='3d', facecolor='black')
            pts = all_data[name][:, idx, :]  # [T, n, 3]

            # Trailing trajectories
            trail_start = max(0, t - TRAIL_LEN)
            for pi in range(len(idx)):
                if t > 0:
                    traj = pts[trail_start:t+1, pi, :]
                    n_seg = len(traj) - 1
                    if n_seg > 0:
                        segments = np.array([[traj[s], traj[s+1]] for s in range(n_seg)])
                        seg_colors = []
                        for s in range(n_seg):
                            global_t = trail_start + s
                            c = list(time_colors[global_t])
                            c[3] = 0.3 + 0.7 * (s / max(n_seg, 1))
                            seg_colors.append(c)
                        lc = Line3DCollection(segments, colors=seg_colors, linewidths=TRAIL_LW * 0.7)
                        ax.add_collection3d(lc)

            # Current points
            cur_pts = pts[t]
            ax.scatter(cur_pts[:, 0], cur_pts[:, 1], cur_pts[:, 2],
                       c=[time_colors[t]], s=PT_SIZE * 0.6, alpha=0.9, edgecolors='none')

            ax.set_xlim(bounds['xlim'])
            ax.set_ylim(bounds['ylim'])
            ax.set_zlim(bounds['zlim'])
            azim = azim_start + t * (120.0 / T)
            ax.view_init(elev=elev, azim=azim)

            ax.set_xlabel('X', color='white', fontsize=6)
            ax.set_ylabel('Y', color='white', fontsize=6)
            ax.set_zlabel('Z', color='white', fontsize=6)
            ax.tick_params(colors='white', labelsize=5)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('gray')
            ax.yaxis.pane.set_edgecolor('gray')
            ax.zaxis.pane.set_edgecolor('gray')
            ax.grid(True, alpha=0.15)
            ax.set_title(name, color=exp_colors[name], fontsize=10, fontweight='bold', pad=5)

        fig.suptitle(f"Frame {t}/{T-1}", color='white', fontsize=14, y=0.98)
        plt.subplots_adjust(wspace=0.05)
        frames.append(fig_to_image(fig))
        plt.close(fig)

        if (t + 1) % 10 == 0:
            print(f"    Comparison frame {t+1}/{T}")

    path = os.path.join(out_dir, "anim_comparison.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=GIF_DURATION, loop=0)
    print(f"  Saved {path} ({len(frames)} frames)")


# ── Main ────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(42)

    print("Loading experiments...")
    all_data = {}
    for name, path in EXPERIMENTS.items():
        print(f"  Loading {name}...")
        means, config = load_all_means(path, DEVICE)
        all_data[name] = means

    # Shared FPS indices from frame 0
    ref_pts = list(all_data.values())[0][0]
    idx = fps_sample(ref_pts, N_PTS)

    print("\n=== Per-experiment GIFs ===")
    for name, means in all_data.items():
        print(f"  [{name}]")
        make_single_experiment_gif(name, means, idx, OUT_DIR)

    print("\n=== Comparison GIF ===")
    make_comparison_gif(all_data, idx, OUT_DIR)

    print(f"\nDone! All GIFs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
