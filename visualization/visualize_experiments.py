"""
Visualize and compare 4D deformation results across experiments.

Generates:
1. Multi-experiment comparison: all experiments' point clouds in the same 3D space
2. Per-experiment trajectory visualization: SpaTracker-style rainbow point tracks
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inference import load_model

# ── Config ──────────────────────────────────────────────────────────
EXPERIMENTS = {
    "follow":       "outputs/20260211_024110_wan2.2-14b_cat_41f_follow_8000",
    "strong_reg":   "outputs/20260210_231645_wan2.2-14b_cat_41f_strong_reg_8000",
    "strong_spatial":"outputs/20260210_223229_wan2.2-14b_cat_41f_strong_spaital_8000",
    "strong_temp":  "outputs/20260210_223157_wan2.2-14b_cat_41f_strong_temp_8000",
}

OUT_DIR = "outputs/visualization_comparison"
DEVICE = "cuda:3"
N_SUBSAMPLE_COMPARISON = 2000   # points for multi-experiment comparison
N_SUBSAMPLE_TRAJECTORY = 400   # points for trajectory visualization
POINT_SIZE = 0.3
TRAJ_LINEWIDTH = 0.6

# ── Helpers ─────────────────────────────────────────────────────────

def fps_sample(points, n):
    """Farthest point sampling on [N, 3] numpy array. Returns indices."""
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
    """Load checkpoint and return all_means [T, N, 3] numpy."""
    ckpt_path = os.path.join(exp_path, "final_checkpoint.pt")
    gs, deform, config = load_model(ckpt_path, device)
    with torch.no_grad():
        all_means, _ = deform.deform_all_frames(gs, use_fine=True)
    return all_means.cpu().numpy(), config  # [T, N, 3]


# ── Visualization 1: Multi-experiment comparison ────────────────────

def plot_comparison(all_data, out_dir):
    """Plot all experiments' point clouds at frame 0 and last frame."""
    os.makedirs(out_dir, exist_ok=True)

    colors_map = {
        "follow":        "#e74c3c",  # red
        "strong_reg":    "#3498db",  # blue
        "strong_spatial":"#2ecc71",  # green
        "strong_temp":   "#f39c12",  # orange
    }

    T = list(all_data.values())[0].shape[0]
    frames_to_show = [0, T // 4, T // 2, 3 * T // 4, T - 1]

    # Get subsample indices from frame 0 (same for all since same PLY)
    ref_pts = list(all_data.values())[0][0]  # [N, 3]
    idx = fps_sample(ref_pts, N_SUBSAMPLE_COMPARISON)

    # Compute global bounds across all experiments and frames
    all_pts = np.concatenate([v[:, idx, :].reshape(-1, 3) for v in all_data.values()])
    margin = 0.1
    xlim = (all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ylim = (all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    zlim = (all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin)

    # ── View 1: Side views for each frame, all experiments overlaid ──
    viewpoints = [
        ("front",  0, 0),
        ("side",   0, 90),
        ("top",    90, 0),
        ("3quarter", 25, 45),
    ]

    for vname, elev, azim in viewpoints:
        fig, axes = plt.subplots(1, len(frames_to_show), figsize=(5 * len(frames_to_show), 5),
                                  subplot_kw={'projection': '3d'})
        for fi, frame in enumerate(frames_to_show):
            ax = axes[fi]
            for exp_name, data in all_data.items():
                pts = data[frame, idx, :]
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                           s=POINT_SIZE, alpha=0.5, c=colors_map[exp_name],
                           label=exp_name if fi == 0 else None)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            ax.set_title(f"Frame {frame}", fontsize=10)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel('X', fontsize=7)
            ax.set_ylabel('Y', fontsize=7)
            ax.set_zlabel('Z', fontsize=7)
            ax.tick_params(labelsize=6)

        fig.legend(loc='upper center', ncol=4, fontsize=9, markerscale=10)
        fig.suptitle(f"Multi-Experiment Comparison ({vname} view)", fontsize=13, y=1.02)
        plt.tight_layout()
        path = os.path.join(out_dir, f"comparison_{vname}.png")
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {path}")

    # ── View 2: Center-of-mass trajectory over time ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axis_labels = [("X", "Y"), ("X", "Z"), ("Y", "Z")]
    for ai, (xlabel, ylabel) in enumerate(axis_labels):
        ax = axes[ai]
        xi = "XYZ".index(xlabel)
        yi = "XYZ".index(ylabel)
        for exp_name, data in all_data.items():
            # Center of mass per frame [T, 3]
            com = data[:, idx, :].mean(axis=1)  # [T, 3]
            ax.plot(com[:, xi], com[:, yi], '-o', markersize=2,
                    color=colors_map[exp_name], label=exp_name, linewidth=1.5)
            # Mark start and end
            ax.plot(com[0, xi], com[0, yi], 's', color=colors_map[exp_name], markersize=8)
            ax.plot(com[-1, xi], com[-1, yi], '*', color=colors_map[exp_name], markersize=12)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Center of Mass: {xlabel}-{ylabel}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    fig.suptitle("Center-of-Mass Trajectory (■=start, ★=end)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "comparison_center_of_mass.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ── Visualization 2: Per-experiment trajectory (SpaTracker style) ──

def plot_trajectories(exp_name, all_means_np, out_dir):
    """Plot point trajectories with rainbow time coloring."""
    os.makedirs(out_dir, exist_ok=True)

    T, N, _ = all_means_np.shape
    idx = fps_sample(all_means_np[0], N_SUBSAMPLE_TRAJECTORY)
    pts = all_means_np[:, idx, :]  # [T, n_sample, 3]

    # Rainbow colormap: frame 0 = blue, frame T-1 = red
    time_colors = cm.rainbow(np.linspace(0, 1, T))  # [T, 4]

    viewpoints = [
        ("front", 0, 0),
        ("side", 0, 90),
        ("top", 90, 0),
        ("3quarter", 25, 45),
    ]

    for vname, elev, azim in viewpoints:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Draw trajectory lines for each point
        for pi in range(len(idx)):
            traj = pts[:, pi, :]  # [T, 3]
            # Create line segments
            segments = np.array([[traj[t], traj[t + 1]] for t in range(T - 1)])
            seg_colors = time_colors[:T - 1]
            lc = Line3DCollection(segments, colors=seg_colors,
                                   linewidths=TRAJ_LINEWIDTH, alpha=0.6)
            ax.add_collection3d(lc)

        # Draw start points (blue) and end points (red)
        ax.scatter(pts[0, :, 0], pts[0, :, 1], pts[0, :, 2],
                   c='blue', s=3, alpha=0.8, label='Start (t=0)')
        ax.scatter(pts[-1, :, 0], pts[-1, :, 1], pts[-1, :, 2],
                   c='red', s=3, alpha=0.8, label=f'End (t={T-1})')

        # Set bounds
        all_flat = pts.reshape(-1, 3)
        margin = 0.05
        ax.set_xlim(all_flat[:, 0].min() - margin, all_flat[:, 0].max() + margin)
        ax.set_ylim(all_flat[:, 1].min() - margin, all_flat[:, 1].max() + margin)
        ax.set_zlim(all_flat[:, 2].min() - margin, all_flat[:, 2].max() + margin)

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
        ax.set_zlabel('Z', fontsize=9)
        ax.legend(fontsize=8)
        ax.set_title(f"{exp_name} — Point Trajectories ({vname})\n"
                     f"Rainbow: blue(t=0) → red(t={T-1})", fontsize=11)

        path = os.path.join(out_dir, f"traj_{exp_name}_{vname}.png")
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {path}")

    # ── 2D projected trajectory plots (cleaner than 3D) ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    projections = [("X", "Y", 0, 1), ("X", "Z", 0, 2), ("Y", "Z", 1, 2)]
    for ai, (xlabel, ylabel, xi, yi) in enumerate(projections):
        ax = axes[ai]
        for pi in range(len(idx)):
            traj = pts[:, pi, :]  # [T, 3]
            points = traj[:, [xi, yi]].reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap='rainbow', norm=plt.Normalize(0, T - 1),
                                linewidths=TRAJ_LINEWIDTH, alpha=0.5)
            lc.set_array(np.arange(T - 1))
            ax.add_collection(lc)

        # Start and end markers
        ax.scatter(pts[0, :, xi], pts[0, :, yi], c='blue', s=2, alpha=0.7, zorder=5)
        ax.scatter(pts[-1, :, xi], pts[-1, :, yi], c='red', s=2, alpha=0.7, zorder=5)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{xlabel}-{ylabel} Projection", fontsize=11)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        # Auto-fit limits
        all_flat = pts.reshape(-1, 3)
        pad = 0.05
        ax.set_xlim(all_flat[:, xi].min() - pad, all_flat[:, xi].max() + pad)
        ax.set_ylim(all_flat[:, yi].min() - pad, all_flat[:, yi].max() + pad)

    # Colorbar
    sm = cm.ScalarMappable(cmap='rainbow', norm=plt.Normalize(0, T - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label('Frame', fontsize=10)

    fig.suptitle(f"{exp_name} — 2D Trajectory Projections (blue=start, red=end)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, f"traj_{exp_name}_2d.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load all experiments
    print("Loading experiments...")
    all_data = {}
    for name, path in EXPERIMENTS.items():
        print(f"  Loading {name} from {path}...")
        means, config = load_all_means(path, DEVICE)
        all_data[name] = means
        print(f"    Shape: {means.shape}, "
              f"frame 0 center: ({means[0].mean(0)[0]:.4f}, {means[0].mean(0)[1]:.4f}, {means[0].mean(0)[2]:.4f}), "
              f"frame -1 center: ({means[-1].mean(0)[0]:.4f}, {means[-1].mean(0)[1]:.4f}, {means[-1].mean(0)[2]:.4f})")

    # Print displacement summary
    print("\n=== Displacement Summary ===")
    for name, means in all_data.items():
        com_start = means[0].mean(axis=0)
        com_end = means[-1].mean(axis=0)
        disp = com_end - com_start
        dist = np.linalg.norm(disp)
        print(f"  {name:20s}: displacement = ({disp[0]:+.4f}, {disp[1]:+.4f}, {disp[2]:+.4f}), "
              f"|d| = {dist:.4f}")

    # Generate visualizations
    print("\n=== Generating comparison plots ===")
    plot_comparison(all_data, OUT_DIR)

    print("\n=== Generating trajectory plots ===")
    for name, means in all_data.items():
        print(f"\n  [{name}]")
        plot_trajectories(name, means, OUT_DIR)

    print(f"\nAll visualizations saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
