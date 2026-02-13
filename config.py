from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ObjectConfig:
    """Per-object configuration for multi-object scenes."""
    name: str = "object"
    ply_path: str = ""
    scene_rotate_x: float = 0.0
    scene_rotate_y: float = 0.0
    scene_rotate_z: float = 0.0
    center_to_origin: bool = True
    position_offset: tuple = (0.0, 0.0, 0.0)
    color_white_point: float = 1.0
    num_coarse_cp: int = None   # None → use global default
    num_fine_cp: int = None
    temp_weight_mult: float = 1.0      # multiplier on global temporal weight
    spatial_weight_mult: float = 1.0   # multiplier on global spatial weight
    displacement_axis_weights: tuple = (0.0, 0.0, 0.0)  # per-axis displacement anchor (X, Y, Z)


@dataclass
class TrainConfig:
    # Data
    ply_path: str = '../data/1.ply'
    text_prompt: str = "A object is moving"

    # Resolution (832x464 matches Wan 2.2 native, no resize needed)
    render_width: int = 832
    render_height: int = 464

    # Time
    num_frames: int = 16

    # Control points
    num_coarse_cp: int = 80
    num_fine_cp: int = 300
    K_neighbors: int = 10

    # Training
    total_iterations: int = 500
    batch_size: int = 4
    fine_start_ratio: float = 0.5
    reinit_step: int = 100

    # Learning rates (log-linear decay) — CHORD paper values
    lr_deformation: float = 0.006
    lr_deformation_end: float = 0.00006
    lr_scale: float = 0.006
    lr_scale_end: float = 0.00006

    # CFG — CHORD paper: 25 → 12
    cfg_scale_start: float = 25.0
    cfg_scale_end: float = 12.0

    # Regularization weights
    temp_weight_start: float = 9.6
    temp_weight_end: float = 1.6
    accel_weight_start: float = 0.5
    accel_weight_end: float = 0.5
    spatial_weight_start: float = 3000.0
    spatial_weight_end: float = 300.0
    num_arap_points: int = 5000

    # Blending weight refresh
    weight_refresh_every: int = 50

    # Reinit: copy ref frame deformation to later frames
    reinit_ref_ratio: float = 0.75  # t_ref = int(num_frames * ratio)

    # Camera
    fovy_deg: float = 49.1
    camera_radius: float = 3.0
    elevation_range: tuple = (-15, 15)
    camera_radius_jitter: float = 0.1  # ±10% radius variation
    camera_follow: bool = False  # Track per-frame object center (for walking animals etc.)

    # Output
    output_dir: str = 'outputs'
    scene_name: str = 'default'
    save_every: int = 100
    log_every: int = 10

    # SDS model (set to None to use fake SDS for testing)
    sds_model_name: str = None

    # Scene transform (applied after loading PLY: rotate then center)
    scene_rotate_x: float = 0.0
    scene_rotate_y: float = 0.0
    scene_rotate_z: float = 0.0
    center_to_origin: bool = True

    # Color levels adjustment (remap [0, white_point] -> [0, 1])
    color_white_point: float = 1.0

    # Device
    device: str = 'cuda:0'

    # Multi-object
    objects: list = field(default_factory=list)  # List[dict] → converted to ObjectConfig at runtime

    # Displacement regularization (anchors objects to original position)
    displacement_weight_start: float = 0.0   # default off for backwards compat
    displacement_weight_end: float = 0.0
    displacement_axis_weights: tuple = (0.0, 0.0, 0.0)  # per-axis weights (X, Y, Z)

    # Contact loss (multi-object only)
    contact_weight_start: float = 100.0
    contact_weight_end: float = 10.0
    contact_target_distance: float = 0.05
    impact_weight: float = 0.0  # impact_deformation_loss weight (0 = disabled)
    impact_influence_radius: float = 0.3

    # Wandb
    use_wandb: bool = True
    wandb_project: str = 'chord-4dgs'
    wandb_run_name: str = None
