from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # Data
    ply_path: str = '../data/1.ply'
    text_prompt: str = "A object is moving"

    # Resolution
    render_width: int = 512
    render_height: int = 288

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

    # Learning rates (log-linear decay)
    lr_deformation: float = 0.002
    lr_deformation_end: float = 0.00002
    lr_scale: float = 0.002
    lr_scale_end: float = 0.00002

    # CFG
    cfg_scale_start: float = 15.0
    cfg_scale_end: float = 7.5

    # Regularization weights
    temp_weight_start: float = 9.6
    temp_weight_end: float = 1.6
    spatial_weight_start: float = 3000.0
    spatial_weight_end: float = 300.0

    # Camera
    fovy_deg: float = 49.1
    camera_radius: float = 3.0
    elevation_range: tuple = (-30, 30)

    # Output
    output_dir: str = 'outputs'
    scene_name: str = 'default'
    save_every: int = 100
    log_every: int = 10

    # SDS model (set to None to use fake SDS for testing)
    sds_model_name: str = None

    # Scene rotation (degrees, applied after loading PLY)
    scene_rotate_x: float = 0.0
    scene_rotate_y: float = 0.0
    scene_rotate_z: float = 0.0

    # Color levels adjustment (remap [0, white_point] -> [0, 1])
    color_white_point: float = 1.0

    # Device
    device: str = 'cuda:0'

    # Wandb
    use_wandb: bool = True
    wandb_project: str = 'chord-4dgs'
    wandb_run_name: str = None
