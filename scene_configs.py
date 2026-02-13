"""
Per-scene preset configurations.

Usage:
    python train.py --scene cat
    python train.py --scene cactus
    python train.py --scene cat --total_iterations 20000   # CLI overrides preset
"""

SCENE_PRESETS = {
    "cactus": {
        "ply_path": "../data/1.ply",
        "text_prompt": "A cactus in a pot swaying left and right gently",
        "scene_name": "cactus",
        "camera_radius": 3.0,
        "num_coarse_cp": 10000,
        "num_fine_cp": 10000,
        # No rotation needed, no color adjustment
        "scene_rotate_x": 0.0,
        "scene_rotate_y": 0.0,
        "scene_rotate_z": 0.0,
        "color_white_point": 1.0,
    },
    "cat": {
        "ply_path": "../data/cat.ply",
        "text_prompt": "A cat walking on the floor",
        "scene_name": "cat",
        "camera_radius": 2.5,
        "num_coarse_cp": 5000,
        "num_fine_cp": 5000,
        # Cat body runs along Y (vertical), +90° X rotation to stand upright
        "scene_rotate_x": 90.0,
        "scene_rotate_y": 0.0,
        "scene_rotate_z": 0.0,
        # Very dark scene (RGB max ~0.27), remap [0, 0.2] -> [0, 1]
        "color_white_point": 0.2,
        # Camera follows the cat as it walks forward
        "camera_follow": True,
        # Displacement: free in X/Z (walking), anchored in Y (stay on ground)
        "displacement_weight_start": 200.0,
        "displacement_weight_end": 20.0,
        "displacement_axis_weights": (0.0, 3.0, 0.0),
    },
    "can": {
        "ply_path": "../sds_data/packofcan/ply/can_1.ply",
        "text_prompt": "A soda can slowly denting and caving inward under pressure",
        "scene_name": "can",
        "camera_radius": 0.4,
        # Can height along Z in PLY, +90° X to stand upright (-Y is up)
        "scene_rotate_x": 90.0,
        "scene_rotate_y": 0.0,
        "scene_rotate_z": 0.0,
        # Very dark scene (RGB max ~0.28), same fix as cat
        "color_white_point": 0.2,
        "elevation_range": (-10, 25),
    },
    "cat_pillow": {
        "text_prompt": "A cat jumps onto a soft pillow, landing gently as the pillow deforms and compresses under the cat's weight",
        "scene_name": "cat_pillow",
        "camera_radius": 4.0,
        "elevation_range": (-10, 25),
        "color_white_point": 0.2,
        "num_coarse_cp": 50,
        "num_fine_cp": 500,
        # Displacement regularization: anchors objects to original positions
        "displacement_weight_start": 500.0,
        "displacement_weight_end": 50.0,
        # Y-down convention: +Y = ground, -Y = up
        # Both objects on the same ground plane, cat in front of pillow (Z offset)
        "objects": [
            {
                "name": "cat",
                "ply_path": "/mnt/workspace/fblan/trajectory-generation/4dgs_generation/data/cat.ply",
                "scene_rotate_x": 90.0,
                # Cat (90°X centered): Y=[-1.0, 0.73] feet=0.73, Z=[-1.07, 1.40]
                # Align feet to pillow bottom (0.51): offset Y = 0.51 - 0.73 = -0.22
                # Place behind pillow: Z offset must be > 0.93+1.07=2.0 to avoid intersection
                "position_offset": [0.0, -0.22, 2.0],
                "color_white_point": 0.2,
                "num_coarse_cp": 50,
                "num_fine_cp": 500,
                "spatial_weight_mult": 0.5,
                # Free in X/Z (walking/jumping), anchored in Y (no floating)
                "displacement_axis_weights": (0.0, 3.0, 0.0),
            },
            {
                "name": "pillow",
                "ply_path": "/mnt/workspace/fblan/trajectory-generation/4dgs_generation/sds_data/single_object/ply/pillow5.ply",
                # 90°X rotation to lay flat; after centering Y=[-0.51, 0.51], bottom=0.51
                "scene_rotate_x": 90.0,
                "position_offset": [0.0, 0.0, 0.0],
                "color_white_point": 0.2,
                "num_coarse_cp": 30,
                "num_fine_cp": 200,
                "temp_weight_mult": 2.0,
                "spatial_weight_mult": 2.0,
                # Anchored in all axes (stays put), strong vertical
                "displacement_axis_weights": (1.0, 5.0, 1.0),
            },
        ],
    },
    "cat_pillow_free_cat": {
        "text_prompt": "A cat jumps onto a soft pillow, landing gently as the pillow deforms and compresses under the cat's weight",
        "scene_name": "cat_pillow_free_cat",
        "camera_radius": 4.0,
        "elevation_range": (-10, 25),
        "color_white_point": 0.2,
        "num_coarse_cp": 50,
        "num_fine_cp": 500,
        # Displacement only on pillow, cat is fully free
        "displacement_weight_start": 500.0,
        "displacement_weight_end": 50.0,
        "objects": [
            {
                "name": "cat",
                "ply_path": "/mnt/workspace/fblan/trajectory-generation/4dgs_generation/data/cat.ply",
                "scene_rotate_x": 90.0,
                "position_offset": [0.0, -0.22, 2.0],
                "color_white_point": 0.2,
                "num_coarse_cp": 50,
                "num_fine_cp": 500,
                "spatial_weight_mult": 0.5,
                # No displacement on cat — fully free to jump/move
                "displacement_axis_weights": (0.0, 0.0, 0.0),
            },
            {
                "name": "pillow",
                "ply_path": "/mnt/workspace/fblan/trajectory-generation/4dgs_generation/sds_data/single_object/ply/pillow5.ply",
                "scene_rotate_x": 90.0,
                "position_offset": [0.0, 0.0, 0.0],
                "color_white_point": 0.2,
                "num_coarse_cp": 30,
                "num_fine_cp": 200,
                "temp_weight_mult": 2.0,
                "spatial_weight_mult": 2.0,
                "displacement_axis_weights": (1.0, 5.0, 1.0),
            },
        ],
    },
    "dog": {
        "ply_path": "../data/dog.ply",
        "text_prompt": "A dog is walking",
        "scene_name": "dog",
        "camera_radius": 2.5,
        "num_coarse_cp": 100,
        "num_fine_cp": 1000,
        "elevation_range": (-15, 15),
        # Dog body along Y, legs in -Z; -90° X to stand upright
        "scene_rotate_x": -90.0,
        "scene_rotate_y": 0.0,
        "scene_rotate_z": 0.0,
        # Very dark scene (RGB max ~0.28), same fix as cat
        "color_white_point": 0.2,
        # Camera follows the dog as it walks forward
        "camera_follow": True,
    },
}


def get_scene_preset(scene_name: str) -> dict:
    """Return preset dict for a scene, or empty dict if unknown."""
    if scene_name not in SCENE_PRESETS:
        available = ", ".join(SCENE_PRESETS.keys())
        raise ValueError(f"Unknown scene '{scene_name}'. Available: {available}")
    return dict(SCENE_PRESETS[scene_name])
