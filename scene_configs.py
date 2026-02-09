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
    },
    "dog": {
        "ply_path": "../data/dog.ply",
        "text_prompt": "A dog is walking",
        "scene_name": "dog",
        "camera_radius": 1.5,
        "num_coarse_cp": 50,
        "num_fine_cp": 500,
        # Dog body along Y, legs in -Z; -90° X to stand upright
        "scene_rotate_x": -90.0,
        "scene_rotate_y": 0.0,
        "scene_rotate_z": 0.0,
        # Very dark scene (RGB max ~0.28), same fix as cat
        "color_white_point": 0.2,
    },
}


def get_scene_preset(scene_name: str) -> dict:
    """Return preset dict for a scene, or empty dict if unknown."""
    if scene_name not in SCENE_PRESETS:
        available = ", ".join(SCENE_PRESETS.keys())
        raise ValueError(f"Unknown scene '{scene_name}'. Available: {available}")
    return dict(SCENE_PRESETS[scene_name])
