"""Check placement: pillow flat on ground, cat beside it on ground."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from PIL import Image

from models.gaussian_model import GaussianModel
from core.renderer import render_gaussians
from utils.camera_utils import orbit_camera, get_intrinsics

CAT_PLY = "/mnt/workspace/fblan/trajectory-generation/4dgs_generation/data/cat.ply"
PILLOW_PLY = "/mnt/workspace/fblan/trajectory-generation/4dgs_generation/sds_data/single_object/ply/pillow5.ply"


def load_obj(ply_path, rotate_x=0, offset=(0,0,0), device='cuda:0'):
    gs = GaussianModel()
    gs.load_ply(ply_path)
    gs.apply_rotation(rotate_x, 0, 0)
    gs.center_to_origin()
    if offset != (0, 0, 0):
        gs.apply_offset(list(offset))
    gs.to(device)
    return gs


def stats(name, gs):
    m = gs.means
    print(f"  [{name}] Y:[{m[:,1].min():.2f},{m[:,1].max():.2f}] "
          f"Z:[{m[:,2].min():.2f},{m[:,2].max():.2f}] "
          f"X:[{m[:,0].min():.2f},{m[:,0].max():.2f}]")


def render_composite(gs_list, viewmat, K, W, H, device='cuda:0'):
    all_means = torch.cat([g.means for g in gs_list], dim=0)
    all_quats = torch.cat([g.quaternions for g in gs_list], dim=0)
    all_scales = torch.cat([g.get_activated_scales() for g in gs_list], dim=0)
    all_opac = torch.cat([g.get_activated_opacities() for g in gs_list], dim=0)
    min_k = min(g.get_sh_coeffs().shape[1] for g in gs_list)
    all_sh = torch.cat([g.get_sh_coeffs()[:, :min_k, :] for g in gs_list], dim=0)
    sd = int(math.sqrt(min_k)) - 1
    img, _ = render_gaussians(all_means, all_quats, all_scales, all_sh, all_opac,
                              viewmat, K, W, H, sh_degree=sd)
    return (img / 0.2).clamp(0, 1)


def test_config(name, cat_offset, pillow_offset, out_dir, device='cuda:0'):
    cat = load_obj(CAT_PLY, rotate_x=90, offset=cat_offset, device=device)
    pillow = load_obj(PILLOW_PLY, rotate_x=90, offset=pillow_offset, device=device)
    print(f"\n[{name}]  cat_off={cat_offset}  pillow_off={pillow_offset}")
    stats("cat", cat)
    stats("pillow", pillow)

    W, H = 512, 288
    K = get_intrinsics(49.1, W, H)
    center = torch.cat([cat.means, pillow.means], dim=0).mean(dim=0).cpu()

    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    for vn, el, az in [('front', 15, 0), ('side', 15, 90), ('diag', 20, 45),
                        ('back', 15, 180), ('top', 55, 45)]:
        vm = orbit_camera(el, az, 4.0, target=center).to(device)
        img = render_composite([cat, pillow], vm, K, W, H, device)
        arr = (img.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(d, f'{vn}.png'))
    print(f"  -> {d}/")


def main():
    device = 'cuda:0'
    out = os.path.join(os.path.dirname(__file__), 'placement_check')

    # Y-down: +Y = ground, -Y = up
    # Cat (90°X centered): Y=[-1.0, 0.73] feet=0.73  Z=[-1.07, 1.40] length=2.47
    # Pillow (90°X centered): Y=[-0.51, 0.51] bottom=0.51  Z=[-0.93, 0.93]

    # Ground level = pillow bottom = 0.51
    # Cat feet at 0.73, need to go to 0.51 → offset Y = 0.51 - 0.73 = -0.22
    # Cat beside pillow in Z: pillow Z edge=0.93, cat Z center=0 → offset Z=+1.5 (behind pillow)

    # Config E: Cat beside pillow (Z offset), both on same ground
    test_config("E_beside_z1.5",
        cat_offset=(0, -0.22, 1.5),
        pillow_offset=(0, 0, 0),
        out_dir=out, device=device)

    # Config F: Cat beside pillow closer
    test_config("F_beside_z1.2",
        cat_offset=(0, -0.22, 1.2),
        pillow_offset=(0, 0, 0),
        out_dir=out, device=device)

    # Config G: Cat in front of pillow (negative Z)
    test_config("G_front_z-1.5",
        cat_offset=(0, -0.22, -1.5),
        pillow_offset=(0, 0, 0),
        out_dir=out, device=device)

    # Config H: Cat beside pillow (X offset instead)
    test_config("H_beside_x1.2",
        cat_offset=(1.2, -0.22, 0),
        pillow_offset=(0, 0, 0),
        out_dir=out, device=device)

    print("\nDone!")

if __name__ == '__main__':
    main()
