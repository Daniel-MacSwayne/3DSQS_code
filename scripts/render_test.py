"""
render_test.py: Visual inspection tests for the CUDA superquadric rasterizer.

Camera setup:
  viewmat = eye(4)  →  camera sits at the world origin, looking along +z.
  All means3D values are in camera space: positive-z = in front of the camera.
  A splat at camera-space [0, 0, 3] is therefore 3 units "ahead" of the camera
  (equivalent to placing the camera at world [0,0,-3] facing the world origin).

Output: <results>/render_tests/
  Left half of each image  = colour render
  Right half               = depth map (lighter pixel = closer to camera)
"""

import sys
sys.path.insert(0, '/home/daniel/Documents/Projects/Sync/3D/3DSQS/submodules/diff-superquadric-rasterization')
from diff_superquadric_rasterization import SuperquadricRasterizationSettings, SuperquadricRasterizer

import torch
import math
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

# ── Configuration ─────────────────────────────────────────────────────────────

OUTPUT_DIR = '/home/daniel/Documents/Projects/Sync/3D/Results/render_tests'
os.makedirs(OUTPUT_DIR, exist_ok=True)

device, dtype = 'cuda', torch.float32
H, W   = 512, 512
FOV    = 60.0                             # horizontal & vertical field of view (degrees)
TANFOV = math.tan(math.radians(FOV / 2))

# Identity viewmatrix: camera at world origin, looking along +z.
# R_cw extracted from this is eye(3), so R_cs = R_ws (splat world rotation).
viewmat = torch.eye(4, device=device, dtype=dtype)
bg      = torch.zeros(3, device=device, dtype=dtype)  # black background

settings = SuperquadricRasterizationSettings(
    image_height=H, image_width=W,
    tanfovx=TANFOV, tanfovy=TANFOV,
    bg=bg, viewmatrix=viewmat)
rast = SuperquadricRasterizer(settings)

# ── Parameter ranges (from GaussianModel2 / Constrain_) ───────────────────────
#   e1, e2 ∈ [0.1, 1.9]   (sigmoid * 1.8 + 0.1)
#     e1=e2=0.1 → box
#     e1=e2=1.0 → sphere / ellipsoid
#     e1=e2=1.9 → diamond / star cross-section
#   e3 ∈ [0.9, 5.0]        (sigmoid * 4.1 + 0.9)
#     small e3 → soft wide falloff
#     large e3 → hard sharp boundary
#   scale: actual scale in world units (GaussianModel2 uses exp(_scaling))

# ── Helpers ───────────────────────────────────────────────────────────────────

def quat_from_axis_angle(axis=(0., 1., 0.), angle_deg=0.):
    """Unit quaternion [q1, q2, q3, q4=w] for a rotation about axis by angle_deg."""
    # Steps:
    #   1. Normalise axis
    #   2. Compute [sin(a/2)*axis | cos(a/2)] half-angle form
    a = math.radians(angle_deg)
    x, y, z = axis
    n = math.sqrt(x*x + y*y + z*z) + 1e-12
    x, y, z = x/n, y/n, z/n
    s = math.sin(a / 2)
    return torch.tensor([[x*s, y*s, z*s, math.cos(a/2)]], device=device, dtype=dtype)

def random_quat(seed=None):
    """Uniformly sampled unit quaternion."""
    if seed is not None:
        torch.manual_seed(seed)
    q = torch.randn(1, 4, device=device, dtype=dtype)
    return q / q.norm(dim=1, keepdim=True)

def make_splat(pos_cam, color, opacity, scale, quat, exps):
    """Pack a single splat into per-tensor tuples expected by the rasterizer.

    Steps:
      1. Convert each scalar list to a (1, D) tensor
      2. Return as a named dict for clarity
    """
    # Step 1
    return dict(
        m = torch.tensor([pos_cam], device=device, dtype=dtype),   # (1, 3) camera-space
        c = torch.tensor([color],   device=device, dtype=dtype),   # (1, 3) RGB in [0,1]
        o = torch.tensor([[opacity]], device=device, dtype=dtype),  # (1, 1)
        s = torch.tensor([scale],   device=device, dtype=dtype),   # (1, 3) actual scale
        q = quat,                                                    # (1, 4) unit quat
        e = torch.tensor([exps],    device=device, dtype=dtype),   # (1, 3) [e1, e2, e3]
    )

def render_scene(splat_list, label=''):
    """Render a list of splat dicts; return a side-by-side colour+depth PIL image.

    Steps:
      1. Concatenate all splat tensors
      2. Call the rasterizer (no grad needed for inspection)
      3. Build colour image and normalised depth image
      4. Concatenate side-by-side and overlay text label
    """
    # Step 1: concatenate
    keys = ['m', 'c', 'o', 's', 'q', 'e']
    m, c, o, s, q, e = [torch.cat([sp[k] for sp in splat_list], dim=0) for k in keys]

    # Step 2: render
    with torch.no_grad():
        color_t, depth_t, radii = rast(m, c, o, s, q, e)

    # Step 3a: colour image (3, H, W) → (H, W, 3) uint8
    img_np = (color_t.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()

    # Step 3b: depth map (WHITE = close, BLACK = far)
    # d_np = depth_t.cpu().numpy()
    # mask = d_np > 0
    
    # depth_gray = np.zeros((H, W), dtype=np.float32)
    
    # if mask.any():
    #     z = d_np[mask]
    
    #     # normalize depth
    #     z = (z - z.min()) / (z.max() - z.min() + 1e-8)
    
    #     # invert so: close -> 1.0 (white), far -> 0.0 (black)
    #     z = 1.0 - z
    
    #     depth_gray[mask] = z
    
    # # convert to RGB (white=near, black=far)
    # depth_np = (depth_gray * 255).astype(np.uint8)
    # depth_np = np.repeat(depth_np[:, :, None], 3, axis=2)


    # Step 3b: depth map — VIRIDIS VISUAL DEBUG (no assumptions about near/far)
    
    d_np = depth_t.cpu().numpy()
    mask = d_np > 0
    
    depth_vis = np.zeros((H, W, 3), dtype=np.uint8)
    
    if mask.any():
        z = d_np[mask]
    
        # IMPORTANT: no normalization across image
        # just compress to [0,1] for visualization stability
        z = (z - z.min()) / (z.max() - z.min() + 1e-8)
    
        cmap = plt.get_cmap("viridis")
        colors = (cmap(z)[:, :3] * 255).astype(np.uint8)
    
        depth_vis[mask] = colors
    
    depth_np = depth_vis



    # Step 4: side-by-side, then label
    combined = np.concatenate([img_np, depth_np], axis=1)
    pil  = Image.fromarray(combined)
    draw = ImageDraw.Draw(pil)
    draw.text((6, 6),  label, fill=(255, 255, 180))
    draw.text((W + 6, 6), 'depth (lighter=closer)', fill=(200, 200, 200))

    return pil, radii.tolist()

def save(pil, name):
    path = os.path.join(OUTPUT_DIR, name)
    pil.save(path)
    print(f'    -> {path}')


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1 — Single splat: sweep over shape parameter combinations
# ═════════════════════════════════════════════════════════════════════════════
print()
print('TEST 1: Single splat — shape sweep')
print('  Splat at camera-space [0, 0, 3]; scale=0.3; random colour; random rotation.')
print()

# Each entry: (filename_tag, e1, e2, e3, scale_xyz, description)
shapes = [
    ('sphere',       1.0,  1.0,  1.0,  [0.30, 0.30, 0.30], 'Sphere  e1=1 e2=1 e3=1'),
    ('box',          0.15, 0.15, 1.0,  [0.30, 0.30, 0.30], 'Box    e1=.15 e2=.15 e3=1'),
    ('diamond',      1.8,  1.8,  1.0,  [0.30, 0.30, 0.30], 'Diamond e1=1.8 e2=1.8 e3=1'),
    ('sharp_edge',   1.0,  1.0,  4.5,  [0.30, 0.30, 0.30], 'Sharp   e3=4.5 (hard boundary)'),
    ('soft',         1.0,  1.0,  0.95, [0.30, 0.30, 0.30], 'Soft    e3=0.95 (wide Gaussian)'),
    ('x_ridge',      0.12, 1.8,  1.0,  [0.30, 0.30, 0.30], 'Ridge  e1=.12 e2=1.8 (one axis sharp)'),
    ('ellipse_flat', 1.0,  1.0,  1.0,  [0.50, 0.15, 0.15], 'Stretched scale [.5,.15,.15]'),
    ('box_sharp',    0.15, 0.15, 4.0,  [0.30, 0.30, 0.30], 'Box+sharp  e1=.15 e2=.15 e3=4'),
    ('random_A',     None, None, None, None,                'Random shape A'),
    ('random_B',     None, None, None, None,                'Random shape B'),
]

for i, (tag, e1, e2, e3, sc, desc) in enumerate(shapes):
    torch.manual_seed(i * 13 + 7)

    # Randomise missing params
    if e1 is None:
        e1 = float(torch.empty(1).uniform_(0.10, 1.90))
        e2 = float(torch.empty(1).uniform_(0.10, 1.90))
        e3 = float(torch.empty(1).uniform_(0.90, 5.00))
        sc = [float(torch.empty(1).uniform_(0.15, 0.45))] * 3

    color = torch.rand(3, device=device, dtype=dtype).tolist()

    sp = make_splat(
        pos_cam = [0., 0., 3.],
        color   = color,
        opacity = 0.95,
        scale   = sc,
        quat    = random_quat(seed=i),
        exps    = [e1, e2, e3],
    )

    label = f'{desc}\ne=[{e1:.2f},{e2:.2f},{e3:.2f}]  sc={[round(v,2) for v in sc]}'
    pil, radii = render_scene([sp], label=label)
    fname = f'test1_{i:02d}_{tag}.png'
    save(pil, fname)
    print(f'  {tag:15s}  e=[{e1:.2f},{e2:.2f},{e3:.2f}]  scale={[round(v,2) for v in sc]}  radius={radii}')


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2 — Two overlapping splats: alpha blending scenarios
# ═════════════════════════════════════════════════════════════════════════════
print()
print('TEST 2: Two overlapping splats — alpha blending')
print()

# identity rotation (axis-aligned) for cleaner inspection
identity_q = quat_from_axis_angle((0, 1, 0), 0)

blend_cases = [
    # tag, front_pos, back_pos, front_col, back_col, front_op, back_op, exps_f, exps_b
    ('full_overlap_equal',
     [0.,0.,2.5], [0.,0.,3.5],
     [1.,0.,0.], [0.,0.,1.],
     0.7, 0.9,
     [1.,1.,1.], [1.,1.,1.]),

    ('full_overlap_opaque_front',
     [0.,0.,2.5], [0.,0.,3.5],
     [1.,1.,0.], [0.,0.,1.],
     0.98, 0.9,
     [1.,1.,1.], [1.,1.,1.]),

    ('full_overlap_thin_front',
     [0.,0.,2.5], [0.,0.,3.5],
     [1.,0.,1.], [0.,1.,0.],
     0.25, 0.9,
     [1.,1.,1.], [1.,1.,1.]),

    ('partial_offset',
     [-0.15,0.,2.5], [0.15,0.,3.0],
     [1.,0.,0.], [0.,0.8,0.2],
     0.85, 0.9,
     [1.,1.,1.], [1.,1.,1.]),

    ('box_over_sphere',
     [0.,0.,2.5], [0.,0.,3.5],
     [1.,0.5,0.], [0.2,0.4,1.],
     0.8, 0.9,
     [0.15,0.15,1.], [1.,1.,1.]),   # front=box, back=sphere

    ('sphere_over_diamond',
     [0.,0.,2.5], [0.,0.,3.5],
     [0.8,0.9,0.2], [0.1,0.3,1.0],
     0.6, 0.9,
     [1.,1.,1.], [1.8,1.8,1.]),     # front=sphere, back=diamond
]

for i, (tag, fp, bp, fc, bc, fo, bo, ef, eb) in enumerate(blend_cases):
    front = make_splat(fp, fc, fo, [0.30,0.30,0.30], identity_q, ef)
    back  = make_splat(bp, bc, bo, [0.35,0.35,0.35], identity_q, eb)

    label = (f'{tag}\n'
             f'front z={fp[2]} op={fo}  {[round(v,1) for v in fc]}\n'
             f'back  z={bp[2]} op={bo}  {[round(v,1) for v in bc]}')
    pil, _ = render_scene([front, back], label=label)
    fname = f'test2_{i:02d}_{tag}.png'
    save(pil, fname)
    print(f'  {tag}')


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3 — Multiple overlapping splats: depth ordering and colour mixing
# ═════════════════════════════════════════════════════════════════════════════
print()
print('TEST 3: 6 overlapping splats — depth ordering and blending')
print()

torch.manual_seed(42)
colours6 = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]]

# 3a: all centred, stacked at different depths
splats_centred = []
for j in range(6):
    e1 = float(torch.empty(1).uniform_(0.3, 1.5))
    e2 = float(torch.empty(1).uniform_(0.3, 1.5))
    e3 = float(torch.empty(1).uniform_(0.9, 3.0))
    splats_centred.append(make_splat(
        pos_cam = [0., 0., 2.5 + j * 0.3],
        color   = colours6[j],
        opacity = 0.55,
        scale   = [0.28, 0.28, 0.28],
        quat    = random_quat(seed=j + 200),
        exps    = [e1, e2, e3],
    ))

pil, _ = render_scene(splats_centred, label='6 centred splats\nz=2.5..4.0  op=0.55 each')
save(pil, 'test3_00_stacked_centre.png')
print('  stacked_centre')

# 3b: scattered offsets
torch.manual_seed(77)
splats_scatter = []
for j in range(6):
    x = float(torch.empty(1).uniform_(-0.25, 0.25))
    y = float(torch.empty(1).uniform_(-0.25, 0.25))
    e1 = float(torch.empty(1).uniform_(0.3, 1.5))
    e2 = float(torch.empty(1).uniform_(0.3, 1.5))
    e3 = float(torch.empty(1).uniform_(0.9, 3.0))
    splats_scatter.append(make_splat(
        pos_cam = [x, y, 2.5 + j * 0.35],
        color   = colours6[j],
        opacity = 0.65,
        scale   = [0.25, 0.25, 0.25],
        quat    = random_quat(seed=j + 300),
        exps    = [e1, e2, e3],
    ))

pil, _ = render_scene(splats_scatter, label='6 scattered splats\nrandom xy offset  op=0.65')
save(pil, 'test3_01_scattered.png')
print('  scattered')

print()
print(f'Done. Images in: {OUTPUT_DIR}')
print('  Left  = colour render')
print('  Right = depth (lighter = closer to camera)')
