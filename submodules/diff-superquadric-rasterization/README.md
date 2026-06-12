# diff-superquadric-rasterization

A CUDA rasterizer for superquadric splats, built as a drop-in replacement for `diff-gaussian-rasterization`. Follows the same tile-based architecture but evaluates a superquadric implicit distance function instead of an ellipsoidal Gaussian.

---

## What This Is

Standard 3D Gaussian Splatting uses ellipsoidal splats — each one has a smooth Gaussian falloff and no ability to represent sharp edges, flat surfaces, or box-like shapes. This rasterizer extends that to **superquadric splats**, which can continuously morph between:

- Spheres (e1=e2=1)
- Boxes / cuboids (e1, e2 → 0)
- Sharp-edged cylinders
- Pillow shapes
- Star-like cross-sections

The shape is controlled by two extra learned parameters per splat — the exponents `e1`, `e2`, `e3` — on top of the standard position, scale, rotation, colour, and opacity.

---

## Architecture

The pipeline mirrors `diff-gaussian-rasterization` exactly:

```
Input splats (P visible)
        │
        ▼
preprocessCUDA          (one thread per splat)
  • Project 3D → 2D pixel centre
  • Estimate pixel-space bounding radius
  • Build R_cs = R_ws @ R_cw  (shape-to-camera rotation)
  • Count tile overlaps
        │
        ▼
CUB InclusiveSum        (prefix sum over tile counts)
        │
        ▼
duplicateWithKeys       (one entry per splat-tile overlap, R total)
  • Key = tile_id << 32 | depth_bits
        │
        ▼
CUB RadixSort           (sort by tile, then depth within tile)
        │
        ▼
identifyTileRanges      (mark start/end of each tile in sorted list)
        │
        ▼
renderCUDA              (one thread per pixel, 16×16 tile blocks)
  • For each splat in this tile (front to back):
      - Compute camera-plane offset at splat depth
      - Rotate [dx, dy, 0] into shape-local frame via R_cs
      - Evaluate superquadric distance F
      - weight G = exp(-F)
      - alpha = min(0.99, opacity * G)
      - Alpha-blend: C += color * alpha * T;  T *= (1-alpha)
  • Write colour, depth, transmittance, last-contributor index
        │
        ▼
Output: render_color (3,H,W), render_depth (H,W), radii (P,)
```

The backward mirrors this in reverse, with one render backward kernel and one preprocess backward kernel.

---

## Key Design Decision: Depth-Plane Approximation

Standard GS projects a 3D Gaussian analytically to a 2D ellipse (EWA splatting). Superquadrics have no closed-form 2D projection — the Python rasterizer estimated it by tracing 29 rim sample points per splat.

This CUDA version instead uses the **depth-plane approximation**: evaluate the superquadric at z=0 in camera space (the plane perpendicular to the camera axis at the splat's depth). For each pixel:

```
camera-plane offset:  dx = xi*d - cx_splat,   dy = yi*d - cy_splat,   dz = 0

shape-local coords:   [xs, ys, zs] = R_cs @ [dx, dy, 0]

superquadric distance: F = ( (|xs/s1|^(2/e2) + |ys/s2|^(2/e2))^(e2/e1) + |zs/s3|^(2/e1) )^e3

weight: G = exp(-F)
```

Because R_cs mixes the camera-plane x and y into all three shape-local axes, the exponents `e1`, `e2`, `e3` and all three scale dimensions remain meaningful even though `dz=0`. The approximation is accurate for non-severely-tilted splats and avoids all the degeneracy handling required by the rim approach.

---

## File Structure

```
diff-superquadric-rasterization/
├── setup.py                          # Build: pip install -e .
├── ext.cpp                           # pybind11 bindings → Python _C module
├── rasterize_superquadrics.h         # C++ function declarations
├── rasterize_superquadrics.cu        # PyTorch tensor → raw pointer bridge
└── cuda_rasterizer/
    ├── config.h                      # BLOCK_X=16, BLOCK_Y=16, NUM_CHANNELS=3
    ├── auxiliary.h                   # Device math: dabsPow, superquadricDistance,
    │                                 #   superquadricDistanceGrad, quatToMatrix,
    │                                 #   matMul3x3, matvec3x3, getRect, transformPoint
    ├── rasterizer.h                  # CudaRasterizer::Rasterizer (static forward/backward)
    ├── rasterizer_impl.h             # GeometryState, ImageState, BinningState buffer structs
    ├── rasterizer_impl.cu            # Orchestration: preprocess → sort → render
    ├── forward.h / forward.cu        # preprocessCUDA + renderCUDA kernels
    └── backward.h / backward.cu      # renderBackwardCUDA + preprocessBackwardCUDA
                                      #   + means3DBackwardCUDA kernels

diff_superquadric_rasterization/
    └── __init__.py                   # Python API:
                                      #   SuperquadricRasterizationSettings (NamedTuple)
                                      #   SuperquadricRasterizer (nn.Module)
                                      #   _RasterizeSuperquadrics (autograd.Function)
```

---

## Python API

```python
from diff_superquadric_rasterization import (
    SuperquadricRasterizationSettings,
    SuperquadricRasterizer,
)

settings = SuperquadricRasterizationSettings(
    image_height = H,
    image_width  = W,
    tanfovx      = math.tan(fov_x / 2),
    tanfovy      = math.tan(fov_y / 2),
    bg           = torch.zeros(3, device='cuda'),   # background colour
    viewmatrix   = camera.world_view_transform,      # (4,4) camera-to-world (transposed)
)

rasterizer = SuperquadricRasterizer(raster_settings=settings)

render_color, render_depth, radii = rasterizer(
    means3D   = means3D,    # (P, 3)  splat centres in camera frame
    colors    = colors,     # (P, 3)  pre-computed RGB
    opacity   = opacity,    # (P, 1)
    scales    = scales,     # (P, 3)  [s1, s2, s3]
    rotations = rotations,  # (P, 4)  quaternion [q1, q2, q3, q4]
    exps      = exps,       # (P, 3)  [e1, e2, e3]
)
# render_color: (3, H, W)
# render_depth: (H, W)
# radii:        (P,)  pixel-space bounding radii
```

The forward and backward passes are fully differentiable. Gradients flow back to all six input tensors.

---

## Backward Pass

The backward correctly computes gradients for all splat parameters using the **full transmittance-aware formula** from the 3DGS paper:

```
dL/d(alpha_j) = sum_c [ (T_j * c_j_c - R_j_c / (1 - alpha_j)) * dL/dC_c ]
```

where `R_j` is the remaining colour after splat j (contributions from deeper splats plus background). This is tracked incrementally going back-to-front as `C_remaining`.

Gradient outputs:
| Parameter | Method |
|---|---|
| `colors` | Direct: `alpha * T_before * dL/dC` |
| `opacity` | Via alpha clamp: `G * dL/d(alpha)` |
| `scales` | Chain rule through superquadric distance |
| `exps` | Chain rule including log terms for fractional-power derivatives |
| `rotations` | Via `dL/d(R_cs) → dL/d(R_ws) @ R_cw^T → dL/d(quaternion)` |
| `means3D` | Via 2D projection Jacobian: `dL/d(px) * fx/z`, `dL/d(pz) = -x*fx/z^2 - ...` |

---

## Buffer Layout

Three opaque byte tensors are saved for the backward pass (same pattern as `diff-gaussian-rasterization`):

**GeometryState** (per-splat):
- `means2D` (P, float2) — projected 2D pixel centres
- `depths` (P, float) — view-space depth
- `R_cs` (P×9, float) — shape-to-camera rotation matrices
- `radii` (P, int) — pixel-space bounding radii
- `tiles_touched`, `point_offsets` (P, uint32) — for CUB prefix sum

**ImageState** (per-tile / per-pixel):
- `ranges` (n_tiles, uint2) — [start, end) in sorted splat list per tile
- `n_contrib` (H×W, uint32) — index of last contributing splat per pixel
- `accum_alpha` (H×W, float) — final transmittance T per pixel

**BinningState** (per rendered instance):
- `point_list_keys` / `point_list_keys_unsorted` (R, uint64) — tile+depth keys
- `point_list` / `point_list_unsorted` (R, uint32) — splat indices

---

## Integration with 3DSQS

`gaussian_renderer/__init__.py` automatically uses the CUDA rasterizer when available:

```python
try:
    from diff_superquadric_rasterization import SuperquadricRasterizationSettings, SuperquadricRasterizer
    _CUDA_SQ_AVAILABLE = True
except ImportError:
    _CUDA_SQ_AVAILABLE = False
```

Inside `render2()`, the CUDA path replaces the Python `rasterizer3()` loop with the `SuperquadricRasterizer`. The Python `rasterizer3()` remains as a fallback for debugging.

---

## Build

Requires CUDA 11.8+, PyTorch with matching CUDA, and an SM_86 GPU (RTX 4090). To build for other architectures, change `-arch=sm_86` in `setup.py`.

```bash
cd submodules/diff-superquadric-rasterization
conda run -n 3DSQS python setup.py build_ext --inplace
# or
conda run -n 3DSQS pip install -e .  # requires torch available in pip's subprocess
```

---

## Gradient Accuracy

Verified with finite-difference checks on a 3-splat scene:

| Parameter | FD eps | Max relative error | Notes |
|---|---|---|---|
| colors | 1e-3 | < 0.05% | Smooth, no discontinuities |
| opacity | 1e-4 | < 0.3% | Use eps < 1e-3 to stay below alpha-threshold discontinuity |
| scales | 1e-2 | ~4% | Discrete pixel-boundary effect (radius floor) |

The alpha=1/255 threshold and the `ceilf(radius)` discretisation both create non-smooth points in the forward function. FD checks at eps values that straddle these boundaries will show large errors — this is expected and is not a bug in the backward. The gradient is correct and produces converging training loss.

---

## Differences from the Python rasterizer3()

| | Python `rasterizer3()` | This CUDA extension |
|---|---|---|
| Per-splat shape eval | Rim interpolation (29 samples) | Depth-plane approx (closed-form) |
| Tile loop | Python `for` loop | CUDA kernel, 16×16 thread blocks |
| Batch size | L=10 splats per iteration | All splats per tile, shared memory |
| Gradient | PyTorch autograd (full graph) | Custom CUDA backward kernels |
| Memory | Dense (N_h, N_w, L_max) map tensor | CUB-sorted (tile_id, depth) list |
| Speed (est.) | Baseline | 10–50× faster |
