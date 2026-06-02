# diff_superquadric_rasterization/__init__.py
#
# Python API for the superquadric CUDA rasterizer.
# Mirrors the structure of diff_gaussian_rasterization/__init__.py:
#   - SuperquadricRasterizationSettings: NamedTuple of camera/render parameters
#   - SuperquadricRasterizer: nn.Module with autograd-compatible forward/backward
#   - _RasterizeSuperquadrics: torch.autograd.Function that calls the C extension

from typing import NamedTuple
import torch
import torch.nn as nn
from . import _C


# ============================================================
#  Settings
# ============================================================

class SuperquadricRasterizationSettings(NamedTuple):
    """Camera and render configuration passed to SuperquadricRasterizer.

    Fields match render2() call-site parameters in gaussian_renderer/__init__.py.
    """
    image_height:  int
    image_width:   int
    tanfovx:       float
    tanfovy:       float
    bg:            torch.Tensor   # (3,) background colour on CUDA
    viewmatrix:    torch.Tensor   # (4,4) world_view_transform (camera-to-world, transposed)
    debug:         bool = False


# ============================================================
#  Autograd Function
# ============================================================

class _RasterizeSuperquadrics(torch.autograd.Function):
    """Differentiable wrapper around the CUDA superquadric rasterizer.

    forward:
      Steps:
        1. Call _C.rasterize_superquadrics to get colour, depth, radii + opaque buffers
        2. Save inputs and buffers for backward
        3. Return (render_color, render_depth, radii)

    backward:
      Steps:
        1. Retrieve saved tensors and settings
        2. Call _C.rasterize_superquadrics_backward with dL/d(render_color)
        3. Return gradient tensors for all differentiable inputs (None for non-tensor args)
    """

    @staticmethod
    def forward(ctx,
                means3D,        # (P,3)
                colors,         # (P,3)
                opacities,      # (P,1)
                scales,         # (P,3)
                rotations,      # (P,4)
                exps,           # (P,3)
                # non-tensor settings passed via ctx
                bg,             # (3,)
                viewmatrix,     # (4,4)
                tan_fovx,
                tan_fovy,
                image_height,
                image_width,
                debug):

        # Step 1: call CUDA forward
        num_rendered, out_color, out_depth, out_radii, \
            geomBuffer, binningBuffer, imgBuffer = _C.rasterize_superquadrics(
                bg, means3D, colors, opacities, scales, rotations, exps,
                viewmatrix, tan_fovx, tan_fovy, image_height, image_width, debug)

        # Step 2: save for backward
        ctx.save_for_backward(
            means3D, colors, opacities, scales, rotations, exps,
            bg, viewmatrix, out_radii, geomBuffer, binningBuffer, imgBuffer)
        ctx.num_rendered  = num_rendered
        ctx.tan_fovx      = tan_fovx
        ctx.tan_fovy      = tan_fovy
        ctx.image_height  = image_height
        ctx.image_width   = image_width
        ctx.debug         = debug

        # Step 3: return differentiable outputs
        return out_color, out_depth, out_radii

    @staticmethod
    def backward(ctx, dL_dcolor, dL_ddepth, _dL_dradii):
        # Step 1: retrieve saved state
        (means3D, colors, opacities, scales, rotations, exps,
         bg, viewmatrix, out_radii,
         geomBuffer, binningBuffer, imgBuffer) = ctx.saved_tensors

        # Step 2: call CUDA backward
        dL_dmeans3D, dL_dcolors, dL_dopacity, \
        dL_dscales, dL_drotations, dL_dexps = _C.rasterize_superquadrics_backward(
            bg, means3D, out_radii, colors, opacities, scales, rotations, exps,
            viewmatrix, ctx.tan_fovx, ctx.tan_fovy,
            dL_dcolor.contiguous(),
            ctx.num_rendered,
            geomBuffer, binningBuffer, imgBuffer,
            ctx.debug)

        # Step 3: return gradients in the same order as forward inputs
        # Non-tensor args (tan_fovx etc.) get None
        return (dL_dmeans3D, dL_dcolors, dL_dopacity,
                dL_dscales, dL_drotations, dL_dexps,
                None, None, None, None, None, None, None)


# ============================================================
#  Public nn.Module
# ============================================================

class SuperquadricRasterizer(nn.Module):
    """Drop-in replacement for GaussianRasterizer, tailored for superquadric splats.

    Usage:
        rasterizer = SuperquadricRasterizer(raster_settings)
        render_color, render_depth, radii = rasterizer(
            means3D, colors_precomp, opacity, scales, rotations, exps)

    Steps (forward):
      1. Unpack camera settings from raster_settings
      2. Call _RasterizeSuperquadrics.apply with all splat parameters
      3. Return (render_color (3,H,W), render_depth (H,W), radii (P,))
    """

    def __init__(self, raster_settings: SuperquadricRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(self,
                means3D:    torch.Tensor,  # (P,3) camera-frame centres
                colors:     torch.Tensor,  # (P,3) pre-computed RGB
                opacity:    torch.Tensor,  # (P,1)
                scales:     torch.Tensor,  # (P,3)
                rotations:  torch.Tensor,  # (P,4) quaternion
                exps:       torch.Tensor,  # (P,3) [e1, e2, e3]
                ):
        # Step 1: unpack settings
        s = self.raster_settings

        # Step 2: call differentiable forward
        out_color, out_depth, radii = _RasterizeSuperquadrics.apply(
            means3D, colors, opacity, scales, rotations, exps,
            s.bg, s.viewmatrix,
            s.tanfovx, s.tanfovy,
            s.image_height, s.image_width,
            s.debug)

        # Step 3: return
        return out_color, out_depth, radii
