// forward.h: Declarations for the superquadric rasterizer forward pass kernels.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace FORWARD {

// preprocess: Per-splat kernel — project to 2D, estimate radius, compute R_cs.
void preprocess(
    int P,
    const float* means3D,    // (P,3) camera-space centres
    const float* scales,     // (P,3)
    const float* rotations,  // (P,4) quaternion
    const float* exps,       // (P,3)
    const float* viewmatrix, // (4,4) camera-to-world (world_view_transform)
    int   W, int H,
    float focal_x, float focal_y,
    float tan_fovx, float tan_fovy,
    float* means2D_out,      // (P,2) output: pixel-space centres
    float* depths_out,       // (P,)  output: depth
    float* R_cs_out,         // (P,9) output: shape-to-camera rotation
    int*   radii_out,        // (P,)  output: pixel radius
    uint32_t* tiles_touched, // (P,)  output: tiles overlapped
    const dim3 grid);        // tile grid dimensions

// render: Per-tile/pixel kernel — alpha-blend splats in depth order.
void render(
    const dim3 grid, dim3 block,
    const uint2*    ranges,        // (n_tiles,) tile→[start,end] in point_list
    const uint32_t* point_list,   // (R,) sorted splat indices
    int W, int H,
    float focal_x, float focal_y,
    const float* bg,              // (3,) background colour
    const float2* means2D,        // (P,2)
    const float*  depths,         // (P,)
    const float*  R_cs,           // (P,9)
    const float*  scales,         // (P,3)
    const float*  exps,           // (P,3)
    const float*  colors,         // (P,3)
    const float*  opacities,      // (P,1)
    float*  final_T,              // (H*W,) output: transmittance at last contributing splat
    uint32_t* n_contrib,          // (H*W,) output: index of last contributor
    float*  out_color,            // (3,H,W) output
    float*  out_depth);           // (H,W)   output

} // namespace FORWARD
