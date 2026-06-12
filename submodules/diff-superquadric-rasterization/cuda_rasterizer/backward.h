// backward.h: Declarations for the superquadric rasterizer backward pass kernels.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace BACKWARD {

// render: Per-tile/pixel backward kernel.
//   Processes splats in reverse depth order to backprop through the
//   front-to-back alpha compositing formula.
void render(
    const dim3 grid, dim3 block,
    const uint2*    ranges,
    const uint32_t* point_list,
    int W, int H,
    float focal_x, float focal_y,
    const float* bg,
    const float2* means2D,
    const float*  depths,
    const float*  R_cs,
    const float*  scales,
    const float*  exps,
    const float*  colors,
    const float*  opacities,
    const float*  final_T,       // (H*W,) from forward
    const uint32_t* n_contrib,   // (H*W,) from forward
    const float*  dL_dpix_color, // (3,H,W) upstream gradient
    // Gradient accumulators (atomically accumulated per splat):
    float* dL_dmeans2D,   // (P,2)
    float* dL_dcolors,    // (P,3)
    float* dL_dopacity,   // (P,1)
    float* dL_dR_cs,      // (P,9) gradient w.r.t. R_cs (for chain rule to rotation)
    float* dL_dscales,    // (P,3)
    float* dL_dexps);     // (P,3)

// preprocess: Per-splat backward kernel.
//   Backprops dL_dR_cs → dL_drotations and dL_dmeans2D → dL_dmeans3D.
void preprocess(
    int P,
    const float* means3D,
    const float* rotations,
    const float* viewmatrix,
    const int*   radii,
    float        focal_x, float focal_y,
    const float* dL_dR_cs,     // (P,9) from render backward
    const float* dL_dmeans2D,  // (P,2) from render backward
    float* dL_drotations,      // (P,4) output
    float* dL_dmeans3D);       // (P,3) output

} // namespace BACKWARD
