// rasterizer.h: Top-level C++ API for the superquadric rasterizer.
//
// Provides the CudaRasterizer::Rasterizer class with static forward/backward methods.
// This is the boundary between the PyTorch tensor wrapper and the CUDA kernels.
//
// Differences from diff-gaussian-rasterization:
//   - Adds 'exps' (M,3) parameter for superquadric shape exponents [e1,e2,e3]
//   - Replaces 3D covariance / conic_opacity with R_cs (shape-to-camera rotation)
//   - Returns render_depth in addition to render_color
//   - No SH evaluation (colours pre-computed before entering the rasterizer)

#pragma once

#include <functional>
#include <cstddef>

namespace CudaRasterizer {

class Rasterizer {
public:

    // forward: Rasterize visible superquadric splats into colour and depth images.
    //
    // Steps:
    //   1. Allocate geometry, image, and binning buffers via the provided lambdas
    //   2. Launch preprocessCUDA: per-splat projection, radius estimation, R_cs computation
    //   3. Prefix-sum tiles_touched to get per-splat offsets
    //   4. Duplicate splat indices once per tile overlap (duplicateWithKeys)
    //   5. Radix-sort the (tile_id | depth) keys
    //   6. Identify per-tile ranges in the sorted list (identifyTileRanges)
    //   7. Launch renderCUDA: per-pixel front-to-back alpha blending
    //
    // Returns: total number of rendered Gaussian-tile instances (R)
    static int forward(
        std::function<char*(size_t)> geometryBuffer,   // allocator for geometry state
        std::function<char*(size_t)> binningBuffer,    // allocator for binning state
        std::function<char*(size_t)> imageBuffer,      // allocator for image state
        int   P,              // number of visible splats (M in Python code)
        const float* bg,      // background colour [3]
        int   width,
        int   height,
        const float* means3D,    // (P,3) splat centres in camera frame
        const float* colors,     // (P,3) pre-computed RGB
        const float* opacities,  // (P,1) opacity
        const float* scales,     // (P,3) [s1, s2, s3]
        const float* rotations,  // (P,4) quaternion [q1,q2,q3,q4]
        const float* exps,       // (P,3) [e1, e2, e3]
        const float* viewmatrix, // (4,4) world_view_transform (camera-to-world, stored transposed)
        float  tan_fovx,
        float  tan_fovy,
        float* out_color,     // (3,H,W) output
        float* out_depth,     // (H,W)   output
        int*   out_radii,     // (P,)    output pixel-space radii
        bool   debug = false);


    // backward: Compute gradients of all splat parameters given dL/d(image).
    //
    // Steps:
    //   1. Reconstruct geometry/image/binning state from saved buffers
    //   2. Launch renderBackwardCUDA: per-pixel backprop through alpha blending
    //   3. Launch preprocessBackwardCUDA: per-splat backprop through projection and R_cs
    static void backward(
        int   P, int R,
        const float* bg,
        int   width, int height,
        const float* means3D,
        const float* colors,
        const float* opacities,
        const float* scales,
        const float* rotations,
        const float* exps,
        const float* viewmatrix,
        float  tan_fovx,
        float  tan_fovy,
        const int*   radii,
        char*  geom_buffer,
        char*  binning_buffer,
        char*  image_buffer,
        const float* dL_dpix_color,   // (3,H,W)
        float* dL_dmeans3D,   // (P,3)
        float* dL_dcolors,    // (P,3)
        float* dL_dopacity,   // (P,1)
        float* dL_dscales,    // (P,3)
        float* dL_drotations, // (P,4)
        float* dL_dexps,      // (P,3)
        bool   debug = false);
};

} // namespace CudaRasterizer
