// rasterize_superquadrics.h: PyTorch-facing C++ interface for the superquadric rasterizer.
//
// These functions are called from rasterize_superquadrics.cu and exposed via ext.cpp.
// They convert PyTorch tensors to raw pointers and call CudaRasterizer::Rasterizer.

#pragma once

#include <torch/extension.h>
#include <tuple>

// RasterizeSuperquadricsCUDA: Forward pass.
//
// Inputs (all on CUDA, float32):
//   background    (3,)
//   means3D       (P, 3)  splat centres in camera frame
//   colors        (P, 3)  pre-computed RGB
//   opacities     (P, 1)
//   scales        (P, 3)
//   rotations     (P, 4)  quaternion [q1,q2,q3,q4]
//   exps          (P, 3)  [e1, e2, e3]
//   viewmatrix    (4, 4)  world_view_transform (camera-to-world, stored transposed)
//   tan_fovx, tan_fovy  floats
//   image_height, image_width  ints
//   debug         bool
//
// Returns tuple of 6:
//   (num_rendered, out_color (3,H,W), out_depth (H,W), out_radii (P,),
//    geomBuffer, binningBuffer, imgBuffer)
//   The last three are opaque byte tensors saved for the backward pass.
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeSuperquadricsCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& exps,
    const torch::Tensor& viewmatrix,
    float tan_fovx, float tan_fovy,
    int image_height, int image_width,
    bool debug);


// RasterizeSuperquadricsBackwardCUDA: Backward pass.
//
// Inputs: all forward inputs + saved buffers + dL_dpix_color (3,H,W)
//
// Returns tuple of 6 gradient tensors:
//   (dL_dmeans3D (P,3), dL_dcolors (P,3), dL_dopacity (P,1),
//    dL_dscales (P,3), dL_drotations (P,4), dL_dexps (P,3))
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeSuperquadricsBackwardCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& radii,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& exps,
    const torch::Tensor& viewmatrix,
    float tan_fovx, float tan_fovy,
    const torch::Tensor& dL_dpix_color,
    int R,
    const torch::Tensor& geomBuffer,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imgBuffer,
    bool debug);
