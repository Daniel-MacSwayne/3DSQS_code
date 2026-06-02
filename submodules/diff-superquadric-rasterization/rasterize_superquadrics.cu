// rasterize_superquadrics.cu: PyTorch tensor wrapper for the superquadric rasterizer.
//
// Converts PyTorch tensors to raw pointers, allocates output tensors,
// and delegates to CudaRasterizer::Rasterizer::forward/backward.
// Mirrors rasterize_points.cu in diff-gaussian-rasterization.

#include "rasterize_superquadrics.h"
#include "cuda_rasterizer/rasterizer.h"
#include <torch/extension.h>
#include <functional>

// resizeFunctional: Returns a lambda that resizes a torch::Tensor on demand.
//   Used to let CudaRasterizer allocate buffers without knowing their size upfront.
//
// Steps:
//   1. Capture a reference to the tensor
//   2. When called with size N, resize the tensor to N bytes and return its data pointer
static std::function<char*(size_t)> resizeFunctional(torch::Tensor& t)
{
    // Step 1-2: lambda captures tensor by reference
    return [&t](size_t N) -> char* {
        t.resize_({ (long long)N });
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
}


// RasterizeSuperquadricsCUDA: Forward pass — tensor interface.
//
// Steps:
//   1. Validate input shapes
//   2. Allocate output tensors (out_color, out_depth, out_radii)
//   3. Allocate opaque buffer tensors for saving intermediate state
//   4. Call CudaRasterizer::Rasterizer::forward via resizeFunctionals
//   5. Return (num_rendered, out_color, out_depth, out_radii, geomBuf, binBuf, imgBuf)
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
    bool debug)
{
    // Step 1: validate
    int P = means3D.size(0);
    TORCH_CHECK(means3D.size(1) == 3,   "means3D must be (P,3)");
    TORCH_CHECK(colors.size(1) == 3,    "colors must be (P,3)");
    TORCH_CHECK(scales.size(1) == 3,    "scales must be (P,3)");
    TORCH_CHECK(rotations.size(1) == 4, "rotations must be (P,4)");
    TORCH_CHECK(exps.size(1) == 3,      "exps must be (P,3)");

    // Step 2: allocate outputs
    auto float_opts = means3D.options().dtype(torch::kFloat32);
    auto int_opts   = means3D.options().dtype(torch::kInt32);
    torch::Tensor out_color = torch::zeros({3, image_height, image_width}, float_opts);
    torch::Tensor out_depth = torch::zeros({image_height, image_width},     float_opts);
    torch::Tensor out_radii = torch::zeros({P},                              int_opts);

    // Step 3: allocate buffer tensors (will be resized by resizeFunctional)
    torch::Tensor geomBuffer   = torch::empty({0}, float_opts.dtype(torch::kByte));
    torch::Tensor binningBuffer= torch::empty({0}, float_opts.dtype(torch::kByte));
    torch::Tensor imgBuffer    = torch::empty({0}, float_opts.dtype(torch::kByte));

    // Step 4: call forward
    int num_rendered = CudaRasterizer::Rasterizer::forward(
        resizeFunctional(geomBuffer),
        resizeFunctional(binningBuffer),
        resizeFunctional(imgBuffer),
        P,
        background.contiguous().data_ptr<float>(),
        image_width, image_height,
        means3D.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        rotations.contiguous().data_ptr<float>(),
        exps.contiguous().data_ptr<float>(),
        viewmatrix.contiguous().data_ptr<float>(),
        tan_fovx, tan_fovy,
        out_color.data_ptr<float>(),
        out_depth.data_ptr<float>(),
        out_radii.data_ptr<int>(),
        debug);

    // Step 5: return
    return { num_rendered, out_color, out_depth, out_radii,
             geomBuffer, binningBuffer, imgBuffer };
}


// RasterizeSuperquadricsBackwardCUDA: Backward pass — tensor interface.
//
// Steps:
//   1. Allocate gradient output tensors (all zeros)
//   2. Call CudaRasterizer::Rasterizer::backward via raw pointers
//   3. Return gradient tensors

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
    bool debug)
{
    int P = means3D.size(0);
    int H = dL_dpix_color.size(1);
    int W = dL_dpix_color.size(2);

    // Step 1: allocate gradient tensors
    auto float_opts = means3D.options().dtype(torch::kFloat32);
    torch::Tensor dL_dmeans3D   = torch::zeros({P, 3}, float_opts);
    torch::Tensor dL_dcolors    = torch::zeros({P, 3}, float_opts);
    torch::Tensor dL_dopacity   = torch::zeros({P, 1}, float_opts);
    torch::Tensor dL_dscales    = torch::zeros({P, 3}, float_opts);
    torch::Tensor dL_drotations = torch::zeros({P, 4}, float_opts);
    torch::Tensor dL_dexps      = torch::zeros({P, 3}, float_opts);

    // Step 2: call backward
    CudaRasterizer::Rasterizer::backward(
        P, R,
        background.contiguous().data_ptr<float>(),
        W, H,
        means3D.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        rotations.contiguous().data_ptr<float>(),
        exps.contiguous().data_ptr<float>(),
        viewmatrix.contiguous().data_ptr<float>(),
        tan_fovx, tan_fovy,
        radii.contiguous().data_ptr<int>(),
        reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(imgBuffer.contiguous().data_ptr()),
        dL_dpix_color.contiguous().data_ptr<float>(),
        dL_dmeans3D.data_ptr<float>(),
        dL_dcolors.data_ptr<float>(),
        dL_dopacity.data_ptr<float>(),
        dL_dscales.data_ptr<float>(),
        dL_drotations.data_ptr<float>(),
        dL_dexps.data_ptr<float>(),
        debug);

    // Step 3: return
    return { dL_dmeans3D, dL_dcolors, dL_dopacity,
             dL_dscales, dL_drotations, dL_dexps };
}
