// ext.cpp: pybind11 bindings exposing the superquadric rasterizer to Python.
//
// Exports two functions into the _C module:
//   rasterize_superquadrics          → forward pass
//   rasterize_superquadrics_backward → backward pass
//
// Steps:
//   1. Include PyTorch extension headers and the C++ interface
//   2. Define the PYBIND11_MODULE block registering both functions

// Step 1: includes
#include <torch/extension.h>
#include "rasterize_superquadrics.h"

// Step 2: module registration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // Forward pass: rasterize visible superquadric splats into colour + depth
    m.def("rasterize_superquadrics",
          &RasterizeSuperquadricsCUDA,
          "Superquadric rasterizer forward pass (CUDA)");

    // Backward pass: compute gradients w.r.t. all splat parameters
    m.def("rasterize_superquadrics_backward",
          &RasterizeSuperquadricsBackwardCUDA,
          "Superquadric rasterizer backward pass (CUDA)");
}
