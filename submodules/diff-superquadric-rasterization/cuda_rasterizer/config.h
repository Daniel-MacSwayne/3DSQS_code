// config.h: Compile-time constants for the superquadric rasterizer.
// Mirror of the constants used in diff-gaussian-rasterization.
// Tile dimensions match the hardware warp structure (256 threads per block).

#pragma once

// Number of output image channels (RGB)
#define NUM_CHANNELS 3

// Tile width in pixels — one CUDA block handles one 16x16 tile
#define BLOCK_X 16
#define BLOCK_Y 16

// Threads per block = BLOCK_X * BLOCK_Y = 256
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)

// Number of CUDA warps per block (32 threads per warp)
#define NUM_WARPS (BLOCK_SIZE / 32)

// RIM_ORTHOGRAPHIC: controls the viewing direction used for rim-depth estimation.
//
//   1 (orthographic): vs = R_cs[:,2]  — all pixels share the same camera z-axis
//     direction.  Matches the Python rim formula exactly; produces correct
//     silhouette shapes (e.g. hexagonal shadow for a cube on its space diagonal)
//     regardless of splat distance or field of view.
//
//   0 (perspective): vs = R_cs @ [xi, yi, 1]  — each pixel uses its own ray
//     direction.  Physically more accurate for large splats or wide FOV, but
//     diverges from the Python rim approach and can distort sharp shapes.
//
// Recommendation: use 1 (orthographic) to match the Python renderer.
#define RIM_ORTHOGRAPHIC 1
