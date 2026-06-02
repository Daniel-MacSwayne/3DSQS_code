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
