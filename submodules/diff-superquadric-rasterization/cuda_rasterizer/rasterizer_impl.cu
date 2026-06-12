// rasterizer_impl.cu: Orchestration for the superquadric rasterizer.
//
// Implements CudaRasterizer::Rasterizer::forward() and ::backward().
// Coordinates buffer allocation, kernel launches, and CUB sort operations.
// Mirrors the structure of rasterizer_impl.cu in diff-gaussian-rasterization.

#include "rasterizer.h"
#include "rasterizer_impl.h"
#include "forward.h"
#include "backward.h"
#include "auxiliary.h"
#include "config.h"

// Include only the CUB primitives we use (avoids histogram template errors in CUDA 11.8)
#include <cub/device/device_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>
#include <functional>

namespace CudaRasterizer {


// ============================================================
//  Helper kernels
// ============================================================

// duplicateWithKeys: For each splat, write one entry per overlapping tile into
//   the keys/values arrays. The key encodes (tile_id << 32 | depth_bits) so
//   that a radix sort on keys produces front-to-back ordering within each tile.
//
// Steps:
//   1. Determine which tiles this splat overlaps (same rect as preprocess)
//   2. For each overlapping tile, write key = (tile_id<<32 | float_to_uint(depth))
//      and value = splat index into the unsorted arrays at the splat's offset
__global__ void duplicateWithKeys(
    int P,
    const float2*    __restrict__ means2D,
    const float*     __restrict__ depths,
    const uint32_t*  __restrict__ point_offsets,
    const int*       __restrict__ radii,
    uint64_t*        __restrict__ keys_unsorted,
    uint32_t*        __restrict__ values_unsorted,
    int W, int H,
    dim3 grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;
    if (radii[idx] == 0) return;

    // Step 1: recompute tile overlap rectangle
    uint2 rect_min, rect_max;
    getRect(means2D[idx], radii[idx], rect_min, rect_max, grid);

    // Depth encoded as uint32 (preserves sort order for positive depths)
    uint32_t depth_bits;
    float d = depths[idx];
    memcpy(&depth_bits, &d, sizeof(float));

    // Step 2: write one entry per overlapping tile.
    // point_offsets is an INCLUSIVE prefix sum: point_offsets[i] = sum(tiles_touched[0..i]).
    // The exclusive start for splat idx is therefore point_offsets[idx-1] (with 0 for idx=0).
    // This mirrors the pattern in diff-gaussian-rasterization/rasterizer_impl.cu.
    uint32_t base = (idx == 0) ? 0 : point_offsets[idx - 1];
    for (uint32_t ty = rect_min.y; ty < rect_max.y; ty++) {
        for (uint32_t tx = rect_min.x; tx < rect_max.x; tx++) {
            uint64_t tile_id = (uint64_t)(ty * grid.x + tx);
            uint64_t key = (tile_id << 32) | (uint64_t)depth_bits;
            keys_unsorted  [base] = key;
            values_unsorted[base] = (uint32_t)idx;
            base++;
        }
    }
}


// identifyTileRanges: After sorting by key, scan the sorted key array to find
//   the [start, end) range of each tile in the sorted list.
//
// Steps:
//   1. Compare key[i] with key[i-1] to detect tile boundaries
//   2. Write start index when a new tile begins, end index when a tile ends
__global__ void identifyTileRanges(
    int R,
    int n_tiles,                            // guard against out-of-bounds tile IDs
    const uint64_t* __restrict__ keys,
    uint2*          __restrict__ ranges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R) return;

    uint32_t tile_curr = (uint32_t)(keys[idx] >> 32);

    // Safety guard: if the CUB sort produced an out-of-range tile ID
    // (can happen if the sort workspace overlapped another buffer), skip silently.
    if ((int)tile_curr >= n_tiles) return;

    // Step 1-2: boundary detection
    if (idx == 0) {
        ranges[tile_curr].x = 0;
    } else {
        uint32_t tile_prev = (uint32_t)(keys[idx-1] >> 32);
        if (tile_curr != tile_prev) {
            if ((int)tile_prev < n_tiles) ranges[tile_prev].y = idx;
            ranges[tile_curr].x = idx;
        }
    }
    if (idx == R - 1) {
        ranges[tile_curr].y = R;
    }
}


// ============================================================
//  Rasterizer::forward
// ============================================================

// forward: Full forward pass — preprocess, sort, render.
//
// Steps:
//   1. Compute focal lengths from FOV tangents and set up tile grid
//   2. Query buffer sizes and allocate geometry/image buffers
//   3. Launch preprocessCUDA (per-splat)
//   4. Prefix-sum tiles_touched → point_offsets to get per-splat write positions
//   5. Copy total R (num rendered instances) to host
//   6. Allocate binning buffer of size R
//   7. Launch duplicateWithKeys: expand splat list to one entry per tile overlap
//   8. CUB radix sort on (tile_id | depth) keys
//   9. Memset tile ranges to 0; launch identifyTileRanges
//  10. Launch renderCUDA (per-tile/pixel)

int Rasterizer::forward(
    std::function<char*(size_t)> geometryBuffer,
    std::function<char*(size_t)> binningBuffer,
    std::function<char*(size_t)> imageBuffer,
    int P,
    const float* bg,
    int W, int H,
    const float* means3D,
    const float* colors,
    const float* opacities,
    const float* scales,
    const float* rotations,
    const float* exps,
    const float* viewmatrix,
    float tan_fovx, float tan_fovy,
    float* out_color,
    float* out_depth,
    int*   out_radii,
    bool   debug)
{
    if (P == 0) return 0;

    // Ensure any prior GPU work (e.g., backward pass kernels) has fully completed
    // before we allocate and use new buffers.  cudaDeviceSynchronize() is cheap
    // relative to the render cost and prevents stale writes corrupting reused memory.
    cudaDeviceSynchronize();

    // Step 1: focal lengths and tile grid
    float focal_x = (float)W / (2.0f * tan_fovx);
    float focal_y = (float)H / (2.0f * tan_fovy);
    dim3 tile_grid((W + BLOCK_X-1)/BLOCK_X, (H + BLOCK_Y-1)/BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);
    int n_tiles = tile_grid.x * tile_grid.y;

    // Step 2: allocate geometry buffer
    //   Query CUB scan size first
    size_t scan_size = 0;
    cub::DeviceScan::InclusiveSum(nullptr, scan_size,
        (uint32_t*)nullptr, (uint32_t*)nullptr, P);
    // geometry buffer size = sum of field sizes + alignment padding + scan workspace
    size_t geom_size =
        P * sizeof(float2)  +   // means2D
        P * sizeof(float)   +   // depths
        P * 9 * sizeof(float)+  // R_cs
        P * sizeof(int)     +   // radii
        P * sizeof(uint32_t)+   // tiles_touched
        P * sizeof(uint32_t)+   // point_offsets
        scan_size + 128 * 8;    // CUB workspace + alignment padding
    char* geom_chunk = geometryBuffer(geom_size);
    GeometryState gs = GeometryState::fromChunk(geom_chunk, P);

    // Allocate image buffer
    size_t img_size =
        n_tiles * sizeof(uint2)   +
        W * H * sizeof(uint32_t)  +
        W * H * sizeof(float)     +
        128 * 4;
    char* img_chunk = imageBuffer(img_size);
    ImageState is = ImageState::fromChunk(img_chunk, n_tiles, W, H);

    // Step 3: launch preprocess
    FORWARD::preprocess(
        P, means3D, scales, rotations, exps, viewmatrix,
        W, H, focal_x, focal_y, tan_fovx, tan_fovy,
        (float*)gs.means2D, gs.depths, gs.R_cs, gs.radii, gs.tiles_touched,
        tile_grid);
    CHECK_CUDA(, debug);

    // Copy radii to output
    cudaMemcpy(out_radii, gs.radii, P * sizeof(int), cudaMemcpyDeviceToDevice);

    // Step 4: prefix sum of tiles_touched → point_offsets
    // Use gs.scanning_space — the pointer left by fromChunk after all field allocations.
    // This is the correct CUB temp storage location inside the geometry buffer.
    gs.scan_size = scan_size;
    cub::DeviceScan::InclusiveSum(
        gs.scanning_space,
        scan_size,
        gs.tiles_touched, gs.point_offsets, P);
    CHECK_CUDA(, debug);

    // Step 5: read total number of rendered instances R
    int R = 0;
    cudaMemcpy(&R, gs.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);
    if (R == 0) return 0;

    // Step 6: allocate binning buffer of size R
    size_t sort_size = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, sort_size,
        (uint64_t*)nullptr, (uint64_t*)nullptr,
        (uint32_t*)nullptr, (uint32_t*)nullptr, R);
    size_t bin_size =
        R * sizeof(uint64_t) * 2 +
        R * sizeof(uint32_t) * 2 +
        sort_size + 128 * 6;
    char* bin_chunk = binningBuffer(bin_size);
    BinningState bs = BinningState::fromChunk(bin_chunk, R);

    // Step 7: expand splat list — one entry per tile overlap
    duplicateWithKeys<<<(P+255)/256, 256>>>(
        P, gs.means2D, gs.depths, gs.point_offsets, gs.radii,
        bs.point_list_keys_unsorted, bs.point_list_unsorted, W, H, tile_grid);
    CHECK_CUDA(, debug);

    // Step 8: radix sort on keys (upper 32 bits = tile_id, lower 32 bits = depth)
    // Use bs.list_sorting_space — the pointer left by fromChunk after all field allocations.
    bs.sorting_size = sort_size;
    int key_bits = 32 + (int)ceil(log2((double)(n_tiles + 1)));
    key_bits = min(key_bits, 64);
    cub::DeviceRadixSort::SortPairs(
        bs.list_sorting_space, sort_size,
        bs.point_list_keys_unsorted, bs.point_list_keys,
        bs.point_list_unsorted,      bs.point_list, R, 0, key_bits);
    CHECK_CUDA(, debug);

    // Step 9: clear ranges and identify per-tile ranges
    cudaMemset(is.ranges, 0, n_tiles * sizeof(uint2));
    if (R > 0)
        identifyTileRanges<<<(R+255)/256, 256>>>(R, n_tiles, bs.point_list_keys, is.ranges);
    CHECK_CUDA(, debug);

    // Step 10: render
    FORWARD::render(
        tile_grid, block,
        is.ranges, bs.point_list,
        W, H, focal_x, focal_y, bg,
        gs.means2D, gs.depths, gs.R_cs, scales, exps, colors, opacities,
        is.accum_alpha, is.n_contrib,
        out_color, out_depth);
    CHECK_CUDA(, debug);

    return R;
}


// ============================================================
//  Rasterizer::backward
// ============================================================

// backward: Full backward pass.
//
// Steps:
//   1. Reconstruct focal lengths and buffer structs from saved buffers
//   2. Launch renderBackwardCUDA (per-tile/pixel, reverse order)
//   3. Launch preprocessBackwardCUDA (per-splat, R_cs → quaternion)

void Rasterizer::backward(
    int P, int R,
    const float* bg,
    int W, int H,
    const float* means3D,
    const float* colors,
    const float* opacities,
    const float* scales,
    const float* rotations,
    const float* exps,
    const float* viewmatrix,
    float tan_fovx, float tan_fovy,
    const int*   radii,
    char*  geom_buffer,
    char*  binning_buffer,
    char*  image_buffer,
    const float* dL_dpix_color,
    float* dL_dmeans3D,
    float* dL_dcolors,
    float* dL_dopacity,
    float* dL_dscales,
    float* dL_drotations,
    float* dL_dexps,
    bool   debug)
{
    if (P == 0 || R == 0) return;

    // Step 1: reconstruct geometry/image/binning state from saved buffers
    float focal_x = (float)W / (2.0f * tan_fovx);
    float focal_y = (float)H / (2.0f * tan_fovy);
    dim3 tile_grid((W + BLOCK_X-1)/BLOCK_X, (H + BLOCK_Y-1)/BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);
    int n_tiles = tile_grid.x * tile_grid.y;

    GeometryState gs = GeometryState::fromChunk(geom_buffer, P);
    BinningState  bs = BinningState::fromChunk(binning_buffer, R);
    ImageState    is = ImageState::fromChunk(image_buffer, n_tiles, W, H);

    // Allocate intermediate gradient buffers.
    // We synchronize before freeing these to ensure all kernels that write to them
    // have completed — cudaFree does not wait for in-flight kernels on all CUDA versions.
    float* dL_dR_cs;
    cudaMalloc(&dL_dR_cs, P * 9 * sizeof(float));
    cudaMemset(dL_dR_cs, 0, P * 9 * sizeof(float));

    float* dL_dmeans2D;
    cudaMalloc(&dL_dmeans2D, P * 2 * sizeof(float));
    cudaMemset(dL_dmeans2D, 0, P * 2 * sizeof(float));

    // Step 2: render backward
    BACKWARD::render(
        tile_grid, block,
        is.ranges, bs.point_list,
        W, H, focal_x, focal_y, bg,
        gs.means2D, gs.depths, gs.R_cs, scales, exps, colors, opacities,
        is.accum_alpha, is.n_contrib,
        dL_dpix_color,
        dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dR_cs,
        dL_dscales, dL_dexps);
    CHECK_CUDA(, debug);

    // Step 3: preprocess backward — R_cs → quaternion, means2D → means3D
    BACKWARD::preprocess(
        P, means3D, rotations, viewmatrix,
        gs.radii, focal_x, focal_y,
        dL_dR_cs, dL_dmeans2D,
        dL_drotations, dL_dmeans3D);
    CHECK_CUDA(, debug);

    // Synchronise before freeing: ensure all backward kernels have finished writing
    // to dL_dR_cs and dL_dmeans2D before releasing the memory.
    cudaDeviceSynchronize();
    cudaFree(dL_dR_cs);
    cudaFree(dL_dmeans2D);
}

} // namespace CudaRasterizer
