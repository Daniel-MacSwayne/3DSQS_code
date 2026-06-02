// rasterizer_impl.h: Buffer state structs for the superquadric rasterizer.
//
// Three buffer types mirror diff-gaussian-rasterization:
//   GeometryState  — per-splat preprocessing results (projection, R_cs, depth, radius)
//   ImageState     — per-pixel auxiliary data (tile ranges, accumulated alpha)
//   BinningState   — sorted splat-tile instance list
//
// Each struct uses the fromChunk() pattern: memory is allocated in a single contiguous
// block and sub-arrays are sliced out with pointer arithmetic and 128-byte alignment.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <functional>

namespace CudaRasterizer {

// obtain128ByteAlignedPointer: advance a char* pointer to the next 128-byte boundary
// and return a typed pointer to that location.
//
// Steps:
//   1. Cast chunk to size_t for arithmetic
//   2. Round up to next 128-byte boundary
//   3. Advance chunk past the allocated region
//   4. Return typed pointer to the aligned start
template <typename T>
inline T* obtain(char*& chunk, size_t count, size_t alignment = 128)
{
    // Step 1-2: align
    size_t addr = reinterpret_cast<size_t>(chunk);
    size_t aligned = (addr + alignment - 1) & ~(alignment - 1);
    // Step 3: advance chunk
    chunk = reinterpret_cast<char*>(aligned) + count * sizeof(T);
    // Step 4: return typed pointer
    return reinterpret_cast<T*>(aligned);
}


// ============================================================
//  GeometryState — per-splat preprocessing results
// ============================================================
struct GeometryState {
    float2*   means2D;      // (P,) projected 2D pixel centres
    float*    depths;       // (P,) view-space depth (z in camera frame)
    float*    R_cs;         // (P,9) shape-to-camera rotation matrix (row-major 3x3)
    int*      radii;        // (P,) pixel-space bounding radius
    uint32_t* tiles_touched; // (P,) number of tiles this splat overlaps
    uint32_t* point_offsets; // (P,) prefix-sum of tiles_touched (for duplicateWithKeys)

    // CUB DeviceScan workspace
    size_t    scan_size;
    char*     scanning_space;

    // fromChunk: slice all fields out of a pre-allocated contiguous buffer.
    //
    // Steps:
    //   1. Slice each field in order, advancing the chunk pointer
    //   2. Allocate CUB workspace (size queried separately, passed as scan_size)
    static GeometryState fromChunk(char*& chunk, size_t P)
    {
        GeometryState gs;
        // Step 1: slice fields
        gs.means2D       = obtain<float2>  (chunk, P);
        gs.depths        = obtain<float>   (chunk, P);
        gs.R_cs          = obtain<float>   (chunk, P * 9);
        gs.radii         = obtain<int>     (chunk, P);
        gs.tiles_touched = obtain<uint32_t>(chunk, P);
        gs.point_offsets = obtain<uint32_t>(chunk, P);
        // Step 2: CUB workspace (filled later with actual size)
        gs.scan_size     = 0;
        gs.scanning_space = chunk;
        return gs;
    }
};


// ============================================================
//  ImageState — per-pixel auxiliary data
// ============================================================
struct ImageState {
    uint2*    ranges;       // (N_tiles,) [start, end] index in sorted point_list per tile
    uint32_t* n_contrib;    // (H*W,) index of the last contributing splat per pixel
    float*    accum_alpha;  // (H*W,) accumulated (1-alpha) product per pixel

    // fromChunk: slice fields for an image of size W x H with n_tiles tiles.
    //
    // Steps:
    //   1. Slice ranges (one uint2 per tile)
    //   2. Slice n_contrib and accum_alpha (one entry per pixel)
    static ImageState fromChunk(char*& chunk, size_t n_tiles, size_t W, size_t H)
    {
        ImageState is;
        // Step 1
        is.ranges      = obtain<uint2>   (chunk, n_tiles);
        // Step 2
        is.n_contrib   = obtain<uint32_t>(chunk, W * H);
        is.accum_alpha = obtain<float>   (chunk, W * H);
        return is;
    }
};


// ============================================================
//  BinningState — sorted splat-tile instance list
// ============================================================
struct BinningState {
    uint64_t* point_list_keys_unsorted;  // (R,) tile_id<<32|depth before sort
    uint64_t* point_list_keys;           // (R,) after sort
    uint32_t* point_list_unsorted;       // (R,) splat indices before sort
    uint32_t* point_list;               // (R,) splat indices after sort

    // CUB DeviceRadixSort workspace
    size_t    sorting_size;
    char*     list_sorting_space;

    // fromChunk: slice fields for R total splat-tile instances.
    //
    // Steps:
    //   1. Slice the four key/value arrays (unsorted and sorted)
    //   2. Reserve CUB sort workspace (filled later)
    static BinningState fromChunk(char*& chunk, size_t R)
    {
        BinningState bs;
        // Step 1
        bs.point_list_keys_unsorted = obtain<uint64_t>(chunk, R);
        bs.point_list_keys          = obtain<uint64_t>(chunk, R);
        bs.point_list_unsorted      = obtain<uint32_t>(chunk, R);
        bs.point_list               = obtain<uint32_t>(chunk, R);
        // Step 2
        bs.sorting_size             = 0;
        bs.list_sorting_space       = chunk;
        return bs;
    }
};

} // namespace CudaRasterizer
