// forward.cu: Forward pass kernels for the superquadric rasterizer.
//
// Contains two CUDA kernels:
//   preprocessCUDA  — one thread per splat; projects to 2D, builds R_cs
//   renderCUDA      — one thread per pixel; alpha-blends splats front-to-back

#include "forward.h"
#include "auxiliary.h"
#include "config.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>


// ============================================================
//  preprocessCUDA
// ============================================================

// preprocessCUDA: Per-splat preprocessing kernel.
//
// Steps:
//   1. Depth cull — skip splats with z <= 0.01 (behind or at camera)
//   2. Project 3D camera-space centre to 2D pixel coordinates
//   3. Compute pixel-space bounding radius from scales and exps
//   4. Image-space frustum cull — skip if bounding circle misses the image entirely
//   5. Determine which tiles the splat overlaps via getRect
//   6. Build R_cs = R_sc.T (camera-to-shape) from precomposed R_sc and store
//   7. Write all outputs to global memory arrays

__global__ void preprocessCUDA(
    int P,
    const float* __restrict__ means3D,    // (P,3)
    const float* __restrict__ scales,     // (P,3)
    const float* __restrict__ rotations,  // (P,4)
    const float* __restrict__ exps,       // (P,3)
    const float* __restrict__ viewmatrix, // (4,4) camera-to-world
    int W, int H,
    float focal_x, float focal_y,
    float tan_fovx, float tan_fovy,
    float f_mean,   // (focal_x + focal_y) / 2
    float2* __restrict__ means2D_out,
    float*  __restrict__ depths_out,
    float*  __restrict__ R_cs_out,
    int*    __restrict__ radii_out,
    uint32_t* __restrict__ tiles_touched,
    dim3 grid)
{
    // One thread per splat using grid-stride loop
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    // Default: invisible
    radii_out[idx]       = 0;
    tiles_touched[idx]   = 0;

    // Step 1: depth cull — skip splats behind the camera
    float z = means3D[idx * 3 + 2];
    if (z <= 0.01f) return;

    float x = means3D[idx * 3 + 0];
    float y = means3D[idx * 3 + 1];

    // Step 2: project to pixel space
    float px = x / z * focal_x + (float)W * 0.5f;
    float py = y / z * focal_y + (float)H * 0.5f;
    means2D_out[idx] = { px, py };
    depths_out[idx]  = z;

    // Step 3: pixel-space bounding radius.
    //
    // Mirror diff-gaussian-rasterization: use 3 * scale_norm / z * f_mean.
    // This is analogous to GS's 3-sigma rule (radius = 3 * sqrt(max 2D eigenvalue))
    // applied to the magnitude of the 3D scale vector without any exponent factors.
    float s0 = scales[idx*3+0], s1 = scales[idx*3+1], s2 = scales[idx*3+2];
    float scale_norm = sqrtf(s0*s0 + s1*s1 + s2*s2);
    int radius = (int)ceilf(3.f * scale_norm / z * f_mean);
    // Cap at image diagonal to prevent degenerate radii from very large splats
    radius = max(1, min(radius, (int)sqrtf((float)(W*W + H*H))));

    // Step 4: frustum cull — mirror GS: reject only if the splat overlaps zero tiles.
    // getRect() clamps to the tile grid, so a splat entirely off-screen produces an
    // empty rectangle (rect_max == rect_min) and is discarded here.
    uint2 rect_min, rect_max;
    getRect({ px, py }, radius, rect_min, rect_max, grid);
    uint32_t n_tiles = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
    if (n_tiles == 0) return;

    radii_out[idx] = radius;

    // Step 5: record tile overlap count
    tiles_touched[idx] = n_tiles;

    // Step 6: build R_cs (camera→shape) = R_sc.T
    //
    //   rotations[] holds R_sc (shape→camera), precomposed in render2() as
    //   quadmultiply(camera_pose, gaussians_rot).  R_cs is its transpose.
    float R_sc[9], R_cs[9];
    quatToMatrix(&rotations[idx * 4], R_sc);

    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            R_cs[r*3+c] = R_sc[c*3+r];

    // Step 7: write R_cs to global memory (row-major, 9 floats per splat)
    for (int i = 0; i < 9; i++) {
        R_cs_out[idx * 9 + i] = R_cs[i];
    }
}


// ============================================================
//  renderCUDA
// ============================================================

// renderCUDA: Per-tile/pixel alpha-blending kernel.
//
// Each CUDA block handles one 16x16 pixel tile. Threads within the block
// cooperatively load splat data into shared memory in batches of BLOCK_SIZE.
//
// Steps:
//   1. Identify this pixel's tile and determine pixel coordinates
//   2. Load tile's sorted splat range from the ranges array
//   3. For each batch of BLOCK_SIZE splats in the tile:
//      a. Cooperative load: each thread fetches one splat's data into shared mem
//      b. For each splat in the shared batch:
//         i.   Compute camera-plane pixel offset from splat centre at splat depth
//         ii.  Find rim depth t* via Regula Falsi on h(t)=n̂(q0+t·v_s)·v_s=0;
//              evaluate distance at the corrected shape-space point q*=q0+t*·v_s
//         iii. Evaluate superquadricDistance in shape frame → F
//         iv.  Gaussian weight G = exp(-F)
//         v.   Compute alpha = min(0.99, opacity * G); skip if negligible
//         vi.  Accumulate colour and depth contributions; update transmittance T
//         vii. Record last contributor index for backward pass
//         viii.Early exit if T < 1e-4 (pixel fully saturated)
//   4. Write final colour + depth + auxiliary data to global memory

__global__ void renderCUDA(
    const uint2*    __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    float focal_x, float focal_y,
    float cx, float cy,             // image centre in pixels
    const float* __restrict__ bg,
    const float2* __restrict__ means2D,
    const float*  __restrict__ depths_arr,
    const float*  __restrict__ R_cs_arr,
    const float*  __restrict__ scales_arr,
    const float*  __restrict__ exps_arr,
    const float*  __restrict__ colors_arr,
    const float*  __restrict__ opacities_arr,
    float*    __restrict__ final_T,
    uint32_t* __restrict__ n_contrib,
    float*    __restrict__ out_color,
    float*    __restrict__ out_depth)
{
    // Step 1: identify tile and pixel
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int px = tile_x * BLOCK_X + threadIdx.x;
    int py = tile_y * BLOCK_Y + threadIdx.y;
    int pix_id = py * W + px;
    bool inside = (px < W && py < H);

    // Normalised image coordinates for this pixel (pixel centre)
    float xi = ((float)px + 0.5f - cx) / focal_x;
    float yi = ((float)py + 0.5f - cy) / focal_y;

    // Step 2: load this tile's range in the sorted list
    int tile_id = tile_y * ((W + BLOCK_X - 1) / BLOCK_X) + tile_x;
    uint2 range = ranges[tile_id];

    // Shared memory: one slot per thread in the block (BLOCK_SIZE splats per batch)
    __shared__ float2   s_means2D  [BLOCK_SIZE];
    __shared__ float    s_depth    [BLOCK_SIZE];
    __shared__ float    s_R_cs     [BLOCK_SIZE * 9];
    __shared__ float    s_scales   [BLOCK_SIZE * 3];
    __shared__ float    s_exps     [BLOCK_SIZE * 3];
    __shared__ float    s_colors   [BLOCK_SIZE * 3];
    __shared__ float    s_opacity  [BLOCK_SIZE];

    // Per-pixel accumulators
    float T = 1.0f;          // transmittance (starts fully transparent)
    float C[3] = {0,0,0};   // accumulated colour
    float D = 0.0f;          // accumulated depth
    uint32_t last_contributor = 0;
    // done=1 means this pixel is saturated; __syncthreads_and() exits when ALL pixels are done
    int done = (!inside) ? 1 : 0;

    // Step 3: iterate over sorted splat list in batches
    // __syncthreads_and(done): returns 1 if ALL threads have done=1 (CUDA 11.8 compatible)
    for (int base = range.x; base < (int)range.y && !__syncthreads_and(done); base += BLOCK_SIZE)
    {
        // Cooperative load — each thread fetches one splat
        int thread_rank = threadIdx.y * BLOCK_X + threadIdx.x;
        int fetch_idx   = base + thread_rank;
        bool valid_fetch = (fetch_idx < (int)range.y);
        uint32_t splat_idx = valid_fetch ? point_list[fetch_idx] : 0;

        // Sync before writing shared memory (replaces block.sync() from cooperative_groups)
        __syncthreads();

        // Step 3a: load all per-splat data into shared memory
        if (valid_fetch) {
            s_means2D[thread_rank] = means2D[splat_idx];
            s_depth  [thread_rank] = depths_arr[splat_idx];
            s_opacity[thread_rank] = opacities_arr[splat_idx];
            for (int c = 0; c < 3; c++) {
                s_colors[thread_rank*3+c] = colors_arr  [splat_idx*3+c];
                s_scales[thread_rank*3+c] = scales_arr  [splat_idx*3+c];
                s_exps  [thread_rank*3+c] = exps_arr    [splat_idx*3+c];
            }
            for (int k = 0; k < 9; k++)
                s_R_cs[thread_rank*9+k] = R_cs_arr[splat_idx*9+k];
        }
        __syncthreads();

        // Step 3b: each pixel-thread processes the loaded batch
        int batch_size = min(BLOCK_SIZE, (int)range.y - base);
        for (int j = 0; j < batch_size && !done; j++)
        {
            // Step 3b-i: camera-plane pixel offset from splat centre at splat depth
            float d  = s_depth[j];
            float mx = s_means2D[j].x;   // splat 2D pixel centre
            float my = s_means2D[j].y;
            // pixel 3D at splat depth: [xi*d, yi*d, d]
            // splat centre in camera: [mx/focal_x * d, my/focal_y * d, d]  (approx)
            // We already have means3D in camera frame, but here we reconstruct the 2D offset:
            //   dx = xi*d - (mx - cx)/focal_x*d = (xi - (mx-cx)/focal_x)*d
            // Simpler: dx = (pixel_cam_x - splat_cam_x) = xi*d - means3D_x
            // But we only have means2D (pixel space). Convert back:
            float splat_cam_x = (mx - cx) / focal_x;
            float splat_cam_y = (my - cy) / focal_y;
            float dx = (xi - splat_cam_x) * d;
            float dy = (yi - splat_cam_y) * d;
            // dz = 0 (depth-plane approximation)

            // Step 3b-ii: find rim depth t* and evaluate at the corrected 3D point.
            //
            //   v_s = R_cs[:,2]  (orthographic: camera z-axis in shape space)
            //   q0  = R_cs @ [dx, dy, 0]           depth-plane base point in shape space
            //   t*  = Regula Falsi root of h(t)=0  where h(t) = n̂(q0+t·v_s)·v_s
            //   q*  = q0 + t*·v_s                  rim surface point in shape space
            //
            // h(t) = 0 is the rim condition: the surface normal is perpendicular to
            // the viewing direction.  For a convex superquadric h is monotone along
            // any ray, so Regula Falsi always converges from the bracket [−2R, +2R].
            // Flat-face pixels (h never changes sign) return t*=0 (depth-plane fallback).
            const float* R  = &s_R_cs  [j * 9];
            const float* sc = &s_scales[j * 3];
            const float* ex = &s_exps  [j * 3];

#if RIM_ORTHOGRAPHIC
            // Orthographic: all pixels share the camera z-axis direction in shape space.
            // vs = R_cs[:,2]  (third column) — constant per splat, matches Python rim.
            float vs[3] = { R[2], R[5], R[8] };
#else
            // Perspective: each pixel uses its own ray direction.
            float vs[3] = { R[0]*xi + R[1]*yi + R[2],
                            R[3]*xi + R[4]*yi + R[5],
                            R[6]*xi + R[7]*yi + R[8] };
#endif
            float q0[3] = { R[0]*dx + R[1]*dy,
                            R[3]*dx + R[4]*dy,
                            R[6]*dx + R[7]*dy };

            float t_star = findRimDepth(q0, vs, sc, ex);
            float xs = q0[0] + vs[0] * t_star;
            float ys = q0[1] + vs[1] * t_star;
            float zs = q0[2] + vs[2] * t_star;

            // Step 3b-iii: evaluate superquadric distance
            float F = superquadricDistance(xs, ys, zs, sc, ex);

            // Step 3b-iv: Gaussian weight
            float G = expf(-F);

            // Step 3b-v: alpha; skip if negligible
            float alpha = fminf(0.99f, s_opacity[j] * G);
            if (alpha < 1.0f / 255.0f) continue;

            // Step 3b-vi: accumulate colour and depth
            float contrib = alpha * T;
            for (int c = 0; c < NUM_CHANNELS; c++)
                C[c] += s_colors[j*3+c] * contrib;
            D += d * contrib;

            // Step 3b-vii: update transmittance and record contributor
            T *= (1.0f - alpha);
            last_contributor = base + j;

            // Step 3b-viii: early exit when pixel is saturated
            if (T < 1e-4f) { done = 1; break; }
        }
    }

    // Step 4: write outputs
    if (!inside) return;

    // Add background contribution proportional to remaining transmittance
    for (int c = 0; c < NUM_CHANNELS; c++)
        out_color[c * H * W + pix_id] = C[c] + T * bg[c];

    out_depth[pix_id]  = D;
    final_T[pix_id]    = T;
    n_contrib[pix_id]  = last_contributor;
}


// ============================================================
//  Kernel launchers
// ============================================================

namespace FORWARD {

void preprocess(
    int P,
    const float* means3D, const float* scales,
    const float* rotations, const float* exps,
    const float* viewmatrix,
    int W, int H,
    float focal_x, float focal_y,
    float tan_fovx, float tan_fovy,
    float* means2D_out, float* depths_out, float* R_cs_out,
    int* radii_out, uint32_t* tiles_touched,
    const dim3 grid)
{
    float f_mean = (focal_x + focal_y) * 0.5f;
    int blocks = (P + 255) / 256;
    preprocessCUDA<<<blocks, 256>>>(
        P, means3D, scales, rotations, exps, viewmatrix,
        W, H, focal_x, focal_y, tan_fovx, tan_fovy, f_mean,
        (float2*)means2D_out, depths_out, R_cs_out,
        radii_out, tiles_touched, grid);
}

void render(
    const dim3 grid, dim3 block,
    const uint2* ranges, const uint32_t* point_list,
    int W, int H,
    float focal_x, float focal_y,
    const float* bg,
    const float2* means2D, const float* depths,
    const float* R_cs, const float* scales, const float* exps,
    const float* colors, const float* opacities,
    float* final_T, uint32_t* n_contrib,
    float* out_color, float* out_depth)
{
    float cx = (float)W * 0.5f;
    float cy = (float)H * 0.5f;
    renderCUDA<<<grid, block>>>(
        ranges, point_list, W, H, focal_x, focal_y, cx, cy,
        bg, means2D, depths, R_cs, scales, exps, colors, opacities,
        final_T, n_contrib, out_color, out_depth);
}

} // namespace FORWARD
