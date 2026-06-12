// backward.cu: Backward pass kernels for the superquadric rasterizer.
//
// Contains:
//   renderBackwardCUDA  — per-pixel backprop through alpha blending and superquadric eval
//   preprocessBackwardCUDA — per-splat backprop from R_cs to quaternion

#include "backward.h"
#include "auxiliary.h"
#include "config.h"
#include <cuda_runtime.h>
#include <cstdio>


// ============================================================
//  renderBackwardCUDA
// ============================================================

// renderBackwardCUDA: Per-tile/pixel backward kernel.
//
// Reconstructs the forward pass in REVERSE splat order (back-to-front) to
// correctly derive gradients through the transmittance product.
//
// Steps:
//   1. Identify pixel and load its forward-pass state (final_T, n_contrib)
//   2. Iterate splats from last contributor back to first, in REVERSE
//   3. For each splat:
//      a. Cooperative load of splat data into shared memory (same as forward)
//      b. Recompute camera-plane offset; find rim depth t*; compute q*, F, G, alpha
//      c. Reconstruct transmittance T at this splat from final_T and alphas (skip trick)
//      d. Backprop through colour accumulation: dL/d(color), dL/d(alpha)
//      e. Backprop through alpha = opacity * G: dL/d(opacity), dL/d(G)
//      f. Backprop through G = exp(-F): dL/d(F)
//      g. Backprop through superquadricDistance: dL/d(xs,ys,zs), dL/d(s), dL/d(e)
//      h. Backprop through rotation: dL/d(R_cs) via chain rule using effective offsets
//         edx=dx+t*xi, edy=dy+t*yi, edz=t* (t* treated as constant)
//      i. Atomically accumulate all gradients to global memory

__global__ void renderBackwardCUDA(
    const uint2*    __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    float focal_x, float focal_y, float cx, float cy,
    const float* __restrict__ bg,
    const float2* __restrict__ means2D,
    const float*  __restrict__ depths_arr,
    const float*  __restrict__ R_cs_arr,
    const float*  __restrict__ scales_arr,
    const float*  __restrict__ exps_arr,
    const float*  __restrict__ colors_arr,
    const float*  __restrict__ opacities_arr,
    const float*    __restrict__ final_T,
    const uint32_t* __restrict__ n_contrib,
    const float*  __restrict__ dL_dpix_color,
    // Gradient accumulators
    float* __restrict__ dL_dmeans2D,
    float* __restrict__ dL_dcolors,
    float* __restrict__ dL_dopacity,
    float* __restrict__ dL_dR_cs,
    float* __restrict__ dL_dscales,
    float* __restrict__ dL_dexps)
{
    // Step 1: identify pixel and early-exit if outside image
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int px = tile_x * BLOCK_X + threadIdx.x;
    int py = tile_y * BLOCK_Y + threadIdx.y;
    int pix_id = py * W + px;
    bool inside = (px < W && py < H);

    float xi = ((float)px + 0.5f - cx) / focal_x;
    float yi = ((float)py + 0.5f - cy) / focal_y;

    // Load tile range
    int tile_id = tile_y * ((W + BLOCK_X - 1) / BLOCK_X) + tile_x;
    uint2 range = ranges[tile_id];

    // Load per-pixel forward state
    float T_final = inside ? final_T[pix_id] : 1.0f;
    uint32_t last = inside ? n_contrib[pix_id] : 0;

    // Upstream gradient from loss w.r.t. this pixel's colour channels
    float dL_dpix[3] = {0,0,0};
    if (inside) {
        for (int c = 0; c < NUM_CHANNELS; c++)
            dL_dpix[c] = dL_dpix_color[c * H * W + pix_id];
    }

    // C_remaining[c]: remaining colour contribution after the current splat
    // (contributions from deeper splats + background).
    // Initialise to the background contribution: bg[c] * T_final.
    // Updated after each splat: C_remaining[c] += color_j[c] * alpha_j * T_before_j
    // This is used to compute the full transmittance-aware gradient:
    //   dL/d(alpha_j) = sum_c [(T_before * c_j - C_remaining / (1 - alpha)) * dL_dpix[c]]
    float C_remaining[3] = {
        bg[0] * T_final,
        bg[1] * T_final,
        bg[2] * T_final
    };

    // Shared memory (same layout as forward)
    __shared__ float2  s_means2D  [BLOCK_SIZE];
    __shared__ float   s_depth    [BLOCK_SIZE];
    __shared__ float   s_R_cs     [BLOCK_SIZE * 9];
    __shared__ float   s_scales   [BLOCK_SIZE * 3];
    __shared__ float   s_exps     [BLOCK_SIZE * 3];
    __shared__ float   s_colors   [BLOCK_SIZE * 3];
    __shared__ float   s_opacity  [BLOCK_SIZE];
    __shared__ uint32_t s_indices [BLOCK_SIZE];

    // Running T reconstruction: start from T_final and work backwards
    float T_running = T_final;

    // Step 2: iterate batches in reverse.
    // The forward loop starts at range.x and increments by BLOCK_SIZE.
    // The last batch starts at: range.x + ((range.y-1 - range.x) / BLOCK_SIZE) * BLOCK_SIZE
    // We must compute the offset relative to range.x, not as an absolute index division.
    int done = (!inside) ? 1 : 0;
    int rel_last = ((int)range.y - 1 - (int)range.x);  // relative index of last element
    int last_batch_base = (int)range.x + (rel_last >= 0 ? (rel_last / BLOCK_SIZE) * BLOCK_SIZE : 0);
    for (int base = last_batch_base;
             base >= (int)range.x;
             base -= BLOCK_SIZE)
    {
        int thread_rank = threadIdx.y * BLOCK_X + threadIdx.x;
        int fetch_idx   = base + thread_rank;
        bool valid_fetch = (fetch_idx < (int)range.y && fetch_idx >= (int)range.x);
        uint32_t splat_idx = valid_fetch ? point_list[fetch_idx] : 0;

        // Step 3a: sync before writing shared memory
        __syncthreads();

        // Load splat data
        if (valid_fetch) {
            s_indices[thread_rank] = splat_idx;
            s_means2D[thread_rank] = means2D[splat_idx];
            s_depth  [thread_rank] = depths_arr[splat_idx];
            s_opacity[thread_rank] = opacities_arr[splat_idx];
            for (int c = 0; c < 3; c++) {
                s_colors[thread_rank*3+c] = colors_arr[splat_idx*3+c];
                s_scales[thread_rank*3+c] = scales_arr[splat_idx*3+c];
                s_exps  [thread_rank*3+c] = exps_arr  [splat_idx*3+c];
            }
            for (int k = 0; k < 9; k++)
                s_R_cs[thread_rank*9+k] = R_cs_arr[splat_idx*9+k];
        }
        __syncthreads();

        // Step 3b-i: process batch in REVERSE order within the batch
        int batch_size = min(BLOCK_SIZE, (int)range.y - base);
        for (int j = batch_size - 1; j >= 0 && !done; j--)
        {
            int global_j = base + j;
            // Only process up to last contributor for this pixel
            if (global_j > (int)last) continue;

            // Step 3b: recompute forward quantities
            float d  = s_depth[j];
            float mx = s_means2D[j].x;
            float my = s_means2D[j].y;

            float splat_cam_x = (mx - cx) / focal_x;
            float splat_cam_y = (my - cy) / focal_y;
            float dx = (xi - splat_cam_x) * d;
            float dy = (yi - splat_cam_y) * d;

            const float* R   = &s_R_cs   [j*9];
            const float* sc  = &s_scales  [j*3];
            const float* ex  = &s_exps    [j*3];

            // Recompute rim depth (same as forward — t* not stored, re-derived here)
#if RIM_ORTHOGRAPHIC
            float vs[3] = { R[2], R[5], R[8] };
#else
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

            float F     = superquadricDistance(xs, ys, zs, sc, ex);
            float G     = expf(-F);
            float raw_alpha = s_opacity[j] * G;
            float alpha = fminf(0.99f, raw_alpha);

            if (alpha < 1.0f / 255.0f) continue;

            // Step 3c: reconstruct T_before_this_splat
            // T_running currently is T_after_this_splat = T_before * (1-alpha)
            // → T_before = T_running / (1-alpha)
            float one_minus_alpha = 1.0f - alpha;
            float T_before = (one_minus_alpha > 1e-6f) ? T_running / one_minus_alpha : 0.0f;

            // Step 3d: backprop through colour accumulation.
            //
            // Full gradient (3DGS paper formulation):
            //   dC/d(alpha_j) = c_j * T_j - R_j / (1 - alpha_j)
            // where R_j = remaining colour after splat j = C_remaining (maintained below).
            //
            //   dL/d(color_j) = alpha_j * T_j * dL/dC   (direct colour contribution)
            //   dL/d(alpha_j) = sum_c [(T_j * c_j - C_remaining_c / (1-alpha_j)) * dL_dpix_c]
            float dL_dalpha = 0.0f;
            uint32_t sidx = s_indices[j];
            float inv_one_minus_alpha = (one_minus_alpha > 1e-6f) ? 1.0f / one_minus_alpha : 0.0f;
            for (int c = 0; c < NUM_CHANNELS; c++) {
                float col_c = s_colors[j*3+c];
                // Step 3d-i: gradient w.r.t. this splat's colour
                atomicAdd(&dL_dcolors[sidx*3+c], alpha * T_before * dL_dpix[c]);
                // Step 3d-ii: full alpha gradient including transmittance term
                dL_dalpha += (T_before * col_c - C_remaining[c] * inv_one_minus_alpha) * dL_dpix[c];
            }

            // Step 3d-iii: update C_remaining with this splat's contribution
            // (must happen AFTER using C_remaining above, before moving to earlier splats)
            for (int c = 0; c < NUM_CHANNELS; c++) {
                C_remaining[c] += s_colors[j*3+c] * alpha * T_before;
            }

            // Step 3e: backprop through alpha = min(0.99, opacity * G)
            //   dL/d(opacity) = G * dL/dalpha  (if alpha < 0.99)
            //   dL/d(G)       = opacity * dL/dalpha
            float dL_dG = 0.0f;
            if (raw_alpha < 0.99f) {
                atomicAdd(&dL_dopacity[sidx], G * dL_dalpha);
                dL_dG = s_opacity[j] * dL_dalpha;
            }

            // Step 3f: backprop through G = exp(-F)
            //   dL/dF = -G * dL/d(G)
            float dL_dF = -G * dL_dG;

            // Step 3g: backprop through superquadricDistance
            float dL_dxyz[3], dL_ds[3], dL_de[3];
            superquadricDistanceGrad(xs, ys, zs, sc, ex, dL_dF,
                                     dL_dxyz, dL_ds, dL_de);

            for (int k = 0; k < 3; k++) {
                atomicAdd(&dL_dscales[sidx*3+k], dL_ds[k]);
                atomicAdd(&dL_dexps  [sidx*3+k], dL_de[k]);
            }

            // Step 3h: backprop through shape coords.
            //
            //   q* = R_cs @ [edx, edy, edz]  where edx = dx + t*·xi,
            //                                      edy = dy + t*·yi,
            //                                      edz = t*
            //   (t* treated as constant — implicit function approximation)
            //
            //   dL/d(R_cs[i,j]) = dL/dq*[i] * [edx, edy, edz][j]
            //
            //   dL/d(dx) uses R_cs column 0 only (d(q*)/d(dx) = R_cs[:,0]):
            //   dL/d(dx) = R[0]*dL/dxs + R[3]*dL/dys + R[6]*dL/dzs  — unchanged
            //   dL/d(dy) = R[1]*dL/dxs + R[4]*dL/dys + R[7]*dL/dzs  — unchanged
            float dL_dxs = dL_dxyz[0], dL_dys = dL_dxyz[1], dL_dzs = dL_dxyz[2];

            // Effective camera-space offsets for R_cs gradient
            float edx = dx + xi * t_star;
            float edy = dy + yi * t_star;
            float edz = t_star;   // non-zero: fixes the zero-gradient-for-z-rotation problem

            // All 9 elements of dL/d(R_cs) are now active
            float dL_dRcs[9] = {
                dL_dxs*edx, dL_dxs*edy, dL_dxs*edz,
                dL_dys*edx, dL_dys*edy, dL_dys*edz,
                dL_dzs*edx, dL_dzs*edy, dL_dzs*edz
            };
            for (int k = 0; k < 9; k++)
                atomicAdd(&dL_dR_cs[sidx*9+k], dL_dRcs[k]);

            // Backprop through dx, dy to means2D (unchanged — t* is constant)
            float dL_ddx = R[0]*dL_dxs + R[3]*dL_dys + R[6]*dL_dzs;
            float dL_ddy = R[1]*dL_dxs + R[4]*dL_dys + R[7]*dL_dzs;
            // dL/d(mx) = -d/focal_x * dL_ddx (mx enters via splat_cam_x)
            atomicAdd(&dL_dmeans2D[sidx*2+0], -d / focal_x * dL_ddx);
            atomicAdd(&dL_dmeans2D[sidx*2+1], -d / focal_y * dL_ddy);

            // Step 3h: update T_running by dividing out this splat's (1-alpha)
            T_running = T_before;

            if (T_running < 1e-4f) { done = 1; break; }
        }
    }
}


// ============================================================
//  preprocessBackwardCUDA
// ============================================================

// preprocessBackwardCUDA: Backprop dL/d(R_cs) to dL/d(quaternion).
//
// Forward: R_cs = R_sc.T  (R_sc = quatToMatrix(rotations[]))
// Backward: dL/d(R_sc)[i,j] = dL/d(R_cs)[j,i]  (gradient transposes through a matrix transpose)
// Then backprop through R_sc = quatToMatrix(q) to get dL/dq.
//
// Steps:
//   1. Compute dL/d(R_sc) = (dL/d(R_cs))^T
//   2. Backprop through quatToMatrix to get dL/dq (Jacobian of rotation matrix w.r.t. quaternion)
//   3. Write dL/drotations

__global__ void preprocessBackwardCUDA(
    int P,
    const float* __restrict__ rotations,
    const float* __restrict__ viewmatrix,
    const float* __restrict__ dL_dR_cs,
    float*       __restrict__ dL_drotations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    // Step 1: dL/d(R_sc)[i,j] = dL/d(R_cs)[j,i]
    const float* dL_dRcs = &dL_dR_cs[idx * 9];
    float dL_dRsc[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            dL_dRsc[i*3+j] = dL_dRcs[j*3+i];

    // Step 2: backprop through R_sc = quatToMatrix(q1,q2,q3,q4)
    //   R_sc[0,0] = 1 - 2*(q2^2+q3^2)    dR00/dq2 = -4*q2, dR00/dq3 = -4*q3
    //   R_sc[0,1] = 2*(q1*q2 - q3*q4)    dR01/dq1 = 2*q2, dR01/dq2 = 2*q1, ...
    //   (full Jacobian — 9 elements x 4 quaternion components)
    float q1 = rotations[idx*4+0], q2 = rotations[idx*4+1];
    float q3 = rotations[idx*4+2], q4 = rotations[idx*4+3];

    // dL/dq = sum_{i,j} dL/dR_sc[i,j] * dR_sc[i,j]/dq
    // Using the standard quaternion Jacobian (derived by differentiating quatToMatrix):
    float dq1 = 0, dq2 = 0, dq3 = 0, dq4 = 0;

    // Row 0: R[0]= 1-2(q2²+q3²), R[1]=2(q1q2-q3q4), R[2]=2(q1q3+q2q4)
    dq2 += dL_dRsc[0] * (-4*q2);
    dq3 += dL_dRsc[0] * (-4*q3);

    dq1 += dL_dRsc[1] * (2*q2);  dq2 += dL_dRsc[1] * (2*q1);
    dq3 += dL_dRsc[1] * (-2*q4); dq4 += dL_dRsc[1] * (-2*q3);

    dq1 += dL_dRsc[2] * (2*q3);  dq2 += dL_dRsc[2] * (2*q4);
    dq3 += dL_dRsc[2] * (2*q1);  dq4 += dL_dRsc[2] * (2*q2);

    // Row 1: R[3]=2(q1q2+q3q4), R[4]=1-2(q1²+q3²), R[5]=2(q2q3-q1q4)
    dq1 += dL_dRsc[3] * (2*q2);  dq2 += dL_dRsc[3] * (2*q1);
    dq3 += dL_dRsc[3] * (2*q4);  dq4 += dL_dRsc[3] * (2*q3);

    dq1 += dL_dRsc[4] * (-4*q1);
    dq3 += dL_dRsc[4] * (-4*q3);

    dq1 += dL_dRsc[5] * (-2*q4); dq2 += dL_dRsc[5] * (2*q3);
    dq3 += dL_dRsc[5] * (2*q2);  dq4 += dL_dRsc[5] * (-2*q1);

    // Row 2: R[6]=2(q1q3-q2q4), R[7]=2(q2q3+q1q4), R[8]=1-2(q1²+q2²)
    dq1 += dL_dRsc[6] * (2*q3);  dq2 += dL_dRsc[6] * (-2*q4);
    dq3 += dL_dRsc[6] * (2*q1);  dq4 += dL_dRsc[6] * (-2*q2);

    dq1 += dL_dRsc[7] * (2*q4);  dq2 += dL_dRsc[7] * (2*q3);
    dq3 += dL_dRsc[7] * (2*q2);  dq4 += dL_dRsc[7] * (2*q1);

    dq1 += dL_dRsc[8] * (-4*q1);
    dq2 += dL_dRsc[8] * (-4*q2);

    // Step 3: write
    dL_drotations[idx*4+0] = dq1;
    dL_drotations[idx*4+1] = dq2;
    dL_drotations[idx*4+2] = dq3;
    dL_drotations[idx*4+3] = dq4;
}


// means3DBackwardCUDA: Backprop dL/d(means2D) to dL/d(means3D).
//
// The projection is: px = x/z * focal_x + W/2,  py = y/z * focal_y + H/2
// So: dL/dx = dL_dmeans2D_x * focal_x / z
//     dL/dy = dL_dmeans2D_y * focal_y / z
//     dL/dz = dL_dmeans2D_x * (-x * focal_x / z^2)
//           + dL_dmeans2D_y * (-y * focal_y / z^2)
//
// Steps:
//   1. Load per-splat 2D gradient and 3D position
//   2. Compute Jacobian terms and accumulate dL/d(means3D)
__global__ void means3DBackwardCUDA(
    int P,
    const float* __restrict__ means3D,
    const float* __restrict__ dL_dmeans2D,  // (P,2) from render backward
    const int*   __restrict__ radii,         // skip if radius == 0 (culled)
    float        focal_x, float focal_y,
    float*       __restrict__ dL_dmeans3D)   // (P,3) output
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;
    if (radii[idx] == 0) return;  // Step 1: skip culled splats

    float x = means3D[idx*3+0];
    float y = means3D[idx*3+1];
    float z = means3D[idx*3+2];
    if (z <= 0.0f) return;

    float g2x = dL_dmeans2D[idx*2+0];
    float g2y = dL_dmeans2D[idx*2+1];

    // Step 2: Jacobian of projection
    float inv_z  = 1.0f / z;
    float inv_z2 = inv_z * inv_z;

    dL_dmeans3D[idx*3+0] = g2x * focal_x * inv_z;
    dL_dmeans3D[idx*3+1] = g2y * focal_y * inv_z;
    dL_dmeans3D[idx*3+2] = -g2x * x * focal_x * inv_z2
                           - g2y * y * focal_y * inv_z2;
}


// ============================================================
//  Kernel launchers
// ============================================================

namespace BACKWARD {

void render(
    const dim3 grid, dim3 block,
    const uint2* ranges, const uint32_t* point_list,
    int W, int H,
    float focal_x, float focal_y,
    const float* bg,
    const float2* means2D, const float* depths,
    const float* R_cs, const float* scales, const float* exps,
    const float* colors, const float* opacities,
    const float* final_T, const uint32_t* n_contrib,
    const float* dL_dpix_color,
    float* dL_dmeans2D, float* dL_dcolors,
    float* dL_dopacity, float* dL_dR_cs,
    float* dL_dscales,  float* dL_dexps)
{
    float cx = (float)W * 0.5f;
    float cy = (float)H * 0.5f;
    renderBackwardCUDA<<<grid, block>>>(
        ranges, point_list, W, H, focal_x, focal_y, cx, cy,
        bg, means2D, depths, R_cs, scales, exps, colors, opacities,
        final_T, n_contrib, dL_dpix_color,
        dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dR_cs,
        dL_dscales, dL_dexps);
}

void preprocess(
    int P,
    const float* means3D,
    const float* rotations,
    const float* viewmatrix,
    const int*   radii,
    float        focal_x, float focal_y,
    const float* dL_dR_cs,
    const float* dL_dmeans2D,
    float*       dL_drotations,
    float*       dL_dmeans3D)
{
    // Launch rotation backward
    preprocessBackwardCUDA<<<(P+255)/256, 256>>>(
        P, rotations, viewmatrix, dL_dR_cs, dL_drotations);

    // Launch means3D backward (projection Jacobian)
    means3DBackwardCUDA<<<(P+255)/256, 256>>>(
        P, means3D, dL_dmeans2D, radii, focal_x, focal_y, dL_dmeans3D);
}

} // namespace BACKWARD
