// auxiliary.h: Device helper functions for the superquadric rasterizer.
//
// Contains:
//   - General geometry helpers  (getRect, ndc2Pix, transformPoint4x3/4x4)
//   - Superquadric math         (dabsPow, superquadricDistance, superquadricDistanceGrad)
//   - Rotation helpers          (quatToMatrix, matMul3x3, matvec3x3)
//   - Rim-depth estimation      (superquadricNormal, evalRimH, findRimDepth)
//   - Debug macro               (CHECK_CUDA)

#pragma once

#include <cuda_runtime.h>
#include "config.h"
#include <cstdio>

// ============================================================
//  Debug helper
// ============================================================

// CHECK_CUDA: If debug=true, synchronise the device and check for CUDA errors.
// Steps:
//   1. Synchronise all GPU work
//   2. Check the last CUDA error
//   3. If an error is found, print a message and abort
#define CHECK_CUDA(A, debug)                                                \
    if (debug) {                                                            \
        cudaDeviceSynchronize();                                            \
        cudaError_t err = cudaGetLastError();                               \
        if (err != cudaSuccess) {                                           \
            printf("[CUDA ERROR] %s at line %d: %s\n",                     \
                   __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(1);                                                        \
        }                                                                   \
    }


// ============================================================
//  General geometry helpers
// ============================================================

// ndc2Pix: Convert a normalised device coordinate in [-1, 1] to a pixel index.
// Steps:
//   1. Shift from [-1,1] to [0,2]
//   2. Scale by half the image dimension
//   3. Subtract 0.5 to centre on pixel
__device__ inline float ndc2Pix(float v, int S)
{
    // Step 1-3: combined formula
    return ((v + 1.0f) * S - 1.0f) * 0.5f;
}

// getRect: Compute the tile-aligned bounding rectangle for a 2D splat.
// Steps:
//   1. Expand the 2D pixel centre by the splat radius (in pixels)
//   2. Clamp to image bounds expressed as tile coordinates
//   3. Write tile-space min/max corners into rect_min and rect_max
__device__ inline void getRect(
    const float2 p,          // 2D pixel-space centre of splat
    int           radius,    // pixel-space radius
    uint2&        rect_min,  // output: top-left tile corner
    uint2&        rect_max,  // output: bottom-right tile corner (exclusive)
    const dim3    grid)      // tile grid dimensions
{
    // Step 1-3: expand by radius, clamp to tile grid
    rect_min = {
        min(grid.x, max(0u, (uint)((p.x - radius) / BLOCK_X))),
        min(grid.y, max(0u, (uint)((p.y - radius) / BLOCK_Y)))
    };
    rect_max = {
        min(grid.x, max(0u, (uint)((p.x + radius + BLOCK_X - 1) / BLOCK_X))),
        min(grid.y, max(0u, (uint)((p.y + radius + BLOCK_Y - 1) / BLOCK_Y)))
    };
}

// transformPoint4x3: Apply a 4x3 matrix (no homogeneous divide) to a float3 point.
// Steps:
//   1. Multiply point by each row of the 4x3 matrix (column-major storage)
//   2. Return the resulting float3
__device__ inline float3 transformPoint4x3(const float3& p, const float* m)
{
    // Step 1-2: mat-vec multiply, column-major 4x4 with implicit w=1
    return {
        m[0]*p.x + m[4]*p.y + m[8] *p.z + m[12],
        m[1]*p.x + m[5]*p.y + m[9] *p.z + m[13],
        m[2]*p.x + m[6]*p.y + m[10]*p.z + m[14]
    };
}

// transformPoint4x4: Apply a 4x4 matrix to a float3 point (returns float4 with w).
// Steps:
//   1. Compute xyz components from rows 0-2
//   2. Compute w component from row 3
__device__ inline float4 transformPoint4x4(const float3& p, const float* m)
{
    // Step 1: xyz
    float3 xyz = transformPoint4x3(p, m);
    // Step 2: w
    float w = m[3]*p.x + m[7]*p.y + m[11]*p.z + m[15];
    return { xyz.x, xyz.y, xyz.z, w };
}


// ============================================================
//  Superquadric math
// ============================================================

// dabsPow: Differentiable absolute-value power: sign(x) * max(|x|, eps)^p
//   This avoids NaN gradients in pow(0, p) for non-integer p.
//   Equivalent to the Python Dabs_(x)^p used throughout Superquadric_Splatting.py.
//
// Steps:
//   1. Take absolute value, clamp to eps to avoid pow(0, p)
//   2. Raise to power p
//   3. Restore sign (for odd-symmetry of the signed version)
__device__ inline float dabsPow(float x, float p)
{
    const float eps = 1e-8f;
    // Step 1: safe absolute value
    float ax = fabsf(x);
    if (ax < eps) ax = eps;
    // Step 2: power
    float r = powf(ax, p);
    // Step 3: sign (we return unsigned here; sign is preserved by caller when needed)
    return r;
}

// signedDabsPow: sign(x) * |x|^p — preserves sign of x through the power operation.
// Used when the signed form is required (e.g., gradient computations).
__device__ inline float signedDabsPow(float x, float p)
{
    const float eps = 1e-8f;
    float ax = fabsf(x);
    if (ax < eps) ax = eps;
    float r = powf(ax, p);
    return (x >= 0.0f) ? r : -r;
}

// superquadricDistance: Evaluate the superquadric implicit distance function F.
//
// The superquadric implicit surface is defined by F(xs, ys, zs) = 1.
// Interior points have F < 1, exterior have F > 1.
// We use F as a soft distance: weight = exp(-F).
//
// Formula (matching Superquadric_Distance_ in Superquadric_Splatting.py):
//   F = ( ( |xs/s1|^(2/e2) + |ys/s2|^(2/e2) )^(e2/e1) + |zs/s3|^(2/e1) )^e3
//
// Steps:
//   1. Normalise coords by scale: u = xs/s1, v = ys/s2, w = zs/s3
//   2. Apply lateral exponent e2: d1 = |u|^(2/e2), d2 = |v|^(2/e2), d3 = |w|^(2/e1)
//   3. Sum lateral terms and apply vertical exponent e1: d4 = (d1+d2)^(e2/e1) + d3
//   4. Clamp and apply outer exponent e3: F = clamp(d4)^e3
__device__ inline float superquadricDistance(
    float xs, float ys, float zs,   // point in shape-local frame
    const float* s,                  // scales [s1, s2, s3]
    const float* e)                  // exponents [e1, e2, e3]
{
    const float eps = 1e-8f;
    float e1 = e[0], e2 = e[1], e3 = e[2];
    float s1 = s[0], s2 = s[1], s3 = s[2];

    // Step 1: normalise by scale
    float u = xs / s1;
    float v = ys / s2;
    float w = zs / s3;

    // Step 2: lateral power 2/e2, vertical power 2/e1
    float d1 = dabsPow(u, 2.0f / e2);
    float d2 = dabsPow(v, 2.0f / e2);
    float d3 = dabsPow(w, 2.0f / e1);

    // Step 3: sum lateral, apply (e2/e1) power, add vertical
    float lat = d1 + d2;
    if (lat < eps) lat = eps;
    float d4 = powf(lat, e2 / e1) + d3;

    // Step 4: clamp and apply outer exponent
    d4 = fmaxf(eps, fminf(d4, 10.0f));
    float F = powf(d4, e3);

    return F;
}

// superquadricDistanceGrad: Compute analytic gradients of F w.r.t. all inputs.
//
// All gradients derived by chain rule through the four steps of superquadricDistance.
//
// Steps:
//   1. Recompute forward intermediates (same as superquadricDistance)
//   2. Backprop through step 4: dF/d(d4)
//   3. Backprop through step 3: d(d4)/d(lat), d(d4)/d(d3)
//   4. Backprop through step 2: d(d1)/d(u), d(d2)/d(v), d(d3)/d(w)
//   5. Backprop through step 1 (scale division): d(u)/d(xs), d/d(s1) etc.
//   6. Compute dF/d(e1), dF/d(e2), dF/d(e3) using d/de[x^(a/e)] = x^(a/e)*ln(x)*(-a/e^2)
__device__ inline void superquadricDistanceGrad(
    float xs, float ys, float zs,
    const float* s, const float* e,
    float dL_dF,          // upstream gradient w.r.t. F
    float* dL_dxyz,       // output: [dL/dxs, dL/dys, dL/dzs]
    float* dL_ds,         // output: [dL/ds1, dL/ds2, dL/ds3]
    float* dL_de)         // output: [dL/de1, dL/de2, dL/de3]
{
    const float eps = 1e-8f;
    float e1 = e[0], e2 = e[1], e3 = e[2];
    float s1 = s[0], s2 = s[1], s3 = s[2];

    // Step 1: recompute forward intermediates
    float u = xs / s1;
    float v = ys / s2;
    float w = zs / s3;

    float au = fmaxf(fabsf(u), eps);
    float av = fmaxf(fabsf(v), eps);
    float aw = fmaxf(fabsf(w), eps);

    float d1 = powf(au, 2.0f / e2);
    float d2 = powf(av, 2.0f / e2);
    float d3 = powf(aw, 2.0f / e1);

    float lat = fmaxf(d1 + d2, eps);
    float lat_e2e1 = powf(lat, e2 / e1);
    float d4 = fmaxf(fminf(lat_e2e1 + d3, 10.0f), eps);
    float F  = powf(d4, e3);

    // Step 2: dF/d(d4) = e3 * d4^(e3-1)
    float dF_dd4 = (d4 > eps && d4 < 10.0f) ? (e3 * powf(d4, e3 - 1.0f)) : 0.0f;
    float g4 = dL_dF * dF_dd4;

    // Step 3: backprop through d4 = lat^(e2/e1) + d3
    // d(d4)/d(lat_e2e1) = 1, d(lat^(e2/e1))/d(lat) = (e2/e1) * lat^(e2/e1 - 1)
    float g_lat_e2e1 = g4;
    float dlatE2e1_dlat = (e2 / e1) * powf(lat, e2 / e1 - 1.0f);
    float g_lat = g_lat_e2e1 * dlatE2e1_dlat;
    // d(d4)/d(d3) = 1
    float g_d3 = g4;

    // Step 4: backprop through d1=|u|^(2/e2), d2=|v|^(2/e2), d3=|w|^(2/e1)
    // d(d1)/d(u) = (2/e2) * |u|^(2/e2 - 1) * sign(u)
    float g_d1 = g_lat;
    float g_d2 = g_lat;

    float dd1_du = (2.0f / e2) * powf(au, 2.0f / e2 - 1.0f) * ((u >= 0.0f) ? 1.0f : -1.0f);
    float dd2_dv = (2.0f / e2) * powf(av, 2.0f / e2 - 1.0f) * ((v >= 0.0f) ? 1.0f : -1.0f);
    float dd3_dw = (2.0f / e1) * powf(aw, 2.0f / e1 - 1.0f) * ((w >= 0.0f) ? 1.0f : -1.0f);

    float g_u = g_d1 * dd1_du;
    float g_v = g_d2 * dd2_dv;
    float g_w = g_d3 * dd3_dw;

    // Step 5: backprop through u=xs/s1, v=ys/s2, w=zs/s3
    dL_dxyz[0] = g_u / s1;
    dL_dxyz[1] = g_v / s2;
    dL_dxyz[2] = g_w / s3;

    dL_ds[0] = -g_u * xs / (s1 * s1);
    dL_ds[1] = -g_v * ys / (s2 * s2);
    dL_ds[2] = -g_w * zs / (s3 * s3);

    // Step 6: gradients w.r.t. exponents
    // d/de2 of d1 = |u|^(2/e2): d = |u|^(2/e2) * ln(|u|) * (-2/e2^2)
    float log_au = logf(au);
    float log_av = logf(av);
    float log_aw = logf(aw);
    float log_lat = logf(lat);

    float dF_de2 =
        g_d1 * d1 * log_au * (-2.0f / (e2 * e2)) +
        g_d2 * d2 * log_av * (-2.0f / (e2 * e2)) +
        // also appears in the exponent of lat^(e2/e1): d/de2 = lat^(e2/e1)*log(lat)*(1/e1)
        g_lat_e2e1 * lat_e2e1 * log_lat / e1;

    // d/de1 of d3 = |w|^(2/e1): (-2/e1^2)*|w|^(2/e1)*ln|w|
    // also in lat^(e2/e1): d/de1 = lat^(e2/e1)*log(lat)*(-e2/e1^2)
    float dF_de1 =
        g_d3 * d3 * log_aw * (-2.0f / (e1 * e1)) +
        g_lat_e2e1 * lat_e2e1 * log_lat * (-e2 / (e1 * e1));

    // d/de3 of F = d4^e3: F*ln(d4)
    float dF_de3 = (d4 > eps) ? (g4 * F * logf(d4) / e3) : 0.0f;
    // Note: g4 already has dL_dF; factor in again only log(d4)*d4^e3, not g4*dF_dd4*...
    // Corrected: dL/de3 = dL_dF * F * ln(d4)
    dF_de3 = dL_dF * ((d4 > eps) ? (F * logf(d4)) : 0.0f);

    dL_de[0] = dF_de1;
    dL_de[1] = dF_de2;
    dL_de[2] = dF_de3;
}


// ============================================================
//  Rotation helpers
// ============================================================

// quatToMatrix: Build a 3x3 rotation matrix from a unit quaternion [q1,q2,q3,q4]
//   where q4 is the scalar component (w). Stored row-major in out[9].
//
// Steps:
//   1. Extract quaternion components
//   2. Compute the 9 elements of the rotation matrix using the standard formula
//   3. Write to output array in row-major order
__device__ inline void quatToMatrix(const float* q, float* out)
{
    // Step 1: extract
    float q1 = q[0], q2 = q[1], q3 = q[2], q4 = q[3];  // q4 = scalar w

    // Step 2-3: fill row-major 3x3
    out[0] = 1.0f - 2.0f*(q2*q2 + q3*q3);
    out[1] = 2.0f*(q1*q2 - q3*q4);
    out[2] = 2.0f*(q1*q3 + q2*q4);

    out[3] = 2.0f*(q1*q2 + q3*q4);
    out[4] = 1.0f - 2.0f*(q1*q1 + q3*q3);
    out[5] = 2.0f*(q2*q3 - q1*q4);

    out[6] = 2.0f*(q1*q3 - q2*q4);
    out[7] = 2.0f*(q2*q3 + q1*q4);
    out[8] = 1.0f - 2.0f*(q1*q1 + q2*q2);
}

// matMul3x3: Multiply two 3x3 row-major matrices: out = A @ B
//
// Steps:
//   1. For each output element (i,j): dot product of row i of A with col j of B
__device__ inline void matMul3x3(const float* A, const float* B, float* out)
{
    // Step 1: standard 3x3 multiply
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            out[i*3 + j] = 0.0f;
            for (int k = 0; k < 3; k++) {
                out[i*3 + j] += A[i*3 + k] * B[k*3 + j];
            }
        }
    }
}

// matvec3x3: Multiply 3x3 row-major matrix by float3 vector: out = M @ v
//
// Steps:
//   1. Dot each row of M with v
__device__ inline float3 matvec3x3(const float* M, float dx, float dy, float dz)
{
    // Step 1: row-dot-vector
    return {
        M[0]*dx + M[1]*dy + M[2]*dz,
        M[3]*dx + M[4]*dy + M[5]*dz,
        M[6]*dx + M[7]*dy + M[8]*dz
    };
}


// ============================================================
//  Rim-depth estimation
// ============================================================

// superquadricNormal: Unit surface normal via direct Cartesian formula (no e3).
//
// e3 only scales F, not the gradient direction, so n_hat = ∇d4/|∇d4| where
// d4 = lat^(e2/e1) + |z/s3|^(2/e1) and lat = |x/s1|^(2/e2) + |y/s2|^(2/e2).
//
// Cartesian normal (Barr 1981):
//   g_x = (1/s1)·|x/s1|^(2/e2−1)·sign(x)·lat^(e2/e1−1)
//   g_y = (1/s2)·|y/s2|^(2/e2−1)·sign(y)·lat^(e2/e1−1)
//   g_z = (1/s3)·|z/s3|^(2/e1−1)·sign(z)
//   n_hat = g / |g|
//
// All base values clamped to eps before raising to fractional/negative powers.
__device__ __forceinline__ void superquadricNormal(
    float xs, float ys, float zs,
    const float* __restrict__ s,
    const float* __restrict__ e,
    float* __restrict__ n_hat)
{
    const float eps = 1e-8f;
    float alpha = 2.0f / e[1];           // 2/e2
    float beta  = e[1] / e[0];           // e2/e1
    float gamma = 2.0f / e[0];           // 2/e1

    float aux = fmaxf(fabsf(xs / s[0]), eps);
    float auy = fmaxf(fabsf(ys / s[1]), eps);
    float auz = fmaxf(fabsf(zs / s[2]), eps);
    float sx = (xs >= 0.0f) ? 1.0f : -1.0f;
    float sy = (ys >= 0.0f) ? 1.0f : -1.0f;
    float sz = (zs >= 0.0f) ? 1.0f : -1.0f;

    float lat    = fmaxf(powf(aux, alpha) + powf(auy, alpha), eps);
    float lat_b1 = powf(lat, beta - 1.0f);

    float gx = beta * lat_b1 * alpha * powf(aux, alpha - 1.0f) / s[0] * sx;
    float gy = beta * lat_b1 * alpha * powf(auy, alpha - 1.0f) / s[1] * sy;
    float gz = gamma * powf(auz, gamma - 1.0f) / s[2] * sz;

    float inv = 1.0f / fmaxf(sqrtf(gx*gx + gy*gy + gz*gz), eps);
    n_hat[0] = gx * inv;
    n_hat[1] = gy * inv;
    n_hat[2] = gz * inv;
}

// evalRimH: h(t) = n̂(q0 + t·vs) · vs  — the rim condition (= 0 at the silhouette).
//
// For a convex superquadric, h is monotone along any ray with direction vs, so
// a bracketed root-finder (Regula Falsi) is guaranteed to converge.
__device__ __forceinline__ float evalRimH(
    float xs, float ys, float zs,
    const float* __restrict__ vs,
    const float* __restrict__ sc,
    const float* __restrict__ ex)
{
    float n[3];
    superquadricNormal(xs, ys, zs, sc, ex, n);
    return n[0]*vs[0] + n[1]*vs[1] + n[2]*vs[2];
}

// findRimDepth: false-position (Regula Falsi) to find t* where n̂(q0+t*·vs)·vs = 0.
//
// Steps:
//   1. Bracket root in [−2R, +2R] where R = max(scales).  Because q0 lies in the
//      plane orthogonal to vs, the z-component of q0 + t·vs equals t exactly, so
//      ±2R always falls outside the shape and h(±2R) ≈ ±1.
//   2. If h doesn't change sign, the ray hits a flat face — no rim exists; return 0.
//   3. Iterate with false position: mid = lo − h_lo·(hi−lo)/(h_hi−h_lo).
//      Converges faster than bisection (superlinear for smooth h) with no derivative.
__device__ __forceinline__ float findRimDepth(
    const float* __restrict__ q0,
    const float* __restrict__ vs,
    const float* __restrict__ sc,
    const float* __restrict__ ex)
{
    const float R  = fmaxf(sc[0], fmaxf(sc[1], sc[2]));
    float lo = -2.0f * R,  hi = 2.0f * R;

    float h_lo = evalRimH(q0[0]+vs[0]*lo, q0[1]+vs[1]*lo, q0[2]+vs[2]*lo, vs, sc, ex);
    float h_hi = evalRimH(q0[0]+vs[0]*hi, q0[1]+vs[1]*hi, q0[2]+vs[2]*hi, vs, sc, ex);

    if (h_lo * h_hi > 0.0f) return 0.0f;   // no sign change — flat face, no rim

    for (int i = 0; i < 16; i++) {
        float mid   = lo - h_lo * (hi - lo) / (h_hi - h_lo);
        float h_mid = evalRimH(q0[0]+vs[0]*mid, q0[1]+vs[1]*mid, q0[2]+vs[2]*mid, vs, sc, ex);
        if (fabsf(h_mid) < 1e-6f) return mid;
        if (h_lo * h_mid <= 0.0f) { hi = mid; h_hi = h_mid; }
        else                       { lo = mid; h_lo = h_mid; }
    }
    return 0.5f * (lo + hi);
}
