# CUDA Superquadric Rasterizer: Technical Addendum

This addendum describes the differentiable CUDA rasterizer implemented in
`submodules/diff-superquadric-rasterization/` and documents the mathematical and
algorithmic changes introduced relative to both the original Python renderer and the
3D Gaussian Splatting (3DGS) rasterizer of Kerbl et al. (2023).  Mathematics that
is identical to 3DGS — tile-sorting, alpha compositing, backward accumulation — is
referenced but not re-derived here.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $\mathbf{c} = (c_x, c_y, c_z)^\top$ | Splat centre in camera space |
| $d = c_z$ | Splat depth (camera-space $z$) |
| $(\xi_i, \eta_i)$ | Normalised camera-plane coordinates for pixel $i$: $\xi_i = (p_x + \tfrac{1}{2} - c_x^{img}) / f_x$ |
| $(s_1, s_2, s_3)$ | Semi-axis scales (world units; activation: $\exp$) |
| $(\varepsilon_1, \varepsilon_2, \varepsilon_3)$ | Shape exponents; $\varepsilon_1, \varepsilon_2 \in (0.1, 1.9)$, $\varepsilon_3 \in (0.5, 1.5)$ |
| $R_{sc}$ | Rotation: shape $\to$ camera frame (precomposed in Python before kernel launch) |
| $R_{cs} = R_{sc}^\top$ | Rotation: camera $\to$ shape frame (built inside the preprocess kernel) |
| $\mathbf{v}_s$ | Camera viewing direction expressed in shape frame |
| $\mathbf{q}_0$ | Depth-plane base point in shape frame |
| $t^*$ | Rim depth offset (scalar) |
| $\mathbf{q}^* = \mathbf{q}_0 + t^* \mathbf{v}_s$ | Surface point used for distance evaluation |
| $F(\mathbf{q}^*)$ | Superquadric implicit distance at $\mathbf{q}^*$ |
| $G$ | Splat weight: $G = e^{-F}$ |
| $\alpha$ | Per-pixel opacity contribution |
| $T$ | Accumulated transmittance |

---

## 2. Pipeline Overview

The CUDA rasterizer follows the same four-stage pipeline as 3DGS:

1. **Preprocess** — one CUDA thread per splat; projects to 2D, computes a conservative pixel-space bounding radius, culls, and stores $R_{cs}$.
2. **Sort** — CUB radix sort of all (tile-id, depth) keys across all surviving splats (identical to 3DGS).
3. **Identify tile ranges** — scan the sorted key array to record the index range belonging to each tile (identical to 3DGS).
4. **Render** — one CUDA thread per pixel; front-to-back alpha-blend within each 16×16 tile.

The novel content is concentrated in stages 1 and 4.

---

## 3. Preprocessing

### 3.1 Camera-Space Projection

Each splat centre is already provided in camera space (the Python caller applies the
world-to-camera transform before invoking the kernel).  The 2D pixel-space projection is
the standard pinhole mapping:

$$p_x = \frac{c_x}{c_z} f_x + \frac{W}{2}, \qquad p_y = \frac{c_y}{c_z} f_y + \frac{H}{2}$$

Splats with $c_z \le 0.01$ are discarded (behind the near plane).

### 3.2 Conservative Bounding Radius

3DGS uses the maximum 2D Gaussian eigenvalue to derive a screen-space radius.  Superquadrics have no closed-form 2D projection, so a geometric approximation is used instead.

Let $\|\mathbf{s}\| = \sqrt{s_1^2 + s_2^2 + s_3^2}$ be the Euclidean norm of the scale vector.  The pixel-space bounding radius is:

$$r = \left\lceil \frac{3\,\|\mathbf{s}\|}{c_z} \cdot \bar{f} \right\rceil, \qquad \bar{f} = \tfrac{1}{2}(f_x + f_y)$$

This mirrors 3DGS's 3-sigma rule: a sphere of radius $\|\mathbf{s}\|$ subtends an angular half-width of $\|\mathbf{s}\|/c_z$ radians at focal length $\bar{f}$.  The shape exponent $\varepsilon_3$ is deliberately excluded; it scales the value of $F$ but not the geometric extent of the surface.

The radius is capped at the image diagonal $\sqrt{W^2 + H^2}$ to handle pathological cases.

### 3.3 Frustum Culling

Tile coverage is computed via `getRect`, which clips the bounding circle to the tile grid.
A splat is discarded if and only if it overlaps zero tiles after clipping:

$$n_{tiles} = (\text{rect\_max}_x - \text{rect\_min}_x)(\text{rect\_max}_y - \text{rect\_min}_y) = 0 \implies \text{discard}$$

This is identical to the 3DGS frustum cull. The CUDA tile-overlap test is strictly more correct because it
accounts for the splat's radius, not just its centre position.

### 3.4 Camera-to-Shape Rotation

The preprocess kernel receives $R_{sc}$ encoded as a unit quaternion (precomposed in the
Python caller as $R_{sc} = R_{wc} \circ R_{sw}$, i.e.\ the product of the
world-to-camera and shape-to-world rotations).  The rotation matrix $R_{sc}$ is built
via the standard quaternion formula and then transposed to yield:

$$R_{cs} = R_{sc}^\top$$

$R_{cs}$ is stored as 9 floats per splat in a global array and loaded into shared memory
during the render kernel.

---

## 4. Per-Pixel Rendering

### 4.1 The Rim Problem

The rim is the set of surface points where the superquadric surface normal is perpendicular to the
viewing direction.  It defines the visible silhouette and, more importantly here, provides
the correct $z$-depth used for depth-of-field and occlusion sorting.

#### 4.2.1 Problem Formulation

Let the camera viewing direction in shape space be:

$$\mathbf{v}_s = R_{cs}\,\hat{\mathbf{z}}_c = R_{cs}[:,2]$$

(Under the orthographic approximation used in the render kernel — `RIM_ORTHOGRAPHIC = 1`
in `config.h` — all pixels within a tile share the same viewing direction $\mathbf{v}_s$,
eliminating a per-pixel matrix-vector product.)

Consider the parametric ray through $\mathbf{q}_0$ in the direction $\mathbf{v}_s$:

$$\mathbf{q}(t) = \mathbf{q}_0 + t\,\mathbf{v}_s, \qquad t \in \mathbb{R}$$

We seek the *rim depth* $t^*$ that satisfies the rim condition:

$$h(t^*) = 0, \qquad h(t) \triangleq \hat{\mathbf{n}}\!\left(\mathbf{q}(t)\right) \cdot \mathbf{v}_s$$

where $\hat{\mathbf{n}}(\mathbf{q})$ is the unit outward normal of the superquadric at $\mathbf{q}$.

#### 4.2.2 Surface Normal

The unnormalised gradient of the superquadric implicit function with respect to
the shape-frame coordinates (Barr 1981) is:

$$g_x = \frac{2}{\varepsilon_1}\,\frac{\mathrm{sgn}(q_x)}{s_1}\left|\frac{q_x}{s_1}\right|^{2/\varepsilon_2 - 1} L^{\varepsilon_2/\varepsilon_1 - 1}$$

$$g_y = \frac{2}{\varepsilon_1}\,\frac{\mathrm{sgn}(q_y)}{s_2}\left|\frac{q_y}{s_2}\right|^{2/\varepsilon_2 - 1} L^{\varepsilon_2/\varepsilon_1 - 1}$$

$$g_z = \frac{2}{\varepsilon_1}\,\frac{\mathrm{sgn}(q_z)}{s_3}\left|\frac{q_z}{s_3}\right|^{2/\varepsilon_1 - 1}$$

where $L = |q_x/s_1|^{2/\varepsilon_2} + |q_y/s_2|^{2/\varepsilon_2}$ is the lateral sum.
The unit normal is $\hat{\mathbf{n}} = \mathbf{g}/\|\mathbf{g}\|$.

Note that $\varepsilon_3$ does not appear in the normal formula: it scales $F$ but not its
gradient direction, so $\hat{\mathbf{n}}$ depends only on $(\varepsilon_1, \varepsilon_2)$.

All base values are clamped to $\epsilon = 10^{-8}$ before exponentiation to avoid
$0^p$ for non-integer $p$.

#### 4.2.3 The Regula Falsi Solver

For a convex superquadric, $h(t)$ is monotone along any ray that does not pass through
the centre: as $t$ increases from $-\infty$ to $+\infty$, the normal rotates continuously
from pointing away from $\mathbf{v}_s$ to pointing toward it, crossing zero exactly once.
This guarantees that any bracket $[t_a, t_b]$ with $h(t_a)\cdot h(t_b) < 0$ contains
exactly one root.

The bracket is initialised as $[-2R,\, +2R]$ where $R = \max(s_1, s_2, s_3)$.  Because
$\mathbf{q}_0$ lies in the plane $\mathbf{v}_s^\top \mathbf{q} = 0$, the component of
$\mathbf{q}(t)$ along $\mathbf{v}_s$ equals $t$, so $\pm 2R$ always lies well outside the
shape boundary and $h(\pm 2R) \approx \pm 1$.

If $h(-2R)$ and $h(+2R)$ share the same sign, the ray traverses a nearly flat face (normal
approximately parallel to $\mathbf{v}_s$ throughout).  In this degenerate case no rim
exists; the kernel falls back to $t^* = 0$ (depth-plane evaluation).

Root-finding is carried out using the **Regula Falsi** method (False Position).  At each
iteration, rather than bisecting the interval, the method fits a linear interpolant through
the two endpoint values and places the new trial point at the zero of that line:

$$t_{k+1} = t_a - h(t_a)\,\frac{t_b - t_a}{h(t_b) - h(t_a)}$$

The bracket is then updated by replacing whichever endpoint has the same sign as $h(t_{k+1})$:

$$[t_a, t_b] \leftarrow \begin{cases} [t_a,\ t_{k+1}] & \text{if } h(t_a)\,h(t_{k+1}) > 0 \\ [t_{k+1},\ t_b] & \text{otherwise} \end{cases}$$

This is distinct from bisection in that the trial point is weighted toward the end where
$h$ has smaller magnitude, giving superlinear convergence when $h$ is smooth.
Convergence is declared when $|h(t_{k+1})| < 10^{-6}$; otherwise, 16 iterations are
performed and the midpoint $\tfrac{1}{2}(t_a + t_b)$ is returned.  In practice, the
method converges to machine precision within 6–8 iterations for all superquadric shapes
tested.

The rim intersection surface point is then:

$$\mathbf{q}^* = \mathbf{q}_0 + t^*\,\mathbf{v}_s$$

---

## 5. Superquadric Distance Field

With $\mathbf{q}^* = (q^*_x, q^*_y, q^*_z)^\top$ in hand, the implicit distance is
evaluated as:

$$F(\mathbf{q}^*) = \left[\,\underbrace{\left(\left|\frac{q^*_x}{s_1}\right|^{2/\varepsilon_2} + \left|\frac{q^*_y}{s_2}\right|^{2/\varepsilon_2}\right)^{\varepsilon_2/\varepsilon_1}}_{\text{lateral term}} + \underbrace{\left|\frac{q^*_z}{s_3}\right|^{2/\varepsilon_1}}_{\text{axial term}}\,\right]^{\varepsilon_3}$$

The outer exponent $\varepsilon_3$ is a superscalar deformation: for $\varepsilon_3 = 1$ the surface is a
standard Barr superquadric; for $\varepsilon_3 < 1$ the interior is inflated; for
$\varepsilon_3 > 1$ it is compressed toward the boundary.

The splat weight is the Gaussian of the distance:

$$G = \exp(-F)$$

For a point exactly on the surface $F = 1$, giving $G = e^{-1} \approx 0.37$.
Interior points ($F < 1$) have $G > e^{-1}$; exterior points have $G < e^{-1}$.
This matches the convention in the Python renderer (`Superquadric_Distance_` / `Weight_`).

---

## 6. Alpha Compositing

Per-pixel opacity is formed as:

$$\alpha_k = \min\!\left(0.99,\; o_k \cdot G_k\right)$$

where $o_k \in (0,1)$ is the learned scalar opacity of splat $k$.
Contributions with $\alpha_k < 1/255$ are skipped.

Splats are composited front-to-back, accumulating colour and depth with transmittance $T$:

$$C_i = \sum_{k=1}^{K} \mathbf{c}_k \,\alpha_k \prod_{j<k}(1-\alpha_j), \qquad \hat{C}_i = C_i + T_K\,\mathbf{c}_{bg}$$

where $T_K = \prod_{k=1}^{K}(1-\alpha_k)$ is the residual transmittance and $\mathbf{c}_{bg}$
is the background colour.  This is identical to 3DGS and requires no per-tile normalisation:
each pixel is owned by exactly one tile and the compositing formula is self-consistent.
The render loop exits early when $T < 10^{-4}$ (pixel fully saturated).

---

## 7. Changes Relative to the Python Renderer

The Python renderer (`rasterizer3` in `gaussian_renderer/__init__.py`) is a tiled
accumulation loop that operates on batched PyTorch tensors.  The key algorithmic
differences are as follows.

### 7.1 Rim Estimation Method

**Python.**  The rim is estimated by evaluating $h(t) = \hat{\mathbf{n}}(\mathbf{q}(t))\cdot\mathbf{v}_s$
on a fixed grid of 29 uniformly-spaced $t$-values spanning $[-2R, +2R]$ and selecting the
index with the smallest $|h|$.  This is a search over a discrete set; it misses the true
root whenever it falls between two sample points.

**CUDA.**  The rim is found to tolerance $10^{-6}$ via Regula Falsi (Section 4.2.3).
The result is exact (up to floating-point precision) regardless of shape parameters, and
requires at most 16 function evaluations rather than a fixed 29.

### 7.2 Frustum Culling

**Python.**  Before calling the CUDA kernel, the Python caller applied an additional
angular visibility filter:

$$|c_x / c_z| < 1.05\,\tan\phi_x \quad \text{and} \quad |c_y / c_z| < 1.05\,\tan\phi_y$$

This test operated on splat *centres* only, discarding large-radius splats whose centres
were slightly outside the field of view even though they would contribute pixels within it.

**CUDA.**  The angular filter has been removed.  Culling is performed entirely inside the
preprocess kernel via the `n_tiles = 0` test (Section 3.3), which correctly accounts for
the splat's radius.  This matches the 3DGS culling strategy exactly.

### 7.3 Bounding Radius Formula

**Python.**  The screen-space radius was computed as:

$$r = \frac{\|\mathbf{s}\|}{c_z} \cdot \bar{f} \cdot 5.5^{1/\varepsilon_3}$$

The factor $5.5^{1/\varepsilon_3}$ was derived from the $\varepsilon_3$-modified distance contour at
opacity threshold $1/255$; however, it coupled the tile-assignment radius to a learned
parameter and caused the radius to diverge for small $\varepsilon_3$.

**CUDA.**  The formula is simplified to:

$$r = \left\lceil \frac{3\,\|\mathbf{s}\|}{c_z}\,\bar{f} \right\rceil$$

mirroring 3DGS's 3-sigma convention.  The $\varepsilon_3$ dependence is removed; the constant
factor of 3 is conservative and provides a clean separation between the rendering radius
and the learned shape parameters.

### 7.4 Splat-Size Pruning Threshold

During training, splats that would cover more than a fraction of the scene are split into
smaller copies.  The threshold for this split operation previously used the same
$5.5^{1/\varepsilon_3}$ factor:

$$\|\mathbf{s}\|\cdot 5.5^{1/\varepsilon_3} > \text{extent}$$

For $\varepsilon_3 = 0.5$ this evaluates to $\|\mathbf{s}\|\cdot 30.25 > \text{extent}$, triggering
splits of splats whose bare scale was only $\text{extent}/30$. The threshold is updated to
match the renderer formula:

$$3\,\|\mathbf{s}\| > \text{extent}$$

### 7.5 Performance

The Python renderer processes splats in batched tensor operations over a Python `for`
loop; each iteration dispatches O(L) CUDA kernels.  The CUDA renderer processes all tiles
in a single kernel launch.  The result is a speedup of approximately 100× at half
resolution for a scene with ${\sim}10^5$ splats.

---

## 8. Learnable Parameters and Activations

For completeness, the full parameter set of each splat and the activation functions applied
before the render kernel are listed below.

| Parameter | Raw shape | Activation | Range | GS equivalent |
|---|---|---|---|---|
| Position $\mathbf{x}$ | $(N,3)$ | identity | $\mathbb{R}^3$ | position |
| Rotation $\mathbf{q}$ | $(N,4)$ | $\ell_2$-normalise | $\|\mathbf{q}\|=1$ | rotation |
| Scale $\mathbf{s}$ | $(N,3)$ | $\exp$ | $(0,\infty)^3$ | scale |
| Opacity $o$ | $(N,1)$ | $\sigma$ | $(0,1)$ | opacity |
| SH coefficients | $(N,C)$ | identity | $\mathbb{R}^C$ | features |
| Lateral exponents $(\varepsilon_1, \varepsilon_2)$ | $(N,2)$ | $\sigma \cdot 1.8 + 0.1$ | $(0.1,\, 1.9)$ | — |
| Outer exponent $\varepsilon_3$ | $(N,1)$ | $\sigma + 0.5$ | $(0.5,\, 1.5)$ | — |

The quaternion convention is $[w, x, y, z]$ (index 0 = scalar), which is remapped to
$[x, y, z, w]$ (index 3 = scalar) inside the `quatToMatrix` CUDA function.  The
remapping is self-consistent in both the forward and backward passes.

$\varepsilon_3$ is initialised to 1.0 (raw value 0, which gives $\sigma(0) + 0.5 = 1.0$).
The gradient of the activation at initialisation is $\sigma'(0) = 0.25$, ensuring
that gradients flow through $\varepsilon_3$ from the first training step.  The earlier
activation $\sigma(x)\cdot 4.1 + 0.9$ placed the initialisation point just below a
hard clamp at 1.0, permanently suppressing $\varepsilon_3$ gradients.
