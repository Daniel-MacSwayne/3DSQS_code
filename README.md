# 3DSQS — Superquadric Splatting

A sparse-view, pose-free scene reconstruction pipeline built on top of
[InstantSplat](https://github.com/NVlabs/InstantSplat) (DUSt3R initialisation +
joint pose/Gaussian optimisation). This fork replaces the ellipsoidal Gaussian
splat with a **superquadric splat**: each primitive has two extra learned
exponents that let it morph between a sphere, a box, a cylinder, or a sharp-edged
pillow shape, instead of always being a smooth Gaussian blob.

A custom CUDA rasterizer (`submodules/diff-superquadric-rasterization`) renders
these splats with the same tile-based architecture as `diff-gaussian-rasterization`,
so training speed is comparable to standard 3DGS.

See [`CUDA_RENDERER_ADDENDUM.md`](CUDA_RENDERER_ADDENDUM.md) and
[`RENDERER_INTERNALS.md`](RENDERER_INTERNALS.md) for the maths behind the
rasterizer, and
[`submodules/diff-superquadric-rasterization/README.md`](submodules/diff-superquadric-rasterization/README.md)
for the CUDA kernel architecture.

## Splat types

`--splat_type` (set per-dataset, see [Training](#training)) controls which
parameters are learned, going from the 3DGS baseline up to the full
superquadric model:

| `splat_type` | Cross-section (`e1`,`e2`) | Boundary sharpness (`e3`) | Equivalent to |
|---|---|---|---|
| `GS`  | fixed (sphere)            | fixed (Gaussian)          | standard 3D Gaussian Splatting |
| `GSE` | fixed (sphere)            | **learned**               | Gaussian with learned hard/soft edges |
| `SQ`  | **learned**                | fixed (Gaussian)          | superquadric cross-section, soft Gaussian falloff |
| `SQE` | **learned**                | **learned**               | full superquadric splat (this project's main method) |

Shape exponents are frozen for the first 15,000 iterations (while positions/poses
settle) and unfrozen afterwards — see `E3_UNFREEZE_STEP` in `train_joint.py`.

---

## Installation

### Tested system spec

This is the exact configuration the code is developed and trained on — a
known-good reference point if you hit build issues:

| Component | Version |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA GeForce RTX 4090 (24 GB VRAM, compute capability **8.9**) |
| NVIDIA driver | 535.309.01 |
| CUDA (via PyTorch wheel) | 11.8 |
| cuDNN | 8.7.0 |
| PyTorch / torchvision | 2.1.2+cu118 / 0.16.2+cu118 |
| Python (conda env) | 3.10 |
| RAM | 64 GB |
| CPU | 32 threads |

> **Known inconsistency**: `submodules/diff-superquadric-rasterization/setup.py`
> hardcodes `-arch=sm_86` and its comment labels this "RTX 4090 (Ada
> Lovelace)" — but the RTX 4090 is actually compute capability **8.9**
> (`sm_89`), not 8.6 (`sm_86` is Ampere, e.g. RTX 3090/A40). In practice the
> `sm_86` build still runs correctly on this 8.9 card, but if you hit CUDA
> errors specific to the superquadric rasterizer, try changing `sm_86` to
> `sm_89` in that `setup.py` and rebuilding first.

### Requirements

- Linux, NVIDIA GPU with CUDA capability ≥ 8.0
- **CUDA 11.8** (matches the reference `Dockerfile`,
  `pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel`, and the tested spec above)
- Conda (or another virtualenv tool)
- ~3 GB disk for the DUSt3R checkpoint

> If you're building on a GPU generation other than Ampere/Ada, edit
> `submodules/diff-superquadric-rasterization/setup.py` and change `sm_86`
> to match your architecture (e.g. `sm_80` for A100, `sm_89` for RTX
> 4080/4090, `sm_90` for H100) before running `pip install -e .`.

### 1. Clone with submodules

```bash
git clone --recursive https://github.com/Daniel-MacSwayne/3DSQS_code.git
cd 3DSQS_code

# if you forgot --recursive:
git submodule update --init --recursive
```

This pulls in `submodules/dust3r`, `submodules/mast3r`, `submodules/simple-knn`,
`submodules/diff-gaussian-rasterization`, and
`submodules/diff-superquadric-rasterization`.

### 2. Create the environment

```bash
conda create -n 3DSQS python=3.10 -y
conda activate 3DSQS

# Install PyTorch matching CUDA 11.8 first
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` was originally copied from upstream InstantSplat and was
missing packages this fork actually imports (`lpips`, `plyfile`, `pandas`,
`open3d`, `plotly`, `scikit-image`, `scikit-learn`, `accelerate`, `torchviz`,
`psutil`) — these have been added with comments noting what each one is for.
If you still hit an `ImportError` at training/render time, it's likely a
transitive DUSt3R/MASt3R dependency that's missing from the list — just
`pip install` it and consider sending a PR to fix `requirements.txt`.

### 4. Build the CUDA extensions

```bash
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
pip install -e submodules/diff-superquadric-rasterization
```

(Optional, speeds up DUSt3R's positional encoding:)

```bash
cd submodules/dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd -
```

### 5. Download the DUSt3R checkpoint

Needed for `--init_type dust3r` (pose-free initialisation from sparse images).
If you only use `--init_type colmap` (COLMAP poses already known), you can skip
this.

```bash
mkdir -p submodules/dust3r/checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
     -P submodules/dust3r/checkpoints/
```

### Docker alternative

A reference `Dockerfile` is included (based on the upstream InstantSplat image,
CUDA 11.8). It clones InstantSplat itself rather than this fork — treat it as a
starting point for the base image/toolchain, then layer this repo's
installation steps (3 and 4 above) on top.

---

## Usage

### Data layout

Each scene needs a folder of sparse-view images:

```
data/<dataset>/<scene>/<N>_views/
└── images/
    ├── 000.jpg
    ├── 001.jpg
    └── ...
```

### Pose initialisation (DUSt3R)

```bash
python coarse_init_infer.py --n_views <N> --img_base_path data/<dataset>/<scene>/<N>_views
```

### Training

```bash
python train_joint.py \
    -s data/<dataset>/<scene>/<N>_views \
    -m output/<scene>/<N>_views \
    --n_views <N> \
    --scene <scene> \
    --iter 1000 \
    --optim_pose \
    --init_type dust3r \
    --results output/<scene>/<N>_views
```

`splat_type` is a dataset/model parameter (`GS`/`GSE`/`SQ`/`SQE`, see the table
above) — set it in code or via the notebook workflow below.

The notebook **`scripts/Train.ipynb`** is the maintained entry point for
running full multi-scene/multi-resolution experiment sweeps (it replaced the
older `scripts/train_all.py`). The shell scripts `scripts/run_train_infer.sh`
and `scripts/run_train_eval.sh` show the equivalent CLI-only workflow
(pose-free inference vs. COLMAP-pose evaluation against ground truth).

### Rendering & video

```bash
python render_by_interp.py \
    -s data/<dataset>/<scene>/<N>_views \
    -m output/<scene>/<N>_views \
    --n_views <N> --scene <scene> --iter 1000 --eval --get_video
```

`Video.ipynb` renders an interpolated camera path through a trained model into
an MP4 (used for inspecting view-dependent artefacts such as anisotropic
splats that change shape with viewing angle).

### Evaluation

```bash
python metrics.py -m output/<scene>/<N>_views --gt_pose_path <colmap_gt_path> --n_views <N>
```

Reports L1 / PSNR / SSIM / LPIPS against held-out test views.

---

## Repository structure

```
3DSQS/
├── train_joint.py              # main training loop (SceneTrainer class)
├── render.py / render_by_interp.py
├── metrics.py                  # PSNR/SSIM/LPIPS evaluation
├── coarse_init_infer.py / coarse_init_eval.py / init_test_pose.py   # DUSt3R pose init
├── arguments/                  # ModelParams / OptimizationParams / PipelineParams
├── scene/                      # GaussianModel (3DGS) / GaussianModel2 (superquadric)
├── gaussian_renderer/          # render() / render2() entry points, Python fallback rasterizer
├── scripts/
│   ├── Train.ipynb             # main experiment-sweep notebook
│   ├── run_train_infer.sh / run_train_eval.sh
│   └── colmap_tankstemples.py
├── Video.ipynb                 # interpolated-path video rendering
├── submodules/
│   ├── dust3r/, mast3r/        # pose-free initialisation
│   ├── simple-knn/             # 3DGS k-NN for densification
│   ├── diff-gaussian-rasterization/      # upstream 3DGS CUDA rasterizer
│   └── diff-superquadric-rasterization/  # this project's CUDA rasterizer
├── CUDA_RENDERER_ADDENDUM.md   # maths/methods writeup of the CUDA renderer
└── RENDERER_INTERNALS.md       # renderer internals deep-dive
```

---

## Acknowledgements

This project builds directly on:

- [InstantSplat](https://github.com/NVlabs/InstantSplat) (NVlabs) — sparse-view,
  pose-free initialisation and joint pose/Gaussian optimisation framework this
  repo is forked from.
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
  (Inria/GRAPHDECO) — `diff-gaussian-rasterization`, `simple-knn`, and the core
  training loop this project's superquadric rasterizer mirrors.
- [DUSt3R](https://github.com/naver/dust3r) / [MASt3R](https://github.com/naver/mast3r)
  (Naver Labs) — pose-free dense stereo used for coarse initialisation.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). Note that the vendored
`diff-gaussian-rasterization`/`simple-knn` submodules carry Inria's
non-commercial research license — check `submodules/*/LICENSE.md` before any
commercial use.
