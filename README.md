# 3DSQS

Sparse-view 3D reconstruction that fits **superquadrics** (instead of, or alongside, standard 3D Gaussians) using a DUSt3R/MASt3R-based coarse geometric initialization and joint pose/scene optimization.

This codebase is built on top of [NVlabs/InstantSplat](https://github.com/NVlabs/InstantSplat), extended with a custom superquadric splatting rasterizer (`submodules/diff-superquadric-rasterization`) alongside the original 3D Gaussian rasterizer (`submodules/diff-gaussian-rasterization`). See `RENDERER_INTERNALS.md` and `CUDA_RENDERER_ADDENDUM.md` for details on the custom renderer.

## Pipeline overview

1. **Coarse geometric initialization** — DUSt3R/MASt3R infers per-image point clouds and camera poses from a handful of unposed images.
2. **Joint training** — `train_joint.py` optimizes the scene representation (Gaussians/superquadrics) and camera poses together.
3. **Render / evaluate** — render novel/interpolated views and video, or compute metrics against ground-truth poses.

## Installation

Requires a CUDA-capable GPU (Linux recommended; the project also runs under WSL). The steps below mirror `Dockerfile` in this repo.

### 1. Clone with submodules

```bash
git clone --recursive <this-repo-url>
cd 3DSQS_code
git submodule update --init --recursive
```

### 2. Create the environment

```bash
conda create -n instantsplat python=3.10 -y
conda activate instantsplat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # match your CUDA version
pip install -r requirements.txt
```

### 3. Build the CUDA submodules

```bash
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
pip install submodules/diff-superquadric-rasterization
pip install plyfile
```

> The upstream InstantSplat Dockerfile patches `diff-gaussian-rasterization`'s near-plane clip
> (`p_view.z <= 0.2f` → `p_view.z <= 0.001f` in `cuda_rasterizer/auxiliary.h`) before building, to
> avoid over-culling points close to the camera in sparse-view setups. Apply the same patch if you
> hit excessive point culling.

Set `TORCH_CUDA_ARCH_LIST` to match your GPU's compute capability before building if needed, e.g.:

```bash
export TORCH_CUDA_ARCH_LIST="8.6+PTX"
```

### 4. Build CroCo's RoPE CUDA kernel (used by DUSt3R/MASt3R)

```bash
cd submodules/dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd -
```

### 5. Download DUSt3R checkpoints

```bash
mkdir -p submodules/dust3r/checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
    -P submodules/dust3r/checkpoints
```

By default `coarse_init_infer.py`/`coarse_init_eval.py` expect the checkpoint at `./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` (see `--model_path`) — either copy/symlink it there or pass `--model_path` explicitly.

### 6. System packages (for rendering/video)

```bash
apt-get install -y libgl1-mesa-dev libglib2.0-0
```

### Docker (alternative)

`Dockerfile` in this repo performs all of the above in a single image based on `pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel`. Note it currently clones InstantSplat from GitHub rather than using this repo's code — treat it as a reference for the install steps rather than a ready-to-use image for 3DSQS.

## Data layout

Top-level convention used by the batch scripts (`scripts/run_train_infer.sh`, `scripts/run_train_eval.sh`):

```
data/<dataset>/<scene>/<N>_views/
```

e.g. `data/sora/santorini/3_views/`. Set `DATA_ROOT_DIR` at the top of those scripts to wherever `data/` lives.

**What actually has to exist before you run anything** is just your input images, dropped in an `images/` subfolder of that view directory:

```
data/<dataset>/<scene>/<N>_views/
└── images/
    ├── 0001.jpg
    ├── 0002.jpg
    └── ...
```

That's it — no poses, no COLMAP files, no depth. `coarse_init_infer.py` (stage 1) reads `<img_base_path>/images/*` and runs DUSt3R to *produce* the poses/point cloud, writing them out as a COLMAP-format sparse model as a sibling directory:

```
data/<dataset>/<scene>/<N>_views/
├── images/                     # your input images (what you provide)
└── dust3r/
    ├── images -> ../images     # symlink, so the scene loader finds them
    └── sparse/0/
        ├── cameras.txt
        ├── images.txt          # DUSt3R-estimated poses
        ├── points3D.ply
        ├── pts_4_3dgs_all.npy
        └── focal.npy
```

`train_joint.py`'s `-s` argument (stage 2) should then point at that `dust3r/` folder (`data/<dataset>/<scene>/<N>_views/dust3r`), since it's the one containing `sparse/0/`. `-m` is just an output directory for the trained model/checkpoints — it's created if it doesn't exist.

For the **evaluation** pipeline (`run_train_eval.sh`), an extra `GT_POSE_PATH` (ground-truth COLMAP poses, e.g. from Tanks & Temples) is required for `metrics.py` to compare against. See `scripts/colmap_tankstemples.py` for building COLMAP-format ground truth from Tanks & Temples raw data (camera `.txt` files → `sparse/0/{cameras,images,points3D}.bin`) — note the paths at the top of that script (`COLMAP`, `ROOT`, `SCENES`) are hardcoded to the original author's machine and need editing before use. `learn_square.py`, `Video.ipynb`, and `Train.ipynb` are exploratory/dataset-prep notebooks, not required for the core pipeline.

## Running

### Option A — batch scripts (recommended)

1. Open `scripts/run_train_infer.sh` (no ground-truth poses needed — just renders a video) or `scripts/run_train_eval.sh` (computes metrics against ground truth).
2. Edit the config block at the top: `GPU_ID`, `DATA_ROOT_DIR`, `DATASETS`, `SCENES`, `N_VIEWS`, `gs_train_iter` (and `GT_POSE_PATH`'s source location, for the eval script).
3. Make sure your images are laid out as described above under each `data/<dataset>/<scene>/<N>_views/images/`.
4. Run from the repo root:

```bash
bash scripts/run_train_infer.sh
# or
bash scripts/run_train_eval.sh
```

Each iterates over every `DATASET`/`SCENE`/`N_VIEW` combination configured, running the full pipeline for each.

### Option B — individual stages

Useful for debugging a single stage, or datasets not in the `data/<dataset>/<scene>/<N>_views/` layout. See `Commands.txt` for real example invocations.

**Inference pipeline** (no ground truth, just reconstruct + render a video):

```bash
# 1. Coarse geometric initialization — reads <img_base_path>/images/*, writes <scene_root>/dust3r/sparse/0/*
python coarse_init_infer.py --n_views 3 --img_base_path data/<dataset>/<scene>/3_views --focal_avg

# 2. Joint training (pose + scene optimization) — note -s points at the dust3r/ output, not images/ directly
python train_joint.py -s data/<dataset>/<scene>/3_views/dust3r -m output/<scene>_3views \
    --n_views 3 --scene <scene_name> --iter 1000 --optim_pose

# 3. Render interpolated camera path + video
python render_by_interp.py -s data/<dataset>/<scene>/3_views/dust3r -m output/<scene>_3views \
    --n_views 3 --scene <scene_name> --iter 1000 --eval --get_video
```

Output video/renders land under the `-m` model path.

**Evaluation pipeline** (against held-out ground-truth poses):

```bash
python coarse_init_eval.py --img_base_path <path/to/24_views> --n_views <N> --focal_avg
python train_joint.py -s <path/to/24_views>/dust3r_<N>_views -m output/eval/<scene>_<N>views \
    --n_views <N> --scene <scene_name> --iter 1000 --optim_pose
python init_test_pose.py --img_base_path <path/to/24_views> --n_views <N> --focal_avg
python render.py -s <path/to/24_views>/dust3r_<N>_views -m output/eval/<scene>_<N>views \
    --n_views <N> --scene <scene_name> --optim_test_pose_iter 500 --iter 1000 --eval
python metrics.py -m output/eval/<scene>_<N>views --gt_pose_path <path/to/gt_colmap> --iter 1000 --n_views <N>
```

### Batch scripts

The full pipelines are wrapped in shell scripts under `scripts/`, confirmed present in this repo:

- **`scripts/run_train_infer.sh`** — inference-only pipeline (coarse init → joint train → render + video) over a configurable list of datasets/scenes/view-counts. Edit `GPU_ID`, `DATA_ROOT_DIR`, `DATASETS`, `SCENES`, and `N_VIEWS` at the top of the script before running.
- **`scripts/run_train_eval.sh`** — full evaluation pipeline (coarse init → joint train → test pose init → render → metrics) against ground-truth COLMAP poses. Also configured via the variables at the top of the script.

Run either with:

```bash
bash scripts/run_train_infer.sh
bash scripts/run_train_eval.sh
```

Other stage-runner scripts present in `scripts/`: `colmap_tankstemples.py` (COLMAP dataset prep), `render_test.py`, `learn_square.py`.

## Key CLI flags

- `--n_views` — number of input sparse views.
- `--focal_avg` — average focal length estimate across views during coarse init.
- `--optim_pose` — jointly optimize camera poses during training (pass to `train_joint.py`).
- `--scene`, `--iter` — scene name and training iteration count (also used to locate/name checkpoints for later stages).
- `--get_video` — render an interpolated video after training.
- `--eval` — evaluation mode (splits held-out views).

## Repo structure

- `arguments/` — argument parsing config for training/optimization.
- `gaussian_renderer/` — rendering entry points for both standard 3D Gaussians and superquadric splatting (`Superquadric_Splatting.py`), plus Plotly visualization helpers.
- `scene/` — dataset loading (COLMAP readers, camera utilities) and the scene/Gaussian model representation.
- `submodules/` — `dust3r`, `mast3r` (coarse geometric init), `diff-gaussian-rasterization`, `diff-superquadric-rasterization`, `simple-knn` (CUDA rasterizers/kernels).
- `utils/` — pose, loss, camera, and general utilities.
- `scripts/` — batch pipeline runners and dataset prep tools.
