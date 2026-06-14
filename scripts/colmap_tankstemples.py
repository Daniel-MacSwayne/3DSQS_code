"""colmap_tankstemples.py: Run COLMAP point triangulation for all Tanks & Temples scenes.

For each scene the pipeline is:
  1. Feature extraction  (SIFT, PINHOLE camera from cam file intrinsics)
  2. Exhaustive feature matching
  3. Convert cam/*.txt poses → COLMAP cameras.txt + images.txt
  4. Triangulate 3D points with known poses → sparse/0/points3D.bin
  5. Convert text model → binary model

Output: <scene>/sparse/0/{cameras.bin, images.bin, points3D.bin}
The scene loader in scene/__init__.py picks up the sparse/ folder automatically.
"""

import os
import subprocess
import sqlite3
import struct
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path

COLMAP   = "/home/daniel/anaconda3/envs/3DSQS/bin/colmap"
ROOT     = Path("/home/daniel/Documents/Projects/Datasets/Static/tanks_temples/intermediate")
SCENES   = ["Family", "Francis", "Horse", "Lighthouse", "M60", "Panther", "Playground", "Train"]


# ── COLMAP binary write helpers ───────────────────────────────────────────────

def write_cameras_bin(path, cameras):
    """Write COLMAP cameras.bin.
    cameras: list of (camera_id, model_id, width, height, params[])
    PINHOLE model_id = 1, params = [fx, fy, cx, cy]
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam_id, model_id, w, h, params in cameras:
            f.write(struct.pack("<i", cam_id))
            f.write(struct.pack("<i", model_id))
            f.write(struct.pack("<Q", w))
            f.write(struct.pack("<Q", h))
            for p in params:
                f.write(struct.pack("<d", p))


def write_images_bin(path, images):
    """Write COLMAP images.bin.
    images: list of (image_id, qvec[4], tvec[3], camera_id, name)
    qvec = [qw, qx, qy, qz]
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img_id, qvec, tvec, cam_id, name in images:
            f.write(struct.pack("<i", img_id))
            for v in qvec:
                f.write(struct.pack("<d", v))
            for v in tvec:
                f.write(struct.pack("<d", v))
            f.write(struct.pack("<i", cam_id))
            name_bytes = name.encode() + b"\x00"
            f.write(name_bytes)
            # No 2D points (triangulator fills these in)
            f.write(struct.pack("<Q", 0))


def write_points3D_bin(path):
    """Write empty COLMAP points3D.bin (triangulator fills it in)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 0))


# ── Per-scene cam file parser ─────────────────────────────────────────────────

def parse_cam_file(cam_path):
    """Parse an MVSNet cam.txt file.
    Returns (R_w2c (3,3), t_w2c (3,), fx, fy, cx, cy).
    Image dimensions are NOT in the cam file (last line is MVSNet depth metadata).
    """
    with open(cam_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    assert lines[0] == "extrinsic", f"Bad format: {cam_path}"
    extr = np.array([[float(v) for v in lines[i].split()] for i in range(1, 5)])
    R_w2c = extr[:3, :3]
    t_w2c = extr[:3,  3]

    ki = lines.index("intrinsic") + 1
    K  = np.array([[float(v) for v in lines[ki + i].split()] for i in range(3)])
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    return R_w2c, t_w2c, fx, fy, cx, cy


def R_to_qvec(R):
    """Rotation matrix → COLMAP quaternion [qw, qx, qy, qz]."""
    q = Rotation.from_matrix(R).as_quat()   # scipy: [x, y, z, w]
    return [q[3], q[0], q[1], q[2]]         # COLMAP: [w, x, y, z]


# ── Main per-scene pipeline ───────────────────────────────────────────────────

def run(cmd, label):
    print(f"\n  [{label}] {' '.join(str(c) for c in cmd[:4])} ...")
    result = subprocess.run([str(c) for c in cmd])
    if result.returncode != 0:
        raise RuntimeError(f"COLMAP step failed: {label}")


def process_scene(scene_name):
    scene_dir   = ROOT / scene_name
    cams_dir    = scene_dir / "cams"
    images_dir  = scene_dir / "images"
    sparse_dir  = scene_dir / "sparse" / "0"
    database    = scene_dir / "database.db"

    sparse_dir.mkdir(parents=True, exist_ok=True)

    cam_files = sorted(f for f in os.listdir(cams_dir) if f.endswith("_cam.txt"))
    if not cam_files:
        raise FileNotFoundError(f"No cam files in {cams_dir}")

    print(f"\n{'='*60}")
    print(f"Scene: {scene_name}  ({len(cam_files)} cameras)")
    print(f"{'='*60}")

    # ── 1. Read intrinsics from first cam (shared across all frames) ──────────
    _, _, fx, fy, cx, cy = parse_cam_file(cams_dir / cam_files[0])
    # Read image size from the first image file
    from PIL import Image as PILImage
    first_img = images_dir / (cam_files[0].replace("_cam.txt", "") + ".jpg")
    W, H = PILImage.open(first_img).size
    print(f"  Intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}  {W}×{H}")

    # ── 2. Feature extraction ─────────────────────────────────────────────────
    if database.exists():
        print("  database.db exists — skipping feature extraction")
    else:
        run([COLMAP, "feature_extractor",
             "--database_path", database,
             "--image_path",    images_dir,
             "--ImageReader.camera_model",  "PINHOLE",
             "--ImageReader.single_camera", "1",
             "--ImageReader.camera_params", f"{fx},{fy},{cx},{cy}",
             "--FeatureExtraction.use_gpu", "1",
        ], "feature_extractor")

    # ── 3. Exhaustive matching ────────────────────────────────────────────────
    # Check if matches already exist
    conn = sqlite3.connect(database)
    n_matches = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    conn.close()
    if n_matches > 0:
        print(f"  {n_matches} match rows already in database — skipping matching")
    else:
        run([COLMAP, "exhaustive_matcher",
             "--database_path", database,
             "--ExhaustiveMatching.block_size", "50",
        ], "exhaustive_matcher")

    # ── 4. Read image IDs from database ──────────────────────────────────────
    conn = sqlite3.connect(database)
    rows = conn.execute("SELECT image_id, name FROM images ORDER BY image_id").fetchall()
    db_cam_rows = conn.execute("SELECT camera_id FROM cameras ORDER BY camera_id").fetchall()
    conn.close()

    name_to_db_id = {name: img_id for img_id, name in rows}
    db_camera_id  = db_cam_rows[0][0]  # single shared camera

    print(f"  Database: {len(rows)} images, camera_id={db_camera_id}")

    # ── 5. Build COLMAP model with known poses ────────────────────────────────
    cameras_bin = sparse_dir / "cameras.bin"
    images_bin  = sparse_dir / "images.bin"
    points_bin  = sparse_dir / "points3D.bin"

    colmap_cameras = [(db_camera_id, 1, W, H, [fx, fy, cx, cy])]  # model_id 1 = PINHOLE

    colmap_images = []
    missing = []
    for cam_file in cam_files:
        stem    = cam_file.replace("_cam.txt", "")
        img_name = stem + ".jpg"
        if img_name not in name_to_db_id:
            missing.append(img_name)
            continue
        img_id = name_to_db_id[img_name]
        R_w2c, t_w2c, *_intrinsics = parse_cam_file(cams_dir / cam_file)
        qvec = R_to_qvec(R_w2c)
        colmap_images.append((img_id, qvec, list(t_w2c), db_camera_id, img_name))

    if missing:
        print(f"  WARNING: {len(missing)} cam files have no matching image in DB: {missing[:3]}")

    colmap_images.sort(key=lambda x: x[0])  # sort by image_id for COLMAP

    write_cameras_bin(cameras_bin, colmap_cameras)
    write_images_bin(images_bin,   colmap_images)
    write_points3D_bin(points_bin)
    print(f"  Wrote sparse/0/ ({len(colmap_images)} images)")

    # ── 6. Point triangulation ────────────────────────────────────────────────
    run([COLMAP, "point_triangulator",
         "--database_path", database,
         "--image_path",    images_dir,
         "--input_path",    sparse_dir,
         "--output_path",   sparse_dir,
         "--Mapper.num_threads", "16",
         "--Mapper.init_min_tri_angle", "4",
    ], "point_triangulator")

    # ── 7. Report ─────────────────────────────────────────────────────────────
    conn = sqlite3.connect(database)
    n_pts = struct.unpack("<Q", open(points_bin, "rb").read(8))[0]
    conn.close()
    print(f"\n  Done: sparse/0/ has {n_pts:,} 3D points")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=SCENES,
                        help="Scenes to process (default: all 8)")
    args = parser.parse_args()

    failures = []
    for scene in args.scenes:
        try:
            process_scene(scene)
        except Exception as e:
            print(f"\n[FAILED] {scene}: {e}")
            failures.append((scene, str(e)))

    print(f"\n{'='*60}")
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for scene, err in failures:
            print(f"  {scene}: {err}")
    else:
        print(f"All {len(args.scenes)} scenes completed successfully.")
