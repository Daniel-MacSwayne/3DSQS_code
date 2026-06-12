"""
train_all.py: Sequential training over all Deep_Blending and Mip_nerf_360 scenes.

Run directly:   python scripts/train_all.py
Or via nohup:   nohup python scripts/train_all.py > logs/train_all.log 2>&1 &

Progress is written to logs/train_all.log and logs/train_all_summary.csv.
Each scene's full stdout/stderr goes to logs/{Dataset}_{Scene}.log.
"""

import subprocess
import os
import sys
import csv
import time
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
PATH          = '/home/daniel/Documents/Projects/'
PROJECT_PATH  = PATH + 'Sync/3D/'
TRAIN_SCRIPT  = PROJECT_PATH + '3DSQS/train_joint.py'
DATASET_ROOT  = PATH + 'Datasets/Static/'
RESULTS_ROOT  = PROJECT_PATH + 'Results/'
LOG_DIR       = PROJECT_PATH + '3DSQS/logs/'
os.makedirs(LOG_DIR, exist_ok=True)

# ── Training hyper-parameters (shared across all scenes) ───────────────────────
SPLAT_TYPE = 'SQE'
ITERATIONS = 10000
DTYPE      = 'fp32'
MAX_SPLATS = 1_000_000
DEVICE     = 'cuda'

# ── Scene lists ─────────────────────────────────────────────────────────────────
# Format: (Scene_name, n_views, resolution)
# Scene_name must match the actual folder name on disk.

DEEP_BLENDING_SCENES = [
    ("Aquarium-20",   19, 4),
    ("Bedroom",       30, 2),
    ("Boats",         30, 8),
    ("Bridge",        30, 2),
    ("CreepyAttic",   30, 2),
    ("DrJohnson",     24, 2),
    ("Hugo-1",        24, 2),
    ("Library",       30, 8),
    ("Lumber",        30, 8),
    ("Museum-1",      27, 4),
    ("Museum-2",      30, 4),
    ("NightSnow",     30, 4),
    ("Playroom",      30, 2),
    ("Ponche",        30, 4),
    ("SaintAnne",     30, 4),
    ("Shed",          30, 8),
    ("Street-10",     13, 4),
    ("Tree-18",       18, 4),
    ("Yellowhouse-12",12, 2),
]

# Mip_nerf_360 folders are lowercase; n_views=30, resolution=1 for all.
MIP_NERF_360_SCENES = [
    # Use images_8/ sub-folder (pre-downscaled 8×) — full-res images are 4-5K
    # and would need 37+ GB GPU just to load. images_8 needs ~600 MB.
    ("bicycle",   30, 1),
    ("bonsai",    30, 1),
    ("counter",   30, 1),
    ("flowers",   30, 1),
    ("garden",    30, 1),
    ("kitchen",   30, 1),
    ("room",      30, 1),
    ("stump",     30, 1),
    ("treehill",  30, 1),
]

# ── Build the list of jobs ──────────────────────────────────────────────────────
# Each job: (dataset_label, scene_path, output_path, scene_name, n_views, res)

def build_jobs():
    jobs = []

    for scene, n_views, res in DEEP_BLENDING_SCENES:
        scene_path  = f'{DATASET_ROOT}deep_blending/{scene}/colmap/'
        output_path = f'{RESULTS_ROOT}Deep_Blending/{scene}/{SPLAT_TYPE}'
        if os.path.exists(scene_path):
            jobs.append(('Deep_Blending', scene_path, output_path, scene, n_views, res))
        else:
            print(f'[SKIP] Deep_Blending/{scene}: {scene_path} not found')

    for scene, n_views, res in MIP_NERF_360_SCENES:
        scene_path  = f'{DATASET_ROOT}mip_nerf_360/{scene}/'
        output_path = f'{RESULTS_ROOT}Mip_nerf_360/{scene}/{SPLAT_TYPE}'
        if os.path.exists(scene_path):
            jobs.append(('Mip_nerf_360', scene_path, output_path, scene, n_views, res))
        else:
            print(f'[SKIP] Mip_nerf_360/{scene}: {scene_path} not found')

    return jobs


# ── Run a single training job ───────────────────────────────────────────────────

def run_job(dataset, scene_path, output_path, scene, n_views, res):
    """Run one scene; stream output to its per-scene log file. Returns exit code."""
    os.makedirs(output_path, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f'{dataset}_{scene}.log')

    cmd = [
        sys.executable, TRAIN_SCRIPT,
        '-s', scene_path,
        '-m', output_path,
        '--scene',       scene,
        '--n_views',     str(n_views),
        '--iter',        str(ITERATIONS),
        '--optim_pose',
        '--results',     output_path,
        '--splat_type',  SPLAT_TYPE,
        '--resolution',  str(res),
        '--dtype',       DTYPE,
        '--max_splats',  str(MAX_SPLATS),
        '--step',        '0',
        '--device',      DEVICE,
    ]

    # Mip_nerf_360 scenes: use images_8/ to avoid loading 4-5K full-res images
    if dataset == 'Mip_nerf_360':
        cmd += ['--images', 'images_8']

    print(f'\n{"="*70}')
    print(f'[{datetime.now().strftime("%H:%M:%S")}] START  {dataset}/{scene}')
    print(f'  scene_path:  {scene_path}')
    print(f'  output_path: {output_path}')
    print(f'  log:         {log_path}')
    print(f'{"="*70}')
    sys.stdout.flush()

    t0 = time.time()
    with open(log_path, 'w') as log_f:
        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)

    elapsed = time.time() - t0
    status  = 'OK' if proc.returncode == 0 else f'FAILED (exit {proc.returncode})'
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {status}  {dataset}/{scene}  ({elapsed/60:.1f} min)')
    sys.stdout.flush()

    return proc.returncode, elapsed


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    jobs = build_jobs()
    total = len(jobs)
    print(f'\nTraining {total} scenes  ({ITERATIONS} iters each, {MAX_SPLATS:,} max splats)')
    print(f'Logs → {LOG_DIR}')
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

    summary_path = os.path.join(LOG_DIR, 'train_all_summary.csv')
    with open(summary_path, 'w', newline='') as sf:
        writer = csv.writer(sf)
        writer.writerow(['Dataset', 'Scene', 'Status', 'ExitCode', 'Minutes'])

        for idx, (dataset, scene_path, output_path, scene, n_views, res) in enumerate(jobs, 1):
            print(f'\n--- Job {idx}/{total} ---')
            sys.stdout.flush()

            retcode, elapsed = run_job(dataset, scene_path, output_path, scene, n_views, res)
            status = 'OK' if retcode == 0 else 'FAILED'
            writer.writerow([dataset, scene, status, retcode, f'{elapsed/60:.1f}'])
            sf.flush()

    print(f'\n\nAll done. Summary → {summary_path}')
    print(f'Finished: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
