#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render2
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, GaussianModel2
from utils.pose_utils import get_tensor_from_camera
from utils.camera_utils import generate_interpolated_path
from utils.camera_utils import visualizer
import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL import Image as PILImage


def save_interpolate_pose(model_path, iter):  # n_views unused — orbit ignores keyframe count
    # save_interpolate_pose: Generate a smooth circular orbit around the scene.
    #
    # Steps:
    #   1. Load optimised poses and extract camera centres
    #   2. PCA of camera centres → find the dominant orbit plane (v1, v2) and orbit axis (v3)
    #   3. Generate exactly 300 poses equally spaced around the orbit circle
    #   4. For each pose: camera sits at orbit_pos, looks at scene_centre, down ≈ v3
    #   5. Save W2C pose sequence to pose_interpolated.npy

    org_pose = np.load(model_path + f"/pose/pose_org.npy")   # (N, 4, 4) W2C

    # Step 1: extract camera centres from W2C matrices
    # camera_centre = -R_w2c.T @ T_w2c
    centers     = np.array([-(p[:3,:3].T @ p[:3,3]) for p in org_pose])
    scene_ctr   = centers.mean(0)

    # Step 2: PCA → orbit plane (v1, v2) + orbit axis (v3)
    centered    = centers - scene_ctr
    _, _, Vt    = np.linalg.svd(centered)
    v1, v2, v3  = Vt[0], Vt[1], Vt[2]

    # Orient v3 to match the average camera "down" direction so images are upright.
    # In OpenCV convention, camera Y = down in camera space → column 1 of C2W = down in world.
    R_c2ws      = org_pose[:,:3,:3].transpose(0,2,1)         # C2W rotation (N,3,3)
    avg_down    = R_c2ws[:,:,1].mean(0)                      # average world "down" direction
    if np.dot(v3, avg_down) < 0:
        v3 = -v3                                              # flip to match camera orientation

    # Orbit radius: median distance from scene centre projected onto the orbit plane
    radius      = np.median(np.linalg.norm(centered @ np.stack([v1, v2], axis=1), axis=1))

    # Step 3: 300 equally-spaced angles
    N           = 300
    thetas      = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # Step 4: build W2C matrix for each orbit position
    orbit_poses = []
    for theta in thetas:
        # Camera position on the orbit circle
        pos     = scene_ctr + radius * (np.cos(theta) * v1 + np.sin(theta) * v2)

        # Camera Z = forward, looking at scene centre
        cam_z   = scene_ctr - pos
        cam_z  /= np.linalg.norm(cam_z)

        # Camera Y = "down" in OpenCV convention; use orbit axis v3, projected ⊥ cam_z
        cam_y   = v3 - np.dot(v3, cam_z) * cam_z
        cam_y  /= np.linalg.norm(cam_y)

        # Camera X = right = down × forward (right-handed, gives correct handedness)
        cam_x   = np.cross(cam_y, cam_z)
        cam_x  /= np.linalg.norm(cam_x)

        # Assemble W2C: R_c2w columns are [right, down, fwd] in world space
        R_c2w   = np.stack([cam_x, cam_y, cam_z], axis=1)   # (3,3)
        R_w2c   = R_c2w.T
        T_w2c   = -R_w2c @ pos

        pose_w2c            = np.eye(4)
        pose_w2c[:3, :3]    = R_w2c
        pose_w2c[:3,  3]    = T_w2c
        orbit_poses.append(pose_w2c)

    inter_pose = np.stack(orbit_poses, 0)                    # (300, 4, 4)
    # Step 5: save
    # visualizer(inter_pose, ["blue" for _ in inter_pose], model_path + "/pose/poses_interpolated.png")
    np.save(model_path + "/pose/pose_interpolated.npy", inter_pose)
    print(f"  Orbit trajectory: {N} poses, radius={radius:.3f}, centre={np.round(scene_ctr,3)}")


def images_to_video(image_folder, output_video_path, fps=30):
    """images_to_video: Compile PNG/JPG frames in a folder into an H.264 MP4.

    Steps:
      1. Collect and sort image files in image_folder (skip subdirectories)
      2. Read frames as RGB numpy arrays via cv2
      3. Write H.264 MP4 via imageio + pyav (libx264)
    """
    # Step 1: sorted image files only (skip subdirectories like depth/)
    filenames = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        and os.path.isfile(os.path.join(image_folder, f))
    ])
    if not filenames:
        print(f"[images_to_video] No image files found in {image_folder}")
        return

    # Step 2: load as RGB (imageio expects RGB; cv2 reads BGR)
    frames = []
    for f in filenames:
        img = cv2.imread(os.path.join(image_folder, f))
        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not frames:
        print(f"[images_to_video] Could not read any frames from {image_folder}")
        return

    # Step 3: write H.264 MP4 via imageio + pyav
    imageio.mimsave(output_video_path, frames, fps=fps, codec='libx264')
    h, w = frames[0].shape[:2]
    print(f"Video saved: {output_video_path}  ({len(frames)} frames @ {fps}fps  {w}×{h})")

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args):
    render_path = os.path.join(args.results, "interp/render")
    depth_path  = os.path.join(args.results, "interp/depth")
    makedirs(render_path, exist_ok=True)
    makedirs(depth_path,  exist_ok=True)

    # views is the smooth interpolated path loaded by the scene (get_video=True path).
    # Clamp to exactly 300 frames so the video is always 10 s at 30 fps.
    views = views[:300]

    cmap = plt.get_cmap('viridis')

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1)).to(device='cpu')

        if gaussians.splat_type == 'GS':
            render_pkg = render(view, gaussians, pipeline, background, camera_pose=camera_pose)
        elif gaussians.splat_type in ['GSE', 'SQ', 'SQE']:
            render_pkg = render2(view, gaussians, pipeline, background, camera_pose=camera_pose)
        rendering = render_pkg["render"]
        depth     = render_pkg["depth"]

        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))

        # Depth — viridis colourmap, yellow=near, purple=far, black=no geometry
        d_np = depth.squeeze().detach().cpu().numpy()
        mask = d_np > 0
        d_norm = np.zeros_like(d_np)
        if mask.any():
            lo, hi = d_np[mask].min(), d_np[mask].max()
            if hi > lo:
                d_norm[mask] = 1.0 - (d_np[mask] - lo) / (hi - lo)
            else:
                d_norm[mask] = 0.5
        depth_rgb = (cmap(d_norm)[:, :, :3] * 255).astype(np.uint8)
        depth_rgb[~mask] = 0
        PILImage.fromarray(depth_rgb).save(os.path.join(depth_path, f"{idx:05d}.png"))


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    args,
):

    splat_type = args.splat_type
    dtype = torch.float32
    device = args.device
    
    
    # save_interpolate_pose(dataset.model_path, iteration, args.n_views)
    save_interpolate_pose(dataset.model_path, iteration)

    with torch.no_grad():

        if dataset.splat_type == 'GS':
            gaussians = GaussianModel(dataset.sh_degree, dtype)
        elif dataset.splat_type in ['GSE', 'SQ', 'SQE']:
            gaussians = GaussianModel2(dataset.sh_degree, dtype, max_splats=200000, device=device)
        # gaussians = GaussianModel(dataset.sh_degree)
        # scene = Scene(dataset, gaussians, load_iteration=iteration, opt=args, shuffle=False)
        scene = Scene(dataset, gaussians, opt=args, shuffle=False)

        scene.gaussians.load_ply(args.results + '/model.ply')
        gaussians.splat_type = dataset.splat_type
        # Note: do NOT modify _exp12/_exp3/_xyz here — render the model as-is
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # render interpolated views
    # render_set(
    #     dataset.model_path,
    #     "interp",
    #     scene.loaded_iter,
    #     scene.getTrainCameras(),
    #     gaussians,
    #     pipeline,
    #     background,
    # )

    render_set(
        dataset.model_path,
        "interp",
        scene.loaded_iter,
        scene.getTrainCameras(),
        gaussians,
        pipeline,
        background,
        args
    )

    if args.get_video:
        stem = f'{args.scene}_{args.splat_type}'
        images_to_video(
            os.path.join(args.results, 'interp', 'render'),
            os.path.join(args.results, f'{stem}_rgb.mp4'),
            fps=30,
        )
        images_to_video(
            os.path.join(args.results, 'interp', 'depth'),
            os.path.join(args.results, f'{stem}_depth.mp4'),
            fps=30,
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--get_video", action="store_true")
    parser.add_argument("--n_views", default=None, type=int)
    parser.add_argument("--scene", default=None, type=str)
    parser.add_argument("--results", type=str)
    parser.add_argument("--device", type=str, default='cuda')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args,
    )
