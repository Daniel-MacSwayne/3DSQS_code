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


def save_interpolate_pose(model_path, iter, n_views):
    # save_interpolate_pose: Interpolate between training camera poses for smooth video.
    #
    # Steps:
    #   1. Load optimised poses from disk
    #   2. Evenly subsample to n_views keyframes (avoids hardcoded indices)
    #   3. Interpolate between consecutive pairs at ~30fps for 10 seconds per pair
    #   4. Save interpolated pose sequence to pose_interpolated.npy

    org_pose = np.load(model_path + f"/pose/pose_org.npy")   # (N, 4, 4)

    # Step 2: evenly subsample to n_views keyframes across the full pose sequence
    n_total   = len(org_pose)
    indices   = np.linspace(0, n_total - 1, min(n_views, n_total), dtype=int)
    org_pose  = org_pose[indices]

    visualizer(org_pose, ["green" for _ in org_pose], model_path + "/pose/poses_optimized.png")

    # Step 3: interpolate between consecutive keyframe pairs
    n_interp = max(2, int(10 * 30 / max(len(org_pose) - 1, 1)))  # frames per segment
    all_inter_pose = []
    for i in range(len(org_pose) - 1):
        tmp_inter_pose = generate_interpolated_path(poses=org_pose[i:i+2], n_interp=n_interp)
        all_inter_pose.append(tmp_inter_pose)
    all_inter_pose = np.array(all_inter_pose).reshape(-1, 3, 4)

    inter_pose_list = []
    for p in all_inter_pose:
        tmp_view = np.eye(4)
        tmp_view[:3, :3] = p[:3, :3]
        tmp_view[:3, 3] = p[:3, 3]
        inter_pose_list.append(tmp_view)
    inter_pose = np.stack(inter_pose_list, 0)
    visualizer(inter_pose, ["blue" for _ in inter_pose], model_path + "/pose/poses_interpolated.png")
    np.save(model_path + "/pose/pose_interpolated.npy", inter_pose)


def images_to_video(image_folder, output_video_path, fps=30):
    """images_to_video: Compile PNG frames in a folder into an MP4.

    Steps:
      1. Collect and sort PNG files in image_folder (skip subdirectories)
      2. Read the first frame to get dimensions
      3. Write all frames to an mp4v VideoWriter
    """
    import cv2

    # Step 1: sorted PNG files only (skip subdirectories like depth/)
    Filenames = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        and os.path.isfile(os.path.join(image_folder, f))
    ])
    if not Filenames:
        print(f"[images_to_video] No image files found in {image_folder}")
        return

    # Step 2: get frame dimensions from first image
    first_path = os.path.join(image_folder, Filenames[0])
    first = cv2.imread(first_path)
    if first is None:
        print(f"[images_to_video] Could not read {first_path}")
        return
    h, w = first.shape[:2]
    frameSize = (w, h)

    # Step 3: write all frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    Video  = cv2.VideoWriter(output_video_path, fourcc, fps, frameSize)
    for File in Filenames:
        frame = cv2.imread(os.path.join(image_folder, File))
        if frame is not None:
            Video.write(frame)
    Video.release()
    print(f"Video saved: {output_video_path}  ({len(Filenames)} frames @ {fps}fps)")

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args):
    # render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_path = os.path.join(args.results, f"interp/render")
    depth_path = os.path.join(args.results, f"interp/depth")
    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1)).to(device='cpu')

        if gaussians.splat_type == 'GS':
            render_pkg = render(view, gaussians, pipeline, background, camera_pose=camera_pose)
        elif gaussians.splat_type in ['GSE', 'SQ', 'SQE']:
            render_pkg = render2(view, gaussians, pipeline, background, camera_pose=camera_pose)
        rendering = render_pkg["render"]
        depth = render_pkg["depth"]
        
        # rendering = render(
            # view, gaussians, pipeline, background, camera_pose=camera_pose
        # )["render"]

        
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )


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
    
    
    # Applying interpolation
    # save_interpolate_pose(dataset.model_path, iteration, args.n_views)
    save_interpolate_pose(dataset.model_path, iteration, 2)

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
        # Render images are saved to interp/render/ — point video compiler there
        image_folder      = os.path.join(args.results, 'interp', 'render')
        output_video_file = os.path.join(args.results, f'{args.scene}_{args.n_views}_{args.splat_type}_view.mp4')
        images_to_video(image_folder, output_video_file, fps=30)


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
