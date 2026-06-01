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

    # org_pose = np.load(model_path + f"/pose/pose_{iter}.npy")
    org_pose = np.load(model_path + f"/pose/pose_org.npy")

    org_pose = org_pose[[22, 19]]
    
    visualizer(org_pose, ["green" for _ in org_pose], model_path + "/pose/poses_optimized.png")
    # n_interp = int(10 * 30 / n_views)  # 10second, fps=30
    n_interp = int(10 * 30 / n_views)  # 10second, fps=30
    all_inter_pose = []
    for i in range(n_views-1):
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
    """
    Convert images in a folder to a video.

    Args:
    - image_folder (str): The path to the folder containing the images.
    - output_video_path (str): The path where the output video will be saved.
    - fps (int): Frames per second for the output video.
    """
    images = []

    # for filename in sorted(os.listdir(image_folder)):
    #     if filename.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
    #         image_path = os.path.join(image_folder, filename)
    #         image = imageio.imread(image_path)
    #         images.append(image)

    # imageio.mimwrite(output_video_path, images, fps=fps)

    import cv2
    Filenames = sorted(os.listdir(image_folder))
    # print(image_folder + '/' + Filenames[0])
    Frame = cv2.imread(image_folder + '/' + Filenames[0], cv2.IMREAD_GRAYSCALE).shape[::-1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    Video = cv2.VideoWriter(output_video_path, fourcc, fps=30, frameSize=Frame)
        
    for File in Filenames:
        Video.write(cv2.imread(image_folder + '/' + File))
            
    Video.release()
    return

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

        scene.gaussians._exp12 = scene.gaussians._exp12 * 0 + 1
        scene.gaussians._exp3 = scene.gaussians._exp3 * 0 + -3.688879454216
        scene.gaussians._xyz += (torch.rand_like(scene.gaussians._xyz) - 0.5)*0.0005
        
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
        # image_folder = os.path.join(args.model_path, f'interp/ours_{args.iteration}/renders')
        # image_folder = os.path.join(args.model_path, f'interp/ours_None/renders')
        image_folder = os.path.join(args.results, f'interp/')
        # print(image_folder)
        # output_video_file = os.path.join(args.results, f'{args.scene}_{args.n_views}_{args.splat_type}_view.mp4')
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
