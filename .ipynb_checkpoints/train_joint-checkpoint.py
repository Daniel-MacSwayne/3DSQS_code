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

import os
import numpy as np
import torch
from torch import nn
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, render2, network_gui
import sys
from scene import Scene, GaussianModel, GaussianModel2
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2_torch
from utils.pose_utils import get_camera_from_tensor
from utils.camera_utils import generate_interpolated_path
from utils.camera_utils import visualizer
import torchvision
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
from time import perf_counter

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import lpips

# torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()
# dtype = torch.float32
# device = torch.device('cuda')
# device = torch.device('cpu')

lpips_model = lpips.LPIPS(net='alex')#.to(device=device)

from torchviz import make_dot  # Install: pip install torchviz

import psutil

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # Resident Set Size: memory in bytes
    mem_mb = mem_bytes / (1024 ** 2)       # Convert to MB
    print(f"Current memory usage: {mem_mb:.2f} MB")


def save_pose(path, quat_pose, train_cams, llffhold=2):
    output_poses=[]
    index_colmap = [cam.colmap_id for cam in train_cams]
    for quat_t in quat_pose:
        w2c = get_camera_from_tensor(quat_t)
        output_poses.append(w2c)
    colmap_poses = []
    for i in range(len(index_colmap)):
        ind = index_colmap.index(i+1)
        bb=output_poses[ind]
        bb = bb#.inverse()
        colmap_poses.append(bb)
    colmap_poses = torch.stack(colmap_poses).detach().cpu().numpy()
    np.save(path, colmap_poses)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, opt=args, shuffle=True)                                                                      
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    train_cams_init = scene.getTrainCameras().copy()
    os.makedirs(scene.model_path + '/pose', exist_ok=True)
    save_pose(scene.model_path + '/pose' + "/pose_org.npy", gaussians.P, train_cams_init)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float16, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    start = perf_counter()
    for iteration in range(first_iter, opt.iterations + 1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if args.optim_pose==False:
            gaussians.P.requires_grad_(False)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        pose = gaussians.get_RT(viewpoint_cam.uid)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()       
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))  
        loss.backward()
        # for param_group in gaussians.optimizer.param_groups:
        #     for param in param_group['params']:
        #         if param is gaussians.P:
        #             print(viewpoint_cam.uid, param.grad)
        #             break
        # print("Gradient of self.P:", gaussians.P.grad)

        iter_end.record()

        with torch.no_grad():

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                save_pose(scene.model_path + '/pose' + f"/pose_{iteration}.npy", gaussians.P, train_cams_init)

            # Densification
            # if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #     gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                
        end = perf_counter()
        train_time = end - start
    
    # We commented out log&save operations, and then calculate train time.
    # train_time = np.array(train_time)
    # print("total_test_time_epoch: ", 1)
    # print("instantsplat_train_time_mean: ", train_time.mean())
    # print("instantsplat_train_time_median: ", np.median(train_time))
    return

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(len(scene.getTrainCameras()))]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if config['name']=="train":
                        pose = scene.gaussians.get_RT(viewpoint.uid)
                    else:
                        pose = scene.gaussians.get_RT_test(viewpoint.uid)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, camera_pose=pose)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

###############################################################################

from utils.trainer import Trainer

class SceneTrainer(Trainer):
    
    def __init__(self, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    
        self.dataset = dataset
        self.opt = opt
        self.pipe = pipe
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations
        self.checkpoint_iterations = checkpoint_iterations
        self.checkpoint = checkpoint
        self.debug_from = debug_from
        self.args = args
        
        self.dtype = {"fp32":torch.float32, "fp16":torch.float16, "bfp16":torch.bfloat16}[args.dtype]
        self.device = args.device
    
        first_iter = 0
        self.tb_writer = prepare_output_and_logger(dataset)

        # print(self.dataset.splat_type)
        # sys.exit()
        
        # if self.dataset.splat_type == 'GS':
        #     self.gaussians = GaussianModel(dataset.sh_degree, self.dtype, self.args.max_splats)
        if self.dataset.splat_type in ['GS', 'GSE', 'SQ', 'SQE']:
            self.gaussians = GaussianModel2(dataset.sh_degree, self.dtype, self.args.max_splats, self.device)

        self.scene = Scene(dataset, self.gaussians, opt=args, shuffle=False)                                                                      

        if self.dataset.splat_type == 'GS':
            self.scene.gaussians._exp12.requires_grad_(False)
            self.scene.gaussians._exp3.requires_grad_(False)
        elif self.dataset.splat_type == 'GSE':
            self.scene.gaussians._exp12.requires_grad_(False)
        elif self.dataset.splat_type == 'SQ':
            self.scene.gaussians._exp3.requires_grad_(False)
        
        self.gaussians.training_setup(opt)
        if self.checkpoint:
            (model_params, first_iter) = torch.load(self.checkpoint)
            self.gaussians.restore(model_params, self.opt)        

        self.train_cams_init = self.scene.getTrainCameras().copy()
        os.makedirs(self.scene.model_path + '/pose', exist_ok=True)
        save_pose(self.scene.model_path + '/pose' + "/pose_org.npy", self.gaussians.P, self.train_cams_init)
        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=self.dtype, device=self.device)
    
        self.iter_start = torch.cuda.Event(enable_timing = True)
        self.iter_end = torch.cuda.Event(enable_timing = True)
    
        self.viewpoint_stack = None
        self.ema_loss_for_log = 0.0
        self.progress_bar = tqdm(range(first_iter, self.opt.iterations), desc="Training progress")
        first_iter += 1
        
        
        super().__init__(model=self.gaussians,
                         train_num_steps=self.opt.iterations)

        if args.step != 0:
            self.gaussians.load_ply(self.args.results + '/model.ply')
            self.step = args.step

        # total_memory = torch.cuda.get_device_properties(0).total_memory / 2**30
        # allocated_memory = torch.cuda.memory_allocated(0) / 2**30
        # reserved_memory = torch.cuda.memory_reserved(0) / 2**30
        # available_memory = total_memory - reserved_memory
        # print(f"Total GPU Memory: {total_memory / (1024**3):.2f} GB")
        # print(f"Reserved GPU Memory 0: {reserved_memory / (1024**3):.2f} GB")
        # print(f"Allocated GPU Memory: {allocated_memory / (1024**3):.2f} GB")         
        # print(f"Available (Unallocated) GPU Memory: {available_memory / (1024**3):.2f} GB")
        return
        
        
    def on_train_step(self):

        os.makedirs(os.path.join(self.args.results, "train"), exist_ok=True)

        self.gaussians._xyz.retain_grad()

        # if self.step < 1000:
        #     self.gaussians._exp3.requires_grad = False
        # else:
        #     self.gaussians._exp3.requires_grad = True
        
        start = perf_counter()
        # for iteration in range(first_iter, opt.iterations + 1):        
        #     if network_gui.conn == None:
        #         network_gui.try_connect()
        #     while network_gui.conn != None:
        #         try:
        #             net_image_bytes = None
        #             custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #             if custom_cam != None:
        #                 net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #                 net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #             network_gui.send(net_image_bytes, dataset.source_path)
        #             if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #                 break
        #         except Exception as e:
        #             network_gui.conn = None

        self.iter_start.record()

        self.gaussians.update_learning_rate(self.step)

        if args.optim_pose==False:
            self.gaussians.P.requires_grad_(False)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.step % 1000 == 0:
            self.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        viewpoint_cam = self.viewpoint_stack.pop(0)
        # viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))
        pose = self.gaussians.get_RT(viewpoint_cam.uid)

        # print(viewpoint_cam.uid)

        # Render
        if (self.step - 1) == self.debug_from:
            self.pipe.debug = True

        bg = torch.rand((3), device=self.device) if self.opt.random_background else self.background

        
        # Free intermediate variables after the backward pass if they're no longer needed
        render_pkg = None  # Free intermediate variable by removing reference
        torch.cuda.empty_cache()
        
        # if self.dataset.splat_type == 'GS':
        #     render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, bg, camera_pose=pose)
        if self.dataset.splat_type in ['GS', 'GSE', 'SQ', 'SQE']:
            render_pkg = render2(viewpoint_cam, self.gaussians, self.pipe, bg, camera_pose=pose)
        
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]
        
        # print(render_pkg)
        image = render_pkg["render"]                                 # (3, H, W)
        depth = render_pkg["depth"]                                  # (H, W)
        # visibility_filter = render_pkg["visibility_filter"]          # (N,)
        # radii = render_pkg["radii"]                                  # (M,)
        # viewspace_point_tensor = render_pkg["viewspace_points"]      # (N,)

        # print(visibility_filter.shape, radii.shape, visibility_filter.all())
        
        # Loss
        gt_image = viewpoint_cam.original_image.to(dtype=self.dtype, device=self.device)
        
        # plt.imshow(gt_image.permute(1, 2, 0).cpu().numpy()), plt.show()
        # plt.imshow(image.detach().permute(1, 2, 0).cpu().numpy()), plt.show()

        I = image.permute(1, 2, 0).detach().cpu().numpy() # (H, W, 3)
        I_GT = gt_image.permute(1, 2, 0).detach().cpu().numpy() # (H, W, 3)
        diff = ((I - I_GT)**2).sum(axis=-1) # (H, W)
        D = depth.clip(0, 3).detach().cpu().numpy() # (H, W)
        
        plt.imshow(I_GT), plt.show()
        plt.imshow(I), plt.show()
        plt.imshow(diff), plt.show()
        plt.imshow(D, cmap='jet_r'), plt.show()

        # print(image.shape, gt_image.shape)
        # print(image.dtype, gt_image.dtype)

        L1 = l1_loss(image, gt_image)
        SSIM = 1.0 - ssim(image, gt_image)
        PSNR = psnr(image, gt_image)
        LPIPS = lpips_model(image, gt_image)
                
        loss = (1.0 - self.opt.lambda_dssim) * L1 + self.opt.lambda_dssim * SSIM

        # with torch.autograd.set_detect_anomaly(True):
        #     # self.accelerator.backward(loss)
        #     loss.backward()
        
        # print(torch.isnan(image).any(), torch.isnan(loss))

        # allocated_memory = torch.cuda.memory_allocated(0) / 2**30
        # print(f"Allocated GPU Memory: {allocated_memory / (1024**3):.2f} GB")

        # next_fn = loss.grad_fn
        # for i in range(1000):
        #     next_fns = next_fn.next_functions
        #     next_fn = next_fns[0][0]
        #     print(next_fns)
        # sys.exit()

        
        # torch.cuda.memory._dump_snapshot("Memory_{}.pickle".format(self.step))

        total_memory = torch.cuda.get_device_properties(0).total_memory / 2**30
        allocated_memory = torch.cuda.memory_allocated(0) / 2**30
        reserved_memory = torch.cuda.memory_reserved(0) / 2**30
        available_memory = total_memory - reserved_memory
        # print(f"Total GPU Memory: {total_memory / (1024**3):.2f} GB")
        # print(f"Reserved GPU Memory: {reserved_memory / (1024**3):.2f} GB")
        print(f"Allocated GPU Memory: {allocated_memory / (1024**3):.2f} GB")
        # print(f"Available (Unallocated) GPU Memory: {available_memory / (1024**3):.2f} GB")

        print_memory_usage()
        
        if self.step == 0:
            print(gt_image.shape)
            # print(gt_image.shape)
        # os.remove(self.path)
        
        # loss.backward()
        # for param_group in gaussians.optimizer.param_groups:
        #     for param in param_group['params']:
        #         if param is gaussians.P:
        #             print(viewpoint_cam.uid, param.grad)
        #             break
        # print("Gradient of self.P:", gaussians.P.grad)

        log_dict = {'total': loss, 'l1': L1,
                    'ssim': 1-SSIM, 'psnr': PSNR}#, 'depth': None, }
        # return loss, log_dict
        
        self.iter_end.record()

        # with torch.no_grad():

        #     # Progress bar
        #     self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
        #     if self.step % 10 == 0:
        #         self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
        #         self.progress_bar.update(10)
        #     if self.step == self.opt.iterations:
        #         self.progress_bar.close()

        #     # Log and save
        #     training_report(self.tb_writer, self.step, L1, loss, l1_loss, self.iter_start.elapsed_time(self.iter_end), self.testing_iterations, self.scene, render, (self.pipe, self.background))
        #     if (self.step in self.saving_iterations):
        #         print("\n[ITER {}] Saving Gaussians".format(self.step))
        #         self.scene.save(self.step)
        #         save_pose(self.scene.model_path + '/pose' + f"/pose_{self.step}.npy", self.gaussians.P, self.train_cams_init)

        # # Densification
        # if self.step < self.opt.densify_until_iter:
        #     # Keep track of max radii in image-space for pruning
        #     print(self.gaussians.max_radii2D.max(), visibility_filter.shape)
            
        #     self.gaussians.max_radii2D[visibility_filter] =                                                                   torch.max(self.gaussians.max_radii2D[visibility_filter], radii)
        #     # self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        #     self.gaussians.add_densification_stats(self.gaussians._xyz, visibility_filter)

        #     print(self.gaussians._xyz.shape)
        #     print(self.gaussians._xyz.grad.shape, self.gaussians._xyz.grad.max())

        #     if self.step > self.opt.densify_from_iter and self.step % self.opt.densification_interval == 0:
        #         size_threshold = 20 if self.step > self.opt.opacity_reset_interval else None
        #         self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005,                                          self.scene.cameras_extent,                              size_threshold)

        #     print(self.gaussians._xyz.shape)
        #     print(self.gaussians._xyz.grad.shape, self.gaussians._xyz.grad.max())

            
            # if self.step % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and self.step ==                          self.opt.densify_from_iter):
            #     self.gaussians.reset_opacity()

            # print(self.gaussians._xyz.grad.max(), self.gaussians._xyz.shape, )
        

        #     # Optimizer step
        #     # if iteration < opt.iterations:
        #         # gaussians.optimizer.step()
        #         # gaussians.optimizer.zero_grad(set_to_none = True)

        #     if (self.step in self.checkpoint_iterations):
        #         print("\n[ITER {}] Saving Checkpoint".format(self.step))
        #         torch.save((self.gaussians.capture(), self.step), self.scene.model_path + "/chkpnt" + str(self.step) + ".pth")
                
        end = perf_counter()
        train_time = end - start
    
        # We commented out log&save operations, and then calculate train time.
        # train_time = np.array(train_time)
        # print("total_test_time_epoch: ", 1)
        # print("instantsplat_train_time_mean: ", train_time.mean())
        # print("instantsplat_train_time_median: ", np.median(train_time))

        if os.path.isfile(self.args.results + 'results_train.csv'):
            results = pd.read_csv(self.args.results + 'results_train.csv', index_col=None)
        else:
            results = pd.DataFrame(columns=['L1', 'SSIM', 'PSNR', 'Loss', 'LPIPS', 'Allocated_GPU', 'Available_GPU'])

        df = pd.DataFrame({'L1':[L1.item()], 'SSIM':[1-SSIM.item()], 'PSNR':[PSNR.item()], 'Loss':[loss.item()], 'LPIPS':[LPIPS.item()], 'Allocated_GPU':[allocated_memory], 'Available_GPU':[available_memory]})
        results = pd.concat([results, df], ignore_index=True)
        results.to_csv(self.args.results + "/results_train.csv", index=False)
            
        img = (image.clamp(0, 1) * 255).to(dtype=torch.uint8).permute(1, 2, 0).detach().cpu().numpy()
        img = Image.fromarray(img)
        # print(img.shape, image.dtype)
        name = '0' * (4 - len(str(self.step))) + str(self.step)
        img.save(self.args.results + f"/train/{name}.png")

        # if (self.step + 1) % 1 == 0
        self.gaussians.save_ply(self.args.results + f'/model.ply')
        # self.gaussians.load_ply(self.args.results + '/model.ply')

        # torch.autograd.set_detect_anomaly(False)
        # loss.backward() 

        # print(torch.isnan(self.gaussians._xyz).any())
        # print(torch.isnan(self.gaussians._exp12).any())
        # print(torch.isnan(self.gaussians._exp3).any())
        # print(torch.isnan(self.gaussians._exp12).any())
        # print(torch.isnan(self.gaussians._exp12).any())
        # print(torch.isnan(self.gaussians._exp12).any())

        # sys.exit()

        return loss, log_dict, render_pkg

    
    def on_densify_step(self, render_pkg):
        print('Densifying')
        
        image = render_pkg["render"]                                 # (3, H, W)
        depth = render_pkg["depth"]                                  # (H, W)
        visibility_filter = render_pkg["visibility_filter"]          # (N,)
        radii = render_pkg["radii"]                                  # (M,)

        # Densification
        # Keep track of max radii in image-space for pruning
        # print(self.gaussians.max_radii2D.max(), visibility_filter.shape)
        
        # self.gaussians.max_radii2D[visibility_filter] =                                                                   torch.max(self.gaussians.max_radii2D[visibility_filter], radii)
        # self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        # self.gaussians.add_densification_stats(self.gaussians._xyz, visibility_filter)

        # print(self.gaussians._xyz.grad.shape, self.gaussians._xyz.grad.max())

        if self.step > self.opt.densify_from_iter and self.step % self.opt.densification_interval == 0:
            size_threshold = 20 if self.step > self.opt.opacity_reset_interval else None
            self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.1,                                          self.scene.cameras_extent, size_threshold)

        # print(self.gaussians._xyz.shape)
        # print(self.gaussians._xyz.grad.shape, self.gaussians._xyz.grad.max())

        return


    def evaluate(self):

        os.makedirs(os.path.join(self.args.results, "eval"), exist_ok=True)

        self.gaussians.load_ply(self.args.results + '/model.ply')

        self.gaussians._exp12 = nn.Parameter(torch.zeros_like(self.gaussians._exp12, dtype=self.dtype, device=self.device).requires_grad_(False))
        
        # self.gaussians.load_ply(self.args.results[:-3] + 'GSE/' + '/model.ply')
        # self.gaussians._exp3 = nn.Parameter(torch.zeros_like(self.gaussians._exp3, dtype=self.dtype, device=self.device-3.688879454216).requires_grad_(True)) 

        # print(torch.isnan(self.gaussians._xyz).sum())
        
        # sys.exit()
        
        results = pd.DataFrame(columns=['L1', 'SSIM', 'PSNR', 'Loss', 'LPIPS', 'Allocated_GPU', 'Available_GPU'])

        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        
        for i in range(0, len(self.viewpoint_stack)):
            
            # Pick a random Camera
            # viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))
            viewpoint_cam = self.viewpoint_stack.pop(0)
            pose = self.gaussians.get_RT(viewpoint_cam.uid)

            bg = torch.rand((3), device=device) if self.opt.random_background else self.background

            # Free intermediate variables after the backward pass if they're no longer needed
            render_pkg = None  # Free intermediate variable by removing reference
            torch.cuda.empty_cache()

            # if self.dataset.splat_type == 'GS':
                # render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, bg, camera_pose=pose)
            if self.dataset.splat_type in ['GS', 'GSE', 'SQ', 'SQE']:
                render_pkg = render2(viewpoint_cam, self.gaussians, self.pipe, bg, camera_pose=pose)

            # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            image = render_pkg["render"]#.permute(2, 0, 1)
            depth = render_pkg["depth"]
            
            # d_m, d_std = depth.mean(), depth.std()
            # depth = depth.clip(d_m * 0, d_m + 2 * d_std)
            # print(depth.mean(), depth.max(), depth.min())
            

            
            # Loss
            gt_image = viewpoint_cam.original_image.to(dtype=self.dtype, device=self.device)
            # print(image.shape, gt_image.shape)
            # print(image.dtype, gt_image.dtype)
            L1 = l1_loss(image, gt_image)
            SSIM = 1.0 - ssim(image, gt_image)
            PSNR = psnr(image, gt_image)
            LPIPS = lpips_model(image, gt_image)

            I = image.permute(1, 2, 0).detach().cpu().numpy() # (H, W, 3)
            I_GT = gt_image.permute(1, 2, 0).detach().cpu().numpy() # (H, W, 3)
            diff = ((I - I_GT)**2).sum(axis=-1) # (H, W)
            D = depth.clip(0, 3).detach().cpu().numpy() # (H, W)
            
            plt.imshow(I_GT), plt.show()
            plt.imshow(I), plt.show()
            plt.imshow(diff), plt.show()
            plt.imshow(D, cmap='jet_r'), plt.show()

            
            

            # log_dict = {'total': loss, 'l1': L1,
                        # 'ssim': SSIM, 'psnr': PSNR}#, 'depth': None, }

            loss = (1.0 - self.opt.lambda_dssim) * L1 + self.opt.lambda_dssim * SSIM
            # torch.cuda.memory._dump_snapshot("Memory_{}.pickle".format(self.step))

            total_memory = torch.cuda.get_device_properties(0).total_memory / 2**30
            allocated_memory = torch.cuda.memory_allocated(0) / 2**30
            reserved_memory = torch.cuda.memory_reserved(0) / 2**30
            available_memory = total_memory - reserved_memory / 2 **30
            # print(f"Allocated GPU Memory: {allocated_memory / (1024**3):.2f} GB            ", f"Available (Unallocated) GPU Memory: {available_memory / (1024**3):.2f} GB")

            
            df = pd.DataFrame({'L1':[L1.item()], 'SSIM':[SSIM.item()], 'PSNR':[PSNR.item()], 'Loss':[loss.item()], 'LPIPS':[LPIPS.item()], 'Allocated_GPU':[allocated_memory], 'Available_GPU':[available_memory]})
            results = pd.concat([results, df], ignore_index=True)
            results.to_csv(self.args.results + "/results_eval.csv", index=False)

            # image = image.clip(0, 1) * 255
            img = (image.clip(0, 1) * 255).to(dtype=torch.uint8).permute(1, 2, 0).detach().cpu().numpy()
            img = Image.fromarray(img)
            # print(img.shape, image.dtype)
            name = '0' * (4 - len(str(i))) + str(i)
            img.save(self.args.results + f"/eval/{name}.png")

            depth_img = (depth - depth.min()) / (depth.max() - depth.min())
            depth_img = (depth_img * 255).to(dtype=torch.uint8).detach().cpu().numpy()
            depth_img = Image.fromarray(depth_img)
            # print(img.shape, image.dtype)
            name = 'd' + '0' * (4 - len(str(i))) + str(i)
            depth_img.save(self.args.results + f"/eval/{name}.png")

        return results

    def on_evaluate_step(self):
        pass

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500, 800, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--n_views", type=int, default=None)
    parser.add_argument("--get_video", action="store_true")
    parser.add_argument("--optim_pose", action="store_true")
    # parser.add_argument("--splat_type", type=str, default='SQE')
    parser.add_argument("--results", type=str)
    parser.add_argument("--dtype", type=str, default='fp32')
    parser.add_argument("--max_splats", type=int, default=200000)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    os.makedirs(args.model_path, exist_ok=True)
    
    print("Optimizing " + args.model_path)

    # sys.exit()

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)
    # training2(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # torch.cuda.memory._record_memory_history(enabled=True)
    
    trainer = SceneTrainer(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from, 
        args)


    
    # trainer.train()
    trainer.evaluate()


    
    
    torch.cuda.memory._record_memory_history(enabled=False)
#     with open("Memory.txt", "w") as f:
#         for entry in history:
#             f.write(str(entry) + "\n")    
    
    # All done
    print("\nTraining complete.")
