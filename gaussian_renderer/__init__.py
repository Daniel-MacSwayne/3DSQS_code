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



import torch as tc
import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.gaussian_model import GaussianModel, GaussianModel2
from utils.sh_utils import eval_sh
from utils.pose_utils import get_camera_from_tensor, quadmultiply

import sys
import pathlib
import matplotlib.pyplot as plt
import gc

from .Superquadric_Splatting import *
# from Plotly_Functions import *

# tc.manual_seed(0)
# np.random.seed(0)
tc.autograd.set_detect_anomaly(False)
# dtype = tc.float32
# device = tc.device('cuda')
# device = tc.device('cpu')


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: tc.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    camera_pose=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (tc.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0)
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # Set camera pose as identity. Then, we will transform the Gaussians around camera_pose
    w2c = tc.eye(4).cuda()
    projmatrix = (w2c.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0))).squeeze(0)
    camera_pos = w2c.inverse()[3, :3]
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        # viewmatrix=viewpoint_camera.world_view_transform,
        # projmatrix=viewpoint_camera.full_proj_transform,
        viewmatrix=w2c,
        projmatrix=projmatrix,
        sh_degree=pc.active_sh_degree,
        # campos=viewpoint_camera.camera_center,
        campos=camera_pos,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    dtype = pc.get_xyz.dtype    
    
    # means3D = pc.get_xyz
    rel_w2c = get_camera_from_tensor(camera_pose).to(dtype=dtype, device=device)
    # Transform mean and rot of Gaussians to camera frame
    gaussians_xyz = pc._xyz.clone()
    gaussians_rot = pc._rotation.clone()

    xyz_ones = tc.ones(gaussians_xyz.shape[0], 1).to(dtype=dtype, device=device)
    xyz_homo = tc.cat((gaussians_xyz, xyz_ones), dim=1)
    gaussians_xyz_trans = (rel_w2c @ xyz_homo.T).T[:, :3]
    gaussians_rot_trans = quadmultiply(camera_pose[:4], gaussians_rot)
    means3D = gaussians_xyz_trans
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = gaussians_rot_trans  # pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = tc.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).

    # print(means3D.dtype, means2D.dtype, colors_precomp.dtype, opacity.dtype, scales.dtype, rotations.dtype)

    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }


###########################################################################

def render2(
    viewpoint_camera,
    pc: GaussianModel2,
    pipe,
    bg_color: tc.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    camera_pose=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    
    dtype = pc.dtype
    device = pc.device


    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = (tc.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")+0)
    # try:
    #     screenspace_points.retain_grad()
    # except:
    #     pass

    # screenspace_points = 
    
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    ###########################################################################

    # means3D = pc.get_xyz
    rel_w2c = get_camera_from_tensor(camera_pose).to(dtype=dtype, device=device)                     # (4, 4)
    # Transform mean and rot of Gaussians to camera frame
    gaussians_xyz = pc._xyz.clone()                                         # (N, 3)
    gaussians_rot = pc._rotation.clone()                                    # (N, 4)
    xyz_ones = tc.ones(gaussians_xyz.shape[0], 1).to(dtype=dtype, device=device)         # (N)
    xyz_homo = tc.cat((gaussians_xyz, xyz_ones), dim=1)                  # (N, 4)
    means3D = (rel_w2c @ xyz_homo.T).T[:, :3]                   # (N, 3)
    # gaussians_xyz_trans = (rel_w2c @ xyz_homo.T).T[:, :3]                   # (N, 3)
    gaussians_rot_trans = quadmultiply(camera_pose[:4], gaussians_rot)      # (N, 4)
    # means3D = gaussians_xyz_trans                                           # (N, 3)
    # means2D = screenspace_points

    # print(means3D.max())
    
    # If points are too near or behind camera, roughly within FOV and not too small.
    visible1 = means3D[:, 2] > 0.01                                          # (N,)
    visible2 = tc.abs(means3D[:, 0] / means3D[:, 2]) < tanfovx * 1.05     # (N,)
    visible3 = tc.abs(means3D[:, 1] / means3D[:, 2]) < tanfovy * 1.05     # (N,)
    # visible4 = tc.norm(scales, axis=-1) / means3D[:, 2] > min(tanfovx/viewpoint_camera.width, tanfovy/viewpoint_camera.height) / 2
    visible = visible1 * visible2 * visible3# * visible4                     # (N,)
    # visible[::2] = False
    # visible[20000:] = False
    
    # print(visible1, visible2, visible3)
    
    means3D = means3D[visible]                                               # (M, 3)
    opacity = pc.get_opacity[visible]                                        # (M, 1)
    exp = pc.get_exp[visible]                                                # (M, 3)
    
    # If precomputed 3d covariance is provided, use it. If not, then it will be
    # computed from scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)[visible]        # (M, 3, 3)
    else:
        scales = pc.get_scaling[visible]                                    # (M, 3)
        rotations = gaussians_rot_trans[visible]                            # (M, 4)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features[visible].transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz[visible] - viewpoint_camera.camera_center.repeat(pc.get_features[visible].shape[0], 1).to(dtype=dtype, device=device)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = tc.clamp_min(sh2rgb + 0.5, 0.0)

        else:
            shs = pc.get_features[visible]                                 # (M, _)
    else:
        colors_precomp = override_color[visible]                           # (M, 3)


    # print(rel_w2c.shape, gaussians_xyz.shape, gaussians_rot.shape, visible.shape, means3D.shape, scales.shape, rotations.shape, colors_precomp.shape)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    render_color, render_depth, radii = rasterizer3(
        camera=viewpoint_camera,
        means3D=means3D,
        # means2D=means2D,
        # shs=shs,
        colors_precomp=colors_precomp,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
        exps=exp,
        # cov3D_precomp=cov3D_precomp,
    )

    render_pkg =  {
        "render": render_color,                           # (3, H, W)
        # "viewspace_points": screenspace_points,           # (N,)
        "visibility_filter": visible,                     # (N,)
        "radii": radii,                                   # (M,)
        "depth": render_depth,                            # (H, W)
        # "alpha": self.render_alpha,
    }
    
    return render_pkg


@tc.no_grad()
def get_rect(means2D, radii, width, height):
    # means2D (M, 2)
    # radii (M,)
    
    k = 2
    radii = radii.reshape(-1, 1)                            # (M, 1)
    rect_min = (means2D - radii*k)                          # (M, 2)
    rect_max = (means2D + radii*k)                          # (M, 2)
    rect_min[..., 0] = rect_min[..., 0].clip(0, width)      # (M, 2)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height)     # (M, 2)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width)      # (M, 2)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height)     # (M, 2)
    return rect_min, rect_max


def rasterizer2(camera, means3D, colors_precomp, opacity, scales, rotations, exps):

    dtype = means3D.dtype
    device = means3D.device
    
    W, H = camera.image_width, camera.image_height
    tanfovx, tanfovy = math.tan(camera.FoVx/2), math.tan(camera.FoVy/2)
    fx, fy = W/(2*tanfovx), H/(2*tanfovy)
    f = tc.tensor([[fx, fy]]).to(dtype=dtype, device=device)
    c = 0.5 * tc.tensor([[W, H]]).to(dtype=dtype, device=device)

    means2D = means3D[:, :2] / means3D[:, 2:] * f + c
    radii = (tc.norm(scales, dim=-1)/exps[:, 2].clamp(0.9, 1)) / means3D[:, 2] * f.mean()
    rect = get_rect(means2D, radii, width=W, height=H)   # (N, 2), (N, 2)
    
    pix_coord = tc.stack(tc.meshgrid(tc.arange(W), tc.arange(H), indexing='xy'), dim=-1).to(dtype=dtype, device=device) * 1.
    pix_coord += 0.5 - c      # (T, T, 2)

    render_color = tc.ones(*pix_coord.shape[:2], 3).to(dtype=dtype, device=device)     # (H, W, 3)
    render_depth = tc.zeros(*pix_coord.shape[:2], 1).to(dtype=dtype, device=device)    # (H, W, 1)
    # self.render_alpha = tc.zeros(*self.pix_coord.shape[:2], 1).to(dtype=dtype, device=device)    # (H, W, 1)
    
    # print(camera.image_width, camera.image_height)
    TILE_SIZE = 16
    for h in range(0, H, TILE_SIZE):
        for w in range(0, W, TILE_SIZE):
            # check if the rectangle penetrate the tile
            over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)                              # (N,) (N,)
            over_br = rect[1][..., 0].clip(max=w+TILE_SIZE), rect[1][..., 1].clip(max=h+TILE_SIZE)      # (N,) (N,)
            in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1])     # 3D gaussian in the tile         (N)

            # in_mask[::2] = False
            
            if not in_mask.sum() > 0:
                continue
            
            P = in_mask.sum()
            # print(P)            
            
            sorted_depths, index = tc.sort(means3D[in_mask][:, 2])                # (P,), (P,)
            
            # C = Histogram2D_(means2D[in_mask], num_bins=4)[-1]            # (P,) int64, Bin counts for each splat 
            # D = tc.arange(P).to(device=device)                                    # (P,) int64, Depth Index
            # Z = tc.argsort(C**2)# * D)                                              # (P,) int64,
            limit = tc.zeros(P, dtype=tc.bool, device=device)                  # (P,) bool
            L = 1000000
            # limit[Z[:L]] = True
            limit[:L] = True
            # limit[tc.randperm(P)[:L]] = True
            L = limit.sum()
            # print(P)
            # print(sorted_depths, index, C, D, Z, limit)
            # sys.exit()
            
            sorted_depths = sorted_depths[limit]                                     # (L, 3)
            sorted_opacity = opacity[in_mask][index][limit]                          # (L, 1)
            sorted_color = colors_precomp[in_mask][index][limit]                     # (L, 3)                       
            sorted_means = means3D[:, :2][in_mask][index][limit].reshape(-1, 1, 1, 2)# (L, 1, 1, 2)       
            sorted_scales = scales[in_mask][index][limit]                            # (L, 3)
            sorted_exps = exps[in_mask][index][limit]                                # (L, 3)
                
            ###################################################################

            tile_coord = pix_coord[h:h+TILE_SIZE, w:w+TILE_SIZE]                   # (T, T, 2)
            h_t, w_t, _ = tile_coord.shape                                             

            ###################################################################

            Xp__ = tile_coord.reshape(1, h_t, w_t, 2)                               # (1, T, T, 2)
            Xi__ = Xp__ / f.reshape(1, 1, 1, 2)                                     # (1, T, T, 2)
            Xc__ = Xi__ * sorted_depths.reshape(-1, 1, 1, 1)                        # (L, T, T, 2)

            # Shift and prevent potential singularities at origin
            I = Xc__ == sorted_means
            Xc__ -= sorted_means * ~I
            Xc__ -= sorted_means * I + 1e-10
            # Xc__ -= sorted_means                                                    # (L, T, T, 2)    

            q_ws = rotations[in_mask][index][limit]                                 # (L, 4)
            R_ws = Construct_(q_ws)                                                 # (L, 3, 3)

            v_cw = camera.world_view_transform.to(dtype=dtype, device=device)       # (4, 4)
            R_cw = v_cw[:3, :3]                                                     # (3, 3)
            T_cw = v_cw[:3, 3]                                                      # (3)

            R_cs = R_ws @ R_cw                                                      # (3, 3)

            G__ = Superquadric_Tile_(Xc__, s_=sorted_scales, e_=sorted_exps, R_cs=R_cs, Show=False)     # (L, T, T)
            gauss_weight = G__.reshape(L, -1).T                                     # (T*T, L)

            # plt.imshow(G__[0].detach().cpu().numpy()), plt.show()
            # sys.exit()
            ###############################################################
            white_bkgd = True
            alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99)                 # (HW, L, 1)
            T = tc.cat([tc.ones_like(alpha[:,:1]), 1-alpha[:,:-1]], dim=1).cumprod(dim=1)     # (HW, L, 1)
            acc_alpha = (alpha * T).sum(dim=1)                                                      # (HW, 1)
            tile_color = (T * alpha * sorted_color[None]).sum(dim=1) + (1-acc_alpha) * (1 if white_bkgd else 0)    # (HW, 3)
            tile_depth = ((T * alpha) * sorted_depths[None,:,None]).sum(dim=1)                                          # (HW, 1)
            render_color[h:h+h_t, w:w+w_t] = tile_color.reshape(h_t, w_t, -1)     # (H, W, 3)
            render_depth[h:h+TILE_SIZE, w:w+TILE_SIZE] = tile_depth.reshape(h_t, w_t, -1)     # (H, W, 1)
            #render_alpha[h:h+TILE_SIZE, w:w+TILE_SIZE] = acc_alpha.reshape(TILE_SIZE, TILE_SIZE, -1)      # (H, W, 1)
            # print(gauss_weight[..., None].shape, sorted_opacity[None].shape, alpha.shape, T.shape, acc_alpha.shape, tile_color.shape, tile_depth.shape)
            # print(tc.cat([tc.ones_like(alpha[:,:1]), 1-alpha[:,:-1]], dim=1).shape)
            # sys.exit()



    # tc.cuda.memory._dump_snapshot(r"C:\Users\danma\Downloads\GPU_Memory.pickle")
    # tc.cuda.memory._record_memory_history(enabled=None)

    # print(render_color.dtype)
    # print(render_color)
    # plt.imshow(render_color.detach().to(dtype=tc.float32, device='cpu').numpy())
    # # plt.savefig('Image{}.png'.format(np.random.randint(1000)))
    # plt.show()
    # sys.exit()
    
    render_pkg =  {
        "render": render_color.permute(2, 0, 1),
        "depth": render_depth,
        # "alpha": self.render_alpha,
        # "viewspace_points": screenspace_points,

        # "visiility_filter": radii > 0,
        # "radii": radii
    }

    try:
        del means3D, means2D, radii, rect, pix_coord, over_tl, over_br, in_mask, sorted_depths, sorted_opacity, sorted_color, sorted_means, sorted_scales, sorted_exps, tile_coord, Xp__, Xi__, Xc__, q_ws, R_ws, v_cw, R_cw, T_cw, R_cs, G__, gauss_weight, alpha, T, acc_alpha, tile_color, tile_depth
    except: pass


    
    return render_pkg


def rasterizer3(camera, means3D, colors_precomp, opacity, scales, rotations, exps):
    # means3D        (M, 3)
    # colors_precomp (M, ...)
    # opacity        (M, 1)
    # scales         (M, 3)
    # rotations      (M, 3, 3)
    # exps           (M, 3)
    # M: Number of visible splats in current view
    # L: Max number of splats in batch
    # l: Actual number of splats in batch

    # H: Image height
    # W: Image width
    # N_h: Number of vertical tiles
    # N_w: Number of horizontal tiles
    # h: Vertical pixels per tile
    # w: Horizontal pixels per tile
    # T: Number of active Tiles
    # U: Number of unique splats in batch

    M = means3D.shape[0]
    L = 10
    dtype = means3D.dtype
    device = means3D.device
    
    ###########################################################################
    # Initialize Image Canvas
    
    W, H = camera.image_width, camera.image_height
    tanfovx, tanfovy = math.tan(camera.FoVx/2), math.tan(camera.FoVy/2)
    fx, fy = W/(2*tanfovx), H/(2*tanfovy)
    f = tc.tensor([[fx, fy]]).to(dtype=dtype, device=device)      # (1, 2)
    c = 0.5 * tc.tensor([[W, H]]).to(dtype=dtype, device=device)  # (1, 2)

    pix_coord = tc.stack(tc.meshgrid(tc.arange(W), tc.arange(H), indexing='xy'), dim=-1).to(dtype=dtype, device=device) * 1. 
    pix_coord += 0.5 - c      # (H, W, 2)

    h, w = 16, 16
    N_h, N_w = H // h, W // w
    pix_coord = pix_coord.reshape(N_h, h, N_w, w, 2).permute(0, 2, 1, 3, 4) # (N_h, N_w, h, w, 2)

    Xp__ = pix_coord                                        # (N_h, N_w, h, w, 2)
    Xi__ = Xp__ / f[0]                                      # (N_h, N_w, h, w, 2)

    del pix_coord, Xp__

    v_cw = camera.world_view_transform.to(dtype=dtype, device=device)       # (4, 4)
    R_cw = v_cw[:3, :3]                                                     # (3, 3)
    T_cw = v_cw[:3, 3]                                                      # (3)

    ###########################################################################
    # Sort Splats by depth
    
    I = tc.argsort(means3D[:, 2])   # (M,)
    
    means3D = means3D[I]            # (M, 3)
    scales = scales[I]              # (M, 3)
    exps = exps[I]                  # (M, 3)
    rotations = rotations[I]        # (M, 4)
    colors = colors_precomp[I]      # (M, 3)
    opacity = opacity[I]        # (M, 1)

    depths = means3D[:, 2:]         # (M, 1)
    means2D = means3D[:, :2] / depths * f + c                                                         # (M, 2)
    radii = (tc.norm(scales, dim=-1) / exps[:, 2].clamp(0.5, 1)) / means3D[:, 2] * f.mean()             # (M,)

    ###########################################################################
    # Which splats are in each tile?

    rect_tl, rect_br = get_rect(means2D, radii, width=W, height=H)                                   # (M, 2), (M, 2)

    # Generate all tile top-left corners (N_h, N_w, 2)
    tile_tl = tc.stack(tc.meshgrid(tc.arange(0, W, w), tc.arange(0, H, h), indexing="xy"), dim=-1).to(device=device)   # (N_h, N_w, 2)
    tile_br = tile_tl + tc.tensor([[[h, w]]]).to(device=device)                                                        # (N_h, N_w, 2)

    over_tl = tc.maximum(tile_tl[:, :, None], rect_tl[None, None])      # (N_h, N_w, M, 2)
    over_br = tc.minimum(tile_br[:, :, None], rect_br[None, None])      # (N_h, N_w, M, 2)

    del rect_tl, rect_br, tile_tl, tile_br
    
    # Intersection exists if width & height are positive
    in_mask = (over_br > over_tl).all(dim=-1)               # (N_h, N_w, M)

    del over_tl, over_br
    
    Counts = in_mask.sum(axis=(-1))                         # (N_h, N_w)
    L_max = Counts.max().item()                             # Find max count in the mask
    
    Map = tc.arange(M, device=device).repeat(N_h, N_w, 1) * 1.             # (N_h, N_w, M) float int
    Map[~in_mask] = tc.inf
    Map, _ = tc.sort(Map, axis=-1)                          # (N_h, N_w, M) float + inf
    Map = Map[:, :, :L_max]                                 # (N_h, N_w, L_max) float + inf
    Map[Map == tc.inf] = -1                                 # (N_h, N_w, L_max) float int
    Map = Map.to(dtype=tc.long)                             # (N_h, N_w, L_max) int

    del in_mask
    
    ###############################################################################
    # Initialize accumulation buffers
    A_acc = tc.zeros((N_h, N_w, h, w, 1), dtype=dtype, device=device)   # (N_h, N_w, h, w, 1)
    C_acc = tc.zeros((N_h, N_w, h, w, 3), dtype=dtype, device=device)   # (N_h, N_w, h, w, 3)
    D_acc = tc.ones((N_h, N_w, h, w, 1), dtype=dtype, device=device)    # (N_h, N_w, h, w, 1)

    # Splats in Tiles  mask
    m0 = (Map != -1)                                        # (N_h, N_w, L_max)
    
    # Initial non-empy tiles mask
    m1 = m0.any(axis=-1)                                    # (N_h, N_w)
    
    # Initial unsaturated pixels mask
    m2 = A_acc[..., 0] < 0.99                               # (N_h, N_w, h, w) 
    
    # Initial unsaturated tiles mask
    m3 = tc.ones((N_h, N_w), dtype=tc.bool, device=device)  # (N_h, N_w)

    # Main Splat Accumulation Loop
    for i in range(0, L_max, L):
        ###########################################################################
        # Efficent Tile/Pixel Management
        
        # Only process tiles that are not empty and have unsaturated pixels
        m4 = m1 & m3                                        # (N_h, N_w)
        T = m4.sum().item()                                 # (T) = (N_h.N_w.)
        
        if T == 0:
            break  # All tiles are saturated or empty
    
        # Gather batch of splat indices
        l = min(L, L_max - i)                               # Might not be a full L batch. l < L
        I0 = Map[:, :, i:i+L][m4]                           # (T, l) int
        
        # Unique Splats in batch    
        I1, I2 = tc.unique(I0, return_inverse=True)         # (U,) (T, l)
        U = len(I1)                                         # (U) = (l.)
        
        # Active splats in active tiles mask
        m5 = m0[m4][:, i:i+L]                               # (T, l)
        
        # Active pixels in active tiles mask
        m6 = m2[m4]                                         # (T, h, w)
        P = m6.sum().item()
    
        # Active splats & pixels in active tiles mask
        m7 = m5[:, :, None, None] & m6[:, None, :, :]       # (T, l, h, w)
        S = m7.sum().item()                                 # (S) = (T.l.h.w.)
        
        # Reshaping for later usage
        m8 = m7[m5].reshape(-1, h*w)                        # (T.l., h*w)
    
        # print('Tiles:', T, ' Pixels:', P, ' Splat-Pixels:', S)
        # plt.imshow(m1), plt.show()
        # plt.imshow(m3), plt.show()
        # plt.imshow(m4), plt.show()
        
        ###########################################################################
        # Begin Rendering
    
        # Grab Splat Parameters
        C__ = colors[I0][:, :, None, None]                  # (T, l, 1, 1, 3)
        A__ = opacity[I0][:, :, None, None]               # (T, l, 1, 1, 1)
        M_ = means3D[:, :2][I0]                             # (T, l, 2)
        D_ = depths[I0][..., None, None]                    # (T, l, 1, 1, 1)
        s_ = scales[I1]                                     # (U, 3)
        e_ = exps[I1]                                       # (U, 3)
        q_ws = rotations[I1]                                # (U, 4)  
        R_ws = Construct_(q_ws)                             # (U, 3, 3)
        R_cs = R_ws @ R_cw                                  # (U, 3, 3)

        ###########################################################################
        # Adjust Splatting Canvas Coordinates
        Xc__ = Xi__[m4][:, None] * D_                       # (T, l, h, w, 2)                                
        Xc__ -= M_[:, :, None, None]                        # (T, l, h, w, 2)

        # Store useful masks
        masks = [m0, m1, m2, m3, m4, m5, m6, m7, m8, I0, I1, I2]
            
        # Compute Gaussian Weighting
        G__ = tc.zeros((T, l, h, w), dtype=dtype, device=device)  # (T, l, h, w)
        G__[m7] = Superquadric_Tile(Xc__, s_, e_, R_cs, masks)   # (T.l.h.w.,) 
        G__ = G__[..., None]                                # (T, l, h, w, 1)

        ###########################################################################
        # Alpha Blending
        a_acc = A_acc[m4][:, None]                          # (T, 1, h, w, 1)
        alpha = (G__ * A__).clip(max=0.99)                  # (T, L, h, w, 1)
        trans = tc.cat([1 - a_acc, 1-alpha[:,:-1]], dim=1)  # (T, L, h, w, 1)
        trans = trans.cumprod(dim=1)                        # (T, L, h, w, 1)
    
        # Compute color contribution for this batch
        C_acc[m4] += (trans * alpha * C__).sum(dim=1)       # (T, h, w, 3)
    
        # Compute depth contribution for this batch
        D_acc[m4] += (trans * alpha * D_).sum(dim=1)        # (T, h, w, 1)
    
        # Update accumulated opacity
        A_acc[m4] += (alpha * trans).sum(dim=1)             # (T, h, w, 1)
        A_acc[m4] = A_acc[m4].clip(max=0.99)                # (T, h, w, 1)

        # plt.imhow(A_acc[..., 0].sum(axis=1).detach().numpy().cpu()), plt.show()
#        plt.imshow(A_acc[..., 0].permute(0, 2, 1, 3).reshape(H, W).cpu().detach().numpy()), plt.show()
        # plt.imshow(C_acc.permute(0, 2, 1, 3, 4).reshape(H, W, 3).cpu().detach().numpy()), plt.show()

        
        ###########################################################################
        # Update Dynamic Tile/Pixel Masks
    
        # Update non-empty tiles mask
        m1[m4] = m5.any(axis=-1)                            # (T, L)
        
        # Update unsaturated pixels mask
        m2[m4] = A_acc[m4][..., 0] < 0.99                   # (T, h, w)
    
        # Update unsaturated tiles mask
        m3[m4] = m2[m4].reshape(T, h*w).any(dim=-1)         # (T,)

        del C__, A__, M_, D_, s_, e_, q_ws, R_ws, R_cs, Xc__, G__, a_acc, alpha, trans
    
    # Apply Background Colour (e.g., white) to all remaining transparent regions
    white_bkgd = False
    if white_bkgd:
        background_color = tc.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device).view(1, 1, 1, 3)  # (1, 1, 1, 3)
        C_acc += (1 - A_acc) * background_color             # (N_h, N_w, h, w, 3)
    
    # Apply far-field depth for transparent background
    # max_depth = 100.0  # Set a reasonable far-field depth
    max_depth = depths.mean()  # Set a reasonable far-field depth
    D_acc += (1 - A_acc) * max_depth                        # (N_h, N_w, h, w, 1)
    
    # Reshape to final image
    C_acc = C_acc.permute(0, 2, 1, 3, 4).reshape(H, W, 3)   # (H, W, 3)
    D_acc = D_acc.permute(0, 2, 1, 3, 4).reshape(H, W)      # (H, W)
    
    render_color = C_acc.permute(2, 0, 1).clip(0, 1)        # (3, H, W)
    render_depth = D_acc                                    # (H, W)
        
    # plt.imshow(render_color.permute(1, 2, 0).cpu().detach().numpy()), plt.show()

    render_pkg =  {
        "render": render_color,
        # "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0),
        "radii": radii,
        "depth": render_depth,
        # "alpha": self.render_alpha,
    }


    # del means3D, scales, exps, rotations, colors, opacity, depths, means2D, radii, Counts, Map, A_acc, C_acc, D_acc, m0, m1, m2, m3, m4, m5, m6, m7, m8, I0, I1, I2, Xi__

    tc.cuda.empty_cache()
    gc.collect()
    # print(tc.cuda.memory_summary(device=None, abbreviated=False))
    
    # return render_pkg
    return render_color, render_depth, radii
