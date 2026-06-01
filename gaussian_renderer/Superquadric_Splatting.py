import sys
import pathlib
import glob
import time

import math
import numpy as np
np.random.seed(0)
# import pandas as pd
# import cv2
# import open3d as o3d

from scipy.spatial.transform import Rotation
import scipy.interpolate as interpolate
from scipy.spatial import KDTree
# import scipy.optimize as spopt
# from sklearn.mixture import GaussianMixture
# from sklearn.preprocessing import OneHotEncoder

import torch as tc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

tc.manual_seed(5)
tc.autograd.set_detect_anomaly(True)
# print('CUDA:', tc.cuda.is_available(), tc.version.cuda)
# device = 'cuda'
# device = 'cpu'
# dtype = tc.float32
# device = tc.device('cuda')
# device = tc.device('cpu')

ε = 1e-10   # Small Value to prevent nans in pow backward pass

import matplotlib.pyplot as plt

Path = str(pathlib.Path().resolve())
ODB_Path = Path[:Path.find('Birmingham')+11]
Projects_Path = ODB_Path + r'/Projects'

# Plotly_Path = Projects_Path + r'/External Repositories/Visual_Data'
# sys.path.insert(0, Plotly_Path)
# from Plotly_Functions import *

# Interp_Path = Projects_Path + r'/External Repositories/torchinterp1d/torchinterp1d'
# sys.path.insert(0, Interp_Path)
# import interp1d

###############################################################################

def Dabs_(x):
    ε = 1e-8   # Small Value to prevent Nans in backward pass
    y = F.smooth_l1_loss(x, tc.zeros_like(x), reduction='none', beta=ε) + ε
    return y


def Constrain_(M_, S_, E_, Q_, C_, A_):
    
    # Uncontrained Position
    m_ = M_
    
    # Bounded Aspect Ratio, Unbounded Scale
    k_max = 5       # No axes can be more than k times longer than another
    k = 1/(k_max-1)
    s1_s = (tc.sigmoid(S_[:, 0]) + k)
    s2_s = (tc.sigmoid(S_[:, 1]) + k)
    s3_s = (tc.sigmoid(S_[:, 2]) + k)
    s = tc.exp(S_[:, 3]) + 0.1                  
    s1 = s1_s * s
    s2 = s2_s * s
    s3 = s3_s * s
    s_ = tc.stack([s1, s2, s3], axis=-1)
    # s_ = tc.exp(S_) + 0.1

    e1 = tc.sigmoid(E_[:, 0]) * 1.8 + 0.1       # 0.1 < e1 < 1.9
    e2 = tc.sigmoid(E_[:, 1]) * 1.8 + 0.1       # 0.1 < e2 < 1.9
    e3 = tc.sigmoid(E_[:, 2]) * 4.9 + 0.1       # 0.1 < e3 < 5
    # e3 = tc.sigmoid(E_[:, 2])**0

    e_ = tc.stack([e1, e2, e3], axis=-1)
    
    q_ = Q_/tc.linalg.norm(Q_, axis=-1).reshape(-1, 1)
    c_ = tc.sigmoid(C_)                         # 0 < c < 1
    a_ = tc.sigmoid(A_)                         # 0 < a < 1
    
    return m_, s_, e_, q_, c_, a_


def Construct_(q_):
    
    q1, q2, q3, q4 = q_.T

    r11 = 1 - 2*q2**2 - 2*q3**2                            # (N)
    r12 = 2*q1*q2 - 2*q3*q4                                # (N)
    r13 = 2*q1*q3 + 2*q2*q4                                # (N)
    r21 = 2*q1*q2 + 2*q3*q4                                # (N)
    r22 = 1 - 2*q1**2 - 2*q3**2                            # (N)
    r23 = 2*q2*q3 - 2*q1*q4                                # (N)
    r31 = 2*q1*q3 - 2*q2*q4                                # (N)
    r32 = 2*q2*q3 + 2*q1*q4                                # (N)
    r33 = 1 - 2*q1**2 - 2*q2**2                            # (N)
    
    R_ = tc.stack([r11, r12, r13, 
                   r21, r22, r23, 
                   r31, r32, r33], 
                   dim=1).reshape(-1, 3, 3)                  # (N, 3, 3)

    return R_





def Interpolate1D_(x1, y1, x2, m_1=None, m_2=None, Interp='Linear'):
    # t0 = time.time()
    
    # x1: (B, N)
    # y1: (B, N, D)
    # x2: (B, S)
    # m_1: (B, N)
    # m_2: (B, S)
    
    # Dimensions
    B, N, D = y1.shape
    S = x2.shape[1]
    device = x1.device
    
    if m_1 == None:
        m_1 = tc.ones_like(x1, dtype=tc.bool, device=device)    # (B, N)
    c1 = m_1.sum().item()                                       # Total Reference Points

    if m_2 == None:
        m_2 = tc.ones_like(x2, dtype=tc.bool, device=device)    # (B, S)
        # m_2[:, S//2:] = False
        # m_2[:, 1:] = False
    c2 = m_2.sum().item()                                       # Total Sample Points

    # Insert Wrap-Around Points
    xs, Is = x1.min(axis=1, keepdims=True)                      # (B, 1) 2
    xe, Ie = x1.max(axis=1, keepdims=True)                      # (B, 1) 2
    xs = xs + 2*tc.pi                                           # (B, 1)
    xe = xe - 2*tc.pi                                           # (B, 1)
    ys = y1.gather(1, Is.view(B, 1, 1).repeat(1, 1, D))         # (B, 1, D)
    ye = y1.gather(1, Ie.view(B, 1, 1).repeat(1, 1, D))         # (B, 1, D)
    
    x1 = tc.cat([xs, xe, x1], axis=1)                           # (B, N+2)
    y1 = tc.cat([ys, ye, y1], axis=1)                           # (B, N+2, D)
    N += 2

    # Re-Sort Control Points
    x1, I0 = x1.sort(axis=1)                                    # (B, N+2)
    y1 = y1.gather(1, I0.view(B, N, 1).repeat(1, 1, D))         # (B, N+2, D)

    # Index of Closest Control Point. Binary Search
    IS = tc.ones((B, S), dtype=tc.int64, device=device) * N//2  # (B, S)

    for j in range(int(np.log2(N))):
        z = x1.gather(1, IS)                                    # (B, S)
        I = tc.zeros_like(m_2, dtype=tc.bool, device=device)    # (B, S)
        I[m_2] = x2[m_2] > z[m_2]                               # (B.S.)
        IS[m_2 & I] += N // 2**(j+2) + 1                         # (B.S.)
        IS[m_2 & ~I] -= N // 2**(j+2) + 1                        # (B.S.)
        IS[m_2] = IS[m_2].clip(0, N-1)                          # (B.S.)

    # Control Point Ahead or Behind?
    D_ = x1.gather(1, IS) - x2                                  # (B, S)

    # Next Control point
    IE = IS - tc.sign(D_).int()                                 # (B, S)
        
    if Interp == 'Linear':
        
        xs = x1.gather(1, IS)                                   # (B, S)
        xe = x1.gather(1, IE)                                   # (B, S)
        
        IS = IS[..., None].expand(-1, -1, D)                    # (B, S, D)
        IE = IE[..., None].expand(-1, -1, D)                    # (B, S, D)

        ys = y1.gather(1, IS)[m_2]                              # (B.S., D)
        ye = y1.gather(1, IE)[m_2]                              # (B.S., D)

        I = xe != xs                                            # (B.S.)
        w = tc.zeros_like(x2[m_2])                              # (B.S.)
        w[I] = (x2[I] - xs[I]) / (xe[I] - xs[I])                # (B.S.)
        w[~I] = tc.rand_like(w)[~I]                             # (B.S.)
        y2 = w[:, None] * (ye - ys) + ys                        # (B.S., D)
        
    elif Interp == 'Rim':
                
        IS = IS[..., None].expand(-1, -1, D)                    # (B, S, D)
        IE = IE[..., None].expand(-1, -1, D)                    # (B, S, D)
        
        r_ws = y1.gather(1, IS)[m_2]                            # (B.S., D)
        r_we = y1.gather(1, IE)[m_2]                            # (B.S., D)
        
        x_rws, y_rws, z_rws = r_ws.T                            # 3 (B.S.)
        x_rwe, y_rwe, z_rwe = r_we.T                            # 3 (B.S.)

        X_w = tc.cos(x2[m_2])                                   # (B.S.)
        Y_w = tc.sin(x2[m_2])                                   # (B.S.)
        
        w = Y_w * (x_rwe - x_rws) - X_w * (y_rwe - y_rws)       # (B.S.)
        
        I = tc.abs(w) > 1e-6
        
        w[I] = (X_w[I] * y_rws[I] - Y_w[I] * x_rws[I])  / w[I]  # (B.S.)
        w[~I] = tc.rand_like(w)[~I]                             # (B.S.)
      
        r_w2 = w[:, None] * (r_we - r_ws) + r_ws                # (B.S., D)  

        y2 = r_w2
        
    else:
        print('Failed')
        pass

    # try:
    #     del 
    # except: pass
    
    return y2

###############################################################################

def Superquadric_Distance_(Xs__, s_, e_):
    # Xs__  (B, 3)
    # s_    (B, 3)
    # s_    (B, 3)

    # t0 = time.time()
    
    xs__, ys__, zs__ = Xs__.T                               # (B,) x3
    s1, s2, s3 = s_.T                                       # (B,) x3
    e1, e2, e3 = e_.T                                       # (B,) x3
    
    d11 = xs__/s1                                           # (B,)
    d12 = ys__/s2                                           # (B,)
    d13 = zs__/s3                                           # (B,)

    d21 = Dabs_(d11)**(2/e2)                                # (B,)
    d22 = Dabs_(d12)**(2/e2)                                # (B,)
    d23 = Dabs_(d13)**(2/e1)                                # (B,)
    
    d31 = d21 + d22                                         # (B,)
    d32 = Dabs_(d31)**(e2/e1)                               # (B,)
    
    d41 = d32 + d23                                         # (B,)

    d41 = tc.clamp(d41, min=1e-8, max=10)                   # (B,)
    d__ = Dabs_(d41)**e3                                    # (B,)
    
    # t1 = time.time()
    # print('        Distance: ', round(t1-t0, 4))
    try:
        del d11, d12, d13, d21, d22, d23, d31, d32, d41
    except: pass
    
    return d__


def Rim_(s_, e_, R_cs):
    # s_    (n_g, 3)
    # e_    (n_g, 3)
    # R_cs  (n_g, 3, 3)
    
    # print(s_.shape, e_.shape, R_cs.shape)
    
    # t0 = time.time()
    dtype = s_.dtype
    device = s_.device

    # Hyper Parameters
    n_g = s_.shape[0]       # Number of Gaussians-Pixels
    n_r = 29                # Number of Rim Points. Currently Only works for n_r = 2^n - 3
    ε = 1e-10               # Small Value to prevent Nans in backward pass    
    
    # Parameters
    s1, s2, s3 = s_.T[..., None]                                # (n_g, 1) x3
    e1, e2, e3 = e_.T[..., None]                                # (n_g, 1) x3
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = R_cs.reshape(-1, 9).T[..., None]     # (n_g, 1) x9
    
    # Initialize Orthographic Polar Rim
    ω_r = tc.zeros((n_g, n_r)).to(dtype=dtype, device=device)   # (n_g, n_r)
    η_r = tc.zeros((n_g, n_r)).to(dtype=dtype, device=device)   # (n_g, n_r)
    
    # Orthographic Rim Sampling Degeneracy Condition
    m1 = (tc.abs(s1 / s3 * r33) < 0.2)[:, 0]                    # (n_g,) Steep Aspect Ratio
    m2 = (tc.abs(s2 / s3 * r33) < 0.2)[:, 0]                    # (n_g,) Steep Aspect Ratio
    m3 = (tc.abs(r33) < 0.2)[:, 0]                              # (n_g,) Shallow Angle
    m4 = ((tc.abs(s2 / s1) < 0.2) + (tc.abs(s1 / s2) < 0.2))[:, 0] # (n_g,) Steep Aspect Ratio
    m5 = (tc.abs(r23) < 0.01)[:, 0]                             # (n_g,) Double Degenerate: Tan Infinities
    
    m6 = m1 + m2 + m3 + m4                                      # (n_g,) Degenerate Cases
    m7 = ~m6                                                    # (n_g,) Regular Cases
    m8 = m6 * ~m5                                               # (n_g,) Single Degenerate Cases
    m9 = m6 * m5                                                # (n_g,) Double Degenerate Cases
    
    # Construct Double Degenerate Orthographic Rim
    ω_r[m9, :(n_r+1)//2] = tc.pi/2                              # (n_g, n_r)
    
    # Construct Single Degenerate Orthographic Rim
    S1 = s2[m8]/s1[m8] * r13[m8]/r23[m8]                        # (n_g,)
    S2 = tc.sign(S1) * Dabs_(S1)**(1/(2-e2[m8]))                # (n_g,)
    S3 = -tc.arctan(S2)                                         # (n_g,)
    
    ω_r[m8, :(n_r+1)//2] = S3                                   # (n_g, n_r)
    ω_r[m6, (n_r+1)//2:] = ω_r[m6, :(n_r-1)//2] + tc.pi         # (n_g, n_r)
    η_r[m6] = tc.linspace(-np.pi, np.pi, n_r).to(dtype=dtype, device=device) % tc.pi - tc.pi/2
     
    # Construct Regular Orthographic Rim
    ω_r[m7] = tc.linspace(-np.pi, np.pi, n_r).to(dtype=dtype, device=device)
    
    c_ωr = tc.cos(ω_r)                                          # (n_g, n_r)
    s_ωr = tc.sin(ω_r)                                          # (n_g, n_r)
    c_ωr2e2 = tc.sign(c_ωr) * Dabs_(c_ωr)**(2-e2)               # (n_g, n_r)
    s_ωr2e2 = tc.sign(s_ωr) * Dabs_(s_ωr)**(2-e2)               # (n_g, n_r)
    c_ωre2 = tc.sign(c_ωr) * Dabs_(c_ωr)**e2                    # (n_g, n_r)
    s_ωre2 = tc.sign(s_ωr) * Dabs_(s_ωr)**e2                    # (n_g, n_r)
    
    # Equation 2.49
    η_r[m7] = (-s3[m7]/r33[m7] * (r13[m7]/s1[m7] * c_ωr2e2[m7] + r23[m7]/s2[m7] * s_ωr2e2[m7]))     # (n_g, n_r)
    η_r[m7] = tc.sign(η_r[m7]) * Dabs_(η_r[m7])**(1/(2 - e1[m7]))  # (n_g, n_r)
    η_r[m7] = tc.arctan(η_r[m7])                                # (n_g, n_r)
    
    # Convert Polar Rim back to Cartesian Shape Coordinates
    c_ηr = tc.cos(η_r)                                          # (n_g, n_r)
    s_ηr = tc.sin(η_r)                                          # (n_g, n_r)
    c_ηre1 = tc.sign(c_ηr) * Dabs_(c_ηr)**e1                    # (n_g, n_r)
    s_ηre1 = tc.sign(s_ηr) * Dabs_(s_ηr)**e1                    # (n_g, n_r)
    
    x_rs = s1 * c_ηre1 * c_ωre2                                 # (n_g, n_r)
    y_rs = s2 * c_ηre1 * s_ωre2                                 # (n_g, n_r)
    z_rs = s3 * s_ηre1                                          # (n_g, n_r)
    r_s = tc.stack([x_rs, y_rs, z_rs], axis=-1)                 # (n_g, n_r, 3)
    
    # Convert to Camera Coordinates
    x_rc = r11 * x_rs + r21 * y_rs + r31 * z_rs                 # (n_g, n_r)
    y_rc = r12 * x_rs + r22 * y_rs + r32 * z_rs                 # (n_g, n_r)
    z_rc = r13 * x_rs + r23 * y_rs + r33 * z_rs                 # (n_g, n_r)
    r_c = tc.stack([x_rc, y_rc, z_rc], axis=-1)                 # (n_g, n_r, 3)

    # Polar Camera Coordinates Centered 
    θ_r = tc.arctan2(y_rc, x_rc)                                # (n_g, n_r)
    Φ_r = tc.arctan(z_rc /(x_rc**2 + y_rc**2)**0.5)             # (n_g, n_r)
    
    # t1 = time.time()
    # print('        Rim: ', round(t1-t0, 4))
    
    return ω_r, η_r, r_s, r_c, θ_r, Φ_r
    

def Superquadric_Tile(Xc__, s_, e_, R_cs, m, Show=False):
    # Xc__  (T, L, h, w, 2)
    # s_    (U, 3)
    # e_    (U, 3)
    # R_cs  (U, 3, 3)
    # m
    
    T, l, h, w, _ = Xc__.shape
    m0, m1, m2, m3, m4, m5, m6, m7, m8, I0, I1, I2 = m

    # Rim - Reference Points
    θ_r, r_c = Rim_(s_, e_, R_cs)[4:2:-1]                   # (U, N_r) (U, N_r, 3)
    
    θ_r = θ_r[I2][m5]                                       # (T.l, N_r)    
    r_c = r_c[I2][m5]                                       # (T.l, N_r, 3)   
    
    # Grid - Sample Points
    Xc__ = Xc__[m5]                                         # (T.l., h, w, 2)
    Xc__ = Xc__.reshape(-1, h*w, 2)                         # (T.l., h*w, 2)
    xc__, yc__ = Xc__[..., 0], Xc__[..., 1]                 # (T.l., h*w) x2

    # Convert Frame to Polar
    R__ = (xc__**2 + yc__**2)**0.5                          # (T.l., h*w)
    θ__ = tc.arctan2(yc__, xc__)                            # (T.l., h*w)

    # Interpolate Rim in Cartesian Coordinates. (B, N) (B, N, D) (B, S)
    θ__ = θ__.reshape(-1, h*w)                              # (T.l., h*w)
    # m8 = m8.reshape(-1, h*w)                              # (T.l., h*w)
    r_c2 = Interpolate1D_(θ_r, r_c, θ__, None, m8, 'Rim')   # (T.l.h.w., 3)
    xrc2__, yrc2__, zrc2__ = r_c2.T                         # (T.l.h.w.) x3

    zc__ = zrc2__ * (xc__[m8]**2 + yc__[m8]**2)**0.5 / (xrc2__**2 + yrc2__**2)**0.5   # (T.l.h.w)

    Xc__ = Xc__[m8]                                         # (T.l.h.w., 2)
    Xc__ = tc.cat([Xc__, zc__[..., None]], axis=-1)         # (T.l.h.w., 3)
    
    s_ = s_[I2][m5][:, None].expand(-1, h*w, -1)[m8]        # (T.l.h.w., 3)    
    e_ = e_[I2][m5][:, None].expand(-1, h*w, -1)[m8]        # (T.l.h.w., 3)    
    R_cs = R_cs[I2][m5][:, None].expand(-1, h*w, -1, -1)[m8]# (T.l.h.w., 3, 3)                                  # (T.l, 3, 3)
    
    # Convert to Shape Coordinates
    Xs__ = (R_cs @ Xc__[..., None])[..., 0]                 # (T.l.h.w, 3) 
    
    # Calculate Shape Distance Function
    d__ =  Superquadric_Distance_(Xs__, s_, e_)             # (T.l.h.w.)
    
    # Gaussianize
    G__ = tc.exp(-d__)                                      # (T.l.h.w.)
    
    return G__
                
###############################################################################



