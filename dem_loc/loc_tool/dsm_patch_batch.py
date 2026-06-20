#!/usr/bin/env python3
"""DSM patch GPU batch 采样 — wrapper，语义与 DsmPatchCropper 单样本一致。"""

import torch
import torch.nn.functional as F

from .dsm_patch import _SOBEL_X, _SOBEL_Y, _apply_deadzone_torch


def build_sample_grid_batch(cropper, particles):
    """particles: [N,3] (gauss_x, gauss_y, yaw_rad) on cropper.device -> grid [N,H,W,2]."""
    if particles.ndim != 2 or particles.shape[1] != 3:
        raise ValueError("particles must be [N, 3]")

    device = cropper.device
    particles = particles.to(device=device, dtype=torch.float32)
    n = int(particles.shape[0])

    x = particles[:, 0].view(n, 1, 1)
    y = particles[:, 1].view(n, 1, 1)
    theta = particles[:, 2].view(n, 1, 1)
    cos_h = torch.cos(theta)
    sin_h = torch.sin(theta)

    gauss_x = x + cropper._x_vehicle * cos_h - cropper._y_vehicle * sin_h
    gauss_y = y + cropper._x_vehicle * sin_h + cropper._y_vehicle * cos_h

    cc = cropper.cc
    u = (gauss_x - cc.tl_gauss_x) / cc.res_gauss_x
    v = -(gauss_y - cc.tl_gauss_y) / cc.res_gauss_y

    dem_w_m1 = max(float(cropper.dem_w - 1), 1.0)
    dem_h_m1 = max(float(cropper.dem_h - 1), 1.0)
    grid_x = 2.0 * u / dem_w_m1 - 1.0
    grid_y = 2.0 * v / dem_h_m1 - 1.0
    return torch.stack((grid_x, grid_y), dim=-1)


def crop_batch(cropper, particles):
    """Batch grid_sample -> elevation patches [N, 1, H, W]."""
    grid = build_sample_grid_batch(cropper, particles)
    n = int(grid.shape[0])
    dem = cropper.dem_tensor.expand(n, -1, -1, -1)
    patches = F.grid_sample(
        dem,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return patches, grid


def _median_filter_3x3_batch(layer_batch):
    """layer_batch: [N,H,W] -> [N,H,W]."""
    x = layer_batch.unsqueeze(1)
    padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    patches = padded.unfold(2, 3, 1).unfold(3, 3, 1)
    n, _, h, w, _, _ = patches.shape
    return patches.reshape(n, 1, h, w, 9).median(dim=-1).values.squeeze(1)


def compute_bev_layers_batch(
    patch_batch,
    resolution,
    sobel_x,
    sobel_y,
    h_rel_deadzone_half=0.20,
    grad_deadzone_half=0.15,
    grad_cap=3.0,
):
    """patch_batch: [N,H,W] -> H_rel, G_long, G_lat each [N,H,W]."""
    if patch_batch.ndim != 3:
        raise ValueError("patch_batch must be [N, H, W]")

    n, rows, cols = patch_batch.shape
    center = patch_batch[:, rows // 2, cols // 2]
    if not torch.isfinite(center).all():
        center = torch.where(
            torch.isfinite(center),
            center,
            torch.nanmedian(patch_batch.reshape(n, -1), dim=1).values,
        )

    h_rel = _apply_deadzone_torch(
        patch_batch - center.view(n, 1, 1), h_rel_deadzone_half
    )
    smoothed = _median_filter_3x3_batch(h_rel)
    s = smoothed.unsqueeze(1)
    grad_index0 = F.conv2d(s, sobel_y, padding=1).squeeze(1)
    grad_index1 = F.conv2d(s, sobel_x, padding=1).squeeze(1)
    scale = 8.0 * max(float(resolution), 1e-3)
    g_long = _apply_deadzone_torch(-grad_index0 / scale, grad_deadzone_half)
    g_lat = _apply_deadzone_torch(-grad_index1 / scale, grad_deadzone_half)
    cap = max(float(grad_cap), 1e-6)
    g_long = torch.clamp(g_long, -cap, cap)
    g_lat = torch.clamp(g_lat, -cap, cap)
    return h_rel, g_long, g_lat


def crop_batch_with_bev_layers(cropper, particles, grad_cap=3.0):
    """Batch DSM patch + BEV layers for score batch."""
    patches, grid = crop_batch(cropper, particles)
    patch_2d = patches.squeeze(1)
    h_rel, g_long, g_lat = compute_bev_layers_batch(
        patch_2d,
        cropper.resolution,
        cropper._sobel_x,
        cropper._sobel_y,
        h_rel_deadzone_half=cropper.h_rel_deadzone_half,
        grad_deadzone_half=cropper.grad_deadzone_half,
        grad_cap=grad_cap,
    )
    return {
        "patches": patches,
        "grid": grid,
        "H_rel_surf": h_rel,
        "G_long_L": g_long,
        "G_lat_L": g_lat,
    }
