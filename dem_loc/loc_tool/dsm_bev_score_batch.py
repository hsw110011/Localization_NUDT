#!/usr/bin/env python3
"""DSM-BEV score GPU batch 计算 — 公式与 dsm_bev_score.py 一致，不修改原语义。"""

import torch

import numpy as np

from .dsm_bev_score import DsmBevScoreConfig, compute_h_max, erode_obs_mask


def _huber_torch(error, delta):
    abs_e = error.abs()
    quad = 0.5 * error * error
    lin = float(delta) * (abs_e - 0.5 * float(delta))
    return torch.where(abs_e <= float(delta), quad, lin)


def _cap_gradient_torch(layer, cap):
    limit = max(float(cap), 1e-6)
    return torch.clamp(layer, -limit, limit)


def _normalize_gradient_torch(layer, cap):
    return _cap_gradient_torch(layer, cap) / max(float(cap), 1e-6)


def _normalize_height_torch(h, h_max):
    h_max = max(float(h_max), 1e-6)
    out = torch.clamp(h, 0.0, h_max) / h_max
    out = torch.where(torch.isfinite(h), out, torch.full_like(out, float("nan")))
    return out


def _sanitize_h_max(h_max, config):
    try:
        value = float(h_max)
    except (TypeError, ValueError):
        return float(config.hmax_min)
    if not np.isfinite(value):
        return float(config.hmax_min)
    return float(np.clip(value, config.hmax_min, config.hmax_max))


def _weighted_mean_batch(values, weights, mask, eps):
    """values/weights/mask: [N,H,W] -> [N]。"""
    valid = mask & torch.isfinite(values) & torch.isfinite(weights) & (weights > 0.0)
    # values 已经是加权后的 penalty（例如 p_h = w_h * loss），这里与
    # dsm_bev_score._weighted_mean_over_obs 保持一致：sum(values) / sum(weights)。
    # 不能用 values * valid.float()：mask 外 NaN * 0 仍为 NaN，会污染 sum。
    num = torch.where(valid, values, torch.zeros_like(values)).sum(dim=(1, 2))
    den = torch.where(valid, weights, torch.zeros_like(weights)).sum(dim=(1, 2)) + float(eps)
    return num / den


def _erode_mask_torch(mask, pixels, device):
    """mask [H,W] bool -> [H,W] bool，复用 numpy erode_obs_mask 语义。"""
    if int(pixels) <= 0:
        return mask.to(device=device)
    m_np = erode_obs_mask(mask.detach().cpu().numpy(), int(pixels))
    return torch.from_numpy(m_np).to(device=device)


def lidar_layers_to_torch(lidar_layers, device):
    out = {}
    for key in ("H_L", "Gx_L", "Gy_L", "M_obs"):
        out[key] = torch.as_tensor(lidar_layers[key], dtype=torch.float32, device=device)
    return out


def expand_lidar_to_batch(lidar_tensor, batch_size):
    """[H,W] -> [N,H,W] via expand (no extra memory copy)."""
    n = int(batch_size)
    return lidar_tensor.unsqueeze(0).expand(n, -1, -1)


def compute_dsm_bev_score_batch(
    H_L,
    Gx_L,
    Gy_L,
    M_obs,
    H_D,
    Gx_D,
    Gy_D,
    config=None,
    h_max=None,
):
    """Batch score: LiDAR [H,W], DSM [N,H,W] -> J_total/score [N] on same device."""
    if config is None:
        config = DsmBevScoreConfig()

    device = H_D.device
    n, h, w = H_D.shape
    if (
        H_L.shape != (h, w)
        or Gx_L.shape != (h, w)
        or Gy_L.shape != (h, w)
        or M_obs.shape != (h, w)
        or Gx_D.shape != (n, h, w)
        or Gy_D.shape != (n, h, w)
    ):
        raise ValueError("batch layer shape mismatch for DSM-BEV score")

    m_obs = torch.isfinite(M_obs) & (M_obs > 0.5)
    valid_pixel_num = int(m_obs.sum().item())
    if valid_pixel_num == 0 or n == 0:
        zeros = torch.zeros(n, device=device, dtype=torch.float32)
        ones = torch.ones(n, device=device, dtype=torch.float32)
        return {
            "J_h": zeros.clone(),
            "J_gx": zeros.clone(),
            "J_gy": zeros.clone(),
            "J_total": zeros.clone(),
            "score": ones.clone(),
            "H_max": torch.full((n,), float(config.hmax_min), device=device),
            "valid_grad_pixel_num": torch.zeros(n, device=device, dtype=torch.int32),
        }

    h_l = torch.where(m_obs, H_L, torch.full_like(H_L, float("nan")))
    h_d = torch.where(m_obs.unsqueeze(0), H_D, torch.full_like(H_D, float("nan")))
    gx_l = torch.where(m_obs, Gx_L, torch.full_like(Gx_L, float("nan")))
    gy_l = torch.where(m_obs, Gy_L, torch.full_like(Gy_L, float("nan")))
    gx_d = torch.where(m_obs.unsqueeze(0), Gx_D, torch.full_like(Gx_D, float("nan")))
    gy_d = torch.where(m_obs.unsqueeze(0), Gy_D, torch.full_like(Gy_D, float("nan")))

    if h_max is None:
        h_max_val = compute_h_max(H_L.detach().cpu().numpy(), m_obs.detach().cpu().numpy(), config)
    else:
        h_max_val = _sanitize_h_max(h_max, config)

    h_max_val = max(float(h_max_val), 1e-6)

    h_l_n = _normalize_height_torch(h_l, h_max_val)
    h_d_n = _normalize_height_torch(h_d, h_max_val)

    gx_l_n = _normalize_gradient_torch(gx_l, config.grad_cap)
    gy_l_n = _normalize_gradient_torch(gy_l, config.grad_cap)
    gx_d_n = _normalize_gradient_torch(gx_d, config.grad_cap)
    gy_d_n = _normalize_gradient_torch(gy_d, config.grad_cap)

    m_grad = _erode_mask_torch(m_obs, config.grad_mask_erode_px, device)
    m_grad_b = m_grad.unsqueeze(0).expand(n, -1, -1)
    m_grad_b = (
        m_grad_b
        & torch.isfinite(gx_l_n).unsqueeze(0).expand(n, -1, -1)
        & torch.isfinite(gx_d_n)
        & torch.isfinite(gy_l_n).unsqueeze(0).expand(n, -1, -1)
        & torch.isfinite(gy_d_n)
    )
    m_obs_b = m_obs.unsqueeze(0).expand(n, -1, -1)

    e_h = h_l_n.unsqueeze(0).expand(n, -1, -1) - h_d_n
    lambda_h = torch.where(
        h_l_n.unsqueeze(0).expand(n, -1, -1) > h_d_n,
        torch.tensor(float(config.lambda_lidar_higher), device=device),
        torch.tensor(float(config.lambda_dsm_higher), device=device),
    )
    w_h = m_obs_b.float() * (
        float(config.w_h_base)
        + float(config.w_h_height)
        * torch.nan_to_num(h_l_n, nan=0.0).unsqueeze(0).expand(n, -1, -1)
    )
    p_h = w_h * lambda_h * _huber_torch(e_h, config.delta_h)
    j_h = _weighted_mean_batch(p_h, w_h, m_obs_b, config.eps)
    j_h = torch.nan_to_num(j_h, nan=0.0, posinf=1e6, neginf=1e6)

    e_gx = gx_l_n.unsqueeze(0).expand(n, -1, -1) - gx_d_n
    w_gx = torch.where(
        m_grad_b,
        float(config.w_g_base)
        + (1.0 - float(config.w_g_base))
        * torch.abs(torch.nan_to_num(gx_l_n, nan=0.0)).unsqueeze(0).expand(n, -1, -1),
        torch.zeros((), device=device),
    )
    p_gx = w_gx * _huber_torch(e_gx, config.delta_g)
    j_gx = _weighted_mean_batch(p_gx, w_gx, m_grad_b, config.eps)
    j_gx = torch.nan_to_num(j_gx, nan=0.0, posinf=1e6, neginf=1e6)

    e_gy = gy_l_n.unsqueeze(0).expand(n, -1, -1) - gy_d_n
    w_gy = torch.where(
        m_grad_b,
        float(config.w_g_base)
        + (1.0 - float(config.w_g_base))
        * torch.abs(torch.nan_to_num(gy_l_n, nan=0.0)).unsqueeze(0).expand(n, -1, -1),
        torch.zeros((), device=device),
    )
    p_gy = w_gy * _huber_torch(e_gy, config.delta_g)
    j_gy = _weighted_mean_batch(p_gy, w_gy, m_grad_b, config.eps)
    j_gy = torch.nan_to_num(j_gy, nan=0.0, posinf=1e6, neginf=1e6)

    j_total = (
        float(config.alpha_h) * j_h
        + float(config.alpha_gx) * j_gx
        + float(config.alpha_gy) * j_gy
    )
    j_total = torch.nan_to_num(j_total, nan=1e6, posinf=1e6, neginf=1e6)
    tau = max(float(config.tau), float(config.eps))
    score = torch.exp(-j_total / tau)

    return {
        "J_h": j_h,
        "J_gx": j_gx,
        "J_gy": j_gy,
        "J_total": j_total,
        "score": score,
        "H_max": torch.full((n,), float(h_max_val), device=device, dtype=torch.float32),
        "valid_grad_pixel_num": m_grad_b.sum(dim=(1, 2)).to(torch.int32),
    }
