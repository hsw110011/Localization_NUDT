#!/usr/bin/env python3
"""GlobalPose DSM-BEV 匹配分数 — 区分度增强版。

单候选: score = exp(-J_total / tau)
多候选: score_i = exp(-(J_total_i - J_min) / tau)
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DsmBevScoreConfig(object):
    alpha_h: float = 0.65
    alpha_gx: float = 0.25
    alpha_gy: float = 0.10

    ground_threshold: float = 0.20
    hmax_min: float = 5.0
    hmax_max: float = 40.0
    height_percentile: float = 98.0
    min_valid_height_points: int = 10

    lambda_lidar_higher: float = 1.5
    lambda_dsm_higher: float = 0.3
    delta_h: float = 0.30
    w_h_base: float = 0.2
    w_h_height: float = 0.8

    delta_g: float = 0.30
    w_g_base: float = 0.05

    tau: float = 0.10
    eps: float = 1e-6

    grad_cap: float = 3.0
    grad_mask_erode_px: int = 1

    dsm_long_sign: float = 1.0
    dsm_lat_sign: float = 1.0


def erode_obs_mask(mask, pixels=1):
    """3x3 二值腐蚀，去掉 M_obs 边界带。"""
    m = np.asarray(mask, dtype=bool)
    px = int(pixels)
    if px <= 0:
        return m.copy()

    def _shift(arr, dr, dc):
        out = np.zeros_like(arr)
        r0, r1 = max(0, dr), arr.shape[0] + min(0, dr)
        c0, c1 = max(0, dc), arr.shape[1] + min(0, dc)
        sr0, sr1 = max(0, -dr), max(0, -dr) + (r1 - r0)
        sc0, sc1 = max(0, -dc), max(0, -dc) + (c1 - c0)
        out[r0:r1, c0:c1] = arr[sr0:sr1, sc0:sc1]
        return out

    out = m
    for _ in range(px):
        eroded = out
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                eroded = eroded & _shift(out, dr, dc)
        out = eroded
    return out


def cap_gradient_layer(layer, cap):
    arr = np.asarray(layer, dtype=np.float64)
    out = arr.copy()
    finite = np.isfinite(out)
    limit = max(float(cap), 1e-6)
    out[finite] = np.clip(out[finite], -limit, limit)
    return out


def normalize_gradient_layer(layer, cap):
    """G_c = clip(G, ±cap) / cap。"""
    return cap_gradient_layer(layer, cap) / max(float(cap), 1e-6)


def huber(error, delta):
    error = np.asarray(error, dtype=np.float64)
    abs_e = np.abs(error)
    quad = 0.5 * error * error
    lin = float(delta) * (abs_e - 0.5 * float(delta))
    return np.where(abs_e <= float(delta), quad, lin)


def compute_h_max(H_L, m_obs, config=None):
    if config is None:
        config = DsmBevScoreConfig()

    h_l = np.asarray(H_L, dtype=np.float64)
    m = np.asarray(m_obs, dtype=bool)
    valid_h = m & np.isfinite(h_l) & (h_l > float(config.ground_threshold))

    if int(valid_h.sum()) < int(config.min_valid_height_points):
        return float(config.hmax_min)

    h_max = float(np.percentile(h_l[valid_h], float(config.height_percentile)))
    if not np.isfinite(h_max):
        return float(config.hmax_min)

    return float(np.clip(h_max, config.hmax_min, config.hmax_max))


def build_obs_mask(M_obs, M_dsm=None):
    m = np.isfinite(M_obs) & (np.asarray(M_obs, dtype=np.float64) > 0.5)
    if M_dsm is not None:
        m = m & np.isfinite(M_dsm) & (np.asarray(M_dsm, dtype=np.float64) > 0.5)
    return m


def apply_obs_mask(layer, m_obs):
    arr = np.asarray(layer, dtype=np.float64).copy()
    arr[~np.asarray(m_obs, dtype=bool)] = np.nan
    return arr


def apply_bev_mask_to_layers(H_L, Gx_L, Gy_L, H_D, Gx_D, Gy_D, M_obs, M_dsm=None):
    m_obs = build_obs_mask(M_obs, M_dsm)
    return m_obs, {
        "H_L": apply_obs_mask(H_L, m_obs),
        "Gx_L": apply_obs_mask(Gx_L, m_obs),
        "Gy_L": apply_obs_mask(Gy_L, m_obs),
        "H_D": apply_obs_mask(H_D, m_obs),
        "Gx_D": apply_obs_mask(Gx_D, m_obs),
        "Gy_D": apply_obs_mask(Gy_D, m_obs),
    }


def _weighted_mean_over_obs(values, weights, m_obs, eps=1e-6):
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    m = np.asarray(m_obs, dtype=bool)
    valid = m & np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if not valid.any():
        return 0.0
    return float(v[valid].sum() / (w[valid].sum() + float(eps)))


def normalize_height(H, h_max):
    h_max = max(float(h_max), 1e-6)
    h = np.asarray(H, dtype=np.float64)
    out = np.full_like(h, np.nan, dtype=np.float64)
    valid = np.isfinite(h)
    out[valid] = np.clip(h[valid], 0.0, h_max) / h_max
    return out


def normalize_candidate_scores(j_totals, config=None):
    """多候选组内 score: exp(-(J_i - J_min)/tau)。"""
    if config is None:
        config = DsmBevScoreConfig()
    j_arr = np.asarray([float(j["J_total"]) for j in j_totals], dtype=np.float64)
    j_min = float(np.min(j_arr)) if j_arr.size else 0.0
    tau = max(float(config.tau), float(config.eps))
    out = []
    for j_dict in j_totals:
        j_total = float(j_dict["J_total"])
        norm_score = float(np.exp(-(j_total - j_min) / tau))
        item = dict(j_dict)
        item["score"] = norm_score
        item["J_min_group"] = j_min
        out.append(item)
    return out


def compute_dsm_bev_score(
    H_L,
    Gx_L,
    Gy_L,
    M_obs,
    H_D,
    Gx_D,
    Gy_D,
    config=None,
    M_dsm=None,
    h_max=None,
):
    if config is None:
        config = DsmBevScoreConfig()

    if not (
        np.asarray(H_L).shape
        == np.asarray(H_D).shape
        == np.asarray(Gx_L).shape
        == np.asarray(Gy_L).shape
        == np.asarray(Gx_D).shape
        == np.asarray(Gy_D).shape
    ):
        raise ValueError("layer shape mismatch for DSM-BEV score inputs")

    m_obs, masked = apply_bev_mask_to_layers(
        H_L, Gx_L, Gy_L, H_D, Gx_D, Gy_D, M_obs, M_dsm=M_dsm
    )
    valid_pixel_num = int(m_obs.sum())
    if valid_pixel_num == 0:
        return {
            "J_h": 0.0,
            "J_gx": 0.0,
            "J_gy": 0.0,
            "J_total": 0.0,
            "score": 1.0,
            "weight": 1.0,
            "H_max": float(config.hmax_min),
            "valid_pixel_num": 0,
            "valid_grad_pixel_num": 0,
            "H_L_n": masked["H_L"].astype(np.float32),
            "H_D_n": masked["H_D"].astype(np.float32),
            "e_h": np.zeros_like(masked["H_L"], dtype=np.float32),
            "P_h": np.zeros_like(masked["H_L"], dtype=np.float32),
            "e_gx": np.zeros_like(masked["Gx_L"], dtype=np.float32),
            "e_gy": np.zeros_like(masked["Gy_L"], dtype=np.float32),
            "M_obs": m_obs.astype(np.float32),
            "M_grad": np.zeros_like(m_obs, dtype=np.float32),
        }

    h_l = masked["H_L"]
    h_d = masked["H_D"]

    if h_max is None:
        h_max = compute_h_max(h_l, m_obs, config)
    else:
        h_max = float(np.clip(float(h_max), config.hmax_min, config.hmax_max))

    h_l_n = normalize_height(h_l, h_max)
    h_d_n = normalize_height(h_d, h_max)

    gx_l = normalize_gradient_layer(masked["Gx_L"], config.grad_cap)
    gy_l = normalize_gradient_layer(masked["Gy_L"], config.grad_cap)
    gx_d = normalize_gradient_layer(masked["Gx_D"], config.grad_cap)
    gy_d = normalize_gradient_layer(masked["Gy_D"], config.grad_cap)

    m_grad = erode_obs_mask(m_obs, config.grad_mask_erode_px)
    m_grad = (
        m_grad
        & np.isfinite(gx_l)
        & np.isfinite(gx_d)
        & np.isfinite(gy_l)
        & np.isfinite(gy_d)
    )

    e_h = h_l_n - h_d_n
    lambda_h = np.where(
        h_l_n > h_d_n,
        float(config.lambda_lidar_higher),
        float(config.lambda_dsm_higher),
    )
    w_h = m_obs.astype(np.float64) * (
        float(config.w_h_base) + float(config.w_h_height) * np.nan_to_num(h_l_n, nan=0.0)
    )
    p_h = w_h * lambda_h * huber(e_h, config.delta_h)
    j_h = _weighted_mean_over_obs(p_h, w_h, m_obs, config.eps)

    e_gx = gx_l - gx_d
    w_gx = np.where(
        m_grad,
        float(config.w_g_base)
        + (1.0 - float(config.w_g_base)) * np.abs(np.nan_to_num(gx_l, nan=0.0)),
        0.0,
    )
    p_gx = w_gx * huber(e_gx, config.delta_g)
    j_gx = _weighted_mean_over_obs(p_gx, w_gx, m_grad, config.eps)

    e_gy = gy_l - gy_d
    w_gy = np.where(
        m_grad,
        float(config.w_g_base)
        + (1.0 - float(config.w_g_base)) * np.abs(np.nan_to_num(gy_l, nan=0.0)),
        0.0,
    )
    p_gy = w_gy * huber(e_gy, config.delta_g)
    j_gy = _weighted_mean_over_obs(p_gy, w_gy, m_grad, config.eps)

    j_total = (
        float(config.alpha_h) * j_h
        + float(config.alpha_gx) * j_gx
        + float(config.alpha_gy) * j_gy
    )
    score = float(np.exp(-j_total / max(float(config.tau), config.eps)))

    return {
        "J_h": j_h,
        "J_gx": j_gx,
        "J_gy": j_gy,
        "J_total": j_total,
        "score": score,
        "weight": 1.0,
        "H_max": float(h_max),
        "valid_pixel_num": valid_pixel_num,
        "valid_grad_pixel_num": int(m_grad.sum()),
        "H_L_n": h_l_n.astype(np.float32),
        "H_D_n": h_d_n.astype(np.float32),
        "e_h": e_h.astype(np.float32),
        "P_h": np.where(m_obs, p_h, 0.0).astype(np.float32),
        "W_h": np.where(m_obs, w_h, 0.0).astype(np.float32),
        "e_gx": e_gx.astype(np.float32),
        "e_gy": e_gy.astype(np.float32),
        "M_obs": m_obs.astype(np.float32),
        "M_grad": m_grad.astype(np.float32),
    }


def dsm_layers_to_score_inputs(dsm_layers, config=None):
    if config is None:
        config = DsmBevScoreConfig()
    return {
        "H_D": dsm_layers["H_rel_surf"],
        "Gx_D": float(config.dsm_lat_sign) * dsm_layers["G_lat_L"],
        "Gy_D": float(config.dsm_long_sign) * dsm_layers["G_long_L"],
    }
