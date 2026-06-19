#!/usr/bin/env python3
"""GlobalPose / LocalPose 双位姿 DSM-BEV 分数计算（库函数，无 ROS 依赖）。"""

import math

from .cinterface import parse_lidar_bev_grid_map
from .dsm_bev_score import (
    DsmBevScoreConfig,
    build_obs_mask,
    compute_dsm_bev_score,
    compute_h_max,
    dsm_layers_to_score_inputs,
    normalize_candidate_scores,
)

PERTURB_DX_M = (-4.0, -2.0, 0.0, 2.0, 4.0)
PERTURB_DY_M = (-4.0, -2.0, 0.0, 2.0, 4.0)
PERTURB_DTHETA_DEG = (-4.0, 0.0, 4.0)


def global_pose_gauss_theta(
    coord_converter,
    global_pose,
    heading_unit="deg",
    heading_convention="math",
):
    gauss_x, gauss_y, heading_math_deg = coord_converter._extract_global_gauss_heading(
        global_pose,
        heading_unit=heading_unit,
        heading_convention=heading_convention,
    )
    return float(gauss_x), float(gauss_y), math.radians(float(heading_math_deg))


def local_pose_gauss_theta(
    coord_converter,
    local_pose,
    local_pose_base,
    local_heading_unit="deg",
    local_heading_convention="math",
):
    global_point = coord_converter.LocalPoseToGlobal(
        local_pose,
        local_pose_base,
        local_heading_unit=local_heading_unit,
        local_heading_convention=local_heading_convention,
    )
    return (
        float(global_point.gauss.x),
        float(global_point.gauss.y),
        math.radians(float(global_point.heading)),
    )


def perturb_pose_gauss_theta(gauss_x, gauss_y, theta, dx_m, dy_m, dtheta_deg):
    """车体坐标系平移 (dx, dy) + 航向扰动 dtheta。"""
    c = math.cos(theta)
    s = math.sin(theta)
    px = float(gauss_x) + float(dx_m) * c - float(dy_m) * s
    py = float(gauss_y) + float(dx_m) * s + float(dy_m) * c
    return px, py, float(theta) + math.radians(float(dtheta_deg))


def score_at_pose(dsm_cropper, gauss_x, gauss_y, theta, lidar_layers, config=None, h_max=None):
    """在指定位姿裁剪 DSM patch 并与 LiDAR BEV 算分。"""
    if config is None:
        config = DsmBevScoreConfig()

    dsm_result = dsm_cropper.crop_with_bev_layers(gauss_x, gauss_y, theta)
    dsm_inputs = dsm_layers_to_score_inputs(dsm_result["layers"], config)
    score = compute_dsm_bev_score(
        H_L=lidar_layers["H_L"],
        Gx_L=lidar_layers["Gx_L"],
        Gy_L=lidar_layers["Gy_L"],
        M_obs=lidar_layers["M_obs"],
        H_D=dsm_inputs["H_D"],
        Gx_D=dsm_inputs["Gx_D"],
        Gy_D=dsm_inputs["Gy_D"],
        config=config,
        h_max=h_max,
    )
    return score, dsm_result


def score_global_and_local(
    coord_converter,
    dsm_cropper,
    global_pose,
    local_pose,
    local_pose_base,
    bev_msg,
    score_config=None,
    global_heading_unit="deg",
    global_heading_convention="math",
    local_heading_unit="deg",
    local_heading_convention="math",
):
    """同一帧 LiDAR BEV 下，分别在 GlobalPose / LocalPose 位置算分。"""
    if score_config is None:
        score_config = DsmBevScoreConfig()

    info = bev_msg.info
    dsm_cropper.set_geometry(
        float(info.length_x), float(info.length_y), float(info.resolution)
    )
    lidar_layers = parse_lidar_bev_grid_map(bev_msg)
    m_obs = build_obs_mask(lidar_layers["M_obs"])
    h_max = compute_h_max(lidar_layers["H_L"], m_obs, score_config)

    g_gauss_x, g_gauss_y, g_theta = global_pose_gauss_theta(
        coord_converter,
        global_pose,
        heading_unit=global_heading_unit,
        heading_convention=global_heading_convention,
    )
    global_score, global_dsm_result = score_at_pose(
        dsm_cropper, g_gauss_x, g_gauss_y, g_theta, lidar_layers, score_config, h_max=h_max
    )

    l_gauss_x, l_gauss_y, l_theta = local_pose_gauss_theta(
        coord_converter,
        local_pose,
        local_pose_base,
        local_heading_unit=local_heading_unit,
        local_heading_convention=local_heading_convention,
    )
    local_score, local_dsm_result = score_at_pose(
        dsm_cropper, l_gauss_x, l_gauss_y, l_theta, lidar_layers, score_config, h_max=h_max
    )

    return {
        "lidar_layers": lidar_layers,
        "h_max": h_max,
        "global": {
            "gauss_x": g_gauss_x,
            "gauss_y": g_gauss_y,
            "theta": g_theta,
            "score": global_score,
            "dsm_result": global_dsm_result,
        },
        "local": {
            "gauss_x": l_gauss_x,
            "gauss_y": l_gauss_y,
            "theta": l_theta,
            "score": local_score,
            "dsm_result": local_dsm_result,
        },
        "delta_score": float(local_score["score"] - global_score["score"]),
    }


def score_perturbation_grid(
    dsm_cropper,
    gauss_x,
    gauss_y,
    theta,
    lidar_layers,
    score_config=None,
    h_max=None,
    dx_values=None,
    dy_values=None,
    dtheta_deg_values=None,
):
    """GlobalPose 周围扰动候选 (Mode B)，组内 score 归一化。"""
    if score_config is None:
        score_config = DsmBevScoreConfig()
    if dx_values is None:
        dx_values = PERTURB_DX_M
    if dy_values is None:
        dy_values = PERTURB_DY_M
    if dtheta_deg_values is None:
        dtheta_deg_values = PERTURB_DTHETA_DEG

    candidates = []
    for dx in dx_values:
        for dy in dy_values:
            for dtheta_deg in dtheta_deg_values:
                px, py, ptheta = perturb_pose_gauss_theta(
                    gauss_x, gauss_y, theta, dx, dy, dtheta_deg
                )
                score, dsm_result = score_at_pose(
                    dsm_cropper,
                    px,
                    py,
                    ptheta,
                    lidar_layers,
                    score_config,
                    h_max=h_max,
                )
                candidates.append(
                    {
                        "dx_m": float(dx),
                        "dy_m": float(dy),
                        "dtheta_deg": float(dtheta_deg),
                        "gauss_x": px,
                        "gauss_y": py,
                        "theta": ptheta,
                        "score_raw": score,
                        "dsm_result": dsm_result,
                    }
                )

    normalized = normalize_candidate_scores(
        [item["score_raw"] for item in candidates],
        score_config,
    )
    for item, norm in zip(candidates, normalized):
        item["score"] = norm

    sorted_by_j = sorted(candidates, key=lambda c: float(c["score_raw"]["J_total"]))
    j_values = [float(c["score_raw"]["J_total"]) for c in sorted_by_j]
    j_best = j_values[0] if j_values else 0.0
    j_second = j_values[1] if len(j_values) > 1 else j_best

    center_idx = None
    for idx, item in enumerate(candidates):
        if (
            abs(item["dx_m"]) < 1e-6
            and abs(item["dy_m"]) < 1e-6
            and abs(item["dtheta_deg"]) < 1e-6
        ):
            center_idx = idx
            break

    center_rank = 1
    if center_idx is not None:
        center_j = float(candidates[center_idx]["score_raw"]["J_total"])
        center_rank = 1 + sum(1 for j in j_values if j < center_j - 1e-9)

    return {
        "candidates": candidates,
        "sorted_by_j": sorted_by_j,
        "J_best": j_best,
        "J_second_best": j_second,
        "J_gap": float(j_second - j_best),
        "center_rank": int(center_rank),
        "candidate_count": len(candidates),
    }


def format_dual_score_logs(result, frame_index, verbose=False):
    """返回一行日志：Global / Local score；verbose 时附带 J 分项。"""
    g_score = float(result["global"]["score"]["score"])
    l_score = float(result["local"]["score"]["score"])
    line = "score #{idx} | Global={gs:.4f} Local={ls:.4f}".format(
        idx=frame_index,
        gs=g_score,
        ls=l_score,
    )
    if not verbose:
        return line
    gs = result["global"]["score"]
    ls = result["local"]["score"]
    line += (
        " | G: J_h={jh:.4f} J_gx={jx:.4f} J_gy={jy:.4f} J={jt:.4f} n={n} ng={ng}".format(
            jh=float(gs["J_h"]),
            jx=float(gs["J_gx"]),
            jy=float(gs["J_gy"]),
            jt=float(gs["J_total"]),
            n=int(gs["valid_pixel_num"]),
            ng=int(gs.get("valid_grad_pixel_num", gs["valid_pixel_num"])),
        )
    )
    line += (
        " | L: J_h={jh:.4f} J_gx={jx:.4f} J_gy={jy:.4f} J={jt:.4f}".format(
            jh=float(ls["J_h"]),
            jx=float(ls["J_gx"]),
            jy=float(ls["J_gy"]),
            jt=float(ls["J_total"]),
        )
    )
    return line


def format_perturbation_score_logs(perturb_result, frame_index):
    """Mode B 扰动候选摘要：J_gap、GlobalPose 排名、top-3。"""
    best = perturb_result["sorted_by_j"][:3]
    parts = [
        "perturb #{idx} N={n} J_gap={gap:.4f} center_rank={rank}".format(
            idx=frame_index,
            n=int(perturb_result["candidate_count"]),
            gap=float(perturb_result["J_gap"]),
            rank=int(perturb_result["center_rank"]),
        )
    ]
    for rank, item in enumerate(best, start=1):
        s = item["score_raw"]
        parts.append(
            "  #{r} dx={dx:+.0f} dy={dy:+.0f} dθ={dt:+.0f}° J={j:.4f} score={sc:.4f} ng={ng}".format(
                r=rank,
                dx=item["dx_m"],
                dy=item["dy_m"],
                dt=item["dtheta_deg"],
                j=float(s["J_total"]),
                sc=float(item["score"]["score"]),
                ng=int(s.get("valid_grad_pixel_num", 0)),
            )
        )
    return "\n".join(parts)
