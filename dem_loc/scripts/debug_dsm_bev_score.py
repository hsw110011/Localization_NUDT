#!/usr/bin/env python3
"""One-shot DSM-BEV score diagnostics (subscribe latest messages, print breakdown)."""

from __future__ import print_function

import math
import os
import sys

import numpy as np
import rospy
from grid_map_msgs.msg import GridMap as GridMapMsg
from self_state.msg import GlobalPose as GlobalPoseMsg
from self_state.msg import LocalPose as LocalPoseMsg

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(SCRIPT_DIR)
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from loc_tool.cinterface import parse_lidar_bev_grid_map
from loc_tool.coord_converter import CoordConverter
from loc_tool.dem_tool import load_dem_tiff
from loc_tool.dsm_bev_score import (
    DsmBevScoreConfig,
    apply_bev_mask_to_layers,
    build_obs_mask,
    compute_dsm_bev_score,
    compute_h_max,
    dsm_layers_to_score_inputs,
)
from loc_tool.dsm_bev_score_runner import (
    format_perturbation_score_logs,
    global_pose_gauss_theta,
    local_pose_gauss_theta,
    score_at_pose,
    score_perturbation_grid,
)
from loc_tool.dsm_patch import DsmPatchCropper


def _layer_stats(name, arr, mask=None):
    arr = np.asarray(arr, dtype=np.float64)
    if mask is None:
        finite = np.isfinite(arr)
    else:
        finite = np.asarray(mask, dtype=bool) & np.isfinite(arr)
    if not finite.any():
        print("  {}: no finite values".format(name))
        return
    vals = arr[finite]
    print(
        "  {}: n={} min={:.4f} max={:.4f} mean={:.4f} p50={:.4f} p98={:.4f} nan={}".format(
            name,
            int(finite.sum()),
            float(vals.min()),
            float(vals.max()),
            float(vals.mean()),
            float(np.percentile(vals, 50)),
            float(np.percentile(vals, 98)),
            int((~np.isfinite(arr)).sum()),
        )
    )


def _print_score_detail(label, score_dict):
    print("\n=== {} ===".format(label))
    for key in (
        "valid_pixel_num",
        "valid_grad_pixel_num",
        "H_max",
        "J_h",
        "J_gx",
        "J_gy",
        "J_total",
        "score",
    ):
        print("  {}: {}".format(key, score_dict.get(key)))


def _print_masked_diff(lidar_layers, dsm_layers, m_obs):
    _, masked = apply_bev_mask_to_layers(
        lidar_layers["H_L"],
        lidar_layers["Gx_L"],
        lidar_layers["Gy_L"],
        dsm_layers["H_rel_surf"],
        dsm_layers["G_lat_L"],
        dsm_layers["G_long_L"],
        lidar_layers["M_obs"],
    )
    m = np.asarray(m_obs, dtype=bool)
    print("\n--- masked layer stats (obs mask) ---")
    for key in ("H_L", "H_D", "Gx_L", "Gx_D", "Gy_L", "Gy_D"):
        _layer_stats(key, masked[key], mask=m)

    h_max = compute_h_max(masked["H_L"], m)
    h_l_n = np.clip(masked["H_L"], 0.0, h_max) / max(h_max, 1e-6)
    h_d_n = np.clip(masked["H_D"], 0.0, h_max) / max(h_max, 1e-6)
    e_h = h_l_n - h_d_n
    e_gx = masked["Gx_L"] - masked["Gx_D"]
    e_gy = masked["Gy_L"] - masked["Gy_D"]
    valid = m & np.isfinite(e_h)
    if valid.any():
        print(
            "  e_h: mean={:.4f} rmse={:.4f} |e|>0.5={}".format(
                float(e_h[valid].mean()),
                float(np.sqrt(np.mean(e_h[valid] ** 2))),
                int((np.abs(e_h[valid]) > 0.5).sum()),
            )
        )
    valid = m & np.isfinite(e_gx)
    if valid.any():
        print(
            "  e_gx: mean={:.4f} rmse={:.4f} |e|>0.5={}".format(
                float(e_gx[valid].mean()),
                float(np.sqrt(np.mean(e_gx[valid] ** 2))),
                int((np.abs(e_gx[valid]) > 0.5).sum()),
            )
        )
    valid = m & np.isfinite(e_gy)
    if valid.any():
        print(
            "  e_gy: mean={:.4f} rmse={:.4f} |e|>0.5={}".format(
                float(e_gy[valid].mean()),
                float(np.sqrt(np.mean(e_gy[valid] ** 2))),
                int((np.abs(e_gy[valid]) > 0.5).sum()),
            )
        )


def main():
    rospy.init_node("debug_dsm_bev_score", anonymous=True)

    dem_path = rospy.get_param(
        "~dem_path", "/home/hsw/catkin_ws/doc/miluo_dsm.tif"
    )
    timeout = float(rospy.get_param("~timeout", 15.0))

    print("Waiting for topics (timeout={}s)...".format(timeout))
    bev_msg = rospy.wait_for_message("/lidar_bev/grid_map", GridMapMsg, timeout=timeout)
    global_pose = rospy.wait_for_message(
        "/self_state/GlobalPose", GlobalPoseMsg, timeout=timeout
    )
    local_pose = rospy.wait_for_message(
        "/self_state/LocalPose", LocalPoseMsg, timeout=timeout
    )

    print("\nBEV info: {}x{} res={:.3f} layers={}".format(
        bev_msg.info.length_x,
        bev_msg.info.length_y,
        bev_msg.info.resolution,
        list(bev_msg.layers),
    ))

    dem_data = load_dem_tiff(dem_path, map_resolution_m=float(bev_msg.info.resolution))
    cc = CoordConverter(dem_data)
    cropper = DsmPatchCropper(
        dem_data,
        cc,
        map_size_x=float(bev_msg.info.length_x),
        map_size_y=float(bev_msg.info.length_y),
        resolution=float(bev_msg.info.resolution),
    )

    lidar_layers = parse_lidar_bev_grid_map(bev_msg)
    m_obs = build_obs_mask(lidar_layers["M_obs"])
    print("\n--- raw LiDAR BEV stats ---")
    print("  M_obs valid pixels: {}".format(int(m_obs.sum())))
    for key in ("H_L", "Gx_L", "Gy_L", "M_obs"):
        _layer_stats(key, lidar_layers[key], mask=m_obs if key != "M_obs" else None)

    h_max = compute_h_max(lidar_layers["H_L"], m_obs)
    print("  H_max from LiDAR: {:.4f}".format(h_max))

    config = DsmBevScoreConfig()
    valid_h = m_obs & np.isfinite(lidar_layers["H_L"]) & (
        lidar_layers["H_L"] > config.ground_threshold
    )
    print(
        "  H_L > {:.2f} count: {} (need {} for percentile H_max)".format(
            config.ground_threshold,
            int(valid_h.sum()),
            config.min_valid_height_points,
        )
    )

    local_pose_base = cc.GetBaseFromLocalPose(
        local_pose,
        global_pose,
        local_heading_unit="deg",
        local_heading_convention="math",
        global_heading_unit="deg",
        global_heading_convention="math",
    )

    g_x, g_y, g_theta = global_pose_gauss_theta(
        cc, global_pose, heading_unit="deg", heading_convention="math"
    )
    l_x, l_y, l_theta = local_pose_gauss_theta(
        cc,
        local_pose,
        local_pose_base,
        local_heading_unit="deg",
        local_heading_convention="math",
    )
    print("\n--- poses ---")
    print("  Global: gauss=({:.2f}, {:.2f}) theta={:.3f} rad".format(g_x, g_y, g_theta))
    print("  Local:  gauss=({:.2f}, {:.2f}) theta={:.3f} rad".format(l_x, l_y, l_theta))
    print(
        "  delta: d={:.2f}m dyaw={:.1f}deg".format(
            math.hypot(l_x - g_x, l_y - g_y),
            math.degrees(l_theta - g_theta),
        )
    )

    global_score, global_dsm = score_at_pose(
        cropper, g_x, g_y, g_theta, lidar_layers, config, h_max=h_max
    )
    local_score, local_dsm = score_at_pose(
        cropper, l_x, l_y, l_theta, lidar_layers, config, h_max=h_max
    )

    _print_score_detail("GlobalPose score", global_score)
    _print_masked_diff(lidar_layers, global_dsm["layers"], m_obs)

    _print_score_detail("LocalPose score", local_score)
    _print_masked_diff(lidar_layers, local_dsm["layers"], m_obs)

    # What-if: clip extreme LiDAR gradients (boundary spike mitigation).
    lidar_clipped = dict(lidar_layers)
    cap = float(config.grad_cap)
    for key in ("Gx_L", "Gy_L"):
        arr = np.asarray(lidar_clipped[key], dtype=np.float32).copy()
        np.clip(arr, -cap, cap, out=arr)
        lidar_clipped[key] = arr
    clip_score, _ = score_at_pose(
        cropper, g_x, g_y, g_theta, lidar_clipped, config, h_max=h_max
    )
    print("\n=== What-if: LiDAR Gx/Gy clipped to [-{:.1f}, {:.1f}] ===".format(cap, cap))
    _print_score_detail("GlobalPose (clipped G)", clip_score)

    perturb = score_perturbation_grid(
        cropper, g_x, g_y, g_theta, lidar_layers, score_config=config, h_max=h_max
    )
    print("\n=== Perturbation grid (Mode B) ===")
    print(format_perturbation_score_logs(perturb, 1))

    print("\nDone.")


if __name__ == "__main__":
    main()
