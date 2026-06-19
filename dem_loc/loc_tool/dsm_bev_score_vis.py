#!/usr/bin/env python3
"""DSM-BEV 分数对比可视化。"""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .dsm_bev_score import apply_bev_mask_to_layers, cap_gradient_layer
from .dsm_patch import (
    draw_center_dot,
    render_colorized_layer,
    render_mask_layer,
    render_masked_colorized_layer,
    robust_layer_range,
)


_LAYER_SPECS = (
    ("H_rel", "H_rel_surf", "H_L"),
    ("G_long", "G_long_L", "Gy_L"),
    ("G_lat", "G_lat_L", "Gx_L"),
)


def add_title_bar(image, title, bar_height=28):
    if cv2 is None:
        raise ImportError("dsm_bev_score_vis requires OpenCV (cv2)")

    img = np.asarray(image, dtype=np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bar = np.zeros((bar_height, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        bar,
        title,
        (8, bar_height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return cv2.vconcat([bar, img])


def _hconcat_panels(panels):
    max_h = max(panel.shape[0] for panel in panels)
    aligned = []
    for panel in panels:
        if panel.shape[0] < max_h:
            pad = np.zeros((max_h - panel.shape[0], panel.shape[1], 3), dtype=np.uint8)
            panel = cv2.vconcat([panel, pad])
        aligned.append(panel)
    return cv2.hconcat(aligned)


def _collect_layer(data, dsm_key, lidar_key, from_lidar):
    if from_lidar:
        return np.asarray(data[lidar_key], dtype=np.float32)
    return np.asarray(data[dsm_key], dtype=np.float32)


def _layer_ranges(lidar_layers, global_layers, local_layers):
    """同一图层在 LiDAR / Global / Local 间共用色标。"""
    ranges = {}
    sources = (
        (lidar_layers, True),
        (global_layers, False),
        (local_layers, False),
    )
    for layer_label, dsm_key, lidar_key in _LAYER_SPECS:
        chunks = []
        for data, from_lidar in sources:
            arr = _collect_layer(data, dsm_key, lidar_key, from_lidar)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                chunks.append(finite)
        if chunks:
            merged = np.concatenate(chunks)
            ranges[layer_label] = robust_layer_range(merged)
        else:
            ranges[layer_label] = (0.0, 1.0)
    return ranges


def build_dual_patch_view(result):
    """3 行竖拼，每行 H_rel | G_long | G_lat | M_L 横排。"""
    if cv2 is None:
        raise ImportError("dsm_bev_score_vis requires OpenCV (cv2)")

    lidar_layers = result["lidar_layers"]
    global_layers = result["global"]["dsm_result"]["layers"]
    local_layers = result["local"]["dsm_result"]["layers"]
    global_score = float(result["global"]["score"]["score"])
    local_score = float(result["local"]["score"]["score"])
    layer_ranges = _layer_ranges(lidar_layers, global_layers, local_layers)
    m_obs = np.asarray(lidar_layers["M_obs"], dtype=np.float32)

    source_rows = (
        ("LiDAR BEV", lidar_layers, True, None),
        ("Global score={:.4f}".format(global_score), global_layers, False, global_score),
        ("Local score={:.4f}".format(local_score), local_layers, False, local_score),
    )

    row_images = []
    for row_title, data, from_lidar, _score in source_rows:
        panels = []
        for layer_label, dsm_key, lidar_key in _LAYER_SPECS:
            arr = _collect_layer(data, dsm_key, lidar_key, from_lidar)
            vmin, vmax = layer_ranges[layer_label]
            panels.append(
                add_title_bar(
                    draw_center_dot(render_colorized_layer(arr, vmin, vmax)),
                    "{} {}".format(row_title.split()[0], layer_label),
                )
            )
        panels.append(
            add_title_bar(
                draw_center_dot(render_mask_layer(m_obs)),
                "{} M_L".format(row_title.split()[0]),
            )
        )
        row_images.append(_hconcat_panels(panels))

    return cv2.vconcat(row_images)


def build_global_masked_patch_view(result, score_config=None):
    """GlobalPose 处 DSM patch，经 LiDAR M_obs 裁剪后单独显示。"""
    if cv2 is None:
        raise ImportError("dsm_bev_score_vis requires OpenCV (cv2)")

    from .dsm_bev_score import DsmBevScoreConfig

    if score_config is None:
        score_config = DsmBevScoreConfig()

    lidar_layers = result["lidar_layers"]
    global_layers = result["global"]["dsm_result"]["layers"]
    global_score = result["global"]["score"]
    g_score = float(global_score["score"])

    m_obs, masked = apply_bev_mask_to_layers(
        lidar_layers["H_L"],
        lidar_layers["Gx_L"],
        lidar_layers["Gy_L"],
        global_layers["H_rel_surf"],
        global_layers["G_lat_L"],
        global_layers["G_long_L"],
        lidar_layers["M_obs"],
    )

    dsm_layers = {
        "H_rel": masked["H_D"],
        "G_long": cap_gradient_layer(masked["Gy_D"], score_config.grad_cap),
        "G_lat": cap_gradient_layer(masked["Gx_D"], score_config.grad_cap),
    }
    ranges = {}
    for key, arr in dsm_layers.items():
        valid = m_obs & np.isfinite(arr)
        if valid.any():
            ranges[key] = robust_layer_range(arr[valid])
        else:
            ranges[key] = (0.0, 1.0)

    panels = []
    for layer_label, key in (("H_rel", "H_rel"), ("G_long", "G_long"), ("G_lat", "G_lat")):
        arr = dsm_layers[key]
        vmin, vmax = ranges[key]
        panels.append(
            add_title_bar(
                draw_center_dot(render_masked_colorized_layer(arr, m_obs, vmin, vmax)),
                "Global {}".format(layer_label),
            )
        )
    panels.append(
        add_title_bar(
            draw_center_dot(render_mask_layer(m_obs)),
            "M_L (LiDAR)",
        )
    )

    body = _hconcat_panels(panels)
    title = "GlobalPose masked DSM | score={:.4f} n={} ng={}".format(
        g_score,
        int(global_score.get("valid_pixel_num", m_obs.sum())),
        int(global_score.get("valid_grad_pixel_num", 0)),
    )
    stub = np.zeros((4, body.shape[1], 3), dtype=np.uint8)
    return cv2.vconcat([add_title_bar(stub, title, bar_height=32), body])
