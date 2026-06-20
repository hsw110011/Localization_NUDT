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


def _layer_ranges_from_sources(sources):
    """sources: [(data_dict, from_lidar), ...]"""
    ranges = {}
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


def _build_data_row(row_title, data, from_lidar, layer_ranges, m_obs, m_label="M_L"):
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
            "{} {}".format(row_title.split()[0], m_label),
        )
    )
    return _hconcat_panels(panels)


def _build_masked_row(row_title, lidar_layers, dsm_layers, layer_ranges, m_obs, score_config):
    _, masked = apply_bev_mask_to_layers(
        lidar_layers["H_L"],
        lidar_layers["Gx_L"],
        lidar_layers["Gy_L"],
        dsm_layers["H_rel_surf"],
        dsm_layers["G_lat_L"],
        dsm_layers["G_long_L"],
        lidar_layers["M_obs"],
    )
    dsm_masked = {
        "H_rel_surf": masked["H_D"],
        "G_long_L": cap_gradient_layer(masked["Gy_D"], score_config.grad_cap),
        "G_lat_L": cap_gradient_layer(masked["Gx_D"], score_config.grad_cap),
    }
    panels = []
    for layer_label, dsm_key, lidar_key in _LAYER_SPECS:
        arr = np.asarray(dsm_masked[dsm_key], dtype=np.float32)
        vmin, vmax = layer_ranges[layer_label]
        panels.append(
            add_title_bar(
                draw_center_dot(render_masked_colorized_layer(arr, m_obs, vmin, vmax)),
                "{} {}".format(row_title.split()[0], layer_label),
            )
        )
    panels.append(
        add_title_bar(
            draw_center_dot(render_mask_layer(m_obs)),
            "{} M_L".format(row_title.split()[0]),
        )
    )
    return _hconcat_panels(panels)


def build_quad_patch_view(
    lidar_layers,
    global_layers,
    global_score,
    pf_best_layers,
    pf_best_score,
    pf_best_pose=None,
    score_config=None,
):
    """4 行 x 4 列: LiDAR | Global | PF best | PF best masked (M_obs)。"""
    if cv2 is None:
        raise ImportError("dsm_bev_score_vis requires OpenCV (cv2)")

    from .dsm_bev_score import DsmBevScoreConfig

    if score_config is None:
        score_config = DsmBevScoreConfig()

    m_obs = np.asarray(lidar_layers["M_obs"], dtype=np.float32)
    layer_ranges = _layer_ranges_from_sources(
        [
            (lidar_layers, True),
            (global_layers, False),
            (pf_best_layers, False),
        ]
    )

    g_score = float(global_score)
    pf_score = float(pf_best_score)
    pf_tag = "PF best score={:.4f}".format(pf_score)
    if pf_best_pose is not None:
        pf_tag += " ({:.1f},{:.1f})".format(
            float(pf_best_pose.get("gauss_x", 0.0)),
            float(pf_best_pose.get("gauss_y", 0.0)),
        )

    rows = [
        _build_data_row("LiDAR BEV", lidar_layers, True, layer_ranges, m_obs),
        _build_data_row(
            "Global score={:.4f}".format(g_score), global_layers, False, layer_ranges, m_obs
        ),
        _build_data_row(pf_tag, pf_best_layers, False, layer_ranges, m_obs),
        _build_masked_row(
            "PF masked score={:.4f}".format(pf_score),
            lidar_layers,
            pf_best_layers,
            layer_ranges,
            m_obs,
            score_config,
        ),
    ]
    return cv2.vconcat(rows)


def build_dual_patch_view(result):
    """3 行竖拼，每行 H_rel | G_long | G_lat | M_L 横排（兼容旧接口）。"""
    if cv2 is None:
        raise ImportError("dsm_bev_score_vis requires OpenCV (cv2)")

    lidar_layers = result["lidar_layers"]
    global_layers = result["global"]["dsm_result"]["layers"]
    local_layers = result["local"]["dsm_result"]["layers"]
    global_score = float(result["global"]["score"]["score"])
    local_score = float(result["local"]["score"]["score"])
    layer_ranges = _layer_ranges_from_sources(
        [
            (lidar_layers, True),
            (global_layers, False),
            (local_layers, False),
        ]
    )
    m_obs = np.asarray(lidar_layers["M_obs"], dtype=np.float32)

    rows = [
        _build_data_row("LiDAR BEV", lidar_layers, True, layer_ranges, m_obs),
        _build_data_row(
            "Global score={:.4f}".format(global_score), global_layers, False, layer_ranges, m_obs
        ),
        _build_data_row(
            "Local score={:.4f}".format(local_score), local_layers, False, layer_ranges, m_obs
        ),
    ]
    return cv2.vconcat(rows)


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

    m_obs = np.asarray(lidar_layers["M_obs"], dtype=np.float32)
    body = _build_masked_row(
        "Global masked",
        lidar_layers,
        global_layers,
        _layer_ranges_from_sources([(lidar_layers, True), (global_layers, False)]),
        m_obs,
        score_config,
    )
    title = "GlobalPose masked DSM | score={:.4f} n={} ng={}".format(
        g_score,
        int(global_score.get("valid_pixel_num", m_obs.sum())),
        int(global_score.get("valid_grad_pixel_num", 0)),
    )
    stub = np.zeros((4, body.shape[1], 3), dtype=np.uint8)
    return cv2.vconcat([add_title_bar(stub, title, bar_height=32), body])
