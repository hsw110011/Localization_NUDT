#!/usr/bin/env python3
"""Particle weight visualization helpers for the DSM particle filter."""

import math
import os

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .dsm_patch import render_colorized_layer, robust_layer_range


def _require_cv2():
    if cv2 is None:
        raise ImportError("particle weight visualization requires OpenCV (cv2)")


def _weight_to_bgr(weight_norm):
    """Map 0..1 to blue..red in OpenCV BGR order."""
    t = float(np.clip(weight_norm, 0.0, 1.0))
    return int(round(255.0 * (1.0 - t))), 0, int(round(255.0 * t))


def _normalize_weights(weights):
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    if w.size == 0:
        return w
    w_min = float(np.min(w))
    w_max = float(np.max(w))
    span = w_max - w_min
    if span <= 1e-12:
        return np.zeros_like(w, dtype=np.float32)
    return ((w - w_min) / span).astype(np.float32)


def _safe_float(value, default=None):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return v


def _fmt_fixed(value, digits=6):
    v = _safe_float(value, None)
    if v is None:
        return "n/a"
    return ("{:.%df}" % int(digits)).format(v)


def _put_text_bounded(
    image,
    text,
    origin,
    max_width,
    font_scale=0.42,
    color=(255, 255, 255),
    thickness=1,
):
    if max_width <= 4:
        return
    clipped = str(text)
    while clipped:
        shown = clipped if len(clipped) == len(str(text)) else clipped + "..."
        text_size, _ = cv2.getTextSize(shown, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if text_size[0] <= max_width:
            cv2.putText(
                image,
                shown,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            return
        clipped = clipped[:-1]


def _layout_scale(image):
    h, w = image.shape[:2]
    return float(np.clip(min(h, w) / 512.0, 0.62, 1.0))


def _particles_to_pixels(particles, center_x, center_y, heading_rad, rows, cols, resolution):
    pts = np.asarray(particles, dtype=np.float32)
    if pts.size == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty, np.zeros((0,), dtype=bool)

    dx = pts[:, 0] - float(center_x)
    dy = pts[:, 1] - float(center_y)
    cos_h = math.cos(float(heading_rad))
    sin_h = math.sin(float(heading_rad))
    x_vehicle = dx * cos_h + dy * sin_h
    y_vehicle = -dx * sin_h + dy * cos_h

    res = max(float(resolution), 1e-6)
    row = 0.5 * float(rows) - x_vehicle / res - 0.5
    col = 0.5 * float(cols) - y_vehicle / res - 0.5
    valid = (row >= 0.0) & (row < float(rows)) & (col >= 0.0) & (col < float(cols))
    return row.astype(np.float32), col.astype(np.float32), valid


def _background_from_patch(patch):
    arr = np.asarray(patch, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size:
        vmin, vmax = robust_layer_range(finite)
    else:
        vmin, vmax = 0.0, 1.0
    return render_colorized_layer(arr, vmin, vmax)


def _resize_for_display(image, resolution, min_size=400):
    h, w = image.shape[:2]
    short_side = min(h, w)
    if short_side >= int(min_size):
        return image, float(resolution)

    scale = float(min_size) / max(float(short_side), 1.0)
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return resized, float(resolution) / scale


def _draw_legend(image, weight_min, weight_max, top_margin=104):
    h, w = image.shape[:2]
    s = _layout_scale(image)
    bar_h = min(int(round(140 * s)), max(int(round(72 * s)), h // 4))
    bar_w = max(9, int(round(14 * s)))
    margin_r = max(20, int(round(46 * s)))
    x0 = max(8, w - margin_r)
    y0 = min(h - bar_h - int(round(34 * s)), int(round(110 * s)))
    y0 = max(int(round(104 * s)), int(top_margin), y0)
    if y0 + bar_h + int(round(24 * s)) >= h:
        y0 = max(int(top_margin) + 4, h - bar_h - int(round(24 * s)))

    bg_x = max(0, x0 - int(round(58 * s)))
    bg_y = max(0, y0 - int(round(24 * s)))
    bg_w = min(w - bg_x, int(round(88 * s)))
    bg_h = min(h - bg_y, bar_h + int(round(50 * s)))
    if bg_w > 0 and bg_h > 0:
        roi = image[bg_y : bg_y + bg_h, bg_x : bg_x + bg_w]
        panel = np.zeros_like(roi)
        cv2.addWeighted(panel, 0.38, roi, 0.62, 0.0, roi)

    for i in range(bar_h):
        t = 1.0 - float(i) / max(float(bar_h - 1), 1.0)
        color = _weight_to_bgr(t)
        cv2.line(image, (x0, y0 + i), (x0 + bar_w, y0 + i), color, 1)

    cv2.rectangle(image, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 1)
    high_x = max(2, x0 - int(round(54 * s)))
    low_x = max(2, x0 - int(round(50 * s)))
    _put_text_bounded(
        image,
        "high {}".format(_fmt_fixed(weight_max, 6)),
        (high_x, max(12, y0 - max(5, int(round(8 * s))))),
        max(4, w - high_x - 8),
        font_scale=0.34 * s,
    )
    _put_text_bounded(
        image,
        "low {}".format(_fmt_fixed(weight_min, 6)),
        (low_x, min(h - 4, y0 + bar_h + int(round(18 * s)))),
        max(4, w - low_x - 8),
        font_scale=0.34 * s,
    )


def _draw_info_panel(
    image,
    frame_id,
    neff,
    total_particles,
    weight_min,
    weight_mean,
    weight_max,
    weight_std,
):
    h, w = image.shape[:2]
    s = _layout_scale(image)
    panel_h = min(int(round(96 * s)), max(int(round(74 * s)), h // 5))
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0.0, image)

    title = "PF weights"
    if frame_id is not None:
        title += " | frame {}".format(int(frame_id))
    x = max(6, int(round(10 * s)))
    _put_text_bounded(image, title, (x, int(round(20 * s))), w - 2 * x, font_scale=0.48 * s)

    neff_text = "Neff {} / N {}".format(_fmt_fixed(neff, 1), int(total_particles))
    _put_text_bounded(image, neff_text, (x, int(round(43 * s))), w - 2 * x, font_scale=0.44 * s)

    score_text = "Score(weight) min {}  mean {}  max {}".format(
        _fmt_fixed(weight_min, 6),
        _fmt_fixed(weight_mean, 6),
        _fmt_fixed(weight_max, 6),
    )
    _put_text_bounded(image, score_text, (x, int(round(66 * s))), w - 2 * x, font_scale=0.39 * s)

    range_text = "Score(weight) std {}  range {}".format(
        _fmt_fixed(weight_std, 6),
        _fmt_fixed(weight_max - weight_min, 6),
    )
    _put_text_bounded(image, range_text, (x, int(round(88 * s))), w - 2 * x, font_scale=0.39 * s)
    return panel_h


def build_particle_weight_view(
    background_patch,
    particles,
    weights,
    center_x,
    center_y,
    heading_rad,
    resolution,
    frame_id=None,
    neff=None,
    best_score=None,
):
    """Build a BGR image with particles colored blue(low weight) to red(high)."""
    _require_cv2()

    image = _background_from_patch(background_patch)
    image, display_resolution = _resize_for_display(image, resolution, min_size=400)
    rows, cols = image.shape[:2]
    layout_s = _layout_scale(image)
    particles_np = np.asarray(particles, dtype=np.float32)
    weights_np = np.asarray(weights, dtype=np.float32).reshape(-1)
    weights_np = np.nan_to_num(weights_np, nan=0.0, posinf=0.0, neginf=0.0)

    if weights_np.size:
        weight_min = float(np.min(weights_np))
        weight_max = float(np.max(weights_np))
        weight_mean = float(np.mean(weights_np))
        weight_std = float(np.std(weights_np))
    else:
        weight_min = weight_max = weight_mean = weight_std = 0.0
    span = weight_max - weight_min
    if abs(span) < 1e-12:
        span = 1.0

    weight_norm = _normalize_weights(weights_np)
    row, col, valid = _particles_to_pixels(
        particles_np, center_x, center_y, heading_rad, rows, cols, display_resolution
    )

    overlay = image.copy()
    best_idx = -1
    best_weight = -1.0
    if particles_np.shape[0] and weights_np.size:
        for idx in range(int(min(particles_np.shape[0], weights_np.size))):
            if idx >= valid.size or not bool(valid[idx]):
                continue
            t = float(weight_norm[idx])
            radius = max(1, int(round((2.0 + 4.0 * t) * layout_s)))
            cv2.circle(
                overlay,
                (int(round(float(col[idx]))), int(round(float(row[idx])))),
                radius,
                _weight_to_bgr(t),
                -1,
                cv2.LINE_AA,
            )
            if float(weights_np[idx]) > best_weight:
                best_weight = float(weights_np[idx])
                best_idx = idx
        image = cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0)

        if best_idx >= 0 and best_idx < valid.size and bool(valid[best_idx]):
            center = (int(round(float(col[best_idx]))), int(round(float(row[best_idx]))))
            cv2.circle(
                image,
                center,
                max(5, int(round(8 * layout_s))),
                (255, 255, 255),
                max(1, int(round(2 * layout_s))),
                cv2.LINE_AA,
            )

    center_px = (int(round(0.5 * cols - 0.5)), int(round(0.5 * rows - 0.5)))
    cv2.drawMarker(
        image,
        center_px,
        (0, 255, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=max(10, int(round(16 * layout_s))),
        thickness=max(1, int(round(2 * layout_s))),
        line_type=cv2.LINE_AA,
    )

    panel_h = _draw_info_panel(
        image,
        frame_id,
        neff,
        int(weights_np.size),
        weight_min,
        weight_mean,
        weight_max,
        weight_std,
    )
    _draw_legend(image, weight_min, weight_max, top_margin=panel_h)
    return image


def save_particle_weight_view(
    dsm_cropper,
    particles,
    weights,
    estimate,
    out_dir,
    timestamp=None,
    return_image=False,
):
    """Crop the DSM once around estimate, draw all particles, and save a PNG."""
    _require_cv2()
    os.makedirs(out_dir, exist_ok=True)

    center_x = float(estimate["x_est"])
    center_y = float(estimate["y_est"])
    heading = float(estimate["yaw_est"])
    patch, _ = dsm_cropper.crop_numpy(center_x, center_y, heading)
    frame_id = estimate.get("frame_id", None)
    image = build_particle_weight_view(
        patch,
        particles,
        weights,
        center_x,
        center_y,
        heading,
        dsm_cropper.resolution,
        frame_id=frame_id,
        neff=estimate.get("neff"),
        best_score=estimate.get("best_score", estimate.get("max_score")),
    )

    if frame_id is None:
        safe_stamp = "nostamp" if timestamp is None else str(timestamp).replace(".", "_")
        filename = "pf_weight_{}.png".format(safe_stamp)
    else:
        filename = "pf_weight_{:06d}.png".format(int(frame_id))
    out_path = os.path.join(out_dir, filename)
    if not cv2.imwrite(out_path, image):
        raise IOError("failed to write particle weight visualization: {}".format(out_path))
    if return_image:
        return out_path, image
    return out_path
