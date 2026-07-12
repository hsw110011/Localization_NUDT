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


def _draw_legend(image, weight_min, weight_max):
    h, w = image.shape[:2]
    bar_h = min(140, max(60, h // 3))
    bar_w = 16
    x0 = max(8, w - 84)
    y0 = 36

    for i in range(bar_h):
        t = 1.0 - float(i) / max(float(bar_h - 1), 1.0)
        color = _weight_to_bgr(t)
        cv2.line(image, (x0, y0 + i), (x0 + bar_w, y0 + i), color, 1)

    cv2.rectangle(image, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 1)
    cv2.putText(
        image,
        "high {:.2e}".format(float(weight_max)),
        (x0 - 44, max(14, y0 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "low {:.2e}".format(float(weight_min)),
        (x0 - 42, y0 + bar_h + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


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
    rows, cols = image.shape[:2]
    particles_np = np.asarray(particles, dtype=np.float32)
    weights_np = np.asarray(weights, dtype=np.float32).reshape(-1)
    weights_np = np.nan_to_num(weights_np, nan=0.0, posinf=0.0, neginf=0.0)
    weight_norm = _normalize_weights(weights_np)
    row, col, valid = _particles_to_pixels(
        particles_np, center_x, center_y, heading_rad, rows, cols, resolution
    )

    overlay = image.copy()
    if particles_np.shape[0] and weights_np.size:
        order = np.argsort(weight_norm)
        for idx in order:
            if idx >= valid.size or not bool(valid[idx]):
                continue
            t = float(weight_norm[idx])
            radius = int(round(2.0 + 4.0 * t))
            cv2.circle(
                overlay,
                (int(round(float(col[idx]))), int(round(float(row[idx])))),
                radius,
                _weight_to_bgr(t),
                -1,
                cv2.LINE_AA,
            )
        image = cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0)

        best_idx = int(np.argmax(weights_np))
        if best_idx < valid.size and bool(valid[best_idx]):
            center = (int(round(float(col[best_idx]))), int(round(float(row[best_idx]))))
            cv2.circle(image, center, 8, (255, 255, 255), 2, cv2.LINE_AA)

    center_px = (int(round(0.5 * cols - 0.5)), int(round(0.5 * rows - 0.5)))
    cv2.drawMarker(
        image,
        center_px,
        (0, 255, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=16,
        thickness=2,
        line_type=cv2.LINE_AA,
    )

    weight_min = float(np.min(weights_np)) if weights_np.size else 0.0
    weight_max = float(np.max(weights_np)) if weights_np.size else 0.0
    title = "PF weights"
    if frame_id is not None:
        title += " frame={}".format(int(frame_id))
    if neff is not None:
        title += " neff={:.1f}".format(float(neff))
    if best_score is not None:
        title += " best_score={:.4f}".format(float(best_score))

    cv2.rectangle(image, (0, 0), (cols, 26), (0, 0, 0), -1)
    cv2.putText(
        image,
        title,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    _draw_legend(image, weight_min, weight_max)
    return image


def save_particle_weight_view(
    dsm_cropper,
    particles,
    weights,
    estimate,
    out_dir,
    timestamp=None,
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
    return out_path
