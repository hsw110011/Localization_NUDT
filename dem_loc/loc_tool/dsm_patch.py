#!/usr/bin/env python3
"""以 Odom 位置为中心、车头朝上，用 PyTorch grid_sample 裁剪局部 DSM patch。

BEV 对齐:
    H_rel_surf = 高程 - 中心点高程，再经 [-h_half, +h_half] 死区
    G_long_L / G_lat_L = 3x3 中值 + Sobel / (8*res)，再经 |g|<=g_half 死区
    显示: H_rel | G_long | G_lat 三列 (各 320×320)，viridis，无 M_L 掩码
    后处理在 GPU 上用 unfold/conv2d，避免 Python 逐像素循环。
"""

import math

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:
    raise ImportError(
        "dsm_patch requires PyTorch. Install torch or activate the environment that provides it."
    ) from exc

try:
    import cv2
except ImportError:
    cv2 = None

_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
).view(1, 1, 3, 3)


def bev_grid_shape(map_size_x, map_size_y, resolution):
    resolution = max(float(resolution), 1e-3)
    rows = max(1, int(round(float(map_size_x) / resolution)))
    cols = max(1, int(round(float(map_size_y) / resolution)))
    return rows, cols


def _build_viridis_lut_bgr():
    if cv2 is None:
        ramp = np.linspace(0, 255, 256, dtype=np.uint8)
        return np.stack([ramp, ramp, ramp], axis=1)
    gray = np.arange(256, dtype=np.uint8).reshape(1, 256)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
    return color.reshape(256, 3)


VIRIDIS_LUT_BGR = _build_viridis_lut_bgr()


def robust_layer_range(values, fallback_min=0.0, fallback_max=1.0):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float(fallback_min), float(fallback_max)
    low = float(np.percentile(finite, 2.0))
    high = float(np.percentile(finite, 98.0))
    if not np.isfinite(low) or not np.isfinite(high) or (high - low) < 1e-3:
        return float(fallback_min), float(fallback_max)
    return low, high


def _median_filter_3x3_torch(layer_2d):
    """3x3 median on [H,W] tensor."""
    x = layer_2d.unsqueeze(0).unsqueeze(0)
    padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    patches = padded.unfold(2, 3, 1).unfold(3, 3, 1)
    h, w = layer_2d.shape
    return patches.reshape(1, 1, h, w, 9).median(dim=-1).values.squeeze(0).squeeze(0)


def _apply_deadzone_torch(values, half_width):
    if half_width <= 0.0:
        return values
    out = values.clone()
    out[out.abs() <= float(half_width)] = 0.0
    return out


def _compute_bev_layers_torch(
    patch_2d,
    resolution,
    sobel_x,
    sobel_y,
    h_rel_deadzone_half=0.20,
    grad_deadzone_half=0.15,
):
    """patch_2d: [H,W] float tensor on any device."""
    rows, cols = patch_2d.shape
    center = patch_2d[rows // 2, cols // 2]
    if not torch.isfinite(center):
        center = torch.nanmedian(patch_2d)
    h_rel = _apply_deadzone_torch(patch_2d - center, h_rel_deadzone_half)

    smoothed = _median_filter_3x3_torch(h_rel)
    s = smoothed.unsqueeze(0).unsqueeze(0)
    grad_index0 = F.conv2d(s, sobel_y, padding=1).squeeze()
    grad_index1 = F.conv2d(s, sobel_x, padding=1).squeeze()
    scale = 8.0 * max(float(resolution), 1e-3)
    g_long = _apply_deadzone_torch(-grad_index0 / scale, grad_deadzone_half)
    g_lat = _apply_deadzone_torch(-grad_index1 / scale, grad_deadzone_half)
    grad_cap = 3.0
    g_long = torch.clamp(g_long, -grad_cap, grad_cap)
    g_lat = torch.clamp(g_lat, -grad_cap, grad_cap)
    return h_rel, g_long, g_lat, center


def compute_bev_layers(
    elevation_patch,
    resolution=0.2,
    device=None,
    h_rel_deadzone_half=0.20,
    grad_deadzone_half=0.15,
):
    """Numpy wrapper (fast torch backend)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    patch = torch.as_tensor(elevation_patch, dtype=torch.float32, device=device)
    sobel_x = _SOBEL_X.to(device)
    sobel_y = _SOBEL_Y.to(device)
    h_rel, g_long, g_lat, center = _compute_bev_layers_torch(
        patch,
        resolution,
        sobel_x,
        sobel_y,
        h_rel_deadzone_half=h_rel_deadzone_half,
        grad_deadzone_half=grad_deadzone_half,
    )
    return {
        "H_rel_surf": h_rel.detach().cpu().numpy().astype(np.float32),
        "G_long_L": g_long.detach().cpu().numpy().astype(np.float32),
        "G_lat_L": g_lat.detach().cpu().numpy().astype(np.float32),
        "center_height": float(center.detach().cpu().item()),
    }


def draw_center_dot(image, radius=4):
    """在图层中心画红点（patch/BEV 几何中心）。"""
    if cv2 is None:
        return np.asarray(image, dtype=np.uint8)

    img = np.asarray(image, dtype=np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = img.copy()
    h, w = img.shape[:2]
    cv2.circle(img, (w // 2, h // 2), int(radius), (0, 0, 255), -1, cv2.LINE_AA)
    return img


def render_mask_layer(mask):
    """M_L / M_obs 二值图：白=有效，黑=无效。"""
    m = np.asarray(mask, dtype=np.float32)
    img = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
    valid = np.isfinite(m) & (m > 0.5)
    img[valid] = (255, 255, 255)
    return img


def render_masked_colorized_layer(layer, mask, min_value, max_value):
    """仅在 mask 有效区域上色，其余为黑。"""
    layer = np.asarray(layer, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    colored = render_colorized_layer(layer, min_value, max_value)
    out = np.zeros_like(colored)
    valid = mask & np.isfinite(layer)
    out[valid] = colored[valid]
    return out


def render_colorized_layer(layer, min_value, max_value):
    layer = np.asarray(layer, dtype=np.float32)
    finite = np.isfinite(layer)
    norm = np.zeros(layer.shape, dtype=np.float32)
    inv_range = 1.0 / max(float(max_value) - float(min_value), 1e-6)
    norm[finite] = np.clip((layer[finite] - float(min_value)) * inv_range, 0.0, 1.0)
    lut_idx = np.clip(np.round(norm * 255.0).astype(np.int32), 0, 255)
    return VIRIDIS_LUT_BGR[lut_idx]


def render_bev_layers_stack(layers, fallback_ranges=None):
    """H_rel | G_long | G_lat，每列 320×320，无掩码。"""
    if fallback_ranges is None:
        fallback_ranges = {
            "H_rel_surf": (0.0, 1.0),
            "G_long_L": (0.0, 1.0),
            "G_lat_L": (0.0, 1.0),
        }

    images = []
    for name in ("H_rel_surf", "G_long_L", "G_lat_L"):
        layer = layers[name]
        finite = np.isfinite(layer)
        vmin, vmax = robust_layer_range(
            layer[finite],
            fallback_min=fallback_ranges[name][0],
            fallback_max=fallback_ranges[name][1],
        )
        images.append(draw_center_dot(render_colorized_layer(layer, vmin, vmax)))

    if cv2 is None:
        return np.concatenate(images, axis=1)
    combined = images[0]
    for image in images[1:]:
        combined = cv2.hconcat([combined, image])
    return combined


class DsmPatchCropper(object):
    """Rotate-sampled local DSM patch aligned with LiDAR BEV frame."""

    def __init__(
        self,
        dem_data,
        coord_converter,
        map_size_x=64.0,
        map_size_y=64.0,
        resolution=0.2,
        device=None,
        fill_value=0.0,
        edge_min_valid_neighbors=5,
        h_rel_deadzone_half=0.20,
        grad_deadzone_half=0.15,
    ):
        self.dem_data = dem_data
        self.cc = coord_converter
        self.map_size_x = float(map_size_x)
        self.map_size_y = float(map_size_y)
        self.resolution = float(resolution)
        self.fill_value = float(fill_value)
        self.edge_min_valid_neighbors = int(edge_min_valid_neighbors)
        self.h_rel_deadzone_half = float(h_rel_deadzone_half)
        self.grad_deadzone_half = float(grad_deadzone_half)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._sobel_x = _SOBEL_X.to(self.device)
        self._sobel_y = _SOBEL_Y.to(self.device)

        elevation = np.asarray(dem_data.raw_elevation_map, dtype=np.float32)
        finite = np.isfinite(elevation)
        if not finite.any():
            elevation = np.zeros_like(elevation, dtype=np.float32)
        else:
            fill = self.fill_value
            if not np.isfinite(fill):
                fill = float(np.nanmin(elevation))
            elevation = np.where(finite, elevation, fill).astype(np.float32)

        self.dem_h, self.dem_w = elevation.shape[:2]
        self.dem_tensor = (
            torch.from_numpy(elevation).to(self.device).unsqueeze(0).unsqueeze(0)
        )

        self.rows = 0
        self.cols = 0
        self._x_vehicle = None
        self._y_vehicle = None
        self.set_geometry(map_size_x, map_size_y, resolution)

    def set_geometry(self, map_size_x, map_size_y, resolution):
        map_size_x = float(map_size_x)
        map_size_y = float(map_size_y)
        resolution = max(float(resolution), 1e-3)
        rows, cols = bev_grid_shape(map_size_x, map_size_y, resolution)

        if (
            abs(self.map_size_x - map_size_x) < 1e-6
            and abs(self.map_size_y - map_size_y) < 1e-6
            and abs(self.resolution - resolution) < 1e-3
            and self.rows == rows
            and self.cols == cols
            and self._x_vehicle is not None
        ):
            return False

        self.map_size_x = map_size_x
        self.map_size_y = map_size_y
        self.resolution = resolution
        self.rows = rows
        self.cols = cols

        row_idx = torch.arange(rows, dtype=torch.float32, device=self.device)
        col_idx = torch.arange(cols, dtype=torch.float32, device=self.device)
        row_grid, col_grid = torch.meshgrid(row_idx, col_idx, indexing="ij")

        self._x_vehicle = (0.5 * rows - (row_grid + 0.5)) * resolution
        self._y_vehicle = (0.5 * cols - (col_grid + 0.5)) * resolution
        return True

    @staticmethod
    def geometry_from_grid_map_msg(msg):
        info = msg.info
        return float(info.length_x), float(info.length_y), float(info.resolution)

    def _build_sample_grid(self, center_gauss_x, center_gauss_y, heading_rad):
        cos_h = math.cos(float(heading_rad))
        sin_h = math.sin(float(heading_rad))

        gauss_x = center_gauss_x + self._x_vehicle * cos_h - self._y_vehicle * sin_h
        gauss_y = center_gauss_y + self._x_vehicle * sin_h + self._y_vehicle * cos_h

        cc = self.cc
        u = (gauss_x - cc.tl_gauss_x) / cc.res_gauss_x
        v = -(gauss_y - cc.tl_gauss_y) / cc.res_gauss_y

        dem_w_m1 = max(float(self.dem_w - 1), 1.0)
        dem_h_m1 = max(float(self.dem_h - 1), 1.0)
        grid_x = 2.0 * u / dem_w_m1 - 1.0
        grid_y = 2.0 * v / dem_h_m1 - 1.0
        return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)

    def crop(self, center_gauss_x, center_gauss_y, heading_rad):
        grid = self._build_sample_grid(center_gauss_x, center_gauss_y, heading_rad)
        patch = F.grid_sample(
            self.dem_tensor,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return {
            "patch": patch,
            "grid": grid,
            "shape": (self.rows, self.cols),
            "resolution": self.resolution,
            "map_size_x": self.map_size_x,
            "map_size_y": self.map_size_y,
        }

    def crop_with_bev_layers(self, center_gauss_x, center_gauss_y, heading_rad):
        result = self.crop(center_gauss_x, center_gauss_y, heading_rad)
        patch_2d = result["patch"].squeeze(0).squeeze(0)

        h_rel, g_long, g_lat, center = _compute_bev_layers_torch(
            patch_2d,
            self.resolution,
            self._sobel_x,
            self._sobel_y,
            h_rel_deadzone_half=self.h_rel_deadzone_half,
            grad_deadzone_half=self.grad_deadzone_half,
        )

        layers = {
            "H_rel_surf": h_rel.detach().cpu().numpy().astype(np.float32),
            "G_long_L": g_long.detach().cpu().numpy().astype(np.float32),
            "G_lat_L": g_lat.detach().cpu().numpy().astype(np.float32),
            "center_height": float(center.detach().cpu().item()),
        }
        result["elevation"] = layers["H_rel_surf"] + layers["center_height"]
        result["layers"] = layers
        result["vis_bgr"] = self.layers_to_vis_bgr(layers)
        return result

    def crop_numpy(self, center_gauss_x, center_gauss_y, heading_rad):
        result = self.crop(center_gauss_x, center_gauss_y, heading_rad)
        patch = result["patch"].detach().cpu().numpy()[0, 0]
        return patch, np.isfinite(patch)

    def layers_to_vis_bgr(self, layers):
        return render_bev_layers_stack(layers)

    def patch_to_vis_bgr(self, patch):
        layers = compute_bev_layers(
            patch,
            resolution=self.resolution,
            device=self.device,
            h_rel_deadzone_half=self.h_rel_deadzone_half,
            grad_deadzone_half=self.grad_deadzone_half,
        )
        return self.layers_to_vis_bgr(layers)
