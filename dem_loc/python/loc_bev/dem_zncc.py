#!/usr/bin/env python3
"""DEM patch sampling and ZNCC scoring for TerrainMap roughness."""

import math

import cv2
import numpy as np


def terrain_array_to_grid(value, width=0, height=0):
    if isinstance(value, (bytes, bytearray)):
        data = np.frombuffer(value, dtype=np.uint8).copy()
    else:
        data = np.asarray(value, dtype=np.uint8)

    if width > 0 and height > 0:
        rows, cols = int(height), int(width)
        if rows * cols != data.size:
            raise ValueError(
                "terrain array size {} does not match {}x{}".format(
                    data.size, rows, cols
                )
            )
    else:
        side = int(math.sqrt(data.size))
        if side * side != data.size:
            raise ValueError("cannot infer square terrain grid from size {}".format(data.size))
        rows, cols = side, side

    return data.reshape((rows, cols))


def heading_to_math_degrees(heading_deg, convention="math"):
    convention = convention.lower()
    if convention == "math":
        return float(heading_deg)
    if convention == "azimuth":
        return 90.0 - float(heading_deg)
    raise ValueError("heading convention must be 'math' or 'azimuth'")


def build_local_grid_offsets(rows, cols, resolution):
    row_idx, col_idx = np.indices((rows, cols), dtype=np.float32)
    half_rows = rows / 2.0
    half_cols = cols / 2.0
    # Vehicle-frame image convention: top is +x/front, left is +y/left.
    car_x = -(row_idx - half_rows) * float(resolution)
    car_y = -(col_idx - half_cols) * float(resolution)
    return car_x.astype(np.float64), car_y.astype(np.float64)


def sample_dem_patch(
    dem_data,
    converter,
    center_lon,
    center_lat,
    heading_deg,
    rows,
    cols,
    terrain_resolution=0.2,
    heading_convention="math",
):
    center_gauss = converter.wgs84_to_gauss(center_lon, center_lat)
    return sample_dem_patch_gauss(
        dem_data,
        converter,
        center_gauss.x,
        center_gauss.y,
        heading_deg,
        rows,
        cols,
        terrain_resolution=terrain_resolution,
        heading_convention=heading_convention,
    )


def sample_dem_patch_gauss(
    dem_data,
    converter,
    center_gauss_x,
    center_gauss_y,
    heading_deg,
    rows,
    cols,
    terrain_resolution=0.2,
    heading_convention="math",
    local_offsets=None,
):
    if local_offsets is None:
        car_x, car_y = build_local_grid_offsets(rows, cols, terrain_resolution)
    else:
        car_x, car_y = local_offsets

    theta = heading_to_math_degrees(heading_deg, heading_convention) * math.pi / 180.0
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    gauss_x = float(center_gauss_x) + cos_t * car_x - sin_t * car_y
    gauss_y = float(center_gauss_y) + sin_t * car_x + cos_t * car_y

    lon, lat = converter.gauss_to_wgs84_np(gauss_x, gauss_y)
    map_x, map_y = converter.wgs84_to_pixel_np(lon, lat)

    valid = (
        (map_x >= 0.0)
        & (map_x <= dem_data.raw_elevation_map.shape[1] - 1)
        & (map_y >= 0.0)
        & (map_y <= dem_data.raw_elevation_map.shape[0] - 1)
    )

    sampled = cv2.remap(
        dem_data.raw_elevation_map,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )
    valid &= np.isfinite(sampled)
    sampled = sampled.astype(np.float32)
    return sampled, valid, map_x, map_y


def _sample_global_xy_patch(
    dem_data,
    converter,
    center_gauss_x,
    center_gauss_y,
    rows,
    cols,
    terrain_resolution=0.2,
):
    # Matches GridMap_v2::opencv_showMaps before rotation: image top is +X,
    # image left is +Y. Rotation then turns that map frame into vehicle frame.
    local_x, local_y = build_local_grid_offsets(rows, cols, terrain_resolution)
    gauss_x = float(center_gauss_x) + local_x
    gauss_y = float(center_gauss_y) + local_y

    lon, lat = converter.gauss_to_wgs84_np(gauss_x, gauss_y)
    map_x, map_y = converter.wgs84_to_pixel_np(lon, lat)

    valid = (
        (map_x >= 0.0)
        & (map_x <= dem_data.raw_elevation_map.shape[1] - 1)
        & (map_y >= 0.0)
        & (map_y <= dem_data.raw_elevation_map.shape[0] - 1)
    )

    sampled = cv2.remap(
        dem_data.raw_elevation_map,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    ).astype(np.float32)
    valid &= np.isfinite(sampled)
    return sampled, valid, map_x, map_y


def _center_crop(image, rows, cols):
    start_row = max(0, (image.shape[0] - rows) // 2)
    start_col = max(0, (image.shape[1] - cols) // 2)
    return image[start_row : start_row + rows, start_col : start_col + cols]


def sample_dem_patch_crop_rotate(
    dem_data,
    converter,
    center_lon,
    center_lat,
    heading_deg,
    rows,
    cols,
    terrain_resolution=0.2,
    heading_convention="math",
    expand_crop=True,
):
    center_gauss = converter.wgs84_to_gauss(center_lon, center_lat)
    return sample_dem_patch_gauss_crop_rotate(
        dem_data,
        converter,
        center_gauss.x,
        center_gauss.y,
        heading_deg,
        rows,
        cols,
        terrain_resolution=terrain_resolution,
        heading_convention=heading_convention,
        expand_crop=expand_crop,
    )


def sample_dem_patch_gauss_crop_rotate(
    dem_data,
    converter,
    center_gauss_x,
    center_gauss_y,
    heading_deg,
    rows,
    cols,
    terrain_resolution=0.2,
    heading_convention="math",
    expand_crop=True,
):
    if expand_crop:
        crop_size = int(math.ceil(math.sqrt(rows * rows + cols * cols)))
        crop_rows = max(rows, crop_size)
        crop_cols = max(cols, crop_size)
    else:
        crop_rows = rows
        crop_cols = cols

    global_patch, global_valid, _map_x, _map_y = _sample_global_xy_patch(
        dem_data,
        converter,
        center_gauss_x,
        center_gauss_y,
        crop_rows,
        crop_cols,
        terrain_resolution=terrain_resolution,
    )

    angle_degrees = -heading_to_math_degrees(heading_deg, heading_convention)
    center = (crop_cols / 2.0, crop_rows / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    rotated = cv2.warpAffine(
        global_patch,
        rotation_matrix,
        (crop_cols, crop_rows),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    ).astype(np.float32)

    valid_u8 = np.where(global_valid, 255, 0).astype(np.uint8)
    rotated_valid_u8 = cv2.warpAffine(
        valid_u8,
        rotation_matrix,
        (crop_cols, crop_rows),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rotated_valid = (rotated_valid_u8 > 0) & np.isfinite(rotated)

    if crop_rows != rows or crop_cols != cols:
        rotated = _center_crop(rotated, rows, cols)
        rotated_valid = _center_crop(rotated_valid, rows, cols)

    return rotated, rotated_valid, None, None


def _odd_kernel_size(value):
    kernel = max(1, int(round(float(value))))
    if kernel % 2 == 0:
        kernel += 1
    return kernel


def dem_patch_to_compare_image(
    dem_patch,
    mode="center_roughness",
    roughness_scale_m=2.5,
    roughness_window_px=5,
):
    mode = mode.lower()
    patch = dem_patch.astype(np.float32, copy=True)

    if mode == "elevation":
        return patch

    finite = np.isfinite(patch)
    if not finite.any():
        return np.zeros_like(patch, dtype=np.float32)

    center_row = patch.shape[0] // 2
    center_col = patch.shape[1] // 2
    center_value = float(patch[center_row, center_col])
    if not np.isfinite(center_value):
        center_value = float(np.nanmean(patch))

    if mode in ("center_relief", "center_roughness"):
        relief = patch - center_value
        if mode == "center_roughness":
            relief = np.clip(relief / max(float(roughness_scale_m), 1e-6), 0.0, 1.0)
            return relief * 255.0
        return relief

    if mode in ("relief", "roughness"):
        local_min = float(np.nanmin(patch))
        relief = patch - local_min
        if mode == "roughness":
            relief = np.clip(relief / max(float(roughness_scale_m), 1e-6), 0.0, 1.0)
            return relief * 255.0
        return relief

    if mode in ("local_roughness", "window_roughness"):
        filled = np.where(finite, patch, float(np.nanmean(patch))).astype(np.float32)
        kernel_size = _odd_kernel_size(roughness_window_px)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        local_max = cv2.dilate(filled, kernel)
        local_min = cv2.erode(filled, kernel)
        roughness = local_max - local_min
        roughness = np.clip(roughness / max(float(roughness_scale_m), 1e-6), 0.0, 1.0)
        return roughness * 255.0

    if mode == "gradient":
        filled = np.where(finite, patch, float(np.nanmean(patch)))
        grad_x = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(grad_x, grad_y)

    raise ValueError("unknown DEM compare mode: {}".format(mode))


def zncc(a, b, mask=None, eps=1e-6):
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    if arr_a.shape != arr_b.shape:
        raise ValueError("ZNCC shape mismatch: {} vs {}".format(arr_a.shape, arr_b.shape))

    valid = np.isfinite(arr_a) & np.isfinite(arr_b)
    if mask is not None:
        valid &= mask.astype(bool)

    count = int(np.count_nonzero(valid))
    if count < 2:
        return float("nan"), count

    va = arr_a[valid]
    vb = arr_b[valid]
    va = va - float(va.mean())
    vb = vb - float(vb.mean())

    denom = float(np.sqrt(np.sum(va * va) * np.sum(vb * vb)))
    if denom <= eps:
        return float("nan"), count
    return float(np.sum(va * vb) / denom), count


def apply_image_transform(image, transform_name):
    name = transform_name.lower()
    if name in ("none", "identity", ""):
        return image
    if name == "rot90_cw":
        return np.rot90(image, k=3)
    if name == "rot90_ccw":
        return np.rot90(image, k=1)
    if name == "rot180":
        return np.rot90(image, k=2)
    if name == "flip_lr":
        return np.fliplr(image)
    if name == "flip_ud":
        return np.flipud(image)
    if name == "transpose":
        return np.transpose(image)
    if name == "anti_transpose":
        return np.flipud(np.fliplr(np.transpose(image)))
    raise ValueError("unknown image transform: {}".format(transform_name))


def candidate_image_transforms():
    return (
        "none",
        "rot90_cw",
        "rot90_ccw",
        "rot180",
        "flip_lr",
        "flip_ud",
        "transpose",
        "anti_transpose",
    )


def normalize_for_display(image):
    data = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros(data.shape, dtype=np.uint8)
    min_value = float(np.nanmin(data))
    max_value = float(np.nanmax(data))
    denom = max(max_value - min_value, 1e-6)
    out = (data - min_value) * 255.0 / denom
    out = np.where(finite, out, 0.0)
    return np.clip(out, 0, 255).astype(np.uint8)
