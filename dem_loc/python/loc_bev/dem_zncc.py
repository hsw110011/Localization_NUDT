#!/usr/bin/env python3
"""DEM patch sampling helpers."""

import math

import cv2
import numpy as np


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
):
    # Crop target patch in map-aligned frame, then rotate to vehicle frame.
    global_patch, global_valid, _map_x, _map_y = _sample_global_xy_patch(
        dem_data,
        converter,
        center_gauss_x,
        center_gauss_y,
        rows,
        cols,
        terrain_resolution=terrain_resolution,
    )

    angle_degrees = -heading_to_math_degrees(heading_deg, heading_convention)
    center = (cols / 2.0, rows / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    rotated = cv2.warpAffine(
        global_patch,
        rotation_matrix,
        (cols, rows),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    ).astype(np.float32)

    valid_u8 = np.where(global_valid, 255, 0).astype(np.uint8)
    rotated_valid_u8 = cv2.warpAffine(
        valid_u8,
        rotation_matrix,
        (cols, rows),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rotated_valid = (rotated_valid_u8 > 0) & np.isfinite(rotated)

    return rotated, rotated_valid, None, None
