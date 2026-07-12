#!/usr/bin/env python3
"""DEM GeoTIFF 加载工具 — 对应 C++ Tool::LoadDemTiff。

主要接口:
    load_dem_tiff(path, map_resolution_m=0.2) -> DemTiffData
    normalize_to_uint8(image, min_value, max_value) -> np.ndarray

被 localization_python.py 用于加载 DSM 并生成伪彩色底图。
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from osgeo import gdal, osr


DEM_MAP_RESOLUTION_M = 0.2


@dataclass
class DemTiffData:
    visual_map: np.ndarray = field(default_factory=lambda: np.empty((0, 0, 3), np.uint8))
    raw_elevation_map: np.ndarray = field(default_factory=lambda: np.empty((0, 0), np.float32))
    geo_bounds: list = field(default_factory=list)
    resolution_x: float = 0.0
    resolution_y: float = 0.0
    resolution: float = 0.0
    min_height: float = 0.0
    max_height: float = 0.0
    zone_num: int = 0
    is_valid: bool = False


def normalize_to_uint8(image, min_value=None, max_value=None):
    data = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros(data.shape, dtype=np.uint8)

    if min_value is None:
        min_value = float(np.nanmin(data))
    if max_value is None:
        max_value = float(np.nanmax(data))

    denom = max(max_value - min_value, 1e-6)
    out = (data - min_value) * (255.0 / denom)
    out = np.where(finite, out, 0.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def load_dem_tiff(tiff_path, map_resolution_m=DEM_MAP_RESOLUTION_M):
    out_data = DemTiffData()
    dataset = None

    try:
        gdal.AllRegister()
        dataset = gdal.Open(tiff_path, gdal.GA_ReadOnly)
        if dataset is None:
            raise RuntimeError("failed to open TIFF file: {}".format(tiff_path))

        width = int(dataset.RasterXSize)
        height = int(dataset.RasterYSize)

        geotransform = dataset.GetGeoTransform(can_return_null=True)
        if geotransform is None:
            raise RuntimeError("GeoTransform not found in TIFF")
        if abs(float(geotransform[2])) > 1e-12 or abs(float(geotransform[4])) > 1e-12:
            raise RuntimeError("rotated GeoTIFF is not supported")

        projection = dataset.GetProjection()
        spatial_ref = osr.SpatialReference()
        if not projection or spatial_ref.ImportFromWkt(projection) != 0:
            raise RuntimeError("GeoTIFF CRS is missing or invalid")
        if not spatial_ref.IsGeographic():
            raise RuntimeError("GeoTIFF must use geographic WGS84 coordinates")
        if (
            abs(float(spatial_ref.GetSemiMajor()) - 6378137.0) > 1e-3
            or abs(float(spatial_ref.GetInvFlattening()) - 298.257223563) > 1e-9
        ):
            raise RuntimeError("GeoTIFF geographic CRS must use the WGS84 ellipsoid")

        out_data.resolution_x = float(geotransform[1])
        out_data.resolution_y = abs(float(geotransform[5]))
        out_data.resolution = 0.0

        min_lon = float(geotransform[0])
        max_lat = float(geotransform[3])
        max_lon = min_lon + width * out_data.resolution_x
        min_lat = max_lat + height * float(geotransform[5])
        out_data.geo_bounds = [min_lon, max_lat, max_lon, min_lat]

        center_lon = (min_lon + max_lon) / 2.0
        out_data.zone_num = int((center_lon + 180.0) / 6.0) + 1

        band = dataset.GetRasterBand(1)
        if band is None:
            raise RuntimeError("first raster band not found")

        min_height = band.GetMinimum()
        max_height = band.GetMaximum()

        raw = band.ReadAsArray().astype(np.float32)
        nodata = band.GetNoDataValue()
        if nodata is not None:
            raw = np.where(raw == nodata, np.nan, raw).astype(np.float32)

        if min_height is None or max_height is None:
            finite = np.isfinite(raw)
            if finite.any():
                out_data.min_height = float(np.nanmin(raw))
                out_data.max_height = float(np.nanmax(raw))
            else:
                out_data.min_height = 0.0
                out_data.max_height = 0.0
        else:
            out_data.min_height = float(min_height)
            out_data.max_height = float(max_height)

        out_data.raw_elevation_map = raw
        gray_map = normalize_to_uint8(raw, out_data.min_height, out_data.max_height)
        out_data.visual_map = cv2.applyColorMap(gray_map, cv2.COLORMAP_JET)
        out_data.is_valid = True

        if map_resolution_m and float(map_resolution_m) > 0.0:
            resample_dem_to_gauss_resolution(out_data, float(map_resolution_m))

        return out_data

    except Exception as exc:
        out_data.is_valid = False
        return out_data

    finally:
        dataset = None


def resample_dem_to_gauss_resolution(dem_data, target_res_m, skip_tolerance=0.01):
    """Resample the full DEM to a uniform UTM grid (meters per pixel).

    Top-left / bottom-right WGS84 bounds are updated to match the new grid.
    Historical ``gauss`` names are retained for API compatibility.
    """
    from .coord_converter import CoordConverter

    target_res_m = float(target_res_m)
    if target_res_m <= 0.0:
        return dem_data

    src = CoordConverter(dem_data)
    if (
        abs(src.res_gauss_x - target_res_m) / target_res_m <= skip_tolerance
        and abs(src.res_gauss_y - target_res_m) / target_res_m <= skip_tolerance
    ):
        dem_data.resolution = target_res_m
        return dem_data

    gauss_tl = src.wgs84_to_gauss(src.tl_lon, src.tl_lat)
    gauss_br = src.wgs84_to_gauss(src.br_lon, src.br_lat)

    extent_x = gauss_br.x - gauss_tl.x
    extent_y = gauss_tl.y - gauss_br.y
    new_width = max(1, int(round(extent_x / target_res_m)) + 1)
    new_height = max(1, int(round(extent_y / target_res_m)) + 1)

    u = np.arange(new_width, dtype=np.float32)
    v = np.arange(new_height, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u, v)

    gauss_x = gauss_tl.x + u_grid.astype(np.float64) * target_res_m
    gauss_y = gauss_tl.y - v_grid.astype(np.float64) * target_res_m

    lon, lat = src.gauss_to_wgs84_np(gauss_x, gauss_y)
    map_x = ((lon - src.tl_lon) / src.res_lon).astype(np.float32)
    map_y = (-(lat - src.tl_lat) / src.res_lat).astype(np.float32)

    resampled = cv2.remap(
        dem_data.raw_elevation_map,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    ).astype(np.float32)

    br_gauss_x = gauss_tl.x + (new_width - 1) * target_res_m
    br_gauss_y = gauss_tl.y - (new_height - 1) * target_res_m
    br_blh = src.gauss_to_wgs84(br_gauss_x, br_gauss_y)

    dem_data.raw_elevation_map = resampled
    dem_data.geo_bounds = [src.tl_lon, src.tl_lat, br_blh.Lon, br_blh.Lat]
    if new_width > 1:
        dem_data.resolution_x = (br_blh.Lon - src.tl_lon) / float(new_width - 1)
    else:
        dem_data.resolution_x = 0.0
    if new_height > 1:
        dem_data.resolution_y = (src.tl_lat - br_blh.Lat) / float(new_height - 1)
    else:
        dem_data.resolution_y = 0.0
    dem_data.resolution = target_res_m

    finite = np.isfinite(resampled)
    if finite.any():
        dem_data.min_height = float(np.nanmin(resampled))
        dem_data.max_height = float(np.nanmax(resampled))
    else:
        dem_data.min_height = 0.0
        dem_data.max_height = 0.0

    gray_map = normalize_to_uint8(
        resampled, dem_data.min_height, dem_data.max_height
    )
    dem_data.visual_map = cv2.applyColorMap(gray_map, cv2.COLORMAP_JET)
    return dem_data
