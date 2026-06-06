#!/usr/bin/env python3
"""DEM GeoTIFF loading utilities ported from Tool::LoadDemTiff."""

from dataclasses import dataclass, field

import cv2
import numpy as np
from osgeo import gdal


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


def load_dem_tiff(tiff_path):
    out_data = DemTiffData()
    dataset = None

    try:
        gdal.AllRegister()
        dataset = gdal.Open(tiff_path, gdal.GA_ReadOnly)
        if dataset is None:
            raise RuntimeError("failed to open TIFF file: {}".format(tiff_path))

        width = int(dataset.RasterXSize)
        height = int(dataset.RasterYSize)
        print("[Info] TIFF Size: {}x{}".format(width, height))

        geotransform = dataset.GetGeoTransform(can_return_null=True)
        if geotransform is None:
            raise RuntimeError("GeoTransform not found in TIFF")

        out_data.resolution_x = float(geotransform[1])
        out_data.resolution_y = abs(float(geotransform[5]))
        out_data.resolution = out_data.resolution_x

        min_lon = float(geotransform[0])
        max_lat = float(geotransform[3])
        max_lon = min_lon + width * out_data.resolution_x
        min_lat = max_lat + height * float(geotransform[5])
        out_data.geo_bounds = [min_lon, max_lat, max_lon, min_lat]

        center_lon = (min_lon + max_lon) / 2.0
        out_data.zone_num = int((center_lon + 180.0) / 6.0) + 1

        print(
            "[Info] Loaded GeoBounds: [L:{}, T:{}, R:{}, B:{}]".format(
                min_lon, max_lat, max_lon, min_lat
            )
        )
        print("[Info] Loaded Resolution: {}".format(out_data.resolution_x))
        print("[Info] UTM Zone: {}".format(out_data.zone_num))

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

        print(
            "[Info] Elevation Range: {} ~ {}".format(
                out_data.min_height, out_data.max_height
            )
        )

        out_data.raw_elevation_map = raw
        gray_map = normalize_to_uint8(raw, out_data.min_height, out_data.max_height)
        out_data.visual_map = cv2.applyColorMap(gray_map, cv2.COLORMAP_JET)
        out_data.is_valid = True
        print("[Info] Generated visual map: {}".format(out_data.visual_map.shape[:2]))
        print("[Info] Successfully loaded TIFF data.")
        return out_data

    except Exception as exc:
        print("[Error] Load TIFF failed: {}".format(exc))
        out_data.is_valid = False
        return out_data

    finally:
        dataset = None
