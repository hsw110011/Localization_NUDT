#!/usr/bin/env python3
"""Coordinate conversion helpers ported from CoordConverter.cpp."""

import math
from dataclasses import dataclass

import numpy as np

from .common_struct import BLH_Point, DEG_TO_RAD, GaussPoint, RAD_TO_DEG


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
UTM_SCALE_FACTOR = 0.9996


@dataclass
class PixelPoint:
    x: float = 0.0
    y: float = 0.0


class CoordConverter(object):
    def __init__(self, dem_tiff_data):
        f = WGS84_F
        self.e2 = 2.0 * f - f * f
        self.e1 = (1.0 - math.sqrt(1.0 - self.e2)) / (
            1.0 + math.sqrt(1.0 - self.e2)
        )

        self.map_width = int(dem_tiff_data.raw_elevation_map.shape[1])
        self.map_height = int(dem_tiff_data.raw_elevation_map.shape[0])
        self.zone_num = int(dem_tiff_data.zone_num)
        self.tl_lon = float(dem_tiff_data.geo_bounds[0])
        self.tl_lat = float(dem_tiff_data.geo_bounds[1])
        self.br_lon = float(dem_tiff_data.geo_bounds[2])
        self.br_lat = float(dem_tiff_data.geo_bounds[3])
        self.res = float(dem_tiff_data.resolution)

        self.res_lon = (self.br_lon - self.tl_lon) / float(self.map_width)
        self.res_lat = (self.tl_lat - self.br_lat) / float(self.map_height)

        gauss_tl = self.wgs84_to_gauss(self.tl_lon, self.tl_lat)
        self.tl_gaussX = gauss_tl.x
        self.tl_gaussY = gauss_tl.y

    def wgs84_to_gauss(self, lon, lat):
        gauss_x, gauss_y = self.wgs84_to_gauss_np(lon, lat)
        return GaussPoint(float(gauss_x), float(gauss_y), 0.0)

    def gauss_to_wgs84(self, gauss_x, gauss_y):
        lon, lat = self.gauss_to_wgs84_np(gauss_x, gauss_y)
        return BLH_Point(float(lon), float(lat), 0.0)

    def gauss_to_pixel(self, gauss_x, gauss_y):
        u = (gauss_x - self.tl_gaussX) / self.res
        v = (self.tl_gaussY - gauss_y) / self.res
        return PixelPoint(float(u), float(v))

    def pixel_to_gauss(self, u, v):
        return GaussPoint(
            float(self.tl_gaussX + u * self.res),
            float(self.tl_gaussY - v * self.res),
            0.0,
        )

    def wgs84_to_pixel(self, lon, lat):
        u = (lon - self.tl_lon) / self.res_lon
        v = (self.tl_lat - lat) / self.res_lat
        return PixelPoint(float(u), float(v))

    def pixel_to_wgs84(self, u, v):
        return BLH_Point(
            float(self.tl_lon + u * self.res_lon),
            float(self.tl_lat - v * self.res_lat),
            0.0,
        )

    def wgs84_to_pixel_np(self, lon, lat):
        lon_arr = np.asarray(lon, dtype=np.float64)
        lat_arr = np.asarray(lat, dtype=np.float64)
        u = (lon_arr - self.tl_lon) / self.res_lon
        v = (self.tl_lat - lat_arr) / self.res_lat
        return u.astype(np.float32), v.astype(np.float32)

    def wgs84_to_gauss_np(self, lon, lat):
        lon_arr = np.asarray(lon, dtype=np.float64)
        lat_arr = np.asarray(lat, dtype=np.float64)

        lon_rad = lon_arr * DEG_TO_RAD
        lat_rad = lat_arr * DEG_TO_RAD

        zone = np.floor((lon_arr + 180.0) / 6.0) + 1.0
        cm_deg = -180.0 + (zone * 6.0) - 3.0
        cm_rad = cm_deg * DEG_TO_RAD

        a = WGS84_A
        k0 = UTM_SCALE_FACTOR
        e2 = self.e2
        ee = e2 / (1.0 - e2)

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        tan_lat = np.tan(lat_rad)

        n_radius = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
        t_val = tan_lat * tan_lat
        c_val = ee * cos_lat * cos_lat
        a_val = (lon_rad - cm_rad) * cos_lat

        meridian = a * (
            (1.0 - e2 / 4.0 - 3.0 * e2 * e2 / 64.0 - 5.0 * e2**3 / 256.0)
            * lat_rad
            - (3.0 * e2 / 8.0 + 3.0 * e2 * e2 / 32.0 + 45.0 * e2**3 / 1024.0)
            * np.sin(2.0 * lat_rad)
            + (15.0 * e2 * e2 / 256.0 + 45.0 * e2**3 / 1024.0)
            * np.sin(4.0 * lat_rad)
            - (35.0 * e2**3 / 3072.0) * np.sin(6.0 * lat_rad)
        )

        gauss_x = k0 * n_radius * (
            a_val
            + (1.0 - t_val + c_val) * a_val**3 / 6.0
            + (5.0 - 18.0 * t_val + t_val * t_val + 72.0 * c_val - 58.0 * ee)
            * a_val**5
            / 120.0
        )
        gauss_y = k0 * (
            meridian
            + n_radius
            * tan_lat
            * (
                a_val**2 / 2.0
                + (5.0 - t_val + 9.0 * c_val + 4.0 * c_val * c_val)
                * a_val**4
                / 24.0
                + (
                    61.0
                    - 58.0 * t_val
                    + t_val * t_val
                    + 600.0 * c_val
                    - 330.0 * ee
                )
                * a_val**6
                / 720.0
            )
        )

        return gauss_x + 500000.0, gauss_y

    def gauss_to_wgs84_np(self, gauss_x, gauss_y):
        x_val = np.asarray(gauss_x, dtype=np.float64) - 500000.0
        y_val = np.asarray(gauss_y, dtype=np.float64)

        a = WGS84_A
        k0 = UTM_SCALE_FACTOR
        e2 = self.e2
        e1 = self.e1

        meridian = y_val / k0
        mu = meridian / (
            a * (1.0 - e2 / 4.0 - 3.0 * e2 * e2 / 64.0 - 5.0 * e2**3 / 256.0)
        )

        phi1 = (
            mu
            + (3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0) * np.sin(2.0 * mu)
            + (21.0 * e1 * e1 / 16.0 - 55.0 * e1**4 / 32.0)
            * np.sin(4.0 * mu)
            + (151.0 * e1**3 / 96.0) * np.sin(6.0 * mu)
        )

        ee = e2 / (1.0 - e2)
        sin_phi1 = np.sin(phi1)
        cos_phi1 = np.cos(phi1)
        tan_phi1 = np.tan(phi1)

        c1 = ee * cos_phi1**2
        t1 = tan_phi1**2
        n1 = a / np.sqrt(1.0 - e2 * sin_phi1**2)
        r1 = a * (1.0 - e2) / np.power(1.0 - e2 * sin_phi1**2, 1.5)
        d_val = x_val / (n1 * k0)

        lat_rad = phi1 - (n1 * tan_phi1 / r1) * (
            d_val**2 / 2.0
            - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1 * c1 - 9.0 * ee)
            * d_val**4
            / 24.0
            + (
                61.0
                + 90.0 * t1
                + 298.0 * c1
                + 45.0 * t1 * t1
                - 252.0 * ee
                - 3.0 * c1 * c1
            )
            * d_val**6
            / 720.0
        )

        lon_rad_diff = (
            d_val
            - (1.0 + 2.0 * t1 + c1) * d_val**3 / 6.0
            + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1 * c1 + 8.0 * ee + 24.0 * t1 * t1)
            * d_val**5
            / 120.0
        ) / cos_phi1

        cm_deg = -180.0 + (self.zone_num * 6.0) - 3.0
        final_lat = lat_rad * RAD_TO_DEG
        final_lon = cm_deg + lon_rad_diff * RAD_TO_DEG
        return final_lon, final_lat
