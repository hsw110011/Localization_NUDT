#!/usr/bin/env python3
"""坐标转换 — 对应 C++ PathAnalysis.cpp / Tool.cpp / CoordConverter.cpp。

坐标系:
    WGS84 (经纬度) <-> UTM (米) <-> DSM 像素 (u/v)

兼容性约定:
    历史接口中的 ``gauss`` 名称保持不变，但其坐标语义统一为
    WGS84 / UTM：x=Easting，y=Northing。

主要接口:
    CoordConverter(dem_data)
    wgs84_to_gauss / gauss_to_wgs84 / wgs84_to_pixel
    GetBaseFromLocalPose / LocalPoseToGlobal / local_pose_to_dem_pixel
    global_pose_to_dem_pixel

被 localization_python.py 用于将 GlobalPose / LocalPose 投影到 DSM 像素坐标。
"""

import math

import numpy as np

from .common_struct import BLH_Point, DEG_TO_RAD, GaussPoint, RAD_TO_DEG, WORLD_POINT, double3D


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
UTM_SCALE_FACTOR = 0.9996
UTM_FALSE_EASTING = 500000.0
UTM_FALSE_NORTHING_SOUTH = 10000000.0
UTM_MIN_LAT = -80.0
UTM_MAX_LAT = 84.0
UTM_INPUT_MARGIN_M = 100.0


class PixelPoint(object):
    __slots__ = ("x", "y")

    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)


def _normalize_heading_deg(heading):
    heading = float(heading)
    while heading < 0.0:
        heading += 360.0
    while heading >= 360.0:
        heading -= 360.0
    return heading


def _angle_to_deg(value, unit="deg"):
    unit = str(unit).lower()
    value = float(value)
    if unit == "rad":
        return value * RAD_TO_DEG
    if unit == "auto" and abs(value) <= (2.0 * math.pi + 0.25):
        return value * RAD_TO_DEG
    return value


def _heading_to_math_deg(value, unit="deg", convention="math"):
    """Convert heading to math convention (East=0 deg, CCW).

    Angle conventions (see degree_set.png):
        LocalPose.dr_heading / GlobalPose.azimuth: math, East=0, CCW (与 C++ 一致)
        base.theta: radians, local frame -> Gauss frame rotation (handled separately)

    ``azimuth`` is kept only for legacy bags that store navigation azimuth (North=0, CW).
    """
    heading = _angle_to_deg(value, unit)
    if str(convention).lower() == "azimuth":
        heading = 90.0 - heading
    return heading


def _finite_pair(x_value, y_value):
    return np.isfinite(float(x_value)) and np.isfinite(float(y_value))


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
        if not 1 <= self.zone_num <= 60:
            raise ValueError("UTM zone_num must be in [1, 60]")

        self.tl_lon = float(dem_tiff_data.geo_bounds[0])
        self.tl_lat = float(dem_tiff_data.geo_bounds[1])
        self.br_lon = float(dem_tiff_data.geo_bounds[2])
        self.br_lat = float(dem_tiff_data.geo_bounds[3])
        center_lat = 0.5 * (self.tl_lat + self.br_lat)
        if not UTM_MIN_LAT <= center_lat <= UTM_MAX_LAT:
            raise ValueError("UTM latitude must be in [-80, 84] degrees")
        self.utm_false_northing = (
            0.0 if center_lat >= 0.0 else UTM_FALSE_NORTHING_SOUTH
        )

        self.res_lon = float(dem_tiff_data.resolution_x)
        self.res_lat = float(dem_tiff_data.resolution_y)

        gauss_tl = self.wgs84_to_gauss(self.tl_lon, self.tl_lat)
        gauss_br = self.wgs84_to_gauss(self.br_lon, self.br_lat)
        self.tl_gauss_x = gauss_tl.x
        self.tl_gauss_y = gauss_tl.y
        corner_lon = np.asarray(
            [self.tl_lon, self.br_lon, self.tl_lon, self.br_lon],
            dtype=np.float64,
        )
        corner_lat = np.asarray(
            [self.tl_lat, self.tl_lat, self.br_lat, self.br_lat],
            dtype=np.float64,
        )
        corner_x, corner_y = self.wgs84_to_gauss_np(corner_lon, corner_lat)
        self.utm_min_x = float(np.min(corner_x))
        self.utm_max_x = float(np.max(corner_x))
        self.utm_min_y = float(np.min(corner_y))
        self.utm_max_y = float(np.max(corner_y))
        map_res = float(dem_tiff_data.resolution)
        if map_res >= 0.01:
            self.res_gauss_x = map_res
            self.res_gauss_y = map_res
        else:
            self.res_gauss_x = (gauss_br.x - gauss_tl.x) / max(float(self.map_width - 1), 1.0)
            self.res_gauss_y = (gauss_tl.y - gauss_br.y) / max(float(self.map_height - 1), 1.0)

    def is_wgs84_in_bounds(self, lon, lat, margin_deg=0.0):
        margin = float(margin_deg)
        lon_v = float(lon)
        lat_v = float(lat)
        return (
            (self.tl_lon - margin) <= lon_v <= (self.br_lon + margin)
            and (self.br_lat - margin) <= lat_v <= (self.tl_lat + margin)
        )

    def is_pixel_in_bounds(self, pixel, margin_px=0.0):
        margin = float(margin_px)
        return (
            pixel is not None
            and -margin <= float(pixel.x) < (self.map_width + margin)
            and -margin <= float(pixel.y) < (self.map_height + margin)
        )

    def is_utm_in_bounds(self, utm_x, utm_y, margin_m=0.0):
        """检查历史 gauss 字段是否符合当前地图的 UTM 坐标范围。"""
        if not _finite_pair(utm_x, utm_y):
            return False
        margin = float(margin_m)
        x_value = float(utm_x)
        y_value = float(utm_y)
        return (
            (self.utm_min_x - margin) <= x_value <= (self.utm_max_x + margin)
            and (self.utm_min_y - margin) <= y_value <= (self.utm_max_y + margin)
        )

    def wgs84_to_gauss(self, lon, lat):
        gauss_x, gauss_y = self.wgs84_to_gauss_np(lon, lat)
        return GaussPoint(float(gauss_x), float(gauss_y), 0.0)

    def gauss_to_wgs84(self, gauss_x, gauss_y):
        lon, lat = self.gauss_to_wgs84_np(gauss_x, gauss_y)
        return BLH_Point(float(lon), float(lat), 0.0)

    def wgs84_to_pixel(self, lon, lat):
        gauss_x, gauss_y = self.wgs84_to_gauss_np(lon, lat)
        u = (gauss_x - self.tl_gauss_x) / self.res_gauss_x
        v = -(gauss_y - self.tl_gauss_y) / self.res_gauss_y
        return PixelPoint(float(u), float(v))

    def wgs84_to_dem_pixel(self, lon, lat):
        lon_v = float(lon)
        lat_v = float(lat)
        if not np.isfinite(lon_v) or not np.isfinite(lat_v):
            return None
        if not self.is_wgs84_in_bounds(lon_v, lat_v):
            return None
        pixel = self.wgs84_to_pixel(lon_v, lat_v)
        if not self.is_pixel_in_bounds(pixel):
            return None
        return pixel

    def gauss_to_pixel(self, gauss_x, gauss_y):
        u = (float(gauss_x) - self.tl_gauss_x) / self.res_gauss_x
        v = -(float(gauss_y) - self.tl_gauss_y) / self.res_gauss_y
        return PixelPoint(float(u), float(v))

    def gauss_to_dem_pixel(self, gauss_x, gauss_y):
        if not _finite_pair(gauss_x, gauss_y):
            return None
        blh = self.gauss_to_wgs84(gauss_x, gauss_y)
        return self.wgs84_to_dem_pixel(blh.Lon, blh.Lat)

    def GetBaseFromLocalPose(
        self,
        local_pose,
        global_pose,
        local_heading_unit="deg",
        local_heading_convention="math",
        global_heading_unit="deg",
        global_heading_convention="math",
    ):
        gauss_x, gauss_y, global_heading = self._extract_global_gauss_heading(
            global_pose,
            heading_unit=global_heading_unit,
            heading_convention=global_heading_convention,
        )
        local_heading = _heading_to_math_deg(
            local_pose.dr_heading,
            local_heading_unit,
            local_heading_convention,
        )
        return self._compute_base(
            float(local_pose.dr_x),
            float(local_pose.dr_y),
            local_heading,
            gauss_x,
            gauss_y,
            global_heading,
        )

    def LocalPoseToGlobal(
        self,
        local_pose,
        base_point,
        local_heading_unit="deg",
        local_heading_convention="math",
    ):
        return self._local_to_global(
            float(local_pose.dr_x),
            float(local_pose.dr_y),
            _heading_to_math_deg(
                local_pose.dr_heading,
                local_heading_unit,
                local_heading_convention,
            ),
            base_point,
            timeflag=float(getattr(local_pose, "local_time", 0.0)),
        )

    def global_pose_to_dem_pixel(self, global_pose):
        lon = float(getattr(global_pose, "longitude", float("nan")))
        lat = float(getattr(global_pose, "latitude", float("nan")))
        if np.isfinite(lon) and np.isfinite(lat):
            pixel = self.wgs84_to_dem_pixel(lon, lat)
            if pixel is not None:
                return pixel

        gauss_xy = self._get_global_pose_gauss_xy(global_pose)
        if gauss_xy is not None:
            pixel = self.gauss_to_pixel(gauss_xy[0], gauss_xy[1])
            if self.is_pixel_in_bounds(pixel):
                return pixel

        return None

    def local_pose_to_dem_pixel(
        self,
        local_pose,
        base_point,
        local_heading_unit="deg",
        local_heading_convention="math",
    ):
        global_point = self.LocalPoseToGlobal(
            local_pose,
            base_point,
            local_heading_unit=local_heading_unit,
            local_heading_convention=local_heading_convention,
        )
        return self.gauss_to_dem_pixel(global_point.gauss.x, global_point.gauss.y)

    def _compute_base(
        self,
        local_x,
        local_y,
        local_heading_deg,
        global_gauss_x,
        global_gauss_y,
        global_heading_deg,
    ):
        theta = (float(global_heading_deg) - float(local_heading_deg)) * DEG_TO_RAD
        base = double3D()
        base.theta = theta
        base.x = float(global_gauss_x) - (
            float(local_x) * math.cos(theta) - float(local_y) * math.sin(theta)
        )
        base.y = float(global_gauss_y) - (
            float(local_x) * math.sin(theta) + float(local_y) * math.cos(theta)
        )
        return base

    def _local_to_global(
        self,
        local_x,
        local_y,
        local_heading_deg,
        base_point,
        timeflag=0.0,
    ):
        theta = float(base_point.theta)
        global_point = WORLD_POINT()
        global_point.timeflag = float(timeflag)
        global_point.gauss.x = (
            float(local_x) * math.cos(theta)
            - float(local_y) * math.sin(theta)
            + float(base_point.x)
        )
        global_point.gauss.y = (
            float(local_x) * math.sin(theta)
            + float(local_y) * math.cos(theta)
            + float(base_point.y)
        )
        global_point.gauss.z = 0.0
        global_point.heading = _normalize_heading_deg(
            float(local_heading_deg) + theta * RAD_TO_DEG
        )
        global_point.BLH = self.gauss_to_wgs84(global_point.gauss.x, global_point.gauss.y)
        global_point.pixel = self.wgs84_to_pixel(global_point.BLH.Lon, global_point.BLH.Lat)
        return global_point

    def _get_global_pose_gauss_xy(self, global_point):
        if hasattr(global_point, "gaussX") and hasattr(global_point, "gaussY"):
            gauss_x = float(global_point.gaussX)
            gauss_y = float(global_point.gaussY)
            if self.is_utm_in_bounds(
                gauss_x, gauss_y, margin_m=UTM_INPUT_MARGIN_M
            ):
                return gauss_x, gauss_y

        if hasattr(global_point, "gauss"):
            gauss_x = float(global_point.gauss.x)
            gauss_y = float(global_point.gauss.y)
            if self.is_utm_in_bounds(
                gauss_x, gauss_y, margin_m=UTM_INPUT_MARGIN_M
            ):
                return gauss_x, gauss_y

        return None

    def _extract_global_gauss_heading(
        self,
        global_point,
        heading_unit="deg",
        heading_convention="math",
    ):
        lon = float(getattr(global_point, "longitude", float("nan")))
        lat = float(getattr(global_point, "latitude", float("nan")))
        gauss_x = float("nan")
        gauss_y = float("nan")
        if (
            np.isfinite(lon)
            and np.isfinite(lat)
            and -180.0 <= lon <= 180.0
            and UTM_MIN_LAT <= lat <= UTM_MAX_LAT
        ):
            projected = self.wgs84_to_gauss(lon, lat)
            if self.is_utm_in_bounds(
                projected.x, projected.y, margin_m=UTM_INPUT_MARGIN_M
            ):
                gauss_x = projected.x
                gauss_y = projected.y

        if not _finite_pair(gauss_x, gauss_y):
            gauss_xy = self._get_global_pose_gauss_xy(global_point)
            if gauss_xy is None:
                raise ValueError(
                    "global_point must provide map-local WGS84 or UTM coordinates"
                )
            gauss_x, gauss_y = gauss_xy

        if not _finite_pair(gauss_x, gauss_y):
            raise ValueError("global_point must provide map-local WGS84 or UTM coordinates")

        heading = float(
            getattr(
                global_point,
                "azimuth",
                getattr(global_point, "heading", 0.0),
            )
        )
        return (
            gauss_x,
            gauss_y,
            _heading_to_math_deg(heading, heading_unit, heading_convention),
        )

    def wgs84_to_pixel_np(self, lon, lat):
        gauss_x, gauss_y = self.wgs84_to_gauss_np(lon, lat)
        u = (gauss_x - self.tl_gauss_x) / self.res_gauss_x
        v = -(gauss_y - self.tl_gauss_y) / self.res_gauss_y
        return u.astype(np.float32), v.astype(np.float32)

    def wgs84_to_gauss_np(self, lon, lat):
        lon_arr = np.asarray(lon, dtype=np.float64)
        lat_arr = np.asarray(lat, dtype=np.float64)

        lon_rad = lon_arr * DEG_TO_RAD
        lat_rad = lat_arr * DEG_TO_RAD

        cm_deg = -180.0 + (float(self.zone_num) * 6.0) - 3.0
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

        return (
            gauss_x + UTM_FALSE_EASTING,
            gauss_y + self.utm_false_northing,
        )

    def gauss_to_wgs84_np(self, gauss_x, gauss_y):
        x_val = np.asarray(gauss_x, dtype=np.float64) - UTM_FALSE_EASTING
        y_val = (
            np.asarray(gauss_y, dtype=np.float64) - self.utm_false_northing
        )

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
