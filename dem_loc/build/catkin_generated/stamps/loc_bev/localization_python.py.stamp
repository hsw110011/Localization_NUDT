#!/usr/bin/env python3
"""Python DSM crop/visualization node."""

import os
import sys
import time
import warnings

_PYTHON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

import cv2
import numpy as np
import rospy

from loc_bev.cinterface import CInterface
from loc_bev.common_struct import InputData
from loc_bev.coord_converter import CoordConverter
from loc_bev.dem_tool import load_dem_tiff, normalize_to_uint8
from loc_bev.dem_zncc import sample_dem_patch, sample_dem_patch_crop_rotate


def _make_dem_color_map(dem_data):
    gray_map = normalize_to_uint8(
        dem_data.raw_elevation_map, dem_data.min_height, dem_data.max_height
    )
    return _apply_bev_colormap(gray_map)


def _bev_colormap_id():
    return getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET)


def _apply_bev_colormap(gray_u8):
    return cv2.applyColorMap(gray_u8.astype(np.uint8), _bev_colormap_id())


def _draw_track(vis_track, pixel, color=(255, 255, 255), previous=None):
    x = int(round(pixel.x))
    y = int(round(pixel.y))
    if 0 <= x < vis_track.shape[1] and 0 <= y < vis_track.shape[0]:
        if previous is not None:
            cv2.line(vis_track, previous, (x, y), color, 2, cv2.LINE_AA)
        cv2.circle(vis_track, (x, y), 5, color, -1)
        return (x, y)
    return previous


def _create_resizable_window(name, width, height):
    flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    if hasattr(cv2, "WINDOW_GUI_NORMAL"):
        flags |= cv2.WINDOW_GUI_NORMAL
    cv2.namedWindow(name, flags)
    cv2.resizeWindow(name, int(width), int(height))


def _read_key_value_ini(path):
    values = {}
    if not path or not os.path.exists(path):
        return values

    with open(path, "r", encoding="utf-8") as fin:
        for raw_line in fin:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                values[parts[0]] = parts[1]
    return values


def _config_float(values, key, fallback):
    try:
        return float(values.get(key, fallback))
    except (TypeError, ValueError):
        return float(fallback)


def _config_int(values, key, fallback):
    try:
        return int(round(float(values.get(key, fallback))))
    except (TypeError, ValueError):
        return int(fallback)


def _param_bool(name, default):
    value = rospy.get_param("~" + name, default)
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(value)


def _robust_normalize(image, mask, lower_percentile, upper_percentile):
    data = np.asarray(image, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(data)
    if np.count_nonzero(valid) < 2:
        return np.zeros(data.shape, dtype=np.float32)

    values = data[valid]
    low = float(np.percentile(values, lower_percentile))
    high = float(np.percentile(values, upper_percentile))
    if high - low <= 1e-6:
        low = float(np.min(values))
        high = float(np.max(values))
    if high - low <= 1e-6:
        return np.zeros(data.shape, dtype=np.float32)

    out = (data - low) * (255.0 / (high - low))
    out = np.where(np.isfinite(out), out, 0.0)
    return np.clip(out, 0.0, 255.0).astype(np.float32)


def _relative_height_from_radius(dsm_patch, valid_mask, resolution, radius_m):
    patch = np.asarray(dsm_patch, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(patch)
    if not np.any(valid):
        return np.full(patch.shape, np.nan, dtype=np.float32), float("nan"), 0

    row_idx, col_idx = np.indices(patch.shape, dtype=np.float32)
    center_r = patch.shape[0] / 2.0
    center_c = patch.shape[1] / 2.0
    dx = (row_idx - center_r) * float(resolution)
    dy = (col_idx - center_c) * float(resolution)
    radius_sq = float(radius_m) * float(radius_m)
    reference_mask = valid & (dx * dx + dy * dy <= radius_sq)

    reference_count = int(np.count_nonzero(reference_mask))
    if reference_count > 0:
        reference_height = float(np.mean(patch[reference_mask]))
    else:
        center_value = patch[int(center_r), int(center_c)]
        reference_height = (
            float(center_value) if np.isfinite(center_value) else float(np.nanmean(patch[valid]))
        )

    relative = patch - reference_height
    relative = np.where(valid, relative, np.nan).astype(np.float32)
    return relative, reference_height, reference_count


def _median_smooth_valid(image, valid_mask, min_valid_neighbors):
    data = np.asarray(image, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(data)
    rows, cols = data.shape
    if rows == 0 or cols == 0:
        return data.copy(), valid.copy()

    padded_data = np.pad(data, ((1, 1), (1, 1)), mode="constant", constant_values=np.nan)
    padded_valid = np.pad(valid, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    data_windows = np.lib.stride_tricks.sliding_window_view(padded_data, (3, 3))
    valid_windows = np.lib.stride_tricks.sliding_window_view(padded_valid, (3, 3))
    window_values = np.where(valid_windows, data_windows, np.nan)
    valid_count = np.sum(valid_windows, axis=(2, 3))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        smoothed = np.nanmedian(window_values, axis=(2, 3)).astype(np.float32)
    smoothed_valid = valid_count >= int(min_valid_neighbors)
    smoothed = np.where(smoothed_valid, smoothed, np.nan).astype(np.float32)
    return smoothed, smoothed_valid


def _bev_style_gradients(relative_height, valid_mask, resolution, min_valid_neighbors):
    smoothed, smoothed_valid = _median_smooth_valid(
        relative_height, valid_mask, min_valid_neighbors
    )
    if not np.any(smoothed_valid):
        zeros = np.zeros(relative_height.shape, dtype=np.float32)
        return zeros, zeros

    fill_value = float(np.nanmean(smoothed[smoothed_valid]))
    filled = np.where(smoothed_valid, smoothed, fill_value).astype(np.float32)
    scale = 8.0 * max(float(resolution), 1e-6)
    grad_col = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3) / scale
    grad_row = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3) / scale
    longitudinal_gradient = np.where(smoothed_valid, -grad_row, 0.0).astype(np.float32)
    lateral_gradient = np.where(smoothed_valid, -grad_col, 0.0).astype(np.float32)
    return longitudinal_gradient, lateral_gradient


def _colorize_masked(image, mask, lower_percentile, upper_percentile):
    normalized = _robust_normalize(image, mask, lower_percentile, upper_percentile)
    color = _apply_bev_colormap(normalized)
    color[~np.asarray(mask, dtype=bool)] = (0, 0, 0)
    return color


def _stack_dsm_debug_images(dsm_products, lower_percentile, upper_percentile):
    valid = dsm_products["valid_mask"]
    height_color = _colorize_masked(
        dsm_products["relative_height"], valid, lower_percentile, upper_percentile
    )
    longitudinal_color = _colorize_masked(
        dsm_products["longitudinal_gradient"], valid, lower_percentile, upper_percentile
    )
    lateral_color = _colorize_masked(
        dsm_products["lateral_gradient"], valid, lower_percentile, upper_percentile
    )
    return np.hstack([height_color, longitudinal_color, lateral_color])


class PythonLocalizationNode(object):
    def __init__(self):
        self.gridmap_init_path = rospy.get_param(
            "~gridmap_init_path",
            "/home/hsw/catkin_ws/program/GridMap_v7_5_ros1/config/GridMapInit.ini",
        )
        self.gridmap_config = _read_key_value_ini(self.gridmap_init_path)
        if self.gridmap_config:
            rospy.loginfo("Loaded GridMap defaults: %s", self.gridmap_init_path)
        else:
            rospy.logwarn("GridMapInit.ini defaults not found: %s", self.gridmap_init_path)

        default_resolution = _config_float(
            self.gridmap_config, "lidar_bev_resolution", 0.2
        )
        default_map_size_x = _config_float(
            self.gridmap_config, "lidar_bev_map_size_x", 100.0
        )
        default_map_size_y = _config_float(
            self.gridmap_config, "lidar_bev_map_size_y", 100.0
        )

        self.dem_path = rospy.get_param("~dem_path", "/home/hsw/catkin_ws/doc/miluo_dsm.tif")
        self.loop_rate = float(rospy.get_param("~rate", 10.0))
        self.show_window = _param_bool("show_window", True)
        self.show_dsm_layers = _param_bool("show_dsm_layers", True)
        self.terrain_resolution = float(
            rospy.get_param("~terrain_resolution", default_resolution)
        )
        self.patch_width = int(
            rospy.get_param(
                "~patch_width",
                max(1, int(round(default_map_size_y / max(self.terrain_resolution, 1e-6)))),
            )
        )
        self.patch_height = int(
            rospy.get_param(
                "~patch_height",
                max(1, int(round(default_map_size_x / max(self.terrain_resolution, 1e-6)))),
            )
        )
        self.heading_source = rospy.get_param("~heading_source", "global_pose")
        self.heading_convention = rospy.get_param("~heading_convention", "auto")
        self.heading_offset_deg = float(rospy.get_param("~heading_offset_deg", 0.0))
        self.dem_sampling_mode = rospy.get_param("~dem_sampling_mode", "crop_rotate")
        self.dsm_reference_radius_m = float(
            rospy.get_param(
                "~dsm_reference_radius_m",
                _config_float(self.gridmap_config, "lidar_bev_near_inner_radius", 2.0),
            )
        )
        self.dsm_min_valid_neighbors = int(
            rospy.get_param(
                "~dsm_min_valid_neighbors",
                _config_int(self.gridmap_config, "lidar_bev_edge_min_valid_neighbors", 3),
            )
        )
        self.dsm_min_valid_neighbors = max(1, min(9, self.dsm_min_valid_neighbors))
        self.norm_lower_percentile = float(rospy.get_param("~norm_lower_percentile", 2.0))
        self.norm_upper_percentile = float(rospy.get_param("~norm_upper_percentile", 98.0))
        self.print_every_n = max(1, int(rospy.get_param("~print_every_n", 1)))
        self.save_debug_dir = rospy.get_param("~save_debug_dir", "")
        self.save_every_n = int(rospy.get_param("~save_every_n", 0))
        self.visual_window_width = int(rospy.get_param("~visual_window_width", 800))
        self.visual_window_height = int(rospy.get_param("~visual_window_height", 800))
        self.track_window_width = int(rospy.get_param("~track_window_width", 800))
        self.track_window_height = int(rospy.get_param("~track_window_height", 800))
        self.debug_window_width = int(rospy.get_param("~debug_window_width", 1500))
        self.debug_window_height = int(rospy.get_param("~debug_window_height", 500))
        self.draw_local_pose_track = _param_bool("draw_local_pose_track", True)
        self.local_pose_track_mode = rospy.get_param("~local_pose_track_mode", "anchor")

        if self.show_window and os.name != "nt" and not os.environ.get("DISPLAY"):
            rospy.logwarn("DISPLAY is not set; disabling OpenCV windows.")
            self.show_window = False

        if self.save_debug_dir:
            os.makedirs(self.save_debug_dir, exist_ok=True)

        rospy.loginfo("Loading DEM: %s", self.dem_path)
        start = time.time()
        self.dem_data = load_dem_tiff(self.dem_path)
        if not self.dem_data.is_valid:
            raise RuntimeError("failed to load DEM: {}".format(self.dem_path))
        rospy.loginfo("DEM loaded in %.3f s", time.time() - start)

        self.coord_converter = CoordConverter(self.dem_data)
        self.interface = CInterface()
        self.input = InputData()

        self.color_map = _make_dem_color_map(self.dem_data)
        self.vis_track = self.color_map.copy()

        self.latest_global_pose = None
        self.latest_local_pose = None
        self.local_pose_anchor = None
        self.global_track_point = None
        self.local_track_point = None
        self.processed_dsm_count = 0

        if self.show_window:
            _create_resizable_window(
                "Visual Map", self.visual_window_width, self.visual_window_height
            )
            cv2.imshow("Visual Map", self.color_map)

            _create_resizable_window(
                "Track", self.track_window_width, self.track_window_height
            )

            if self.show_dsm_layers:
                _create_resizable_window(
                    "DSM H_rel | DSM G_long | DSM G_lat",
                    self.debug_window_width,
                    self.debug_window_height,
                )
            cv2.waitKey(1)

    def spin(self):
        rate = rospy.Rate(self.loop_rate)
        rospy.loginfo("Python localization node started.")

        while not rospy.is_shutdown():
            self.interface.ConvertToLocalData(self.input)

            if self.input.GlobalPose_refreshflag:
                self._handle_global_pose(self.input.GlobalPose)

            if self.input.LocalPose_refreshflag:
                self._handle_local_pose(self.input.LocalPose)

            if self.input.TerrainMap_refreshflag:
                self._handle_terrain_map(self.input.TerrainMap)

            if self.show_window:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    rospy.signal_shutdown("closed by keyboard")

            rate.sleep()

        if self.show_window:
            cv2.destroyAllWindows()

    def _handle_global_pose(self, msg):
        self.latest_global_pose = msg

        pixel = self.coord_converter.wgs84_to_pixel(msg.longitude, msg.latitude)
        self.global_track_point = _draw_track(
            self.vis_track, pixel, color=(255, 255, 255), previous=self.global_track_point
        )

        if self.show_window:
            cv2.imshow("Track", self.vis_track)

    def _handle_local_pose(self, msg):
        self.latest_local_pose = msg
        if not self.draw_local_pose_track:
            return

        pixel = self._local_pose_to_dem_pixel(msg)
        if pixel is None:
            return

        self.local_track_point = _draw_track(
            self.vis_track, pixel, color=(0, 0, 255), previous=self.local_track_point
        )

        if self.show_window:
            cv2.imshow("Track", self.vis_track)

    def _local_pose_to_dem_pixel(self, msg):
        mode = self.local_pose_track_mode.lower()
        if mode in ("gauss", "direct", "direct_gauss"):
            return self.coord_converter.gauss_to_pixel(float(msg.dr_x), float(msg.dr_y))

        if self.local_pose_anchor is None:
            if self.latest_global_pose is None:
                return None
            anchor_gauss = self.coord_converter.wgs84_to_gauss(
                float(self.latest_global_pose.longitude),
                float(self.latest_global_pose.latitude),
            )
            self.local_pose_anchor = (
                float(msg.dr_x),
                float(msg.dr_y),
                anchor_gauss.x,
                anchor_gauss.y,
            )
            rospy.loginfo(
                (
                    "LocalPose track anchor: local=(%.3f, %.3f), "
                    "gauss=(%.3f, %.3f), mode=%s"
                ),
                self.local_pose_anchor[0],
                self.local_pose_anchor[1],
                self.local_pose_anchor[2],
                self.local_pose_anchor[3],
                self.local_pose_track_mode,
            )

        local_x0, local_y0, gauss_x0, gauss_y0 = self.local_pose_anchor
        gauss_x = gauss_x0 + (float(msg.dr_x) - local_x0)
        gauss_y = gauss_y0 + (float(msg.dr_y) - local_y0)
        return self.coord_converter.gauss_to_pixel(gauss_x, gauss_y)

    def _handle_terrain_map(self, terrain_msg):
        if self.latest_global_pose is None:
            rospy.logwarn_throttle(
                2.0, "Received TerrainMap but no /self_state/GlobalPose yet; skip DSM update."
            )
            return

        center_lon = float(self.latest_global_pose.longitude)
        center_lat = float(self.latest_global_pose.latitude)
        terrain_heading = float(terrain_msg.localPoseStamped.theta)
        global_azimuth = float(self.latest_global_pose.azimuth)
        if self.heading_source == "terrain_local_pose":
            heading = terrain_heading
        else:
            heading = global_azimuth
        heading += self.heading_offset_deg
        heading_convention = self._resolve_heading_convention()

        dsm_products = self._build_dsm_at_pose(
            center_lon, center_lat, heading, heading_convention
        )

        self.processed_dsm_count += 1
        index = self.processed_dsm_count
        valid_count = int(np.count_nonzero(dsm_products["valid_mask"]))

        if index % self.print_every_n == 0:
            pixel = self.coord_converter.wgs84_to_pixel(center_lon, center_lat)
            rospy.loginfo(
                (
                    "DSM patch #%d: valid=%d/%d (%.2f%%), "
                    "center_lon=%.9f, center_lat=%.9f, heading=%.3f(%s/%s), "
                    "terrain_theta=%.3f, global_azimuth=%.3f, sampling=%s, "
                    "dem_pixel=(%.2f, %.2f), dsm_ref=%.3fm/%dpx"
                ),
                index,
                valid_count,
                dsm_products["valid_mask"].size,
                100.0 * float(valid_count) / max(float(dsm_products["valid_mask"].size), 1.0),
                center_lon,
                center_lat,
                heading,
                self.heading_source,
                heading_convention,
                terrain_heading,
                global_azimuth,
                self.dem_sampling_mode,
                pixel.x,
                pixel.y,
                dsm_products["reference_height"],
                dsm_products["reference_count"],
            )

        dsm_debug_image = None
        if self.show_window or (self.save_debug_dir and self.save_every_n > 0):
            dsm_debug_image = _stack_dsm_debug_images(
                dsm_products,
                self.norm_lower_percentile,
                self.norm_upper_percentile,
            )

        if self.show_window and self.show_dsm_layers and dsm_debug_image is not None:
            cv2.imshow("DSM H_rel | DSM G_long | DSM G_lat", dsm_debug_image)

        if (
            self.save_debug_dir
            and self.save_every_n > 0
            and index % self.save_every_n == 0
            and dsm_debug_image is not None
        ):
            dsm_path = os.path.join(
                self.save_debug_dir, "dsm_height_gradient_{:06d}.png".format(index)
            )
            cv2.imwrite(dsm_path, dsm_debug_image)

    def _resolve_heading_convention(self):
        if self.heading_convention != "auto":
            return self.heading_convention
        return "math"

    def _sample_dsm_patch(self, center_lon, center_lat, heading, heading_convention):
        if self.dem_sampling_mode.lower() in ("crop_rotate", "crop_then_rotate", "opencv"):
            return sample_dem_patch_crop_rotate(
                self.dem_data,
                self.coord_converter,
                center_lon,
                center_lat,
                heading,
                self.patch_height,
                self.patch_width,
                terrain_resolution=self.terrain_resolution,
                heading_convention=heading_convention,
            )
        return sample_dem_patch(
            self.dem_data,
            self.coord_converter,
            center_lon,
            center_lat,
            heading,
            self.patch_height,
            self.patch_width,
            terrain_resolution=self.terrain_resolution,
            heading_convention=heading_convention,
        )

    def _build_dsm_at_pose(self, center_lon, center_lat, heading, heading_convention):
        dem_patch, valid_dem, _map_x, _map_y = self._sample_dsm_patch(
            center_lon, center_lat, heading, heading_convention
        )
        return self._build_dsm_products(dem_patch, valid_dem)

    def _build_dsm_products(self, dem_patch, valid_dem):
        relative_height, reference_height, reference_count = _relative_height_from_radius(
            dem_patch,
            valid_dem,
            self.terrain_resolution,
            self.dsm_reference_radius_m,
        )
        longitudinal_gradient, lateral_gradient = _bev_style_gradients(
            relative_height,
            valid_dem,
            self.terrain_resolution,
            self.dsm_min_valid_neighbors,
        )
        valid = np.asarray(valid_dem, dtype=bool) & np.isfinite(relative_height)
        return {
            "relative_height": relative_height,
            "longitudinal_gradient": longitudinal_gradient,
            "lateral_gradient": lateral_gradient,
            "valid_mask": valid,
            "reference_height": reference_height,
            "reference_count": reference_count,
        }


def main():
    rospy.init_node("localization_python_node")
    node = PythonLocalizationNode()
    node.spin()


if __name__ == "__main__":
    main()
