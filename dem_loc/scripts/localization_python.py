#!/usr/bin/env python3
"""Python port of localization.cpp with TerrainMap roughness DEM ZNCC."""

import os
import sys
import time
import csv

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
from loc_bev.dem_zncc import (
    dem_patch_to_compare_image,
    normalize_for_display,
    sample_dem_patch,
    sample_dem_patch_crop_rotate,
    terrain_array_to_grid,
    zncc,
)


def _make_dem_color_map(dem_data):
    gray_map = normalize_to_uint8(
        dem_data.raw_elevation_map, dem_data.min_height, dem_data.max_height
    )
    return cv2.applyColorMap(gray_map, cv2.COLORMAP_HOT)


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


def _stack_debug_images(roughness_grid, dem_compare, valid_mask):
    rough_u8 = roughness_grid.astype(np.uint8)
    dem_u8 = normalize_for_display(dem_compare)

    valid_u8 = np.where(valid_mask, 255, 0).astype(np.uint8)
    rough_color = cv2.applyColorMap(rough_u8, cv2.COLORMAP_JET)
    dem_color = cv2.applyColorMap(dem_u8, cv2.COLORMAP_JET)
    valid_color = cv2.cvtColor(valid_u8, cv2.COLOR_GRAY2BGR)
    return np.hstack([rough_color, dem_color, valid_color])


def _gradient_magnitude(image):
    data = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(data)
    if finite.any():
        data = np.where(finite, data, float(np.nanmean(data))).astype(np.float32)
    else:
        data = np.zeros(data.shape, dtype=np.float32)
    grad_x = cv2.Sobel(data, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(data, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y)


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


def _edge_iou(image_a, image_b, mask, percentile):
    valid = np.asarray(mask, dtype=bool) & np.isfinite(image_a) & np.isfinite(image_b)
    if np.count_nonzero(valid) < 2:
        return float("nan")

    values_a = image_a[valid]
    values_b = image_b[valid]
    threshold_a = float(np.percentile(values_a, percentile))
    threshold_b = float(np.percentile(values_b, percentile))
    edge_a = valid & (image_a >= threshold_a)
    edge_b = valid & (image_b >= threshold_b)
    union = int(np.count_nonzero(edge_a | edge_b))
    if union == 0:
        return float("nan")
    return float(np.count_nonzero(edge_a & edge_b)) / float(union)


def _finite_or(value, fallback=0.0):
    if np.isfinite(value):
        return float(value)
    return float(fallback)


class PythonLocalizationNode(object):
    def __init__(self):
        self.dem_path = rospy.get_param("~dem_path", "/home/hsw/catkin_ws/doc/miluo_dsm.tif")
        self.loop_rate = float(rospy.get_param("~rate", 10.0))
        self.show_window = bool(rospy.get_param("~show_window", True))
        self.terrain_resolution = float(rospy.get_param("~terrain_resolution", 0.2))
        self.roughness_width = int(rospy.get_param("~roughness_width", 500))
        self.roughness_height = int(rospy.get_param("~roughness_height", 500))
        self.heading_source = rospy.get_param("~heading_source", "global_pose")
        self.heading_convention = rospy.get_param("~heading_convention", "auto")
        self.heading_offset_deg = float(rospy.get_param("~heading_offset_deg", 0.0))
        self.matcher_mode = rospy.get_param("~matcher_mode", "feature")
        self.dem_sampling_mode = rospy.get_param("~dem_sampling_mode", "crop_rotate")
        self.dem_crop_rotate_expand = bool(rospy.get_param("~dem_crop_rotate_expand", True))
        self.dem_compare_mode = rospy.get_param("~dem_compare_mode", "local_roughness")
        self.roughness_scale_m = float(rospy.get_param("~roughness_scale_m", 2.5))
        self.dem_roughness_window_m = float(rospy.get_param("~dem_roughness_window_m", 1.0))
        self.mask_zero_roughness = bool(rospy.get_param("~mask_zero_roughness", True))
        self.roughness_mask_min = float(rospy.get_param("~roughness_mask_min", 5.0))
        self.roughness_mask_max = float(rospy.get_param("~roughness_mask_max", 255.0))
        self.norm_lower_percentile = float(rospy.get_param("~norm_lower_percentile", 2.0))
        self.norm_upper_percentile = float(rospy.get_param("~norm_upper_percentile", 98.0))
        self.norm_zncc_weight = float(rospy.get_param("~norm_zncc_weight", 0.55))
        self.gradient_zncc_weight = float(rospy.get_param("~gradient_zncc_weight", 0.45))
        self.edge_iou_weight = float(rospy.get_param("~edge_iou_weight", 0.0))
        self.edge_percentile = float(rospy.get_param("~edge_percentile", 85.0))
        self.min_valid_ratio = float(rospy.get_param("~min_valid_ratio", 0.02))
        self.score_csv_path = rospy.get_param(
            "~score_csv_path",
            "/home/hsw/catkin_ws/program/dem_loc/match_scores.csv",
        )
        self.score_csv_append = bool(rospy.get_param("~score_csv_append", False))
        self.score_csv_every_n = max(1, int(rospy.get_param("~score_csv_every_n", 1)))
        self.print_every_n = max(1, int(rospy.get_param("~print_every_n", 1)))
        self.save_debug_dir = rospy.get_param("~save_debug_dir", "")
        self.save_every_n = int(rospy.get_param("~save_every_n", 0))
        self.visual_window_width = int(rospy.get_param("~visual_window_width", 800))
        self.visual_window_height = int(rospy.get_param("~visual_window_height", 800))
        self.track_window_width = int(rospy.get_param("~track_window_width", 800))
        self.track_window_height = int(rospy.get_param("~track_window_height", 800))
        self.debug_window_width = int(rospy.get_param("~debug_window_width", 1500))
        self.debug_window_height = int(rospy.get_param("~debug_window_height", 500))
        self.draw_local_pose_track = bool(rospy.get_param("~draw_local_pose_track", True))
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
        self.processed_terrain_count = 0
        self.score_csv_file = None
        self.score_csv_writer = None
        self._init_score_csv()

        if self.show_window:
            _create_resizable_window(
                "Visual Map", self.visual_window_width, self.visual_window_height
            )
            cv2.imshow("Visual Map", self.color_map)

            _create_resizable_window(
                "Track", self.track_window_width, self.track_window_height
            )

            _create_resizable_window(
                "Roughness | DEM | Valid",
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
        if self.score_csv_file is not None:
            self.score_csv_file.close()

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
                2.0, "Received TerrainMap but no /self_state/GlobalPose yet; skip ZNCC."
            )
            return

        roughness = terrain_array_to_grid(
            terrain_msg.roughness,
            width=self.roughness_width,
            height=self.roughness_height,
        )

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

        match = self._match_same_pose(
            roughness, center_lon, center_lat, heading, heading_convention
        )

        dem_compare = match["dem_compare"]
        valid_dem = match["valid_mask"]
        score = match["score"]
        valid_count = match["valid_count"]

        self.processed_terrain_count += 1
        index = self.processed_terrain_count

        if self.score_csv_writer is not None and index % self.score_csv_every_n == 0:
            self._write_score_csv_rows(
                index,
                terrain_msg,
                roughness,
                center_lon,
                center_lat,
                heading,
                heading_convention,
                match,
            )

        if index % self.print_every_n == 0:
            pixel = self.coord_converter.wgs84_to_pixel(center_lon, center_lat)
            rospy.loginfo(
                (
                    "TerrainMap roughness DEM match #%d: mode=%s/%s, score=%.6f "
                    "(raw=%.6f, norm=%.6f, grad=%.6f, edge_iou=%.6f), "
                    "valid=%d/%d, mask=%.2f%%, "
                    "center_lon=%.9f, center_lat=%.9f, heading=%.3f(%s/%s), "
                    "terrain_theta=%.3f, global_azimuth=%.3f, "
                    "sampling=%s, rough_win=%.2fm, norm_p=[%.1f, %.1f], "
                    "dem_pixel=(%.2f, %.2f), roughness[min=%d max=%d mean=%.3f]"
                ),
                index,
                self.matcher_mode,
                self.dem_compare_mode,
                score,
                match["raw_score"],
                match["norm_score"],
                match["gradient_score"],
                match["edge_iou"],
                valid_count,
                roughness.size,
                100.0 * float(valid_count) / max(float(roughness.size), 1.0),
                center_lon,
                center_lat,
                heading,
                self.heading_source,
                heading_convention,
                terrain_heading,
                global_azimuth,
                self.dem_sampling_mode,
                self.dem_roughness_window_m,
                self.norm_lower_percentile,
                self.norm_upper_percentile,
                pixel.x,
                pixel.y,
                int(roughness.min()),
                int(roughness.max()),
                float(roughness.mean()),
            )

        debug_image = None
        if self.show_window or (self.save_debug_dir and self.save_every_n > 0):
            debug_image = _stack_debug_images(
                match["roughness_display"], match["dem_display"], valid_dem
            )

        if self.show_window and debug_image is not None:
            cv2.imshow("Roughness | DEM | Valid", debug_image)

        if (
            self.save_debug_dir
            and self.save_every_n > 0
            and index % self.save_every_n == 0
            and debug_image is not None
        ):
            path = os.path.join(self.save_debug_dir, "roughness_dem_{:06d}.png".format(index))
            cv2.imwrite(path, debug_image)

    def _resolve_heading_convention(self):
        if self.heading_convention != "auto":
            return self.heading_convention
        return "math"

    def _roughness_window_px(self):
        return max(1, int(round(self.dem_roughness_window_m / self.terrain_resolution)))

    def _init_score_csv(self):
        if not self.score_csv_path:
            return
        score_dir = os.path.dirname(os.path.abspath(self.score_csv_path))
        if score_dir:
            os.makedirs(score_dir, exist_ok=True)

        file_exists = (
            self.score_csv_append
            and os.path.exists(self.score_csv_path)
            and os.path.getsize(self.score_csv_path) > 0
        )
        self.score_csv_file = open(
            self.score_csv_path, "a" if self.score_csv_append else "w", newline=""
        )
        self.score_csv_writer = csv.DictWriter(
            self.score_csv_file,
            fieldnames=[
                "frame_index",
                "local_time",
                "utc_time",
                "sample_name",
                "radius_m",
                "direction",
                "dx_vehicle_m",
                "dy_vehicle_m",
                "center_lon",
                "center_lat",
                "sample_lon",
                "sample_lat",
                "heading_deg",
                "heading_convention",
                "dem_sampling_mode",
                "dem_mode",
                "score",
                "raw_score",
                "norm_score",
                "gradient_score",
                "edge_iou",
                "valid_count",
                "valid_ratio",
                "rough_min",
                "rough_max",
                "rough_mean",
                "roughness_scale_m",
                "dem_roughness_window_m",
                "norm_lower_percentile",
                "norm_upper_percentile",
            ],
        )
        if not file_exists:
            self.score_csv_writer.writeheader()
            self.score_csv_file.flush()
        rospy.loginfo("Writing DEM match score CSV: %s", self.score_csv_path)

    def _match_same_pose(self, roughness, center_lon, center_lat, heading, heading_convention):
        if self.dem_sampling_mode.lower() in ("crop_rotate", "crop_then_rotate", "opencv"):
            dem_patch, valid_dem, _map_x, _map_y = sample_dem_patch_crop_rotate(
                self.dem_data,
                self.coord_converter,
                center_lon,
                center_lat,
                heading,
                roughness.shape[0],
                roughness.shape[1],
                terrain_resolution=self.terrain_resolution,
                heading_convention=heading_convention,
                expand_crop=self.dem_crop_rotate_expand,
            )
        else:
            dem_patch, valid_dem, _map_x, _map_y = sample_dem_patch(
                self.dem_data,
                self.coord_converter,
                center_lon,
                center_lat,
                heading,
                roughness.shape[0],
                roughness.shape[1],
                terrain_resolution=self.terrain_resolution,
                heading_convention=heading_convention,
            )
        dem_compare = dem_patch_to_compare_image(
            dem_patch,
            mode=self.dem_compare_mode,
            roughness_scale_m=self.roughness_scale_m,
            roughness_window_px=self._roughness_window_px(),
        )

        match = self._score_feature_arrays(roughness, dem_compare, valid_dem)
        match["dem_compare"] = dem_compare
        match["valid_mask"] = match["mask"]
        match["transform"] = "none"
        return match

    def _write_score_csv_rows(
        self,
        index,
        terrain_msg,
        roughness,
        center_lon,
        center_lat,
        heading,
        heading_convention,
        center_match,
    ):
        rough = np.asarray(roughness, dtype=np.float32)
        rough_min = int(np.nanmin(rough))
        rough_max = int(np.nanmax(rough))
        rough_mean = float(np.nanmean(rough))
        valid_count = int(center_match["valid_count"])

        self.score_csv_writer.writerow(
            {
                "frame_index": index,
                "local_time": float(getattr(terrain_msg, "local_time", 0.0)),
                "utc_time": float(getattr(terrain_msg, "UTC_time", 0.0)),
                "sample_name": "center",
                "radius_m": 0.0,
                "direction": "center",
                "dx_vehicle_m": 0.0,
                "dy_vehicle_m": 0.0,
                "center_lon": center_lon,
                "center_lat": center_lat,
                "sample_lon": center_lon,
                "sample_lat": center_lat,
                "heading_deg": heading,
                "heading_convention": heading_convention,
                "dem_sampling_mode": self.dem_sampling_mode,
                "dem_mode": self.dem_compare_mode,
                "score": center_match["score"],
                "raw_score": center_match["raw_score"],
                "norm_score": center_match["norm_score"],
                "gradient_score": center_match["gradient_score"],
                "edge_iou": center_match["edge_iou"],
                "valid_count": valid_count,
                "valid_ratio": float(valid_count) / max(float(roughness.size), 1.0),
                "rough_min": rough_min,
                "rough_max": rough_max,
                "rough_mean": rough_mean,
                "roughness_scale_m": self.roughness_scale_m,
                "dem_roughness_window_m": self.dem_roughness_window_m,
                "norm_lower_percentile": self.norm_lower_percentile,
                "norm_upper_percentile": self.norm_upper_percentile,
            }
        )
        self.score_csv_file.flush()

    def _build_valid_mask(self, roughness, dem_compare, valid_dem):
        rough = np.asarray(roughness, dtype=np.float32)
        dem = np.asarray(dem_compare, dtype=np.float32)
        base = np.asarray(valid_dem, dtype=bool) & np.isfinite(rough) & np.isfinite(dem)

        mask = base.copy()
        if self.mask_zero_roughness:
            mask &= rough > 0.0
        mask &= rough >= self.roughness_mask_min
        if self.roughness_mask_max < 255.0:
            mask &= rough <= self.roughness_mask_max

        min_count = max(100, int(round(rough.size * self.min_valid_ratio)))
        if np.count_nonzero(mask) >= min_count:
            return mask

        fallback = base.copy()
        if self.mask_zero_roughness:
            fallback &= rough > 0.0
        if np.count_nonzero(fallback) >= min_count:
            return fallback
        return base

    def _score_feature_arrays(self, roughness, dem_compare, valid_dem):
        rough = np.asarray(roughness, dtype=np.float32)
        dem = np.asarray(dem_compare, dtype=np.float32)
        mask = self._build_valid_mask(rough, dem, valid_dem)

        raw_score, raw_count = zncc(rough, dem, mask)
        rough_norm = _robust_normalize(
            rough, mask, self.norm_lower_percentile, self.norm_upper_percentile
        )
        dem_norm = _robust_normalize(
            dem, mask, self.norm_lower_percentile, self.norm_upper_percentile
        )
        norm_score, norm_count = zncc(rough_norm, dem_norm, mask)

        rough_grad = _gradient_magnitude(rough_norm)
        dem_grad = _gradient_magnitude(dem_norm)
        gradient_score, gradient_count = zncc(rough_grad, dem_grad, mask)
        edge_iou = _edge_iou(rough_grad, dem_grad, mask, self.edge_percentile)

        norm_value = _finite_or(norm_score)
        gradient_value = _finite_or(gradient_score)
        edge_value = _finite_or(edge_iou, 0.0) * 2.0 - 1.0
        weight_sum = max(
            self.norm_zncc_weight + self.gradient_zncc_weight + self.edge_iou_weight,
            1e-6,
        )
        score = (
            self.norm_zncc_weight * norm_value
            + self.gradient_zncc_weight * gradient_value
            + self.edge_iou_weight * edge_value
        ) / weight_sum

        return {
            "score": float(score),
            "raw_score": raw_score,
            "norm_score": norm_score,
            "gradient_score": gradient_score,
            "edge_iou": edge_iou,
            "valid_count": int(max(raw_count, norm_count, gradient_count)),
            "mask": mask,
            "roughness_display": rough_norm,
            "dem_display": dem_norm,
        }

def main():
    rospy.init_node("localization_python_node")
    node = PythonLocalizationNode()
    node.spin()


if __name__ == "__main__":
    main()
