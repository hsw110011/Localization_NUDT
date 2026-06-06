#!/usr/bin/env python3
"""Python port of localization.cpp with TerrainMap roughness DEM ZNCC."""

import os
import sys
import time

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
    terrain_array_to_grid,
    zncc,
)


def _make_dem_color_map(dem_data):
    gray_map = normalize_to_uint8(
        dem_data.raw_elevation_map, dem_data.min_height, dem_data.max_height
    )
    return cv2.applyColorMap(gray_map, cv2.COLORMAP_HOT)


def _draw_track(vis_track, pixel, color=(255, 255, 255)):
    x = int(round(pixel.x))
    y = int(round(pixel.y))
    if 0 <= x < vis_track.shape[1] and 0 <= y < vis_track.shape[0]:
        cv2.circle(vis_track, (x, y), 5, color, -1)


def _stack_debug_images(roughness_grid, dem_compare, valid_mask):
    rough_u8 = roughness_grid.astype(np.uint8)
    dem_u8 = normalize_for_display(dem_compare)

    valid_u8 = np.where(valid_mask, 255, 0).astype(np.uint8)
    rough_color = cv2.applyColorMap(rough_u8, cv2.COLORMAP_JET)
    dem_color = cv2.applyColorMap(dem_u8, cv2.COLORMAP_JET)
    valid_color = cv2.cvtColor(valid_u8, cv2.COLOR_GRAY2BGR)
    return np.hstack([rough_color, dem_color, valid_color])


class PythonLocalizationNode(object):
    def __init__(self):
        self.dem_path = rospy.get_param("~dem_path", "/home/hsw/catkin_ws/doc/miluo_dsm.tif")
        self.loop_rate = float(rospy.get_param("~rate", 10.0))
        self.show_window = bool(rospy.get_param("~show_window", True))
        self.terrain_resolution = float(rospy.get_param("~terrain_resolution", 0.2))
        self.roughness_width = int(rospy.get_param("~roughness_width", 500))
        self.roughness_height = int(rospy.get_param("~roughness_height", 500))
        self.heading_convention = rospy.get_param("~heading_convention", "math")
        self.dem_compare_mode = rospy.get_param("~dem_compare_mode", "roughness")
        self.roughness_scale_m = float(rospy.get_param("~roughness_scale_m", 2.5))
        self.mask_zero_roughness = bool(rospy.get_param("~mask_zero_roughness", True))
        self.print_every_n = max(1, int(rospy.get_param("~print_every_n", 1)))
        self.save_debug_dir = rospy.get_param("~save_debug_dir", "")
        self.save_every_n = int(rospy.get_param("~save_every_n", 0))

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
        self.processed_terrain_count = 0

        if self.show_window:
            cv2.namedWindow("Visual Map", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Visual Map", 800, 800)
            cv2.imshow("Visual Map", self.color_map)

            cv2.namedWindow("Track", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Track", 800, 800)

            cv2.namedWindow("Roughness | DEM | Valid", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Roughness | DEM | Valid", 1200, 500)
            cv2.waitKey(1)

    def spin(self):
        rate = rospy.Rate(self.loop_rate)
        rospy.loginfo("Python localization node started.")

        while not rospy.is_shutdown():
            self.interface.ConvertToLocalData(self.input)

            if self.input.GlobalPose_refreshflag:
                self._handle_global_pose(self.input.GlobalPose)

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
        _draw_track(self.vis_track, pixel)

        if self.show_window:
            cv2.imshow("Track", self.vis_track)

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
        heading = float(self.latest_global_pose.azimuth)

        dem_patch, valid_dem, _map_x, _map_y = sample_dem_patch(
            self.dem_data,
            self.coord_converter,
            center_lon,
            center_lat,
            heading,
            roughness.shape[0],
            roughness.shape[1],
            terrain_resolution=self.terrain_resolution,
            heading_convention=self.heading_convention,
        )
        dem_compare = dem_patch_to_compare_image(
            dem_patch,
            mode=self.dem_compare_mode,
            roughness_scale_m=self.roughness_scale_m,
        )

        valid_mask = valid_dem
        if self.mask_zero_roughness:
            valid_mask = valid_mask & (roughness > 0)

        score, valid_count = zncc(roughness.astype(np.float32), dem_compare, valid_mask)
        self.processed_terrain_count += 1
        index = self.processed_terrain_count

        if index % self.print_every_n == 0:
            pixel = self.coord_converter.wgs84_to_pixel(center_lon, center_lat)
            rospy.loginfo(
                (
                    "TerrainMap roughness DEM ZNCC #%d: score=%.6f, valid=%d/%d, "
                    "center_lon=%.9f, center_lat=%.9f, heading=%.3f, "
                    "dem_pixel=(%.2f, %.2f), roughness[min=%d max=%d mean=%.3f]"
                ),
                index,
                score,
                valid_count,
                roughness.size,
                center_lon,
                center_lat,
                heading,
                pixel.x,
                pixel.y,
                int(roughness.min()),
                int(roughness.max()),
                float(roughness.mean()),
            )

        debug_image = None
        if self.show_window or (self.save_debug_dir and self.save_every_n > 0):
            debug_image = _stack_debug_images(roughness, dem_compare, valid_mask)

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


def main():
    rospy.init_node("localization_python_node")
    node = PythonLocalizationNode()
    node.spin()


if __name__ == "__main__":
    main()
