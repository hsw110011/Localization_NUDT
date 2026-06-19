#!/usr/bin/env python3
"""DSM 三轨迹可视化。

1. GlobalPose（白）:
   读取 longitude/latitude -> coord_converter.wgs84_to_dem_pixel -> 画在 DSM 上

2. LocalPose（黄）:
   首帧 GetBaseFromLocalPose(local, global)，以 GlobalPose 高斯坐标为全局参考；
   每帧 LocalPoseToGlobal -> gauss -> 经纬度 -> pixel -> 画在 DSM 上

3. OdomDR（红）:
   首帧位置 = 对齐时刻 GlobalPose 的高斯坐标；
   之后每帧将 LocalPose 局部位移增量用 base.theta 转到全局坐标系并累加。

4. DSM Patch（Odom 位置）:
   三列 320×320：H_rel | G_long | G_lat（viridis，无掩码）；梯度同 BEV 算法，GPU 加速。

用法:
    python3 localization_python.py
    # 三条轨迹默认全开；关闭某条: _draw_odom_track:=false
    # 关闭 patch 窗口: _show_dsm_patch_window:=false
"""

import math
import os
import sys
import time

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import cv2
import numpy as np
import rospy

from loc_tool.cinterface import CInterface
from loc_tool.common_struct import InputData
from loc_tool.coord_converter import CoordConverter, _heading_to_math_deg
from loc_tool.dem_tool import load_dem_tiff, normalize_to_uint8
from loc_tool.dsm_patch import DsmPatchCropper, bev_grid_shape


DEM_PATH = "/home/hsw/catkin_ws/doc/miluo_dsm.tif"
DEM_MAP_RESOLUTION_M = 0.2

WINDOW_DSM_TRACK = "DSM Track"
WINDOW_DSM_PATCH = "DSM H_rel | G_long | G_lat"

DEFAULT_BEV_MAP_SIZE_X = 64.0
DEFAULT_BEV_MAP_SIZE_Y = 64.0
DEFAULT_BEV_RESOLUTION = 0.2

COLOR_GLOBAL = (255, 255, 255)
COLOR_LOCAL = (0, 255, 255)
COLOR_ODOM = (0, 0, 255)


def _make_dem_color_map(dem_data):
    gray_map = normalize_to_uint8(
        dem_data.raw_elevation_map, dem_data.min_height, dem_data.max_height
    )
    colormap = getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET)
    return cv2.applyColorMap(gray_map.astype(np.uint8), colormap)


def _draw_track(vis_track, pixel, color=(255, 255, 255), previous=None, line_width=2, radius=4):
    x = int(round(pixel.x))
    y = int(round(pixel.y))
    if 0 <= x < vis_track.shape[1] and 0 <= y < vis_track.shape[0]:
        if previous is not None:
            cv2.line(vis_track, previous, (x, y), color, int(line_width), cv2.LINE_AA)
        cv2.circle(vis_track, (x, y), int(radius), color, -1, cv2.LINE_AA)
        return (x, y)
    return previous


def _draw_track_legend(image, draw_global, draw_local, draw_odom):
    if image is None or image.size == 0:
        return
    items = []
    if draw_global:
        items.append(("GlobalPose", COLOR_GLOBAL))
    if draw_local:
        items.append(("LocalPose", COLOR_LOCAL))
    if draw_odom:
        items.append(("OdomDR", COLOR_ODOM))
    x0, y0, font = 16, 24, cv2.FONT_HERSHEY_SIMPLEX
    for index, (label, color) in enumerate(items):
        y = y0 + index * 24
        cv2.circle(image, (x0, y - 5), 5, color, -1, cv2.LINE_AA)
        cv2.putText(image, label, (x0 + 14, y), font, 0.55, color, 1, cv2.LINE_AA)


def _create_resizable_window(name, width, height):
    flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    if hasattr(cv2, "WINDOW_GUI_NORMAL"):
        flags |= cv2.WINDOW_GUI_NORMAL
    cv2.namedWindow(name, flags)
    cv2.resizeWindow(name, int(width), int(height))


def _cli_sets_param(name):
    prefix = "_{}:".format(name)
    return any(arg.startswith(prefix) for arg in sys.argv)


def _ensure_track_draw_defaults():
    """三条轨迹默认开启；忽略 roscore 里残留的旧 false，除非命令行显式传参。"""
    for name in (
        "draw_global_pose_track",
        "draw_local_pose_track",
        "draw_odom_track",
    ):
        if not _cli_sets_param(name):
            rospy.set_param("~" + name, True)


def _param_bool(name, default):
    param_name = "~" + name
    if not rospy.has_param(param_name):
        return bool(default)
    value = rospy.get_param(param_name)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "default"):
            return bool(default)
        return text not in ("0", "false", "no", "off")
    return bool(value)


class PythonLocalizationNode(object):
    def __init__(self):
        self.dem_path = DEM_PATH
        self.dem_map_resolution = DEM_MAP_RESOLUTION_M
        self.loop_rate = float(rospy.get_param("~rate", 10.0))
        self.show_window = _param_bool("show_window", True)
        self.track_window_width = int(rospy.get_param("~track_window_width", 900))
        self.track_window_height = int(rospy.get_param("~track_window_height", 900))
        self.draw_global_pose_track = _param_bool("draw_global_pose_track", True)
        self.draw_local_pose_track = _param_bool("draw_local_pose_track", True)
        self.draw_odom_track = _param_bool("draw_odom_track", True)
        self.log_every_n = max(1, int(rospy.get_param("~log_every_n", 10)))
        self.heartbeat_interval_sec = float(rospy.get_param("~heartbeat_interval_sec", 3.0))
        self.local_heading_unit = rospy.get_param("~local_heading_unit", "deg")
        self.global_heading_unit = rospy.get_param("~global_heading_unit", "deg")
        self.local_heading_convention = rospy.get_param("~local_heading_convention", "math")
        # GlobalPose.azimuth: math 约定，东向 0°、逆时针（见 degree_set.png）
        self.global_heading_convention = rospy.get_param("~global_heading_convention", "math")
        default_track_image_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "output", "track_on_dsm.png")
        )
        self.save_track_image = _param_bool("save_track_image", False)
        self.track_image_path = rospy.get_param("~track_image_path", default_track_image_path)
        self.save_track_every_n = max(1, int(rospy.get_param("~save_track_every_n", 20)))
        self.enable_dsm_patch = _param_bool("enable_dsm_patch", True)
        self.show_dsm_patch_window = _param_bool("show_dsm_patch_window", True)
        self.sync_bev_from_topic = _param_bool("sync_bev_from_topic", True)
        self.bev_map_size_x = float(rospy.get_param("~bev_map_size_x", DEFAULT_BEV_MAP_SIZE_X))
        self.bev_map_size_y = float(rospy.get_param("~bev_map_size_y", DEFAULT_BEV_MAP_SIZE_Y))
        self.bev_resolution = float(rospy.get_param("~bev_resolution", DEFAULT_BEV_RESOLUTION))
        self.bev_edge_min_valid_neighbors = max(
            1, int(rospy.get_param("~bev_edge_min_valid_neighbors", 5))
        )
        self.dsm_patch_every_n = max(1, int(rospy.get_param("~dsm_patch_every_n", 1)))
        default_patch_image_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "output", "odom_dsm_patch.png")
        )
        self.save_dsm_patch_image = _param_bool("save_dsm_patch_image", False)
        self.dsm_patch_image_path = rospy.get_param("~dsm_patch_image_path", default_patch_image_path)

        if self.show_window and os.name != "nt" and not os.environ.get("DISPLAY"):
            rospy.logwarn("DISPLAY is not set; disabling OpenCV windows.")
            self.show_window = False

        rospy.loginfo("Loading DSM: %s", self.dem_path)
        self.dem_data = load_dem_tiff(self.dem_path, map_resolution_m=self.dem_map_resolution)
        if not self.dem_data.is_valid:
            raise RuntimeError("failed to load DEM: {}".format(self.dem_path))

        self.coord_converter = CoordConverter(self.dem_data)
        self.interface = CInterface()
        self.input = InputData()

        self.dsm_patch_cropper = None
        if self.enable_dsm_patch:
            self.dsm_patch_cropper = DsmPatchCropper(
                self.dem_data,
                self.coord_converter,
                map_size_x=self.bev_map_size_x,
                map_size_y=self.bev_map_size_y,
                resolution=self.bev_resolution,
                edge_min_valid_neighbors=self.bev_edge_min_valid_neighbors,
            )

        self.vis_track = _make_dem_color_map(self.dem_data)
        _draw_track_legend(
            self.vis_track,
            self.draw_global_pose_track,
            self.draw_local_pose_track,
            self.draw_odom_track,
        )

        self.latest_global_pose = None
        self.latest_local_pose = None
        self.local_pose_base = None
        self.first_frame_aligned = False

        self.odom_gauss_x = 0.0
        self.odom_gauss_y = 0.0
        self.odom_theta = 0.0
        self.prev_local_x = 0.0
        self.prev_local_y = 0.0
        self.odom_anchor_drawn = False
        self.odom_step_count = 0

        self.global_track_point = None
        self.local_track_point = None
        self.odom_track_point = None
        self.global_pose_count = 0
        self.local_pose_count = 0
        self.track_draw_count = 0
        self.dsm_patch_count = 0
        self._last_heartbeat_time = time.time()

        if self.show_window:
            _create_resizable_window(
                WINDOW_DSM_TRACK, self.track_window_width, self.track_window_height
            )
            cv2.imshow(WINDOW_DSM_TRACK, self.vis_track)
            if self.enable_dsm_patch and self.show_dsm_patch_window:
                rows, cols = bev_grid_shape(
                    self.bev_map_size_x, self.bev_map_size_y, self.bev_resolution
                )
                _create_resizable_window(WINDOW_DSM_PATCH, cols * 3, rows)
            cv2.waitKey(1)

        rospy.loginfo(
            "Track draw: GlobalPose=%s LocalPose=%s OdomDR=%s",
            self.draw_global_pose_track,
            self.draw_local_pose_track,
            self.draw_odom_track,
        )
        rospy.loginfo(
            "Heading convention: local=%s/%s global=%s/%s (math=East 0 deg CCW)",
            self.local_heading_unit,
            self.local_heading_convention,
            self.global_heading_unit,
            self.global_heading_convention,
        )
        if self.enable_dsm_patch:
            rows, cols = bev_grid_shape(
                self.bev_map_size_x, self.bev_map_size_y, self.bev_resolution
            )
            rospy.loginfo(
                "DSM patch: 3x %dx%d px, %.3f m/px, physical %.1fx%.1f m",
                cols,
                rows,
                self.bev_resolution,
                self.bev_map_size_x,
                self.bev_map_size_y,
            )
        rospy.loginfo("Waiting for GlobalPose + LocalPose to align first frame ...")

    def spin(self):
        rate = rospy.Rate(self.loop_rate)

        while not rospy.is_shutdown():
            self.interface.ConvertToLocalData(self.input)
            updated = False

            if self.input.GlobalPose_refreshflag:
                self.latest_global_pose = self.input.GlobalPose
                self._try_align_first_frame()
                self._handle_global_pose(self.input.GlobalPose)
                updated = True

            if self.input.LocalPose_refreshflag:
                self.latest_local_pose = self.input.LocalPose
                self._try_align_first_frame()
                self._handle_local_pose(self.input.LocalPose)
                updated = True

            if self.sync_bev_from_topic and self.input.LidarBevGridMap_refreshflag:
                self._sync_bev_geometry_from_msg(self.input.LidarBevGridMap)
                updated = True

            if not updated:
                self._log_waiting_heartbeat()

            if self.show_window:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    rospy.signal_shutdown("closed by keyboard")

            rate.sleep()

        if self.show_window:
            cv2.destroyAllWindows()

    def _log_waiting_heartbeat(self):
        now = time.time()
        if now - self._last_heartbeat_time < self.heartbeat_interval_sec:
            return
        self._last_heartbeat_time = now
        if not self.first_frame_aligned:
            rospy.loginfo("Waiting for GlobalPose + LocalPose ...")
        else:
            rospy.loginfo(
                "Running: global=%d local=%d odom_steps=%d dsm_patch=%d.",
                self.global_pose_count,
                self.local_pose_count,
                self.odom_step_count,
                self.dsm_patch_count,
            )

    def _sync_bev_geometry_from_msg(self, msg):
        if not self.enable_dsm_patch or self.dsm_patch_cropper is None:
            return
        try:
            map_size_x, map_size_y, resolution = DsmPatchCropper.geometry_from_grid_map_msg(msg)
        except AttributeError:
            return
        if self.dsm_patch_cropper.set_geometry(map_size_x, map_size_y, resolution):
            self.bev_map_size_x = map_size_x
            self.bev_map_size_y = map_size_y
            self.bev_resolution = resolution
            rows, cols = bev_grid_shape(map_size_x, map_size_y, resolution)
            rospy.loginfo(
                "Synced DSM patch geometry from BEV topic: %dx%d px, %.3f m/px, %.1fx%.1f m",
                cols,
                rows,
                resolution,
                map_size_x,
                map_size_y,
            )
            if self.show_window and self.show_dsm_patch_window:
                _create_resizable_window(WINDOW_DSM_PATCH, cols * 3, rows)

    def _odom_heading_rad(self, local_pose):
        local_heading_deg = _heading_to_math_deg(
            local_pose.dr_heading,
            self.local_heading_unit,
            self.local_heading_convention,
        )
        return math.radians(local_heading_deg) + float(self.odom_theta)

    def _update_dsm_patch(self, local_pose):
        if not self.enable_dsm_patch or self.dsm_patch_cropper is None:
            return
        if self.dsm_patch_count % self.dsm_patch_every_n != 0 and self.dsm_patch_count > 0:
            return

        heading_rad = self._odom_heading_rad(local_pose)
        patch_result = self.dsm_patch_cropper.crop_with_bev_layers(
            self.odom_gauss_x,
            self.odom_gauss_y,
            heading_rad,
        )
        layers = patch_result["layers"]
        self.dsm_patch_count += 1

        if self.dsm_patch_count % self.log_every_n == 0:
            h_rel = layers["H_rel_surf"]
            finite = np.isfinite(h_rel)
            center_h = layers.get("center_height", float("nan"))
            h_min = float(np.min(h_rel[finite])) if finite.any() else float("nan")
            h_max = float(np.max(h_rel[finite])) if finite.any() else float("nan")
            g_long = layers["G_long_L"]
            g_lat = layers["G_lat_L"]
            g_finite = np.isfinite(g_long) & np.isfinite(g_lat)
            rospy.loginfo(
                "DSM patch #%d: odom_gauss=(%.2f, %.2f) heading=%.2f deg "
                "center_h=%.2f H_rel=[%.2f, %.2f] grad_valid=%.1f%%",
                self.dsm_patch_count,
                self.odom_gauss_x,
                self.odom_gauss_y,
                math.degrees(heading_rad),
                center_h,
                h_min,
                h_max,
                float(g_finite.mean()) * 100.0 if g_finite.size else 0.0,
            )

        if self.show_window and self.show_dsm_patch_window:
            vis = patch_result["vis_bgr"]
            rows, cols = vis.shape[0], vis.shape[1] // 3
            for panel in range(3):
                cx = panel * cols + cols // 2
                cv2.circle(vis, (cx, rows // 2), 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.imshow(WINDOW_DSM_PATCH, vis)

        if self.save_dsm_patch_image:
            vis = patch_result["vis_bgr"]
            patch_dir = os.path.dirname(self.dsm_patch_image_path)
            if patch_dir:
                os.makedirs(patch_dir, exist_ok=True)
            try:
                cv2.imwrite(self.dsm_patch_image_path, vis)
            except cv2.error as exc:
                rospy.logwarn_throttle(
                    5.0, "Failed to save DSM patch '%s': %s", self.dsm_patch_image_path, exc
                )

    def _global_pose_gauss(self, global_pose):
        gauss_x, gauss_y, _ = self.coord_converter._extract_global_gauss_heading(
            global_pose,
            heading_unit=self.global_heading_unit,
            heading_convention=self.global_heading_convention,
        )
        return float(gauss_x), float(gauss_y)

    def _try_align_first_frame(self):
        if self.first_frame_aligned:
            return
        if self.latest_global_pose is None or self.latest_local_pose is None:
            return

        local_pose = self.latest_local_pose
        global_pose = self.latest_global_pose
        cc = self.coord_converter

        self.local_pose_base = cc.GetBaseFromLocalPose(
            local_pose,
            global_pose,
            local_heading_unit=self.local_heading_unit,
            local_heading_convention=self.local_heading_convention,
            global_heading_unit=self.global_heading_unit,
            global_heading_convention=self.global_heading_convention,
        )
        self.odom_theta = float(self.local_pose_base.theta)

        anchor_gauss_x, anchor_gauss_y = self._global_pose_gauss(global_pose)
        self.odom_gauss_x = anchor_gauss_x
        self.odom_gauss_y = anchor_gauss_y
        self.prev_local_x = float(local_pose.dr_x)
        self.prev_local_y = float(local_pose.dr_y)
        self.odom_anchor_drawn = False
        self.odom_step_count = 0
        self.first_frame_aligned = True

        global_point = cc.LocalPoseToGlobal(
            local_pose,
            self.local_pose_base,
            local_heading_unit=self.local_heading_unit,
            local_heading_convention=self.local_heading_convention,
        )
        anchor_pixel = cc.global_pose_to_dem_pixel(global_pose)
        rospy.loginfo(
            "First-frame aligned: base=(%.3f, %.3f) theta=%.4f rad | "
            "GlobalPose anchor gauss=(%.3f, %.3f) pixel=(%.1f, %.1f) | "
            "LocalPose dr=(%.3f, %.3f, h=%.2f) -> global gauss=(%.3f, %.3f) ll=(%.9f, %.9f)",
            self.local_pose_base.x,
            self.local_pose_base.y,
            self.odom_theta,
            anchor_gauss_x,
            anchor_gauss_y,
            anchor_pixel.x if anchor_pixel else float("nan"),
            anchor_pixel.y if anchor_pixel else float("nan"),
            local_pose.dr_x,
            local_pose.dr_y,
            local_pose.dr_heading,
            global_point.gauss.x,
            global_point.gauss.y,
            global_point.BLH.Lon,
            global_point.BLH.Lat,
        )
        if self.draw_global_pose_track and anchor_pixel is not None:
            self.global_track_point = _draw_track(
                self.vis_track,
                anchor_pixel,
                color=COLOR_GLOBAL,
                previous=self.global_track_point,
                line_width=2,
                radius=5,
            )
            self._refresh_track_view()

    def _handle_global_pose(self, msg):
        self.global_pose_count += 1
        if not self.draw_global_pose_track:
            return
        self._draw_global_pose_point(msg)

    def _draw_global_pose_point(self, msg):
        pixel = self.coord_converter.global_pose_to_dem_pixel(msg)
        if pixel is None:
            lon = float(getattr(msg, "longitude", float("nan")))
            lat = float(getattr(msg, "latitude", float("nan")))
            gauss_x = float(getattr(msg, "gaussX", float("nan")))
            gauss_y = float(getattr(msg, "gaussY", float("nan")))
            rospy.logwarn_throttle(
                2.0,
                "GlobalPose outside DSM: ll=(%.9f, %.9f) gauss=(%.2f, %.2f)",
                lon,
                lat,
                gauss_x,
                gauss_y,
            )
            return

        previous = self.global_track_point
        self.global_track_point = _draw_track(
            self.vis_track,
            pixel,
            color=COLOR_GLOBAL,
            previous=previous,
            line_width=2,
            radius=5,
        )
        self._refresh_track_view()

    def _handle_local_pose(self, msg):
        self.local_pose_count += 1

        if not self.first_frame_aligned:
            rospy.loginfo_throttle(
                3.0,
                "LocalPose #%d waiting for first-frame alignment.",
                self.local_pose_count,
            )
            return

        # 先画 Odom，再画 LocalPose，避免黄线完全盖住红色 Odom 轨迹
        if self.draw_odom_track:
            self._update_odom_track(msg)

        if self.draw_local_pose_track:
            self._draw_local_pose_track(msg)

    def _draw_local_pose_track(self, local_pose):
        """LocalPose -> WORLD_POINT -> gauss -> 经纬度 -> pixel。"""
        cc = self.coord_converter
        global_point = cc.LocalPoseToGlobal(
            local_pose,
            self.local_pose_base,
            local_heading_unit=self.local_heading_unit,
            local_heading_convention=self.local_heading_convention,
        )
        pixel = cc.gauss_to_dem_pixel(global_point.gauss.x, global_point.gauss.y)
        if pixel is None:
            rospy.logwarn_throttle(
                2.0,
                "LocalPose outside DSM: ll=(%.9f, %.9f) gauss=(%.2f, %.2f)",
                global_point.BLH.Lon,
                global_point.BLH.Lat,
                global_point.gauss.x,
                global_point.gauss.y,
            )
            return

        if self.local_pose_count % self.log_every_n == 0:
            rospy.loginfo(
                "LocalPose #%d: dr=(%.3f, %.3f) -> gauss=(%.3f, %.3f) "
                "ll=(%.9f, %.9f) pixel=(%.1f, %.1f)",
                self.local_pose_count,
                local_pose.dr_x,
                local_pose.dr_y,
                global_point.gauss.x,
                global_point.gauss.y,
                global_point.BLH.Lon,
                global_point.BLH.Lat,
                pixel.x,
                pixel.y,
            )

        self.local_track_point = _draw_track(
            self.vis_track,
            pixel,
            color=COLOR_LOCAL,
            previous=self.local_track_point,
            line_width=2,
            radius=3,
        )
        self._refresh_track_view()

    def _update_odom_track(self, local_pose):
        """首帧锚定在 GlobalPose，之后累加局部位移增量。"""
        cc = self.coord_converter

        if not self.odom_anchor_drawn:
            pixel = cc.gauss_to_pixel(self.odom_gauss_x, self.odom_gauss_y)
            if not cc.is_pixel_in_bounds(pixel):
                pixel = cc.gauss_to_dem_pixel(self.odom_gauss_x, self.odom_gauss_y)
            if pixel is None:
                blh = cc.gauss_to_wgs84(self.odom_gauss_x, self.odom_gauss_y)
                rospy.logwarn_throttle(
                    2.0,
                    "Odom anchor outside DSM: ll=(%.9f, %.9f)",
                    blh.Lon,
                    blh.Lat,
                )
                return
            self.odom_track_point = _draw_track(
                self.vis_track,
                pixel,
                color=COLOR_ODOM,
                previous=None,
                line_width=2,
                radius=5,
            )
            self.odom_anchor_drawn = True
            self._refresh_track_view()
            self._update_dsm_patch(local_pose)
            return

        delta_local_x = float(local_pose.dr_x) - self.prev_local_x
        delta_local_y = float(local_pose.dr_y) - self.prev_local_y
        cos_t = math.cos(self.odom_theta)
        sin_t = math.sin(self.odom_theta)
        delta_global_x = delta_local_x * cos_t - delta_local_y * sin_t
        delta_global_y = delta_local_x * sin_t + delta_local_y * cos_t

        self.odom_gauss_x += delta_global_x
        self.odom_gauss_y += delta_global_y
        self.prev_local_x = float(local_pose.dr_x)
        self.prev_local_y = float(local_pose.dr_y)
        self.odom_step_count += 1

        pixel = cc.gauss_to_pixel(self.odom_gauss_x, self.odom_gauss_y)
        if not cc.is_pixel_in_bounds(pixel):
            pixel = cc.gauss_to_dem_pixel(self.odom_gauss_x, self.odom_gauss_y)
        if pixel is None:
            blh = cc.gauss_to_wgs84(self.odom_gauss_x, self.odom_gauss_y)
            rospy.logwarn_throttle(
                2.0,
                "Odom outside DSM: ll=(%.9f, %.9f) gauss=(%.2f, %.2f)",
                blh.Lon,
                blh.Lat,
                self.odom_gauss_x,
                self.odom_gauss_y,
            )
            return

        if self.odom_step_count % self.log_every_n == 0:
            blh = cc.gauss_to_wgs84(self.odom_gauss_x, self.odom_gauss_y)
            rospy.loginfo(
                "Odom step #%d: d_local=(%.3f, %.3f) -> d_global=(%.3f, %.3f) "
                "accum_gauss=(%.3f, %.3f) ll=(%.9f, %.9f) pixel=(%.1f, %.1f)",
                self.odom_step_count,
                delta_local_x,
                delta_local_y,
                delta_global_x,
                delta_global_y,
                self.odom_gauss_x,
                self.odom_gauss_y,
                blh.Lon,
                blh.Lat,
                pixel.x,
                pixel.y,
            )

        self.odom_track_point = _draw_track(
            self.vis_track,
            pixel,
            color=COLOR_ODOM,
            previous=self.odom_track_point,
            line_width=2,
            radius=4,
        )
        self._refresh_track_view()
        self._update_dsm_patch(local_pose)

    def _refresh_track_view(self, force_save=False):
        self.track_draw_count += 1
        if self.show_window:
            cv2.imshow(WINDOW_DSM_TRACK, self.vis_track)
        if not self.save_track_image:
            return
        if not force_save and self.track_draw_count % self.save_track_every_n != 0:
            return
        track_dir = os.path.dirname(self.track_image_path)
        if track_dir:
            os.makedirs(track_dir, exist_ok=True)
        try:
            cv2.imwrite(self.track_image_path, self.vis_track)
        except cv2.error as exc:
            rospy.logwarn_throttle(
                5.0, "Failed to save track image '%s': %s", self.track_image_path, exc
            )


def main():
    rospy.init_node("localization_python_node")
    _ensure_track_draw_defaults()
    node = PythonLocalizationNode()
    node.spin()


if __name__ == "__main__":
    main()
