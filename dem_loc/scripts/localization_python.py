#!/usr/bin/env python3
"""DSM-BEV x/y 粒子滤波定位：/lio/odom 位移，LocalPose 航向。

两个窗口：
1. 左侧三行四列：感知 BEV / GPS 中心 DSM / PF 输出中心 DSM；右侧全局轨迹。
   列依次为 H_rel、G_long、G_lat、M_L，保留像素悬停读数。
2. PF 粒子分布与权重，布局保持不变。

用法: python3 localization_python.py
"""

import math
import os
import sys
import time
import csv

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import cv2
import numpy as np
import rospy

from loc_tool.cinterface import CInterface
from loc_tool.common_struct import InputData
from loc_tool.global_track_view import GlobalTrackView
from loc_tool.coord_converter import CoordConverter, PixelPoint
from loc_tool.dem_tool import load_dem_tiff, normalize_to_uint8
from loc_tool.cinterface import parse_lidar_bev_grid_map
from loc_tool.dsm_bev_score import DsmBevScoreConfig, build_obs_mask, compute_h_max
from loc_tool.dsm_bev_score_runner import (
    global_pose_gauss_theta, local_pose_gauss_theta, score_pose_batch,
)
from loc_tool.dsm_bev_score_vis import (
    build_quad_patch_view,
)
from loc_tool.dsm_patch import DsmPatchCropper, bev_grid_shape
from loc_tool.particle_filter import load_particle_filter_config, load_particle_filter_ini_defaults
from loc_tool.particle_filter_runner import ParticleFilterRunner, format_pf_frame_log, odometry_xy_yaw

from self_state.msg import GlobalPose as GlobalPoseMsg


#DEM_PATH = "/home/hsw/catkin_ws/doc/baotou_dem.tif"
#DEM_PATH = "/home/hsw/catkin_ws/doc/miluo_dsm.tif"

DEM_PATH = "/home/hsw/Perception/河西遥感/hexi_dsm.tif"
DEM_MAP_RESOLUTION_M = 0.2

WINDOW_DSM_BEV_SCORE = "BEV / GPS DSM / PF DSM"
WINDOW_PF_WEIGHT = "PF Particle Weights"

DEFAULT_BEV_MAP_SIZE_X = 64.0
DEFAULT_BEV_MAP_SIZE_Y = 64.0
DEFAULT_BEV_RESOLUTION = 0.2

COLOR_GLOBAL = (255, 255, 255)
COLOR_LOCAL = (0, 255, 255)
COLOR_ODOM = (0, 0, 255)
COLOR_PF = (0, 255, 0)


def _make_dem_color_map(dem_data):
    # Keep every loaded DSM pixel; the viewport is resized, not the backing map.
    elevation = dem_data.raw_elevation_map
    gray_map = normalize_to_uint8(elevation, dem_data.min_height, dem_data.max_height)
    colormap = getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET)
    return cv2.applyColorMap(gray_map.astype(np.uint8), colormap)


def _draw_track_legend(image, draw_global, draw_local, draw_odom, draw_pf=False):
    if image is None or image.size == 0:
        return
    items = []
    if draw_global:
        items.append(("GlobalPose", COLOR_GLOBAL))
    if draw_local:
        items.append(("LocalPose", COLOR_LOCAL))
    if draw_odom:
        items.append(("OdomDR", COLOR_ODOM))
    if draw_pf:
        items.append(("PF", COLOR_PF))
    x0, y0, font = 16, 24, cv2.FONT_HERSHEY_SIMPLEX
    for index, (label, color) in enumerate(items):
        y = y0 + index * 24
        cv2.circle(image, (x0, y - 5), 5, color, -1, cv2.LINE_AA)
        cv2.putText(image, label, (x0 + 14, y), font, 0.55, color, 1, cv2.LINE_AA)


def _create_resizable_window(name, width, height):
    flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    if hasattr(cv2, "WINDOW_GUI_EXPANDED"):
        flags |= cv2.WINDOW_GUI_EXPANDED
    elif hasattr(cv2, "WINDOW_GUI_NORMAL"):
        flags |= cv2.WINDOW_GUI_NORMAL
    cv2.namedWindow(name, flags)
    cv2.resizeWindow(name, int(width), int(height))


def _imshow_named(name, image):
    cv2.imshow(name, image)


def _format_hover_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(number):
        return "nan"
    return "{:.4f}".format(number)


def _draw_hover_overlay(base_image, lines, x, y):
    image = base_image.copy()
    h, w = image.shape[:2]
    x = int(np.clip(int(x), 0, max(0, w - 1)))
    y = int(np.clip(int(y), 0, max(0, h - 1)))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    padding = 6
    line_gap = 4
    text_sizes = [cv2.getTextSize(str(line), font, font_scale, thickness)[0] for line in lines]
    box_w = max(size[0] for size in text_sizes) + padding * 2
    box_h = sum(size[1] for size in text_sizes) + line_gap * (len(lines) - 1) + padding * 2
    box_x = x + 12
    if box_x + box_w >= w:
        box_x = x - box_w - 12
    box_y = y + 12
    if box_y + box_h >= h:
        box_y = y - box_h - 12
    box_x = int(np.clip(box_x, 0, max(0, w - box_w - 1)))
    box_y = int(np.clip(box_y, 0, max(0, h - box_h - 1)))

    overlay = image.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.76, image, 0.24, 0.0, image)
    cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), (230, 230, 230), 1)

    text_y = box_y + padding
    for line, size in zip(lines, text_sizes):
        text_y += size[1]
        cv2.putText(
            image,
            str(line),
            (box_x + padding, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        text_y += line_gap

    cv2.drawMarker(
        image,
        (x, y),
        (0, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=12,
        thickness=1,
        line_type=cv2.LINE_AA,
    )
    return image


def _cli_sets_param(name):
    prefix = "_{}:".format(name)
    return any(arg.startswith(prefix) for arg in sys.argv)


def _ensure_track_draw_defaults():
    """三条轨迹默认开启；忽略 roscore 里残留的旧 false，除非命令行显式传参。"""
    for name in (
        "draw_global_pose_track",
        "draw_local_pose_track",
        "draw_odom_track",
        "draw_pf_track",
    ):
        if not _cli_sets_param(name):
            rospy.set_param("~" + name, True)


def _load_pf_ini_defaults():
    ini_path = os.path.join(_PKG_ROOT, "config", "particle_filter.ini")
    defaults = load_particle_filter_ini_defaults(ini_path)
    for key, value in defaults.items():
        param_name = "~" + key
        if not rospy.has_param(param_name):
            if value.lower() in ("true", "false"):
                rospy.set_param(param_name, value.lower() == "true")
            else:
                try:
                    if "." in value or "e" in value.lower():
                        rospy.set_param(param_name, float(value))
                    else:
                        rospy.set_param(param_name, int(value))
                except ValueError:
                    rospy.set_param(param_name, value)


def _make_dsm_patch_cropper(node_self):
    return DsmPatchCropper(
        node_self.dem_data,
        node_self.coord_converter,
        map_size_x=node_self.bev_map_size_x,
        map_size_y=node_self.bev_map_size_y,
        resolution=node_self.bev_resolution,
        edge_min_valid_neighbors=node_self.bev_edge_min_valid_neighbors,
        grad_deadzone_half=node_self.dsm_grad_deadzone_half,
        device=node_self.pf_device if node_self.enable_particle_filter else None,
    )


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
        self.loop_rate = float(rospy.get_param("~rate", 100.0))
        self.show_window = _param_bool("show_window", True)
        self.draw_global_pose_track = _param_bool("draw_global_pose_track", True)
        self.draw_local_pose_track = _param_bool("draw_local_pose_track", True)
        self.draw_odom_track = _param_bool("draw_odom_track", True)
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
        track_enabled = self.show_window or self.save_track_image
        self.draw_global_pose_track &= track_enabled
        self.draw_local_pose_track &= track_enabled
        self.draw_odom_track &= track_enabled
        self.track_image_path = rospy.get_param("~track_image_path", default_track_image_path)
        self.save_track_every_n = max(1, int(rospy.get_param("~save_track_every_n", 20)))
        self.sync_bev_from_topic = _param_bool("sync_bev_from_topic", True)
        self.bev_map_size_x = float(rospy.get_param("~bev_map_size_x", DEFAULT_BEV_MAP_SIZE_X))
        self.bev_map_size_y = float(rospy.get_param("~bev_map_size_y", DEFAULT_BEV_MAP_SIZE_Y))
        self.bev_resolution = float(rospy.get_param("~bev_resolution", DEFAULT_BEV_RESOLUTION))
        self.bev_edge_min_valid_neighbors = max(
            1, int(rospy.get_param("~bev_edge_min_valid_neighbors", 5))
        )
        self.enable_dsm_bev_score = _param_bool("enable_dsm_bev_score", True)
        self.show_dsm_bev_score_window = _param_bool("show_dsm_bev_score_window", True)
        self.show_pf_weight_window = _param_bool("show_pf_weight_window", True)
        self.dsm_bev_score_every_n = max(1, int(rospy.get_param("~dsm_bev_score_every_n", 1)))
        default_score_debug_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "output", "dsm_bev_score_debug")
        )
        self.save_dsm_bev_score_debug = _param_bool("save_dsm_bev_score_debug", False)
        self.dsm_bev_score_debug_dir = rospy.get_param("~dsm_bev_score_debug_dir", default_score_debug_dir)
        self.dsm_bev_score_debug_every_n = max(
            1, int(rospy.get_param("~dsm_bev_score_debug_every_n", 10))
        )
        self.score_config = DsmBevScoreConfig(
            alpha_h=float(rospy.get_param("~alpha_h", 0.65)),
            alpha_gx=float(rospy.get_param("~alpha_gx", 0.25)),
            alpha_gy=float(rospy.get_param("~alpha_gy", 0.10)),
            lambda_lidar_higher=float(rospy.get_param("~lambda_lidar_higher", 1.5)),
            lambda_dsm_higher=float(rospy.get_param("~lambda_dsm_higher", 1.5)),
            delta_h=float(rospy.get_param("~delta_h", 0.30)),
            w_h_base=float(rospy.get_param("~w_h_base", 0.2)),
            w_h_height=float(rospy.get_param("~w_h_height", 0.8)),
            delta_g=float(rospy.get_param("~delta_g", 0.30)),
            w_g_base=float(rospy.get_param("~w_g_base", 0.05)),
            tau=float(rospy.get_param("~tau", 0.10)),
            grad_cap=float(rospy.get_param("~grad_cap", 3.0)),
            grad_mask_erode_px=int(rospy.get_param("~grad_mask_erode_px", 1)),
            dsm_long_sign=float(rospy.get_param("~dsm_long_sign", 1.0)),
            dsm_lat_sign=float(rospy.get_param("~dsm_lat_sign", 1.0)),
        )
        self.dsm_grad_deadzone_half = float(
            rospy.get_param("~dsm_grad_deadzone_half", 0.15)
        )

        self.pf_config = load_particle_filter_config(rospy.get_param)
        self.enable_particle_filter = bool(self.pf_config.enable)
        self.pf_device = str(self.pf_config.device)
        self.draw_pf_track = track_enabled and _param_bool("draw_pf_track", self.enable_particle_filter)
        self.pf_log_every_n = max(1, int(self.pf_config.pf_log_every_n))
        default_pf_debug_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", self.pf_config.debug_dir)
        )
        pf_debug_dir = rospy.get_param("~pf_debug_dir", default_pf_debug_dir)
        if not os.path.isabs(pf_debug_dir):
            pf_debug_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", pf_debug_dir)
            )
        self.pf_config.debug_dir = pf_debug_dir
        default_pose_csv_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "output", "pose_compare.csv")
        )
        self.save_pose_csv = _param_bool("save_pose_csv", True)
        self.pose_csv_path = rospy.get_param("~pose_csv_path", default_pose_csv_path)
        if not os.path.isabs(self.pose_csv_path):
            self.pose_csv_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", self.pose_csv_path)
            )
        self.pose_csv_file = None
        self.pose_csv_writer = None
        self._init_pose_csv()

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
        if self.enable_dsm_bev_score or self.enable_particle_filter:
            self.dsm_patch_cropper = _make_dsm_patch_cropper(self)
            dsm_tensor_device = str(getattr(self.dsm_patch_cropper, "device", "unknown"))
            dsm_tensor_mb = (
                float(getattr(self.dsm_patch_cropper, "dem_tensor_bytes", 0))
                / (1024.0 * 1024.0)
            )
            rospy.loginfo(
                "DSM tensor device: %s, size=%.1f MB",
                dsm_tensor_device,
                dsm_tensor_mb,
            )
            if dsm_tensor_device.startswith("cuda"):
                rospy.loginfo(
                    "DSM GPU load: YES, full DSM tensor is resident on %s.",
                    dsm_tensor_device,
                )
            else:
                rospy.logwarn(
                    "DSM GPU load: NO, full DSM tensor is on %s.",
                    dsm_tensor_device,
                )

        self.pf_runner = None
        self.pf_pose_pub = None
        self.pf_track_point = None
        self.pf_update_count = 0
        if self.enable_particle_filter:
            if self.dsm_patch_cropper is None:
                raise RuntimeError("particle filter requires DSM patch cropper")
            self.pf_runner = ParticleFilterRunner(
                self.pf_config,
                self.score_config,
                self.coord_converter,
                self.dsm_patch_cropper,
                global_heading_unit=self.global_heading_unit,
                global_heading_convention=self.global_heading_convention,
                local_heading_unit=self.local_heading_unit,
                local_heading_convention=self.local_heading_convention,
            )
            self.pf_pose_pub = rospy.Publisher(
                self.pf_config.pf_pose_topic,
                GlobalPoseMsg,
                queue_size=10,
            )

        self.vis_track = None
        self.track_view = None
        self._main_vis_canvas = None
        self._track_rect = None
        self._track_drag = None
        self._track_view_dirty = False
        if track_enabled:
            self.vis_track = _make_dem_color_map(self.dem_data)
            self.track_view = GlobalTrackView(self.vis_track)

        self.latest_global_pose = None
        self.latest_local_pose = None
        self.latest_lio_odom = None
        self.latest_bev_msg = None
        self.local_pose_base = None
        self.first_frame_aligned = False

        self.odom_gauss_x = 0.0
        self.odom_gauss_y = 0.0
        self.odom_theta = 0.0
        self.prev_odom_x = 0.0
        self.prev_odom_y = 0.0
        self.odom_anchor_drawn = False
        self.odom_step_count = 0

        self.global_track_point = None
        self.local_track_point = None
        self.odom_track_point = None
        self.global_pose_count = 0
        self.local_pose_count = 0
        self.track_draw_count = 0
        self.dsm_bev_score_count = 0
        self._last_pf_estimate = None
        self._last_bev_vis_cache = None
        self._last_bev_vis_hover_meta = None
        self._last_heartbeat_time = time.time()

        if self.show_window:
            if self.show_window and (self.enable_dsm_bev_score or self.enable_particle_filter) and self.show_dsm_bev_score_window:
                rows, cols = bev_grid_shape(
                    self.bev_map_size_x, self.bev_map_size_y, self.bev_resolution
                )
                _create_resizable_window(
                    WINDOW_DSM_BEV_SCORE, cols * 4 + self._track_sidebar_width(), (rows + 28) * 3
                )
                waiting = np.zeros(((rows + 28) * 3, cols * 4, 3), dtype=np.uint8)
                cv2.putText(waiting, "Waiting for BEV / pose data", (16, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1, cv2.LINE_AA)
                self._set_bev_vis_cache(waiting, {"panels": []})
                self._show_bev_score_vis()
            if self.enable_particle_filter and self.show_pf_weight_window:
                rows, cols = bev_grid_shape(
                    self.bev_map_size_x, self.bev_map_size_y, self.bev_resolution
                )
                _create_resizable_window(WINDOW_PF_WEIGHT, cols, rows + 26)
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
        if self.enable_dsm_bev_score:
            rows, cols = bev_grid_shape(
                self.bev_map_size_x, self.bev_map_size_y, self.bev_resolution
            )
            rospy.loginfo(
                "DSM-BEV score: 3x4 grid %dx%d px/layer (H_rel|G_long|G_lat|M_L), alpha=(%.2f, %.2f, %.2f)",
                cols,
                rows,
                self.score_config.alpha_h,
                self.score_config.alpha_gx,
                self.score_config.alpha_gy,
            )
        if self.enable_particle_filter:
            rospy.loginfo(
                "Particle filter: N=%d device=%s topic=%s",
                self.pf_config.num_particles,
                self.pf_device,
                self.pf_config.pf_pose_topic,
            )
        rospy.loginfo("Waiting for GlobalPose + LocalPose + /lio/odom to align first frame ...")

    def spin(self):
        rate = rospy.Rate(self.loop_rate)

        while not rospy.is_shutdown():
            self.interface.ConvertToLocalData(self.input)
            updated = False
            if self.input.GlobalPose_refreshflag:
                self.latest_global_pose = self.input.GlobalPose
            if self.input.LocalPose_refreshflag:
                self.latest_local_pose = self.input.LocalPose
            if self.input.LioOdom_refreshflag:
                self.latest_lio_odom = self.input.LioOdom
            self._try_align_first_frame()

            if self.input.GlobalPose_refreshflag:
                self._handle_global_pose(self.input.GlobalPose)
                updated = True

            if self.input.LocalPose_refreshflag:
                self._handle_local_pose(self.input.LocalPose)
                updated = True

            if self.input.LioOdom_refreshflag:
                if self.first_frame_aligned:
                    if self.draw_odom_track or self.save_pose_csv:
                        self._update_odom_track(self.input.LioOdom)
                    if self.pf_runner is not None:
                        self.pf_runner.on_odometry(self.input.LioOdom)
                updated = True

            if self.sync_bev_from_topic and self.input.LidarBevGridMap_refreshflag:
                self.latest_bev_msg = self.input.LidarBevGridMap
                self._sync_bev_geometry_from_msg(self.input.LidarBevGridMap)
                updated = True

            if self.enable_dsm_bev_score or self.enable_particle_filter:
                self._try_update_bev_frame()

            if not updated:
                self._log_waiting_heartbeat()

            if self.show_window:
                if self._track_view_dirty and self._last_bev_vis_cache is not None:
                    self._blit_track_view()
                    self._show_bev_score_vis()
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
            rospy.loginfo("Waiting for GlobalPose + LocalPose + /lio/odom ...")
        else:
            rospy.loginfo(
                "Running: global=%d local=%d odom_steps=%d score=%d pf=%d.",
                self.global_pose_count,
                self.local_pose_count,
                self.odom_step_count,
                self.dsm_bev_score_count,
                self.pf_update_count,
            )

    def _sync_bev_geometry_from_msg(self, msg):
        if self.dsm_patch_cropper is None:
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
            if self.show_window and (self.enable_dsm_bev_score or self.enable_particle_filter) and self.show_dsm_bev_score_window:
                _create_resizable_window(WINDOW_DSM_BEV_SCORE, cols * 4 + self._track_sidebar_width(), (rows + 28) * 3)
            if self.show_window and self.enable_particle_filter and self.show_pf_weight_window:
                _create_resizable_window(WINDOW_PF_WEIGHT, cols, rows + 26)

    def _parse_bev_layers(self):
        """解析 LiDAR BEV 一次，供 PF / 可视化复用。"""
        bev_msg = self.latest_bev_msg
        info = bev_msg.info
        self.dsm_patch_cropper.set_geometry(
            float(info.length_x), float(info.length_y), float(info.resolution)
        )
        lidar_layers = parse_lidar_bev_grid_map(bev_msg)
        m_obs = build_obs_mask(lidar_layers["M_obs"])
        h_max = compute_h_max(lidar_layers["H_L"], m_obs, self.score_config)
        return lidar_layers, h_max

    def _show_pf_weight_vis(self, pf_estimate):
        if not (
            self.show_window
            and self.enable_particle_filter
            and self.show_pf_weight_window
            and pf_estimate is not None
        ):
            return
        image = pf_estimate.get("weight_vis_image")
        out_path = pf_estimate.get("weight_vis_path")
        if image is None and out_path:
            image = cv2.imread(out_path, cv2.IMREAD_COLOR)
        if image is None:
            return
        _imshow_named(WINDOW_PF_WEIGHT, image)

    def _track_sidebar_width(self):
        return 536 if self.track_view is not None else 0

    def _draw_global_track(self, pixel, color, previous=None):
        point = self.track_view.add_point(pixel, color, previous)
        self._track_view_dirty = True
        return point

    def _blit_track_view(self):
        if self._track_rect is None:
            return
        x, y, width, height = self._track_rect
        canvas = self._main_vis_canvas
        canvas[:y, x:x + width] = 0
        for text, row in (
            ("Global DSM tracks   {:.2f}x (1.00x = native pixels)".format(self.track_view.scale), 20),
            ("Wheel: zoom | Left drag: pan | Double-click: fit", 42),
            ("Right-click: native 1:1", 62),
        ):
            cv2.putText(canvas, text, (x + 4, row), cv2.FONT_HERSHEY_SIMPLEX,
                        .43, (255, 255, 255), 1, cv2.LINE_AA)
        items = [("GPS", COLOR_GLOBAL, self.draw_global_pose_track),
                 ("LocalPose", COLOR_LOCAL, self.draw_local_pose_track),
                 ("LIO", COLOR_ODOM, self.draw_odom_track),
                 ("PF", COLOR_PF, self.draw_pf_track)]
        for i, (label, color, enabled) in enumerate(items):
            if enabled:
                cv2.line(canvas, (x + i * 125 + 4, 84), (x + i * 125 + 20, 84), color, 1)
                cv2.putText(canvas, label, (x + i * 125 + 26, 89),
                            cv2.FONT_HERSHEY_SIMPLEX, .43, color, 1, cv2.LINE_AA)
        canvas[y:y + height, x:x + width] = self.track_view.image
        self._track_view_dirty = False

    def _set_bev_vis_cache(self, image, hover_meta=None):
        if self.track_view is not None:
            height, width = image.shape[:2]
            self.track_view.resize(520, max(100, height - 100))
            track_h, track_w = self.track_view.image.shape[:2]
            shape = (max(height, track_h + 100), width + track_w + 16, 3)
            if self._main_vis_canvas is None or self._main_vis_canvas.shape != shape:
                self._main_vis_canvas = np.zeros(shape, dtype=np.uint8)
            self._main_vis_canvas[:height, :width] = image
            self._track_rect = (width + 8, 100, track_w, track_h)
            self._blit_track_view()
            image = self._main_vis_canvas
        self._last_bev_vis_cache = image
        self._last_bev_vis_hover_meta = hover_meta

    def _on_track_mouse(self, event, x, y, flags):
        if self.track_view is None or self._track_rect is None:
            return False
        x0, y0, width, height = self._track_rect
        inside = x0 <= x < x0 + width and y0 <= y < y0 + height
        if event == cv2.EVENT_LBUTTONUP and self._track_drag is not None:
            self._track_drag = None
            return True
        changed = False
        if event == cv2.EVENT_MOUSEMOVE and self._track_drag is not None:
            old_x, old_y = self._track_drag
            self.track_view.pan(x - old_x, y - old_y)
            self._track_drag = (x, y)
            changed = True
        elif not inside:
            return False
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = (flags >> 16) & 0xffff
            if delta >= 0x8000:
                delta -= 0x10000
            self.track_view.zoom_at(x - x0, y - y0, 1.25 ** (delta / 120.))
            changed = True
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self._track_drag = None
            self.track_view.fit()
            changed = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.track_view.zoom_at(x - x0, y - y0)
            changed = True
        elif event == cv2.EVENT_LBUTTONDOWN:
            self._track_drag = (x, y)
        if changed:
            self._blit_track_view()
        if inside or changed:
            cv2.imshow(WINDOW_DSM_BEV_SCORE, self._last_bev_vis_cache)
        return inside or changed

    def _lookup_bev_hover_lines(self, x, y):
        meta = self._last_bev_vis_hover_meta or {}
        for panel in meta.get("panels", []):
            x0 = int(panel.get("x0", 0))
            y0 = int(panel.get("y0", 0))
            x1 = int(panel.get("x1", 0))
            y1 = int(panel.get("y1", 0))
            bar_height = int(panel.get("bar_height", 0))
            if not (x0 <= x < x1 and y0 + bar_height <= y < y1):
                continue

            values = panel.get("values")
            if values is None:
                continue
            arr = np.asarray(values)
            if arr.ndim < 2 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
                continue

            local_x = float(x - x0)
            local_y = float(y - y0 - bar_height)
            image_w = max(1.0, float(panel.get("image_width", x1 - x0)))
            image_h = max(1.0, float(panel.get("image_height", y1 - y0 - bar_height)))
            col = int(np.clip(np.floor(local_x * arr.shape[1] / image_w), 0, arr.shape[1] - 1))
            row = int(np.clip(np.floor(local_y * arr.shape[0] / image_h), 0, arr.shape[0] - 1))
            value = arr[row, col]
            return [
                "{}  row={} col={}".format(panel.get("title", "panel"), row, col),
                "value={}".format(_format_hover_number(value)),
            ]
        return None

    def _on_bev_score_mouse(self, event, x, y, flags, userdata):
        if self._on_track_mouse(event, x, y, flags):
            return
        if event not in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            return
        if self._last_bev_vis_cache is None:
            return

        lines = self._lookup_bev_hover_lines(int(x), int(y))
        if not lines:
            cv2.imshow(WINDOW_DSM_BEV_SCORE, self._last_bev_vis_cache)
            return
        hover_image = _draw_hover_overlay(self._last_bev_vis_cache, lines, x, y)
        cv2.imshow(WINDOW_DSM_BEV_SCORE, hover_image)

    def _show_bev_score_vis(self):
        if self._last_bev_vis_cache is None:
            return
        _imshow_named(WINDOW_DSM_BEV_SCORE, self._last_bev_vis_cache)
        try:
            cv2.setMouseCallback(WINDOW_DSM_BEV_SCORE, self._on_bev_score_mouse)
        except cv2.error as exc:
            rospy.logwarn_throttle(
                5.0, "Failed to install DSM-BEV score mouse callback: %s", exc
            )

    def _init_pose_csv(self):
        if not self.save_pose_csv:
            return
        out_dir = os.path.dirname(self.pose_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        try:
            self.pose_csv_file = open(
                self.pose_csv_path, "w", newline="", encoding="utf-8"
            )
            self.pose_csv_writer = csv.writer(self.pose_csv_file)
            self.pose_csv_writer.writerow(
                [
                    "timestamp",
                    "odom_x",
                    "odom_y",
                    "odom_theta",
                    "loc_x",
                    "loc_y",
                    "loc_theta",
                    "gps_x",
                    "gps_y",
                    "gps_theta",
                ]
            )
            self.pose_csv_file.flush()
            rospy.on_shutdown(self._close_pose_csv)
            rospy.loginfo("Pose comparison CSV: %s", self.pose_csv_path)
        except OSError as exc:
            self.pose_csv_file = None
            self.pose_csv_writer = None
            self.save_pose_csv = False
            rospy.logwarn("Failed to open pose comparison CSV '%s': %s", self.pose_csv_path, exc)

    def _close_pose_csv(self):
        if self.pose_csv_file is None:
            return
        try:
            self.pose_csv_file.flush()
            self.pose_csv_file.close()
        finally:
            self.pose_csv_file = None
            self.pose_csv_writer = None

    @staticmethod
    def _csv_float(value):
        if value is None:
            return ""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return ""
        if not math.isfinite(v):
            return ""
        return "{:.6f}".format(v)

    def _csv_degrees(self, value):
        try:
            return self._csv_float(math.degrees(float(value)))
        except (TypeError, ValueError):
            return ""

    def _write_pose_csv(self, pf_estimate, timestamp=None):
        if self.pose_csv_writer is None or pf_estimate is None:
            return

        stamp = timestamp
        if stamp is None and self.latest_global_pose is not None:
            stamp = getattr(self.latest_global_pose, "local_time", None)
        if stamp is None:
            stamp = rospy.Time.now().to_sec()

        odom_x = self.odom_gauss_x if self.first_frame_aligned else None
        odom_y = self.odom_gauss_y if self.first_frame_aligned else None
        odom_theta = self.odom_theta
        if self.latest_local_pose is not None and self.local_pose_base is not None:
            try:
                _, _, odom_theta = local_pose_gauss_theta(
                    self.coord_converter,
                    self.latest_local_pose,
                    self.local_pose_base,
                    local_heading_unit=self.local_heading_unit,
                    local_heading_convention=self.local_heading_convention,
                )
            except (KeyError, ValueError, AttributeError):
                pass

        gps_x = gps_y = gps_theta = None
        if self.latest_global_pose is not None:
            try:
                gps_x, gps_y, gps_theta = global_pose_gauss_theta(
                    self.coord_converter,
                    self.latest_global_pose,
                    heading_unit=self.global_heading_unit,
                    heading_convention=self.global_heading_convention,
                )
            except (KeyError, ValueError, AttributeError):
                gps_x = gps_y = gps_theta = None

        self.pose_csv_writer.writerow(
            [
                self._csv_float(stamp),
                self._csv_float(odom_x),
                self._csv_float(odom_y),
                self._csv_degrees(odom_theta),
                self._csv_float(pf_estimate.get("x_est")),
                self._csv_float(pf_estimate.get("y_est")),
                self._csv_degrees(pf_estimate.get("yaw_est")),
                self._csv_float(gps_x),
                self._csv_float(gps_y),
                self._csv_degrees(gps_theta),
            ]
        )
        self.pose_csv_file.flush()

    def _try_update_bev_frame(self):
        need_pf = self.enable_particle_filter and self.pf_runner is not None
        if not (self.enable_dsm_bev_score or need_pf):
            return
        if (
            self.dsm_patch_cropper is None
            or not self.first_frame_aligned
            or self.latest_bev_msg is None
            or not self.input.LidarBevGridMap_refreshflag
        ):
            return

        try:
            lidar_layers, h_max = self._parse_bev_layers()
        except (KeyError, ValueError) as exc:
            rospy.logwarn_throttle(5.0, "BEV parse skipped: %s", exc)
            return

        self.dsm_bev_score_count += 1
        timestamp = getattr(self.latest_global_pose, "local_time", None)
        if need_pf:
            try:
                estimate = self.pf_runner.on_bev_message(
                    self.latest_bev_msg, lidar_layers=lidar_layers,
                    h_max=h_max, timestamp=timestamp,
                    template_global_pose=self.latest_global_pose,
                )
            except (KeyError, ValueError, RuntimeError) as exc:
                rospy.logwarn_throttle(5.0, "Particle filter update skipped: %s", exc)
                estimate = None
            if estimate is not None:
                self._last_pf_estimate = estimate
                self.pf_update_count += 1
                # Publish as soon as filtering finishes, before drawing patches.
                pf_msg = estimate["pf_msg"]
                if self.pf_pose_pub is not None:
                    self.pf_pose_pub.publish(pf_msg)
                self._write_pose_csv(estimate, timestamp=timestamp)
                self._show_pf_weight_vis(estimate)
                if self.pf_update_count % self.pf_log_every_n == 0:
                    rospy.loginfo(format_pf_frame_log(
                        self.pf_update_count, global_score=None, local_score=None,
                        pf_best_score=estimate["max_score"], estimate=estimate,
                    ))
                if self.draw_pf_track:
                    pixel = self.coord_converter.global_pose_to_dem_pixel(pf_msg)
                    if pixel is not None:
                        self.pf_track_point = self._draw_global_track(
                            pixel, color=COLOR_PF,
                            previous=self.pf_track_point,
                        )
                        self._refresh_track_view()

        show_view = (
            self.show_window and self.show_dsm_bev_score_window
            and (self.dsm_bev_score_count - 1) % self.dsm_bev_score_every_n == 0
        )
        save_view = (
            self.save_dsm_bev_score_debug
            and self.dsm_bev_score_count % self.dsm_bev_score_debug_every_n == 0
        )
        if not (show_view or save_view) or self.latest_global_pose is None:
            return
        if not need_pf:
            return
        pf_estimate = self._last_pf_estimate
        if pf_estimate is None:
            # At rest the PF skips observations; still display its initialized pose.
            pf_estimate = self.pf_runner.pf.estimate_pose()
        try:
            gps_pose = global_pose_gauss_theta(
                self.coord_converter, self.latest_global_pose,
                heading_unit=self.global_heading_unit,
                heading_convention=self.global_heading_convention,
            )
            pf_pose = (pf_estimate["x_est"], pf_estimate["y_est"], self.pf_runner.pf.heading_rad)
            scores, patches = score_pose_batch(
                self.dsm_patch_cropper, [gps_pose, pf_pose], lidar_layers,
                self.score_config, h_max=h_max, return_dsm_results=True,
            )
            vis, hover_meta = build_quad_patch_view(
                lidar_layers, patches[0]["layers"], scores[0]["score"],
                patches[1]["layers"], scores[1]["score"],
                pf_best_pose={"gauss_x": pf_pose[0], "gauss_y": pf_pose[1]},
                pf_label="PF", return_meta=True,
            )
            self._set_bev_vis_cache(vis, hover_meta)
            if save_view:
                os.makedirs(self.dsm_bev_score_debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(
                    self.dsm_bev_score_debug_dir,
                    "score_{:06d}.png".format(self.dsm_bev_score_count),
                ), vis)
            if show_view:
                self._show_bev_score_vis()
        except (KeyError, ValueError, cv2.error) as exc:
            rospy.logwarn_throttle(5.0, "DSM-BEV vis skipped: %s", exc)

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
        if (self.latest_global_pose is None or self.latest_local_pose is None
                or self.latest_lio_odom is None):
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
        _, _, heading = local_pose_gauss_theta(
            cc, local_pose, self.local_pose_base,
            local_heading_unit=self.local_heading_unit,
            local_heading_convention=self.local_heading_convention,
        )
        odom_x, odom_y, odom_yaw = odometry_xy_yaw(self.latest_lio_odom)
        self.odom_theta = heading - odom_yaw

        anchor_gauss_x, anchor_gauss_y = self._global_pose_gauss(global_pose)
        self.odom_gauss_x = anchor_gauss_x
        self.odom_gauss_y = anchor_gauss_y
        self.prev_odom_x = odom_x
        self.prev_odom_y = odom_y
        self.odom_anchor_drawn = False
        self.odom_step_count = 0
        self.first_frame_aligned = True

        if self.pf_runner is not None:
            g_x, g_y, heading = self.pf_runner.initialize(
                global_pose, local_pose, self.local_pose_base, self.latest_lio_odom,
            )
            rospy.loginfo(
                "XY particle filter initialized: gauss=(%.2f, %.2f) "
                "init_std_xy=(%.1f, %.1f) m N=%d; xy=/lio/odom, yaw=LocalPose (%.2f deg)",
                g_x, g_y,
                self.pf_runner.pf.pf_config.init_std_x,
                self.pf_runner.pf.pf_config.init_std_y,
                self.pf_runner.pf.pf_config.num_particles,
                math.degrees(heading),
            )

        anchor_pixel = cc.global_pose_to_dem_pixel(global_pose)
        rospy.loginfo("First-frame aligned.")
        if self.draw_global_pose_track and anchor_pixel is not None:
            self.global_track_point = self._draw_global_track(
                anchor_pixel,
                color=COLOR_GLOBAL,
                previous=self.global_track_point,
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
            rospy.logwarn_throttle(2.0, "GlobalPose outside DSM.")
            return

        previous = self.global_track_point
        self.global_track_point = self._draw_global_track(
            pixel,
            color=COLOR_GLOBAL,
            previous=previous,
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

        if self.pf_runner is not None:
            self.pf_runner.on_local_pose(msg)

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
            rospy.logwarn_throttle(2.0, "LocalPose outside DSM.")
            return

        self.local_track_point = self._draw_global_track(
            pixel,
            color=COLOR_LOCAL,
            previous=self.local_track_point,
        )
        self._refresh_track_view()

    def _update_odom_track(self, odometry):
        """首帧锚定在 GlobalPose，之后累加 /lio/odom 固定坐标系位移。"""
        cc = self.coord_converter

        delta_local_x = float(odometry.pose.pose.position.x) - self.prev_odom_x
        delta_local_y = float(odometry.pose.pose.position.y) - self.prev_odom_y
        cos_t = math.cos(self.odom_theta)
        sin_t = math.sin(self.odom_theta)
        delta_global_x = delta_local_x * cos_t - delta_local_y * sin_t
        delta_global_y = delta_local_x * sin_t + delta_local_y * cos_t

        self.odom_gauss_x += delta_global_x
        self.odom_gauss_y += delta_global_y
        self.prev_odom_x = float(odometry.pose.pose.position.x)
        self.prev_odom_y = float(odometry.pose.pose.position.y)
        self.odom_step_count += 1

        if not self.odom_anchor_drawn:
            if not self.draw_odom_track:
                self.odom_anchor_drawn = True
                return
            pixel = cc.gauss_to_pixel(self.odom_gauss_x, self.odom_gauss_y)
            if not cc.is_pixel_in_bounds(pixel):
                pixel = cc.gauss_to_dem_pixel(self.odom_gauss_x, self.odom_gauss_y)
            if pixel is None:
                rospy.logwarn_throttle(2.0, "Odom anchor outside DSM.")
                return
            self.odom_track_point = self._draw_global_track(
                pixel,
                color=COLOR_ODOM,
                previous=None,
            )
            self.odom_anchor_drawn = True
            self._refresh_track_view()
            return

        if not self.draw_odom_track:
            return

        pixel = cc.gauss_to_pixel(self.odom_gauss_x, self.odom_gauss_y)
        if not cc.is_pixel_in_bounds(pixel):
            pixel = cc.gauss_to_dem_pixel(self.odom_gauss_x, self.odom_gauss_y)
        if pixel is None:
            rospy.logwarn_throttle(2.0, "Odom outside DSM.")
            return

        self.odom_track_point = self._draw_global_track(
            pixel,
            color=COLOR_ODOM,
            previous=self.odom_track_point,
        )
        self._refresh_track_view()

    def _refresh_track_view(self, force_save=False):
        self.track_draw_count += 1
        if not self.save_track_image:
            return
        if not force_save and self.track_draw_count % self.save_track_every_n != 0:
            return
        track_dir = os.path.dirname(self.track_image_path)
        if track_dir:
            os.makedirs(track_dir, exist_ok=True)
        try:
            image = self.track_view.export_image()
            _draw_track_legend(image, self.draw_global_pose_track, self.draw_local_pose_track,
                               self.draw_odom_track, self.draw_pf_track)
            cv2.imwrite(self.track_image_path, image)
        except cv2.error as exc:
            rospy.logwarn_throttle(
                5.0, "Failed to save track image '%s': %s", self.track_image_path, exc
            )


def main():
    rospy.init_node("localization_python_node")
    rospy.loginfo("Localization script: %s", os.path.realpath(__file__))
    _ensure_track_draw_defaults()
    _load_pf_ini_defaults()
    node = PythonLocalizationNode()
    node.spin()


if __name__ == "__main__":
    main()
