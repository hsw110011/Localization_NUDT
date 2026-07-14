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

4. DSM-BEV Score（GlobalPose / LocalPose）:
   3 行：LiDAR BEV / Global / Local；每行内 H_rel | G_long | G_lat 横排。

用法:
    python3 localization_python.py
    # 关闭分数对比: _enable_dsm_bev_score:=false
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
from loc_tool.coord_converter import CoordConverter
from loc_tool.dem_tool import load_dem_tiff, normalize_to_uint8
from loc_tool.cinterface import parse_lidar_bev_grid_map
from loc_tool.dsm_bev_score import DsmBevScoreConfig, build_obs_mask, compute_h_max
from loc_tool.dsm_bev_score_runner import (
    format_dual_score_logs,
    format_perturbation_score_logs,
    global_pose_gauss_theta,
    local_pose_gauss_theta,
    score_at_pose,
    score_global_and_local,
    score_perturbation_grid,
)
from loc_tool.dsm_bev_score_vis import (
    build_dual_patch_view,
    build_quad_patch_view,
)
from loc_tool.dsm_patch import DsmPatchCropper, bev_grid_shape
from loc_tool.particle_filter import load_particle_filter_config, load_particle_filter_ini_defaults
from loc_tool.particle_filter_runner import ParticleFilterRunner, format_pf_frame_log

from self_state.msg import GlobalPose as GlobalPoseMsg


DEM_PATH = "/home/hsw/catkin_ws/doc/miluo_dsm.tif"
DEM_MAP_RESOLUTION_M = 0.2

WINDOW_DSM_TRACK = "DSM Track"
WINDOW_DSM_BEV_SCORE = "DSM-BEV 4x4"
WINDOW_PF_WEIGHT = "PF Particle Weights"

DEFAULT_BEV_MAP_SIZE_X = 64.0
DEFAULT_BEV_MAP_SIZE_Y = 64.0
DEFAULT_BEV_RESOLUTION = 0.2

COLOR_GLOBAL = (255, 255, 255)
COLOR_LOCAL = (0, 255, 255)
COLOR_ODOM = (0, 0, 255)
COLOR_PF = (0, 255, 0)


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
    flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    if hasattr(cv2, "WINDOW_GUI_EXPANDED"):
        flags |= cv2.WINDOW_GUI_EXPANDED
    elif hasattr(cv2, "WINDOW_GUI_NORMAL"):
        flags |= cv2.WINDOW_GUI_NORMAL
    cv2.namedWindow(name, flags)
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
        h_rel_deadzone_half=node_self.dsm_h_rel_deadzone_half,
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
        self.loop_rate = float(rospy.get_param("~rate", 10.0))
        self.show_window = _param_bool("show_window", True)
        self.track_window_width = int(rospy.get_param("~track_window_width", 900))
        self.track_window_height = int(rospy.get_param("~track_window_height", 900))
        self.draw_global_pose_track = _param_bool("draw_global_pose_track", True)
        self.draw_local_pose_track = _param_bool("draw_local_pose_track", True)
        self.draw_odom_track = _param_bool("draw_odom_track", True)
        self.log_every_n = max(1, int(rospy.get_param("~log_every_n", 10)))
        self.score_verbose = _param_bool("score_verbose", False)
        self.score_verbose_below = float(rospy.get_param("~score_verbose_below", 0.05))
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
        self.enable_perturbation_score_debug = _param_bool("enable_perturbation_score_debug", False)
        self.perturbation_score_every_n = max(
            1, int(rospy.get_param("~perturbation_score_every_n", 10))
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
        self.dsm_h_rel_deadzone_half = float(
            rospy.get_param("~dsm_h_rel_deadzone_half", 0.20)
        )
        self.dsm_grad_deadzone_half = float(
            rospy.get_param("~dsm_grad_deadzone_half", 0.15)
        )

        self.pf_config = load_particle_filter_config(rospy.get_param)
        self.enable_particle_filter = bool(self.pf_config.enable)
        self.pf_device = str(self.pf_config.device)
        self.draw_pf_track = _param_bool("draw_pf_track", self.enable_particle_filter)
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

        self.vis_track = _make_dem_color_map(self.dem_data)
        _draw_track_legend(
            self.vis_track,
            self.draw_global_pose_track,
            self.draw_local_pose_track,
            self.draw_odom_track,
            draw_pf=self.draw_pf_track,
        )

        self.latest_global_pose = None
        self.latest_local_pose = None
        self.latest_bev_msg = None
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
        self.dsm_bev_score_count = 0
        self._last_global_ref_score = None
        self._last_bev_vis_cache = None
        self._last_bev_vis_hover_meta = None
        self._last_heartbeat_time = time.time()

        if self.show_window:
            _create_resizable_window(
                WINDOW_DSM_TRACK, self.track_window_width, self.track_window_height
            )
            _imshow_named(WINDOW_DSM_TRACK, self.vis_track)
            if self.show_window and (self.enable_dsm_bev_score or self.enable_particle_filter) and self.show_dsm_bev_score_window:
                rows, cols = bev_grid_shape(
                    self.bev_map_size_x, self.bev_map_size_y, self.bev_resolution
                )
                _create_resizable_window(
                    WINDOW_DSM_BEV_SCORE, cols * 4, (rows + 28) * 4
                )
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
                "DSM-BEV score: 4x4 grid %dx%d px/layer (H_rel|G_long|G_lat|M_L), alpha=(%.2f, %.2f, %.2f)",
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
                self.latest_bev_msg = self.input.LidarBevGridMap
                self._sync_bev_geometry_from_msg(self.input.LidarBevGridMap)
                updated = True

            if self.enable_dsm_bev_score or self.enable_particle_filter:
                self._try_update_bev_frame()

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
                _create_resizable_window(WINDOW_DSM_BEV_SCORE, cols * 4, (rows + 28) * 4)
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

    def _set_bev_vis_cache(self, image, hover_meta=None):
        self._last_bev_vis_cache = image
        self._last_bev_vis_hover_meta = hover_meta

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
        need_score = self.enable_dsm_bev_score
        need_pf = self.enable_particle_filter and self.pf_runner is not None
        if not (need_score or need_pf):
            return
        if (
            self.dsm_patch_cropper is None
            or not self.first_frame_aligned
            or self.latest_bev_msg is None
        ):
            return
        if need_score and (
            self.local_pose_base is None
            or self.latest_global_pose is None
            or self.latest_local_pose is None
        ):
            return
        if need_pf and not self.pf_runner.initialized:
            return
        if not self.input.LidarBevGridMap_refreshflag:
            return

        try:
            lidar_layers, h_max = self._parse_bev_layers()
        except (KeyError, ValueError) as exc:
            rospy.logwarn_throttle(5.0, "BEV parse skipped: %s", exc)
            return

        pf_estimate = None
        global_ref = None
        want_score_view = (
            (self.show_window and self.show_dsm_bev_score_window)
            or self.save_dsm_bev_score_debug
        )
        if need_pf:
            timestamp = None
            if self.latest_global_pose is not None:
                timestamp = getattr(self.latest_global_pose, "local_time", None)
            try:
                pf_estimate = self.pf_runner.on_bev_message(
                    self.latest_bev_msg,
                    lidar_layers=lidar_layers,
                    h_max=h_max,
                    timestamp=timestamp,
                    template_global_pose=self.latest_global_pose,
                )
            except (KeyError, ValueError, RuntimeError) as exc:
                rospy.logwarn_throttle(5.0, "Particle filter update skipped: %s", exc)
                pf_estimate = None

            if pf_estimate is not None:
                self._show_pf_weight_vis(pf_estimate)
                self._write_pose_csv(pf_estimate, timestamp=timestamp)
                next_pf_update_count = self.pf_update_count + 1
                need_pf_log = next_pf_update_count % self.pf_log_every_n == 0
                need_global_ref = (
                    self.latest_global_pose is not None
                    and (
                        need_pf_log
                        or (need_score and want_score_view)
                    )
                )
                if need_global_ref:
                    try:
                        g_gauss_x, g_gauss_y, g_theta = global_pose_gauss_theta(
                            self.coord_converter,
                            self.latest_global_pose,
                            heading_unit=self.global_heading_unit,
                            heading_convention=self.global_heading_convention,
                        )
                        global_score, global_dsm = score_at_pose(
                            self.dsm_patch_cropper,
                            g_gauss_x,
                            g_gauss_y,
                            g_theta,
                            lidar_layers,
                            self.score_config,
                            h_max=h_max,
                            return_dsm_result=want_score_view,
                        )
                        global_ref = (global_score, global_dsm)
                        self._last_global_ref_score = float(global_score["score"])
                    except (KeyError, ValueError):
                        global_ref = None

                self.pf_update_count = next_pf_update_count
                pf_msg = pf_estimate.get("pf_msg")
                if pf_msg is not None and self.pf_pose_pub is not None:
                    self.pf_pose_pub.publish(pf_msg)

                if need_pf_log:
                    g_sc = (
                        float(self._last_global_ref_score)
                        if self._last_global_ref_score is not None
                        else None
                    )
                    g_detail = global_ref[0] if global_ref is not None else None
                    l_sc = None
                    l_detail = None
                    if self.latest_local_pose is not None and self.local_pose_base is not None:
                        try:
                            l_gauss_x, l_gauss_y, l_theta = local_pose_gauss_theta(
                                self.coord_converter,
                                self.latest_local_pose,
                                self.local_pose_base,
                                local_heading_unit=self.local_heading_unit,
                                local_heading_convention=self.local_heading_convention,
                            )
                            local_score_dict, _ = score_at_pose(
                                self.dsm_patch_cropper,
                                l_gauss_x,
                                l_gauss_y,
                                l_theta,
                                lidar_layers,
                                self.score_config,
                                h_max=h_max,
                                return_dsm_result=False,
                            )
                            l_sc = float(local_score_dict["score"])
                            l_detail = local_score_dict
                        except (KeyError, ValueError):
                            pass

                    pf_best_sc = float(pf_estimate.get("max_score", float("nan")))
                    pf_best = pf_estimate.get("pf_best")
                    if pf_best is not None:
                        pf_best_sc = float(pf_best.get("score", pf_best_sc))

                    verbose = self.score_verbose
                    if g_sc is not None and g_sc < self.score_verbose_below:
                        verbose = True
                    if l_sc is not None and l_sc < self.score_verbose_below:
                        verbose = True

                    rospy.loginfo(
                        format_pf_frame_log(
                            self.pf_update_count,
                            global_score=g_sc,
                            local_score=l_sc,
                            pf_best_score=pf_best_sc,
                            estimate=pf_estimate,
                            verbose=verbose,
                            global_score_detail=g_detail,
                            local_score_detail=l_detail,
                            motion=getattr(self.pf_runner.pf, "last_motion", None),
                        )
                    )

                if self.draw_pf_track and pf_msg is not None:
                    pixel = self.coord_converter.global_pose_to_dem_pixel(pf_msg)
                    if pixel is not None:
                        self.pf_track_point = _draw_track(
                            self.vis_track,
                            pixel,
                            color=COLOR_PF,
                            previous=self.pf_track_point,
                            line_width=2,
                            radius=4,
                        )
                        self._refresh_track_view()

        run_score_vis = need_score and (
            self.dsm_bev_score_count % self.dsm_bev_score_every_n == 0
            or self.dsm_bev_score_count == 0
        )
        use_pf_quad = (
            need_score
            and want_score_view
            and pf_estimate is not None
            and self.latest_global_pose is not None
        )

        if not run_score_vis and not use_pf_quad:
            return

        try:
            if use_pf_quad:
                if global_ref is None or global_ref[1] is None:
                    g_gauss_x, g_gauss_y, g_theta = global_pose_gauss_theta(
                        self.coord_converter,
                        self.latest_global_pose,
                        heading_unit=self.global_heading_unit,
                        heading_convention=self.global_heading_convention,
                    )
                    global_score, global_dsm = score_at_pose(
                        self.dsm_patch_cropper,
                        g_gauss_x,
                        g_gauss_y,
                        g_theta,
                        lidar_layers,
                        self.score_config,
                        h_max=h_max,
                        return_dsm_result=True,
                    )
                    global_ref = (global_score, global_dsm)
                    self._last_global_ref_score = float(global_score["score"])
                else:
                    global_score, global_dsm = global_ref
                pf_output_pose = {
                    "gauss_x": float(pf_estimate["x_est"]),
                    "gauss_y": float(pf_estimate["y_est"]),
                    "yaw": float(pf_estimate["yaw_est"]),
                }
                pf_output_score, pf_output_dsm = score_at_pose(
                    self.dsm_patch_cropper,
                    pf_output_pose["gauss_x"],
                    pf_output_pose["gauss_y"],
                    pf_output_pose["yaw"],
                    lidar_layers,
                    self.score_config,
                    h_max=h_max,
                    return_dsm_result=True,
                )
                vis, hover_meta = build_quad_patch_view(
                    lidar_layers,
                    global_dsm["layers"],
                    global_score["score"],
                    pf_output_dsm["layers"],
                    pf_output_score["score"],
                    pf_best_pose=pf_output_pose,
                    score_config=self.score_config,
                    pf_label="PF output",
                    return_meta=True,
                )
                self._set_bev_vis_cache(vis, hover_meta)
                if need_score:
                    self.dsm_bev_score_count += 1

                if self.save_dsm_bev_score_debug and (
                    self.dsm_bev_score_count % self.dsm_bev_score_debug_every_n == 0
                ):
                    os.makedirs(self.dsm_bev_score_debug_dir, exist_ok=True)
                    out_path = os.path.join(
                        self.dsm_bev_score_debug_dir,
                        "score_{:06d}.png".format(self.dsm_bev_score_count),
                    )
                    try:
                        cv2.imwrite(out_path, vis)
                    except cv2.error as exc:
                        rospy.logwarn_throttle(
                            5.0, "Failed to save DSM-BEV score debug '%s': %s", out_path, exc
                        )
            elif need_score:
                result = score_global_and_local(
                    self.coord_converter,
                    self.dsm_patch_cropper,
                    self.latest_global_pose,
                    self.latest_local_pose,
                    self.local_pose_base,
                    self.latest_bev_msg,
                    score_config=self.score_config,
                    global_heading_unit=self.global_heading_unit,
                    global_heading_convention=self.global_heading_convention,
                    local_heading_unit=self.local_heading_unit,
                    local_heading_convention=self.local_heading_convention,
                    lidar_layers=lidar_layers,
                    h_max=h_max,
                    return_dsm_results=want_score_view,
                )
                self.dsm_bev_score_count += 1
                self._last_global_ref_score = float(result["global"]["score"]["score"])

                if self.dsm_bev_score_count % self.log_every_n == 0:
                    g_score = float(result["global"]["score"]["score"])
                    l_score = float(result["local"]["score"]["score"])
                    verbose = self.score_verbose or max(g_score, l_score) < self.score_verbose_below
                    rospy.loginfo(
                        format_dual_score_logs(
                            result, self.dsm_bev_score_count, verbose=verbose
                        )
                    )

                if (
                    self.enable_perturbation_score_debug
                    and self.dsm_bev_score_count % self.perturbation_score_every_n == 0
                ):
                    try:
                        perturb = score_perturbation_grid(
                            self.dsm_patch_cropper,
                            result["global"]["gauss_x"],
                            result["global"]["gauss_y"],
                            result["global"]["theta"],
                            lidar_layers,
                            score_config=self.score_config,
                            h_max=h_max,
                        )
                        rospy.loginfo(
                            format_perturbation_score_logs(perturb, self.dsm_bev_score_count)
                        )
                    except (KeyError, ValueError) as exc:
                        rospy.logwarn_throttle(
                            5.0, "DSM-BEV perturbation score skipped: %s", exc
                        )

                if want_score_view:
                    vis, hover_meta = build_dual_patch_view(result, return_meta=True)
                    self._set_bev_vis_cache(vis, hover_meta)

                    if self.save_dsm_bev_score_debug and (
                        self.dsm_bev_score_count % self.dsm_bev_score_debug_every_n == 0
                    ):
                        os.makedirs(self.dsm_bev_score_debug_dir, exist_ok=True)
                        out_path = os.path.join(
                            self.dsm_bev_score_debug_dir,
                            "score_{:06d}.png".format(self.dsm_bev_score_count),
                        )
                        try:
                            cv2.imwrite(out_path, vis)
                        except cv2.error as exc:
                            rospy.logwarn_throttle(
                                5.0,
                                "Failed to save DSM-BEV score debug '%s': %s",
                                out_path,
                                exc,
                            )
        except (KeyError, ValueError) as exc:
            rospy.logwarn_throttle(5.0, "DSM-BEV vis skipped: %s", exc)
            return

        if self.show_window and self.show_dsm_bev_score_window and self._last_bev_vis_cache is not None:
            self._show_bev_score_vis()

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

        if self.pf_runner is not None:
            g_x, g_y, _ = self.pf_runner.initialize_from_global_pose(global_pose)
            self.pf_runner.set_odom_anchor(
                self.odom_theta, self.prev_local_x, self.prev_local_y
            )
            rospy.loginfo(
                "Particle filter initialized at GlobalPose (GNSS): gauss=(%.2f, %.2f) "
                "init_std=(%.1f, %.1f, %.3f) rad N=%d",
                g_x,
                g_y,
                self.pf_runner.pf.pf_config.init_std_x,
                self.pf_runner.pf.pf_config.init_std_y,
                self.pf_runner.pf.pf_config.init_std_yaw,
                self.pf_runner.pf.pf_config.num_particles,
            )

        anchor_pixel = cc.global_pose_to_dem_pixel(global_pose)
        rospy.loginfo("First-frame aligned.")
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
            rospy.logwarn_throttle(2.0, "GlobalPose outside DSM.")
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
                rospy.logwarn_throttle(2.0, "Odom anchor outside DSM.")
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
            rospy.logwarn_throttle(2.0, "Odom outside DSM.")
            return

        self.odom_track_point = _draw_track(
            self.vis_track,
            pixel,
            color=COLOR_ODOM,
            previous=self.odom_track_point,
            line_width=2,
            radius=4,
        )
        self._refresh_track_view()

    def _refresh_track_view(self, force_save=False):
        self.track_draw_count += 1
        if self.show_window:
            _imshow_named(WINDOW_DSM_TRACK, self.vis_track)
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
    _load_pf_ini_defaults()
    node = PythonLocalizationNode()
    node.spin()


if __name__ == "__main__":
    main()
