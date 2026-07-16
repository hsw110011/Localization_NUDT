#!/usr/bin/env python3
"""粒子滤波 ROS 集成 — 复用 localization_python 里程计递推与 BEV 解析。"""

import copy
import math

from self_state.msg import GlobalPose as GlobalPoseMsg

from .coord_converter import _heading_to_math_deg
from .cinterface import parse_lidar_bev_grid_map
from .dsm_bev_score_runner import global_pose_gauss_theta
from .particle_filter import ParticleFilter


def compute_odom_delta_local(local_pose, prev_local_x, prev_local_y):
    """与 localization_python._update_odom_track 相同的局部位移增量。"""
    delta_local_x = float(local_pose.dr_x) - float(prev_local_x)
    delta_local_y = float(local_pose.dr_y) - float(prev_local_y)
    return delta_local_x, delta_local_y


def extract_local_speed_mps(local_pose):
    """优先使用 LocalPose.vehicle_speed，缺失时用 speed_x/speed_y 合成。"""
    try:
        speed = float(getattr(local_pose, "vehicle_speed"))
        if math.isfinite(speed):
            return abs(speed)
    except (AttributeError, TypeError, ValueError):
        pass

    try:
        speed_x = float(getattr(local_pose, "speed_x", 0.0))
        speed_y = float(getattr(local_pose, "speed_y", 0.0))
        speed = math.hypot(speed_x, speed_y)
        if math.isfinite(speed):
            return abs(speed)
    except (TypeError, ValueError):
        pass
    return 0.0


def compute_local_heading_delta_rad(
    local_pose,
    prev_heading_rad,
    local_heading_unit="deg",
    local_heading_convention="math",
):
    """LocalPose.dr_heading 增量（math 弧度）。"""
    heading_deg = _heading_to_math_deg(
        local_pose.dr_heading,
        local_heading_unit,
        local_heading_convention,
    )
    heading_rad = math.radians(float(heading_deg))
    if prev_heading_rad is None:
        return 0.0, heading_rad
    delta = float(heading_rad) - float(prev_heading_rad)
    delta = math.atan2(math.sin(delta), math.cos(delta))
    return delta, heading_rad


def gauss_pose_to_global_pose_msg(
    coord_converter,
    gauss_x,
    gauss_y,
    yaw_rad,
    template_msg=None,
    heading_unit="deg",
):
    """Gauss + math yaw -> GlobalPose 消息（兼容现有绘图链路）。"""
    msg = GlobalPoseMsg() if template_msg is None else copy.copy(template_msg)
    blh = coord_converter.gauss_to_wgs84(float(gauss_x), float(gauss_y))
    msg.longitude = float(blh.Lon)
    msg.latitude = float(blh.Lat)
    msg.gaussX = float(gauss_x)
    msg.gaussY = float(gauss_y)
    if heading_unit == "rad":
        msg.azimuth = float(yaw_rad)
    else:
        msg.azimuth = float(math.degrees(float(yaw_rad)))
    return msg


def format_pf_log_line(frame_id, estimate):
    """单行 PF 摘要（兼容旧调用）。"""
    return format_pf_frame_log(
        frame_id,
        global_score=None,
        local_score=None,
        pf_best_score=float(estimate.get("max_score", float("nan"))),
        estimate=estimate,
        verbose=False,
    ).strip()


def format_pf_frame_log(
    frame_id,
    global_score,
    local_score,
    pf_best_score,
    estimate,
    verbose=False,
    global_score_detail=None,
    local_score_detail=None,
    motion=None,
):
    """单行打印 PF 最优粒子的三个直接评分项。"""

    def _fmt_score(value):
        if value is None:
            return "n/a"
        try:
            v = float(value)
            if not math.isfinite(v):
                return "n/a"
            return "{:.4f}".format(v)
        except (TypeError, ValueError):
            return "n/a"

    return (
        "PF particle score #{fid}: J_h={jh} J_gx={jgx} J_gy={jgy} "
        "J={jt} score={sc}"
    ).format(
        fid=int(frame_id),
        jh=_fmt_score(estimate.get("best_J_h", float("nan"))),
        jgx=_fmt_score(estimate.get("best_J_gx", float("nan"))),
        jgy=_fmt_score(estimate.get("best_J_gy", float("nan"))),
        jt=_fmt_score(estimate.get("best_J_total", estimate.get("best_score", float("nan")))),
        sc=_fmt_score(estimate.get("best_score_value", estimate.get("max_score", float("nan")))),
    )


class ParticleFilterRunner(object):
    """封装 PF 初始化、传播、观测更新与 GlobalPose 发布。"""

    def __init__(
        self,
        pf_config,
        score_config,
        coord_converter,
        dsm_cropper,
        global_heading_unit="deg",
        global_heading_convention="math",
        local_heading_unit="deg",
        local_heading_convention="math",
    ):
        self.pf_config = pf_config
        self.score_config = score_config
        self.coord_converter = coord_converter
        self.dsm_cropper = dsm_cropper
        self.global_heading_unit = global_heading_unit
        self.global_heading_convention = global_heading_convention
        self.local_heading_unit = local_heading_unit
        self.local_heading_convention = local_heading_convention

        self.pf = ParticleFilter(pf_config, score_config, dsm_cropper)
        self.initialized = False
        self.pending_propagation = False
        self.odom_theta = 0.0
        self.prev_local_x = 0.0
        self.prev_local_y = 0.0
        self.prev_local_heading_rad = None
        self.last_delta_local = (0.0, 0.0)
        self.last_delta_yaw = 0.0
        self.update_count = 0
        self.last_lidar_layers = None
        self.last_speed_mps = 0.0
        self.stationary = False

    def initialize_from_global_pose(self, global_pose):
        gauss_x, gauss_y, theta = global_pose_gauss_theta(
            self.coord_converter,
            global_pose,
            heading_unit=self.global_heading_unit,
            heading_convention=self.global_heading_convention,
        )
        self.pf.initialize(gauss_x, gauss_y, theta)
        self.initialized = True
        self.pending_propagation = False
        self.prev_local_heading_rad = None
        return gauss_x, gauss_y, theta

    def set_odom_anchor(self, odom_theta, prev_local_x, prev_local_y):
        self.odom_theta = float(odom_theta)
        self.prev_local_x = float(prev_local_x)
        self.prev_local_y = float(prev_local_y)

    def on_local_pose(self, local_pose):
        if not self.initialized:
            return False
        delta_local_x, delta_local_y = compute_odom_delta_local(
            local_pose, self.prev_local_x, self.prev_local_y
        )
        delta_yaw, heading_rad = compute_local_heading_delta_rad(
            local_pose,
            self.prev_local_heading_rad,
            local_heading_unit=self.local_heading_unit,
            local_heading_convention=self.local_heading_convention,
        )
        speed_mps = extract_local_speed_mps(local_pose)
        self.last_speed_mps = speed_mps
        self.last_delta_local = (delta_local_x, delta_local_y)
        self.last_delta_yaw = float(delta_yaw)
        stationary = speed_mps <= float(self.pf_config.stationary_speed_threshold)
        self.stationary = bool(stationary)
        if stationary:
            self.prev_local_x = float(local_pose.dr_x)
            self.prev_local_y = float(local_pose.dr_y)
            self.prev_local_heading_rad = heading_rad
            self.pending_propagation = False
            self.pf.last_motion = {
                "delta_local_x": delta_local_x,
                "delta_local_y": delta_local_y,
                "delta_yaw": delta_yaw,
                "delta_global_x": 0.0,
                "delta_global_y": 0.0,
                "motion_std_lx": 0.0,
                "motion_std_ly": 0.0,
                "motion_std_yaw": 0.0,
                "speed_mps": speed_mps,
                "speed_noise_scale": 1.0,
                "stationary_skip": True,
            }
            return False

        self.pf.propagate(
            delta_local_x,
            delta_local_y,
            self.odom_theta,
            delta_yaw=delta_yaw,
            speed_mps=speed_mps,
        )
        self.prev_local_x = float(local_pose.dr_x)
        self.prev_local_y = float(local_pose.dr_y)
        self.prev_local_heading_rad = heading_rad
        self.pending_propagation = True
        return True

    def on_bev_message(
        self,
        bev_msg,
        lidar_layers=None,
        h_max=None,
        timestamp=None,
        template_global_pose=None,
    ):
        if not self.initialized:
            return None

        if self.stationary:
            self.update_count += 1
            self.pending_propagation = False
            return None

        if self.update_count % self.pf_config.pf_update_every_n != 0 and self.update_count > 0:
            self.update_count += 1
            return None

        info = bev_msg.info
        self.dsm_cropper.set_geometry(
            float(info.length_x), float(info.length_y), float(info.resolution)
        )
        if lidar_layers is None:
            lidar_layers = parse_lidar_bev_grid_map(bev_msg)
        self.last_lidar_layers = lidar_layers

        estimate = self.pf.observe_and_resample(lidar_layers, timestamp=timestamp, h_max=h_max)
        self.update_count += 1
        self.pending_propagation = False

        pf_msg = gauss_pose_to_global_pose_msg(
            self.coord_converter,
            estimate["x_est"],
            estimate["y_est"],
            estimate["yaw_est"],
            template_msg=template_global_pose,
            heading_unit="deg",
        )
        estimate["pf_msg"] = pf_msg
        estimate["lidar_layers"] = lidar_layers
        if self.pf.last_best_vis is not None:
            estimate["pf_best"] = dict(self.pf.last_best_vis)
        return estimate
