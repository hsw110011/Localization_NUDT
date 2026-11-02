#!/usr/bin/env python3
"""粒子滤波 ROS 集成 — 复用 localization_python 里程计递推与 BEV 解析。"""

import copy
import math

from self_state.msg import GlobalPose as GlobalPoseMsg

from .cinterface import parse_lidar_bev_grid_map
from .dsm_bev_score_runner import global_pose_gauss_theta, local_pose_gauss_theta
from .particle_filter import ParticleFilter


def odometry_xy_yaw(odometry):
    """nav_msgs/Odometry 在固定 LIO 坐标系中的位置与 math 弧度航向。"""
    pose = odometry.pose.pose
    q = pose.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                     1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return float(pose.position.x), float(pose.position.y), yaw


def body_translation_delta(prev_x, prev_y, prev_yaw, x, y):
    """T_prev^-1 * T_now 的平移部分，属于上一帧车体坐标系。"""
    dx, dy = float(x) - float(prev_x), float(y) - float(prev_y)
    c, s = math.cos(float(prev_yaw)), math.sin(float(prev_yaw))
    return c * dx + s * dy, -s * dx + c * dy


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
        self.prev_odom_yaw = 0.0
        self.prev_motion_heading = 0.0
        self.prev_odom_x = 0.0
        self.prev_odom_y = 0.0
        self.last_delta_local = (0.0, 0.0)
        self.update_count = 0
        self.last_lidar_layers = None
        self.last_speed_mps = 0.0
        self.stationary = False

    def initialize(self, global_pose, local_pose, local_pose_base, odometry):
        gauss_x, gauss_y, _ = global_pose_gauss_theta(
            self.coord_converter, global_pose,
            heading_unit=self.global_heading_unit,
            heading_convention=self.global_heading_convention,
        )
        self.local_pose_base = local_pose_base
        _, _, heading = local_pose_gauss_theta(
            self.coord_converter, local_pose, local_pose_base,
            local_heading_unit=self.local_heading_unit,
            local_heading_convention=self.local_heading_convention,
        )
        self.pf.initialize(gauss_x, gauss_y, heading)
        self.prev_odom_x, self.prev_odom_y, self.prev_odom_yaw = odometry_xy_yaw(odometry)
        # This heading belongs to the previous odometry sample, not to the
        # latest LocalPose callback received before the next motion update.
        self.prev_motion_heading = self.pf.heading_rad
        self.initialized = True
        self.pending_propagation = False
        self.on_local_pose(local_pose)
        return gauss_x, gauss_y, heading

    def on_local_pose(self, local_pose):
        if not self.initialized:
            return False
        _, _, heading = local_pose_gauss_theta(
            self.coord_converter, local_pose, self.local_pose_base,
            local_heading_unit=self.local_heading_unit,
            local_heading_convention=self.local_heading_convention,
        )
        # Also update heading while stationary; neither position nor yaw is
        # propagated from LocalPose's dr_x/dr_y or from LIO orientation deltas.
        self.pf.set_heading(heading)
        self.last_speed_mps = extract_local_speed_mps(local_pose)
        self.stationary = self.last_speed_mps <= float(self.pf_config.stationary_speed_threshold)
        return True

    def on_odometry(self, odometry):
        if not self.initialized:
            return False
        x, y, yaw = odometry_xy_yaw(odometry)
        dx, dy = body_translation_delta(self.prev_odom_x, self.prev_odom_y,
                                        self.prev_odom_yaw, x, y)
        previous_heading = self.prev_motion_heading
        self.prev_odom_x, self.prev_odom_y, self.prev_odom_yaw = x, y, yaw
        self.prev_motion_heading = self.pf.heading_rad
        self.last_delta_local = (dx, dy)
        if self.stationary:
            self.pending_propagation = False
            self.pf.last_motion = {
                "delta_local_x": dx, "delta_local_y": dy, "delta_yaw": 0.0,
                "delta_global_x": 0.0, "delta_global_y": 0.0,
                "motion_std_lx": 0.0, "motion_std_ly": 0.0, "motion_std_yaw": 0.0,
                "speed_mps": self.last_speed_mps, "speed_noise_scale": 1.0,
                "stationary_skip": True,
            }
            return False
        self.pf.propagate(dx, dy, previous_heading, speed_mps=self.last_speed_mps)
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
