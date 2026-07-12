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
    ).strip().split("\n")[1]


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
    """多行帧日志：首行 Global/Local/PF 分数，随后 PF 状态，帧间用分界线分隔。"""
    sep = "=" * 76

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

    lines = [
        sep,
        "[#{fid}]  Global={g}  Local={l}  PF_best={p}".format(
            fid=int(frame_id),
            g=_fmt_score(global_score),
            l=_fmt_score(local_score),
            p=_fmt_score(pf_best_score),
        ),
    ]

    if verbose:
        if global_score_detail is not None:
            gs = global_score_detail
            lines.append(
                "  Global detail: J={jt:.4f}  n={n}  ng={ng}".format(
                    jt=float(gs.get("J_total", float("nan"))),
                    n=int(gs.get("valid_pixel_num", 0)),
                    ng=int(gs.get("valid_grad_pixel_num", gs.get("valid_pixel_num", 0))),
                )
            )
        if local_score_detail is not None:
            ls = local_score_detail
            lines.append(
                "  Local  detail: J={jt:.4f}  n={n}  ng={ng}".format(
                    jt=float(ls.get("J_total", float("nan"))),
                    n=int(ls.get("valid_pixel_num", 0)),
                    ng=int(ls.get("valid_grad_pixel_num", ls.get("valid_pixel_num", 0))),
                )
            )

    if motion is not None:
        speed_part = ""
        if "speed_mps" in motion:
            speed_part = "  speed={:.3f} m/s  scale={:.2f}".format(
                float(motion.get("speed_mps", 0.0)),
                float(motion.get("speed_noise_scale", 1.0)),
            )
        skip_part = "  stationary_skip=1" if motion.get("stationary_skip", False) else ""
        lines.append(
            "  odom  local=({dx:.3f}, {dy:.3f}) m  dyaw={dyaw:.4f} rad  "
            "noise_std=({sx:.4f}, {sy:.4f}, {st:.4f}){speed}{skip}".format(
                dx=float(motion.get("delta_local_x", 0.0)),
                dy=float(motion.get("delta_local_y", 0.0)),
                dyaw=float(motion.get("delta_yaw", 0.0)),
                sx=float(motion.get("motion_std_lx", 0.0)),
                sy=float(motion.get("motion_std_ly", 0.0)),
                st=float(motion.get("motion_std_yaw", 0.0)),
                speed=speed_part,
                skip=skip_part,
            )
        )

    lines.extend(
        [
            "-" * 76,
            "  mode={mode}  resampled={rs}".format(
                mode=str(estimate.get("estimate_mode", "?")),
                rs=int(bool(estimate.get("resampled", False))),
            ),
            "  est   gauss=({x:.2f}, {y:.2f})  yaw={yaw:.3f} rad".format(
                x=float(estimate["x_est"]),
                y=float(estimate["y_est"]),
                yaw=float(estimate["yaw_est"]),
            ),
            "  best  gauss=({bx:.2f}, {by:.2f})  yaw={byaw:.3f} rad  score={bs:.4f}".format(
                bx=float(estimate["best_x"]),
                by=float(estimate["best_y"]),
                byaw=float(estimate.get("best_yaw", float("nan"))),
                bs=float(estimate.get("best_score", estimate.get("max_score", float("nan")))),
            ),
            "  elite_k={ek}  neff={ne:.1f}  valid_scores={vc}".format(
                ek=int(estimate.get("elite_count", 0)),
                ne=float(estimate["neff"]),
                vc=int(estimate.get("valid_score_count", 0)),
            ),
            "  spread  x={sx:.2f} m  y={sy:.2f} m  yaw={syaw:.3f} rad".format(
                sx=float(estimate.get("spread_x", 0.0)),
                sy=float(estimate.get("spread_y", 0.0)),
                syaw=float(estimate.get("spread_yaw", 0.0)),
            ),
            "  weights  min={wmin:.2e}  max={wmax:.2e}".format(
                wmin=float(estimate.get("weight_min", 0.0)),
                wmax=float(estimate.get("weight_max", 0.0)),
            ),
            "  weight_math  mode={wm}  j_scale={js:.3e}  j_spread={jsp:.3e}  log_range={lr:.2f}".format(
                wm=str(estimate.get("weight_mode", "")),
                js=float(estimate.get("weight_j_scale", float("nan"))),
                jsp=float(estimate.get("weight_j_spread", float("nan"))),
                lr=float(estimate.get("weight_log_range", float("nan"))),
            ),
            "  J_total  [{jmin:.4f}, {jmax:.4f}]  mean={jmean:.4f}".format(
                jmin=float(estimate.get("J_min", float("nan"))),
                jmax=float(estimate.get("J_max", float("nan"))),
                jmean=float(estimate.get("J_mean", float("nan"))),
            ),
            "  score    [{mn:.4f}, {mx:.4f}]  mean={ms:.4f}".format(
                mn=float(estimate["min_score"]),
                mx=float(estimate["max_score"]),
                ms=float(estimate.get("mean_score", float("nan"))),
            ),
            sep,
            "",
        ]
    )
    return "\n".join(lines)


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
