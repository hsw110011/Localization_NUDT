#!/usr/bin/env python3
"""Standalone 3-DoF particle filter localization node.

The ParticleFilter class is deliberately independent from DSM/BEV scoring.
Observation modules should provide one scalar cost per particle through
compute_observation_costs().
"""

import configparser
import csv
import math
import os
import threading
from dataclasses import dataclass

if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rospy
from geometry_msgs.msg import PoseArray, PoseStamped
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Path
from self_state.msg import GlobalPose, LocalPose


def wrap_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def yaw_to_quaternion(theta):
    half = 0.5 * float(theta)
    return 0.0, 0.0, math.sin(half), math.cos(half)


def parse_bool(value, fallback=False):
    if value is None:
        return bool(fallback)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")


def angle_to_rad(value, unit="auto"):
    value = float(value)
    unit = str(unit).lower()
    if unit == "rad":
        return value
    if unit == "deg":
        return math.radians(value)
    if abs(value) > (2.0 * math.pi + 0.25):
        return math.radians(value)
    return value


def heading_to_math_rad(value, unit="deg", convention="math"):
    if str(unit).lower() == "rad":
        heading_deg = math.degrees(float(value))
    elif str(unit).lower() == "auto" and abs(float(value)) <= (2.0 * math.pi + 0.25):
        heading_deg = math.degrees(float(value))
    else:
        heading_deg = float(value)

    if str(convention).lower() == "azimuth":
        heading_deg = 90.0 - heading_deg
    return float(wrap_to_pi(math.radians(heading_deg)))


def resolve_path(path, package_root):
    if not path:
        return path
    if os.path.isabs(path):
        return path
    package_name = os.path.basename(package_root.rstrip(os.sep))
    if path == package_name or path.startswith(package_name + os.sep):
        return os.path.abspath(os.path.join(os.path.dirname(package_root), path))
    return os.path.abspath(os.path.join(package_root, path))


class ConfigView(object):
    def __init__(self, path):
        self.path = path
        self.parser = configparser.ConfigParser()
        self.parser.read(path, encoding="utf-8")

    def get(self, section, key, fallback=None):
        param_name = "~" + key
        if rospy.has_param(param_name):
            return rospy.get_param(param_name)
        if self.parser.has_option(section, key):
            return self.parser.get(section, key)
        return fallback

    def get_str(self, section, key, fallback=""):
        return str(self.get(section, key, fallback))

    def get_int(self, section, key, fallback=0):
        return int(round(float(self.get(section, key, fallback))))

    def get_float(self, section, key, fallback=0.0):
        return float(self.get(section, key, fallback))

    def get_bool(self, section, key, fallback=False):
        return parse_bool(self.get(section, key, fallback), fallback)


@dataclass
class ParticleFilterConfig:
    num_particles: int = 512
    init_std_x: float = 5.0
    init_std_y: float = 5.0
    init_std_theta_deg: float = 10.0
    motion_noise_x: float = 0.15
    motion_noise_y: float = 0.10
    motion_noise_theta_deg: float = 1.0
    resample_threshold_ratio: float = 0.5
    min_effective_particles_ratio: float = 0.5
    score_tau: float = 0.20
    max_cost: float = 10.0
    random_seed: int = 0


class ParticleFilter:
    def __init__(self, config):
        self.config = config
        self.num_particles = int(config.num_particles)
        self.rng = np.random.default_rng(
            None if int(config.random_seed) == 0 else int(config.random_seed)
        )
        self.particles = np.zeros((self.num_particles, 3), dtype=np.float64)
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles, dtype=np.float64)
        self.last_costs = np.zeros(self.num_particles, dtype=np.float64)
        self.initialized = False

    def initialize(self, init_x, init_y, init_theta):
        self.particles[:, 0] = self.rng.normal(float(init_x), self.config.init_std_x, self.num_particles)
        self.particles[:, 1] = self.rng.normal(float(init_y), self.config.init_std_y, self.num_particles)
        init_std_theta = math.radians(float(self.config.init_std_theta_deg))
        self.particles[:, 2] = self.rng.normal(float(init_theta), init_std_theta, self.num_particles)
        self.particles[:, 2] = wrap_to_pi(self.particles[:, 2])
        self.weights.fill(1.0 / self.num_particles)
        self.initialized = True

    def predict(self, delta_x_local, delta_y_local, delta_theta):
        if not self.initialized:
            return

        theta = self.particles[:, 2]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        dx = float(delta_x_local)
        dy = float(delta_y_local)
        dtheta = float(delta_theta)

        self.particles[:, 0] += cos_t * dx - sin_t * dy
        self.particles[:, 1] += sin_t * dx + cos_t * dy
        self.particles[:, 2] += dtheta

        self.particles[:, 0] += self.rng.normal(0.0, self.config.motion_noise_x, self.num_particles)
        self.particles[:, 1] += self.rng.normal(0.0, self.config.motion_noise_y, self.num_particles)
        theta_noise = math.radians(float(self.config.motion_noise_theta_deg))
        self.particles[:, 2] += self.rng.normal(0.0, theta_noise, self.num_particles)
        self.particles[:, 2] = wrap_to_pi(self.particles[:, 2])

    def update_weights_from_cost(self, costs):
        if not self.initialized:
            return False

        costs = np.asarray(costs, dtype=np.float64).reshape(-1)
        if costs.shape[0] != self.num_particles:
            raise ValueError("costs must have shape [{}]".format(self.num_particles))

        clean_costs = np.where(np.isfinite(costs), costs, float(self.config.max_cost))
        clean_costs = np.clip(clean_costs, 0.0, float(self.config.max_cost))
        self.last_costs = clean_costs

        cost_min = float(np.min(clean_costs))
        tau = max(float(self.config.score_tau), 1e-9)
        likelihood = np.exp(-(clean_costs - cost_min) / tau)
        likelihood = np.where(np.isfinite(likelihood), likelihood, 0.0)

        if float(np.sum(likelihood)) <= 1e-300:
            self.weights.fill(1.0 / self.num_particles)
            return False

        self.weights *= likelihood
        return self.normalize_weights()

    def normalize_weights(self):
        weight_sum = float(np.sum(self.weights))
        if not np.isfinite(weight_sum) or weight_sum <= 1e-300:
            self.weights.fill(1.0 / self.num_particles)
            return False
        self.weights /= weight_sum
        return True

    def effective_particle_number(self):
        denom = float(np.sum(self.weights ** 2))
        if not np.isfinite(denom) or denom <= 1e-300:
            return 0.0
        return 1.0 / denom

    def resample_if_needed(self):
        threshold = self.num_particles * float(self.config.resample_threshold_ratio)
        if self.effective_particle_number() >= threshold:
            return False
        self._systematic_resample()
        return True

    def _systematic_resample(self):
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        step = 1.0 / self.num_particles
        start = self.rng.uniform(0.0, step)
        positions = start + step * np.arange(self.num_particles)
        indexes = np.searchsorted(cumulative, positions, side="left")
        indexes = np.clip(indexes, 0, self.num_particles - 1)
        self.particles = self.particles[indexes].copy()
        self.weights.fill(1.0 / self.num_particles)

    def estimate(self):
        x_est = float(np.sum(self.weights * self.particles[:, 0]))
        y_est = float(np.sum(self.weights * self.particles[:, 1]))
        sin_sum = float(np.sum(self.weights * np.sin(self.particles[:, 2])))
        cos_sum = float(np.sum(self.weights * np.cos(self.particles[:, 2])))
        theta_est = math.atan2(sin_sum, cos_sum)
        return np.array([x_est, y_est, theta_est], dtype=np.float64)

    def get_particles(self):
        return self.particles.copy(), self.weights.copy()


def compute_observation_costs(particles, bev_msg, dsm_map, config):
    """Placeholder observation model.

    Replace this function with LiDAR relative-height BEV + DSM patch scoring.
    The contract is one finite scalar cost per particle, where lower is better.
    """
    return np.zeros(len(particles), dtype=np.float64)


class TrajectoryRecorder:
    HEADER = (
        "timestamp",
        "pf_x",
        "pf_y",
        "pf_theta",
        "odom_x",
        "odom_y",
        "odom_theta",
        "gps_x",
        "gps_y",
        "gps_theta",
    )

    def __init__(self, path, enabled):
        self.path = path
        self.enabled = bool(enabled)
        if self.enabled:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.exists(path):
                with open(path, "w", newline="", encoding="utf-8") as fout:
                    csv.writer(fout).writerow(self.HEADER)

    def append(self, stamp, pf_state, odom_state, gps_state):
        if not self.enabled:
            return
        row = [
            float(stamp),
            pf_state[0],
            pf_state[1],
            pf_state[2],
            odom_state[0] if odom_state is not None else np.nan,
            odom_state[1] if odom_state is not None else np.nan,
            odom_state[2] if odom_state is not None else np.nan,
            gps_state[0] if gps_state is not None else np.nan,
            gps_state[1] if gps_state is not None else np.nan,
            gps_state[2] if gps_state is not None else np.nan,
        ]
        with open(self.path, "a", newline="", encoding="utf-8") as fout:
            csv.writer(fout).writerow(row)


class ParticleFilterNode:
    def __init__(self):
        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        default_config = os.path.join(package_root, "config", "particle_filter.ini")
        config_path = rospy.get_param("~config_path", default_config)
        self.config = ConfigView(config_path)
        rospy.loginfo("Loaded particle filter config: %s", config_path)

        pf_config = ParticleFilterConfig(
            num_particles=self.config.get_int("particle_filter", "num_particles", 512),
            init_std_x=self.config.get_float("particle_filter", "init_std_x", 5.0),
            init_std_y=self.config.get_float("particle_filter", "init_std_y", 5.0),
            init_std_theta_deg=self.config.get_float("particle_filter", "init_std_theta_deg", 10.0),
            motion_noise_x=self.config.get_float("particle_filter", "motion_noise_x", 0.15),
            motion_noise_y=self.config.get_float("particle_filter", "motion_noise_y", 0.10),
            motion_noise_theta_deg=self.config.get_float("particle_filter", "motion_noise_theta_deg", 1.0),
            resample_threshold_ratio=self.config.get_float("particle_filter", "resample_threshold_ratio", 0.5),
            min_effective_particles_ratio=self.config.get_float(
                "particle_filter", "min_effective_particles_ratio", 0.5
            ),
            score_tau=self.config.get_float("observation", "score_tau", 0.20),
            max_cost=self.config.get_float("observation", "max_cost", 10.0),
            random_seed=self.config.get_int("particle_filter", "random_seed", 0),
        )
        self.pf = ParticleFilter(pf_config)

        self.localpose_topic = self.config.get_str("topics", "localpose_topic", "/localpose")
        self.globalpose_topic = self.config.get_str("topics", "globalpose_topic", "/globalpose")
        self.lidar_bev_topic = self.config.get_str("topics", "lidar_bev_topic", "/lidar_bev/grid_map")
        self.output_frame_id = self.config.get_str("frames", "output_frame_id", "map")

        self.use_globalpose_init = self.config.get_bool("initial_pose", "use_globalpose_init", True)
        self.init_x = self.config.get_float("initial_pose", "init_x", 0.0)
        self.init_y = self.config.get_float("initial_pose", "init_y", 0.0)
        self.init_theta = math.radians(self.config.get_float("initial_pose", "init_theta_deg", 0.0))
        self.global_heading_convention = self.config.get_str(
            "initial_pose", "global_heading_convention", "azimuth"
        )
        self.global_heading_unit = self.config.get_str("initial_pose", "global_heading_unit", "deg")
        self.local_heading_unit = self.config.get_str("initial_pose", "local_heading_unit", "deg")
        self.localpose_delta_mode = self.config.get_str(
            "initial_pose", "localpose_delta_mode", "odom_frame"
        )

        self.publish_particles = self.config.get_bool("output", "publish_particles", True)
        self.publish_path = self.config.get_bool("output", "publish_path", True)
        self.save_debug = self.config.get_bool("output", "save_debug", True)
        self.show_plot = self.config.get_bool("output", "show_plot", False)
        self.plot_update_interval = max(1, self.config.get_int("output", "plot_update_interval", 10))
        self.debug_dir = resolve_path(
            self.config.get_str("output", "debug_dir", "dem_loc/output/debug"), package_root
        )
        trajectory_path = resolve_path(
            self.config.get_str("output", "trajectory_path", "dem_loc/output/pf_trajectory.csv"),
            package_root,
        )
        self.recorder = TrajectoryRecorder(
            trajectory_path, self.config.get_bool("output", "save_trajectory", True)
        )
        if self.save_debug:
            os.makedirs(self.debug_dir, exist_ok=True)

        self.pose_pub = rospy.Publisher(
            self.config.get_str("output", "pf_pose_topic", "/pf_pose"),
            PoseStamped,
            queue_size=1,
        )
        self.particles_pub = rospy.Publisher(
            self.config.get_str("output", "pf_particles_topic", "/pf_particles"),
            PoseArray,
            queue_size=1,
        )
        self.path_pub = rospy.Publisher(
            self.config.get_str("output", "pf_path_topic", "/pf_path"),
            Path,
            queue_size=1,
        )

        self.lock = threading.RLock()
        self.previous_local_state = None
        self.latest_odom_state = None
        self.latest_gps_state = None
        self.pf_history = []
        self.odom_history = []
        self.gps_history = []
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.output_frame_id
        self.update_count = 0
        self.dsm_map = None

        if not self.use_globalpose_init:
            self.pf.initialize(self.init_x, self.init_y, self.init_theta)
            rospy.loginfo(
                "Initialized PF from config: x=%.3f y=%.3f theta=%.3fdeg",
                self.init_x,
                self.init_y,
                math.degrees(self.init_theta),
            )

        self.local_sub = rospy.Subscriber(self.localpose_topic, LocalPose, self._localpose_cb, queue_size=20)
        self.global_sub = rospy.Subscriber(self.globalpose_topic, GlobalPose, self._globalpose_cb, queue_size=20)
        self.bev_sub = rospy.Subscriber(self.lidar_bev_topic, GridMap, self._bev_cb, queue_size=1)
        rospy.loginfo(
            "particle_filter_node subscribed: local=%s global=%s bev=%s",
            self.localpose_topic,
            self.globalpose_topic,
            self.lidar_bev_topic,
        )

    def _localpose_to_state(self, msg):
        return np.array(
            [
                float(msg.dr_x),
                float(msg.dr_y),
                angle_to_rad(float(msg.dr_heading), self.local_heading_unit),
            ],
            dtype=np.float64,
        )

    def _globalpose_to_state(self, msg):
        theta = heading_to_math_rad(
            float(msg.azimuth), self.global_heading_unit, self.global_heading_convention
        )
        return np.array([float(msg.gaussX), float(msg.gaussY), theta], dtype=np.float64)

    def _localpose_delta(self, previous, current):
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        dtheta = float(wrap_to_pi(current[2] - previous[2]))
        if self.localpose_delta_mode.lower() in ("body", "body_frame", "local_delta"):
            return dx, dy, dtheta

        theta = previous[2]
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        delta_x_local = cos_t * dx + sin_t * dy
        delta_y_local = -sin_t * dx + cos_t * dy
        return delta_x_local, delta_y_local, dtheta

    def _globalpose_cb(self, msg):
        with self.lock:
            gps_state = self._globalpose_to_state(msg)
            self.latest_gps_state = gps_state
            self.gps_history.append(gps_state.copy())

            if self.use_globalpose_init and not self.pf.initialized:
                self.pf.initialize(gps_state[0], gps_state[1], gps_state[2])
                rospy.loginfo(
                    "Initialized PF from globalpose: x=%.3f y=%.3f theta=%.3fdeg",
                    gps_state[0],
                    gps_state[1],
                    math.degrees(gps_state[2]),
                )

    def _localpose_cb(self, msg):
        with self.lock:
            current = self._localpose_to_state(msg)
            self.latest_odom_state = current
            self.odom_history.append(current.copy())

            if self.previous_local_state is not None and self.pf.initialized:
                delta = self._localpose_delta(self.previous_local_state, current)
                self.pf.predict(delta[0], delta[1], delta[2])
            self.previous_local_state = current

    def _bev_cb(self, msg):
        with self.lock:
            if not self.pf.initialized:
                rospy.logwarn_throttle(2.0, "Waiting for PF initialization.")
                return

            particles, _weights = self.pf.get_particles()
            costs = compute_observation_costs(particles, msg, self.dsm_map, self.config)
            self.pf.update_weights_from_cost(costs)
            resampled = self.pf.resample_if_needed()
            estimate = self.pf.estimate()
            self.pf_history.append(estimate.copy())
            self.update_count += 1

            stamp = msg.info.header.stamp if msg.info.header.stamp != rospy.Time() else rospy.Time.now()
            self._publish_pose(stamp, estimate)
            if self.publish_particles:
                self._publish_particles(stamp)
            if self.publish_path:
                self._publish_path(stamp, estimate)
            self.recorder.append(
                stamp.to_sec(), estimate, self.latest_odom_state, self.latest_gps_state
            )
            if self.show_plot and self.update_count % self.plot_update_interval == 0:
                self.visualize()

            rospy.loginfo_throttle(
                1.0,
                "PF update: neff=%.1f resampled=%s estimate=(%.3f, %.3f, %.2fdeg)",
                self.pf.effective_particle_number(),
                str(resampled),
                estimate[0],
                estimate[1],
                math.degrees(estimate[2]),
            )

    def _publish_pose(self, stamp, state):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.output_frame_id
        msg.pose.position.x = float(state[0])
        msg.pose.position.y = float(state[1])
        msg.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion(state[2])
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.pose_pub.publish(msg)

    def _publish_particles(self, stamp):
        particles, _weights = self.pf.get_particles()
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.output_frame_id
        for particle in particles:
            pose = PoseStamped().pose
            pose.position.x = float(particle[0])
            pose.position.y = float(particle[1])
            pose.position.z = 0.0
            qx, qy, qz, qw = yaw_to_quaternion(particle[2])
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw
            msg.poses.append(pose)
        self.particles_pub.publish(msg)

    def _publish_path(self, stamp, state):
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.output_frame_id
        pose.pose.position.x = float(state[0])
        pose.pose.position.y = float(state[1])
        pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion(state[2])
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        self.path_msg.header.stamp = stamp
        self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)

    def visualize(self):
        if not self.save_debug and not self.show_plot:
            return

        particles, weights = self.pf.get_particles()
        fig, ax = plt.subplots(figsize=(8, 8))
        if self.odom_history:
            odom = np.asarray(self.odom_history)
            ax.plot(odom[:, 0], odom[:, 1], "r-", label="localpose")
        if self.gps_history:
            gps = np.asarray(self.gps_history)
            ax.plot(gps[:, 0], gps[:, 1], "k-", label="globalpose")
        if self.pf_history:
            pf = np.asarray(self.pf_history)
            ax.plot(pf[:, 0], pf[:, 1], "g-", label="pf")
        if particles.size:
            sizes = 10.0 + 200.0 * weights / max(float(np.max(weights)), 1e-12)
            ax.scatter(particles[:, 0], particles[:, 1], s=sizes, c="tab:blue", alpha=0.35, label="particles")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True)
        ax.legend(loc="best")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Particle Filter Localization")
        if self.save_debug:
            out_path = os.path.join(self.debug_dir, "pf_debug_{:06d}.png".format(self.update_count))
            fig.savefig(out_path, dpi=150)
        if self.show_plot:
            plt.pause(0.001)
        plt.close(fig)

    def spin(self):
        rate = rospy.Rate(self.config.get_float("runtime", "rate", 30.0))
        while not rospy.is_shutdown():
            rate.sleep()


def main():
    rospy.init_node("particle_filter_node")
    node = ParticleFilterNode()
    node.spin()


if __name__ == "__main__":
    main()
