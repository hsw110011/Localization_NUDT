#!/usr/bin/env python3
"""Particle filter localization using LiDAR relative-height BEV and DSM."""

import math
import os
import sys
import threading
import time

_PYTHON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

import cv2
import numpy as np
import rospy
import torch
import torch.nn.functional as F
from geometry_msgs.msg import PoseArray, PoseStamped
from grid_map_msgs.msg import GridMap
from self_state.msg import GlobalPose, LocalPose
from std_msgs.msg import Float32MultiArray

from loc_bev.coord_converter import UTM_SCALE_FACTOR, WGS84_A, CoordConverter
from loc_bev.dem_tool import load_dem_tiff, normalize_to_uint8
from loc_bev.dem_zncc import build_local_grid_offsets


def _read_key_value_ini(path):
    values = {}
    if not path or not os.path.exists(path):
        return values
    with open(path, "r", encoding="utf-8") as fin:
        for raw_line in fin:
            line = raw_line.split("#", 1)[0].split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
            else:
                parts = line.split(None, 1)
                if len(parts) >= 2:
                    values[parts[0]] = parts[1].strip().strip("\"'")
    return values


def _config_float(values, key, fallback):
    try:
        return float(values.get(key, fallback))
    except (TypeError, ValueError):
        return float(fallback)


def _to_bool(value):
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(value)


def _to_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _angle_to_rad(value, unit):
    unit = str(unit).lower()
    value = float(value)
    if unit == "deg":
        return math.radians(value)
    if unit == "rad":
        return value
    if abs(value) > (2.0 * math.pi + 0.25):
        return math.radians(value)
    return value


def _angle_to_math_rad(value, unit, convention):
    if str(unit).lower() == "rad":
        angle_deg = math.degrees(float(value))
    elif str(unit).lower() == "auto" and abs(float(value)) <= (2.0 * math.pi + 0.25):
        angle_deg = math.degrees(float(value))
    else:
        angle_deg = float(value)

    if str(convention).lower() == "azimuth":
        angle_deg = 90.0 - angle_deg
    return _normalize_angle(math.radians(angle_deg))


def _normalize_angle(theta):
    return math.atan2(math.sin(theta), math.cos(theta))


def _yaw_to_quaternion(theta):
    half = 0.5 * float(theta)
    return 0.0, 0.0, math.sin(half), math.cos(half)


def _make_dem_color_map(dem_data):
    gray = normalize_to_uint8(
        dem_data.raw_elevation_map, dem_data.min_height, dem_data.max_height
    )
    return cv2.applyColorMap(gray.astype(np.uint8), cv2.COLORMAP_JET)


def _draw_track(image, pixel, color, previous=None, radius=4):
    x = int(round(pixel.x))
    y = int(round(pixel.y))
    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        if previous is not None:
            cv2.line(image, previous, (x, y), color, 2, cv2.LINE_AA)
        cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)
        return (x, y)
    return previous


def _create_resizable_window(name, width, height):
    flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    if hasattr(cv2, "WINDOW_GUI_NORMAL"):
        flags |= cv2.WINDOW_GUI_NORMAL
    cv2.namedWindow(name, flags)
    cv2.resizeWindow(name, int(width), int(height))


def _multi_array_to_matrix(array_msg, outer_start_index=0, inner_start_index=0):
    dims = list(array_msg.layout.dim)
    data_offset = int(array_msg.layout.data_offset)
    data = np.asarray(array_msg.data[data_offset:], dtype=np.float32)
    if len(dims) < 2:
        raise ValueError("GridMap layer has no 2D layout")

    first_label = str(dims[0].label).lower()
    first_size = int(dims[0].size)
    second_size = int(dims[1].size)
    if first_size <= 0 or second_size <= 0:
        raise ValueError("GridMap layer has invalid layout size")
    if data.size < first_size * second_size:
        raise ValueError("GridMap layer data is shorter than layout size")
    data = data[: first_size * second_size]

    if "column" in first_label:
        matrix = data.reshape((first_size, second_size)).T
        row_start = int(inner_start_index)
        col_start = int(outer_start_index)
    else:
        matrix = data.reshape((first_size, second_size))
        row_start = int(outer_start_index)
        col_start = int(inner_start_index)

    if row_start:
        matrix = np.roll(matrix, -row_start, axis=0)
    if col_start:
        matrix = np.roll(matrix, -col_start, axis=1)
    return matrix.astype(np.float32, copy=False)


def _find_grid_layer(grid_msg, preferred_name, fallback_names):
    names = [preferred_name] + list(fallback_names)
    layer_to_index = {name: index for index, name in enumerate(grid_msg.layers)}
    for name in names:
        if name in layer_to_index:
            index = layer_to_index[name]
            return name, _multi_array_to_matrix(
                grid_msg.data[index],
                grid_msg.outer_start_index,
                grid_msg.inner_start_index,
            )
    raise KeyError("none of layers {} found in {}".format(names, list(grid_msg.layers)))


class HeightParticleFilter(object):
    def __init__(self, node):
        self.node = node
        self.device = node.device
        self.num_particles = int(node.num_particles)
        self.particles = torch.zeros((self.num_particles, 3), dtype=torch.float32, device=self.device)
        self.weights = torch.full(
            (self.num_particles,),
            1.0 / float(self.num_particles),
            dtype=torch.float32,
            device=self.device,
        )
        self.last_cost = torch.zeros((self.num_particles,), dtype=torch.float32, device=self.device)
        self.initialized = False

    def initialize(self, x, y, theta):
        std_x = float(self.node.init_std_x)
        std_y = float(self.node.init_std_y)
        std_theta = math.radians(float(self.node.init_std_theta_deg))
        mean = torch.tensor([x, y, theta], dtype=torch.float32, device=self.device)
        std = torch.tensor([std_x, std_y, std_theta], dtype=torch.float32, device=self.device)
        self.particles = mean + torch.randn_like(self.particles) * std
        self.particles[:, 2] = torch.atan2(torch.sin(self.particles[:, 2]), torch.cos(self.particles[:, 2]))
        self.weights.fill_(1.0 / float(self.num_particles))
        self.initialized = True

    def predict(self, delta_x_local, delta_y_local, delta_theta):
        if not self.initialized:
            return

        theta = self.particles[:, 2]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        dx = float(delta_x_local)
        dy = float(delta_y_local)

        self.particles[:, 0] += cos_t * dx - sin_t * dy
        self.particles[:, 1] += sin_t * dx + cos_t * dy
        self.particles[:, 2] += float(delta_theta)

        if self.node.motion_noise_x > 0.0:
            self.particles[:, 0] += torch.randn(self.num_particles, device=self.device) * float(self.node.motion_noise_x)
        if self.node.motion_noise_y > 0.0:
            self.particles[:, 1] += torch.randn(self.num_particles, device=self.device) * float(self.node.motion_noise_y)
        theta_noise = math.radians(float(self.node.motion_noise_theta_deg))
        if theta_noise > 0.0:
            self.particles[:, 2] += torch.randn(self.num_particles, device=self.device) * theta_noise
        self.particles[:, 2] = torch.atan2(torch.sin(self.particles[:, 2]), torch.cos(self.particles[:, 2]))

    def update(self, lidar_height_np, obs_mask_np):
        if not self.initialized:
            return None

        height = np.asarray(lidar_height_np, dtype=np.float32)
        mask = (np.asarray(obs_mask_np, dtype=np.float32) > 0.5) & np.isfinite(height)
        if height.ndim != 2 or mask.shape != height.shape:
            raise ValueError("height and mask must be matching 2D arrays")

        observed_count = int(np.count_nonzero(mask))
        if observed_count < int(self.node.min_obs_cells):
            rospy.logwarn_throttle(
                2.0,
                "Skip PF update: observed BEV cells %d < %d",
                observed_count,
                int(self.node.min_obs_cells),
            )
            return None

        h_l = torch.from_numpy(np.where(np.isfinite(height), height, 0.0)).to(self.device)
        m_obs = torch.from_numpy(mask.astype(np.float32)).to(self.device)

        valid_h = h_l[(m_obs > 0.5) & (h_l > float(self.node.ground_eps))]
        if int(valid_h.numel()) < int(self.node.hmax_min_valid_count):
            h_max = torch.tensor(float(self.node.hmax_default), dtype=torch.float32, device=self.device)
        else:
            q = float(self.node.height_percentile) / 100.0
            h_max = torch.quantile(valid_h, q)
        h_max = torch.clamp(
            h_max,
            min=float(self.node.hmax_min),
            max=float(self.node.hmax_max),
        )

        h_l_n = torch.clamp(h_l, min=0.0, max=float(h_max.item())) / h_max
        denom = torch.sum(m_obs) + float(self.node.score_eps)

        costs = []
        dsm_valid_ratios = []
        chunk_size = max(1, int(self.node.particle_batch_size))
        for start in range(0, self.num_particles, chunk_size):
            end = min(start + chunk_size, self.num_particles)
            dsm_rel, dsm_valid_ratio = self.node.sample_dsm_relative_height(self.particles[start:end])
            h_d_n = torch.clamp(dsm_rel, min=0.0, max=float(h_max.item())) / h_max
            e_h = h_l_n.unsqueeze(0) - h_d_n
            asym_lambda = torch.where(
                h_l_n.unsqueeze(0) > h_d_n,
                torch.tensor(float(self.node.lambda_lidar_higher), device=self.device),
                torch.tensor(float(self.node.lambda_dsm_higher), device=self.device),
            )
            abs_e = torch.abs(e_h)
            delta = float(self.node.huber_delta)
            huber = torch.where(
                abs_e <= delta,
                0.5 * e_h * e_h,
                delta * (abs_e - 0.5 * delta),
            )
            penalty = m_obs.unsqueeze(0) * asym_lambda * huber
            cost = torch.sum(penalty, dim=(1, 2)) / denom
            invalid = dsm_valid_ratio < float(self.node.dsm_min_valid_ratio)
            if torch.any(invalid):
                cost = torch.where(
                    invalid,
                    cost + float(self.node.invalid_dsm_cost),
                    cost,
                )
            costs.append(cost)
            dsm_valid_ratios.append(dsm_valid_ratio)

        self.last_cost = torch.cat(costs, dim=0)
        score = torch.exp(-self.last_cost / float(self.node.score_tau))
        self.weights *= score
        weight_sum = torch.sum(self.weights)
        if not torch.isfinite(weight_sum) or float(weight_sum.item()) <= 1e-12:
            rospy.logwarn("PF weights collapsed; reset to uniform weights.")
            self.weights.fill_(1.0 / float(self.num_particles))
        else:
            self.weights /= weight_sum

        neff = self.neff()
        if neff < float(self.node.resample_threshold_ratio) * float(self.num_particles):
            self.resample()

        valid_ratio_all = torch.cat(dsm_valid_ratios, dim=0)
        best_index = int(torch.argmin(self.last_cost).item())
        return {
            "h_max": float(h_max.item()),
            "observed_count": observed_count,
            "neff": float(neff),
            "best_index": best_index,
            "best_cost": float(self.last_cost[best_index].item()),
            "best_dsm_valid_ratio": float(valid_ratio_all[best_index].item()),
        }

    def neff(self):
        return float((1.0 / torch.sum(self.weights * self.weights)).item())

    def resample(self):
        cumulative = torch.cumsum(self.weights, dim=0)
        step = 1.0 / float(self.num_particles)
        start = float(torch.rand((), device=self.device).item()) * step
        positions = start + step * torch.arange(self.num_particles, dtype=torch.float32, device=self.device)
        indices = torch.searchsorted(cumulative, positions).clamp(max=self.num_particles - 1)
        self.particles = self.particles[indices].clone()
        self.weights.fill_(1.0 / float(self.num_particles))

    def estimate(self):
        x = torch.sum(self.particles[:, 0] * self.weights)
        y = torch.sum(self.particles[:, 1] * self.weights)
        sin_t = torch.sum(torch.sin(self.particles[:, 2]) * self.weights)
        cos_t = torch.sum(torch.cos(self.particles[:, 2]) * self.weights)
        theta = torch.atan2(sin_t, cos_t)
        return float(x.item()), float(y.item()), float(theta.item())


class LidarHeightPFLocalizationNode(object):
    def _param(self, name, fallback):
        return rospy.get_param("~" + name, self.pf_config.get(name, fallback))

    def _param_str(self, name, fallback):
        return str(self._param(name, fallback))

    def _param_float(self, name, fallback):
        return float(self._param(name, fallback))

    def _param_int(self, name, fallback):
        return int(round(float(self._param(name, fallback))))

    def _param_bool(self, name, fallback):
        return _to_bool(self._param(name, fallback))

    def _param_list(self, name, fallback):
        return _to_list(self._param(name, fallback))

    def __init__(self):
        default_pf_config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "config", "lidar_height_pf.ini")
        )
        self.pf_config_path = rospy.get_param("~pf_config_path", default_pf_config_path)
        self.pf_config = _read_key_value_ini(self.pf_config_path)
        if self.pf_config:
            rospy.loginfo("Loaded PF config: %s", self.pf_config_path)
        else:
            rospy.logwarn("PF config not found or empty: %s", self.pf_config_path)

        self.gridmap_init_path = self._param_str(
            "gridmap_init_path",
            "/home/hsw/catkin_ws/program/GridMap_v7_5_ros1/config/GridMapInit.ini",
        )
        self.gridmap_config = _read_key_value_ini(self.gridmap_init_path)
        default_resolution = _config_float(self.gridmap_config, "lidar_bev_resolution", 0.2)

        self.dem_path = self._param_str("dem_path", "/home/hsw/catkin_ws/doc/miluo_dsm.tif")
        self.lidar_bev_topic = self._param_str("lidar_bev_topic", "/lidar_bev/grid_map")
        self.localpose_topic = self._param_str("localpose_topic", "/self_state/LocalPose")
        self.globalpose_topic = self._param_str("globalpose_topic", "/self_state/GlobalPose")
        self.particles_topic = self._param_str("particles_topic", "/pf_particles")
        self.pose_topic = self._param_str("pose_topic", "/pf_pose")
        self.cost_topic = self._param_str("cost_topic", "/pf_height_cost")
        self.output_frame_id = self._param_str(
            "output_frame_id", self.gridmap_config.get("lidar_bev_frame_id", "map")
        )

        self.height_layer = self._param_str("height_layer", "H_rel_surf")
        self.mask_layer = self._param_str("mask_layer", "M_L")
        self.height_layer_fallbacks = self._param_list("height_layer_fallbacks", ["H_L", "H_clean"])
        self.mask_layer_fallbacks = self._param_list("mask_layer_fallbacks", ["M_obs"])

        self.num_particles = self._param_int("num_particles", 512)
        self.init_std_x = self._param_float("init_std_x", 5.0)
        self.init_std_y = self._param_float("init_std_y", 5.0)
        self.init_std_theta_deg = self._param_float("init_std_theta_deg", 10.0)
        self.motion_noise_x = self._param_float("motion_noise_x", 0.10)
        self.motion_noise_y = self._param_float("motion_noise_y", 0.10)
        self.motion_noise_theta_deg = self._param_float("motion_noise_theta_deg", 1.0)
        self.resample_threshold_ratio = self._param_float("resample_threshold_ratio", 0.5)
        self.particle_batch_size = self._param_int("particle_batch_size", 64)

        self.terrain_resolution = self._param_float("terrain_resolution", default_resolution)
        self.dsm_reference_radius_m = float(
            self._param(
                "dsm_reference_radius_m",
                _config_float(self.gridmap_config, "lidar_bev_near_inner_radius", 2.0),
            )
        )
        self.dsm_min_valid_ratio = self._param_float("dsm_min_valid_ratio", 0.8)
        self.invalid_dsm_cost = self._param_float("invalid_dsm_cost", 1000.0)

        self.height_percentile = self._param_float("height_percentile", 98.0)
        self.ground_eps = self._param_float("ground_eps", 0.20)
        self.hmax_min_valid_count = self._param_int("hmax_min_valid_count", 50)
        self.hmax_default = self._param_float("hmax_default", 5.0)
        self.hmax_min = self._param_float("hmax_min", 5.0)
        self.hmax_max = self._param_float("hmax_max", 40.0)
        self.lambda_lidar_higher = self._param_float("lambda_lidar_higher", 1.0)
        self.lambda_dsm_higher = self._param_float("lambda_dsm_higher", 0.3)
        self.huber_delta = self._param_float("huber_delta", 0.20)
        self.score_tau = self._param_float("score_tau", 0.20)
        self.score_eps = self._param_float("score_eps", 1e-6)
        self.min_obs_cells = self._param_int("min_obs_cells", 50)

        self.local_heading_unit = self._param_str("local_heading_unit", "deg")
        self.global_heading_unit = self._param_str("global_heading_unit", "deg")
        self.global_heading_convention = self._param_str("global_heading_convention", "math")
        self.localpose_delta_mode = self._param_str("localpose_delta_mode", "odom_frame")
        self.initialize_from_global = self._param_bool("initialize_from_global", True)
        self.local_pose_track_mode = self._param_str("local_pose_track_mode", "anchor")

        use_gpu = self._param_bool("use_gpu", True)
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        if use_gpu and self.device.type != "cuda":
            rospy.logwarn("use_gpu=true but CUDA is unavailable; using CPU.")

        self.show_window = self._param_bool("show_window", True)
        self.track_window_width = self._param_int("track_window_width", 800)
        self.track_window_height = self._param_int("track_window_height", 800)
        self.print_every_n = max(1, self._param_int("print_every_n", 1))
        self.loop_rate = self._param_float("rate", 30.0)

        if self.show_window and os.name != "nt" and not os.environ.get("DISPLAY"):
            rospy.logwarn("DISPLAY is not set; disabling OpenCV windows.")
            self.show_window = False

        rospy.loginfo("Loading DSM: %s", self.dem_path)
        start = time.time()
        self.dem_data = load_dem_tiff(self.dem_path)
        if not self.dem_data.is_valid:
            raise RuntimeError("failed to load DSM: {}".format(self.dem_path))
        self.coord_converter = CoordConverter(self.dem_data)
        rospy.loginfo("DSM loaded in %.3f s", time.time() - start)

        dem_raw = np.asarray(self.dem_data.raw_elevation_map, dtype=np.float32)
        dem_valid_np = np.isfinite(dem_raw).astype(np.float32)
        dem_filled = np.where(np.isfinite(dem_raw), dem_raw, 0.0).astype(np.float32)
        self.dem_tensor = torch.from_numpy(dem_filled).to(self.device).view(1, 1, dem_raw.shape[0], dem_raw.shape[1])
        self.dem_valid_tensor = torch.from_numpy(dem_valid_np).to(self.device).view(1, 1, dem_raw.shape[0], dem_raw.shape[1])

        self.local_offsets_key = None
        self.local_x = None
        self.local_y = None
        self.reference_mask = None

        self.pf = HeightParticleFilter(self)
        self.latest_global_pose = None
        self.latest_local_pose = None
        self.previous_local_state = None
        self.local_pose_anchor = None
        self.global_track_point = None
        self.local_track_point = None
        self.pf_track_point = None
        self.vis_track = None
        self.frame_count = 0
        self.lock = threading.RLock()

        if self.show_window:
            self.vis_track = _make_dem_color_map(self.dem_data)
            _create_resizable_window("PF Track", self.track_window_width, self.track_window_height)
            cv2.imshow("PF Track", self.vis_track)
            cv2.waitKey(1)

        self.particles_pub = rospy.Publisher(self.particles_topic, PoseArray, queue_size=1)
        self.pose_pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=1)
        self.cost_pub = rospy.Publisher(self.cost_topic, Float32MultiArray, queue_size=1)

        self.lidar_sub = rospy.Subscriber(self.lidar_bev_topic, GridMap, self._lidar_bev_callback, queue_size=1)
        self.local_sub = rospy.Subscriber(self.localpose_topic, LocalPose, self._local_pose_callback, queue_size=20)
        self.global_sub = rospy.Subscriber(self.globalpose_topic, GlobalPose, self._global_pose_callback, queue_size=20)

        rospy.loginfo(
            "LiDAR height PF started: particles=%d device=%s height_layer=%s mask_layer=%s",
            self.num_particles,
            self.device,
            self.height_layer,
            self.mask_layer,
        )

    def _ensure_local_offsets(self, rows, cols):
        key = (int(rows), int(cols), float(self.terrain_resolution), float(self.dsm_reference_radius_m))
        if self.local_offsets_key == key:
            return
        car_x, car_y = build_local_grid_offsets(rows, cols, self.terrain_resolution)
        self.local_x = torch.from_numpy(car_x.astype(np.float32)).to(self.device)
        self.local_y = torch.from_numpy(car_y.astype(np.float32)).to(self.device)
        radius_sq = float(self.dsm_reference_radius_m) * float(self.dsm_reference_radius_m)
        ref = ((self.local_x * self.local_x + self.local_y * self.local_y) <= radius_sq).float()
        self.reference_mask = ref.view(1, 1, rows, cols)
        self.local_offsets_key = key

    def sample_dsm_relative_height(self, particles):
        count = int(particles.shape[0])
        rows = int(self.local_x.shape[0])
        cols = int(self.local_x.shape[1])

        theta = particles[:, 2].view(count, 1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        center_x = particles[:, 0].view(count, 1, 1)
        center_y = particles[:, 1].view(count, 1, 1)

        gauss_x = center_x + cos_t * self.local_x.unsqueeze(0) - sin_t * self.local_y.unsqueeze(0)
        gauss_y = center_y + sin_t * self.local_x.unsqueeze(0) + cos_t * self.local_y.unsqueeze(0)
        lon, lat = self._gauss_to_wgs84_torch(gauss_x, gauss_y)
        map_x = (lon - float(self.coord_converter.tl_lon)) / float(self.coord_converter.res_lon)
        map_y = (float(self.coord_converter.tl_lat) - lat) / float(self.coord_converter.res_lat)

        dem_h = int(self.dem_data.raw_elevation_map.shape[0])
        dem_w = int(self.dem_data.raw_elevation_map.shape[1])
        grid_x = 2.0 * map_x / max(float(dem_w - 1), 1.0) - 1.0
        grid_y = 2.0 * map_y / max(float(dem_h - 1), 1.0) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)

        sampled = F.grid_sample(
            self.dem_tensor.expand(count, -1, -1, -1),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled_valid = F.grid_sample(
            self.dem_valid_tensor.expand(count, -1, -1, -1),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        valid = (sampled_valid > 0.999).float()
        ref_mask = self.reference_mask.expand(count, -1, -1, -1)
        ref_valid = valid * ref_mask
        ref_count = torch.sum(ref_valid, dim=(1, 2, 3), keepdim=True)
        reference = torch.sum(sampled * ref_valid, dim=(1, 2, 3), keepdim=True) / torch.clamp(ref_count, min=1.0)
        relative = (sampled - reference) * valid
        valid_ratio = torch.mean(valid.view(count, -1), dim=1)
        return relative[:, 0, :, :], valid_ratio

    def _gauss_to_wgs84_torch(self, gauss_x, gauss_y):
        x_val = gauss_x - 500000.0
        y_val = gauss_y

        a = float(WGS84_A)
        k0 = float(UTM_SCALE_FACTOR)
        e2 = float(self.coord_converter.e2)
        e1 = float(self.coord_converter.e1)

        meridian = y_val / k0
        mu = meridian / (
            a * (1.0 - e2 / 4.0 - 3.0 * e2 * e2 / 64.0 - 5.0 * e2**3 / 256.0)
        )

        phi1 = (
            mu
            + (3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0) * torch.sin(2.0 * mu)
            + (21.0 * e1 * e1 / 16.0 - 55.0 * e1**4 / 32.0) * torch.sin(4.0 * mu)
            + (151.0 * e1**3 / 96.0) * torch.sin(6.0 * mu)
        )

        ee = e2 / (1.0 - e2)
        sin_phi1 = torch.sin(phi1)
        cos_phi1 = torch.cos(phi1)
        tan_phi1 = torch.tan(phi1)

        c1 = ee * cos_phi1 * cos_phi1
        t1 = tan_phi1 * tan_phi1
        n1 = a / torch.sqrt(1.0 - e2 * sin_phi1 * sin_phi1)
        r1 = a * (1.0 - e2) / torch.pow(1.0 - e2 * sin_phi1 * sin_phi1, 1.5)
        d_val = x_val / (n1 * k0)

        lat_rad = phi1 - (n1 * tan_phi1 / r1) * (
            d_val * d_val / 2.0
            - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1 * c1 - 9.0 * ee)
            * torch.pow(d_val, 4)
            / 24.0
            + (
                61.0
                + 90.0 * t1
                + 298.0 * c1
                + 45.0 * t1 * t1
                - 252.0 * ee
                - 3.0 * c1 * c1
            )
            * torch.pow(d_val, 6)
            / 720.0
        )

        lon_rad_diff = (
            d_val
            - (1.0 + 2.0 * t1 + c1) * torch.pow(d_val, 3) / 6.0
            + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1 * c1 + 8.0 * ee + 24.0 * t1 * t1)
            * torch.pow(d_val, 5)
            / 120.0
        ) / cos_phi1

        cm_deg = -180.0 + (float(self.coord_converter.zone_num) * 6.0) - 3.0
        rad_to_deg = 180.0 / math.pi
        lat = lat_rad * rad_to_deg
        lon = cm_deg + lon_rad_diff * rad_to_deg
        return lon, lat

    def _global_pose_callback(self, msg):
        with self.lock:
            self.latest_global_pose = msg
            if self.initialize_from_global and not self.pf.initialized:
                self._initialize_from_global_pose(msg)
            if self.show_window:
                pixel = self.coord_converter.wgs84_to_pixel(float(msg.longitude), float(msg.latitude))
                self.global_track_point = _draw_track(
                    self.vis_track, pixel, (255, 255, 255), self.global_track_point
                )
                cv2.imshow("PF Track", self.vis_track)
                cv2.waitKey(1)

    def _local_pose_callback(self, msg):
        with self.lock:
            current = self._local_pose_state(msg)
            if self.previous_local_state is not None and self.pf.initialized:
                delta = self._local_pose_delta(self.previous_local_state, current)
                self.pf.predict(delta[0], delta[1], delta[2])
            self.previous_local_state = current
            self.latest_local_pose = msg

            if self.show_window:
                pixel = self._local_pose_to_dem_pixel(msg)
                if pixel is not None:
                    self.local_track_point = _draw_track(
                        self.vis_track, pixel, (0, 0, 255), self.local_track_point
                    )
                    cv2.imshow("PF Track", self.vis_track)
                    cv2.waitKey(1)

    def _lidar_bev_callback(self, msg):
        with self.lock:
            if not self.pf.initialized:
                if self.latest_global_pose is not None:
                    self._initialize_from_global_pose(self.latest_global_pose)
                else:
                    rospy.logwarn_throttle(2.0, "Waiting for GlobalPose before PF initialization.")
                    return

            try:
                height_name, height = _find_grid_layer(
                    msg, self.height_layer, self.height_layer_fallbacks
                )
                mask_name, mask = _find_grid_layer(msg, self.mask_layer, self.mask_layer_fallbacks)
            except Exception as exc:
                rospy.logwarn_throttle(2.0, "Failed to parse LiDAR BEV GridMap: %s", exc)
                return

            self._ensure_local_offsets(height.shape[0], height.shape[1])
            stats = self.pf.update(height, mask)
            if stats is None:
                return

            self.frame_count += 1
            estimate = self.pf.estimate()
            stamp = msg.info.header.stamp if msg.info.header.stamp != rospy.Time() else rospy.Time.now()
            frame_id = self.output_frame_id
            self._publish_estimate(stamp, frame_id, estimate)
            self._publish_particles(stamp, frame_id)
            self._publish_costs()

            if self.frame_count % self.print_every_n == 0:
                rospy.loginfo(
                    (
                        "PF update #%d: layers=(%s,%s) obs=%d hmax=%.3f "
                        "best_cost=%.5f best_dsm_valid=%.2f neff=%.1f "
                        "pose=(%.3f, %.3f, %.3fdeg)"
                    ),
                    self.frame_count,
                    height_name,
                    mask_name,
                    stats["observed_count"],
                    stats["h_max"],
                    stats["best_cost"],
                    stats["best_dsm_valid_ratio"],
                    stats["neff"],
                    estimate[0],
                    estimate[1],
                    math.degrees(estimate[2]),
                )

            if self.show_window:
                pixel = self.coord_converter.gauss_to_pixel(estimate[0], estimate[1])
                self.pf_track_point = _draw_track(
                    self.vis_track, pixel, (0, 255, 0), self.pf_track_point, radius=5
                )
                self._draw_particles_on_track()
                cv2.imshow("PF Track", self.vis_track)
                cv2.waitKey(1)

    def _initialize_from_global_pose(self, msg):
        gauss = self.coord_converter.wgs84_to_gauss(float(msg.longitude), float(msg.latitude))
        theta = _angle_to_math_rad(
            float(msg.azimuth),
            self.global_heading_unit,
            self.global_heading_convention,
        )
        self.pf.initialize(gauss.x, gauss.y, theta)
        rospy.loginfo(
            "Initialized PF from GlobalPose: gauss=(%.3f, %.3f), theta=%.3fdeg",
            gauss.x,
            gauss.y,
            math.degrees(theta),
        )

    def _local_pose_state(self, msg):
        return (
            float(msg.dr_x),
            float(msg.dr_y),
            _angle_to_rad(float(msg.dr_heading), self.local_heading_unit),
            float(msg.local_time),
        )

    def _local_pose_delta(self, previous, current):
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        dtheta = _normalize_angle(current[2] - previous[2])
        if str(self.localpose_delta_mode).lower() in ("local_delta", "body", "body_frame"):
            return dx, dy, dtheta

        prev_theta = previous[2]
        cos_t = math.cos(prev_theta)
        sin_t = math.sin(prev_theta)
        local_dx = cos_t * dx + sin_t * dy
        local_dy = -sin_t * dx + cos_t * dy
        return local_dx, local_dy, dtheta

    def _local_pose_to_dem_pixel(self, msg):
        mode = str(self.local_pose_track_mode).lower()
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
        local_x0, local_y0, gauss_x0, gauss_y0 = self.local_pose_anchor
        return self.coord_converter.gauss_to_pixel(
            gauss_x0 + (float(msg.dr_x) - local_x0),
            gauss_y0 + (float(msg.dr_y) - local_y0),
        )

    def _publish_estimate(self, stamp, frame_id, estimate):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.pose.position.x = estimate[0]
        msg.pose.position.y = estimate[1]
        msg.pose.position.z = 0.0
        qx, qy, qz, qw = _yaw_to_quaternion(estimate[2])
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.pose_pub.publish(msg)

    def _publish_particles(self, stamp, frame_id):
        particles = self.pf.particles.detach().cpu().numpy()
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        for particle in particles:
            pose = PoseStamped().pose
            pose.position.x = float(particle[0])
            pose.position.y = float(particle[1])
            pose.position.z = 0.0
            qx, qy, qz, qw = _yaw_to_quaternion(float(particle[2]))
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw
            msg.poses.append(pose)
        self.particles_pub.publish(msg)

    def _publish_costs(self):
        msg = Float32MultiArray()
        msg.data = self.pf.last_cost.detach().cpu().numpy().astype(np.float32).tolist()
        self.cost_pub.publish(msg)

    def _draw_particles_on_track(self):
        if self.vis_track is None:
            return
        particles = self.pf.particles.detach().cpu().numpy()
        weights = self.pf.weights.detach().cpu().numpy()
        if particles.shape[0] > 128:
            chosen = np.argsort(weights)[-128:]
        else:
            chosen = np.arange(particles.shape[0])
        overlay = self.vis_track.copy()
        for index in chosen:
            pixel = self.coord_converter.gauss_to_pixel(float(particles[index, 0]), float(particles[index, 1]))
            x = int(round(pixel.x))
            y = int(round(pixel.y))
            if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
                cv2.circle(overlay, (x, y), 1, (0, 200, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.35, self.vis_track, 0.65, 0.0, self.vis_track)

    def spin(self):
        rate = rospy.Rate(float(self.loop_rate))
        while not rospy.is_shutdown():
            if self.show_window:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    rospy.signal_shutdown("closed by keyboard")
            rate.sleep()
        if self.show_window:
            cv2.destroyAllWindows()


def main():
    rospy.init_node("lidar_height_pf_localization")
    node = LidarHeightPFLocalizationNode()
    node.spin()


if __name__ == "__main__":
    main()
