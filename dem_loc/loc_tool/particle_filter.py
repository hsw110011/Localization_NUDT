#!/usr/bin/env python3
"""粒子滤波定位 — 复用里程计递推与 DSM-BEV GPU batch score。"""

import csv
import math
import os
from dataclasses import dataclass

import numpy as np
import torch

from .dsm_bev_score import DsmBevScoreConfig, build_obs_mask, compute_h_max
from .dsm_bev_score_batch import compute_dsm_bev_score_batch, lidar_layers_to_torch, _sanitize_h_max
from .dsm_patch_batch import crop_batch_with_bev_layers
from .particle_weight_vis import save_particle_weight_view


def _normalize_angle(yaw):
    return torch.atan2(torch.sin(yaw), torch.cos(yaw))


def _percentile_1d(values, q):
    values = values.flatten()
    if values.numel() == 0:
        return torch.tensor(float("nan"), dtype=torch.float32, device=values.device)
    sorted_values = torch.sort(values).values
    idx = int(round((sorted_values.numel() - 1) * float(q)))
    idx = max(0, min(int(sorted_values.numel()) - 1, idx))
    return sorted_values[idx]


@dataclass
class ParticleFilterConfig(object):
    enable: bool = False
    num_particles: int = 256
    random_seed: int = 42
    device: str = "cuda:0"

    init_std_x: float = 8.0
    init_std_y: float = 8.0
    init_std_yaw: float = 0.10

    motion_std_x: float = 0.30  # 平移噪声系数：std_x = coeff * |delta_x| (m)
    motion_std_y: float = 0.15  # 横向噪声系数：std_y = coeff * (|delta_y| + 0.5 * |delta_x|)
    motion_std_yaw: float = 0.004  # std_yaw = coeff * |delta_yaw| (rad)
    stationary_speed_threshold: float = 0.05  # LocalPose.vehicle_speed 低于该值时跳过 PF 更新
    motion_speed_noise_gain: float = 0.10  # 速度噪声放大：std *= 1 + gain * |speed_mps|

    score_is_cost: bool = True
    # 组内权重锐度：1.0=仅按 score/max(score)；<1 进一步放大 top 粒子权重
    score_temperature: float = 1.0
    # 粒子权重区分度模式：absolute=固定 tau*temperature；adaptive=按当前帧 J spread 自适应拉开权重
    weight_contrast_mode: str = "adaptive"
    weight_contrast_target_log_range: float = 3.0
    weight_contrast_min_j_spread: float = 1e-4
    weight_contrast_max_sharpen: float = 20.0
    nan_score_penalty: float = 1e6
    min_valid_weight_sum: float = 1e-12

    # 最终位姿估计模式：weighted=全粒子加权；elite_weighted=分数前若干比例加权。
    estimate_mode: str = "elite_weighted"
    elite_top_fraction: float = 0.50

    resample_threshold: float = 0.5
    enable_roughening: bool = True
    roughening_std_x: float = 0.02
    roughening_std_y: float = 0.02
    roughening_std_yaw: float = 0.001

    save_debug: bool = True
    save_particle_csv: bool = False
    save_score_csv: bool = True
    save_weight_vis: bool = True
    weight_vis_every_n: int = 1
    weight_vis_dir: str = "pf_weight_vis"
    debug_dir: str = "output/particle_filter_debug"

    pf_pose_topic: str = "/pf_global_pose"
    pf_update_every_n: int = 1
    pf_log_every_n: int = 10


def load_particle_filter_config(get_param):
    """从 ROS param 读取 PF 配置（默认值与 particle_filter.ini 一致）。"""

    def _p(name, default):
        return get_param("~" + name, default)

    return ParticleFilterConfig(
        enable=bool(_p("pf_enable", False)),
        num_particles=int(_p("pf_num_particles", 256)),
        random_seed=int(_p("pf_random_seed", 42)),
        device=str(_p("pf_device", "cuda:0")),
        init_std_x=float(_p("pf_init_std_x", 8.0)),
        init_std_y=float(_p("pf_init_std_y", 8.0)),
        init_std_yaw=float(_p("pf_init_std_yaw", 0.10)),
        motion_std_x=float(_p("pf_motion_std_x", 0.30)),
        motion_std_y=float(_p("pf_motion_std_y", 0.15)),
        motion_std_yaw=float(_p("pf_motion_std_yaw", 0.004)),
        stationary_speed_threshold=float(_p("pf_stationary_speed_threshold", 0.05)),
        motion_speed_noise_gain=float(_p("pf_motion_speed_noise_gain", 0.10)),
        score_is_cost=bool(_p("pf_score_is_cost", True)),
        score_temperature=float(_p("pf_score_temperature", 1.0)),
        weight_contrast_mode=str(_p("pf_weight_contrast_mode", "adaptive")),
        weight_contrast_target_log_range=float(_p("pf_weight_contrast_target_log_range", 3.0)),
        weight_contrast_min_j_spread=float(_p("pf_weight_contrast_min_j_spread", 1e-4)),
        weight_contrast_max_sharpen=float(_p("pf_weight_contrast_max_sharpen", 20.0)),
        estimate_mode=str(_p("pf_estimate_mode", "elite_weighted")),
        elite_top_fraction=float(_p("pf_elite_top_fraction", 0.50)),
        nan_score_penalty=float(_p("pf_nan_score_penalty", 1e6)),
        min_valid_weight_sum=float(_p("pf_min_valid_weight_sum", 1e-12)),
        resample_threshold=float(_p("pf_resample_threshold", 0.5)),
        enable_roughening=bool(_p("pf_enable_roughening", True)),
        roughening_std_x=float(_p("pf_roughening_std_x", 0.02)),
        roughening_std_y=float(_p("pf_roughening_std_y", 0.02)),
        roughening_std_yaw=float(_p("pf_roughening_std_yaw", 0.001)),
        save_debug=bool(_p("pf_save_debug", True)),
        save_particle_csv=bool(_p("pf_save_particle_csv", False)),
        save_score_csv=bool(_p("pf_save_score_csv", True)),
        save_weight_vis=bool(_p("pf_save_weight_vis", True)),
        weight_vis_every_n=max(1, int(_p("pf_weight_vis_every_n", 1))),
        weight_vis_dir=str(_p("pf_weight_vis_dir", "pf_weight_vis")),
        debug_dir=str(_p("pf_debug_dir", "output/particle_filter_debug")),
        pf_pose_topic=str(_p("pf_pose_topic", "/pf_global_pose")),
        pf_update_every_n=max(1, int(_p("pf_update_every_n", 1))),
        pf_log_every_n=max(1, int(_p("pf_log_every_n", 10))),
    )


def load_particle_filter_ini_defaults(ini_path):
    """读取 ini 默认值，供 rosparam load 前或离线脚本使用。"""
    import configparser

    parser = configparser.ConfigParser()
    if not os.path.isfile(ini_path):
        return {}
    parser.read(ini_path, encoding="utf-8")
    if not parser.has_section("particle_filter"):
        return {}
    section = parser["particle_filter"]
    mapping = {
        "enable": "pf_enable",
        "num_particles": "pf_num_particles",
        "random_seed": "pf_random_seed",
        "device": "pf_device",
        "init_std_x": "pf_init_std_x",
        "init_std_y": "pf_init_std_y",
        "init_std_yaw": "pf_init_std_yaw",
        "motion_std_x": "pf_motion_std_x",
        "motion_std_y": "pf_motion_std_y",
        "motion_std_yaw": "pf_motion_std_yaw",
        "stationary_speed_threshold": "pf_stationary_speed_threshold",
        "motion_speed_noise_gain": "pf_motion_speed_noise_gain",
        "score_is_cost": "pf_score_is_cost",
        "score_temperature": "pf_score_temperature",
        "weight_contrast_mode": "pf_weight_contrast_mode",
        "weight_contrast_target_log_range": "pf_weight_contrast_target_log_range",
        "weight_contrast_min_j_spread": "pf_weight_contrast_min_j_spread",
        "weight_contrast_max_sharpen": "pf_weight_contrast_max_sharpen",
        "estimate_mode": "pf_estimate_mode",
        "elite_top_fraction": "pf_elite_top_fraction",
        "nan_score_penalty": "pf_nan_score_penalty",
        "min_valid_weight_sum": "pf_min_valid_weight_sum",
        "resample_threshold": "pf_resample_threshold",
        "enable_roughening": "pf_enable_roughening",
        "roughening_std_x": "pf_roughening_std_x",
        "roughening_std_y": "pf_roughening_std_y",
        "roughening_std_yaw": "pf_roughening_std_yaw",
        "save_debug": "pf_save_debug",
        "save_particle_csv": "pf_save_particle_csv",
        "save_score_csv": "pf_save_score_csv",
        "save_weight_vis": "pf_save_weight_vis",
        "weight_vis_every_n": "pf_weight_vis_every_n",
        "weight_vis_dir": "pf_weight_vis_dir",
        "debug_dir": "pf_debug_dir",
        "pf_pose_topic": "pf_pose_topic",
        "pf_update_every_n": "pf_update_every_n",
        "pf_log_every_n": "pf_log_every_n",
    }
    out = {}
    for ini_key, param_key in mapping.items():
        if ini_key in section:
            out[param_key] = section.get(ini_key)
    if parser.has_section("dsm_bev_score"):
        score_section = parser["dsm_bev_score"]
        score_mapping = {
            "alpha_h": "alpha_h",
            "alpha_gx": "alpha_gx",
            "alpha_gy": "alpha_gy",
            "lambda_lidar_higher": "lambda_lidar_higher",
            "lambda_dsm_higher": "lambda_dsm_higher",
            "delta_h": "delta_h",
            "w_h_base": "w_h_base",
            "w_h_height": "w_h_height",
            "delta_g": "delta_g",
            "w_g_base": "w_g_base",
            "tau": "tau",
            "grad_cap": "grad_cap",
            "grad_mask_erode_px": "grad_mask_erode_px",
            "dsm_long_sign": "dsm_long_sign",
            "dsm_lat_sign": "dsm_lat_sign",
        }
        for ini_key, param_key in score_mapping.items():
            if ini_key in score_section:
                out[param_key] = score_section.get(ini_key)
    if parser.has_section("topics") and "pf_pose_topic" in parser["topics"]:
        out["pf_pose_topic"] = parser["topics"].get("pf_pose_topic")
    return out


class ParticleFilter(object):
    """DSM-LiDAR 粒子滤波器。"""

    def __init__(self, pf_config, score_config, dsm_cropper):
        self.pf_config = pf_config
        self.score_config = score_config
        self.dsm_cropper = dsm_cropper

        cropper_device = getattr(dsm_cropper, "device", None)
        if cropper_device is not None:
            self.device = torch.device(cropper_device)
            if str(pf_config.device).startswith("cuda") and self.device.type != "cuda":
                import warnings

                warnings.warn(
                    "DSM cropper is on CPU, particle filter falls back to CPU tensors."
                )
        elif torch.cuda.is_available() and str(pf_config.device).startswith("cuda"):
            self.device = torch.device(pf_config.device)
        else:
            self.device = torch.device("cpu")
            if str(pf_config.device).startswith("cuda"):
                import warnings

                warnings.warn("CUDA unavailable, particle filter falls back to CPU tensors.")

        self.rng = np.random.default_rng(int(pf_config.random_seed))
        self.torch_gen = torch.Generator(device=self.device)
        self.torch_gen.manual_seed(int(pf_config.random_seed))

        n = int(pf_config.num_particles)
        self.particles = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.weights = torch.full((n,), 1.0 / max(n, 1), dtype=torch.float32, device=self.device)
        self.scores = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.j_totals = torch.zeros(n, dtype=torch.float32, device=self.device)

        self.initialized = False
        self.frame_id = 0
        self.h_max = None
        self._lidar_torch = None
        self.last_best_vis = None
        self.last_debug = {}
        self.last_motion = {}
        self.last_weight_debug = {}
        self._weight_vis_failed = False

        if pf_config.save_debug:
            os.makedirs(pf_config.debug_dir, exist_ok=True)
            self._init_debug_csv()

    def _init_debug_csv(self):
        score_path = os.path.join(self.pf_config.debug_dir, "pf_score.csv")
        if self.pf_config.save_score_csv and not os.path.isfile(score_path):
            with open(score_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "frame_id",
                        "x_est",
                        "y_est",
                        "yaw_est",
                        "best_x",
                        "best_y",
                        "best_yaw",
                        "best_score",
                        "mean_score",
                        "min_score",
                        "max_score",
                        "neff",
                        "resampled",
                        "weight_max",
                        "weight_min",
                        "weight_mode",
                        "weight_j_scale",
                        "weight_j_spread",
                        "weight_base_scale",
                        "weight_log_range",
                        "valid_score_count",
                        "cuda_memory_allocated",
                        "cuda_memory_reserved",
                    ]
                )
        particle_path = os.path.join(self.pf_config.debug_dir, "pf_particles.csv")
        if self.pf_config.save_particle_csv and (
            not os.path.isfile(particle_path) or os.path.getsize(particle_path) == 0
        ):
            with open(particle_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "frame_id",
                        "particle_id",
                        "x",
                        "y",
                        "yaw",
                        "score",
                        "j_total",
                        "weight",
                    ]
                )

    def initialize(self, gauss_x, gauss_y, yaw_rad):
        n = int(self.pf_config.num_particles)
        cfg = self.pf_config
        noise = self.rng.normal(
            0.0,
            [cfg.init_std_x, cfg.init_std_y, cfg.init_std_yaw],
            size=(n, 3),
        ).astype(np.float32)
        state = np.array([float(gauss_x), float(gauss_y), float(yaw_rad)], dtype=np.float32)
        particles = state.reshape(1, 3) + noise
        particles[:, 2] = np.arctan2(np.sin(particles[:, 2]), np.cos(particles[:, 2]))
        self.particles = torch.as_tensor(particles, dtype=torch.float32, device=self.device)
        self.weights.fill_(1.0 / max(n, 1))
        self.scores.zero_()
        self.j_totals.zero_()
        self.last_weight_debug = {}
        self.initialized = True
        self.frame_id = 0
        self.h_max = None
        self._lidar_torch = None

    def propagate(self, delta_local_x, delta_local_y, odom_theta, delta_yaw=0.0, speed_mps=0.0):
        """里程计位移 + 航向增量传播；运动噪声与位移和速度正相关。"""
        if not self.initialized:
            return

        cfg = self.pf_config
        dx = float(delta_local_x)
        dy = float(delta_local_y)
        dyaw = float(delta_yaw)
        try:
            speed = abs(float(speed_mps))
            if not math.isfinite(speed):
                speed = 0.0
        except (TypeError, ValueError):
            speed = 0.0
        speed_noise_scale = 1.0 + max(0.0, float(cfg.motion_speed_noise_gain)) * speed
        cos_t = math.cos(float(odom_theta))
        sin_t = math.sin(float(odom_theta))
        delta_global_x = dx * cos_t - dy * sin_t
        delta_global_y = dx * sin_t + dy * cos_t

        self.particles[:, 0] += delta_global_x
        self.particles[:, 1] += delta_global_y
        if abs(dyaw) > 1e-12:
            self.particles[:, 2] += dyaw

        longitudinal_motion = abs(dx)
        lateral_motion = abs(dy)
        lateral_motion_with_forward = lateral_motion + 0.5 * longitudinal_motion
        std_lx = float(cfg.motion_std_x) * longitudinal_motion * speed_noise_scale
        std_ly = float(cfg.motion_std_y) * lateral_motion_with_forward * speed_noise_scale
        std_yaw = float(cfg.motion_std_yaw) * abs(dyaw) * speed_noise_scale
        if std_lx <= 0.0 and std_ly <= 0.0 and std_yaw <= 0.0:
            self.particles[:, 2] = _normalize_angle(self.particles[:, 2])
            self.last_motion = {
                "delta_local_x": dx,
                "delta_local_y": dy,
                "delta_yaw": dyaw,
                "delta_global_x": delta_global_x,
                "delta_global_y": delta_global_y,
                "motion_std_lx": std_lx,
                "motion_std_ly": std_ly,
                "motion_std_yaw": std_yaw,
                "speed_mps": speed,
                "speed_noise_scale": speed_noise_scale,
                "stationary_skip": False,
            }
            return

        motion_noise = torch.randn(
            self.particles.shape,
            generator=self.torch_gen,
            device=self.device,
            dtype=torch.float32,
        )
        noise_lx = motion_noise[:, 0] * std_lx
        noise_ly = motion_noise[:, 1] * std_ly
        motion_noise[:, 0] = noise_lx * cos_t - noise_ly * sin_t
        motion_noise[:, 1] = noise_lx * sin_t + noise_ly * cos_t
        motion_noise[:, 2] *= std_yaw
        self.particles += motion_noise
        self.particles[:, 2] = _normalize_angle(self.particles[:, 2])
        self.last_motion = {
            "delta_local_x": dx,
            "delta_local_y": dy,
            "delta_yaw": dyaw,
            "delta_global_x": delta_global_x,
            "delta_global_y": delta_global_y,
            "motion_std_lx": std_lx,
            "motion_std_ly": std_ly,
            "motion_std_yaw": std_yaw,
            "speed_mps": speed,
            "speed_noise_scale": speed_noise_scale,
            "stationary_skip": False,
        }

    def _prepare_lidar(self, lidar_layers, h_max=None):
        self._lidar_torch = lidar_layers_to_torch(lidar_layers, self.device)
        if h_max is None:
            m_obs = build_obs_mask(lidar_layers["M_obs"])
            h_max = compute_h_max(lidar_layers["H_L"], m_obs, self.score_config)
        self.h_max = _sanitize_h_max(h_max, self.score_config)

    def _cache_best_vis(self, bi):
        idx = int(bi)
        self.last_best_vis = {
            "gauss_x": float(self.particles[idx, 0].item()),
            "gauss_y": float(self.particles[idx, 1].item()),
            "yaw": float(self.particles[idx, 2].item()),
            "score": float(self.scores[idx].item()),
            "J_total": float(self.j_totals[idx].item()),
        }

    def _compute_log_weights(self, j_total, score):
        cfg = self.pf_config
        eps = max(float(getattr(self.score_config, "eps", 1e-6)), 1e-12)
        tau = max(float(self.score_config.tau), eps)
        temperature = max(float(cfg.score_temperature), 1e-6)
        base_scale = max(tau * temperature, eps)
        debug = {
            "weight_mode": "cost_absolute",
            "weight_j_scale": base_scale,
            "weight_j_spread": 0.0,
            "weight_base_scale": base_scale,
            "weight_log_range": 0.0,
        }

        if not bool(cfg.score_is_cost):
            score_safe = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
            score_max = torch.max(score_safe)
            if float(score_max.item()) > 1e-30:
                log_weights = torch.log(torch.clamp(score_safe / score_max, min=1e-30))
                if abs(temperature - 1.0) > 1e-6:
                    log_weights = log_weights / temperature
            else:
                log_weights = torch.zeros_like(score_safe)
            finite_log = log_weights[torch.isfinite(log_weights)]
            if finite_log.numel() > 0:
                debug["weight_log_range"] = float((finite_log.max() - finite_log.min()).item())
            debug["weight_mode"] = "score_relative"
            self.last_weight_debug = debug
            return log_weights

        finite_mask = torch.isfinite(j_total)
        finite_j = j_total[finite_mask]
        if finite_j.numel() == 0:
            debug["weight_mode"] = "uniform_invalid_cost"
            self.last_weight_debug = debug
            return torch.zeros_like(j_total)

        j_min = torch.min(finite_j)
        invalid_delta = max(float(cfg.nan_score_penalty), base_scale * 80.0)
        j_safe = torch.where(
            finite_mask,
            j_total,
            torch.full_like(j_total, float(j_min.item()) + invalid_delta),
        )
        delta_j = torch.clamp(j_safe - j_min, min=0.0)
        finite_delta = torch.clamp(finite_j - j_min, min=0.0)

        j_spread = 0.0
        if finite_delta.numel() >= 2:
            q10 = _percentile_1d(finite_delta, 0.10)
            q90 = _percentile_1d(finite_delta, 0.90)
            if torch.isfinite(q10) and torch.isfinite(q90):
                j_spread = max(0.0, float((q90 - q10).item()))

        mode_req = str(cfg.weight_contrast_mode).strip().lower()
        j_scale = base_scale
        if mode_req in ("adaptive", "contrast", "contrastive"):
            min_spread = max(float(cfg.weight_contrast_min_j_spread), 0.0)
            if j_spread >= min_spread:
                target_log_range = max(float(cfg.weight_contrast_target_log_range), 1e-6)
                adaptive_scale = max(j_spread / target_log_range, eps)
                max_sharpen = max(float(cfg.weight_contrast_max_sharpen), 1.0)
                min_scale = max(base_scale / max_sharpen, eps)
                j_scale = min(base_scale, max(adaptive_scale, min_scale))
                debug["weight_mode"] = "adaptive_contrast"
            else:
                debug["weight_mode"] = "adaptive_flat"
        elif mode_req in ("absolute", "fixed", "legacy"):
            debug["weight_mode"] = "cost_absolute"
        else:
            debug["weight_mode"] = "cost_absolute_unknown_mode"

        log_weights = -delta_j / max(j_scale, eps)
        if finite_delta.numel() > 0:
            debug["weight_log_range"] = float((finite_delta.max() / max(j_scale, eps)).item())
        debug["weight_j_scale"] = float(j_scale)
        debug["weight_j_spread"] = float(j_spread)
        debug["weight_base_scale"] = float(base_scale)
        self.last_weight_debug = debug
        return log_weights

    def update(self, lidar_layers, h_max=None):
        if not self.initialized:
            raise RuntimeError("particle filter not initialized")

        self._prepare_lidar(lidar_layers, h_max=h_max)
        lt = self._lidar_torch
        n = int(self.particles.shape[0])

        with torch.inference_mode():
            dsm = crop_batch_with_bev_layers(
                self.dsm_cropper,
                self.particles,
                grad_cap=self.score_config.grad_cap,
            )
            h_d = dsm["H_rel_surf"]
            g_long = dsm["G_long_L"] * float(self.score_config.dsm_long_sign)
            g_lat = dsm["G_lat_L"] * float(self.score_config.dsm_lat_sign)

            batch_scores = compute_dsm_bev_score_batch(
                lt["H_L"],
                lt["Gx_L"],
                lt["Gy_L"],
                lt["M_obs"],
                h_d,
                g_lat,
                g_long,
                config=self.score_config,
                h_max=self.h_max,
            )

        j_total = batch_scores["J_total"]
        score = batch_scores["score"]
        self.j_totals = j_total
        self.scores = score

        best_idx = int(torch.argmax(score).item())
        self._cache_best_vis(best_idx)

        cfg = self.pf_config
        log_likelihood = self._compute_log_weights(j_total, score)

        bad = ~torch.isfinite(log_likelihood)
        if bad.any():
            log_likelihood = torch.where(
                bad,
                torch.full_like(log_likelihood, -float(cfg.nan_score_penalty)),
                log_likelihood,
            )

        prior = torch.nan_to_num(self.weights, nan=0.0, posinf=0.0, neginf=0.0)
        prior = torch.clamp(prior, min=0.0)
        if prior.numel() != n:
            prior = torch.full((n,), 1.0 / max(n, 1), device=self.device)
        else:
            prior_sum = prior.sum()
            if (not torch.isfinite(prior_sum)) or float(prior_sum.item()) <= 1e-12:
                prior = torch.full((n,), 1.0 / max(n, 1), device=self.device)
            else:
                prior = prior / prior_sum

        # 标准 SIR 递推：后验权重 = 上一帧 prior 权重 * 当前观测似然。
        log_weights = torch.log(torch.clamp(prior, min=1e-12)) + log_likelihood
        log_weights = log_weights - torch.max(log_weights)
        weights = torch.exp(log_weights)
        weight_sum = weights.sum()
        if (not torch.isfinite(weight_sum)) or float(weight_sum.item()) < float(
            cfg.min_valid_weight_sum
        ):
            weights = torch.full((n,), 1.0 / max(n, 1), device=self.device)
        else:
            weights = weights / weight_sum
        self.weights = weights

    def effective_sample_size(self):
        return float(1.0 / torch.sum(self.weights * self.weights).item())

    def systematic_resample(self):
        n = int(self.weights.shape[0])
        if n <= 0:
            return False

        positions = (
            torch.arange(n, device=self.device, dtype=torch.float32)
            + torch.rand(1, device=self.device, generator=self.torch_gen).item()
        ) / float(n)
        cumsum = torch.cumsum(self.weights, dim=0)
        indices = torch.searchsorted(cumsum, positions)
        indices = torch.clamp(indices, max=n - 1)
        self.particles = self.particles[indices]
        self.scores = self.scores[indices]
        self.j_totals = self.j_totals[indices]
        self.weights.fill_(1.0 / max(n, 1))

        cfg = self.pf_config
        if cfg.enable_roughening:
            noise = torch.randn(
                self.particles.shape,
                generator=self.torch_gen,
                device=self.device,
                dtype=torch.float32,
            )
            noise[:, 0] *= float(cfg.roughening_std_x)
            noise[:, 1] *= float(cfg.roughening_std_y)
            noise[:, 2] *= float(cfg.roughening_std_yaw)
            self.particles += noise
            self.particles[:, 2] = _normalize_angle(self.particles[:, 2])
        return True

    def maybe_resample(self):
        neff = self.effective_sample_size()
        threshold = float(self.pf_config.resample_threshold) * int(self.pf_config.num_particles)
        if neff < threshold:
            return self.systematic_resample()
        return False

    def _mean_pose_from_indices(self, indices, weights=None):
        sel = self.particles[indices]
        if sel.shape[0] == 0:
            sel = self.particles
        if weights is None:
            w = torch.full(
                (int(sel.shape[0]),),
                1.0 / max(int(sel.shape[0]), 1),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            w = torch.nan_to_num(weights[indices], nan=0.0, posinf=0.0, neginf=0.0)
            w = torch.clamp(w, min=0.0)
            w_sum = torch.sum(w)
            if (not torch.isfinite(w_sum)) or float(w_sum.item()) <= 1e-12:
                w = torch.full(
                    (int(sel.shape[0]),),
                    1.0 / max(int(sel.shape[0]), 1),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                w = w / w_sum
        x = float(torch.sum(w * sel[:, 0]).item())
        y = float(torch.sum(w * sel[:, 1]).item())
        sin_sum = float(torch.sum(w * torch.sin(sel[:, 2])).item())
        cos_sum = float(torch.sum(w * torch.cos(sel[:, 2])).item())
        yaw = float(math.atan2(sin_sum, cos_sum))
        return x, y, yaw

    def _elite_indices(self):
        n = int(self.particles.shape[0])
        fraction = min(max(float(self.pf_config.elite_top_fraction), 1e-6), 1.0)
        k = max(1, int(math.ceil(n * fraction)))
        if self.pf_config.score_is_cost:
            metric = torch.nan_to_num(self.j_totals, nan=float("inf"), posinf=float("inf"))
            return torch.topk(metric, k, largest=False).indices
        metric = torch.nan_to_num(self.scores, nan=-float("inf"), neginf=-float("inf"))
        return torch.topk(metric, k, largest=True).indices

    def estimate_pose(self):
        w = torch.nan_to_num(self.weights, nan=0.0, posinf=0.0, neginf=0.0)
        w_sum = torch.sum(w)
        if (not torch.isfinite(w_sum)) or float(w_sum.item()) <= 1e-12:
            n = int(self.particles.shape[0])
            w = torch.full((n,), 1.0 / max(n, 1), dtype=torch.float32, device=self.device)
        else:
            w = w / w_sum

        x_w = float(torch.sum(w * self.particles[:, 0]).item())
        y_w = float(torch.sum(w * self.particles[:, 1]).item())
        sin_w = float(torch.sum(w * torch.sin(self.particles[:, 2])).item())
        cos_w = float(torch.sum(w * torch.cos(self.particles[:, 2])).item())
        yaw_w = float(math.atan2(sin_w, cos_w))

        elite_idx = self._elite_indices()
        x_e, y_e, yaw_e = self._mean_pose_from_indices(elite_idx, weights=w)
        configured_mode = str(self.pf_config.estimate_mode).strip().lower()
        if configured_mode in ("elite", "top", "top_weighted", "elite_weighted"):
            mode = "elite_weighted"
            x_est, y_est, yaw_est = x_e, y_e, yaw_e
        else:
            mode = "weighted"
            x_est, y_est, yaw_est = x_w, y_w, yaw_w

        if self.pf_config.score_is_cost:
            finite_j = self.j_totals[torch.isfinite(self.j_totals)]
            if finite_j.numel() == 0:
                best_idx = int(torch.argmax(self.scores).item())
                best_metric = float("nan")
            else:
                best_idx = int(torch.argmin(self.j_totals).item())
                best_metric = float(self.j_totals[best_idx].item())
        else:
            best_idx = int(torch.argmax(self.scores).item())
            best_metric = float(self.scores[best_idx].item())

        parts = self.particles.detach().cpu().numpy()
        spread_x = float(np.std(parts[:, 0]))
        spread_y = float(np.std(parts[:, 1]))
        spread_yaw = float(np.std(parts[:, 2]))

        best = self.particles[best_idx]
        finite_scores = self.scores[torch.isfinite(self.scores)]
        if finite_scores.numel() == 0:
            mean_score = min_score = max_score = 0.0
            valid_score_count = 0
        else:
            mean_score = float(finite_scores.mean().item())
            min_score = float(finite_scores.min().item())
            max_score = float(finite_scores.max().item())
            valid_score_count = int(finite_scores.numel())

        finite_j = self.j_totals[torch.isfinite(self.j_totals)]
        if finite_j.numel() == 0:
            j_min = j_max = j_mean = float("nan")
        else:
            j_min = float(finite_j.min().item())
            j_max = float(finite_j.max().item())
            j_mean = float(finite_j.mean().item())

        return {
            "x_est": x_est,
            "y_est": y_est,
            "yaw_est": yaw_est,
            "x_est_weighted": x_w,
            "y_est_weighted": y_w,
            "yaw_est_weighted": yaw_w,
            "x_est_elite": x_e,
            "y_est_elite": y_e,
            "yaw_est_elite": yaw_e,
            "elite_count": int(elite_idx.numel()),
            "estimate_mode": mode,
            "configured_estimate_mode": configured_mode,
            "best_x": float(best[0].item()),
            "best_y": float(best[1].item()),
            "best_yaw": float(best[2].item()),
            "best_score": best_metric,
            "mean_score": mean_score,
            "min_score": min_score,
            "max_score": max_score,
            "neff": self.effective_sample_size(),
            "weight_max": float(torch.max(self.weights).item()),
            "weight_min": float(torch.min(self.weights).item()),
            "weight_mode": str(self.last_weight_debug.get("weight_mode", "")),
            "weight_j_scale": float(self.last_weight_debug.get("weight_j_scale", float("nan"))),
            "weight_j_spread": float(self.last_weight_debug.get("weight_j_spread", float("nan"))),
            "weight_base_scale": float(
                self.last_weight_debug.get("weight_base_scale", float("nan"))
            ),
            "weight_log_range": float(
                self.last_weight_debug.get("weight_log_range", float("nan"))
            ),
            "valid_score_count": valid_score_count,
            "J_min": j_min,
            "J_max": j_max,
            "J_mean": j_mean,
            "spread_x": spread_x,
            "spread_y": spread_y,
            "spread_yaw": spread_yaw,
        }

    def observe_and_resample(self, lidar_layers, timestamp=None, h_max=None):
        self.update(lidar_layers, h_max=h_max)
        estimate = self.estimate_pose()
        threshold = float(self.pf_config.resample_threshold) * int(self.pf_config.num_particles)
        should_resample = float(estimate["neff"]) < threshold
        estimate["resampled"] = bool(should_resample)
        estimate["frame_id"] = self.frame_id
        estimate["timestamp"] = timestamp

        if self.device.type == "cuda":
            estimate["cuda_memory_allocated"] = float(
                torch.cuda.memory_allocated(self.device) / (1024.0 * 1024.0)
            )
            estimate["cuda_memory_reserved"] = float(
                torch.cuda.memory_reserved(self.device) / (1024.0 * 1024.0)
            )
        else:
            estimate["cuda_memory_allocated"] = 0.0
            estimate["cuda_memory_reserved"] = 0.0

        self.last_debug = estimate
        self._save_debug(timestamp, estimate)
        self._save_weight_vis(timestamp, estimate)
        if should_resample:
            self.systematic_resample()
        self.frame_id += 1
        return estimate

    def _weight_vis_output_dir(self):
        vis_dir = str(self.pf_config.weight_vis_dir).strip()
        if not vis_dir:
            vis_dir = "pf_weight_vis"
        if os.path.isabs(vis_dir):
            return vis_dir
        return os.path.join(self.pf_config.debug_dir, vis_dir)

    def _save_debug(self, timestamp, estimate):
        cfg = self.pf_config
        if not cfg.save_debug:
            return

        if cfg.save_score_csv:
            score_path = os.path.join(cfg.debug_dir, "pf_score.csv")
            with open(score_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        timestamp if timestamp is not None else "",
                        estimate.get("frame_id", ""),
                        estimate["x_est"],
                        estimate["y_est"],
                        estimate["yaw_est"],
                        estimate["best_x"],
                        estimate["best_y"],
                        estimate["best_yaw"],
                        estimate["best_score"],
                        estimate["mean_score"],
                        estimate["min_score"],
                        estimate["max_score"],
                        estimate["neff"],
                        int(estimate.get("resampled", False)),
                        estimate["weight_max"],
                        estimate["weight_min"],
                        estimate.get("weight_mode", ""),
                        estimate.get("weight_j_scale", ""),
                        estimate.get("weight_j_spread", ""),
                        estimate.get("weight_base_scale", ""),
                        estimate.get("weight_log_range", ""),
                        estimate["valid_score_count"],
                        estimate.get("cuda_memory_allocated", 0.0),
                        estimate.get("cuda_memory_reserved", 0.0),
                    ]
                )

        if cfg.save_particle_csv:
            particle_path = os.path.join(cfg.debug_dir, "pf_particles.csv")
            particles = self.particles.detach().cpu().numpy()
            weights = self.weights.detach().cpu().numpy()
            scores = self.scores.detach().cpu().numpy()
            j_totals = self.j_totals.detach().cpu().numpy()
            with open(particle_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for pid in range(particles.shape[0]):
                    writer.writerow(
                        [
                            timestamp if timestamp is not None else "",
                            estimate.get("frame_id", ""),
                            pid,
                            particles[pid, 0],
                            particles[pid, 1],
                            particles[pid, 2],
                            scores[pid],
                            j_totals[pid],
                            weights[pid],
                        ]
                    )

    def _save_weight_vis(self, timestamp, estimate):
        cfg = self.pf_config
        if not cfg.save_weight_vis or self._weight_vis_failed:
            return

        frame_id = int(estimate.get("frame_id", self.frame_id))
        every_n = max(1, int(cfg.weight_vis_every_n))
        if frame_id % every_n != 0:
            return

        particles = self.particles.detach().cpu().numpy()
        weights = self.weights.detach().cpu().numpy()
        try:
            out_path = save_particle_weight_view(
                self.dsm_cropper,
                particles,
                weights,
                estimate,
                self._weight_vis_output_dir(),
                timestamp=timestamp,
                return_image=True,
            )
            if isinstance(out_path, tuple):
                out_path, image = out_path
                estimate["weight_vis_image"] = image
            estimate["weight_vis_path"] = out_path
        except Exception as exc:
            self._weight_vis_failed = True
            estimate["weight_vis_error"] = str(exc)
            print("PF weight visualization disabled: {}".format(exc))
