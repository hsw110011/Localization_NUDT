#pragma once
#include "SemLocConfig.h"
#include <torch/torch.h>

namespace sem_loc {
// 三种评分及路由均实现在 ParticleFilter.cpp；输入为 NCHW，最后一通道为有效观测 mask。
torch::Tensor ObservationWeights(const torch::Tensor& observation, ScoreMode mode);
torch::Tensor ComputeParticleCosts(const torch::Tensor& tdf, const torch::Tensor& grids,
                                   const torch::Tensor& weights);
// TDF 已归一化到 [0,1]；对应通道逐元素相乘后直接求和，越界取 TDF=1。
// CPU 逐粒子 OpenCV 采样。pixel_poses: Nx3 float64 [col, row, heading_deg]。
torch::Tensor ComputeForloopParticleCosts(const cv::Mat& tdf, const torch::Tensor& pixel_poses,
                                          const torch::Tensor& weights);
}
