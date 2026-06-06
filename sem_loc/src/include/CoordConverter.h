#ifndef COORD_CONVERTER_H
#define COORD_CONVERTER_H

#include <cmath>
#include <tuple>
#include "CommonStruct.h"
#include "Tool.h"

struct SatelliteData; // Forward declaration

class CoordConverter 
{
public:

    
    // 新增：直接使用 SatelliteData 构造
    CoordConverter(const SatelliteData& map_data);
   
    // WGS84 <-> Gauss
    GaussPoint wgs84_to_gauss(double lon, double lat);
    BLH_Point gauss_to_wgs84(double gauss_x, double gauss_y);

    // Gauss <-> Pixel (OpenCV)
    cv::Point2f gauss_to_pixel(double gauss_x, double gauss_y);
    GaussPoint pixel_to_gauss(double u, double v);

    // WGS84 <-> Pixel
    cv::Point2f wgs84_to_pixel(double lon, double lat);
    BLH_Point pixel_to_wgs84(double u, double v);

    std::pair<torch::Tensor, torch::Tensor> wgs84_to_gauss_gpu(const torch::Tensor& lon, const torch::Tensor& lat);
    std::pair<torch::Tensor, torch::Tensor> gauss_to_wgs84_gpu(const torch::Tensor& gauss_x, const torch::Tensor& gauss_y);

    // 2. Gauss <-> Pixel (GPU)
    std::pair<torch::Tensor, torch::Tensor> gauss_to_pixel_gpu(const torch::Tensor& gauss_x, const torch::Tensor& gauss_y);
    std::pair<torch::Tensor, torch::Tensor> pixel_to_gauss_gpu(const torch::Tensor& u, const torch::Tensor& v);

    // 3. WGS84 <-> Pixel (GPU) - 线性近似
    std::pair<torch::Tensor, torch::Tensor> wgs84_to_pixel_gpu(const torch::Tensor& lon, const torch::Tensor& lat);
    std::pair<torch::Tensor, torch::Tensor> pixel_to_wgs84_gpu(const torch::Tensor& u, const torch::Tensor& v);


private:
    // 常量定义
    static constexpr double WGS84_A = 6378137.0;
    static constexpr double WGS84_F = 1.0 / 298.257223563;
    static constexpr double UTM_SCALE_FACTOR = 0.9996;

    // 地图参数
    double tl_lon_, tl_lat_;
    double br_lon_, br_lat_;
    double res_; // 米/像素 resolution

    int map_width_, map_height_;
    double res_lon_; // resolution (经纬度)
    double res_lat_;
    double tl_gaussX_;
    double tl_gaussY_;
    int zone_num_;

    // 预计算的椭球参数
    double e2_; // 第一偏心率平方
    double e1_; // 第二偏心率相关参数
};

#endif // COORD_CONVERTER_H