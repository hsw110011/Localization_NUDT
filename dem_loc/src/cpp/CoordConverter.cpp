#include "CoordConverter.h"
#include "Tool.h"
#include <cmath>


CoordConverter::CoordConverter(const DemTiffData& dem_tiff_data)
{
    // 预计算椭球参数
    double f = WGS84_F;
    e2_ = 2 * f - f * f;
    e1_ = (1 - std::sqrt(1 - e2_)) / (1 + std::sqrt(1 - e2_));

    map_width_ = dem_tiff_data.visual_map.cols;
    map_height_ = dem_tiff_data.visual_map.rows;
    zone_num_ = dem_tiff_data.zone_num;
    tl_lon_ = dem_tiff_data.geo_bounds[0];
    tl_lat_ = dem_tiff_data.geo_bounds[1];
    br_lon_ = dem_tiff_data.geo_bounds[2];
    br_lat_ = dem_tiff_data.geo_bounds[3];
    res_ = dem_tiff_data.resolution;    

    res_lon_ = (br_lon_ - tl_lon_) / map_width_;
    res_lat_ = (tl_lat_ - br_lat_) / map_height_;
    cout << "res_lon_: " << res_lon_ << ", res_lat_: " << res_lat_ << endl;
    GaussPoint gauss_tl = wgs84_to_gauss(tl_lon_, tl_lat_);
}

// ==========================================
// 1. WGS84 <-> Gauss 实现
// ==========================================

GaussPoint CoordConverter::wgs84_to_gauss(double lon, double lat) 
{
    double lon_rad = lon * DEG_TO_RAD;
    double lat_rad = lat * DEG_TO_RAD;

    // 确定中央经线
    // 使用类成员变量 zone_num_，确保与地图投影带一致，而不是根据每点的经度动态计算
    // 这样可以处理跨越投影带边缘的地图（强行投影到同一个带上，保持平面连续性）
    int zone = int((lon + 180) / 6) + 1; 
    double cm_deg = -180.0 + (zone * 6.0) - 3.0;
    double cm_rad = cm_deg * DEG_TO_RAD;

    double a = WGS84_A; 
    double k0 = UTM_SCALE_FACTOR;
    double e2 = e2_;
    double ee = e2 / (1 - e2);

    double sin_lat = std::sin(lat_rad);
    double cos_lat = std::cos(lat_rad);
    double tan_lat = std::tan(lat_rad);

    // 卯酉圈曲率半径 (N)
    double N = a / std::sqrt(1 - e2 * sin_lat * sin_lat);

    double T = tan_lat * tan_lat;
    double C = ee * cos_lat * cos_lat;
    double A = (lon_rad - cm_rad) * cos_lat;

    // 子午线弧长 (M)
    double M = a * ((1 - e2 / 4 - 3 * e2 * e2 / 64 - 5 * std::pow(e2, 3) / 256) * lat_rad
                    - (3 * e2 / 8 + 3 * e2 * e2 / 32 + 45 * std::pow(e2, 3) / 1024) * std::sin(2 * lat_rad)
                    + (15 * e2 * e2 / 256 + 45 * std::pow(e2, 3) / 1024) * std::sin(4 * lat_rad)
                    - (35 * std::pow(e2, 3) / 3072) * std::sin(6 * lat_rad));

    // 计算 Easting (X)
    double x_val = k0 * N * (A + (1 - T + C) * std::pow(A, 3) / 6.0 +
                             (5 - 18 * T + T * T + 72 * C - 58 * ee) * std::pow(A, 5) / 120.0);

    // 计算 Northing (Y)
    double y_val = k0 * (M + N * tan_lat * (std::pow(A, 2) / 2.0 +
                                            (5 - T + 9 * C + 4 * C * C) * std::pow(A, 4) / 24.0 +
                                            (61 - 58 * T + T * T + 600 * C - 330 * ee) * std::pow(A, 6) / 720.0));

    // False Easting
    GaussPoint gauss;
    gauss.x = x_val + 500000.0;
    gauss.y = y_val;
    gauss.z = 0.0;
    return gauss;
}

BLH_Point CoordConverter::gauss_to_wgs84(double gauss_x, double gauss_y) 
{
    double x_val = gauss_x - 500000.0; // 去掉 False Easting
    
    double a = WGS84_A;
    double k0 = UTM_SCALE_FACTOR;
    double e2 = e2_;
    double e1 = e1_;
    
    double M = gauss_y / k0;
    double mu = M / (a * (1 - e2 / 4 - 3 * e2 * e2 / 64 - 5 * std::pow(e2, 3) / 256));

    // 底点纬度 phi1
    double phi1 = mu + (3 * e1 / 2 - 27 * std::pow(e1, 3) / 32) * std::sin(2 * mu) +
                  (21 * e1 * e1 / 16 - 55 * std::pow(e1, 4) / 32) * std::sin(4 * mu) +
                  (151 * std::pow(e1, 3) / 96) * std::sin(6 * mu);

    double ee = e2 / (1 - e2);
    double C1 = ee * std::pow(std::cos(phi1), 2);
    double T1 = std::pow(std::tan(phi1), 2);
    double N1 = a / std::sqrt(1 - e2 * std::pow(std::sin(phi1), 2));
    double R1 = a * (1 - e2) / std::pow(1 - e2 * std::pow(std::sin(phi1), 2), 1.5);
    double D = x_val / (N1 * k0);

    // 计算 Lat
    double lat_rad = phi1 - (N1 * std::tan(phi1) / R1) * (std::pow(D, 2) / 2 -
                     (5 + 3 * T1 + 10 * C1 - 4 * C1 * C1 - 9 * ee) * std::pow(D, 4) / 24 +
                     (61 + 90 * T1 + 298 * C1 + 45 * T1 * T1 - 252 * ee - 3 * C1 * C1) * std::pow(D, 6) / 720);

    // 计算 Lon
    double lon_rad = (D - (1 + 2 * T1 + C1) * std::pow(D, 3) / 6 +
                      (5 - 2 * C1 + 28 * T1 - 3 * C1 * C1 + 8 * ee + 24 * T1 * T1) * std::pow(D, 5) / 120) / std::cos(phi1);

    // 使用构造时确定的中央经线
    double cm_deg = -180.0 + (zone_num_ * 6.0) - 3.0;
    
    double final_lat = lat_rad * RAD_TO_DEG;
    double final_lon = cm_deg + (lon_rad * RAD_TO_DEG);
    BLH_Point blh;
    blh.Lat = final_lat;
    blh.Lon = final_lon;
    blh.Height = 0.0;
    return blh;
}

// ==========================================
// 2. Gauss <-> Pixel 实现
// ==========================================

cv::Point2f CoordConverter::gauss_to_pixel(double gauss_x, double gauss_y)
{
    double u = (gauss_x - tl_gaussX_) / res_;
    double v = (tl_gaussY_ - gauss_y) / res_;
    return cv::Point2f(static_cast<float>(u), static_cast<float>(v));
}

GaussPoint CoordConverter::pixel_to_gauss(double u, double v)
{
    GaussPoint gauss;
    gauss.x = tl_gaussX_ + u * res_;
    gauss.y = tl_gaussY_ - v * res_;
    gauss.z = 0.0; // 默认平面 z=0
    return gauss;
}

// ==========================================
// 3. WGS84 <-> Pixel 实现 (组合)
// ==========================================

cv::Point2f CoordConverter::wgs84_to_pixel(double lon, double lat)
{
    cv::Point2f pixel;
    pixel.x = (lon- tl_lon_)/ res_lon_;
    pixel.y = (tl_lat_ - lat)/ res_lat_;
    return pixel;
}

BLH_Point CoordConverter::pixel_to_wgs84(double u, double v)
{
    BLH_Point blh;
    blh.Lon = tl_lon_ + u * res_lon_;
    blh.Lat = tl_lat_ - v * res_lat_;
    blh.Height = 0.0;
    return blh;
}



// ==========================================
// 1. WGS84 -> Gauss (GPU Tensor版)
// ==========================================
std::pair<torch::Tensor, torch::Tensor> CoordConverter::wgs84_to_gauss_gpu(const torch::Tensor& lon, const torch::Tensor& lat) 
{
    // 确保计算在输入 Tensor 所在的设备上进行
    auto device = lon.device();
    auto dtype = lon.dtype();

    // 1. 角度转弧度
    auto lon_rad = lon * DEG_TO_RAD;
    auto lat_rad = lat * DEG_TO_RAD;

    // 2. 确定中央经线 (Vectorized)
    // CPU: int zone = int((lon + 180) / 6) + 1;
    // GPU: 对应 float 操作，使用 floor
    auto zone = torch::floor((lon + 180.0) / 6.0) + 1.0;
    auto cm_deg = -180.0 + (zone * 6.0) - 3.0;
    auto cm_rad = cm_deg * DEG_TO_RAD;

    // 3. 准备参数 (标量会自动广播)
    double a = WGS84_A; 
    double k0 = UTM_SCALE_FACTOR;
    double e2 = e2_;
    double ee = e2 / (1.0 - e2);

    auto sin_lat = torch::sin(lat_rad);
    auto cos_lat = torch::cos(lat_rad);
    auto tan_lat = torch::tan(lat_rad);

    // 4. 卯酉圈曲率半径 (N)
    // N = a / sqrt(1 - e2 * sin^2)
    auto N = a / torch::sqrt(1.0 - e2 * torch::pow(sin_lat, 2));

    auto T = torch::pow(tan_lat, 2);
    auto C = ee * torch::pow(cos_lat, 2);
    auto A = (lon_rad - cm_rad) * cos_lat;

    // 5. 子午线弧长 (M) - 这是一个长多项式
    // 为了性能和可读性，将系数提取
    double c1 = 1.0 - e2 / 4.0 - 3.0 * std::pow(e2, 2) / 64.0 - 5.0 * std::pow(e2, 3) / 256.0;
    double c2 = 3.0 * e2 / 8.0 + 3.0 * std::pow(e2, 2) / 32.0 + 45.0 * std::pow(e2, 3) / 1024.0;
    double c3 = 15.0 * std::pow(e2, 2) / 256.0 + 45.0 * std::pow(e2, 3) / 1024.0;
    double c4 = 35.0 * std::pow(e2, 3) / 3072.0;

    auto M = a * (c1 * lat_rad
                  - c2 * torch::sin(2.0 * lat_rad)
                  + c3 * torch::sin(4.0 * lat_rad)
                  - c4 * torch::sin(6.0 * lat_rad));

    // 6. 计算 Easting (X)
    auto A2 = torch::pow(A, 2);
    auto A3 = torch::pow(A, 3);
    auto A4 = torch::pow(A, 4);
    auto A5 = torch::pow(A, 5);
    auto A6 = torch::pow(A, 6); // 预计算幂次

    auto x_val = k0 * N * (A + (1.0 - T + C) * A3 / 6.0 +
                           (5.0 - 18.0 * T + T * T + 72.0 * C - 58.0 * ee) * A5 / 120.0);

    // 7. 计算 Northing (Y)
    auto y_val = k0 * (M + N * tan_lat * (A2 / 2.0 +
                                          (5.0 - T + 9.0 * C + 4.0 * C * C) * A4 / 24.0 +
                                          (61.0 - 58.0 * T + T * T + 600.0 * C - 330.0 * ee) * A6 / 720.0));

    // 8. False Easting
    auto gauss_x = x_val + 500000.0;
    auto gauss_y = y_val;

    return {gauss_x, gauss_y};
}

// ==========================================
// Gauss -> WGS84 (GPU Tensor版)
// ==========================================
std::pair<torch::Tensor, torch::Tensor> CoordConverter::gauss_to_wgs84_gpu(const torch::Tensor& gauss_x, const torch::Tensor& gauss_y) 
{
    // 去掉 False Easting
    auto x_val = gauss_x - 500000.0; 
    
    double a = WGS84_A;
    double k0 = UTM_SCALE_FACTOR;
    double e2 = e2_;
    double e1 = e1_;
    
    auto M = gauss_y / k0;

    double mu_denom = a * (1.0 - e2 / 4.0 - 3.0 * std::pow(e2, 2) / 64.0 - 5.0 * std::pow(e2, 3) / 256.0);
    auto mu = M / mu_denom;

    // 底点纬度 phi1
    double c1 = 3.0 * e1 / 2.0 - 27.0 * std::pow(e1, 3) / 32.0;
    double c2 = 21.0 * std::pow(e1, 2) / 16.0 - 55.0 * std::pow(e1, 4) / 32.0;
    double c3 = 151.0 * std::pow(e1, 3) / 96.0;

    auto phi1 = mu + c1 * torch::sin(2.0 * mu) +
                     c2 * torch::sin(4.0 * mu) +
                     c3 * torch::sin(6.0 * mu);

    double ee = e2 / (1.0 - e2);
    
    // 三角函数缓存
    auto sin_phi1 = torch::sin(phi1);
    auto cos_phi1 = torch::cos(phi1);
    auto tan_phi1 = torch::tan(phi1);
    
    auto C1 = ee * torch::pow(cos_phi1, 2);
    auto T1 = torch::pow(tan_phi1, 2);
    auto N1 = a / torch::sqrt(1.0 - e2 * torch::pow(sin_phi1, 2));
    auto R1 = a * (1.0 - e2) / torch::pow(1.0 - e2 * torch::pow(sin_phi1, 2), 1.5);
    auto D = x_val / (N1 * k0);

    // 预计算 D 的幂次
    auto D2 = torch::pow(D, 2);
    auto D3 = torch::pow(D, 3);
    auto D4 = torch::pow(D, 4);
    auto D5 = torch::pow(D, 5);
    auto D6 = torch::pow(D, 6);

    // 计算 Lat
    auto lat_rad = phi1 - (N1 * tan_phi1 / R1) * (D2 / 2.0 -
                     (5.0 + 3.0 * T1 + 10.0 * C1 - 4.0 * C1 * C1 - 9.0 * ee) * D4 / 24.0 +
                     (61.0 + 90.0 * T1 + 298.0 * C1 + 45.0 * T1 * T1 - 252.0 * ee - 3.0 * C1 * C1) * D6 / 720.0);

    // 计算 Lon
    auto lon_rad_diff = (D - (1.0 + 2.0 * T1 + C1) * D3 / 6.0 +
                      (5.0 - 2.0 * C1 + 28.0 * T1 - 3.0 * C1 * C1 + 8.0 * ee + 24.0 * T1 * T1) * D5 / 120.0) / cos_phi1;

    // 使用构造时确定的中央经线 (zone_num_)
    double cm_deg = -180.0 + (zone_num_ * 6.0) - 3.0;
    
    auto final_lat = lat_rad * RAD_TO_DEG;
    auto final_lon = cm_deg + (lon_rad_diff * RAD_TO_DEG);

    return {final_lon, final_lat};
}


// ==========================================
// 2. Gauss -> Pixel (GPU Tensor版)
// ==========================================
std::pair<torch::Tensor, torch::Tensor> CoordConverter::gauss_to_pixel_gpu(const torch::Tensor& gauss_x, const torch::Tensor& gauss_y)
{
    // double u = (gauss_x - tl_gaussX_) / res_;
    // double v = (tl_gaussY_ - gauss_y) / res_;
    
    // LibTorch 会自动处理 tensor - scalar 的广播
    auto u = (gauss_x - tl_gaussX_) / res_;
    auto v = (tl_gaussY_ - gauss_y) / res_;
    
    // 如果需要转成 float32 (类似 cv::Point2f)，可以加 .to(torch::kFloat32)
    // 这里保持精度，返回与输入相同的类型
    return {u, v};
}

// ==========================================
// Pixel -> Gauss (GPU Tensor版)
// ==========================================
std::pair<torch::Tensor, torch::Tensor> CoordConverter::pixel_to_gauss_gpu(const torch::Tensor& u, const torch::Tensor& v)
{
    // gauss.x = tl_gaussX_ + u * res_;
    // gauss.y = tl_gaussY_ - v * res_;

    auto gauss_x = tl_gaussX_ + u * res_;
    auto gauss_y = tl_gaussY_ - v * res_;
    
    return {gauss_x, gauss_y};
}



// ==========================================
// 3. WGS84 -> Pixel (GPU Tensor版) - 线性近似
// ==========================================
std::pair<torch::Tensor, torch::Tensor> CoordConverter::wgs84_to_pixel_gpu(const torch::Tensor& lon, const torch::Tensor& lat)
{
    // pixel.x = (lon - tl_lon_) / res_lon_;
    // pixel.y = (tl_lat_ - lat) / res_lat_;

    auto u = (lon - tl_lon_) / res_lon_;
    auto v = (tl_lat_ - lat) / res_lat_;

    return {u, v};
}

// ==========================================
// Pixel -> WGS84 (GPU Tensor版) - 线性近似
// ==========================================
std::pair<torch::Tensor, torch::Tensor> CoordConverter::pixel_to_wgs84_gpu(const torch::Tensor& u, const torch::Tensor& v)
{
    // blh.Lon = tl_lon_ + u * res_lon_;
    // blh.Lat = tl_lat_ - v * res_lat_;

    auto lon = tl_lon_ + u * res_lon_;
    auto lat = tl_lat_ - v * res_lat_;

    return {lon, lat};
}