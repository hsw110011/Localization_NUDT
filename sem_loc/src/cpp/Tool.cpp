#include "Tool.h"
#include "cnpy.h"
Tool::Tool()
{
}
// C++ 原生读取特定的 NPZ 数据
bool Tool::LoadSatelliteNpz(const std::string& npz_path, SatelliteData& out_data)
{
    try {
        // 1. 加载整个压缩包
        std::map<std::string, cnpy::NpyArray> arrays = cnpy::npz_load(npz_path);

        // 2. 读取 Semantic Map (H, W, 4)
        if (arrays.count("semantic_map"))
        {
            cnpy::NpyArray& arr = arrays["semantic_map"];
            int h = arr.shape[0];
            int w = arr.shape[1];
            int c = arr.shape.size() > 2 ? arr.shape[2] : 1;

            // 根据数据类型构造 Mat
            if (arr.word_size == 1) // uint8
            {
                out_data.semantic_map = cv::Mat(h, w, CV_8UC(c));
                memcpy(out_data.semantic_map.data, arr.data<unsigned char>(), arr.num_bytes());
            }
            else if (arr.word_size == 4) // float32
            {
                out_data.semantic_map = cv::Mat(h, w, CV_32FC(c));
                memcpy(out_data.semantic_map.data, arr.data<float>(), arr.num_bytes());
            }
            else
            {
                std::cout << "[Warning] semantic_map unknown dtype size: " << arr.word_size << std::endl;
            }
            std::cout << "[Info] Loaded semantic_map: " << out_data.semantic_map.size() << " Channels: " << out_data.semantic_map.channels() << std::endl;
        }

        // 3. 读取 TDF Map (H, W, N)
        if (arrays.count("tdf_map"))
        {
            cnpy::NpyArray& arr = arrays["tdf_map"];
            int h = arr.shape[0];
            int w = arr.shape[1];
            int c = arr.shape.size() > 2 ? arr.shape[2] : 1;

            // TDF 通常是 float
            out_data.tdf_map = cv::Mat(h, w, CV_32FC(c));
            if (arr.word_size == 4)
            {
                memcpy(out_data.tdf_map.data, arr.data<float>(), arr.num_bytes());
            }
            else if (arr.word_size == 8) // double
            {
                 // OpenCV 默认常用 float32 处理数据，如果源是非常高精度 double，可能需要转 CV_64F
                 cv::Mat temp(h, w, CV_64FC(c));
                 memcpy(temp.data, arr.data<double>(), arr.num_bytes());
                 temp.convertTo(out_data.tdf_map, CV_32F); // 统一转为 float32 方便后续处理
            }
            std::cout << "[Info] Loaded tdf_map: " << out_data.tdf_map.size() << " Channels: " << out_data.tdf_map.channels() << std::endl;
        }

        // 4. 读取 Geo Bounds
        if (arrays.count("geo_bounds"))
        {
            cnpy::NpyArray& arr = arrays["geo_bounds"];
            size_t num = arr.shape[0]; // 应该是一维数组
            out_data.geo_bounds.resize(num);

            if (arr.word_size == 8) // double
            {
                double* data = arr.data<double>();
                for(size_t i=0; i<num; i++) out_data.geo_bounds[i] = data[i];
            }
            else if (arr.word_size == 4)  // float
            {
                float* data = arr.data<float>();
                for(size_t i=0; i<num; i++) out_data.geo_bounds[i] = (double)data[i];
            }

            std::cout << "[Info] Loaded geo_bounds: [";
            for(auto v : out_data.geo_bounds) std::cout << v << " ";
            std::cout << "]" << std::endl;
        }

        // 5. 读取 Gauss Origin X
        if (arrays.count("Origin_X"))
        {
            cnpy::NpyArray& arr = arrays["Origin_X"];
            if (arr.word_size == 8) out_data.origin_x = arr.data<double>()[0];
            else if (arr.word_size == 4) out_data.origin_x = (double)arr.data<float>()[0];
            std::cout << "[Info] Loaded Origin_X: " << out_data.origin_x << std::endl;
        }

        // 6. 读取 Gauss Origin Y
        if (arrays.count("Origin_Y"))
        {
            cnpy::NpyArray& arr = arrays["Origin_Y"];
            if (arr.word_size == 8) out_data.origin_y = arr.data<double>()[0];
            else if (arr.word_size == 4) out_data.origin_y = (double)arr.data<float>()[0];
            std::cout << "[Info] Loaded Origin_Y: " << out_data.origin_y << std::endl;
        }

        // 7. 读取 Resolution
        if (arrays.count("Resolution"))
        {
            cnpy::NpyArray& arr = arrays["Resolution"];
            if (arr.word_size == 8) out_data.resolution = arr.data<double>()[0];
            else if (arr.word_size == 4) out_data.resolution = (double)arr.data<float>()[0];
            std::cout << "[Info] Loaded Resolution: " << out_data.resolution << std::endl;
        }

        // 8. 读取 Zone Num
        if (arrays.count("Zone_Num"))
        {
            cnpy::NpyArray& arr = arrays["Zone_Num"];
            // Zone Num 通常是 int，但 numpy 可能存成 long/int64
            if (arr.word_size == 8) out_data.zone_num = (int)arr.data<long>()[0]; // int64
            else if (arr.word_size == 4) out_data.zone_num = arr.data<int>()[0];   // int32
            std::cout << "[Info] Loaded Zone_Num: " << out_data.zone_num << std::endl;
        }

        // 9. 读取 Satellite Map (H, W, 3) - RGB/BGR
        if (arrays.count("satellite_map"))
        {
            cnpy::NpyArray& arr = arrays["satellite_map"];
            int h = arr.shape[0];
            int w = arr.shape[1];
            int c = arr.shape.size() > 2 ? arr.shape[2] : 1;

            // 卫星图通常是 uint8 BGR
            if (arr.word_size == 1) // uint8
            {
                out_data.satellite_map = cv::Mat(h, w, CV_8UC(c));
                memcpy(out_data.satellite_map.data, arr.data<unsigned char>(), arr.num_bytes());
            }
            else
            {
                std::cout << "[Warning] satellite_map unknown dtype size: " << arr.word_size << std::endl;
            }
            std::cout << "[Info] Loaded satellite_map: " << out_data.satellite_map.size() << " Channels: " << out_data.satellite_map.channels() << std::endl;
        }

        out_data.is_valid = !out_data.semantic_map.empty();
        return true;

    }
    catch (const std::exception& e)
    {
        std::cerr << "[Error] Load NPZ failed: " << e.what() << std::endl;
        return false;
    }
}


        /*所以这个找局部坐标系原点在全局坐标位置的函数*/
double3D Tool::GetBase(const nav_msgs::Odometry *odom , WORLD_POINT globalpoint)
{
    double3D base;

    double siny_cosp = 2.0 * (odom->pose.pose.orientation.w * odom->pose.pose.orientation.z + odom->pose.pose.orientation.x * odom->pose.pose.orientation.y);
    double cosy_cosp = 1.0 - 2.0 * (odom->pose.pose.orientation.y * odom->pose.pose.orientation.y + odom->pose.pose.orientation.z * odom->pose.pose.orientation.z);
    double odom_heading = atan2(siny_cosp, cosy_cosp) * RAD_TO_DEG; // CCW,

    double global_math_heading = globalpoint.heading;    //输入就是角度，单位degree，坐标系(东向为0度)，bag中的amzith

    double odom_x = odom->pose.pose.position.x;
    double odom_y = odom->pose.pose.position.y;

    base.theta = (global_math_heading - odom_heading)*DEG_TO_RAD;
    base.x = globalpoint.gauss.x-(odom_x*cos(base.theta)-odom_y*sin(base.theta));
    base.y = globalpoint.gauss.y-(odom_x*sin(base.theta)+odom_y*cos(base.theta));
    return base;                                                                //得到的是gauss坐标系下的基点
}

WORLD_POINT Tool::LocalToGlobal(const nav_msgs::Odometry *odom,double3D BasePoint)
{
    WORLD_POINT global_point;
    global_point.timeflag = odom->header.stamp.toSec() * 1000.0;

    double siny_cosp = 2.0 * (odom->pose.pose.orientation.w * odom->pose.pose.orientation.z + odom->pose.pose.orientation.x * odom->pose.pose.orientation.y);
    double cosy_cosp = 1.0 - 2.0 * (odom->pose.pose.orientation.y * odom->pose.pose.orientation.y + odom->pose.pose.orientation.z * odom->pose.pose.orientation.z);
    double odom_heading = atan2(siny_cosp, cosy_cosp) * RAD_TO_DEG;

    double odom_x = odom->pose.pose.position.x;
    double odom_y = odom->pose.pose.position.y;

    // 1. Transform Position
    global_point.gauss.x = odom_x* cos(BasePoint.theta)-odom_y*sin(BasePoint.theta)+ BasePoint.x;
    global_point.gauss.y = odom_x* sin(BasePoint.theta)+odom_y*cos(BasePoint.theta)+ BasePoint.y;

    // 2. Transform Heading
    // Global_Math (CCW) = Local_Math (CCW) + Theta (CCW)
    double global_math_heading = odom_heading + (BasePoint.theta)*RAD_TO_DEG;

    global_point.heading = global_math_heading;

    // Normalize to 0-360
    while(global_point.heading < 0) global_point.heading += 360.0;
    while(global_point.heading >= 360.0) global_point.heading -= 360.0;

    return global_point;
}

