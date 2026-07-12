#ifndef COMMONSTRUCT_H
#define COMMONSTRUCT_H
#include "ros/ros.h"
#include "self_state/GlobalPose.h"
#include "self_state/LocalPose.h"
#include "self_state/LidarLocalPose.h"
#include "nav_msgs/Odometry.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/NavSatFix.h"
#include "novatel_msgs/INSPVAX.h"
#include "world_state/ColorMap.h"
#include "world_state/EntityMap.h"
#include "world_state/SemanticMap.h"
#include "world_state/TerrainMap.h"
#include "world_state/SimilarityMap.h"
#include "behavior/ReferencePath.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include <sys/unistd.h>
#include <torch/script.h>
#include <torch/csrc/api/include/torch/torch.h>
#include <fstream>


#define MAP_RESOLUTION 0.15
#define PATH_RESOLUTION 1.0
#define BASE_X  19695752.27
#define BASE_Y  3125228.07

#define ZONEWIDE_XG 6.0001
#define ZONEWIDE 6.0

using namespace std;

#ifndef DEG_TO_RAD
#define DEG_TO_RAD (M_PI / 180.0)
#endif
#ifndef RAD_TO_DEG
#define RAD_TO_DEG (180.0 / M_PI)
#endif

typedef struct {
    double Lon;
    double Lat;
    double Height;
}BLH_Point;
typedef struct{
    double x;
    double y;
    double z;
}GaussPoint;


struct double3D
{
    double x;
    double y;
    double theta;
};

struct double2D
{
    double x;
    double y;
};


struct PointxyRGB{
    float x;
    float y;
    float z;
    cv::Vec3b color;
};//lidarpoint

struct InputData{
    world_state::ColorMap *ColorMap;
    world_state::SimilarityMap *SimilarityMap;
    world_state::EntityMap *EntityMap;
    world_state::TerrainMap *TerrainMap;
    world_state::SemanticMap *SemanticMap;
    self_state::GlobalPose *GlobalPose;
    self_state::LocalPose  *LocalPose;
    self_state::LidarLocalPose *LidarLocalPose;
    behavior::ReferencePath *ReferencePath;

    // New inputs
    nav_msgs::Odometry *Odom;
    sensor_msgs::Image *BevMasks;
    sensor_msgs::Image *BevProbs;
    novatel_msgs::INSPVAX *Inspvax;
    nav_msgs::Odometry *NavsatOdom;
    sensor_msgs::NavSatFix *KittiGpsFix;


    bool GlobalPose_refreshflag;
    bool LocalPose_refreshflag;
    bool LidarLocalPose_refreshflag;
    bool PointCloud_refreshflag;
    bool SematicMap_refreshflag;
    bool ColorMap_refreshflag;
    bool SimilarityMap_refreshflag;
    bool TerrainMap_refreshflag;
    bool EntityMap_refreshflag;
    bool ReferencePath_refreshflag;

    // New refresh flags
    bool Odom_refreshflag;
    bool BevMasks_refreshflag;
    bool BevProbs_refreshflag;
    bool Inspvax_refreshflag;
    bool NavsatOdom_refreshflag;
    bool KittiGpsFix_refreshflag;

//    InputData() {
//        ColorMap = nullptr;
//        SimilarityMap = nullptr;
//        EntityMap = nullptr;
//        TerrainMap = nullptr;
//        SemanticMap = nullptr;
//        GlobalPose = nullptr;
//        LocalPose = nullptr;
//        LidarLocalPose = nullptr;
//        TaskUpdate = nullptr;
//    }
};

struct Pose
{
    Pose(double x_=0.,double y_=0.,double yaw_=0.) {x = x_;y=y_;yaw=yaw_;}
    double x;
    double y;
    double yaw;
};

struct Pointxyz{
    float x;
    float y;
    float z;
};//lidarpoint

typedef struct{
    double timeflag;
    BLH_Point BLH;
    GaussPoint gauss;
    cv::Point pixel;
    double heading;
    double distance;
    double std_var;
}WORLD_POINT;

struct Paras{
    string mapPath;
    double topLeftX;
    double topLeftY;
    double lowRightX;
    double lowRightY;
    int speedUpperLimits;
    int PathResource;
};

struct MAP_INFO{
    WORLD_POINT left_top;
    WORLD_POINT right_down;
    double gauss_x_resolution;
    double gauss_y_resolution;
    double BLH_lon_resolution;
    double BLH_lat_resolution;
    int rows;
    int cols;
};

typedef struct{
    double delta_x;
    double delta_y;
    double delta_theta;
    double scale;
    double score;
}MATCH_STATE;

typedef struct{
    WORLD_POINT                 path_center;
    int                         path_index;
    WORLD_POINT                 vehicle_frozen_global;
    self_state::LocalPose       vehicle_frozen_local;
    self_state::LidarLocalPose  vehicle_frozen_lidar;
    WORLD_POINT                 vehicle_frozen_state;
    MATCH_STATE                 match_state;
    WORLD_POINT                 map_position;
    double                      match_time;
    int                         site_type;
    double                      std_var;
    WORLD_POINT                 diff_pathl;
}MATCH_SITE;


class CommonStruct
{
public:
    CommonStruct();
};

#endif // COMMONSTRUCT_H
