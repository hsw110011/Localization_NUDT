#ifndef CINTERFACE_H
#define CINTERFACE_H
/*
    Coder:BuYafeng
 */
#include <CommonStruct.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/NavSatFix.h>
#include <novatel_msgs/INSPVAX.h>

class CInterface
{
public:
    CInterface(ros::NodeHandle nh);
    ~CInterface();
    bool ConvertToLocalData(InputData *input);


private:
    void LocalPoseCallback(const self_state::LocalPose::ConstPtr &msg);
    void LidarLocalPoseCallback(const self_state::LidarLocalPose::ConstPtr &msg);
    void GlobalPoseCallback(const self_state::GlobalPose::ConstPtr &msg);
    void ColorMapCallback(const world_state::ColorMapConstPtr& msg );
    void EntityMapCallback(const world_state::EntityMap::ConstPtr &msg);
    void TerrainMapCallback(const world_state::TerrainMap::ConstPtr &msg);
    void SemanticMapCallback(const world_state::SemanticMap::ConstPtr &msg);
    void SimilarityMapCallback(const world_state::SimilarityMap::ConstPtr &msg);
    void ReferencePathCallback(const behavior::ReferencePath::ConstPtr &msg);

    // New Callbacks
    void OdomCallback(const nav_msgs::Odometry::ConstPtr &msg);
    void BevMasksCallback(const sensor_msgs::Image::ConstPtr &msg);
    void InspvaxCallback(const novatel_msgs::INSPVAX::ConstPtr &msg);
    void NavsatOdomCallback(const nav_msgs::Odometry::ConstPtr &msg);
    void KittiGpsFixCallback(const sensor_msgs::NavSatFix::ConstPtr &msg);

    ros::Subscriber ColorMapSub;
    ros::Subscriber EntityMapSub;
    ros::Subscriber TerrainMapSub;
    ros::Subscriber SemanticMapSub;
    ros::Subscriber SimilarityMapSub;
    ros::Subscriber LocalPoseSub;
    ros::Subscriber LidarLocalPoseSub;
    ros::Subscriber GlobalPoseSub;
    ros::Subscriber ImgSub;
    ros::Subscriber ReferencePathSub;
    
    // New Subscribers
    ros::Subscriber OdomSub;
    ros::Subscriber BevMasksSub;
    ros::Subscriber InspvaxSub;
    ros::Subscriber NavsatOdomSub;
    ros::Subscriber KittiGpsFixSub;


    world_state::ColorMap *ColorMap;

    world_state::SimilarityMap *SimilarityMap;
    /// ROS input DATA!!!
    world_state::EntityMap *EntityMap;
    /// ROS input DATA!!!
    world_state::TerrainMap *TerrainMap;
    /// ROS input DATA!!!
    world_state::SemanticMap *SemanticMap;

    behavior::ReferencePath *ReferencePath;

    /// ROS input DATA!!!
    self_state::GlobalPose *GlobalPose;
    /// ROS input DATA!!!
    self_state::LocalPose  *LocalPose;

    self_state::LocalPose *CurrentLocalPose;

    self_state::LidarLocalPose *LidarLocalPose;

    // New Data pointers
    nav_msgs::Odometry *Odom;
    sensor_msgs::Image *BevMasks;
    novatel_msgs::INSPVAX *Inspvax;
    nav_msgs::Odometry *NavsatOdom;
    sensor_msgs::NavSatFix *KittiGpsFix;


    ros::NodeHandle nh;

    bool LidarLocalPose_refreshflag;
    bool LocalPose_refreshflag;
    bool GlobalPose_refreshflag;

    bool SematicMap_refreshflag;
    bool ColorMap_refreshflag;
    bool SimilarityMap_refreshflag;
    bool TerrainMap_refreshflag;
    bool EntityMap_refreshflag;
    bool ReferencePath_refreshflag;

    // New Refresh Flags
    bool Odom_refreshflag;
    bool BevMasks_refreshflag;
    bool Inspvax_refreshflag;
    bool NavsatOdom_refreshflag;
    bool KittiGpsFix_refreshflag;

    std::mutex mutex_LocalPose;
    std::mutex mutex_cloud;
    std::mutex mutex_Img;
};

#endif // CINTERFACE_H
