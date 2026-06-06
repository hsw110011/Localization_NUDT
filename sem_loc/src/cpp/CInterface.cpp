#include "CInterface.h"
#include <stdio.h>
CInterface::CInterface(ros::NodeHandle _nh)
{
    nh = _nh;

//    GlobalPoseSub = nh.subscribe("/self_state/GlobalPose",1,&CInterface::GlobalPoseCallback,this);

    GlobalPoseSub = nh.subscribe("/self_state/GlobalPose",1,&CInterface::GlobalPoseCallback,this);
    LocalPoseSub = nh.subscribe("/self_state/LocalPose",1,&CInterface::LocalPoseCallback,this);
    LidarLocalPoseSub = nh.subscribe("/self_state/LidarLocalPose",1,&CInterface::LidarLocalPoseCallback,this);
    ColorMapSub = nh.subscribe("/world_state/ColorMap",1,&CInterface::ColorMapCallback,this);
    EntityMapSub = nh.subscribe("/world_state/EntityMap_false",1,&CInterface::EntityMapCallback,this);
    TerrainMapSub = nh.subscribe("/world_state/TerrainMap",1,&CInterface::TerrainMapCallback,this);
    SemanticMapSub = nh.subscribe("/world_state/SemanticMap",1,&CInterface::SemanticMapCallback,this);
    SimilarityMapSub = nh.subscribe("/world_state/SimilarityMap",1,&CInterface::SimilarityMapCallback,this);
    ReferencePathSub = nh.subscribe("/behavior/ReferencePath",1,&CInterface::ReferencePathCallback, this);

    // New Subscribers
    OdomSub = nh.subscribe("/Odometry", 1, &CInterface::OdomCallback, this);
    BevMasksSub = nh.subscribe("/Semantic_Bev/ClassMask", 1, &CInterface::BevMasksCallback, this);
    BevProbsSub = nh.subscribe("/Semantic_Bev/Probs", 1, &CInterface::BevProbsCallback, this);
    // InspvaxSub = nh.subscribe("/novatel_data/inspvax", 1, &CInterface::InspvaxCallback, this);
    InspvaxSub = nh.subscribe("/kitti/oxts/gps/inspvax", 1, &CInterface::InspvaxCallback, this);
    NavsatOdomSub = nh.subscribe("/navsat/odom", 1, &CInterface::NavsatOdomCallback, this);
    KittiGpsFixSub = nh.subscribe("/kitti/oxts/gps/fix", 1, &CInterface::KittiGpsFixCallback, this);

    LocalPose = new self_state::LocalPose;
    LidarLocalPose = new self_state::LidarLocalPose;
    CurrentLocalPose = new self_state::LocalPose;
    GlobalPose = new self_state::GlobalPose;

    ColorMap = new world_state::ColorMap;
    EntityMap= new world_state::EntityMap;
    TerrainMap= new world_state::TerrainMap;
    SemanticMap= new world_state::SemanticMap;
    SimilarityMap= new world_state::SimilarityMap;
    ReferencePath  = new behavior::ReferencePath;

    // Allocate memory for new data
    Odom = new nav_msgs::Odometry;
    BevMasks = new sensor_msgs::Image;
    BevProbs = new sensor_msgs::Image;
    Inspvax = new novatel_msgs::INSPVAX;
    NavsatOdom = new nav_msgs::Odometry;
    KittiGpsFix = new sensor_msgs::NavSatFix;

    SematicMap_refreshflag=false;
    ColorMap_refreshflag=false;
    SimilarityMap_refreshflag=false;
    TerrainMap_refreshflag=false;
    EntityMap_refreshflag=false;
    LocalPose_refreshflag=false;
    GlobalPose_refreshflag=false;
    LidarLocalPose_refreshflag=false;
    ReferencePath_refreshflag = false;

    // Initialize new refresh flags
    Odom_refreshflag = false;
    BevMasks_refreshflag = false;
    BevProbs_refreshflag = false;
    Inspvax_refreshflag = false;
    NavsatOdom_refreshflag = false;
    KittiGpsFix_refreshflag = false;

}
CInterface::~CInterface()
{
    if(LocalPose)        delete LocalPose;
    if(GlobalPose)       delete GlobalPose;
    if(LidarLocalPose)   delete LidarLocalPose;
    if(CurrentLocalPose) delete CurrentLocalPose;

    if(ColorMap) delete ColorMap;
    if(EntityMap) delete EntityMap;
    if(TerrainMap) delete TerrainMap;
    if(SimilarityMap) delete SimilarityMap;
    if(SemanticMap)   delete SemanticMap;
    if(ReferencePath)  delete ReferencePath;

    // Delete new data
    if(Odom) delete Odom;
    if(BevProbs) delete BevProbs;
    if(BevMasks) delete BevMasks;
    if(Inspvax) delete Inspvax;
    if(NavsatOdom) delete NavsatOdom;
    if(KittiGpsFix) delete KittiGpsFix;

}
void CInterface::SimilarityMapCallback(const world_state::SimilarityMap::ConstPtr &msg)
{
    *SimilarityMap = *msg;
    SimilarityMap_refreshflag=true;
}

void CInterface::SemanticMapCallback(const world_state::SemanticMap::ConstPtr &msg)
{
    *SemanticMap = *msg;
    SematicMap_refreshflag=true;
}

void CInterface::ReferencePathCallback(const behavior::ReferencePath::ConstPtr &msg)
{
    *ReferencePath = *msg;
    cout << "###  进入回调  ###" << endl;
    ReferencePath_refreshflag = true;
}

void CInterface::TerrainMapCallback(const world_state::TerrainMap::ConstPtr &msg)
{
    *TerrainMap = *msg;
    TerrainMap_refreshflag=true;
}

void CInterface::EntityMapCallback(const world_state::EntityMap::ConstPtr &msg)
{
    *EntityMap = *msg;
    EntityMap_refreshflag=true;
}

void CInterface::ColorMapCallback(const world_state::ColorMapConstPtr &msg){
    *ColorMap=*msg;
    ColorMap_refreshflag=true;
}

void CInterface::GlobalPoseCallback(const self_state::GlobalPose::ConstPtr &msg)
{
    *GlobalPose = *msg;
    GlobalPose_refreshflag = true;
}
void CInterface::LocalPoseCallback(const self_state::LocalPose::ConstPtr &msg){
    *LocalPose = *msg;
    LocalPose_refreshflag = true;

}
void CInterface::LidarLocalPoseCallback(const self_state::LidarLocalPose::ConstPtr &msg){
    *LidarLocalPose = *msg;
    LidarLocalPose_refreshflag = true;

}

// New Callback Implementations
void CInterface::OdomCallback(const nav_msgs::Odometry::ConstPtr &msg) {
    *Odom = *msg;
    Odom_refreshflag = true;
}

void CInterface::BevMasksCallback(const sensor_msgs::Image::ConstPtr &msg) {
    *BevMasks = *msg;
    BevMasks_refreshflag = true;
}

void CInterface::BevProbsCallback(const sensor_msgs::Image::ConstPtr &msg) {
    *BevProbs = *msg;
    BevProbs_refreshflag = true;
}   


void CInterface::InspvaxCallback(const novatel_msgs::INSPVAX::ConstPtr &msg) {
    *Inspvax = *msg;
    Inspvax_refreshflag = true;
}

void CInterface::NavsatOdomCallback(const nav_msgs::Odometry::ConstPtr &msg) {
    *NavsatOdom = *msg;
    NavsatOdom_refreshflag = true;
}

void CInterface::KittiGpsFixCallback(const sensor_msgs::NavSatFix::ConstPtr &msg) {
    *KittiGpsFix = *msg;
    KittiGpsFix_refreshflag = true;
}


bool CInterface::ConvertToLocalData(InputData *input)
{
    input->LocalPose = LocalPose;
    input->LidarLocalPose = LidarLocalPose;
    input->ColorMap = ColorMap;
    input->GlobalPose = GlobalPose;
    input->TerrainMap = TerrainMap;
    input->BevProbs = BevProbs;
    input->SemanticMap = SemanticMap;
    input->SimilarityMap=SimilarityMap;
    input->ReferencePath = ReferencePath;   
    
    // Copy newly collected pointers
    input->Odom = Odom;
    input->BevMasks = BevMasks;
    input->Inspvax = Inspvax;
    input->NavsatOdom = NavsatOdom;
    input->KittiGpsFix = KittiGpsFix;


    input->LocalPose_refreshflag = LocalPose_refreshflag;
    input->LidarLocalPose_refreshflag = LidarLocalPose_refreshflag;
    input->GlobalPose_refreshflag = GlobalPose_refreshflag;
    input->ColorMap_refreshflag = ColorMap_refreshflag;
    input->EntityMap_refreshflag = EntityMap_refreshflag;
    input->BevProbs_refreshflag = BevProbs_refreshflag;
    input->SematicMap_refreshflag = SematicMap_refreshflag;
    input->SimilarityMap_refreshflag = SimilarityMap_refreshflag;
    input->TerrainMap_refreshflag = TerrainMap_refreshflag;
    input->ReferencePath_refreshflag = ReferencePath_refreshflag;

    // Copy new refresh flags
    input->Odom_refreshflag = Odom_refreshflag;
    input->BevMasks_refreshflag = BevMasks_refreshflag;
    input->Inspvax_refreshflag = Inspvax_refreshflag;
    input->NavsatOdom_refreshflag = NavsatOdom_refreshflag;
    input->KittiGpsFix_refreshflag = KittiGpsFix_refreshflag;


    LocalPose_refreshflag = false;
    LidarLocalPose_refreshflag = false;
    GlobalPose_refreshflag = false;
    ColorMap_refreshflag = false;
    EntityMap_refreshflag = false;
    SematicMap_refreshflag = false;
    SimilarityMap_refreshflag = false;
    TerrainMap_refreshflag = false;
    ReferencePath_refreshflag = false;

    // Reset new refresh flags
    Odom_refreshflag = false;
    BevMasks_refreshflag = false;
    BevProbs_refreshflag = false;
    Inspvax_refreshflag = false;
    NavsatOdom_refreshflag = false;
    KittiGpsFix_refreshflag = false;


    return true;
}

