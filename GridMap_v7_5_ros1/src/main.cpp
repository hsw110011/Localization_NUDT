#include "ros/ros.h"
#include "map/sensor_map.h"

int main(int argc, char *argv[])
{
    setlocale(LC_ALL,"");
    //节点初始化
    ros::init(argc,argv,"GradMap");
    
    SensorMap sensor_map;
    sensor_map.output();
    return 0;
}
