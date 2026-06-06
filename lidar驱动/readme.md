## 激光雷达：

驱动配置方法：进入`/home/xyf/lbk/rslidar_sdk_tuwei/build`下删除文件：

```
cmake ..
make -j8
```

修改配置文件：

`/home/xyf/lbk/rslidar_sdk_tuwei/config`的`config.yaml`文件

其中需要注意的

```
msg_source: 2   #包的来源选择ros而不是1即在线雷达
frame_id: radar_1   #框架选择与毫米波雷达相同的radar_1
roll: 0		#激光雷达的坐标朝向
pitch: 0
yaw: 0
```

原本的topic：

<img src="/home/xyf/文档/typora文档/typora图片/7月河西_雷达联合标定md图/2024-07-16 20-37-21 的屏幕截图.png" alt="2024-07-16 20-37-21 的屏幕截图"  />

使用rviz查看激光雷达点云之前，先运行：

```
cd /home/xyf/Store/lbk/rslidar_sdk_tuwei/build
./rslidar_sdk_node_tuwei
```

出现新的topic，雷达点云在这个topic里面：

![2024-07-16 20-38-36 的屏幕截图](/home/xyf/文档/typora文档/typora图片/7月河西_雷达联合标定md图/2024-07-16 20-38-36 的屏幕截图-17211335635911.png)