#!/usr/bin/env python3
"""rospy version of src/include/CInterface.h and src/cpp/CInterface.cpp."""

import threading

import rospy
from behavior.msg import ReferencePath as ReferencePathMsg
from nav_msgs.msg import Odometry
from novatel_msgs.msg import INSPVAX
from self_state.msg import GlobalPose as GlobalPoseMsg
from self_state.msg import LidarLocalPose as LidarLocalPoseMsg
from self_state.msg import LocalPose as LocalPoseMsg
from sensor_msgs.msg import Image, NavSatFix
from world_state.msg import ColorMap as ColorMapMsg
from world_state.msg import EntityMap as EntityMapMsg
from world_state.msg import SemanticMap as SemanticMapMsg
from world_state.msg import SimilarityMap as SimilarityMapMsg
from world_state.msg import TerrainMap as TerrainMapMsg

from .common_struct import INPUT_DATA_FIELDS, INPUT_REFRESH_FIELDS, InputData


TOPIC_DEFINITIONS = (
    (
        "GlobalPose",
        "/self_state/GlobalPose",
        GlobalPoseMsg,
        "GlobalPoseCallback",
        "GlobalPoseSub",
        "GlobalPose_refreshflag",
    ),
    (
        "LocalPose",
        "/self_state/LocalPose",
        LocalPoseMsg,
        "LocalPoseCallback",
        "LocalPoseSub",
        "LocalPose_refreshflag",
    ),
    (
        "LidarLocalPose",
        "/self_state/LidarLocalPose",
        LidarLocalPoseMsg,
        "LidarLocalPoseCallback",
        "LidarLocalPoseSub",
        "LidarLocalPose_refreshflag",
    ),
    (
        "ColorMap",
        "/world_state/ColorMap",
        ColorMapMsg,
        "ColorMapCallback",
        "ColorMapSub",
        "ColorMap_refreshflag",
    ),
    (
        "EntityMap",
        "/world_state/EntityMap_false",
        EntityMapMsg,
        "EntityMapCallback",
        "EntityMapSub",
        "EntityMap_refreshflag",
    ),
    (
        "TerrainMap",
        "/world_state/TerrainMap",
        TerrainMapMsg,
        "TerrainMapCallback",
        "TerrainMapSub",
        "TerrainMap_refreshflag",
    ),
    (
        "SemanticMap",
        "/world_state/SemanticMap",
        SemanticMapMsg,
        "SemanticMapCallback",
        "SemanticMapSub",
        "SematicMap_refreshflag",
    ),
    (
        "SimilarityMap",
        "/world_state/SimilarityMap",
        SimilarityMapMsg,
        "SimilarityMapCallback",
        "SimilarityMapSub",
        "SimilarityMap_refreshflag",
    ),
    (
        "ReferencePath",
        "/behavior/ReferencePath",
        ReferencePathMsg,
        "ReferencePathCallback",
        "ReferencePathSub",
        "ReferencePath_refreshflag",
    ),
    (
        "Odom",
        "/Odometry",
        Odometry,
        "OdomCallback",
        "OdomSub",
        "Odom_refreshflag",
    ),
    (
        "BevMasks",
        "/Semantic_Bev/ClassMask",
        Image,
        "BevMasksCallback",
        "BevMasksSub",
        "BevMasks_refreshflag",
    ),
    (
        "Inspvax",
        "/kitti/oxts/gps/inspvax",
        INSPVAX,
        "InspvaxCallback",
        "InspvaxSub",
        "Inspvax_refreshflag",
    ),
    (
        "NavsatOdom",
        "/navsat/odom",
        Odometry,
        "NavsatOdomCallback",
        "NavsatOdomSub",
        "NavsatOdom_refreshflag",
    ),
    (
        "KittiGpsFix",
        "/kitti/oxts/gps/fix",
        NavSatFix,
        "KittiGpsFixCallback",
        "KittiGpsFixSub",
        "KittiGpsFix_refreshflag",
    ),
)


class CInterface(object):
    """Python subscriber interface with the same ConvertToLocalData call style."""

    def __init__(self, nh=None, queue_size=1):
        self.nh = nh
        self._lock = threading.RLock()
        self._subscribers = []

        for field_name, _topic, msg_type, _cb, _sub, _flag in TOPIC_DEFINITIONS:
            setattr(self, field_name, msg_type())
        self.CurrentLocalPose = LocalPoseMsg()

        for flag_name in INPUT_REFRESH_FIELDS:
            setattr(self, flag_name, False)

        for (
            _field_name,
            topic,
            msg_type,
            callback_name,
            subscriber_name,
            _flag_name,
        ) in TOPIC_DEFINITIONS:
            subscriber = rospy.Subscriber(
                topic,
                msg_type,
                getattr(self, callback_name),
                queue_size=queue_size,
            )
            setattr(self, subscriber_name, subscriber)
            self._subscribers.append(subscriber)

    def _store_message(self, field_name, flag_name, msg):
        with self._lock:
            setattr(self, field_name, msg)
            setattr(self, flag_name, True)

    def GlobalPoseCallback(self, msg):
        self._store_message("GlobalPose", "GlobalPose_refreshflag", msg)

    def LocalPoseCallback(self, msg):
        self._store_message("LocalPose", "LocalPose_refreshflag", msg)

    def LidarLocalPoseCallback(self, msg):
        self._store_message("LidarLocalPose", "LidarLocalPose_refreshflag", msg)

    def ColorMapCallback(self, msg):
        self._store_message("ColorMap", "ColorMap_refreshflag", msg)

    def EntityMapCallback(self, msg):
        self._store_message("EntityMap", "EntityMap_refreshflag", msg)

    def TerrainMapCallback(self, msg):
        self._store_message("TerrainMap", "TerrainMap_refreshflag", msg)

    def SemanticMapCallback(self, msg):
        self._store_message("SemanticMap", "SematicMap_refreshflag", msg)

    def SimilarityMapCallback(self, msg):
        self._store_message("SimilarityMap", "SimilarityMap_refreshflag", msg)

    def ReferencePathCallback(self, msg):
        self._store_message("ReferencePath", "ReferencePath_refreshflag", msg)

    def OdomCallback(self, msg):
        self._store_message("Odom", "Odom_refreshflag", msg)

    def BevMasksCallback(self, msg):
        self._store_message("BevMasks", "BevMasks_refreshflag", msg)

    def InspvaxCallback(self, msg):
        self._store_message("Inspvax", "Inspvax_refreshflag", msg)

    def NavsatOdomCallback(self, msg):
        self._store_message("NavsatOdom", "NavsatOdom_refreshflag", msg)

    def KittiGpsFixCallback(self, msg):
        self._store_message("KittiGpsFix", "KittiGpsFix_refreshflag", msg)

    def ConvertToLocalData(self, input_data):
        if input_data is None:
            raise ValueError("input_data must be an InputData instance")

        with self._lock:
            for field_name in INPUT_DATA_FIELDS:
                setattr(input_data, field_name, getattr(self, field_name, None))

            for flag_name in INPUT_REFRESH_FIELDS:
                setattr(input_data, flag_name, getattr(self, flag_name, False))

            for flag_name in INPUT_REFRESH_FIELDS:
                setattr(self, flag_name, False)

        return True


def make_input_data():
    return InputData()
