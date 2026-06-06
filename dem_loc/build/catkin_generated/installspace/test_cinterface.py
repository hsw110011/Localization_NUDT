#!/usr/bin/env python3
"""Print data received by the Python CInterface subscribers."""

import math
import os
import sys

_PYTHON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

import rospy

from loc_bev.cinterface import CInterface, TOPIC_DEFINITIONS
from loc_bev.common_struct import InputData


def _fmt(value, precision=3):
    if isinstance(value, float):
        if math.isfinite(value):
            return ("{:.%df}" % precision).format(value)
        return str(value)
    return str(value)


def _stamp_summary(header):
    if header is None:
        return ""

    parts = []
    stamp = getattr(header, "stamp", None)
    if stamp is not None and hasattr(stamp, "to_sec"):
        parts.append("stamp={:.3f}".format(stamp.to_sec()))

    frame_id = getattr(header, "frame_id", "")
    if frame_id:
        parts.append("frame_id={}".format(frame_id))

    gps_week = getattr(header, "gps_week", None)
    gps_seconds = getattr(header, "gps_week_seconds", None)
    if gps_week is not None and gps_seconds is not None:
        parts.append("gps_week={}".format(gps_week))
        parts.append("gps_week_seconds={}".format(gps_seconds))

    return ", ".join(parts)


def _len_summary(value):
    try:
        return len(value)
    except TypeError:
        return None


def _array_summary(name, value):
    size = _len_summary(value)
    if size is None:
        return "{}={}".format(name, value)
    if size == 0:
        return "{}=[]".format(name)
    preview = []
    try:
        for item in list(value[:3]):
            preview.append(_fmt(item))
    except TypeError:
        preview = []
    suffix = " first=[{}]".format(", ".join(preview)) if preview else ""
    return "{}_len={}{}".format(name, size, suffix)


def _local_time_parts(msg):
    parts = []
    for name in ("local_time", "UTC_time", "message_num"):
        if hasattr(msg, name):
            parts.append("{}={}".format(name, _fmt(getattr(msg, name))))
    return parts


def _pose2d_summary(label, pose):
    if pose is None:
        return ""
    return "{}=(x={}, y={}, theta={})".format(
        label,
        _fmt(getattr(pose, "x", 0.0)),
        _fmt(getattr(pose, "y", 0.0)),
        _fmt(getattr(pose, "theta", 0.0)),
    )


def _odom_summary(name, msg):
    header = _stamp_summary(getattr(msg, "header", None))
    pose = msg.pose.pose
    p = pose.position
    q = pose.orientation
    twist = msg.twist.twist
    parts = [
        header,
        "child_frame_id={}".format(getattr(msg, "child_frame_id", "")),
        "position=({}, {}, {})".format(_fmt(p.x), _fmt(p.y), _fmt(p.z)),
        "orientation=({}, {}, {}, {})".format(
            _fmt(q.x), _fmt(q.y), _fmt(q.z), _fmt(q.w)
        ),
        "linear=({}, {}, {})".format(
            _fmt(twist.linear.x), _fmt(twist.linear.y), _fmt(twist.linear.z)
        ),
        "angular=({}, {}, {})".format(
            _fmt(twist.angular.x), _fmt(twist.angular.y), _fmt(twist.angular.z)
        ),
    ]
    return "{}: {}".format(name, ", ".join(part for part in parts if part))


def _generic_summary(name, msg):
    parts = _local_time_parts(msg)
    for slot_name in getattr(msg, "__slots__", ()):
        if slot_name in ("local_time", "UTC_time", "message_num"):
            continue
        value = getattr(msg, slot_name)
        if isinstance(value, (int, float, bool, str)):
            parts.append("{}={}".format(slot_name, _fmt(value)))
        elif isinstance(value, (list, tuple, bytes, bytearray)):
            parts.append(_array_summary(slot_name, value))
        elif len(parts) < 8:
            nested_type = value.__class__.__name__
            parts.append("{}=<{}>".format(slot_name, nested_type))
        if len(parts) >= 10:
            break
    return "{}: {}".format(name, ", ".join(parts))


def summarize_message(name, msg):
    if msg is None:
        return "{}: None".format(name)

    if name in ("Odom", "NavsatOdom"):
        return _odom_summary(name, msg)

    if name == "BevMasks":
        parts = [
            _stamp_summary(getattr(msg, "header", None)),
            "height={}".format(msg.height),
            "width={}".format(msg.width),
            "encoding={}".format(msg.encoding),
            "step={}".format(msg.step),
            _array_summary("data", msg.data),
        ]
        return "{}: {}".format(name, ", ".join(part for part in parts if part))

    if name == "KittiGpsFix":
        parts = [
            _stamp_summary(getattr(msg, "header", None)),
            "status={}".format(getattr(msg.status, "status", "")),
            "latitude={}".format(_fmt(msg.latitude, 7)),
            "longitude={}".format(_fmt(msg.longitude, 7)),
            "altitude={}".format(_fmt(msg.altitude)),
        ]
        return "{}: {}".format(name, ", ".join(part for part in parts if part))

    if name == "Inspvax":
        parts = [
            _stamp_summary(getattr(msg, "header", None)),
            "ins_status={}".format(msg.ins_status),
            "position_type={}".format(msg.position_type),
            "latitude={}".format(_fmt(msg.latitude, 7)),
            "longitude={}".format(_fmt(msg.longitude, 7)),
            "altitude={}".format(_fmt(msg.altitude)),
            "azimuth={}".format(_fmt(msg.azimuth)),
        ]
        return "{}: {}".format(name, ", ".join(part for part in parts if part))

    if name == "GlobalPose":
        parts = _local_time_parts(msg)
        parts.extend(
            [
                "gauss=({}, {}, {})".format(
                    _fmt(msg.gaussX), _fmt(msg.gaussY), _fmt(msg.height)
                ),
                "ll=({}, {})".format(_fmt(msg.latitude, 7), _fmt(msg.longitude, 7)),
                "rpy=({}, {}, {})".format(
                    _fmt(msg.roll), _fmt(msg.pitch), _fmt(msg.azimuth)
                ),
            ]
        )
        return "{}: {}".format(name, ", ".join(parts))

    if name == "LocalPose":
        parts = _local_time_parts(msg)
        parts.extend(
            [
                "dr=({}, {}, {})".format(_fmt(msg.dr_x), _fmt(msg.dr_y), _fmt(msg.dr_z)),
                "rph=({}, {}, {})".format(
                    _fmt(msg.dr_roll), _fmt(msg.dr_pitch), _fmt(msg.dr_heading)
                ),
                "vehicle_speed={}".format(_fmt(msg.vehicle_speed)),
            ]
        )
        return "{}: {}".format(name, ", ".join(parts))

    if name == "LidarLocalPose":
        parts = _local_time_parts(msg)
        parts.extend(
            [
                "pos_type={}".format(msg.pos_type),
                "xyz=({}, {}, {})".format(_fmt(msg.x), _fmt(msg.y), _fmt(msg.z)),
                "azimuth={}".format(_fmt(msg.azimuth)),
            ]
        )
        return "{}: {}".format(name, ", ".join(parts))

    if name == "ReferencePath":
        parts = _local_time_parts(msg)
        if hasattr(msg, "path_source"):
            parts.append("path_source={}".format(msg.path_source))
        parts.append("effectPointsNum={}".format(msg.effectPointsNum))
        parts.append(_array_summary("Points", msg.Points))
        return "{}: {}".format(name, ", ".join(parts))

    if name == "EntityMap":
        parts = _local_time_parts(msg)
        parts.extend(
            [
                "pedestrianNum={}".format(msg.pedestrianNum),
                "vehicleNum={}".format(msg.vehicleNum),
                "generalNum={}".format(msg.generalNum),
            ]
        )
        return "{}: {}".format(name, ", ".join(parts))

    if name == "ColorMap":
        parts = _local_time_parts(msg)
        parts.extend(
            [
                _pose2d_summary("localPoseStamped", msg.localPoseStamped),
                _array_summary("r", msg.r),
                _array_summary("g", msg.g),
                _array_summary("b", msg.b),
                "residual=({}, {})".format(_fmt(msg.residual_x), _fmt(msg.residual_y)),
            ]
        )
        return "{}: {}".format(name, ", ".join(part for part in parts if part))

    if name == "TerrainMap":
        parts = _local_time_parts(msg)
        parts.extend(
            [
                _array_summary("label", msg.label),
                _array_summary("slope", msg.slope),
                _array_summary("roughness", msg.roughness),
                "residual=({}, {})".format(_fmt(msg.residual_x), _fmt(msg.residual_y)),
            ]
        )
        return "{}: {}".format(name, ", ".join(parts))

    if name == "SemanticMap":
        parts = _local_time_parts(msg)
        parts.extend(
            [
                _array_summary("semantic", msg.semantic),
                "residual=({}, {})".format(_fmt(msg.residual_x), _fmt(msg.residual_y)),
            ]
        )
        return "{}: {}".format(name, ", ".join(parts))

    if name == "SimilarityMap":
        parts = _local_time_parts(msg)
        parts.extend(
            [
                _pose2d_summary("lidarlocalPoseStamped", msg.lidarlocalPoseStamped),
                _array_summary("similarity", msg.similarity),
            ]
        )
        return "{}: {}".format(name, ", ".join(part for part in parts if part))

    return _generic_summary(name, msg)


def _trim(text, max_chars):
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... <truncated, set _max_full_chars:=0 for all data>"


def main():
    rospy.init_node("test_cinterface_py", anonymous=True)

    rate_hz = rospy.get_param("~rate", 2.0)
    print_full_message = rospy.get_param("~print_full_message", False)
    print_only_changed = rospy.get_param("~print_only_changed", True)
    max_full_chars = rospy.get_param("~max_full_chars", 4000)
    once = rospy.get_param("~once", False)

    interface = CInterface()
    input_data = InputData()
    rate = rospy.Rate(rate_hz)
    last_wait_log = rospy.Time.now()

    rospy.loginfo("Listening to %d topics with Python CInterface.", len(TOPIC_DEFINITIONS))
    rospy.loginfo("Use _print_full_message:=true to print raw ROS message bodies.")

    while not rospy.is_shutdown():
        interface.ConvertToLocalData(input_data)
        printed = False

        for field_name, topic, _msg_type, _callback, _subscriber, flag_name in TOPIC_DEFINITIONS:
            if print_only_changed and not getattr(input_data, flag_name, False):
                continue

            msg = getattr(input_data, field_name)
            if print_full_message:
                body = _trim(str(msg), max_full_chars)
            else:
                body = summarize_message(field_name, msg)
            rospy.loginfo("[%s] %s\n%s", field_name, topic, body)
            printed = True

        if once and printed:
            return

        now = rospy.Time.now()
        if not printed and (now - last_wait_log).to_sec() >= 5.0:
            rospy.loginfo("Waiting for subscribed topic data...")
            last_wait_log = now

        rate.sleep()


if __name__ == "__main__":
    main()
