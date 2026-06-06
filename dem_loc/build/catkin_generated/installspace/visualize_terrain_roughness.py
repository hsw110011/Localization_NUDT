#!/usr/bin/env python3
"""Subscribe to TerrainMap and visualize its roughness grid."""

import math
import os
import threading

import cv2
import numpy as np
import rospy
from world_state.msg import TerrainMap


def _as_uint8_array(value):
    if isinstance(value, (bytes, bytearray)):
        return np.frombuffer(value, dtype=np.uint8).copy()
    return np.asarray(value, dtype=np.uint8)


def _infer_grid_shape(size, width, height):
    if width > 0 and height > 0:
        if width * height != size:
            raise ValueError(
                "roughness size {} does not match width*height {}*{}".format(
                    size, width, height
                )
            )
        return height, width

    side = int(math.sqrt(size))
    if side * side == size:
        return side, side

    return 1, size


def _build_display_image(grid, auto_contrast, color_map_name):
    if auto_contrast:
        min_value = int(grid.min())
        max_value = int(grid.max())
        if max_value > min_value:
            image = ((grid.astype(np.float32) - min_value) * 255.0)
            image = (image / float(max_value - min_value)).clip(0, 255).astype(np.uint8)
        else:
            image = np.zeros_like(grid, dtype=np.uint8)
    else:
        image = grid

    color_map_name = color_map_name.lower()
    if color_map_name == "gray":
        return image

    color_maps = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "bone": cv2.COLORMAP_BONE,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }
    if hasattr(cv2, "COLORMAP_TURBO"):
        color_maps["turbo"] = cv2.COLORMAP_TURBO

    if color_map_name not in color_maps:
        rospy.logwarn_once(
            "Unknown color_map '%s', using gray. Valid values: %s",
            color_map_name,
            ", ".join(sorted(color_maps.keys()) + ["gray"]),
        )
        return image

    return cv2.applyColorMap(image, color_maps[color_map_name])


class TerrainRoughnessViewer(object):
    def __init__(self):
        self.topic = rospy.get_param("~topic", "/world_state/TerrainMap")
        self.width = int(rospy.get_param("~width", 0))
        self.height = int(rospy.get_param("~height", 0))
        self.display_scale = float(rospy.get_param("~display_scale", 1.0))
        self.auto_contrast = bool(rospy.get_param("~auto_contrast", False))
        self.color_map = rospy.get_param("~color_map", "gray")
        self.print_full = bool(rospy.get_param("~print_full", False))
        self.print_first_n = int(rospy.get_param("~print_first_n", 20))
        self.print_every_n = max(1, int(rospy.get_param("~print_every_n", 1)))
        self.save_dir = rospy.get_param("~save_dir", "")
        self.save_every_n = int(rospy.get_param("~save_every_n", 0))
        self.show_window = bool(rospy.get_param("~show_window", True))

        if self.show_window and os.name != "nt" and not os.environ.get("DISPLAY"):
            rospy.logwarn("DISPLAY is not set; disabling OpenCV window.")
            self.show_window = False

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._latest = None
        self._latest_index = 0
        self._displayed_index = 0
        self._received_count = 0

        self._subscriber = rospy.Subscriber(
            self.topic, TerrainMap, self._callback, queue_size=1
        )

        if self.show_window:
            cv2.namedWindow("TerrainMap roughness", cv2.WINDOW_AUTOSIZE)

    def _callback(self, msg):
        roughness = _as_uint8_array(msg.roughness)
        try:
            rows, cols = _infer_grid_shape(roughness.size, self.width, self.height)
        except ValueError as exc:
            rospy.logerr("%s", exc)
            return

        grid = roughness.reshape((rows, cols))
        image = _build_display_image(grid, self.auto_contrast, self.color_map)
        if self.display_scale > 0.0 and self.display_scale != 1.0:
            image = cv2.resize(
                image,
                None,
                fx=self.display_scale,
                fy=self.display_scale,
                interpolation=cv2.INTER_NEAREST,
            )

        with self._lock:
            self._received_count += 1
            self._latest_index = self._received_count
            self._latest = (msg, roughness, grid, image, rows, cols, self._latest_index)

    def spin(self):
        rate = rospy.Rate(float(rospy.get_param("~rate", 30.0)))
        rospy.loginfo("Subscribing %s and visualizing TerrainMap.roughness", self.topic)

        while not rospy.is_shutdown():
            item = None
            with self._lock:
                if self._latest and self._latest_index != self._displayed_index:
                    item = self._latest
                    self._displayed_index = self._latest_index

            if item is not None:
                msg, roughness, grid, image, rows, cols, index = item
                self._print_roughness(msg, roughness, grid, rows, cols, index)
                self._save_image(image, index)

                if self.show_window:
                    cv2.imshow("TerrainMap roughness", image)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        rospy.signal_shutdown("viewer closed by keyboard")
            elif self.show_window:
                cv2.waitKey(1)

            rate.sleep()

        if self.show_window:
            cv2.destroyAllWindows()

    def _print_roughness(self, msg, roughness, grid, rows, cols, index):
        if index % self.print_every_n != 0:
            return

        local_pose = msg.localPoseStamped
        summary = (
            "roughness frame #{index}: shape={rows}x{cols}, size={size}, "
            "min={min_value}, max={max_value}, mean={mean:.3f}, "
            "local_time={local_time:.3f}, UTC_time={utc_time:.3f}, "
            "localPose=(x={x:.3f}, y={y:.3f}, theta={theta:.3f}), "
            "residual=({rx:.3f}, {ry:.3f})"
        ).format(
            index=index,
            rows=rows,
            cols=cols,
            size=roughness.size,
            min_value=int(grid.min()),
            max_value=int(grid.max()),
            mean=float(grid.mean()),
            local_time=float(msg.local_time),
            utc_time=float(msg.UTC_time),
            x=float(local_pose.x),
            y=float(local_pose.y),
            theta=float(local_pose.theta),
            rx=float(msg.residual_x),
            ry=float(msg.residual_y),
        )
        rospy.loginfo(summary)

        if self.print_first_n > 0:
            first_values = roughness[: self.print_first_n].tolist()
            rospy.loginfo("roughness first %d values: %s", self.print_first_n, first_values)

        if self.print_full:
            rospy.loginfo("roughness values: %s", roughness.tolist())

    def _save_image(self, image, index):
        if not self.save_dir or self.save_every_n <= 0:
            return
        if index % self.save_every_n != 0:
            return

        path = os.path.join(self.save_dir, "roughness_{:06d}.png".format(index))
        if cv2.imwrite(path, image):
            rospy.loginfo("Saved roughness image: %s", path)
        else:
            rospy.logerr("Failed to save roughness image: %s", path)


def main():
    rospy.init_node("visualize_terrain_roughness", anonymous=True)
    viewer = TerrainRoughnessViewer()
    viewer.spin()


if __name__ == "__main__":
    main()
