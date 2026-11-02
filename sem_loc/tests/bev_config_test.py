#!/usr/bin/env python3
"""在独立 ROS master 上验证直接读取 config 的 BEV 模式、通道、EMA 和配置校验。

用法：source ROS 环境后运行 python3 bev_pipeline_test.py /path/to/bev
"""
import math
import copy
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import xmlrpc.client

import numpy as np


BEV_BINARY = str(Path(sys.argv.pop(1)).resolve())


class BevPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logs = tempfile.TemporaryDirectory(prefix="sem_loc_ros_test_")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        os.environ["ROS_MASTER_URI"] = f"http://127.0.0.1:{port}"
        os.environ["ROS_HOSTNAME"] = "127.0.0.1"
        os.environ["ROS_LOG_DIR"] = cls.logs.name
        cls.master_log = open(Path(cls.logs.name) / "master.log", "w")
        cls.master = subprocess.Popen(["roscore", "-p", str(port)], stdout=cls.master_log, stderr=subprocess.STDOUT)
        with xmlrpc.client.ServerProxy(os.environ["ROS_MASTER_URI"]) as master:
            for _ in range(100):
                try:
                    if master.getPid("sem_loc_test")[0] == 1:
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                cls.master.terminate()
                cls.master.wait(timeout=10)
                raise RuntimeError("test ROS master did not start")

        global rospy, Odometry, PointCloud2, PointField, Image, point_cloud2
        import rospy
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import PointCloud2, PointField, Image
        from sensor_msgs import point_cloud2
        rospy.init_node("sem_loc_pipeline_test", disable_signals=True)

    @classmethod
    def tearDownClass(cls):
        rospy.signal_shutdown("test complete")
        cls.master.terminate()
        cls.master.wait(timeout=10)
        cls.master_log.close()
        cls.logs.cleanup()

    def setUp(self):
        self.config = {
            "bev": {"mode": "uncertain", "resolution": 1.0, "view_range": 8.0,
                    "visualize": 0, "probability_channels": [0, 1, 2], "input_cloud_in_world": 0,
                    "history_frames": 10, "surface_band_m": 0.5,
                    "accumulation_update_distance_m": 0.0, "accumulation_update_yaw_deg": 0.0,
                    "accumulation_max_interval_sec": 0.0},
            "particle_filter": {"method": "ours_entropy", "backend": "gpu",
                                "score_lambda": 0.001, "tdf_value_scale": 100.0},
        }
        self.config_path = Path(self.logs.name) / (self._testMethodName + ".yaml")
        self.node_log = open(Path(self.logs.name) / f"{self._testMethodName}.log", "w+")
        self.node = None
        self.subscriber = None
        self.messages = []
        self.odom_pub = rospy.Publisher("/Odometry", Odometry, queue_size=1, latch=True)
        self.cloud_pub = rospy.Publisher("/cloud_registered_body", PointCloud2, queue_size=1)

    def tearDown(self):
        if self.subscriber:
            self.subscriber.unregister()
        if self.node and self.node.poll() is None:
            self.node.terminate()
            self.node.wait(timeout=10)
        self.odom_pub.unregister()
        self.cloud_pub.unregister()
        self.node_log.close()

    def start_node(self, certain=False, odom_x=0.0):
        if certain:
            self.config["bev"]["mode"] = "certain"
            self.config["particle_filter"]["method"] = "baseline"
        topic = "/Semantic_Bev/ClassMask" if certain else "/Semantic_Bev/Probs"
        self.subscriber = rospy.Subscriber(topic, Image, self.messages.append, queue_size=10)
        self.write_config()
        self.node = subprocess.Popen([BEV_BINARY, "_config:=" + str(self.config_path),
                                      "__name:=" + self._testMethodName,
                                      "/lio/odom:=/Odometry", "/lio/cloud_world:=/cloud_registered_body"],
                                     stdout=self.node_log, stderr=subprocess.STDOUT)
        for _ in range(100):
            if self.node.poll() is not None:
                self.node_log.seek(0)
                self.fail(self.node_log.read())
            if (self.cloud_pub.get_num_connections() and self.odom_pub.get_num_connections()
                    and self.subscriber.get_num_connections()):
                break
            time.sleep(0.05)
        else:
            self.fail("BEV input/output connections did not become ready")
        odom = Odometry()
        odom.header.frame_id = "world"
        odom.header.stamp = rospy.Time.now()
        odom.pose.pose.position.x = odom_x
        odom.pose.pose.orientation.z = math.sin(math.pi / 4)
        odom.pose.pose.orientation.w = math.cos(math.pi / 4)
        self.odom_template = odom
        self.odom_pub.publish(odom)
        time.sleep(0.1)

    def cloud(self, probs=(0.7, 0.2, 0.1), label=1, z=1.0, include_probs=True, x=0.0, frame="base_link", points=None, stamp=None):
        from std_msgs.msg import Header
        fields = [PointField("x", 0, PointField.FLOAT32, 1),
                  PointField("y", 4, PointField.FLOAT32, 1),
                  PointField("z", 8, PointField.FLOAT32, 1),
                  PointField("rgb", 12, PointField.UINT32, 1),
                  PointField("label", 16, PointField.UINT32, 1)]
        point = (x, 0.0, z, 0x112233, label)
        if include_probs:
            fields.append(PointField("merged_probs", 20, PointField.FLOAT32, 3))
            point += probs
        return point_cloud2.create_cloud(Header(stamp=stamp or rospy.Time.now(), frame_id=frame),
                                         fields, points if points is not None else [point])

    def send(self, cloud, sync_pose=True):
        if sync_pose:
            odom = copy.deepcopy(self.odom_template)
            odom.header.stamp = cloud.header.stamp
            self.odom_pub.publish(odom)
        self.cloud_pub.publish(cloud)

    def output(self, cloud, certain=False, sync_pose=True):
        self.messages.clear()
        self.send(cloud, sync_pose)
        time.sleep(0.2)  # 等待点云回调和下一次 10 Hz 渲染完成。
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            if self.messages and self.messages[-1].header.stamp == cloud.header.stamp:
                msg = self.messages[-1]
                dtype, channels = (np.uint8, 4) if certain else (np.dtype("<f4"), 5)
                image = np.ndarray((msg.height, msg.width, channels), dtype=dtype, buffer=msg.data,
                                   strides=(msg.step, channels * np.dtype(dtype).itemsize, np.dtype(dtype).itemsize))
                if np.any(image):
                    self.assertEqual(msg.encoding, "8UC4" if certain else "32FC5")
                    self.assertEqual(msg.header.frame_id, "base_link")
                    return image.copy()
            time.sleep(0.03)
        self.node_log.seek(0)
        self.fail("no nonempty BEV output; node log:\n" + self.node_log.read())

    def test_certain_channel_order_and_observed_mask(self):
        self.start_node(certain=True)
        points = [(float(i), 0., 1., 0x112233, i) for i in range(4)]
        image = self.output(self.cloud(points=points, include_probs=False), certain=True)
        for label in range(3):
            expected = [0, 0, 0, 1]
            expected[label] = 1
            np.testing.assert_array_equal(image[4-label, 4], expected)
        np.testing.assert_array_equal(image[1, 4], [0, 0, 0, 1]) # others 也有几何观测
        np.testing.assert_array_equal(image[0, 0], [0, 0, 0, 0])
        self.assertTrue(np.all((image == 0) | (image == 1)))

    def test_world_cloud_is_not_transformed_twice(self):
        self.config["bev"]["input_cloud_in_world"] = 1
        self.start_node(certain=True, odom_x=10.0)
        image = self.output(self.cloud(x=10.0, frame="world", include_probs=False), certain=True)
        np.testing.assert_array_equal(image[4, 4], [0, 1, 0, 1])

    def test_shipped_config_starts_without_override(self):
        self.node = subprocess.Popen([BEV_BINARY, "__name:=" + self._testMethodName],
                                     stdout=self.node_log, stderr=subprocess.STDOUT)
        for _ in range(100):
            self.node_log.seek(0)
            output = self.node_log.read()
            if "[BEV] Started:" in output:
                self.assertIsNone(self.node.poll())
                return
            if self.node.poll() is not None:
                self.fail(output)
            time.sleep(0.05)
        self.fail("project YAML did not produce BEV startup message")

    def test_uncertain_order_entropy_mask_and_probability_mass(self):
        self.start_node()
        image = self.output(self.cloud())
        np.testing.assert_allclose(image[4, 4, :3], [0.7, 0.2, 0.1], atol=1e-6)
        expected_entropy = -sum(p * math.log(p) for p in [0.7, 0.2, 0.1]) / math.log(3)
        self.assertAlmostEqual(image[4, 4, 3], expected_entropy, places=5)
        self.assertEqual(image[4, 4, 4], 1)
        np.testing.assert_array_equal(image[0, 0], [0, 0, 0, 0, 0])
        # 其他类的概率质量不应被强制分配到这三类。
        image = self.output(self.cloud(x=1., probs=(0.1, 0.2, 0.1)))
        np.testing.assert_allclose(image[3, 4, :3], [.1, .2, .1], atol=1e-6)
        self.assertEqual(image[3, 4, 4], 1)
        self.assertTrue(np.all((image[:, :, 4] == 0) | (image[:, :, 4] == 1)))

    def test_fusion_is_per_frame_and_order_independent(self):
        self.start_node()
        road = (0., 0., 1., 0x112233, 0, 1., 0., 0.)
        building = (0., 0., 1., 0x112233, 1, 0., 1., 0.)
        image = self.output(self.cloud(points=[road, building]))
        np.testing.assert_allclose(image[4, 4, :3], [.5, .5, 0], atol=1e-6)
        image = self.output(self.cloud(points=[building, road]))
        np.testing.assert_allclose(image[4, 4, :3], [.5, .5, 0], atol=1e-6)
        # 一帧 100 点和另一帧 1 点各贡献一次观测。
        image = self.output(self.cloud(x=1., points=[(1.,0.,1.,0x112233,0,1.,0.,0.)]*100))
        image = self.output(self.cloud(x=1., probs=(0, 1, 0)))
        np.testing.assert_allclose(image[3, 4, :3], [.5, .5, 0], atol=1e-6)

    def test_lower_points_do_not_overwrite_top_surface(self):
        self.start_node()
        road = (0.,0.,0.,0x112233,0,1.,0.,0.)
        building = (0.,0.,2.,0x112233,1,0.,1.,0.)
        image = self.output(self.cloud(points=[building]+[road]*100))
        np.testing.assert_allclose(image[4,4,:3], [0,1,0], atol=1e-6)

    def test_per_cell_fusion_capacity_does_not_expire_map(self):
        self.config["bev"]["history_frames"] = 2
        self.start_node()
        for t, p in [(100., (1,0,0)), (100.1, (0,1,0)), (100.2, (0,0,1))]:
            image = self.output(self.cloud(probs=p, stamp=rospy.Time.from_sec(t)))
        np.testing.assert_allclose(image[4,4,:3], [0,.5,.5], atol=1e-6)
        image = self.output(self.cloud(x=1., probs=(1,0,0), stamp=rospy.Time.from_sec(101.)))
        self.assertEqual(image[4,4,4], 1)
        np.testing.assert_allclose(image[4,4,:3], [0,.5,.5], atol=1e-6)
        self.assertEqual(image[3,4,4], 1)

    def test_sparse_window_publishes_and_survives_long_gap(self):
        self.start_node()
        first=self.output(self.cloud(probs=(1,0,0),stamp=rospy.Time(100,0)))
        self.assertEqual(first.shape,(8,8,5))
        self.assertGreater(first[:,:,4].sum(),0)
        self.assertLess(first[:,:,4].sum(),64) # 未填满窗口也立即发布。
        # 一百秒后没有新增点，窗口内旧格子仍然存在。
        later=self.output(self.cloud(points=[],stamp=rospy.Time(200,0)))
        np.testing.assert_array_equal(later,first)

    def test_spatial_window_eviction_does_not_resurrect_old_cells(self):
        self.config["bev"]["input_cloud_in_world"]=1
        self.start_node(certain=True)
        self.output(self.cloud(x=0.,label=0,frame="world",include_probs=False),certain=True)
        self.odom_template.pose.pose.position.x=20.
        moved=self.output(self.cloud(x=20.,label=1,frame="world",include_probs=False),certain=True)
        np.testing.assert_array_equal(moved[4,4],[0,1,0,1])
        self.odom_template.pose.pose.position.x=0.
        returned=self.output(self.cloud(x=1.,label=2,frame="world",include_probs=False),certain=True)
        np.testing.assert_array_equal(returned[4,4],[0,0,0,0]) # 已移出过窗口，不能重新出现。
        np.testing.assert_array_equal(returned[4,5],[0,0,1,1])

    def test_keyframe_gate(self):
        self.config["bev"].update(accumulation_update_distance_m=1., accumulation_update_yaw_deg=90.,
                                   accumulation_max_interval_sec=1.)
        self.start_node()
        self.output(self.cloud(probs=(1,0,0), stamp=rospy.Time.from_sec(100.)))
        image = self.output(self.cloud(probs=(0,1,0), stamp=rospy.Time.from_sec(100.1)))
        np.testing.assert_allclose(image[4,4,:3], [1,0,0], atol=1e-6)
        image = self.output(self.cloud(probs=(0,1,0), stamp=rospy.Time.from_sec(101.1)))
        np.testing.assert_allclose(image[4,4,:3], [.5,.5,0], atol=1e-6)

    def test_rewind_clears_accumulation(self):
        self.start_node()
        self.output(self.cloud(probs=(1,0,0), stamp=rospy.Time.from_sec(100.)))
        image = self.output(self.cloud(probs=(0,1,0), stamp=rospy.Time.from_sec(90.)))
        np.testing.assert_allclose(image[4,4,:3], [0,1,0], atol=1e-6)

    def test_world_history_stays_fixed_when_vehicle_moves(self):
        self.config["bev"]["input_cloud_in_world"] = 1
        self.start_node(certain=True, odom_x=10.)
        self.output(self.cloud(x=10., label=0, frame="world", include_probs=False), certain=True)
        self.odom_template.pose.pose.position.x = 11.
        image = self.output(self.cloud(x=11., label=1, frame="world", include_probs=False), certain=True)
        np.testing.assert_array_equal(image[4,3], [1,0,0,1], err_msg=str(np.argwhere(image[:,:,:3])))
        np.testing.assert_array_equal(image[4,4], [0,1,0,1])

    def test_cloud_uses_interpolated_capture_pose(self):
        # 使用非栅格边界位置，避免 Time.from_sec 纳秒舍入改变边界归属。
        self.start_node(certain=True, odom_x=.25)
        self.output(self.cloud(label=0, include_probs=False, stamp=rospy.Time.from_sec(100.)), certain=True)
        future = copy.deepcopy(self.odom_template)
        future.header.stamp = rospy.Time.from_sec(100.2)
        future.pose.pose.position.x = 2.25
        self.odom_pub.publish(future)
        time.sleep(.05)
        image = self.output(self.cloud(label=1, include_probs=False, stamp=rospy.Time.from_sec(100.1)),
                            certain=True, sync_pose=False)
        np.testing.assert_array_equal(image[4,3], [1,0,0,1], err_msg=str(np.argwhere(image[:,:,:3])))
        np.testing.assert_array_equal(image[4,4], [0,1,0,1])

    def test_duplicate_cloud_is_not_counted_twice(self):
        self.start_node()
        first=self.cloud(probs=(1,0,0),stamp=rospy.Time(100,0))
        self.output(first)
        self.messages.clear()
        self.send(first)
        time.sleep(.2)
        self.assertFalse(self.messages)
        image=self.output(self.cloud(probs=(0,1,0),stamp=rospy.Time(100,100000000)))
        np.testing.assert_allclose(image[4,4,:3],[.5,.5,0],atol=1e-6)

    def test_zero_semantic_mass_still_has_observation_mask(self):
        self.start_node()
        image=self.output(self.cloud(probs=(0,0,0),label=3))
        np.testing.assert_array_equal(image[4,4],[0,0,0,0,1])

    def test_negative_position_and_rotated_binary_mask(self):
        self.config["bev"]["input_cloud_in_world"]=1
        self.start_node(odom_x=-.25)
        self.odom_template.pose.pose.orientation.z=math.sin(math.pi/8)
        self.odom_template.pose.pose.orientation.w=math.cos(math.pi/8)
        image=self.output(self.cloud(x=-.25,frame="world"))
        np.testing.assert_allclose(image[4,4,:3],[.7,.2,.1],atol=1e-6)
        self.assertEqual(image[4,4,4],1)
        self.assertTrue(np.all((image[:,:,4]==0)|(image[:,:,4]==1)))
        self.assertTrue(np.all(image[:,:,:4][image[:,:,4]==0]==0))

    def test_unmatched_pose_does_not_publish(self):
        self.start_node()
        self.send(self.cloud(stamp=rospy.Time.from_sec(10.)), sync_pose=False)
        time.sleep(.3)
        self.assertFalse(self.messages)

    def test_uncertain_requires_probability_field(self):
        self.start_node()
        self.cloud_pub.publish(self.cloud(include_probs=False))
        time.sleep(0.4)
        self.assertFalse(self.messages)
        self.assertIsNone(self.node.poll())

    def test_certain_requires_label_field(self):
        self.start_node(certain=True)
        cloud=self.cloud(include_probs=False)
        cloud.fields=[field for field in cloud.fields if field.name!="label"]
        self.send(cloud)
        time.sleep(.3)
        self.assertFalse(self.messages)
        self.assertIsNone(self.node.poll())

    def test_no_entropy_configuration(self):
        self.config["particle_filter"]["method"] = "ours_no_entropy"
        self.config["particle_filter"]["backend"] = "forloop"
        self.config["bev"]["probability_channels"] = [0, 2, 1]
        self.start_node()
        image = self.output(self.cloud())
        np.testing.assert_allclose(image[4, 4, :3], [0.7, 0.1, 0.2], atol=1e-6)

    def write_config(self):
        # OpenCV FileStorage 使用带版本头的 YAML；仅写本测试的标量和整数数组。
        lines = ["%YAML:1.0", "---"]
        for section, values in self.config.items():
            lines.append(section + ":")
            for key, value in values.items():
                lines.append(f"  {key}: {value}")
        self.config_path.write_text("\n".join(lines) + "\n")

    def test_invalid_mode_combinations_fail_at_start(self):
        for section, key, value in [("bev", "mode", "typo"), ("bev", "mode", "certain"),
                                     ("particle_filter", "method", "typo"),
                                     ("particle_filter", "backend", "typo"),
                                     ("particle_filter", "score_lambda", 0),
                                     ("particle_filter", "score_lambda", -1),
                                     ("particle_filter", "tdf_value_scale", 0),
                                     ("particle_filter", "tdf_value_scale", -1),
                                     ("bev", "probability_channels", [0, 0, 2])]:
            old = self.config[section][key]
            self.config[section][key] = value
            self.write_config()
            with subprocess.Popen([BEV_BINARY, "_config:=" + str(self.config_path)],
                                  stdout=self.node_log, stderr=subprocess.STDOUT) as process:
                self.assertEqual(process.wait(timeout=10), 1)
            self.config[section][key] = old


if __name__ == "__main__":
    unittest.main()
