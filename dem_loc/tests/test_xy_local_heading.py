"""XY filtering must preserve the external heading through motion and resampling."""
import importlib.util
import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from loc_tool.dem_tool import DemTiffData
from loc_tool.coord_converter import CoordConverter
from loc_tool.dsm_patch import DsmPatchCropper
from loc_tool.dsm_bev_score import DsmBevScoreConfig
from loc_tool.particle_filter import ParticleFilter, ParticleFilterConfig

ROS_AVAILABLE = importlib.util.find_spec("self_state") is not None
if ROS_AVAILABLE:
    from nav_msgs.msg import Odometry
    from self_state.msg import GlobalPose, LocalPose
    from loc_tool.particle_filter_runner import ParticleFilterRunner


def make_map(device):
    data = DemTiffData(raw_elevation_map=np.zeros((128, 128), np.float32),
                       geo_bounds=[113., 28.00023, 113.00026, 28.],
                       resolution=.2, zone_num=49)
    cc = CoordConverter(data)
    crop = DsmPatchCropper(data, cc, map_size_x=4., map_size_y=4., device=device)
    return cc, crop


class XYFilterTest(unittest.TestCase):
    def test_heading_survives_motion_observation_and_roughening(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            cc, crop = make_map(device)
            cfg = ParticleFilterConfig(num_particles=16, device=device, save_debug=False,
                                       save_weight_vis=False)
            pf = ParticleFilter(cfg, DsmBevScoreConfig(), crop)
            origin = cc.wgs84_to_gauss(113.00013, 28.000115)
            pf.initialize(origin.x, origin.y, math.radians(179.))
            np.testing.assert_array_equal(pf.particles[:, 2].cpu().numpy(), pf.heading_rad)
            pf.set_heading(math.radians(181.))
            pf.propagate(.2, .1, .4, speed_mps=2.)
            layers = {key: np.zeros((20, 20), np.float32) for key in ("H_L", "Gx_L", "Gy_L")}
            layers["M_obs"] = np.ones((20, 20), np.float32)
            pf.update(layers, h_max=3.)
            pf.weights.zero_()
            pf.weights[0] = 1.
            pf.systematic_resample()
            np.testing.assert_array_equal(pf.particles[:, 2].cpu().numpy(), pf.heading_rad)
            self.assertAlmostEqual(pf.heading_rad, math.radians(-179.))
            estimate = pf.estimate_pose()
            for key in ("yaw_est", "yaw_est_weighted", "yaw_est_elite", "best_yaw"):
                self.assertEqual(estimate[key], pf.heading_rad)
            self.assertLess(estimate["spread_yaw"], 1e-12)
            self.assertGreater(float(torch.std(pf.particles[:, 0]).item()), 0.)


@unittest.skipUnless(ROS_AVAILABLE, "source catkin_ws/devel/setup.bash for ROS messages")
class RunnerHeadingTest(unittest.TestCase):
    def setUp(self):
        self.cc, self.crop = make_map("cpu")
        cfg = ParticleFilterConfig(num_particles=8, device="cpu", save_debug=False,
                                  save_weight_vis=False, init_std_x=0., init_std_y=0.,
                                  motion_std_x=0., motion_std_y=0., enable_roughening=False)
        self.runner = ParticleFilterRunner(cfg, DsmBevScoreConfig(), self.cc, self.crop)
        self.global_pose = GlobalPose(longitude=113.00013, latitude=28.000115, azimuth=90.)
        self.local = LocalPose(dr_x=5., dr_y=7., dr_heading=20., vehicle_speed=1.)

    @staticmethod
    def odom(x, y, yaw_deg):
        msg = Odometry()
        msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
        msg.pose.pose.orientation.z = math.sin(math.radians(yaw_deg) / 2.)
        msg.pose.pose.orientation.w = math.cos(math.radians(yaw_deg) / 2.)
        return msg

    def initialize(self, odom=None):
        base = self.cc.GetBaseFromLocalPose(
            self.local, self.global_pose,
            local_heading_unit=self.runner.local_heading_unit,
            local_heading_convention=self.runner.local_heading_convention)
        self.runner.initialize(self.global_pose, self.local, base,
                               self.odom(10., 20., 30.) if odom is None else odom)
        return self.runner.pf.particles[:, :2].cpu().numpy().copy()

    def test_first_translation_uses_previous_heading_and_ignores_local_position(self):
        before = self.initialize()
        self.local.dr_x, self.local.dr_y, self.local.dr_heading = 500., -700., 40.
        self.runner.on_local_pose(self.local)
        np.testing.assert_array_equal(self.runner.pf.particles[:, :2].cpu().numpy(), before)
        self.runner.on_odometry(self.odom(11., 22., 150.))
        theta = math.radians(60.)  # map initial yaw 90 - LIO initial yaw 30
        delta = [math.cos(theta) - 2. * math.sin(theta), math.sin(theta) + 2. * math.cos(theta)]
        np.testing.assert_allclose(self.runner.pf.particles[:, :2].cpu().numpy(), before + delta)
        self.assertAlmostEqual(self.runner.pf.heading_rad, math.radians(110.))

    def test_stationary_heading_updates_and_skipped_motion_does_not_accumulate(self):
        self.local.vehicle_speed = 0.
        before = self.initialize()
        self.local.dr_heading = 50.
        self.runner.on_local_pose(self.local)
        self.assertFalse(self.runner.on_odometry(self.odom(15., 20., 90.)))
        np.testing.assert_array_equal(self.runner.pf.particles[:, :2].numpy(), before)
        self.assertAlmostEqual(self.runner.pf.heading_rad, math.radians(120.))
        self.local.vehicle_speed = 1.
        self.runner.on_local_pose(self.local)
        self.runner.on_odometry(self.odom(16., 20., 170.))
        np.testing.assert_allclose(self.runner.pf.particles[:, :2].numpy(),
                                   before + [math.cos(math.pi/6.), math.sin(math.pi/6.)])

    def test_local_heading_changes_the_next_motion_direction(self):
        before = self.initialize(self.odom(0., 0., 0.))
        self.local.dr_heading = 110.  # initial map yaw 90 + local change 90 = 180
        self.runner.on_local_pose(self.local)
        self.runner.on_odometry(self.odom(1., 0., 0.))
        # First increment still belongs to prior heading 90, not latest 180.
        np.testing.assert_allclose(self.runner.pf.particles[:, :2].numpy(), before + [0., 1.])
        self.local.dr_heading = 140.  # current map yaw -150; previous remains 180
        self.runner.on_local_pose(self.local)
        self.runner.on_odometry(self.odom(2., 0., 0.))
        np.testing.assert_allclose(self.runner.pf.particles[:, :2].numpy(), before + [-1., 1.])
        self.assertAlmostEqual(self.runner.pf.heading_rad, math.radians(-150.))

    def test_noise_axes_rotate_with_vehicle_heading(self):
        cfg = ParticleFilterConfig(num_particles=4096, device="cpu", save_debug=False,
                                  save_weight_vis=False, init_std_x=0., init_std_y=0.,
                                  motion_std_x=.8, motion_std_y=.05, motion_speed_noise_gain=0.)
        pf = ParticleFilter(cfg, DsmBevScoreConfig(), self.crop)
        pf.initialize(700000., 3173000., 0.)
        before = pf.particles[:, :2].numpy().copy()
        pf.propagate(1., 0., math.pi / 2., speed_mps=1.)
        noise = pf.particles[:, :2].numpy() - before - [0., 1.]
        self.assertGreater(np.std(noise[:, 1]), 20. * np.std(noise[:, 0]))
        np.testing.assert_allclose(np.mean(noise, axis=0), [0., 0.], atol=.04)

    def test_radian_azimuth_heading_conversion_and_wrap(self):
        self.runner.local_heading_unit = "rad"
        self.runner.local_heading_convention = "azimuth"
        self.local.dr_heading = 0.
        self.global_pose.azimuth = 170.
        self.initialize()
        self.local.dr_heading = math.radians(-20.)
        self.runner.on_local_pose(self.local)
        self.assertAlmostEqual(self.runner.pf.heading_rad, math.radians(-170.))

    def test_bev_result_publishes_local_heading_with_filtered_position(self):
        self.initialize()
        self.local.dr_heading = 35.
        self.runner.on_local_pose(self.local)
        layers = {key: np.zeros((20, 20), np.float32) for key in ("H_L", "Gx_L", "Gy_L")}
        layers["M_obs"] = np.ones((20, 20), np.float32)
        bev = SimpleNamespace(info=SimpleNamespace(length_x=4., length_y=4., resolution=.2))
        estimate = self.runner.on_bev_message(bev, lidar_layers=layers, h_max=3.,
                                               template_global_pose=self.global_pose)
        self.assertAlmostEqual(estimate["pf_msg"].azimuth, 105.)
        self.assertEqual(estimate["pf_msg"].gaussX, estimate["x_est"])
        self.assertEqual(estimate["pf_msg"].gaussY, estimate["y_est"])

    def test_node_routes_heading_before_lio_motion_and_waits_for_lio(self):
        from unittest.mock import Mock, patch
        from test_localization_ui import app
        node = app.PythonLocalizationNode.__new__(app.PythonLocalizationNode)
        node.coord_converter = self.cc
        node.global_heading_unit = node.local_heading_unit = "deg"
        node.global_heading_convention = node.local_heading_convention = "math"
        node.latest_global_pose, node.latest_local_pose = self.global_pose, self.local
        node.latest_lio_odom = None
        node.first_frame_aligned = False
        node.pf_runner = self.runner
        node.draw_global_pose_track = node.draw_local_pose_track = node.draw_odom_track = False
        node.save_pose_csv = False
        with patch.object(app.rospy, "loginfo"):
            node._try_align_first_frame()
            self.assertFalse(node.first_frame_aligned)
            node.latest_lio_odom = self.odom(10., 20., 30.)
            node._try_align_first_frame()
        self.assertTrue(node.first_frame_aligned)
        before = self.runner.pf.particles[:, :2].numpy().copy()
        self.local.dr_heading = 40.
        self.local.dr_x = 500.
        node.input = SimpleNamespace(GlobalPose_refreshflag=False,
                                     LocalPose_refreshflag=True, LocalPose=self.local,
                                     LioOdom_refreshflag=True, LioOdom=self.odom(11., 20., 150.))
        node.interface = SimpleNamespace(ConvertToLocalData=lambda data: None)
        node.sync_bev_from_topic = node.enable_dsm_bev_score = node.show_window = False
        node.enable_particle_filter = True
        node.loop_rate = 100.
        node.local_pose_count = 0
        node._try_update_bev_frame = Mock()
        with patch.object(app.rospy, "Rate"), patch.object(app.rospy, "is_shutdown", side_effect=[False, True]):
            node.spin()
        np.testing.assert_allclose(self.runner.pf.particles[:, :2].numpy(),
                                   before + [math.cos(math.pi/3.), math.sin(math.pi/3.)])
        self.assertAlmostEqual(self.runner.pf.heading_rad, math.radians(110.))
        node._try_update_bev_frame.assert_called_once()

    def test_lio_inbox_has_an_independent_refresh_flag(self):
        import threading
        from loc_tool.cinterface import CInterface
        from loc_tool.common_struct import InputData
        interface = CInterface.__new__(CInterface)
        interface._lock = threading.RLock()
        msg = self.odom(1., 2., 30.)
        interface.LioOdomCallback(msg)
        data = InputData()
        interface.ConvertToLocalData(data)
        self.assertIs(data.LioOdom, msg)
        self.assertTrue(data.LioOdom_refreshflag)
        self.assertFalse(data.LocalPose_refreshflag)
        interface.ConvertToLocalData(data)
        self.assertFalse(data.LioOdom_refreshflag)
