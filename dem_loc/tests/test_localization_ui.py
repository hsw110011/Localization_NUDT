"""Native DSM zoom/pan must preserve track detail without redrawing per sample."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from loc_tool.global_track_view import GlobalTrackView

ROS_MESSAGES_AVAILABLE = importlib.util.find_spec('self_state') is not None
if ROS_MESSAGES_AVAILABLE:
    spec = importlib.util.spec_from_file_location(
        'localization_ui_test_target', Path(__file__).resolve().parents[1] / 'scripts' / 'localization_python.py',
    )
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)


class TrackViewportTest(unittest.TestCase):
    def test_native_coordinates_zoom_anchor_and_pan(self):
        background = np.zeros((1200, 1800, 3), np.uint8)
        view = GlobalTrackView(background, 520, 700)
        anchor = view.view_to_map(150, 260)
        view.zoom_at(150, 260, 2.)
        np.testing.assert_allclose(view.view_to_map(150, 260), anchor)
        view.zoom_at(150, 260)  # Native 1:1, at the cursor.
        self.assertEqual(view.scale, 1.)
        np.testing.assert_allclose(view.view_to_map(150, 260), anchor)
        mapped = view.map_to_view(*anchor)
        view.pan(30, -20)
        np.testing.assert_allclose(view.map_to_view(*anchor), mapped + [30, -20])
        view.fit()
        self.assertEqual(view.scale, view.fit_scale)
        self.assertIs(view.background, background)

    def test_track_updates_do_not_rebuild_background(self):
        background = np.zeros((1200, 1800, 3), np.uint8)
        with patch.object(cv2, 'warpAffine', wraps=cv2.warpAffine) as warp:
            view = GlobalTrackView(background)
            previous = None
            for x in range(400, 500):
                previous = view.add_point(SimpleNamespace(x=x + .125, y=600.25), (0, 255, 0), previous)
            self.assertEqual(warp.call_count, 1)
            self.assertEqual(view.paths[(0, 255, 0)][0][-1], (499.125, 600.25))
            self.assertFalse(np.any(background))
            anchor = view.map_to_view(450., 600.)
            view.zoom_at(*anchor)
            self.assertEqual(warp.call_count, 2)
            self.assertTrue(np.any(view.image[:, :, 1]))
            exported = view.export_image()
            self.assertEqual(exported.shape, background.shape)
            self.assertTrue(np.any(exported[600, 400:501, 1]))

    def test_nearby_tracks_remain_separate_at_native_zoom(self):
        view = GlobalTrackView(np.zeros((1200, 1800, 3), np.uint8), 520, 700)
        for y, color in ((600., (0, 255, 0)), (603., (0, 0, 255))):
            previous = None
            for x in (880., 920.):
                previous = view.add_point(SimpleNamespace(x=x, y=y), color, previous)
        view.zoom_at(259.5, 349.5)
        for y, channel in ((600., 1), (603., 2)):
            x, py = np.rint(view.map_to_view(900., y)).astype(int)
            self.assertGreater(view.image[py, x, channel], 180)
            self.assertEqual(view.image[py, x, 3 - channel], 0)


@unittest.skipUnless(ROS_MESSAGES_AVAILABLE, 'source catkin_ws/devel/setup.bash for ROS message types')
class OverviewTest(unittest.TestCase):
    def test_full_resolution_sidebar_mouse_controls_and_canvas_cache(self):
        from loc_tool.dem_tool import DemTiffData
        data = DemTiffData(raw_elevation_map=np.arange(1200 * 800, dtype=np.float32).reshape(1200, 800),
                           min_height=0., max_height=1200 * 800.)
        with patch.object(cv2, 'resize', wraps=cv2.resize) as resize:
            background = app._make_dem_color_map(data)
            self.assertEqual(background.shape, (1200, 800, 3))
            self.assertEqual(resize.call_count, 0)
            node = app.PythonLocalizationNode.__new__(app.PythonLocalizationNode)
            node.vis_track = background
            node.track_view = GlobalTrackView(background)
            node._main_vis_canvas = node._track_rect = node._track_drag = None
            node.draw_global_pose_track = node.draw_local_pose_track = True
            node.draw_odom_track = node.draw_pf_track = True
            node._set_bev_vis_cache(np.zeros((1044, 1280, 3), np.uint8), {'panels': []})
            canvas = node._main_vis_canvas
            node._set_bev_vis_cache(np.zeros((1044, 1280, 3), np.uint8), {'panels': []})
            self.assertIs(node._main_vis_canvas, canvas)
            self.assertEqual(resize.call_count, 0)
            self.assertEqual(canvas.shape, (1044, 1816, 3))
            x0, y0, width, height = node._track_rect
            with patch.object(cv2, 'imshow'):
                node._on_bev_score_mouse(cv2.EVENT_RBUTTONDOWN, x0 + 260, y0 + 400, 0, None)
                self.assertEqual(node.track_view.scale, 1.)
                center = node.track_view.center.copy()
                node._on_bev_score_mouse(cv2.EVENT_LBUTTONDOWN, x0 + 260, y0 + 400, 0, None)
                node._on_bev_score_mouse(cv2.EVENT_MOUSEMOVE, x0 + 280, y0 + 430, cv2.EVENT_FLAG_LBUTTON, None)
                np.testing.assert_allclose(node.track_view.center, center - [20., 30.])
                node._on_bev_score_mouse(cv2.EVENT_LBUTTONUP, x0 + 280, y0 + 430, 0, None)
                self.assertIsNone(node._track_drag)
                node._on_bev_score_mouse(cv2.EVENT_MOUSEWHEEL, x0 + 260, y0 + 400, 120 << 16, None)
                self.assertEqual(node.track_view.scale, 1.25)
                node._on_bev_score_mouse(cv2.EVENT_LBUTTONDBLCLK, x0 + 260, y0 + 400, 0, None)
                self.assertEqual(node.track_view.scale, node.track_view.fit_scale)
            point = node._draw_global_track(app.PixelPoint(400.125, 600.25), color=(0, 255, 0))
            self.assertEqual(point, (400.125, 600.25))
            self.assertTrue(node._track_view_dirty)
            node._blit_track_view()
            np.testing.assert_array_equal(canvas[y0:y0 + height, x0:x0 + width], node.track_view.image)
            self.assertFalse(node._track_view_dirty)
            self.assertEqual(resize.call_count, 0)


if __name__ == '__main__':
    unittest.main()
