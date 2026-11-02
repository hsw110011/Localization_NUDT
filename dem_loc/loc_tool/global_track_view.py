"""Full-resolution DSM with a cached, zoomable trajectory viewport (no ROS)."""
import cv2
import numpy as np


class GlobalTrackView:
    def __init__(self, background, width=520, height=520):
        self.background = background
        self.map_h, self.map_w = background.shape[:2]
        self.paths = {}
        self.width, self.height = int(width), int(height)
        self.center = np.array([(self.map_w - 1) / 2., (self.map_h - 1) / 2.])
        self.scale = self.fit_scale
        self._redraw()

    @property
    def fit_scale(self):
        return min(self.width / self.map_w, self.height / self.map_h)

    def map_to_view(self, x, y):
        return (np.asarray([x, y]) - self.center) * self.scale + self._view_center()

    def view_to_map(self, x, y):
        return (np.asarray([x, y]) - self._view_center()) / self.scale + self.center

    def _view_center(self):
        return np.array([(self.width - 1) / 2., (self.height - 1) / 2.])

    def resize(self, width, height):
        width, height = int(width), int(height)
        if (width, height) == (self.width, self.height):
            return
        fitted = abs(self.scale - self.fit_scale) < 1e-9
        self.width, self.height = width, height
        if fitted:
            self.scale = self.fit_scale
        self._redraw()

    def fit(self):
        self.center[:] = [(self.map_w - 1) / 2., (self.map_h - 1) / 2.]
        self.scale = self.fit_scale
        self._redraw()

    def zoom_at(self, x, y, factor=None):
        anchor = self.view_to_map(x, y)
        scale = 1.0 if factor is None else self.scale * float(factor)
        self.scale = min(max(scale, min(self.fit_scale, 1.0)), max(8.0, self.fit_scale))
        self.center = anchor - (np.array([x, y]) - self._view_center()) / self.scale
        self._redraw()

    def pan(self, dx, dy):
        self.center -= np.array([dx, dy]) / self.scale
        self._redraw()

    def add_point(self, pixel, color, previous=None):
        # Retain native floating-point positions, including sub-pixel motion.
        point = (float(pixel.x), float(pixel.y))
        x, y = point
        if not (0 <= x < self.map_w and 0 <= y < self.map_h):
            return previous
        color = tuple(color)
        paths = self.paths.setdefault(color, [])
        if previous is None or not paths or paths[-1][-1] != previous:
            paths.append([previous] if previous is not None else [])
        paths[-1].append(point)
        self._draw_path(self.image, [previous, point] if previous is not None else [point], color)
        return point

    def _draw_path(self, image, points, color, native=False):
        points = np.asarray(points, dtype=np.float64)
        if not native:
            points = (points - self.center) * self.scale + self._view_center()
        points = np.rint(points).astype(np.int32).reshape(-1, 1, 2)
        if len(points) > 1:
            cv2.polylines(image, [points], False, color, 1, cv2.LINE_AA)
        else:
            cv2.circle(image, tuple(points[0, 0]), 0, color, -1, cv2.LINE_AA)

    def _redraw(self):
        # Warp only the fixed-size viewport, and only on zoom/pan/resize.
        offset = self._view_center() - self.scale * self.center
        matrix = np.array([[self.scale, 0., offset[0]], [0., self.scale, offset[1]]])
        self.image = cv2.warpAffine(
            self.background, matrix, (self.width, self.height),
            flags=cv2.INTER_NEAREST if self.scale >= 1.0 else cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(24, 24, 24),
        )
        for color, paths in self.paths.items():
            for path in paths:
                self._draw_path(self.image, path, color)

    def export_image(self):
        image = self.background.copy()
        for color, paths in self.paths.items():
            for path in paths:
                self._draw_path(image, path, color, native=True)
        return image
