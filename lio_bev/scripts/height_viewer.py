#!/usr/bin/env python3
"""Independent OpenCV viewer for actual BEV heights; no main-node processing."""
import math
import os
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

# Also works through catkin's devel-space executable wrapper.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from surface_diagnostics import unpack_grid_map


COLORMAP = getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET)


def colorize_height(height, valid, lower, upper):
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError('height color range must be finite and increasing')
    finite = np.asarray(valid, dtype=bool) & np.isfinite(height)
    scaled = np.zeros(height.shape, dtype=np.uint8)
    scaled[finite] = np.rint(np.clip((height[finite]-lower)/(upper-lower), 0, 1)*255).astype(np.uint8)
    image = cv2.applyColorMap(scaled, COLORMAP)
    image[~finite] = 0
    return image


def profile_values(height, valid, y, half_width=1.0):
    band = np.abs(y) <= half_width
    profile = np.full(height.shape[0], np.nan)
    for row in range(height.shape[0]):
        values = height[row, band & valid[row] & np.isfinite(height[row])]
        if len(values):
            profile[row] = np.median(values)
    return profile


def text(image, label, xy, color=(225, 225, 225), scale=0.48):
    cv2.putText(image, label, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_height(height, valid, x, y, lower, upper, label='', mode='FIXED'):
    rows, cols = height.shape
    scale = min(2.0, 720.0/max(rows, cols))
    width, size_y = max(1, round(cols*scale)), max(1, round(rows*scale))
    left, top = 58, 60
    image = np.full((top+size_y+245, left+width+210, 3), 28, dtype=np.uint8)
    colored = colorize_height(height, valid, lower, upper)
    enlarged = cv2.resize(colored, (width, size_y), interpolation=cv2.INTER_NEAREST)
    image[top:top+size_y, left:left+width] = enlarged
    text(image, 'BEV actual height: H_stable_surf [m]', (left, 22), scale=.58)
    text(image, '+X forward = up; +Y left = left. Black = unobserved.', (left, 44))
    for fraction in (0, .25, .5, .75, 1):
        row = min(rows-1, round(fraction*(rows-1)))
        col = min(cols-1, round(fraction*(cols-1)))
        py = top+min(size_y-1, round(fraction*(size_y-1)))
        px = left+min(width-1, round(fraction*(width-1)))
        text(image, '{:+.0f}'.format(x[row]), (5, py+5))
        text(image, '{:+.0f}'.format(y[col]), (px-12, top+size_y+19))
    # Only mark the body origin. Do not cover measured cells with an ego rectangle.
    if x.min() <= 0 <= x.max() and y.min() <= 0 <= y.max():
        rr, cc = int(np.argmin(np.abs(x))), int(np.argmin(np.abs(y)))
        cx, cy = left+round((cc+.5)*width/cols), top+round((rr+.5)*size_y/rows)
        cv2.drawMarker(image, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
    bar_x = left+width+30
    bar_height = max(120, min(size_y, 420))
    ramp = np.linspace(upper, lower, bar_height)[:, None]
    bar = colorize_height(ramp, np.ones(ramp.shape, bool), lower, upper)
    image[top:top+bar_height, bar_x:bar_x+24] = np.repeat(bar, 24, axis=1)
    text(image, 'Height [m]', (bar_x-10, top-14))
    for f in (0, .25, .5, .75, 1):
        value = upper-f*(upper-lower)
        text(image, '{:+.2f}'.format(value), (bar_x+31, top+round(f*(bar_height-1))+5))
    text(image, mode+' color range', (bar_x-10, top+bar_height+30))
    finite = np.asarray(valid, bool) & np.isfinite(height)
    values = height[finite]
    if len(values):
        labels = ['Valid: {}'.format(len(values)),
                  'Min: {:+.3f} m'.format(float(values.min())),
                  'Max: {:+.3f} m'.format(float(values.max())),
                  'Below range: {:.1f}%'.format(100*np.mean(values < lower)),
                  'Above range: {:.1f}%'.format(100*np.mean(values > upper))]
        for i, item in enumerate(labels):
            text(image, item, (bar_x-10, top+bar_height+55+i*22))
    # A numeric longitudinal profile makes gradual slopes and actual steps visible.
    plot_top, plot_bottom = top+size_y+62, top+size_y+175
    plot_left, plot_right = left, left+width-1
    text(image, 'Center-strip median height: |y| <= 1 m', (left, plot_top-15))
    cv2.rectangle(image, (plot_left, plot_top), (plot_right, plot_bottom), (95, 95, 95), 1)
    for f in (0, .5, 1):
        yy = round(plot_top+f*(plot_bottom-plot_top))
        text(image, '{:+.2f}'.format(upper-f*(upper-lower)), (4, yy+4), scale=.4)
        cv2.line(image, (plot_left, yy), (plot_right, yy), (55, 55, 55), 1)
    profile = profile_values(height, finite, y)
    previous = None
    for row, value in enumerate(profile):
        if not np.isfinite(value):
            previous = None
            continue
        px = plot_left+round((x[row]-x.min())/max(float(np.ptp(x)), 1e-9)*(plot_right-plot_left))
        py = plot_bottom-round(np.clip((value-lower)/(upper-lower), 0, 1)*(plot_bottom-plot_top))
        current = (px, py)
        if previous is not None:
            cv2.line(image, previous, current, (100, 240, 230), 1, cv2.LINE_AA)
        previous = current
    text(image, '{:+.1f} m (rear)'.format(float(x.min())), (left, plot_bottom+18), scale=.42)
    text(image, '{:+.1f} m (front)'.format(float(x.max())), (max(left, plot_right-110), plot_bottom+18), scale=.42)
    text(image, label, (left, plot_bottom+40), scale=.42)
    text(image, 'A: auto/fixed  R: reset  [ / ]: narrow/widen  S: save  Q/Esc: close',
         (left, plot_bottom+62), scale=.42)
    return image, (left, top, width, size_y)


def main():
    import rospy
    from grid_map_msgs.msg import GridMap
    from sensor_msgs.msg import Image
    rospy.init_node('lio_bev_height_viewer')
    cv2.setNumThreads(1)
    lower = float(rospy.get_param('~height_min', -2.0))
    upper = float(rospy.get_param('~height_max', 2.0))
    rate = float(rospy.get_param('~rate_hz', 5.0))
    show = bool(rospy.get_param('~show_window', True))
    if not all(math.isfinite(v) for v in (lower, upper, rate)) or upper <= lower or not 0 < rate <= 30:
        raise ValueError('invalid height range or display rate')
    if show and not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        raise RuntimeError('No desktop display. Use _show_window:=false for image-topic-only output.')
    topic = str(rospy.get_param('~input_topic', '/lidar_bev/grid_map'))
    image_topic = str(rospy.get_param('~image_topic', '/lidar_bev/debug/height_image'))
    snapshot_dir = Path(rospy.get_param('~snapshot_dir', '/tmp/lio_bev_height_viewer'))
    pub = rospy.Publisher(image_topic, Image, queue_size=1, latch=True)
    latest = [None, None]
    mouse = [-1, -1]
    mutex = threading.Lock()

    def receive(msg):
        with mutex:
            latest[:] = [msg, time.monotonic()]

    def on_mouse(event, x, y, flags, param):
        mouse[:] = [x, y]

    sub = rospy.Subscriber(topic, GridMap, receive, queue_size=1, buff_size=8*1024*1024)
    stop = threading.Event()
    rospy.on_shutdown(stop.set)
    window = 'LiDAR BEV height [m]'
    if show:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window, 900, 980)
        cv2.setMouseCallback(window, on_mouse)
    fixed = (lower, upper)
    automatic = False
    previous, cached, base, rectangle, received_at = None, None, None, None, None
    next_render = 0.0
    force = False
    rospy.loginfo('OpenCV height viewer: %s, fixed range [%g, %g] m, %.1f Hz', topic, lower, upper, rate)
    try:
        while not rospy.is_shutdown():
            with mutex:
                msg, received = latest
            now = time.monotonic()
            if msg is not None and (force or (msg is not previous and now >= next_render)):
                try:
                    h, valid, x, y = unpack_grid_map(msg)
                    if automatic:
                        values = h[valid & np.isfinite(h)]
                        if len(values):
                            lo, hi = np.percentile(values, [2, 98])
                            if hi-lo < .02:
                                lo, hi = lo-.01, hi+.01
                            lower, upper = float(lo), float(hi)
                    label = 'Frame: {}  stamp: {:.3f}  x/y tick units: m'.format(
                        msg.info.header.frame_id, msg.info.header.stamp.to_sec())
                    base, rectangle = render_height(h, valid, x, y, lower, upper,
                                                    label, 'AUTO 2-98%' if automatic else 'FIXED')
                    cached = (h, valid, x, y, msg)
                    received_at = received
                    previous, force, next_render = msg, False, now+1/rate
                    image_msg = Image()
                    image_msg.header = copy_header(msg.info.header)
                    image_msg.height, image_msg.width = base.shape[:2]
                    image_msg.encoding = 'bgr8'
                    image_msg.step = image_msg.width*3
                    image_msg.data = base.tobytes()
                    pub.publish(image_msg)
                except (ValueError, IndexError) as error:
                    rospy.logwarn_throttle(2, 'height viewer: %s', str(error))
                    previous = msg
            if show:
                if base is None:
                    display = np.full((200, 850, 3), 28, dtype=np.uint8)
                    text(display, 'Waiting for '+topic, (30, 85), scale=.7)
                    text(display, 'Continue rosbag playback / start the BEV publisher.', (30, 120))
                else:
                    display = base.copy()
                    h, valid, x, y, shown = cached
                    left, top, width, size_y = rectangle
                    mx, my = mouse
                    row, col = int((my-top)*h.shape[0]/size_y), int((mx-left)*h.shape[1]/width)
                    if left <= mx < left+width and top <= my < top+size_y:
                        value = '{:+.4f} m'.format(h[row, col]) if valid[row, col] and np.isfinite(h[row, col]) else 'unobserved'
                        label = 'x={:+.2f}m  y={:+.2f}m  height={}'.format(x[row], y[col], value)
                        cv2.rectangle(display, (left, top+4), (left+width, top+30), (20, 20, 20), -1)
                        text(display, label, (left+8, top+24), scale=.52)
                    age = now-received_at
                    if age > 1:
                        text(display, 'Last frame received {:.1f}s ago'.format(age), (left, 44), (50, 180, 255))
                cv2.imshow(window, display)
                key = cv2.waitKey(20) & 0xff
                if key in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord('a'):
                    automatic = not automatic
                    if not automatic:
                        lower, upper = fixed
                    force = True
                elif key == ord('r'):
                    lower, upper = fixed
                    automatic, force = False, True
                elif key in (ord('['), ord(']')):
                    center, half = (lower+upper)/2, (upper-lower)/2
                    half = max(.01, min(1000, half*(.5 if key == ord('[') else 2)))
                    lower, upper = center-half, center+half
                    automatic, force = False, True
                elif key == ord('s') and cached is not None:
                    snapshot_dir.mkdir(parents=True, exist_ok=True)
                    path = snapshot_dir/('height_{}.png'.format(cached[4].info.header.stamp.to_nsec()))
                    if not cv2.imwrite(str(path), display):
                        rospy.logerr('Failed saving %s', path)
                    else:
                        rospy.loginfo('Saved %s', path)
            else:
                stop.wait(.02)
    finally:
        sub.unregister()
        if show:
            cv2.destroyAllWindows()
        rospy.signal_shutdown('height viewer closed')


def copy_header(header):
    from std_msgs.msg import Header
    return Header(seq=header.seq, stamp=header.stamp, frame_id=header.frame_id)


if __name__ == '__main__':
    main()
