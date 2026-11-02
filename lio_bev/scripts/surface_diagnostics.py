#!/usr/bin/env python3
"""Optional, read-only BEV surface diagnostics. No GUI, no main-node changes.

Fits a robust plane in a configurable road ROI. Residuals are diagnostic only:
they do not flatten or replace the localization map, and do not establish ground truth.
"""
import copy
import json
import math
import threading
import time

import numpy as np


def analyze_surface(height, valid, x, y, roi=(-15.0, 15.0, -1.5, 1.5),
                    inlier_threshold=0.10, min_cells=50):
    """All coordinates/heights in meters. Keep outliers in the reported residuals."""
    h = np.asarray(height, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    selected = (np.asarray(valid, dtype=bool) & np.isfinite(h) &
                (xx >= roi[0]) & (xx <= roi[1]) &
                (yy >= roi[2]) & (yy <= roi[3]))
    rows, cols = np.where(selected)
    if len(rows) < min_cells:
        raise ValueError('insufficient valid cells in road ROI: {}'.format(len(rows)))
    design = np.column_stack((xx[selected], yy[selected], np.ones(len(rows))))
    values = h[selected]
    if np.ptp(design[:, 0]) < 2.0 or np.ptp(design[:, 1]) < 0.5:
        raise ValueError('road ROI has insufficient two-dimensional coverage')
    # Bound fit work without losing the spatial ordering/coverage of a large ROI.
    indices = np.linspace(0, len(values)-1, min(6000, len(values)), dtype=int)
    fit_x, fit_z = design[indices], values[indices]
    weights = np.ones(len(fit_z))
    for _ in range(6):
        root = np.sqrt(weights)
        coeff, _, rank, _ = np.linalg.lstsq(fit_x*root[:, None], fit_z*root, rcond=None)
        if rank != 3 or not np.all(np.isfinite(coeff)):
            raise ValueError('degenerate plane fit')
        error = fit_z - fit_x @ coeff
        centered = error - np.median(error)
        scale = max(0.01, 1.4826*float(np.median(np.abs(centered))))
        weights = np.minimum(1.0, 1.5*scale/np.maximum(np.abs(error), 1e-12))
    predicted = coeff[0]*xx + coeff[1]*yy + coeff[2]
    residual = np.full(h.shape, np.nan, dtype=np.float32)
    residual[selected] = (h-predicted)[selected]
    plane = np.full(h.shape, np.nan, dtype=np.float32)
    plane[selected] = predicted[selected]
    road_height = np.full(h.shape, np.nan, dtype=np.float32)
    road_height[selected] = h[selected]
    jumps = np.full(h.shape, np.nan, dtype=np.float32)
    edge_values = []
    for axis in (0, 1):
        first = [slice(None), slice(None)]
        second = list(first)
        first[axis] = slice(None, -1)
        second[axis] = slice(1, None)
        first, second = tuple(first), tuple(second)
        finite = selected[first] & selected[second]
        delta = np.abs(residual[first]-residual[second])
        edge_values.append(delta[finite])
        for part in (first, second):
            target = jumps[part]
            target[finite] = np.fmax(target[finite], delta[finite])
    errors = residual[selected].astype(np.float64)
    absolute = np.abs(errors)
    inliers = absolute <= inlier_threshold
    edges = np.concatenate(edge_values)
    stats = {
        'roi_m': list(roi), 'valid_cells': int(selected.sum()),
        'plane_z_ax_by_c': coeff.tolist(),
        'plane_slope_deg': math.degrees(math.atan(math.hypot(coeff[0], coeff[1]))),
        'height_min_m': float(values.min()), 'height_max_m': float(values.max()),
        'residual_abs_p50_m': float(np.percentile(absolute, 50)),
        'residual_abs_p95_m': float(np.percentile(absolute, 95)),
        'residual_abs_max_m': float(absolute.max()),
        'inlier_threshold_m': float(inlier_threshold),
        'inlier_fraction': float(inliers.mean()),
        'inlier_rmse_m': float(np.sqrt(np.mean(errors[inliers]**2))) if inliers.any() else None,
        'neighbor_residual_jump_p95_m': float(np.percentile(edges, 95)) if len(edges) else None,
        'neighbor_residual_jump_max_m': float(edges.max()) if len(edges) else None,
    }
    profiles = []
    for lo in np.arange(roi[0], roi[1], 1.0):
        take = selected & (xx >= lo) & (xx < min(lo+1.0, roi[1]))
        if take.any():
            profiles.append({'x_m': float(lo+0.5), 'cells': int(take.sum()),
                             'height_median_m': float(np.median(h[take])),
                             'residual_median_m': float(np.median(residual[take]))})
    stats['longitudinal_profile'] = profiles
    return {'height': road_height, 'plane': plane, 'residual': residual,
            'neighbor_jump': jumps}, stats


def unpack_grid_map(msg):
    rows = int(round(msg.info.length_x/msg.info.resolution))
    cols = int(round(msg.info.length_y/msg.info.resolution))
    if rows <= 0 or cols <= 0:
        raise ValueError('invalid grid geometry')
    layers = {}
    for name in ('H_stable_surf', 'M_L'):
        if name not in msg.layers:
            raise ValueError('missing input layer '+name)
        data = np.asarray(msg.data[msg.layers.index(name)].data, dtype=np.float32)
        if data.size != rows*cols:
            raise ValueError('layer size does not match grid geometry')
        data = data.reshape((rows, cols), order='F')
        layers[name] = np.roll(data, (-msg.outer_start_index, -msg.inner_start_index), axis=(0, 1))
    x = msg.info.pose.position.x + msg.info.length_x/2-(np.arange(rows)+0.5)*msg.info.resolution
    y = msg.info.pose.position.y + msg.info.length_y/2-(np.arange(cols)+0.5)*msg.info.resolution
    return layers['H_stable_surf'], layers['M_L'] > 0.5, x, y


def make_points(layer, x, y, header, intensity=None):
    from sensor_msgs.msg import PointCloud2, PointField
    take = np.isfinite(layer)
    rows, cols = np.where(take)
    packed = np.empty((len(rows), 4), dtype='<f4')
    packed[:, 0] = x[rows]
    packed[:, 1] = y[cols]
    packed[:, 2] = layer[take]
    packed[:, 3] = layer[take] if intensity is None else intensity[take]
    msg = PointCloud2()
    msg.header = copy.deepcopy(header)
    msg.height = 1
    msg.width = len(rows)
    msg.fields = [PointField(name=n, offset=4*i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate(('x', 'y', 'z', 'intensity'))]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16*len(rows)
    msg.data = packed.tobytes()
    msg.is_dense = True
    return msg


def make_debug_map(source, layers):
    from grid_map_msgs.msg import GridMap
    from std_msgs.msg import Float32MultiArray, MultiArrayDimension
    result = GridMap()
    result.info = copy.deepcopy(source.info)
    result.layers = list(layers)
    result.basic_layers = ['height']
    for values in layers.values():
        rows, cols = values.shape
        array = Float32MultiArray()
        array.layout.dim = [MultiArrayDimension(label='column_index', size=cols, stride=rows*cols),
                            MultiArrayDimension(label='row_index', size=rows, stride=rows)]
        array.data = values.flatten(order='F').tolist()
        result.data.append(array)
    return result


def main():
    import rospy
    from grid_map_msgs.msg import GridMap
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import String
    rospy.init_node('lio_bev_surface_diagnostics')
    rate = float(rospy.get_param('~rate_hz', 1.0))
    roi = tuple(float(rospy.get_param('~'+key, default)) for key, default in
                [('x_min', -15), ('x_max', 15), ('y_min', -1.5), ('y_max', 1.5)])
    threshold = float(rospy.get_param('~inlier_threshold_m', 0.10))
    if not all(math.isfinite(v) for v in (rate, threshold)+roi) or not 0 < rate <= 10:
        raise ValueError('invalid diagnostic rate/ROI')
    if roi[0] >= roi[1] or roi[2] >= roi[3] or threshold <= 0:
        raise ValueError('invalid diagnostic ROI/threshold')
    input_topic = rospy.get_param('~input_topic', '/lidar_bev/grid_map')
    prefix = str(rospy.get_param('~output_prefix', '/lidar_bev/debug')).rstrip('/')
    if rospy.resolve_name(prefix+'/grid_map') == rospy.resolve_name(input_topic):
        raise ValueError('diagnostic output must differ from the input topic')
    pub_surface = rospy.Publisher(prefix+'/road_points', PointCloud2, queue_size=1, latch=True)
    pub_plane = rospy.Publisher(prefix+'/plane_points', PointCloud2, queue_size=1, latch=True)
    pub_residual = rospy.Publisher(prefix+'/residual_points', PointCloud2, queue_size=1, latch=True)
    pub_map = rospy.Publisher(prefix+'/grid_map', GridMap, queue_size=1, latch=True)
    pub_stats = rospy.Publisher(prefix+'/stats', String, queue_size=1, latch=True)
    latest = [None]
    mutex = threading.Lock()

    def receive(msg):
        with mutex:
            latest[0] = msg

    rospy.Subscriber(input_topic, GridMap, receive, queue_size=1, buff_size=8*1024*1024)
    previous = None
    stop = threading.Event()
    rospy.on_shutdown(stop.set)
    rospy.loginfo('BEV diagnostics: %s -> %s/*; %.1f Hz; ROI x=%s y=%s; no GUI',
                  input_topic, prefix, rate, roi[:2], roi[2:])
    while not rospy.is_shutdown():
        begin = time.monotonic()
        with mutex:
            msg = latest[0]
        if msg is not None and msg is not previous:
            previous = msg
            try:
                h, valid, x, y = unpack_grid_map(msg)
                layers, stats = analyze_surface(h, valid, x, y, roi, threshold)
                header = msg.info.header
                stats['stamp_sec'] = header.stamp.to_sec()
                stats['frame_id'] = header.frame_id
                pub_surface.publish(make_points(layers['height'], x, y, header, layers['residual']))
                pub_plane.publish(make_points(layers['plane'], x, y, header))
                pub_residual.publish(make_points(layers['residual'], x, y, header))
                pub_map.publish(make_debug_map(msg, layers))
                stats['processing_ms'] = (time.monotonic()-begin)*1000
                pub_stats.publish(String(data=json.dumps(stats, allow_nan=False)))
                rospy.loginfo('BEV road: cells=%d slope=%.2fdeg |residual|p95=%.3fm '
                              'neighbor_jump_max=%s m inliers=%.1f%% process=%.1fms',
                              stats['valid_cells'], stats['plane_slope_deg'],
                              stats['residual_abs_p95_m'], stats['neighbor_residual_jump_max_m'],
                              stats['inlier_fraction']*100, stats['processing_ms'])
            except (ValueError, IndexError, np.linalg.LinAlgError) as error:
                rospy.logwarn_throttle(5.0, 'BEV diagnostics unavailable: %s', str(error))
        stop.wait(max(0, 1.0/rate-(time.monotonic()-begin)))


if __name__ == '__main__':
    main()
