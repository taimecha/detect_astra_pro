#!/usr/bin/env python3
"""Publish a separate, range-limited depth preview from three measured anchors."""

import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField


def correct_ray_range(ray_range, raw_ranges, measured_ranges, boundary_margin_m):
    """Return measured ray ranges; use NaN outside the measured interval."""
    valid = (
        np.isfinite(ray_range)
        & (ray_range >= raw_ranges[0] - boundary_margin_m)
        & (ray_range <= raw_ranges[-1] + boundary_margin_m)
    )
    corrected = np.full(ray_range.shape, np.nan, dtype=np.float32)
    if np.any(valid):
        values = ray_range[valid]
        lower = values <= raw_ranges[1]
        mapped = np.empty_like(values, dtype=np.float32)
        mapped[lower] = measured_ranges[0] + (
            (values[lower] - raw_ranges[0])
            * (measured_ranges[1] - measured_ranges[0])
            / (raw_ranges[1] - raw_ranges[0])
        )
        mapped[~lower] = measured_ranges[1] + (
            (values[~lower] - raw_ranges[1])
            * (measured_ranges[2] - measured_ranges[1])
            / (raw_ranges[2] - raw_ranges[1])
        )
        corrected[valid] = mapped
    return corrected


def correct_depth(depth, ray_norm, raw_ranges, measured_ranges, boundary_margin_m=0.0):
    """Map optical-axis depth through a piecewise-linear ray-range calibration."""
    ray_range = depth * ray_norm
    corrected_range = correct_ray_range(
        ray_range, raw_ranges, measured_ranges, boundary_margin_m
    )
    return corrected_range / ray_norm


def cloud_xyz_view(msg, data):
    """View the XYZ fields of an organized cloud without changing other fields."""
    xyz_fields = {field.name: field for field in msg.fields if field.name in ('x', 'y', 'z')}
    if (
        set(xyz_fields) != {'x', 'y', 'z'}
        or any(field.datatype != PointField.FLOAT32 or field.count != 1
               or field.offset + 4 > msg.point_step for field in xyz_fields.values())
        or msg.point_step <= 0 or msg.row_step < msg.width * msg.point_step
        or len(msg.data) < msg.row_step * msg.height
    ):
        raise ValueError('Expected PointCloud2 with float32 x, y, z fields')

    endian = '>' if msg.is_bigendian else '<'
    xyz_type = np.dtype({
        'names': ['x', 'y', 'z'],
        'formats': [endian + 'f4'] * 3,
        'offsets': [xyz_fields[name].offset for name in ('x', 'y', 'z')],
        'itemsize': msg.point_step,
    })
    return np.ndarray(
        shape=(msg.height, msg.width), dtype=xyz_type, buffer=data,
        strides=(msg.row_step, msg.point_step)
    )


def cloud_with_data(msg, data):
    result = PointCloud2()
    result.header = msg.header
    result.height = msg.height
    result.width = msg.width
    result.fields = msg.fields
    result.is_bigendian = msg.is_bigendian
    result.point_step = msg.point_step
    result.row_step = msg.row_step
    result.data = data
    result.is_dense = False
    return result


def correct_point_cloud(msg, raw_ranges, measured_ranges, boundary_margin_m):
    """Scale measured XYZ rays while retaining their RGB and timestamp."""
    data = bytearray(msg.data)
    points = cloud_xyz_view(msg, data)
    x, y, z = (points[name] for name in ('x', 'y', 'z'))
    ray_range = np.sqrt(x * x + y * y + z * z)
    corrected_range = correct_ray_range(
        ray_range, raw_ranges, measured_ranges, boundary_margin_m
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        scale = corrected_range / ray_range
    for component in (x, y, z):
        component *= scale

    return cloud_with_data(msg, data)


def uncalibrated_point_cloud(msg, raw_ranges, boundary_margin_m):
    """Keep original XYZ only outside the measured calibration band."""
    data = bytearray(msg.data)
    points = cloud_xyz_view(msg, data)
    x, y, z = (points[name] for name in ('x', 'y', 'z'))
    ray_range = np.sqrt(x * x + y * y + z * z)
    calibrated_band = (
        np.isfinite(ray_range)
        & (ray_range >= raw_ranges[0] - boundary_margin_m)
        & (ray_range <= raw_ranges[-1] + boundary_margin_m)
    )
    for component in (x, y, z):
        component[calibrated_band] = np.nan
    return cloud_with_data(msg, data)


class DepthCalibrationPreview(Node):
    def __init__(self):
        super().__init__('ai_depth_calibration_preview')
        default_config = (
            Path(__file__).resolve().parent.parent
            / 'config'
            / 'ai_depth_range_calibration.json'
        )
        config_path = Path(
            self.declare_parameter('calibration_file', str(default_config)).value
        )
        config = json.loads(config_path.read_text())
        self.raw_ranges = np.asarray(config['raw_ray_ranges_m'], dtype=np.float32)
        self.measured_ranges = np.asarray(
            config['measured_ray_ranges_m'], dtype=np.float32
        )
        self.boundary_margin_m = float(config['boundary_margin_m'])
        if (
            len(self.raw_ranges) != 3
            or len(self.measured_ranges) != 3
            or not np.all(np.isfinite(self.raw_ranges))
            or not np.all(np.isfinite(self.measured_ranges))
            or not np.all(np.diff(self.raw_ranges) > 0)
            or not np.all(np.diff(self.measured_ranges) > 0)
            or not 0.0 <= self.boundary_margin_m <= 0.1
            or config.get('outside_calibrated_range') != 'nan'
        ):
            raise ValueError('Calibration requires three increasing ray-range anchors')

        input_topic = self.declare_parameter(
            'input_depth_topic', '/camera/ai_depth/image'
        ).value
        info_topic = self.declare_parameter(
            'camera_info_topic', '/camera/rgb/camera_info'
        ).value
        output_topic = self.declare_parameter(
            'output_depth_topic', '/camera/ai_depth_calibrated/image'
        ).value
        input_cloud_topic = self.declare_parameter(
            'input_cloud_topic', '/camera/ai_depth/points'
        ).value
        output_cloud_topic = self.declare_parameter(
            'output_cloud_topic', '/camera/ai_depth_calibrated/points'
        ).value
        uncalibrated_cloud_topic = self.declare_parameter(
            'uncalibrated_cloud_topic', '/camera/ai_depth_uncalibrated/points'
        ).value
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.publisher = self.create_publisher(Image, output_topic, qos)
        self.cloud_publisher = self.create_publisher(PointCloud2, output_cloud_topic, qos)
        self.uncalibrated_cloud_publisher = self.create_publisher(
            PointCloud2, uncalibrated_cloud_topic, qos
        )
        self.create_subscription(CameraInfo, info_topic, self.on_info, qos)
        self.create_subscription(Image, input_topic, self.on_depth, qos)
        self.create_subscription(PointCloud2, input_cloud_topic, self.on_cloud, qos)
        self.info = None
        self.ray_norm = None
        self.ray_norm_key = None
        self.get_logger().info(
            f'Preview calibration active: {input_topic} -> {output_topic}, '
            f'{input_cloud_topic} -> {output_cloud_topic} and '
            f'{uncalibrated_cloud_topic}; '
            f'valid AI ray range '
            f'{self.raw_ranges[0] - self.boundary_margin_m:.2f}-'
            f'{self.raw_ranges[-1] + self.boundary_margin_m:.2f} m'
        )

    def on_info(self, msg):
        self.info = msg

    def on_depth(self, msg):
        info = self.info
        if info is None:
            return
        if msg.encoding != '32FC1' or msg.step < msg.width * 4 or msg.step % 4:
            self.get_logger().error('Expected a 32FC1 depth image with 4-byte row stride')
            return
        if (msg.width, msg.height) != (info.width, info.height):
            self.get_logger().error('Depth image and CameraInfo dimensions differ')
            return
        if msg.header.frame_id != info.header.frame_id:
            self.get_logger().error('Depth image and CameraInfo frame IDs differ')
            return
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        if fx <= 0 or fy <= 0:
            self.get_logger().error('CameraInfo has invalid focal lengths')
            return
        key = (msg.width, msg.height, fx, fy, cx, cy)
        if key != self.ray_norm_key:
            x = (np.arange(msg.width, dtype=np.float32) - cx) / fx
            y = (np.arange(msg.height, dtype=np.float32) - cy) / fy
            self.ray_norm = np.sqrt(1.0 + x[None, :] ** 2 + y[:, None] ** 2)
            self.ray_norm_key = key

        dtype = '>f4' if msg.is_bigendian else '<f4'
        depth = np.frombuffer(msg.data, dtype=dtype).reshape(
            msg.height, msg.step // 4
        )[:, :msg.width]
        calibrated = correct_depth(
            depth, self.ray_norm, self.raw_ranges, self.measured_ranges,
            self.boundary_margin_m
        )
        result = Image()
        result.header = msg.header
        result.height = msg.height
        result.width = msg.width
        result.encoding = '32FC1'
        result.is_bigendian = False
        result.step = msg.width * 4
        result.data = calibrated.tobytes()
        self.publisher.publish(result)

    def on_cloud(self, msg):
        try:
            result = correct_point_cloud(
                msg, self.raw_ranges, self.measured_ranges, self.boundary_margin_m
            )
            uncalibrated = uncalibrated_point_cloud(
                msg, self.raw_ranges, self.boundary_margin_m
            )
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return
        self.cloud_publisher.publish(result)
        self.uncalibrated_cloud_publisher.publish(uncalibrated)


def main():
    rclpy.init()
    node = DepthCalibrationPreview()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
