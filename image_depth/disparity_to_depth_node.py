#!/usr/bin/env python3
"""
视差 → 深度 转换节点 (disparity_to_depth_node)

功能：
  订阅 stereo_image_proc 输出的视差话题 (stereo_msgs/Disparity)，
  转换为深度图像 (sensor_msgs/Image, 32FC1, 单位:米)，
  供 detect_3d_node 使用。

深度公式:  Z = f * B / d = t / d

其中:
  f = 焦距 (pixel)
  B = 基线 (米)
  d = 视差 (pixel)
  t = disparity.t  (f * B 的乘积，单位为米×像素)

用法:
  python3 disparity_to_depth_node.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage
import numpy as np


class DisparityToDepthNode(Node):
    """
    将 stereo_msgs/Disparity 转换为深度图像。
    """

    def __init__(self):
        super().__init__("disparity_to_depth_node")

        # ---- 参数 ----
        self.declare_parameter("disparity_topic", "/left/disparity")
        self.declare_parameter("depth_topic", "/left/depth")
        self.declare_parameter("depth_info_topic", "/left/camera_info")
        self.declare_parameter("max_depth", 10.0)  # 最大有效深度 (米)
        self.declare_parameter("min_depth", 0.1)  # 最小有效深度 (米)

        disparity_topic = (
            self.get_parameter("disparity_topic").get_parameter_value().string_value
        )
        depth_topic = (
            self.get_parameter("depth_topic").get_parameter_value().string_value
        )
        self.max_depth = (
            self.get_parameter("max_depth").get_parameter_value().double_value
        )
        self.min_depth = (
            self.get_parameter("min_depth").get_parameter_value().double_value
        )

        # ---- 发布者 ----
        self.depth_pub = self.create_publisher(Image, depth_topic, 10)

        # ---- 订阅者 ----
        self.disparity_sub = self.create_subscription(
            DisparityImage, disparity_topic, self.disparity_cb, 10
        )

        self.get_logger().info(
            f"DisparityToDepthNode started.\n"
            f"  订阅:  {disparity_topic}\n"
            f"  发布:  {depth_topic}\n"
            f"  深度范围: {self.min_depth} ~ {self.max_depth} m"
        )

    def disparity_cb(self, msg: DisparityImage):
        """
        将视差消息转换为深度图像并发布。
        深度公式: Z = f * T / d
          f = msg.f (焦距, 像素)
          T = msg.t (基线, 米)
          d = 视差值 (像素)
        """
        img_msg = msg.image

        # 手动解码图像数据（无需 cv_bridge）
        try:
            dtype = self._encoding_to_dtype(img_msg.encoding)
            disparity_img = np.frombuffer(
                img_msg.data, dtype=dtype
            ).reshape(img_msg.height, img_msg.width).astype(np.float32)
        except Exception as e:
            self.get_logger().warn(f"视差解码失败: {e}", throttle_duration_sec=5.0)
            return

        # Z = f * T / d
        f = msg.f   # 焦距 (像素)
        t = msg.t   # 基线 (米)
        if f <= 0 or t <= 0:
            return

        valid = disparity_img > 0.0
        depth_img = np.zeros_like(disparity_img)
        depth_img[valid] = (f * t) / disparity_img[valid]
        depth_img = np.clip(depth_img, self.min_depth, self.max_depth)
        depth_img[~valid] = 0.0

        # 手动构造深度图 Image 消息
        depth_msg = Image()
        depth_msg.header = msg.header
        depth_msg.height = img_msg.height
        depth_msg.width = img_msg.width
        depth_msg.encoding = "32FC1"
        depth_msg.is_bigendian = False
        depth_msg.step = img_msg.width * 4  # float32 = 4 bytes
        depth_msg.data = depth_img.tobytes()
        self.depth_pub.publish(depth_msg)

    @staticmethod
    def _encoding_to_dtype(encoding: str):
        """将 ROS Image encoding 转换为 numpy dtype"""
        enc_map = {
            "32FC1": np.float32,
            "32FC3": np.float32,
            "16SC1": np.int16,
            "16UC1": np.uint16,
            "8UC1": np.uint8,
            "8UC3": np.uint8,
            "mono8": np.uint8,
            "mono16": np.uint16,
            "bgr8": np.uint8,
            "rgb8": np.uint8,
        }
        if encoding in enc_map:
            return enc_map[encoding]
        # fallback: try to parse generic encoding like "32FC1"
        import re
        m = re.match(r"(\d+)([A-Za-z]+)(\d*)", encoding)
        if m:
            bits = int(m.group(1))
            signed = m.group(2).startswith("S")
            if bits == 32:
                return np.float32
            elif bits == 16:
                return np.int16 if signed else np.uint16
            elif bits == 8:
                return np.int8 if signed else np.uint8
        return np.float32  # default


def main(args=None):
    rclpy.init(args=args)
    node = DisparityToDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
