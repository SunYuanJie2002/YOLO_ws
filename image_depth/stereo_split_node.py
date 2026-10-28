#!/usr/bin/env python3
"""
双目图像拆分 ROS2 节点 (stereo_split_node)

功能：
  从 ZMQ 获取完整双目图像 (480×1280)，拆分为左/右视图 (480×640)，
  并发布到 ROS2 话题上，供 stereo_image_proc 做立体匹配生成深度图。

流程：
  1. ZMQ SUB 连接到 teleimager (tcp://192.168.123.164:55555)
  2. 接收双目图像 → 拆分为 left/right 各 480×640
  3. 发布 left/image_raw, right/image_raw, left/camera_info, right/camera_info
  4. stereo_image_proc 消费这些话题 → 生成深度图

用法：
  # 单独运行
  python3 stereo_split_node.py

  # 或通过启动脚本
  bash run_stereo_yolo.sh

依赖：
  source /opt/ros/humble/setup.bash
  source /home/unitree/YOLO_ws/install/setup.bash
  pip install zmq
"""

import cv2
import numpy as np
import zmq

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import Header
from cv_bridge import CvBridge


class StereoSplitNode(Node):
    """
    双目图像拆分节点。

    从 ZMQ 读取双目图像，拆分为左右视图，分别发布图像和相机内参。
    """

    # ============================================================
    # 标定结果（来自 calibrate_stereo.py 的 stereoCalibrate）
    # 标定时间: 2026-06-04
    # ============================================================
    IMG_HEIGHT = 480
    IMG_WIDTH = 640

    FX_DEFAULT = 274.8
    FY_DEFAULT = 272.6
    CX_DEFAULT = 320.0
    CY_DEFAULT = 240.0
    BASELINE_DEFAULT = 0.0835  # 8.35 cm

    def __init__(self):
        super().__init__("stereo_split_node")

        # ---- 参数 ----
        self.declare_parameter("zmq_host", "192.168.123.164")
        self.declare_parameter("zmq_port", 55555)
        self.declare_parameter("zmq_timeout_ms", 5)
        self.declare_parameter("zmq_poll_period", 0.01)

        self.declare_parameter("fx", self.FX_DEFAULT)
        self.declare_parameter("fy", self.FY_DEFAULT)
        self.declare_parameter("cx", self.CX_DEFAULT)
        self.declare_parameter("cy", self.CY_DEFAULT)
        self.declare_parameter("baseline", self.BASELINE_DEFAULT)

        self.zmq_host = (
            self.get_parameter("zmq_host").get_parameter_value().string_value
        )
        self.zmq_port = (
            self.get_parameter("zmq_port").get_parameter_value().integer_value
        )
        self.zmq_timeout_ms = (
            self.get_parameter("zmq_timeout_ms").get_parameter_value().integer_value
        )
        self.zmq_poll_period = (
            self.get_parameter("zmq_poll_period").get_parameter_value().double_value
        )

        self.fx = self.get_parameter("fx").get_parameter_value().double_value
        self.fy = self.get_parameter("fy").get_parameter_value().double_value
        self.cx = self.get_parameter("cx").get_parameter_value().double_value
        self.cy = self.get_parameter("cy").get_parameter_value().double_value
        self.baseline = self.get_parameter("baseline").get_parameter_value().double_value

        # ---- 发布者 ----
        self.left_pub = self.create_publisher(Image, "left/image_raw", 10)
        self.right_pub = self.create_publisher(Image, "right/image_raw", 10)
        self.left_compressed_pub = self.create_publisher(
            CompressedImage, "left/image_raw/compressed", 10
        )
        self.left_info_pub = self.create_publisher(
            CameraInfo, "left/camera_info", 10
        )
        self.right_info_pub = self.create_publisher(
            CameraInfo, "right/camera_info", 10
        )

        # ---- ZMQ ----
        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.SUB)
        self._zmq_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._zmq_socket.setsockopt(zmq.LINGER, 0)
        self._zmq_socket.setsockopt(zmq.CONFLATE, 1)
        self._zmq_socket.setsockopt(zmq.RCVHWM, 1)
        self._zmq_socket.setsockopt(zmq.RCVTIMEO, self.zmq_timeout_ms)
        endpoint = f"tcp://{self.zmq_host}:{self.zmq_port}"
        self._zmq_socket.connect(endpoint)
        self.get_logger().info(f"Connected to ZMQ: {endpoint}")

        # ---- 帧计数器（压缩图像跳帧用）----
        self._frame_count = 0

        # ---- 定时器轮询 ----
        self._timer = self.create_timer(self.zmq_poll_period, self._poll_zmq)

        self.get_logger().info(
            f"StereoSplitNode started. "
            f"Publishing left/right images to "
            f"'left/image_raw', 'right/image_raw'"
        )

    # ------------------------------------------------------------------
    #  ZMQ 轮询
    # ------------------------------------------------------------------

    def _poll_zmq(self):
        """
        从 ZMQ socket 轮询一帧双目图像 → 拆分 → 发布。
        """
        try:
            msg_parts = self._zmq_socket.recv_multipart(flags=zmq.NOBLOCK)
            # 排空旧帧，只保留最新
            while True:
                msg_parts = self._zmq_socket.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            if "msg_parts" not in locals():
                return
        except Exception as exc:
            self.get_logger().warn(f"ZMQ error: {exc}", throttle_duration_sec=5.0)
            return

        # 解码图像
        cv_image = self._decode_zmq_image(msg_parts)

        if cv_image is None:
            return

        # 检查尺寸
        if cv_image.shape[0] != self.IMG_HEIGHT or cv_image.shape[1] != self.IMG_WIDTH * 2:
            self.get_logger().warn(
                f"Unexpected frame shape: {cv_image.shape}, "
                f"expected ({self.IMG_HEIGHT}, {self.IMG_WIDTH * 2})",
                throttle_duration_sec=5.0,
            )
            return

        # ---- 拆分左右 ----
        left_img = cv_image[:, : self.IMG_WIDTH]
        right_img = cv_image[:, self.IMG_WIDTH :]

        # ---- 时间戳 ----
        stamp = self.get_clock().now().to_msg()

        # ---- 帧计数 ----
        self._frame_count += 1

        # ---- 发布图像（原始 + 压缩）----
        self._publish_image(self.left_pub, left_img, "left_camera", stamp)
        self._publish_image(self.right_pub, right_img, "right_camera", stamp)
        # 压缩版仅发布左目，且每 2 帧跳过 1 帧以降低 CPU 和网络负载
        if self._frame_count % 2 == 0:
            self._publish_compressed(self.left_compressed_pub, left_img, "left_camera", stamp)

        # ---- 发布 CameraInfo ----
        self._publish_camera_info(self.left_info_pub, "left_camera", stamp)
        self._publish_camera_info(self.right_info_pub, "right_camera", stamp)

    # ------------------------------------------------------------------
    #  图像解码（与 yolo_node 相同的逻辑）
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_zmq_image(msg_parts):
        """解码 ZMQ 载荷为 OpenCV BGR 图像"""
        if not msg_parts:
            return None

        payload = msg_parts[-1]
        image_array = np.frombuffer(payload, dtype=np.uint8)

        # 尝试 JPEG 解码
        cv_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if cv_image is not None:
            return cv_image

        height = StereoSplitNode.IMG_HEIGHT
        width = StereoSplitNode.IMG_WIDTH * 2

        # 尝试原始 RGB uint8
        if image_array.size == height * width * 3:
            return image_array.reshape(height, width, 3)

        # 尝试原始灰度 uint8
        if image_array.size == height * width:
            gray = image_array.reshape(height, width)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return None

    # ------------------------------------------------------------------
    #  发布 ROS2 Image
    # ------------------------------------------------------------------

    def _publish_image(self, publisher, cv_img, frame_id, stamp):
        """发布 OpenCV 图像为 ROS2 Image 消息（无需 cv_bridge）"""
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = cv_img.shape[0]
        msg.width = cv_img.shape[1]
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = cv_img.shape[1] * 3
        msg.data = cv_img.tobytes()
        publisher.publish(msg)

    def _publish_compressed(self, publisher, cv_img, frame_id, stamp):
        """发布 OpenCV 图像为压缩 JPEG 消息，降低网络带宽"""
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.format = "jpeg"
        success, encoded = cv2.imencode(".jpg", cv_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if success:
            msg.data = encoded.tobytes()
            publisher.publish(msg)

    # ------------------------------------------------------------------
    #  发布 CameraInfo
    # ------------------------------------------------------------------

    def _publish_camera_info(self, publisher, frame_id, stamp):
        """
        发布相机内参。

        注意：内参为估算值，标定后应替换。
        """
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id

        msg.height = self.IMG_HEIGHT
        msg.width = self.IMG_WIDTH

        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # 无畸变（假设已标定校正）

        msg.k = [
            self.fx, 0.0, self.cx,
            0.0, self.fy, self.cy,
            0.0, 0.0, 1.0,
        ]

        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        # 投影矩阵 P = [fx' 0  cx'  Tx; 0  fy'  cy'  0; 0  0  1  0]
        # Tx = -fx' * baseline  (右目相对于左目)
        if frame_id == "left_camera":
            tx = 0.0
        else:
            tx = -self.fx * self.baseline

        msg.p = [
            self.fx, 0.0, self.cx, tx,
            0.0, self.fy, self.cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]

        publisher.publish(msg)


# ======================================================================
#  Main
# ======================================================================

def main(args=None):
    rclpy.init(args=args)
    node = StereoSplitNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as exc:
        node.get_logger().error(f"Unexpected error: {exc}")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
