#!/usr/bin/env python3
"""
YOLO 检测结果回显工具（含 OBB 旋转角度，角度以"度"直出）。

用于替代 `ros2 topic echo /yolo/detections`（后者只能打印弧度 theta），
本工具把每个检测框的角度直接打印成"度"。

用法:
    # 默认监听 /yolo/detections (带命名空间)
    python3 echo_angle.py

    # 无命名空间时
    python3 echo_angle.py /detections

依赖:
    source /opt/ros/humble/setup.bash
    source /home/unitree/YOLO_ws/install/setup.bash
    conda activate YOLO (或 unitree_lerobot)
"""

import sys
import math
import rclpy
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from yolo_msgs.msg import DetectionArray


class AngleEcho:
    def __init__(self):
        rclpy.init()
        self.node = rclpy.create_node("echo_angle")
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        # YOLO 节点默认在 yolo 命名空间下，话题为 /yolo/detections
        self.topic = sys.argv[1] if len(sys.argv) > 1 else "/yolo/detections"
        self.sub = self.node.create_subscription(
            DetectionArray, self.topic, self.callback, qos
        )
        print("=" * 68)
        print("  YOLO 检测结果回显（OBB 角度单位：度）")
        print(f"  监听话题: {self.topic}")
        print("  用法: python3 echo_angle.py [话题名]")
        print("      默认 /yolo/detections | 无命名空间时用 /detections")
        print("  按 Ctrl+C 退出")
        print("=" * 68)

    def callback(self, msg):
        for det in msg.detections:
            bbox = det.bbox
            theta_rad = bbox.center.theta
            theta_deg = math.degrees(theta_rad)
            print(
                f"[{det.class_name:10s}]  "
                f"置信度={det.score:.2f}  "
                f"中心=({bbox.center.position.x:6.1f},{bbox.center.position.y:6.1f})  "
                f"尺寸=({bbox.size.x:5.1f}x{bbox.size.y:5.1f})  "
                f"角度={theta_deg:7.1f} 度  (弧度 {theta_rad:.4f})"
            )

    def run(self):
        try:
            rclpy.spin(self.node)
        except KeyboardInterrupt:
            print("\n退出回显。")
        finally:
            self.node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    try:
        AngleEcho().run()
    except rclpy.exceptions.RCLError as e:
        print(f"[错误] ROS2 初始化失败: {e}")
        print("请确保已执行:")
        print("  source /opt/ros/humble/setup.bash")
        print("  source /home/unitree/YOLO_ws/install/setup.bash")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] {e}")
        sys.exit(1)
