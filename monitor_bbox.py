#!/usr/bin/env python3
"""
YOLO 检测边界框面积监测工具。
用于观察 cloth 类别的边界框大小，帮助确定合适的停止阈值。

用法:
    # 默认监听 /yolo/detections (带命名空间)
    python3 monitor_bbox.py

    # 无命名空间时
    python3 monitor_bbox.py /detections

依赖:
    source /opt/ros/humble/setup.bash
    source /home/unitree/YOLO_ws/install/setup.bash
    conda activate YOLO (或 unitree_lerobot)
"""

import sys
import rclpy
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from yolo_msgs.msg import DetectionArray


class BboxMonitor:
    def __init__(self):
        rclpy.init()
        self.node = rclpy.create_node("bbox_monitor")
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        # YOLO 节点默认在 yolo 命名空间下，话题为 /yolo/detections
        # 如果启动时用了 namespace:=/ 则在 /detections
        import sys
        self.topic = sys.argv[1] if len(sys.argv) > 1 else "/yolo/detections"
        self.sub = self.node.create_subscription(
            DetectionArray, self.topic, self.callback, qos
        )
        print("=" * 60)
        print("  YOLO 边界框面积监测工具")
        print(f"  监听话题: {self.topic}")
        print("  用法: python3 monitor_bbox.py [话题名]")
        print("      默认 /yolo/detections | 无命名空间时用 /detections")
        print("  按 Ctrl+C 退出")
        print("=" * 60)

    def callback(self, msg):
        for det in msg.detections:
            bbox = det.bbox
            area = bbox.size.x * bbox.size.y
            max_possible = 480 * 640  # 图像分辨率
            ratio = area / max_possible * 100
            print(
                f"[{det.class_name:10s}]  "
                f"置信度={det.score:.2f}  "
                f"bbox=({bbox.size.x:.0f}×{bbox.size.y:.0f})  "
                f"面积={area:>8.0f}  "
                f"占画面={ratio:>5.1f}%"
            )

    def run(self):
        try:
            rclpy.spin(self.node)
        except KeyboardInterrupt:
            print("\n退出监测。")
        finally:
            self.node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    try:
        monitor = BboxMonitor()
        monitor.run()
    except rclpy.exceptions.RCLError as e:
        print(f"[错误] ROS2 初始化失败: {e}")
        print("请确保已执行:")
        print("  source /opt/ros/humble/setup.bash")
        print("  source /home/unitree/YOLO_ws/install/setup.bash")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] {e}")
        sys.exit(1)
