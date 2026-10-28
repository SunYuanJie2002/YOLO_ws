#!/usr/bin/env python3
"""
双目相机标定脚本（无需 cv_bridge / camera_calibration）

手动从 ROS2 话题读取左右目图像，用 OpenCV 检测棋盘格角点，
完成立体标定，输出内参 + 外参。

用法:
  1. 启动 stereo_split_node (python3 stereo_split_node.py)
  2. 运行本脚本:  python3 calibrate_stereo.py
  3. 将打印的棋盘格放在相机前移动，终端会显示检测状态
  4. 收集约 20~30 帧有效图像后自动标定
  5. 标定结果保存到 calibration_result.yaml

棋盘格参数 (默认): 内角点 9×7, 每格 25mm
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import sys
import os
import yaml
import time


# ============================================================
#  配置参数
# ============================================================
PATTERN_SIZE = (9, 7)       # 内角点数 (宽, 高)
SQUARE_SIZE = 0.025          # 每格边长 (米) = 25mm
MIN_FRAMES = 25              # 最少有效帧数
MAX_FRAMES = 40              # 最多有效帧数
SHOW_PREVIEW = False         # 是否显示预览窗口（无 GUI 时设为 False）


class StereoCalibrator(Node):
    def __init__(self):
        super().__init__("stereo_calibrator")

        self.left_images = []   # 存储左目角点
        self.right_images = []  # 存储右目角点
        self.left_img_shape = None
        self.right_img_shape = None
        self.frame_count = 0
        self.calibrated = False

        # 准备棋盘格 3D 点 (所有帧共享)
        self.objp = np.zeros(
            (PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32
        )
        self.objp[:, :2] = np.mgrid[
            0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]
        ].T.reshape(-1, 2) * SQUARE_SIZE
        self.object_points = []  # 3D 点集

        # 订阅左右目图像
        self.left_sub = self.create_subscription(
            Image, "/left/image_raw", self.left_cb, 10
        )
        self.right_sub = self.create_subscription(
            Image, "/right/image_raw", self.right_cb, 10
        )

        self.left_latest = None
        self.right_latest = None
        self.lock = False

        self.get_logger().info(
            "=" * 60
        )
        self.get_logger().info("  双目相机标定工具")
        self.get_logger().info(f"  棋盘格: {PATTERN_SIZE[0]}x{PATTERN_SIZE[1]} 内角点, {SQUARE_SIZE*1000:.0f}mm/格")
        self.get_logger().info(f"  需要: {MIN_FRAMES}~{MAX_FRAMES} 帧有效图像")
        self.get_logger().info("=" * 60)
        self.get_logger().info("")
        self.get_logger().info("【操作说明】")
        self.get_logger().info("  将打印的棋盘格放在相机前, 缓慢移动")
        self.get_logger().info("  改变角度、距离、倾斜 (远/近/左/右/俯/仰)")
        self.get_logger().info("  检测到角点时终端会打印 + 计数")
        self.get_logger().info("  收集足够帧后自动标定")
        self.get_logger().info("")

        if SHOW_PREVIEW:
            cv2.namedWindow("Left Camera", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Left Camera", 640, 480)
            cv2.namedWindow("Right Camera", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Right Camera", 640, 480)

    def left_cb(self, msg: Image):
        if self.lock or self.calibrated:
            return
        img = self._msg_to_cv2(msg)
        if img is not None:
            self.left_latest = (img, msg.header.stamp)

    def right_cb(self, msg: Image):
        if self.lock or self.calibrated:
            return
        img = self._msg_to_cv2(msg)
        if img is not None:
            self.right_latest = (img, msg.header.stamp)

    def _msg_to_cv2(self, msg: Image):
        """手动将 ROS Image 转为 OpenCV BGR (无需 cv_bridge)"""
        dtype_map = {
            "bgr8": np.uint8,
            "rgb8": np.uint8,
            "mono8": np.uint8,
            "16UC1": np.uint16,
            "32FC1": np.float32,
        }
        dt = dtype_map.get(msg.encoding, np.uint8)
        try:
            arr = np.frombuffer(msg.data, dtype=dt).reshape(
                msg.height, msg.width, -1
            )
            if arr.shape[2] == 1:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif msg.encoding == "rgb8":
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return arr
        except Exception as e:
            self.get_logger().warn(f"图像转换失败: {e}")
            return None

    def process_frame(self):
        """处理一帧: 检测棋盘格角点"""
        if self.left_latest is None or self.right_latest is None:
            return

        l_img, l_stamp = self.left_latest
        r_img, r_stamp = self.right_latest

        # 【核心修正】: 严格比对时间戳，杜绝左右眼接收时间差带来的巨大运动视差！
        if l_stamp.sec != r_stamp.sec or l_stamp.nanosec != r_stamp.nanosec:
            return

        self.lock = True
        left_img = l_img
        right_img = r_img
        self.left_latest = None
        self.right_latest = None

        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        if self.left_img_shape is None:
            self.left_img_shape = gray_left.shape[::-1]
            self.right_img_shape = gray_right.shape[::-1]

        found_left, corners_left = cv2.findChessboardCorners(
            gray_left, PATTERN_SIZE, None,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        found_right, corners_right = cv2.findChessboardCorners(
            gray_right, PATTERN_SIZE, None,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found_left and found_right:
            # 亚像素精化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

            # 修正由于 9x7 对称棋盘格导致的角点颠倒 (180度翻转) 问题
            dist_normal = np.linalg.norm(corners_left[0] - corners_right[0])
            dist_flipped = np.linalg.norm(corners_left[0] - corners_right[-1])
            if dist_flipped < dist_normal:
                corners_right = corners_right[::-1]

            self.object_points.append(self.objp)
            self.left_images.append(corners_left)
            self.right_images.append(corners_right)
            self.frame_count += 1

            # 绘制角点
            cv2.drawChessboardCorners(left_img, PATTERN_SIZE, corners_left, found_left)
            cv2.drawChessboardCorners(right_img, PATTERN_SIZE, corners_right, found_right)

            self.get_logger().info(
                f"  ✓ 第 {self.frame_count:2d}/{MAX_FRAMES} 帧  "
                f"检测到棋盘格!  ({len(self.left_images)} 帧已收集)"
            )

            if self.frame_count >= MAX_FRAMES:
                self.run_calibration()
        else:
            side = "左" if not found_left else ""
            side += "右" if not found_right else ""
            self.get_logger().info(
                f"  ✗ {side}目未检测到棋盘格，换个角度试试",
                throttle_duration_sec=2.0
            )

        if SHOW_PREVIEW:
            cv2.imshow("Left Camera", left_img)
            cv2.imshow("Right Camera", right_img)
            cv2.waitKey(1)
        elif self.frame_count in [1, 5, 10, 20, 30, MAX_FRAMES]:
            # 保存示例图像供远程查看
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_samples")
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(f"{out_dir}/left_{self.frame_count:02d}.png", left_img)
            cv2.imwrite(f"{out_dir}/right_{self.frame_count:02d}.png", right_img)

        self.lock = False

    def run_calibration(self):
        """执行立体标定"""
        if len(self.object_points) < MIN_FRAMES:
            self.get_logger().warn(
                f"帧数不足 ({len(self.object_points)} < {MIN_FRAMES})，继续收集"
            )
            return

        self.calibrated = True
        self.get_logger().info("")
        self.get_logger().info("=" * 60)
        self.get_logger().info("  正在运行立体标定...")
        self.get_logger().info("=" * 60)

        # 初始内参猜测
        w, h = self.left_img_shape
        K_init_l = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
        D_init_l = np.zeros((5, 1), dtype=np.float64)
        K_init_r = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
        D_init_r = np.zeros((5, 1), dtype=np.float64)

        # 先分别标定左右单目
        self.get_logger().info("  标定左目...")
        ret_l, K_l, D_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
            self.object_points, self.left_images,
            self.left_img_shape, K_init_l, D_init_l,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT
        )
        self.get_logger().info(f"    左目重投影误差: {ret_l:.4f} px")

        self.get_logger().info("  标定右目...")
        ret_r, K_r, D_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
            self.object_points, self.right_images,
            self.right_img_shape, K_init_r, D_init_r,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT
        )
        self.get_logger().info(f"    右目重投影误差: {ret_r:.4f} px")

        # 立体标定
        self.get_logger().info("  立体标定 (stereoCalibrate)...")
        flags = cv2.CALIB_FIX_INTRINSIC
        ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            self.object_points,
            self.left_images, self.right_images,
            K_l, D_l, K_r, D_r,
            self.left_img_shape,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
            flags=flags
        )
        self.get_logger().info(f"    立体标定误差: {ret:.4f} px")

        # 立体校正
        self.get_logger().info("  计算立体校正映射...")
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2,
            self.left_img_shape, R, T,
            alpha=0,  # 0=裁剪黑边, 1=保留全部
        )

        # 计算基线
        baseline = np.linalg.norm(T)

        # ---- 输出结果 ----
        self.get_logger().info("")
        self.get_logger().info("=" * 60)
        self.get_logger().info("  ★ 标定完成! 结果如下:")
        self.get_logger().info("=" * 60)
        self.get_logger().info("")

        result = {
            "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_width": w,
            "image_height": h,
            "pattern_size": list(PATTERN_SIZE),
            "square_size_m": SQUARE_SIZE,
            "num_frames": len(self.object_points),
            "stereo_error_px": float(ret),

            "left": {
                "k": K1.flatten().tolist(),
                "d": D1.flatten().tolist(),
                "reprojection_error_px": float(ret_l),
            },
            "right": {
                "k": K2.flatten().tolist(),
                "d": D2.flatten().tolist(),
                "reprojection_error_px": float(ret_r),
            },
            "stereo": {
                "R": R.flatten().tolist(),
                "T": T.flatten().tolist(),
                "baseline_m": float(baseline),
                "R1": R1.flatten().tolist(),
                "R2": R2.flatten().tolist(),
                "P1": P1.flatten().tolist(),
                "P2": P2.flatten().tolist(),
                "Q": Q.flatten().tolist(),
            },
        }

        # 保存到文件
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "calibration_result.yaml"
        )
        with open(output_path, "w") as f:
            yaml.dump(result, f, default_flow_style=None, sort_keys=False)

        self.get_logger().info(f"  结果已保存到: {output_path}")
        self.get_logger().info("")

        # 打印易读摘要
        self._print_summary(result, K1, D1, K2, D2, R, T, baseline)

        self.get_logger().info("")
        self.get_logger().info("按 Ctrl+C 退出")

    def _print_summary(self, result, K1, D1, K2, D2, R, T, baseline):
        """打印易读的标定摘要"""

        def fmt_mat(name, mat):
            if mat.ndim == 1:
                vals = ", ".join(f"{v:.6f}" for v in mat)
            else:
                lines = []
                for row in mat:
                    lines.append("    [" + ", ".join(f"{v:>10.4f}" for v in row) + "]")
                vals = "\n" + "\n".join(lines)
            return f"  {name}: {vals}"

        self.get_logger().info("【标定参数摘要】")
        self.get_logger().info(f"  图像尺寸: {result['image_width']}x{result['image_height']}")
        self.get_logger().info(f"  标定帧数: {result['num_frames']}")
        self.get_logger().info(f"  立体误差: {result['stereo_error_px']:.4f} px")
        self.get_logger().info("")
        self.get_logger().info("【左目内参 K1】")
        self.get_logger().info(fmt_mat("K1", K1).replace("\n", "\n  "))
        self.get_logger().info(f"  畸变 D1: {D1.flatten().tolist()}")
        self.get_logger().info("")
        self.get_logger().info("【右目内参 K2】")
        self.get_logger().info(fmt_mat("K2", K2).replace("\n", "\n  "))
        self.get_logger().info(f"  畸变 D2: {D2.flatten().tolist()}")
        self.get_logger().info("")
        self.get_logger().info("【双目外参】")
        self.get_logger().info(f"  基线距离: {baseline*100:.1f} cm")
        self.get_logger().info(fmt_mat("R (旋转)", R))
        self.get_logger().info(fmt_mat("T (平移)", T.flatten()))
        self.get_logger().info("")
        self.get_logger().info("【更新 stereo_split_node.py 的参数】")
        self.get_logger().info(f"  fx={K1[0,0]:.1f}   fy={K1[1,1]:.1f}")
        self.get_logger().info(f"  cx={K1[0,2]:.1f}   cy={K1[1,2]:.1f}")
        self.get_logger().info(f"  baseline={baseline:.4f}  # {baseline*100:.1f} cm")
        self.get_logger().info("")
        self.get_logger().info("启动时传入:")
        self.get_logger().info(f"  python3 stereo_split_node.py \\")
        self.get_logger().info(f"    --ros-args -p fx:={K1[0,2]:.1f} -p fy:={K1[1,1]:.1f} \\")
        self.get_logger().info(f"    -p cx:={K1[0,2]:.1f} -p cy:={K1[1,2]:.1f} -p baseline:={baseline:.4f}")


def main(args=None):
    rclpy.init(args=args)
    calib = StereoCalibrator()

    # 定时器: 5Hz 处理帧
    timer = calib.create_timer(0.2, calib.process_frame)

    try:
        rclpy.spin(calib)
    except KeyboardInterrupt:
        pass
    finally:
        if not calib.calibrated and calib.frame_count >= MIN_FRAMES:
            calib.run_calibration()
        elif not calib.calibrated:
            calib.get_logger().info(f"仅收集了 {calib.frame_count} 帧，未达到 {MIN_FRAMES} 帧，不执行标定")

        if SHOW_PREVIEW:
            cv2.destroyAllWindows()
        calib.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
