#!/usr/bin/env python3
"""
立体匹配节点 (sgbm_stereo_node)

用 OpenCV 立体匹配算法（SGBM 或轻量 BM）替代 stereo_image_proc。
订阅左右目图像 + CameraInfo，输出视差图和深度图。
默认使用 StereoBM（对 CPU/边缘设备友好），可通过参数切换为 SGBM。

发布话题:
  /left/disparity   — 视差图 (stereo_msgs/DisparityImage)
  /left/depth       — 深度图 (sensor_msgs/Image, 32FC1, 米)

参数:
  stereo_algo        str    "bm" (默认, 快) 或 "sgbm" (精确但慢)
  num_disparities    int    视差等级数 (默认 48)
  downsample_factor  float  下采样比例 (默认 0.5, 越小越快但越粗糙)
  hole_fill_mode     str    孔洞填充: "none"/"light"(默认)/"medium"/"aggressive"
  temporal_smooth    float  时域平滑系数 0~0.9 (默认 0.0=关闭)
  use_clahe          bool   CLAHE 对比度增强 (默认 True)
  use_wls            bool   WLS 边缘引导滤波 (默认 True, 需 opencv-contrib)
  wls_lambda         float  WLS 平滑强度 (默认 6000)
  wls_sigma          float  WLS 边缘保持系数 (默认 1.2)
  max_depth          float  最大深度 (米)
  min_depth          float  最小深度 (米)
  calibration_file   str    标定文件路径

用法:
  python3 sgbm_stereo_node.py
  # 或带参数:
  # ros2 run ... sgbm_stereo_node --ros-args -p stereo_algo:=sgbm -p num_disparities:=48
"""

import cv2
import numpy as np
import yaml
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, RegionOfInterest, CompressedImage
from stereo_msgs.msg import DisparityImage

# ---- WLS 滤波器（opencv-contrib-python），不可用时静默降级 ----
try:
    from cv2 import ximgproc
    _WLS_AVAILABLE = True
except ImportError:
    _WLS_AVAILABLE = False


class SgbmStereoNode(Node):
    def __init__(self):
        super().__init__("sgbm_stereo_node")

        self.declare_parameter("max_depth", 10.0)
        self.declare_parameter("min_depth", 0.2)
        self.declare_parameter("calibration_file", "")
        self.declare_parameter("stereo_algo", "bm")         # "bm" 或 "sgbm"
        self.declare_parameter("num_disparities", 48)       # 视差等级数
        self.declare_parameter("downsample_factor", 0.5)    # 下采样比例
        self.declare_parameter("hole_fill_mode", "medium")   # none/light/medium/aggressive
        self.declare_parameter("temporal_smooth", 0.0)      # 时域平滑 0~0.9
        self.declare_parameter("use_clahe", True)           # CLAHE 对比度增强
        self.declare_parameter("use_wls", True)             # WLS 边缘引导滤波
        self.declare_parameter("wls_lambda", 6000.0)        # WLS 平滑强度
        self.declare_parameter("wls_sigma", 1.2)            # WLS 边缘保持系数
        self.max_depth = self.get_parameter("max_depth").value
        self.min_depth = self.get_parameter("min_depth").value
        self.stereo_algo = self.get_parameter("stereo_algo").value
        self.num_disparities = self.get_parameter("num_disparities").value
        self.downsample_factor = self.get_parameter("downsample_factor").value
        self.hole_fill_mode = self.get_parameter("hole_fill_mode").value
        self.temporal_smooth = max(0.0, min(0.9, self.get_parameter("temporal_smooth").value))
        self.use_clahe = self.get_parameter("use_clahe").value
        self.use_wls = self.get_parameter("use_wls").value and _WLS_AVAILABLE
        self.wls_lambda = self.get_parameter("wls_lambda").value
        self.wls_sigma = self.get_parameter("wls_sigma").value
        calib_path = self.get_parameter("calibration_file").value

        # CLAHE 增强器（提升低对比度图像纹理，帮助 BM 匹配）
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # WLS 滤波器缓存（运行时按需创建）
        self._wls_filter = None

        # 最近帧缓存
        self.left_img = None
        self.right_img = None
        self.left_info = None
        # 时域滤波用 — 上一帧深度
        self._prev_depth = None

        # 订阅
        self.create_subscription(Image, "/left/image_raw", self.left_cb, 10)
        self.create_subscription(Image, "/right/image_raw", self.right_cb, 10)
        self.create_subscription(CameraInfo, "/left/camera_info", self.info_cb, 10)

        # 发布
        self.disparity_pub = self.create_publisher(DisparityImage, "/left/disparity", 10)
        self.depth_pub = self.create_publisher(Image, "/left/depth", 10)
        self.depth_compressed_pub = self.create_publisher(
            CompressedImage, "/left/depth/compressed", 10
        )

        # 帧计数器
        self._frame_count = 0

        # ---- 加载标定参数，计算校正映射表 ----
        self.map1_l = self.map2_l = None
        self.map1_r = self.map2_r = None
        self.fx_rect = None
        self.baseline_rect = None

        self._load_calibration(calib_path)

        # 定时器 (10Hz)
        self.create_timer(0.1, self.process)

        self.get_logger().info(
            f"SgbmStereoNode 已启动 (算法={self.stereo_algo}, "
            f"视差数={self.num_disparities}, 降采样={self.downsample_factor}, "
            f"CLAHE={self.use_clahe}, WLS={self.use_wls}, "
            f"填洞={self.hole_fill_mode}, 时域平滑={self.temporal_smooth})"
        )

    # ------------------------------------------------------------------
    #  标定加载 & 校正映射
    # ------------------------------------------------------------------

    def _load_calibration(self, calib_path):
        """从 calibration_result.yaml 加载标定数据，生成校正映射表"""
        if not calib_path or not os.path.exists(calib_path):
            # 尝试默认路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            calib_path = os.path.join(script_dir, "calibration_result.yaml")
        if not os.path.exists(calib_path):
            self.get_logger().warn(
                f"标定文件未找到 ({calib_path})，使用无校正模式"
            )
            return

        with open(calib_path) as f:
            calib = yaml.safe_load(f)

        img_w = calib.get("image_width", 640)
        img_h = calib.get("image_height", 480)

        # 左目内参 & 畸变
        K1 = np.array(calib["left"]["k"], dtype=np.float64).reshape(3, 3)
        D1 = np.array(calib["left"]["d"], dtype=np.float64)
        # 右目内参 & 畸变
        K2 = np.array(calib["right"]["k"], dtype=np.float64).reshape(3, 3)
        D2 = np.array(calib["right"]["d"], dtype=np.float64)

        # 校正旋转矩阵 & 投影矩阵
        R1 = np.array(calib["stereo"]["R1"], dtype=np.float64).reshape(3, 3)
        R2 = np.array(calib["stereo"]["R2"], dtype=np.float64).reshape(3, 3)
        P1 = np.array(calib["stereo"]["P1"], dtype=np.float64).reshape(3, 4)
        P2 = np.array(calib["stereo"]["P2"], dtype=np.float64).reshape(3, 4)

        # 计算校正映射表
        self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(
            K1, D1, R1, P1[:3, :3], (img_w, img_h), cv2.CV_32FC1
        )
        self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(
            K2, D2, R2, P2[:3, :3], (img_w, img_h), cv2.CV_32FC1
        )

        # 校正后的焦距和基线
        self.fx_rect = P1[0, 0]
        self.baseline_rect = abs(P2[0, 3] / P1[0, 0]) if P1[0, 0] > 0 else 0.0426

        self.get_logger().info(
            f"极线校正已加载: fx={self.fx_rect:.1f}, "
            f"baseline={self.baseline_rect:.4f}m"
        )

    # ------------------------------------------------------------------

    def _decode_img(self, msg):
        """ROS Image → numpy array (无需 cv_bridge C++ 扩展)"""
        enc = msg.encoding
        if enc == "bgr8":
            dt = np.uint8
            ch = 3
        elif enc == "mono8" or enc == "8UC1":
            dt = np.uint8
            ch = 1
        elif enc == "32FC1":
            dt = np.float32
            ch = 1
        elif enc == "16UC1":
            dt = np.uint16
            ch = 1
        else:
            dt = np.uint8
            ch = 3
        arr = np.frombuffer(msg.data, dtype=dt).reshape(msg.height, msg.width, ch)
        if ch == 3:
            return arr
        return arr[:, :, 0] if ch == 1 else arr

    def left_cb(self, msg):
        try:
            self.left_img = self._decode_img(msg)
        except Exception as e:
            self.get_logger().warn(f"left decode failed: {e}")

    def right_cb(self, msg):
        try:
            self.right_img = self._decode_img(msg)
        except Exception as e:
            self.get_logger().warn(f"right decode failed: {e}")

    def info_cb(self, msg):
        self.left_info = msg

    # ------------------------------------------------------------------
    #  视差孔洞填充
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_disparity_holes(disp, mode="light", guide=None):
        """
        多级孔洞填充，从浅到深三种模式。

        Args:
            disp:   float32 视差图，无效像素 = 0
            mode:   "light" / "medium" / "aggressive"
            guide:  uint8 引导图（左目灰度），用于 aggressive 模式的边缘保持

        Returns:
            填充后的视差图（原地修改）
        """
        if mode == "light":
            # ---- 轻量模式：形态学闭运算 + 小孔均值填充 ----
            # Step 1: 闭运算弥合小裂缝
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            valid = (disp > 0).astype(np.uint8)
            closed = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, kernel, iterations=1)
            # 填充闭运算弥合的区域
            mask = (closed > 0) & (disp == 0)
            if mask.any():
                mean_disp = cv2.boxFilter(disp, -1, (3, 3),
                                          borderType=cv2.BORDER_CONSTANT)
                disp[mask] = mean_disp[mask]

            # Step 2: 3x3 邻域至少 5 个有效 → 用均值填充
            kernel2 = np.ones((3, 3), np.uint8)
            valid2 = (disp > 0).astype(np.uint8)
            neighbor_cnt = cv2.filter2D(valid2, -1, kernel2,
                                        borderType=cv2.BORDER_CONSTANT)
            fill2 = (neighbor_cnt >= 5) & (disp == 0)
            if fill2.any():
                mean2 = cv2.boxFilter(disp, -1, (3, 3),
                                      borderType=cv2.BORDER_CONSTANT)
                disp[fill2] = mean2[fill2]

            return disp

        elif mode == "medium":
            # ---- 中等模式：更大核闭运算 + 多级填充 ----
            # Step 1: 5x5 闭运算
            kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            valid = (disp > 0).astype(np.uint8)
            closed = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, kernel5, iterations=1)
            mask = (closed > 0) & (disp == 0)
            if mask.any():
                mean_disp = cv2.boxFilter(disp, -1, (5, 5),
                                          borderType=cv2.BORDER_CONSTANT)
                disp[mask] = mean_disp[mask]

            # Step 2: 5x5 邻域至少 9 个有效
            kernel_box = np.ones((5, 5), np.uint8)
            valid2 = (disp > 0).astype(np.uint8)
            neighbor_cnt = cv2.filter2D(valid2, -1, kernel_box,
                                        borderType=cv2.BORDER_CONSTANT)
            fill2 = (neighbor_cnt >= 9) & (disp == 0)
            if fill2.any():
                mean2 = cv2.boxFilter(disp, -1, (5, 5),
                                      borderType=cv2.BORDER_CONSTANT)
                disp[fill2] = mean2[fill2]

            # Step 3: 对剩余空洞做快速行进修复
            mask_remain = (disp == 0)
            if mask_remain.any():
                mask_ui8 = mask_remain.astype(np.uint8)
                d_max = disp.max()
                if d_max > 0:
                    # 归一化到 uint8 范围再修复
                    disp_norm = (disp / d_max * 255).astype(np.uint8)
                    disp_filled = cv2.inpaint(disp_norm, mask_ui8, 3, cv2.INPAINT_TELEA)
                    disp[mask_remain] = disp_filled[mask_remain].astype(np.float32) / 255.0 * d_max

            return disp

        else:  # "aggressive"
            # ---- 激进模式：大范围填充 + 双边滤波平滑 ----
            # Step 1: 7x7 闭运算 ×2
            kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            valid = (disp > 0).astype(np.uint8)
            closed = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, kernel7, iterations=2)
            mask = (closed > 0) & (disp == 0)
            if mask.any():
                mean_disp = cv2.boxFilter(disp, -1, (7, 7),
                                          borderType=cv2.BORDER_CONSTANT)
                disp[mask] = mean_disp[mask]

            # Step 2: 多尺度均值填充 (7→15→31)
            for ksize in [7, 15, 31]:
                valid2 = (disp > 0).astype(np.uint8)
                kernel_box = np.ones((ksize, ksize), np.uint8)
                neighbor_cnt = cv2.filter2D(valid2, -1, kernel_box,
                                            borderType=cv2.BORDER_CONSTANT)
                threshold = ksize * ksize // 3
                fill2 = (neighbor_cnt >= threshold) & (disp == 0)
                if fill2.any():
                    mean2 = cv2.boxFilter(disp, -1, (ksize, ksize),
                                          borderType=cv2.BORDER_CONSTANT)
                    disp[fill2] = mean2[fill2]
                if not (disp == 0).any():
                    break

            # Step 3: 对残留空洞做各向异性扩散填充
            mask_remain = (disp == 0)
            if mask_remain.any():
                # 使用快速行进修复（需 uint8 输入）
                disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                mask_ui8 = mask_remain.astype(np.uint8)
                disp_filled = cv2.inpaint(disp_norm, mask_ui8, 5, cv2.INPAINT_NS)
                # 还原尺度
                disp_filled = disp_filled.astype(np.float32) / 255.0 * disp.max()
                # 只填充原本为空的位置
                disp[mask_remain] = disp_filled[mask_remain]

            # Step 4: 边缘引导平滑（使用左目灰度图保持边缘）
            if guide is not None and guide.shape[:2] == disp.shape[:2]:
                guide_8u = guide if guide.dtype == np.uint8 else cv2.normalize(
                    guide, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                disp_blur = cv2.bilateralFilter(disp, d=7, sigmaColor=75, sigmaSpace=15)
                smooth_mask = disp > 0
                disp[smooth_mask] = 0.6 * disp[smooth_mask] + 0.4 * disp_blur[smooth_mask]

            return disp

    # ------------------------------------------------------------------

    def process(self):
        if self.left_img is None or self.right_img is None:
            return

        left = self.left_img
        right = self.right_img
        self.left_img = None
        self.right_img = None

        # ---- 极线校正 ----
        if self.map1_l is not None and self.map1_r is not None:
            left_rect = cv2.remap(left, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right, self.map1_r, self.map2_r, cv2.INTER_LINEAR)
        else:
            left_rect = left
            right_rect = right

        # 转灰度
        if left_rect.ndim == 3:
            gray_l = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        else:
            gray_l = left_rect
            gray_r = right_rect

        # ---- CLAHE 对比度增强（提升低纹理区域的 BM 匹配成功率）----
        if self.use_clahe:
            gray_l = self._clahe.apply(gray_l)
            gray_r = self._clahe.apply(gray_r)

        # ---- 降采样加速立体匹配 ----
        gray_l_small = cv2.resize(gray_l, (0, 0), fx=self.downsample_factor, fy=self.downsample_factor,
                                   interpolation=cv2.INTER_LINEAR)
        gray_r_small = cv2.resize(gray_r, (0, 0), fx=self.downsample_factor, fy=self.downsample_factor,
                                   interpolation=cv2.INTER_LINEAR)

        # ---- 立体匹配（根据参数选择算法）----
        num_disp = self.num_disparities
        num_disp = ((num_disp + 15) // 16) * 16  # 必须是 16 的倍数

        # --- 创建左目匹配器 ---
        if self.stereo_algo == "bm":
            left_matcher = cv2.StereoBM_create(
                numDisparities=num_disp, blockSize=7,
            )
            left_matcher.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
            left_matcher.setPreFilterSize(9)
            left_matcher.setPreFilterCap(31)
            left_matcher.setTextureThreshold(3)
            left_matcher.setUniquenessRatio(3)
            left_matcher.setSpeckleWindowSize(100)
            left_matcher.setSpeckleRange(4)
            left_matcher.setDisp12MaxDiff(1)
        else:
            left_matcher = cv2.StereoSGBM_create(
                minDisparity=0, numDisparities=num_disp, blockSize=7,
                P1=8 * 3 * 7 ** 2, P2=32 * 3 * 7 ** 2,
                disp12MaxDiff=1, uniquenessRatio=5,
                speckleWindowSize=200, speckleRange=4, preFilterCap=31,
                mode=cv2.STEREO_SGBM_MODE_SGBM,
            )

        # --- WLS 路径：左右交叉匹配 + 边缘引导滤波（平滑 + 边缘对齐 + 填洞） ---
        if self.use_wls:
            try:
                # 创建右目匹配器（与左目对称）
                right_matcher = ximgproc.createRightMatcher(left_matcher)

                # 低分辨率下计算左右视差
                left_disp_small = left_matcher.compute(gray_l_small, gray_r_small).astype(np.float32) / 16.0
                right_disp_small = right_matcher.compute(gray_r_small, gray_l_small).astype(np.float32) / 16.0

                # 快速上采样到原始分辨率（WLS 要求引导图与视差图尺寸一致）
                disp_up = cv2.resize(left_disp_small, (gray_l.shape[1], gray_l.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
                right_up = cv2.resize(right_disp_small, (gray_l.shape[1], gray_l.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)

                # 创建 WLS 滤波器
                wls_filter = ximgproc.createDisparityWLSFilter(left_matcher)
                wls_filter.setLambda(self.wls_lambda)
                wls_filter.setSigmaColor(self.wls_sigma)

                # ★ 核心：WLS 以全分辨率左图作为引导，平滑视差、保边、填洞
                disparity = wls_filter.filter(
                    disp_up, gray_l, disparity_map_right=right_up
                )
                disparity[disparity < 0] = 0.0

                # WLS 已填平大部分空洞，只做轻量扫尾
                if self.hole_fill_mode != "none":
                    disparity = self._fill_disparity_holes(disparity, "light", guide=gray_l)

            except Exception as wls_err:
                self.get_logger().warn(f"WLS 滤波失败 ({wls_err})，降级为原始路径")
                # 降级：用左视差 + INTER_CUBIC 上采样
                left_disp_small = left_matcher.compute(gray_l_small, gray_r_small).astype(np.float32) / 16.0
                disparity = cv2.resize(left_disp_small, (gray_l.shape[1], gray_l.shape[0]),
                                       interpolation=cv2.INTER_CUBIC)
                disparity[disparity <= 0] = 0.0
                if self.hole_fill_mode != "none":
                    disparity = self._fill_disparity_holes(disparity, self.hole_fill_mode, guide=gray_l)

        else:
            # --- 原始路径：低分辨率计算 → INTER_CUBIC 上采样 → 填洞 ---
            left_disp_small = left_matcher.compute(gray_l_small, gray_r_small).astype(np.float32) / 16.0
            disparity = cv2.resize(left_disp_small, (gray_l.shape[1], gray_l.shape[0]),
                                   interpolation=cv2.INTER_CUBIC)
            disparity[disparity <= 0] = 0.0
            if self.hole_fill_mode != "none":
                disparity = self._fill_disparity_holes(disparity, self.hole_fill_mode, guide=gray_l)

        # ---- 深度计算 (使用校正后的参数) ----
        if self.fx_rect is not None:
            fx = self.fx_rect
            baseline = self.baseline_rect
        else:
            fx = self.left_info.k[0] if self.left_info else 297.9
            baseline = 0.0426

        depth = np.zeros_like(disparity)
        valid = disparity > 0.3  # 原 0.5 → 0.3（允许更小的有效视差）
        depth[valid] = (fx * baseline) / disparity[valid]
        depth = np.clip(depth, self.min_depth, self.max_depth)
        depth[~valid] = 0.0

        # ---- 时域平滑（可选） ----
        if self.temporal_smooth > 0 and self._prev_depth is not None:
            if depth.shape == self._prev_depth.shape:
                alpha = self.temporal_smooth
                # 只对当前帧有效区域做平滑，无效区域沿用上帧
                smooth = alpha * self._prev_depth + (1.0 - alpha) * depth
                depth[valid] = smooth[valid]
                # 如果当前帧无效但上帧有效，继承上帧（衰减）
                inherit = (~valid) & (self._prev_depth > 0)
                if inherit.any():
                    depth[inherit] = self._prev_depth[inherit] * 0.9
        self._prev_depth = depth.copy()

        stamp = self.get_clock().now().to_msg()

        # ---- 发布视差图 ----
        inner_img = Image()
        inner_img.header.stamp = stamp
        inner_img.header.frame_id = "left_camera"
        inner_img.height = gray_l.shape[0]
        inner_img.width = gray_l.shape[1]
        inner_img.encoding = "32FC1"
        inner_img.is_bigendian = False
        inner_img.step = gray_l.shape[1] * 4
        inner_img.data = disparity.tobytes()

        disp_msg = DisparityImage()
        disp_msg.header.stamp = stamp
        disp_msg.header.frame_id = "left_camera"
        disp_msg.image = inner_img
        disp_msg.f = fx
        disp_msg.t = baseline
        disp_msg.min_disparity = 0.0
        disp_msg.max_disparity = float(num_disp)
        disp_msg.delta_d = 1.0 / 16.0
        self.disparity_pub.publish(disp_msg)

        # ---- 发布深度图 ----
        depth_msg = Image()
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = "left_camera"
        depth_msg.height = depth.shape[0]
        depth_msg.width = depth.shape[1]
        depth_msg.encoding = "32FC1"
        depth_msg.is_bigendian = False
        depth_msg.step = depth.shape[1] * 4
        depth_msg.data = depth.tobytes()
        self.depth_pub.publish(depth_msg)

        # ---- 发布压缩深度图（16位 PNG，隔帧发布）----
        self._frame_count += 1
        if self._frame_count % 2 == 0:
            depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
            success, encoded = cv2.imencode(
                ".png", depth_mm,
                [cv2.IMWRITE_PNG_COMPRESSION, 6]
            )
            if success:
                compressed_msg = CompressedImage()
                compressed_msg.header.stamp = stamp
                compressed_msg.header.frame_id = "left_camera"
                compressed_msg.format = "16UC1; png compressed"
                compressed_msg.data = encoded.tobytes()
                self.depth_compressed_pub.publish(compressed_msg)


# ======================================================================

def main(args=None):
    rclpy.init(args=args)
    node = SgbmStereoNode()
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
