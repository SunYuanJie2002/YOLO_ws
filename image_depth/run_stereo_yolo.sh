#!/usr/bin/env bash
# ============================================================
# YOLO + 双目深度 3D 检测 一键启动脚本
# ============================================================
# 启动的节点:
#   1. stereo_split_node        — ZMQ 双目图 → left/right 话题
#   2. sgbm_stereo_node         — SGBM 立体匹配 → 视差 + 深度
#   3. static_transform_publisher — left_camera → base_link TF
#   4. yolo.launch.py           — YOLO 检测 + 3D 坐标转换
#
# 需要两个 conda 环境:
#   - teleimager: 双目拆分 + SGBM (有 zmq + opencv)
#   - YOLO:       YOLO 检测 (需 numpy<2)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 读取运行配置（超参数与模型都改这里: config/run_config.sh）----
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/../config/run_config.sh}"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    echo "[配置] 已加载 $CONFIG_FILE"
else
    echo "[警告] 未找到配置文件 $CONFIG_FILE，使用脚本内置默认值"
fi

echo "=========================================="
echo "  YOLO + 双目深度 3D 检测"
echo "=========================================="

# ---- ROS2 环境 ----
if [ -z "$ROS_DISTRO" ]; then
    if [ -f /opt/ros/humble/setup.bash ]; then
        source /opt/ros/humble/setup.bash
    fi
fi
if [ -f "$SCRIPT_DIR/../install/setup.bash" ]; then
    source "$SCRIPT_DIR/../install/setup.bash"
fi
# 使用自定义修复后的 rmw_cyclonedds_cpp（抑制 serdata.cpp 序列化警告）
if [ -f ~/ros2_ws/install/setup.bash ]; then
    source ~/ros2_ws/install/setup.bash
fi
echo "ROS_DISTRO=$ROS_DISTRO"

# ---- 清理旧进程 ----
echo ""
echo "[1/6] 清理旧进程..."
for p in stereo_split_node sgbm_stereo static_transform yolo_node tracking_node detect_3d debug_node; do
    pkill -f "$p" 2>/dev/null || true
done
sleep 2

# ---- 1. 双目拆分 (teleimager 环境) ----
echo ""
echo "[2/6] 启动 stereo_split_node (双目拆分)..."
source /home/unitree/miniconda3/bin/activate teleimager 2>/dev/null
python3 "$SCRIPT_DIR/stereo_split_node.py" &
PID_SPLIT=$!
sleep 3
kill -0 $PID_SPLIT 2>/dev/null || { echo "[错误] stereo_split_node 启动失败"; exit 1; }
echo "  PID: $PID_SPLIT"

# ---- 2. 立体匹配 (teleimager 环境) ----
echo ""
echo "[3/6] 启动 sgbm_stereo_node (立体匹配 + 极线校正)..."
echo "     算法=StereoBM | CLAHE增强 | WLS边缘引导滤波"
# teleimager 环境已在上一步激活
python3 "$SCRIPT_DIR/sgbm_stereo_node.py" \
    --ros-args -p calibration_file:="$SCRIPT_DIR/calibration_result.yaml" &
PID_SGBM=$!
sleep 2
kill -0 $PID_SGBM 2>/dev/null || { echo "[错误] sgbm_stereo_node 启动失败"; exit 1; }
echo "  PID: $PID_SGBM"

# ---- 3. 静态 TF (使用系统 ROS2) ----
echo ""
echo "[4/6] 发布静态 TF left_camera → base_link..."
source /opt/ros/humble/setup.bash
ros2 run tf2_ros static_transform_publisher \
    0.054 0.018 0.474 -0.383 0 0 0.924 \
    left_camera base_link &
PID_TF=$!
sleep 1
kill -0 $PID_TF 2>/dev/null || { echo "[错误] static_transform_publisher 启动失败"; exit 1; }
echo "  PID: $PID_TF"
echo "  (来自 URDF: 向前 5.4cm, 向左 1.8cm, 向上 47.4cm, 俯仰 45°)"

# ---- 4. YOLO 3D 检测 (YOLO 环境, 需 numpy<2) ----
echo ""
echo "[5/6] 启动 YOLO 检测 + 3D..."
echo "     模型: ${YOLO_MODEL:-$SCRIPT_DIR/../models/best.pt}"
echo "     threshold=${YOLO_THRESHOLD:-0.5} iou=${YOLO_IOU:-0.7} imgsz=${YOLO_IMGSZ_WIDTH:-640}x${YOLO_IMGSZ_HEIGHT:-480} half=${YOLO_HALF:-False}"
source /home/unitree/miniconda3/bin/activate YOLO
ros2 launch yolo_bringup yolo.launch.py \
    use_3d:="${YOLO_USE_3D:-True}" \
    use_tracking:="${YOLO_USE_TRACKING:-True}" \
    use_debug:="${YOLO_USE_DEBUG:-True}" \
    input_image_topic:="${YOLO_INPUT_IMAGE_TOPIC:-/left/image_raw}" \
    input_depth_topic:="${YOLO_INPUT_DEPTH_TOPIC:-/left/depth}" \
    input_depth_info_topic:="${YOLO_INPUT_DEPTH_INFO_TOPIC:-/left/camera_info}" \
    depth_image_units_divisor:="${YOLO_DEPTH_UNITS_DIVISOR:-1}" \
    target_frame:="${YOLO_TARGET_FRAME:-base_link}" \
    threshold:="${YOLO_THRESHOLD:-0.5}" \
    iou:="${YOLO_IOU:-0.7}" \
    imgsz_height:="${YOLO_IMGSZ_HEIGHT:-480}" \
    imgsz_width:="${YOLO_IMGSZ_WIDTH:-640}" \
    half:="${YOLO_HALF:-False}" \
    max_det:="${YOLO_MAX_DET:-300}" \
    augment:="${YOLO_AUGMENT:-False}" \
    agnostic_nms:="${YOLO_AGNOSTIC_NMS:-False}" \
    retina_masks:="${YOLO_RETINA_MASKS:-False}" \
    containment_threshold:="${YOLO_CONTAINMENT_THRESHOLD:-0.7}" \
    model:="${YOLO_MODEL:-$SCRIPT_DIR/../models/best.pt}" \
    model_type:="${YOLO_MODEL_TYPE:-YOLO}" \
    device:="${YOLO_DEVICE:-cuda:0}" \
    fuse_model:="${YOLO_FUSE_MODEL:-False}" \
    enable:="${YOLO_ENABLE:-True}" \
    namespace:="${YOLO_NAMESPACE:-yolo}" \
    image_reliability:="${YOLO_IMAGE_RELIABILITY:-1}" &
PID_YOLO=$!
sleep 3
echo "  PID: $PID_YOLO"

echo ""
echo "[6/6] 验证话题..."
sleep 2
source /opt/ros/humble/setup.bash
echo "  left 话题:"
ros2 topic list 2>/dev/null | grep "^/left" | while read t; do echo "    $t"; done
echo "  yolo 话题:"
ros2 topic list 2>/dev/null | grep "^/yolo" | while read t; do echo "    $t"; done

echo ""
echo "=========================================="
echo "  所有节点已启动!"
echo "=========================================="
echo ""
echo "可用话题:"
echo "  /left/image_raw      — 左目图像 (bgr8)"
echo "  /right/image_raw     — 右目图像 (bgr8)"
echo "  /left/camera_info    — 左目相机内参"
echo "  /left/disparity      — 视差图 (DisparityImage)"
echo "  /left/depth          — 深度图 (32FC1, 米)"
echo "  /yolo/detections     — YOLO 2D 检测"
echo "  /yolo/tracking       — 目标跟踪 (ByteTrack)"
echo "  /yolo/detections_3d  — ★ 3D 检测结果 (相对 base_link)"
echo "  /yolo/dbg_image      — 可视化图像"
echo ""
echo "查看深度:      ros2 topic echo /left/depth --once | head -5"
echo "查看3D结果:    ros2 topic echo /yolo/detections_3d --once"
echo "RViz:          ros2 run rviz2 rviz2"
echo ""
echo "按 Ctrl+C 停止所有节点"

# ---- 等待 + 清理 ----
trap 'echo "停止中..."; kill $PID_SPLIT $PID_SGBM $PID_TF 2>/dev/null; exit 0' INT TERM
wait
