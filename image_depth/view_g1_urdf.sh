#!/bin/bash
# =============================================================================
# view_g1_urdf.sh - G1 机器人 URDF 模型可视化启动脚本
#
# 用法：
#   bash view_g1_urdf.sh              # 在本机 (有显示器) 启动 RViz 可视化
#   bash view_g1_urdf.sh --headless   # 在机器人端 (无显示器) 只启动 ROS 节点
#   bash view_g1_urdf.sh --stop       # 停止已启动的后台节点
#
# 功能：
#   1. g1_lowstate_to_joint_states — 将 /lowstate 转为 /joint_states
#   2. robot_state_publisher      — 加载 URDF，发布所有 link 的 TF
#   3. base_link → pelvis 静态 TF  — 连接检测参考系与 URDF 根 link
#   4. rviz2 (可选)               — 显示机器人模型 + TF 树
#
# TF 树结构：
#   left_camera → base_link → pelvis → [所有 URDF link]
#
# 典型场景：
#   - 机器人端 (无显示器): bash view_g1_urdf.sh --headless
#   - PC 端 (有显示器):     bash view_g1_urdf.sh
#     或手动:              ros2 run rviz2 rviz2 -d g1_urdf.rviz
# =============================================================================

set -e

# ---------- 路径 ----------
URDF_PATH="$HOME/gb_repositories/unitree_ros2/foxglove/g1_description/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RVIZ_CFG="$SCRIPT_DIR/g1_urdf.rviz"

# ---------- 解析参数 ----------
MODE="normal"  # normal | headless | stop
for arg in "$@"; do
    case "$arg" in
        --headless) MODE="headless" ;;
        --stop)     MODE="stop" ;;
    esac
done

# ---------- 停止功能 ----------
stop_nodes() {
    echo " [停止] 停止后台节点 ..."
    for pname in "g1_lowstate_to_joint_states" "robot_state_publisher"; do
        pkill -x "$pname" 2>/dev/null || true
    done
    pkill -f "static_transform_publisher.*base_link pelvis" 2>/dev/null || true
    rm -f /tmp/g1_robot_temp.urdf 2>/dev/null || true
    echo " 完成。"
    exit 0
}

if [ "$MODE" = "stop" ]; then
    stop_nodes
fi

# ---------- 环境 ----------
source /opt/ros/humble/setup.bash
if [ -f "$HOME/gb_repositories/unitree_ros2/example/install/setup.bash" ]; then
    source "$HOME/gb_repositories/unitree_ros2/example/install/setup.bash"
fi

echo "=============================================="
echo " G1 机器人模型可视化"
echo "=============================================="
echo " URDF: $URDF_PATH"
echo " 模式: $MODE"
echo ""

# ---------- 清理旧进程 ----------
for pname in "g1_lowstate_to_joint_states" "robot_state_publisher"; do
    if pgrep -x "$pname" > /dev/null 2>&1; then
        echo " [INFO] 停止已有 $pname 进程..."
        pkill -x "$pname" 2>/dev/null || true
        sleep 0.5
    fi
done

# ---------- 1. lowstate → joint_states ----------
echo " [1/3] 启动 g1_lowstate_to_joint_states ..."
ros2 run unitree_ros2_example g1_lowstate_to_joint_states \
    --ros-args \
    -p urdf_path:="$URDF_PATH" \
    -p lowstate_topic:=/lowstate \
    -p joint_state_topic:=/joint_states \
    --log-level warn \
    &
sleep 1

# ---------- 2. robot_state_publisher ----------
echo " [2/3] 启动 robot_state_publisher ..."
# robot_state_publisher 仅需 URDF 的运动学信息（关节、link 坐标变换），
# 不需要加载 mesh 文件，因此直接使用原始 URDF 文件。
# 保留 package:// 路径，让 PC 端的 RViz 自行解析 mesh 位置。
ros2 run robot_state_publisher robot_state_publisher \
    "$URDF_PATH" \
    --ros-args --log-level warn \
    &
sleep 1

# ---------- 3. base_link → pelvis 静态 TF（抬起机器人使其脚底落在地面）----------
echo " [3/3] 发布 base_link → pelvis 静态 TF ..."
# 注意：
#   - 已有 left_camera → base_link 静态 TF（由 run_stereo_yolo.sh 启动）
#   - URDF 的根 link 是 pelvis（腰部），脚踝在 pelvis 下方约 0.71m
#   - 通过此 TF 将机器人抬升，使脚底近似落在 RViz 网格平面 (z=0) 上
#   - 此高度为当前站立姿态的近似值，随关节角度变化会有微小差异
#   - YOLO 3D 检测的 target_frame=base_link 不受影响
HEIGHT_OFFSET=0.71
ros2 run tf2_ros static_transform_publisher \
    0 0 "$HEIGHT_OFFSET" 0 0 0 1 \
    base_link pelvis \
    --ros-args --log-level warn \
    &

echo ""
echo " G1 机器人 ROS 节点已启动！"
echo ""

# ---------- 4. RViz (仅 normal 模式) ----------
if [ "$MODE" = "headless" ]; then
    echo " 机器人端无显示器，已跳过 RViz。"
    echo ""
    echo " 请在你的 PC 上执行以下命令连接 RViz："
    echo "-----------------------------------------------------------"
    echo "  # 在 PC 上 source ROS 环境后："
    echo "  source /opt/ros/humble/setup.bash"
    echo "  source ~/YOLO_ws/install/setup.bash  # 可选，使 yolo_msgs 可用"
    echo ""
    echo "  # 如果跨机器通信，需设置 ROS_DOMAIN_ID（两端一致）："
    echo "  export ROS_DOMAIN_ID=0"
    echo ""
    echo "  # 启动 RViz"
    echo "  cd ~/YOLO_ws/image_depth"
    echo "  ros2 run rviz2 rviz2 -d g1_urdf.rviz"
    echo "-----------------------------------------------------------"
    echo ""
    echo " 完成后在机器人端停止节点："
    echo "  bash view_g1_urdf.sh --stop"
    echo ""
    echo " 或者在本机 (有显示器) 运行直接打开 RViz："
    echo "  bash view_g1_urdf.sh"
    exit 0
fi

# ---------- 4. RViz 正常启动 ----------
echo " [启动] RViz2 ..."
echo ""
echo "  RViz 中操作提示："
echo "  - Fixed Frame 已设为 base_link"
echo "  - 左侧 Displays 面板有 RobotModel + TF"
echo "  - 可添加 /yolo/dgb_bb_markers (MarkerArray) 查看 3D 检测框"
echo "  - 可添加 /yolo/dbg_image/compressed (Image) 查看检测画面"
echo "  - 鼠标中键平移，右键旋转，滚轮缩放"
echo ""

ros2 run rviz2 rviz2 \
    -d "$RVIZ_CFG" \
    --ros-args --log-level warn

# ---------- 清理 ----------
echo ""
echo " [完毕] RViz 已关闭，清理后台节点 ..."
pkill -x "g1_lowstate_to_joint_states" 2>/dev/null || true
pkill -x "robot_state_publisher" 2>/dev/null || true
pkill -f "static_transform_publisher.*base_link pelvis" 2>/dev/null || true
echo " 完成。"
