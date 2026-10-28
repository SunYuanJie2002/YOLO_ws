#!/usr/bin/env bash
# =============================================================================
# switch_model.sh — 不重启进程，通过 ROS2 生命周期状态机切换 YOLO 模型
#
# 原理：yolo_node 是 LifecycleNode，模型只在 activate 时加载。
#       通过 deactivate -> cleanup -> configure -> activate 让节点
#       重新读取 model 参数并加载新模型，进程全程不退出。
#
# 本脚本直接用 ROS2 服务调用（service call）实现，绕开
# ros2 lifecycle / ros2 param CLI 的 graph 查询（本机 CycloneDDS
# discovery 存在 "empty node name" 问题，CLI 命令不可用）。
#
# 用法:
#   ./switch_model.sh <模型路径或名称> [model_type]
#
# 示例:
#   ./switch_model.sh /home/unitree/YOLO_ws/models/Arm_holding.pt
#   ./switch_model.sh /home/unitree/YOLO_ws/models/bin_and_cloth-OBB.pt
#   ./switch_model.sh yolov8s-worldv2.pt World        # 顺带切换模型类型
#
# 注意:
#   - 切换期间检测会中断几秒~几十秒（模型加载耗时），ZMQ 会断开重连。
#   - 只改 model / model_type，zmq_*、source_* 等仍不支持运行中修改。
# =============================================================================
set -euo pipefail

NODE="/yolo/yolo_node"

# ---- 环境（source 期间临时关闭 set -u，避免 ROS setup.bash 报 unbound variable）----
set +u
if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
fi
if [ -f /home/unitree/YOLO_ws/install/setup.bash ]; then
  source /home/unitree/YOLO_ws/install/setup.bash
fi
set -u

# ---- 参数 ----
MODEL="${1:?用法: $0 <模型路径或名称> [model_type]}"
MODEL_TYPE="${2:-}"

# ---- 辅助：调用生命周期/参数服务 ----
call_service() {  # $1=服务后缀  $2=服务类型  $3=请求
  timeout 20 ros2 service call "${NODE}/$1" "$2" "$3" 2>/dev/null
}

# ---- 探测节点是否可达 ----
if ! call_service get_state lifecycle_msgs/srv/GetState "{}" >/dev/null; then
  echo "❌ 节点 ${NODE} 不可达（服务调用超时），请先启动 yolo launch。" >&2
  exit 1
fi

state_resp="$(call_service get_state lifecycle_msgs/srv/GetState "{}")"
cur_id="$(echo "${state_resp}" | grep -oE "id=[0-9]+" | head -1 | cut -d= -f2)"
cur_label="$(echo "${state_resp}" | grep -oE "label='[^']*'" | head -1 | cut -d"'" -f2)"
echo "当前生命周期状态: ${cur_label} (id=${cur_id})"

# 生命周期状态 id: 1=unconfigured 2=inactive 3=active 4=finalized
# 生命周期转换 id: 1=configure 2=cleanup 3=activate 4=deactivate 5=shutdown
change_state() {  # $1=转换id  $2=说明
  echo ">>> lifecycle: ${2} (id=${1})"
  local resp
  resp="$(call_service change_state lifecycle_msgs/srv/ChangeState "{transition: {id: ${1}}}")"
  if ! echo "${resp}" | grep -q "success=True"; then
    echo "❌ ${2} 失败: ${resp}" >&2
    exit 1
  fi
}

# ---- 回到 unconfigured（仅当节点处于 active / inactive）----
case "${cur_id}" in
  3)  # active
    change_state 4 "deactivate"
    change_state 2 "cleanup"
    ;;
  2)  # inactive
    change_state 2 "cleanup"
    ;;
  1)  # unconfigured
    echo ">>> 节点已在 unconfigured，跳过 deactivate/cleanup"
    ;;
  *)
    echo "❌ 不支持的当前状态 id=${cur_id}" >&2
    exit 1
    ;;
esac

# ---- 设置模型参数（必须在 configure 之前）----
echo ">>> 设置 model=${MODEL}"
set_resp="$(call_service set_parameters rcl_interfaces/srv/SetParameters \
  "{parameters: [{name: 'model', value: {type: 4, string_value: '${MODEL}'}}]}")"
if ! echo "${set_resp}" | grep -q "successful=True"; then
  echo "❌ 设置 model 参数失败: ${set_resp}" >&2
  exit 1
fi

if [ -n "${MODEL_TYPE}" ]; then
  echo ">>> 设置 model_type=${MODEL_TYPE}"
  set_resp="$(call_service set_parameters rcl_interfaces/srv/SetParameters \
    "{parameters: [{name: 'model_type', value: {type: 4, string_value: '${MODEL_TYPE}'}}]}")"
  if ! echo "${set_resp}" | grep -q "successful=True"; then
    echo "❌ 设置 model_type 参数失败: ${set_resp}" >&2
    exit 1
  fi
fi

# ---- configure + activate（activate 会加载新模型，可能较慢）----
change_state 1 "configure"
change_state 3 "activate"

# ---- 验证 ----
get_resp="$(call_service get_parameters rcl_interfaces/srv/GetParameters "{names: ['model']}")"
new_model="$(echo "${get_resp}" | grep -oE "string_value='[^']*'" | head -1 | cut -d"'" -f2)"
echo ""
echo "✅ 模型切换完成（进程未重启）"
echo "   新模型: ${new_model}"
echo "   状态:   $(call_service get_state lifecycle_msgs/srv/GetState "{}" | grep -oE "label='[^']*'" | head -1 | cut -d"'" -f2)"
