#!/usr/bin/env bash
# =============================================================================
# run_config.sh — YOLO 运行配置（唯一的"改参数 / 改模型"入口）
#
# 说明:
#   - 本文件被 run_stereo_yolo.sh 等启动脚本 source，改完保存即可生效，
#     无需重新编译、无需改 launch 文件。
#   - 每个变量都带默认值兜底：即使本文件缺失或某项没写，脚本也能用内置默认值启动。
#   - 变量名 = YOLO_ + launch 参数名（对应 yolo.launch.py 的 DeclareLaunchArgument）。
#
# 常见用法:
#   ① 换模型    → 改 YOLO_MODEL（绝对路径或模型名）
#   ② 调阈值    → 改 YOLO_THRESHOLD / YOLO_IOU
#   ③ 提帧率    → YOLO_HALF=True，或调小 YOLO_IMGSZ_*
#   ④ 换设备    → YOLO_DEVICE=cpu 或 cuda:0
# =============================================================================

# ---------------------------------------------------------------------------
# 模型（即将启用的模型）
# ---------------------------------------------------------------------------
YOLO_MODEL="/home/unitree/YOLO_ws/models/bin_and_cloth-OBB.pt"   # 模型路径或名称
YOLO_MODEL_TYPE="YOLO"          # 模型类型: YOLO / World / YOLOE

# ---------------------------------------------------------------------------
# 推理超参数
# ---------------------------------------------------------------------------
YOLO_THRESHOLD="0.5"            # 置信度阈值（越大越严）
YOLO_IOU="0.7"                  # NMS IoU 阈值
YOLO_IMGSZ_HEIGHT="480"         # 推理输入高（像素）
YOLO_IMGSZ_WIDTH="640"          # 推理输入宽（像素）
YOLO_HALF="False"               # FP16 半精度加速（True 提帧率）
YOLO_MAX_DET="300"              # 每帧最大检测数
YOLO_AUGMENT="False"            # 测试时增强 TTA（慢，一般不开启）
YOLO_AGNOSTIC_NMS="False"       # 跨类别 NMS
YOLO_RETINA_MASKS="False"       # 高分辨率分割掩码
YOLO_CONTAINMENT_THRESHOLD="0.7" # 包含过滤重叠阈值（0~1，1=完全包含）

# ---------------------------------------------------------------------------
# 设备 / 运行
# ---------------------------------------------------------------------------
YOLO_DEVICE="cuda:0"            # 推理设备: cuda:0 / cpu
YOLO_FUSE_MODEL="False"         # 是否 fuse 模型加速推理
YOLO_ENABLE="True"              # 启动时是否启用检测

# ---------------------------------------------------------------------------
# 管线开关（跟踪 / 3D / 可视化）
# ---------------------------------------------------------------------------
YOLO_USE_TRACKING="True"        # 是否启用跟踪节点
YOLO_USE_3D="True"              # 是否启用 3D 检测（需双目深度链路在跑）
YOLO_USE_DEBUG="True"           # 是否启用可视化节点
YOLO_TARGET_FRAME="base_link"   # 3D 结果目标坐标系
YOLO_DEPTH_UNITS_DIVISOR="1"    # 深度值→米 除数（毫米图填 1000，已是米填 1）

# ---------------------------------------------------------------------------
# 话题 / QoS / 命名空间
# ---------------------------------------------------------------------------
YOLO_INPUT_IMAGE_TOPIC="/left/image_raw"      # 输入图像话题
YOLO_INPUT_DEPTH_TOPIC="/left/depth"          # 输入深度话题
YOLO_INPUT_DEPTH_INFO_TOPIC="/left/camera_info" # 深度相机内参话题
YOLO_NAMESPACE="yolo"           # 节点命名空间（话题前缀 /yolo/*）
YOLO_IMAGE_RELIABILITY="1"      # 图像 QoS: 0=默认 1=Reliable 2=BestEffort
