# YOLO 检测节点 (YOLO_ws) 使用说明

> 系统：ROS2 Humble + Python 3.10 + CUDA
> 位置：`/home/unitree/YOLO_ws`
> 运行环境：`YOLO`（Conda）+ `teleimager`（Conda）
> 上游项目：[mgonzs13/yolo_ros](https://github.com/mgonzs13/yolo_ros) v4.6.0（含本地定制扩展）

---

## 目录

1. [这个项目做什么](#1-这个项目做什么)
2. [工作原理](#2-工作原理)
3. [快速开始](#3-快速开始)
4. [工程结构](#4-工程结构)
5. [配置说明](#5-配置说明)
6. [对外 API（ROS2 话题 / 服务 / 消息）](#6-对外-apiros2-话题--服务--消息)
7. [核心功能](#7-核心功能)
   - [7.1 ZMQ 图像输入与左半裁切](#71-zmq-图像输入与左半裁切)
   - [7.2 目标跟踪](#72-目标跟踪)
   - [7.3 3D 检测（双目深度）](#73-3d-检测双目深度)
   - [7.4 包含关系过滤](#74-包含关系过滤)
   - [7.5 YOLO-World 开放词汇检测](#75-yolo-world-开放词汇检测)
   - [7.6 边界框面积监测](#76-边界框面积监测)
   - [7.7 OBB 旋转角度输出](#77-obb-旋转角度输出)
8. [日志查看](#8-日志查看)
9. [部署与运维](#9-部署与运维)
10. [改代码后怎么上线](#10-改代码后怎么上线)
11. [二次开发](#11-二次开发)
12. [常见问题排查](#12-常见问题排查)
13. [领域基础知识](#13-领域基础知识)
14. [第三方 SDK / 库速查表](#14-第三方-sdk--库速查表)

---

## 1. 这个项目做什么

**一句话定位**：给 G1 人形机器人装上一双"眼睛"——把相机画面里的目标（当前主要是**衣物 cloth**）实时识别出来，并给出它在画面里的位置（2D 框）和相对机器人的空间坐标（3D 框），供下游的抓取、停止、分拣等动作决策使用。

**简单类比**：这就像给机器人配了一位"视力员"——摄像头每秒把画面（480×1280 双目图）送进来，这位"视力员"（YOLO 模型）不断喊出"左边 2 米处有一包衣服"，并把框画出来，机器人据此决定该不该停、该往哪抓。

**技术栈**：ROS2 Humble（`rclpy`）+ Python 3.10 + Ultralytics YOLO（v8.4.6）+ PyTorch CUDA + ZeroMQ + OpenCV，运行在 Conda 环境中。

**主要功能（3 条）**：

| # | 功能 | 说明 | 输出 |
|---|------|------|------|
| 1 | 目标检测 | 基于 ZMQ 图传流做 YOLO 推理，输出 2D 检测框 | `/yolo/detections` |
| 2 | 3D 定位 | 结合双目深度把 2D 框换算成相对机器人的 3D 坐标 | `/yolo/detections_3d` |
| 3 | 后处理定制 | 目标跟踪、包含关系过滤、开放词汇检测 | `/yolo/tracking` 等 |

**当前工作空间内可用的模型**（`/home/unitree/YOLO_ws/models/`）：

| 模型 | 类别 | 用途 |
|------|------|------|
| `best.pt` | `{0: 'cloth'}`（衣物） | 当前主力模型，2D / 3D 检测管线默认使用 |
| `best-OBB.pt` | 旋转框检测 | 输出带旋转角度的边界框 |
| `robot.pt` | 机器人相关 | 备用模型 |

---

## 2. 工作原理

### 2.1 数据流图（ASCII）

```
┌─────────────┐   ZMQ SUB (tcp://192.168.123.164:55555)   ┌──────────────────────────────┐
│  teleimager │ ──────────────── 480×1280 双目图 ─────────▶│  yolo_node（核心检测）        │
│  (图传相机)  │                                           │  ① 解码 JPEG / 原始 uint8     │
└─────────────┘                                           │  ② 裁切左半 480×640           │
                                                          │  ③ YOLO 推理                 │
                                                          │  ④ 包含过滤（可选）            │
                                                          └──────┬───────────┬───────────┘
                                                                 │           │
                                          /yolo/image_raw         │           │ /yolo/detections
                                                                 ▼           ▼
                                            ┌─────────────┐   ┌────────────────────┐
                                            │ stereo_     │   │  tracking_node     │
                                            │ split_node  │   │  (ByteTrack)       │
                                            │ (双目拆分)   │   └─────────┬──────────┘
                                            └──────┬──────┘             │ /yolo/tracking
                                                   │ /left/image_raw     ▼
                                                   ▼              ┌────────────────────┐
                                            ┌─────────────┐       │  detect_3d_node    │
                                            │ sgbm_stereo │       │  (2D→3D + TF)      │
                                            │ _node       │       └─────────┬──────────┘
                                            │ (SGBM深度)  │                 │ /yolo/detections_3d
                                            └──────┬──────┘                 ▼
                                                   │ /left/depth     ┌────────────────────┐
                                                   ▼                 │  debug_node        │
                                         ┌─────────────────┐         │  (可视化画框)       │
                                         │ left_camera ──▶ │         └─────────┬──────────┘
                                         │ base_link (TF)  │                   │ /yolo/dbg_image/compressed
                                         └─────────────────┘                   ▼
                                                                       RViz / 下游决策
```

### 2.2 启动到运行的全过程

1. **准备图传**：远端/本机 `teleimager` 已把相机画面通过 ZMQ PUB 推到 `tcp://192.168.123.164:55555`（480×1280 双目左右拼接图）。
2. **启动双目链路**：`stereo_split_node` 订阅同一路 ZMQ 流，拆成左/右目发布到 `/left/image_raw`、`/right/image_raw`，并带相机内参 `/left/camera_info`（`stereo_split_node.py` 内置标定参数：fx=274.8, fy=272.6, 基线 8.35cm）。
3. **生成深度**：`sgbm_stereo_node` 用 StereoBM/SGBM 做立体匹配，发布视差 `/left/disparity` 和深度 `/left/depth`（单位：米）。
4. **发布 TF**：`static_transform_publisher` 发布 `left_camera → base_link` 的静态变换，让 3D 结果能换算到机器人本体坐标系。
5. **YOLO 推理**：`yolo_node` 从 ZMQ 取帧 → 裁切左半（480×640）→ 推理 → 发布 `/yolo/detections`（`DetectionArray`）与 `/yolo/image_raw`。
6. **目标跟踪**：`tracking_node` 同步 `image_raw + detections`，用 ByteTrack 给每个目标分配稳定 ID，发布 `/yolo/tracking`。
7. **3D 转换**：`detect_3d_node` 同步 `detections + depth + camera_info`，做前景聚类/深度采样，经 TF 转到 `base_link`，发布 `/yolo/detections_3d`。
8. **可视化**：`debug_node` 在图上画框/掩码/关键点，发布 `/yolo/dbg_image/compressed`（JPEG）及 RViz 用 Marker。
9. **消费结果**：下游（如 `eval_g1_yolo_stop.py`、`monitor_bbox.py`）订阅 `/yolo/detections*` 做停止判断 / 面积监测等。

> 生命周期说明：四个节点都是 **LifecycleNode**，`main()` 会自动执行 `trigger_configure() → trigger_activate() → rclpy.spin()`，无需手动操作。

---

## 3. 快速开始

> 目标：**5 分钟内**确认检测节点在跑、能出结果。

### 3.1 环境准备（首次）

```bash
# 激活 YOLO 环境（ROS2 + Ultralytics + torch 都装在这里）
conda activate YOLO

# 加载 ROS2 与工作空间
source /opt/ros/humble/setup.bash
source /home/unitree/YOLO_ws/install/setup.bash

# 验证关键依赖（必须全部 OK）
python3 -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
python3 -c "import ultralytics, numpy; print('ultralytics', ultralytics.__version__, 'numpy', numpy.__version__)"
```

### 3.2 启动（本地/开发机）

```bash
conda activate YOLO
source /opt/ros/humble/setup.bash
source /home/unitree/YOLO_ws/install/setup.bash
ros2 launch yolo_bringup yolo.launch.py \
    model:=/home/unitree/YOLO_ws/models/best.pt \
    device:=cuda:0 \
    threshold:=0.5 \
    use_tracking:=False \
    use_3d:=False \
    use_debug:=True
```

### 3.3 启动（生产/完整管线，一键）

```bash
bash /home/unitree/YOLO_ws/image_depth/run_stereo_yolo.sh
```

该脚本自动完成：清旧进程 → 启双目拆分 → 启立体匹配 → 发静态 TF → 启 YOLO 3D 管线 → 打印话题清单。

### 3.4 健康检查（新开一个终端）

```bash
source /opt/ros/humble/setup.bash
source /home/unitree/YOLO_ws/install/setup.bash

# ① 节点是否活着（期望看到 yolo_node 等）
ros2 node list | grep yolo

# ② ZMQ 图像是否在收（期望每秒 ~10Hz 以上）
ros2 topic hz /yolo/image_raw --window 20

# ③ 检测结果是否在出（期望有 detections 字段）
ros2 topic echo /yolo/detections --once
```

| 检查项 | 命令 | 预期结果 |
|--------|------|---------|
| 节点存活 | `ros2 node list` | 出现 `yolo_node`（及 tracking/detect_3d/debug） |
| 图像流入 | `ros2 topic hz /yolo/image_raw` | `average rate: > 10.0` |
| 检测输出 | `ros2 topic echo /yolo/detections --once` | `detections:` 非空、含 `class_name: cloth` |
| GPU 正常 | `nvidia-smi` | YOLO 进程占用显存 |

> 若 `②` 为 0Hz：说明 ZMQ 图传没连上（见 [12. 常见问题排查](#12-常见问题排查)）。
> 若 `③` 为空：说明模型没框到目标，可降低阈值或调整提示词（见 [7.5](#75-yolo-world-开放词汇检测)）。

---

## 4. 工程结构

```
/home/unitree/YOLO_ws/
├── README.md                       # 本文档
├── yolo_node_readme.md             # 旧版说明（保留参考）
├── readme 提示词.md                 # 文档写作模板（本 README 的写作依据）
├── monitor_bbox.py                 # 边界框面积监测工具（调停止阈值用）
├── echo_angle.py                   # 检测结果回显（OBB 角度，单位度）
├── carry_state/                    # 预留：搬运状态相关
├── image_depth/                    # ★ 双目深度 3D 检测（独立子模块）
│   ├── run_stereo_yolo.sh          # ★ 一键启动脚本（生产入口）
│   ├── stereo_split_node.py        # ZMQ 双目 → left/right 话题
│   ├── sgbm_stereo_node.py         # 立体匹配 → 视差/深度
│   ├── disparity_to_depth_node.py  # （备用）视差→深度
│   ├── calibrate_stereo.py         # 双目相机标定
│   ├── calibration_result.yaml     # 标定结果
│   ├── checkerboard_9x7_25mm.png   # 打印用棋盘格
│   ├── view_g1_urdf.sh             # G1 URDF + RViz 可视化
│   └── g1_urdf.rviz                # RViz 配置
├── models/                         # 模型文件
├── src/yolo_ros-main/              # ★ ROS2 包源码（上游 + 本地定制）
│   ├── pyproject.toml              # Python 依赖（numpy<2, ultralytics==8.4.6...）
│   ├── yolo_msgs/                  # 消息定义包（.msg / .srv）
│   ├── yolo_ros/                   # 节点包
│   │   └── yolo_ros/
│   │       ├── yolo_node.py        # 核心检测节点（ZMQ 版，本地定制）
│   │       ├── tracking_node.py    # 目标跟踪节点
│   │       ├── detect_3d_node.py   # 3D 检测节点
│   │       ├── debug_node.py       # 可视化节点
│   │       └── fix_indent.py       # 缩进修复小工具
│   └── yolo_bringup/               # 启动包
│       └── launch/
│           ├── yolo.launch.py      # 通用启动文件（最常用）
│           ├── yolo-world.launch.py# YOLO-World 启动
│           ├── yoloe.launch.py
│           └── yolov5~v26.launch.py# 各版本模型启动
├── build/                          # colcon 构建产物
└── install/                        # colcon 安装目录（source 用）
```

**模块职责**：

| 目录/文件 | 职责 |
|-----------|------|
| `src/yolo_ros-main/yolo_ros` | 4 个节点的实现（检测/跟踪/3D/可视化） |
| `src/yolo_ros-main/yolo_msgs` | 自定义消息与服务（DetectionArray、SetClasses 等） |
| `src/yolo_ros-main/yolo_bringup` | 各类启动文件（launch） |
| `image_depth/` | 双目拆分 + SGBM 深度，提供 3D 检测所需深度与 TF |
| `models/` | 训练好的 `.pt` 模型 |
| `monitor_bbox.py` | 独立工具：打印每个检测框面积，用于标定停止阈值 |
| `echo_angle.py` | 独立工具：回显检测框与 OBB 旋转角度（度） |
| `install/` `build/` | colcon 产物；`install` 需要被 `source` |

---

## 5. 配置说明

> 本项目**没有 yml/conf/env 配置文件**，所有配置都是 **launch 参数 → ROS2 参数** 的方式下发。改配置有两条通道：启动时传参、运行中 `ros2 param set`。

### 5.1 启动参数全字段（`yolo.launch.py`，按行注释）

| 参数 | 默认值 | 必填 | 说明 |
|------|--------|------|------|
| `model_type` | `YOLO` | 否 | 模型类型：`YOLO` / `World` / `YOLOE` |
| `model` | `yolov8m.pt` | 是* | 模型名或绝对路径（生产必填，如 `models/best.pt`） |
| `tracker` | `bytetrack.yaml` | 否 | 跟踪器配置（`bytetrack.yaml` / `botsort.yaml`） |
| `device` | `cuda:0` | 否 | 推理设备（`cuda:0` / `cpu`） |
| `fuse_model` | `False` | 否 | 是否 fuse 模型加速推理 |
| `yolo_encoding` | `bgr8` | 否 | 图像编码 |
| `enable` | `True` | 否 | 启动时是否启用检测 |
| `threshold` | `0.5` | 否 | 置信度阈值（越大越严） |
| `iou` | `0.7` | 否 | NMS IoU 阈值 |
| `imgsz_height/width` | `480/640` | 否 | 推理输入尺寸 |
| `half` | `False` | 否 | FP16 半精度加速 |
| `max_det` | `300` | 否 | 每帧最大检测数 |
| `augment` | `False` | 否 | 测试时增强（TTA，慢） |
| `agnostic_nms` | `False` | 否 | 跨类别 NMS |
| `retina_masks` | `False` | 否 | 高分辨率分割掩码 |
| `containment_threshold` | `0.7` | 否 | 包含过滤重叠阈值（见 [7.4](#74-包含关系过滤)） |
| `input_image_topic` | `/camera/rgb/image_raw` | 否 | 图像话题（仅影响 tracking/debug 的订阅；`yolo_node` 用 ZMQ 取图） |
| `image_reliability` | `1` | 否 | 图像 QoS：`0`=默认 `1`=Reliable `2`=BestEffort |
| `input_depth_topic` | `/camera/depth/image_raw` | 否 | 深度图话题 |
| `input_depth_info_topic` | `/camera/depth/camera_info` | 否 | 深度相机内参话题 |
| `depth_image_reliability` | `1` | 否 | 深度图 QoS |
| `depth_info_reliability` | `1` | 否 | 深度内参 QoS |
| `target_frame` | `base_link` | 否 | 3D 结果的目标坐标系 |
| `depth_image_units_divisor` | `1000` | 否 | 深度值→米 的除数（毫米图填 `1000`，已是米填 `1`） |
| `namespace` | `yolo` | 否 | 节点命名空间（影响话题前缀 `/yolo/*`） |
| `use_tracking` | `True` | 否 | 是否启用跟踪节点 |
| `use_3d` | `False` | 否 | 是否启用 3D 检测节点 |
| `use_debug` | `True` | 否 | 是否启用可视化节点 |

\* `model` 不传时用官方预训练模型（首次自动下载），生产环境务必显式指定 `models/best.pt`。

### 5.2 节点内部参数（`yolo_node.py` 声明，ZMQ 相关）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `zmq_host` | `192.168.123.164` | 图传 ZMQ 地址 |
| `zmq_port` | `55555` | 图传 ZMQ 端口 |
| `zmq_timeout_ms` | `5` | ZMQ 接收超时（毫秒） |
| `zmq_poll_period` | `0.01` | ZMQ 轮询周期（秒，约 100Hz） |
| `source_height` | `480` | 源图高（解码用） |
| `source_width` | `1280` | 源图宽（双目拼接，取左半） |

> `stereo_split_node.py` 也有同名 `zmq_*` 参数，以及标定参数 `fx/fy/cx/cy/baseline`（内置默认值来自 2026-06-04 标定）。

### 5.3 配置覆盖的两种方式

**方式 A：启动时覆盖（推荐）**

```bash
ros2 launch yolo_bringup yolo.launch.py model:=/home/unitree/YOLO_ws/models/best.pt threshold:=0.3
```

**方式 B：运行中覆盖（立即生效，无需重启）**

```bash
# 改置信度阈值
ros2 param set /yolo/yolo_node threshold 0.3
# 改包含过滤阈值
ros2 param set /yolo/yolo_node containment_threshold 0.5
# 查看当前所有参数
ros2 param get /yolo/yolo_node threshold
```

> 注意：`zmq_*`、`source_*` 参数在 `on_configure` 阶段读取，运行中 `ros2 param set` **不生效**，需要重启节点。

**③ 切换模型（不重启进程）**

`yolo_node` 是 LifecycleNode，模型只在 `activate` 时加载。通过生命周期状态机
`deactivate -> cleanup -> configure -> activate` 可让节点重新读取 `model` 参数并加载新模型，**进程不退出**。

**方式一：一键脚本（推荐，带状态探测与成功校验）**

```bash
# 一键切换（用 service call 实现，绕开本机 CLI graph 查询问题）
/home/unitree/YOLO_ws/switch_model.sh /home/unitree/YOLO_ws/models/bin_and_cloth-OBB.pt
# 顺带切换模型类型（可选第二个参数）
/home/unitree/YOLO_ws/switch_model.sh yolov8s-worldv2.pt World
```

> 注意：切换期间检测会中断几秒~几十秒（模型加载耗时），ZMQ 会断开重连。
> 本机 CycloneDDS discovery 存在 `empty node name` 问题，`ros2 lifecycle` / `ros2 param set` CLI 不可用，脚本直接用服务调用完成。

**方式二：纯 ROS 命令手动切换（不用脚本）**

脚本本质就是下面这串 `ros2 service call` 命令的封装，完全可以在终端里手敲（先 source 环境）：

```bash
source /opt/ros/humble/setup.bash
source /home/unitree/YOLO_ws/install/setup.bash

# ① 查看当前生命周期状态（可选）
ros2 service call /yolo/yolo_node/get_state lifecycle_msgs/srv/GetState "{}"

# ② 回退到 unconfigured（按当前状态选择）
#    当前 active(3)     → 先 deactivate(4)，再 cleanup(2)
#    当前 inactive(2)   → 直接 cleanup(2)
#    当前 unconfigured(1) → 跳过本步
ros2 service call /yolo/yolo_node/change_state lifecycle_msgs/srv/ChangeState "{transition: {id: 4}}"
ros2 service call /yolo/yolo_node/change_state lifecycle_msgs/srv/ChangeState "{transition: {id: 2}}"

# ③ 设置新模型参数（type 4 = string）
ros2 service call /yolo/yolo_node/set_parameters rcl_interfaces/srv/SetParameters \
  "{parameters: [{name: 'model', value: {type: 4, string_value: '/home/unitree/YOLO_ws/models/bin_and_cloth-OBB.pt'}}]}"
# 如需顺带切换模型类型（YOLO / World / YOLOE），再执行一句：
# ros2 service call /yolo/yolo_node/set_parameters rcl_interfaces/srv/SetParameters \
#   "{parameters: [{name: 'model_type', value: {type: 4, string_value: 'World'}}]}"

# ④ configure + activate（activate 才真正加载新模型，可能较慢）
ros2 service call /yolo/yolo_node/change_state lifecycle_msgs/srv/ChangeState "{transition: {id: 1}}"
ros2 service call /yolo/yolo_node/change_state lifecycle_msgs/srv/ChangeState "{transition: {id: 3}}"
```

**生命周期状态 / 转换 id 速查**

| 类型 | id | 含义 |
|------|----|------|
| 状态 | 1 | `unconfigured` |
| 状态 | 2 | `inactive` |
| 状态 | 3 | `active` |
| 状态 | 4 | `finalized` |
| 转换 | 1 | `configure` |
| 转换 | 2 | `cleanup` |
| 转换 | 3 | `activate` |
| 转换 | 4 | `deactivate` |
| 转换 | 5 | `shutdown` |

**为什么只能用 `service call`，不能直接用 `ros2 lifecycle set` / `ros2 param set`？**

`ros2 lifecycle`、`ros2 param` 这两个 CLI 内部要先通过 DDS graph 解析出**节点名**，再去找该节点下的服务/参数。本机（含远端 DDS 发现）存在 `empty node name` 问题：`ros2 node list` 里看不到 yolo 节点，`ros2 lifecycle get /yolo/yolo_node`、`ros2 param get /yolo/yolo_node model` 均报 `Node not found`。

而 `ros2 service call` 按**完整服务路径**（如 `/yolo/yolo_node/change_state`）直接寻址，不依赖 graph 节点名解析，因此可用。方式一脚本与上面的手动命令本质完全相同，只是额外做了状态探测与每一步的成功校验。

### 5.4 "Profile 切换"（不同场景 = 不同 launch 文件）

| 场景 | 使用的 launch | 要点 |
|------|--------------|------|
| 通用检测 | `yolo.launch.py` | 全参数可配，最常用 |
| 开放词汇检测 | `yolo-world.launch.py` | 自动用 `yolov8s-worldv2.pt`，threshold=0.25, max_det=10 |
| 特定版本模型 | `yolov5~yolov26.launch.py` | 如 `yolov11.launch.py` |
| 完整 3D 管线 | `run_stereo_yolo.sh` | 一键拉起双目+深度+检测+3D |

### 5.5 环境变量（Docker 场景说明）

本项目**默认不走 Docker**（直接 Conda + ROS2 进程）。若未来容器化部署，ROS2 参数不读环境变量，需要在容器入口把 `ENV` 映射为 launch 参数，例如：

```bash
docker run -e MODEL=/home/unitree/YOLO_ws/models/best.pt -e THRESHOLD=0.5 ... \
    bash -c "source install/setup.bash && ros2 launch yolo_bringup yolo.launch.py \
    model:=\$MODEL threshold:=\$THRESHOLD"
```

---

## 6. 对外 API（ROS2 话题 / 服务 / 消息）

> 本项目是 ROS2 节点，没有 REST 接口。"对外 API" = ROS2 话题、服务、消息。默认命名空间 `yolo`，以下话题实际为 `/yolo/*`。

### 6.1 话题总览

| 话题 | 消息类型 | 发布者 | 说明 |
|------|---------|--------|------|
| `/yolo/detections` | `yolo_msgs/DetectionArray` | yolo_node | 2D 检测结果 |
| `/yolo/image_raw` | `sensor_msgs/Image` | yolo_node | 裁切后的左目图（bgr8） |
| `/yolo/tracking` | `yolo_msgs/DetectionArray` | tracking_node | 带跟踪 ID 的结果 |
| `/yolo/detections_3d` | `yolo_msgs/DetectionArray` | detect_3d_node | 3D 检测（相对 `base_link`） |
| `/yolo/dbg_image/compressed` | `sensor_msgs/CompressedImage` | debug_node | 画框后的 JPEG 图 |
| `/yolo/dgb_bb_markers` | `visualization_msgs/MarkerArray` | debug_node | 3D 框 Marker（RViz） |
| `/yolo/dgb_kp_markers` | `visualization_msgs/MarkerArray` | debug_node | 3D 关键点 Marker（RViz） |
| `/left/image_raw` `/right/image_raw` | `sensor_msgs/Image` | stereo_split_node | 左右目原始图 |
| `/left/camera_info` | `sensor_msgs/CameraInfo` | stereo_split_node | 左目相机内参 |
| `/left/disparity` | `stereo_msgs/DisparityImage` | sgbm_stereo_node | 视差图 |
| `/left/depth` | `sensor_msgs/Image` (32FC1) | sgbm_stereo_node | 深度图（米） |

**订阅示例（消费检测结果）**：

```bash
ros2 topic echo /yolo/detections --once
ros2 topic hz /yolo/detections
```

### 6.2 服务总览

| 服务 | 类型 | 提供者 | 说明 |
|------|------|--------|------|
| `/yolo/enable` | `std_srvs/SetBool` | yolo_node | 暂停/恢复检测 |
| `/yolo/set_classes` | `yolo_msgs/SetClasses` | yolo_node | 设置检测类别（仅 World/YOLOE 模型有） |
| `/yolo/set_containment_rule` | `yolo_msgs/SetContainmentRule` | yolo_node | 设置包含过滤规则 |

**① 暂停/恢复检测**

```bash
# 暂停
ros2 service call /yolo/enable std_srvs/srv/SetBool "{data: false}"
# 恢复
ros2 service call /yolo/enable std_srvs/srv/SetBool "{data: true}"
```

**② 设置检测类别（World 模型）**

```bash
ros2 service call /yolo/set_classes yolo_msgs/srv/SetClasses "{classes: ['cloth', 'person']}"
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `classes`（请求） | `string[]` | 要检测的类别名列表 |
| 响应 | — | 无返回字段（成功与否看节点日志） |

**③ 设置包含过滤规则**

```bash
ros2 service call /yolo/set_containment_rule yolo_msgs/srv/SetContainmentRule \
  "{outer_class: 'blue material box', inner_class: 'transparent plastic bag with clothes'}"
# 清除规则（outer_class 传空串）
ros2 service call /yolo/set_containment_rule yolo_msgs/srv/SetContainmentRule \
  "{outer_class: '', inner_class: ''}"
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `outer_class`（请求） | `string` | 容器类别名 |
| `inner_class`（请求） | `string` | 被包含类别名 |
| `success`（响应） | `bool` | 是否设置成功 |
| `message`（响应） | `string` | 状态消息 |

**错误/异常情况**：

| 场景 | 现象 | 处理 |
|------|------|------|
| 模型不支持 set_classes | 节点里没有 `/yolo/set_classes` 服务 | 用 World 类型模型（`model_type:=World`） |
| `clip` 未安装 | 日志报 `No module named 'clip'` | 按 [12.Q3](#常见问题排查) 安装 CLIP |
| 类别名不匹配 | 包含过滤后内层目标被全滤掉 | 确保 outer/inner 与 set_classes 完全一致 |

### 6.3 核心消息结构（`yolo_msgs`）

```
DetectionArray.msg
  std_msgs/Header header
  yolo_msgs/Detection[] detections

Detection.msg
  int32 class_id            # 类别 ID
  string class_name         # 类别名（如 "cloth"）
  float64 score             # 置信度
  string id                 # 跟踪 ID
  BoundingBox2D bbox        # 2D 框（像素，xywh+theta）
  BoundingBox3D bbox3d      # 3D 框（米）
  Mask mask                 # 分割掩码
  KeyPoint2DArray keypoints    # 2D 关键点
  KeyPoint3DArray keypoints3d  # 3D 关键点
```

> `BoundingBox2D.center` 是 `Pose2D`，其 `theta` 字段保存 **OBB 旋转角（弧度）**：
> 普通（非 OBB）模型恒为 `0.0`；OBB 模型（`best-OBB.pt`）为实际旋转角（约 -90°~90°）。

### 6.4 参数速查（对外可调）

| 参数 | 节点 | 说明 |
|------|------|------|
| `threshold` | yolo_node | 置信度阈值 |
| `iou` | yolo_node | NMS IoU |
| `imgsz_height/width` | yolo_node | 推理分辨率 |
| `half` | yolo_node | FP16 |
| `containment_threshold` | yolo_node | 包含过滤阈值 |
| `tracker` | tracking_node | 跟踪器配置 |
| `target_frame` | detect_3d_node | 3D 目标坐标系 |
| `depth_image_units_divisor` | detect_3d_node | 深度单位换算 |
| `jpeg_quality` | debug_node | 调试图 JPEG 质量（默认 80） |

---

## 7. 核心功能

### 7.1 ZMQ 图像输入与左半裁切

`yolo_node` **不订阅 ROS2 图像话题**，而是直接连 ZMQ SUB 拿图，好处是低延迟、不占 ROS 带宽。

- 默认端点：`tcp://192.168.123.164:55555`
- 协议：ZMQ SUB，`CONFLATE=1`（只留最新帧，防堆积）
- 支持载荷：JPEG 编码、原始 RGB uint8（480×1280×3）、原始灰度（480×1280）
- 裁切：取左半 `480×640` 作为推理输入，`header.frame_id = "zmq_left_camera"`

```bash
# 改图传地址/端口（需重启生效）
ros2 launch yolo_bringup yolo.launch.py \
    zmq_host:=192.168.123.164 zmq_port:=55555
```

### 7.2 目标跟踪

`tracking_node` 用 `ApproximateTimeSynchronizer` 对齐 `image_raw + detections`，为每个目标分配稳定的 `id` 字段。

```bash
# 换跟踪器
ros2 launch yolo_bringup yolo.launch.py tracker:=botsort.yaml

# 查看带 ID 的结果
ros2 topic echo /yolo/tracking --once
```

### 7.3 3D 检测（双目深度）

`detect_3d_node` 同步 `detections + depth + camera_info` 三路消息，把 2D 框转为 3D 坐标：

- 基于密度分析的前景聚类（区分前景/背景）
- 空间加权 + 自适应阈值 + 多模态深度处理
- 有分割掩码时按掩码采样，否则按框内采样
- 经 TF 转到 `target_frame`（默认 `base_link`）

```bash
# 深度图已是米（sgbm_stereo_node 输出），所以 divisor:=1
ros2 launch yolo_bringup yolo.launch.py \
    use_3d:=True \
    input_image_topic:=/left/image_raw \
    input_depth_topic:=/left/depth \
    input_depth_info_topic:=/left/camera_info \
    depth_image_units_divisor:=1 \
    target_frame:=base_link

# 看 3D 结果
ros2 topic echo /yolo/detections_3d --once
```

> 生产环境直接用 `run_stereo_yolo.sh`，参数已配好。3D 精度依赖：深度图质量、相机内参、TF 正确性。

### 7.4 包含关系过滤

> 本地定制功能（上游 yolo_ros 无此能力）：判断"A 是否在 B 容器内"，常用于"衣物是否放进了蓝色物料箱"。

**逻辑**：设 `inner_class` 必须被 `outer_class` 框包含（重叠比例 ≥ `containment_threshold`）才发布，否则静默丢弃。

```bash
# ① 设置两个类别
ros2 service call /yolo/set_classes yolo_msgs/srv/SetClasses \
  "{classes: ['blue material box', 'transparent plastic bag with clothes']}"

# ② 设置包含规则
ros2 service call /yolo/set_containment_rule yolo_msgs/srv/SetContainmentRule \
  "{outer_class: 'blue material box', inner_class: 'transparent plastic bag with clothes'}"

# ③ 调宽松度（0~1，默认 0.7）
ros2 param set /yolo/yolo_node containment_threshold 0.5   # 更宽松
ros2 param set /yolo/yolo_node containment_threshold 1.0   # 严格完全包含
```

**效果表**（threshold=0.7）：

| 衣物框与蓝箱框关系 | 重叠比例 | 输出 |
|---|---|---|
| 完全在箱内 | 1.0 | ✅ 衣物 + 蓝箱都发布 |
| 大部分在箱内（边缘） | ≥0.7 | ✅ 都发布 |
| 少部分在箱内 | <0.7 | ❌ 衣物被过滤，只发蓝箱 |
| 完全在箱外 / 没检测到箱 | 0.0 | ❌ 衣物被过滤 |

**注意**：`max_det` 要够大（world launch 默认 10，够用）；类别名必须与 `set_classes` 完全一致。

### 7.5 YOLO-World 开放词汇检测

用自然语言描述"要检测什么"，无需预训练类别。底层用 CLIP 文本编码器把提示词转成特征与图像特征匹配。

```bash
# 启动 World 节点
ros2 launch yolo_bringup yolo-world.launch.py \
    model:=yolov8s-worldv2.pt device:=cuda:0 use_debug:=True

# 设置提示词（核心用法）
ros2 service call /yolo/set_classes yolo_msgs/srv/SetClasses \
  "{classes: ['cloth', 'clothes in plastic bag', 'jacket']}"
```

**提示词技巧**：

| 场景 | 推荐提示词 |
|------|-----------|
| 裸眼可见衣物 | `cloth` / `clothing` / `garment` |
| 袋装衣物 | `clothes in plastic bag` / `packaged clothing` |
| 具体服装 | `jacket` / `down jacket` / `T-shirt` |
| 中文 | `衣服` / `羽绒服` / `包装袋里的衣服` |

**核心原则**：描述**视觉外观**而非功能（❌ `warm clothes` → ✅ `red jacket`）；被遮挡就带上容器描述；一次设多个近义词提高命中率；检测不到就先降阈值 `ros2 param set /yolo/yolo_node threshold 0.25`。

### 7.6 边界框面积监测

`monitor_bbox.py` 用于观察 cloth 框面积占比，帮助确定"停止/抓取"阈值。

```bash
conda activate YOLO
source /opt/ros/humble/setup.bash
source /home/unitree/YOLO_ws/install/setup.bash
python3 /home/unitree/YOLO_ws/monitor_bbox.py          # 默认监听 /yolo/detections
python3 /home/unitree/YOLO_ws/monitor_bbox.py /detections  # 无命名空间时
```

输出示例：

```
[cloth      ]  置信度=0.87  bbox=(312×260)  面积=   81120  占画面= 26.4%
```

### 7.7 OBB 旋转角度输出

**前置条件**：必须用 OBB 模型 `best-OBB.pt` 启动，普通 `best.pt` 的 `theta` 恒为 `0.0`。

#### a. 启动 OBB 检测

```bash
conda activate YOLO
source /opt/ros/humble/setup.bash
source /home/unitree/YOLO_ws/install/setup.bash

ros2 launch yolo_bringup yolo.launch.py \
    model:=/home/unitree/YOLO_ws/models/best-OBB.pt \
    device:=cuda:0 \
    threshold:=0.5 \
    use_tracking:=False \
    use_3d:=False \
    use_debug:=True
```

#### b. 直接查看原始消息（弧度）

旋转角存放在 `detections[i].bbox.center.theta`，单位是**弧度**（范围约 -90°~90°，即 $\pm\pi/2$）：

```bash
ros2 topic echo /yolo/detections --once
```

输出示例：

```
detections:
-
  class_name: cloth
  score: 0.87
  bbox:
    center:
      position:
        x: 320.5
        y: 240.2
      theta: 0.5235        # ★ 旋转角，弧度
    size:
      x: 123.4
      y: 56.7
```

> 若 `theta` 一直是 `0.0`，说明模型没走 OBB 分支——确认加载的是 `best-OBB.pt`。

#### c. 一行命令转成角度（度）

```bash
ros2 topic echo /yolo/detections --once | python3 -c "
import sys, re, math
data = sys.stdin.read()
for m in re.finditer(r'theta: ([\-\d.]+)', data):
    print(f'旋转角 = {math.degrees(float(m.group(1))):.1f} 度')
"
```

#### d. 直接回显角度（推荐）

`ros2 topic echo` 只能打印消息原始字段（`theta` 为弧度），无法直接转度数。
用 `echo_angle.py`（与 `monitor_bbox.py` 风格一致）即可**角度直出**：

```bash
conda activate YOLO
source /opt/ros/humble/setup.bash
source /home/unitree/YOLO_ws/install/setup.bash
python3 /home/unitree/YOLO_ws/echo_angle.py            # 默认监听 /yolo/detections
python3 /home/unitree/YOLO_ws/echo_angle.py /detections  # 无命名空间时
```

输出示例（角度直接是**度**）：

```
[cloth      ]  置信度=0.87  中心=( 320.5, 240.2)  尺寸=(123.4x 56.7)  角度=   30.0 度  (弧度 0.5235)
```

> 也可加别名方便敲：`alias yolo_echo='python3 /home/unitree/YOLO_ws/echo_angle.py'`

#### e. 调试画面标签显示角度

`debug_node` 的框标签现在会直接显示旋转角（仅当 `theta != 0` 时，普通模型不显示）：

```
cloth (0.870) ang=30.0
```

---

## 8. 日志查看

> 本项目无容器，节点日志直接打到 stdout/stderr，同时 ROS2 会归档到 `~/.ros/log/`。

### a. 进程日志的 6+ 种查看方式

```bash
# ① 前台启动（最常用）：日志直接滚动在终端
ros2 launch yolo_bringup yolo.launch.py model:=/home/unitree/YOLO_ws/models/best.pt

# ② 后台启动 + 实时跟踪
nohup ros2 launch yolo_bringup yolo.launch.py model:=/home/unitree/YOLO_ws/models/best.pt \
    > /tmp/yolo.log 2>&1 &
tail -f /tmp/yolo.log

# ③ 只看最近 N 行
tail -n 100 /tmp/yolo.log

# ④ 按时间筛选（结合 grep 时间戳）
grep '2026-08-03' /tmp/yolo.log

# ⑤ less 翻页（/ 搜索，q 退出）
less /tmp/yolo.log

# ⑥ 若走 systemd：journalctl
journalctl -u yolo -f

# ⑦ GUI 日志查看器
rqt_console
```

### b. ROS2 日志文件查看（按时间归档）

ROS2 自动把每次运行的日志写到 `~/.ros/log/`，目录名含时间戳，可用于回溯历史：

```bash
ls -lt ~/.ros/log/ | head -10        # 按时间倒序看
# 例：~/.ros/log/yolo_node_2886_1785718674039.log

# 打开最新一次 yolo_node 日志
tail -f $(ls -t ~/.ros/log/yolo_node_*.log | head -1)
```

### c. 关键词检索（统计 / 去重 / 找错）

```bash
LOG=/tmp/yolo.log

# 统计出现次数
grep -c "Connected to ZMQ" $LOG
# 只看错误级别
grep -E "\[ERROR\]|\[WARN\]" $LOG
# 去重后看（适合刷屏的警告）
grep -E "\[WARN\]" $LOG | sort | uniq -c | sort -rn
# 错误前 3 行上下文
grep -B3 -A1 "ERROR" $LOG
# 看某进程的日志（ROS2 归档按节点分文件）
grep "Filtered out" ~/.ros/log/yolo_node_*.log
```

### d. 从另一台机器/另一个终端看

```bash
# 远程（把日志拷回来看）
ssh user@host "tail -f /home/unitree/YOLO_ws/log/yolo.log"
# 或直接在远程订阅话题看数据（不依赖日志）
ros2 topic echo /yolo/detections --once
```

### e. 清空日志

```bash
# 清自定义日志文件
> /tmp/yolo.log
# 清 ROS2 归档（建议先备份）
rm -rf ~/.ros/log/*
# 清工作空间构建日志
rm -rf /home/unitree/YOLO_ws/log/*
```

### 关键日志含义

| 日志行 | 含义 | 正常？ |
|--------|------|--------|
| `[yolo_node] Configuring...` → `Configured` | 读取参数、建发布者成功 | ✅ |
| `[yolo_node] Activating...` → `Activated` | 加载模型、连 ZMQ、建服务成功 | ✅ |
| `Connected to ZMQ image source: tcp://192.168.123.164:55555` | ZMQ 已连接（不代表有数据） | ✅ |
| `Trying to fuse model...` | 正在 fuse 模型加速 | ✅ 可选 |
| `Model file 'xxx' does not exists` | 模型路径错误 | ❌ 检查 model 参数 |
| `Failed to receive ZMQ image: ...` | ZMQ 收帧异常（网络/图传中断） | ❌ 检查图传 |
| `Received undecodable ZMQ image payload` | 帧格式无法解析 | ❌ 检查图传格式 |
| `Unexpected frame shape (H,W) expected at least (480,1280)` | 帧尺寸不对 | ❌ 检查 source_* 参数 |
| `Setting classes: [...]` / `New classes: {...}` | set_classes 生效 | ✅ |
| `Containment rule: 'A' must be inside 'B'` | 包含规则已设置 | ✅ |
| `Filtered out 'cloth' (best containment ratio 0.2 < 0.7)` | 目标因未在容器内被过滤 | ⭐ 预期行为 |
| `Clearing CUDA cache` | 节点 deactivate，释放显存 | ✅ |

---

## 9. 部署与运维

### 9.1 一键部署 / 启动

```bash
# 生产完整管线（双目 + 深度 + 检测 + 3D）
bash /home/unitree/YOLO_ws/image_depth/run_stereo_yolo.sh
```

### 9.2 常用运维命令

```bash
# 查看所有相关进程
ps aux | grep -E "yolo_node|tracking_node|detect_3d|debug_node|stereo_split|sgbm_stereo" | grep -v grep

# 停止所有相关进程（脚本内部也是这么清的）
pkill -f "yolo_node"; pkill -f "tracking_node"; pkill -f "detect_3d"
pkill -f "debug_node"; pkill -f "stereo_split"; pkill -f "sgbm_stereo"

# 查看节点/话题/服务是否齐全
ros2 node list
ros2 topic list | grep -E "yolo|left"
ros2 service list | grep yolo

# 占用显存检查
nvidia-smi

# 手动重建（改了 msg/launch 后）
cd /home/unitree/YOLO_ws && source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### 9.3 Conda 环境管理

```bash
# YOLO 环境（检测节点用；必须 numpy<2 以兼容 cv_bridge）
conda activate YOLO
pip install "numpy<2" ultralytics==8.4.6 pyzmq

# teleimager 环境（双目拆分 / 立体匹配用）
conda activate teleimager
pip install pyzmq opencv-python numpy
```

### 9.4 开机自启

当前 YOLO 节点为**手动/脚本启动**。若需开机自启，可配置 systemd（示例，`/etc/systemd/system/yolo.service`）：

```ini
[Unit]
Description=YOLO 3D detection pipeline
After=network.target

[Service]
Type=simple
User=unitree
Environment="PATH=/home/unitree/miniconda3/envs/YOLO/bin:/usr/bin:/bin"
ExecStart=/bin/bash /home/unitree/YOLO_ws/image_depth/run_stereo_yolo.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now yolo
sudo systemctl status yolo
```

> 图传侧 `teleimager` 自带 `setup_autostart.sh`，可参考其自启方式。

---

## 10. 改代码后怎么上线

### 10.1 Git 流程

```bash
# 查看改动
cd /home/unitree/YOLO_ws && git status
# 提交
git add src/yolo_ros-main/yolo_ros/yolo_ros/yolo_node.py
git commit -m "feat: xxx"
git push
# 从上游拉更新（注意：上游可能覆盖本地定制，见 10.3）
git pull
```

### 10.2 改动类型 → 需要做什么

| 改动内容 | 重建 colcon？ | 重启节点？ | 说明 |
|---------|--------------|-----------|------|
| 改 Python 业务代码（yolo_node.py 等） | 否（`--symlink-install`） | 是 | 重启对应节点/launch 即生效 |
| 改 launch 文件 | 否 | 是 | 重启 launch 即生效 |
| 改 `.msg` / `.srv` 消息 | **是** | 是 | 必须 `colcon build` + 重新 source |
| 新增节点/改 setup.py | **是** | 是 | 必须重建 |
| 换模型 `.pt` | 否 | 是 | 改 `model:=` 参数重启即可 |
| 改 `package.xml` 依赖 | 是 | 是 | 重建 + 重装依赖 |

### 10.3 特别提醒（本地定制风险）

1. **ZMQ 版 `yolo_node.py` 是本地的关键定制**（上游默认版本是订阅 ROS 图像话题，不连 ZMQ）。`colcon build` / `git pull` 上游**可能覆盖**它。构建/更新后请确认：
   ```bash
   grep -c "zmq_host" /home/unitree/YOLO_ws/src/yolo_ros-main/yolo_ros/yolo_ros/yolo_node.py
   # 期望输出 ≥ 1（有 zmq 参数 = 定制版还在）
   ```
2. **入口脚本 shebang**：`install/yolo_ros/lib/yolo_ros/` 下 4 个入口脚本的 shebang 必须指向 YOLO 环境 Python（否则 `ModuleNotFoundError: ultralytics`）：
   ```bash
   head -1 /home/unitree/YOLO_ws/install/yolo_ros/lib/yolo_ros/yolo_node
   # 期望：#!/home/unitree/miniconda3/envs/YOLO/bin/python
   # 重建后若被改回 /usr/bin/python3，手动改回
   ```
3. **tracking_node 兼容**：Ultralytics 新版 `BYTETracker.__init__()` 只接受 `args`，不要再传 `frame_rate`（已修复，别回退）。

---

## 11. 二次开发

### 11.1 业务代码加在哪里

| 想做什么 | 改哪个文件 | 关键位置 |
|---------|-----------|---------|
| 检测后处理（过滤/规则） | `yolo_ros/yolo_ros/yolo_node.py` | `process_image()` → `_apply_containment_filter()` |
| 改 ZMQ 取图/裁切逻辑 | 同上 | `zmq_image_cb()` / `_decode_zmq_image()` |
| 新增检测类型/输出字段 | `yolo_msgs/msg/*.msg` + `yolo_node.py` | 先加消息，再填字段 |
| 新增一个节点 | `yolo_ros/yolo_ros/` 新建 `.py` | 见 11.3 |
| 改启动组合 | `yolo_bringup/launch/yolo.launch.py` | 增删 `Node(...)` |
| 深度/3D 算法 | `detect_3d_node.py` | `detections_cb()` 附近的采样/聚类 |
| 双目/深度链路 | `image_depth/stereo_split_node.py`、`sgbm_stereo_node.py` | 各自 `__init__` 声明参数处 |

### 11.2 核心 API 速查（`yolo_node` 内部）

| API | 类型 | 说明 |
|-----|------|------|
| `YoloNode(LifecycleNode)` | 类 | 生命周期节点，重写 `on_configure/on_activate/on_deactivate` |
| `self.declare_parameter(name, default)` | 方法 | 声明参数（`__init__` 中） |
| `self.get_parameter(name).get_parameter_value()` | 方法 | 读取参数（`on_configure` 中） |
| `self.create_lifecycle_publisher(type, topic, qos)` | 方法 | 发布检测结果（生命周期发布者） |
| `self.create_publisher(type, topic, qos)` | 方法 | 普通发布（如 `image_raw`） |
| `self.create_service(srv_type, name, cb)` | 方法 | 对外服务 |
| `self.create_timer(period, cb)` | 方法 | 周期任务（ZMQ 轮询） |
| `self.yolo.predict(source=..., conf=..., iou=..., imgsz=...)` | 方法 | Ultralytics 推理入口 |
| `parse_hypothesis / parse_boxes / parse_masks / parse_keypoints` | 方法 | 把 YOLO 结果转成 ROS 消息 |

### 11.3 新增一个节点的步骤

```bash
# ① 在 yolo_ros 包下新建文件
#    复制 tracking_node.py 的骨架，改类名与话题
# ② 注册入口（setup.py）
#    entry_points["console_scripts"] 加一行: "my_node = yolo_ros.my_node:main"
# ③ 在 launch 里加 Node(...)（带 condition 可开关）
# ④ 重建
cd /home/unitree/YOLO_ws && source /opt/ros/humble/setup.bash
colcon build --symlink-install && source install/setup.bash
ros2 run yolo_ros my_node
```

### 11.4 与下游集成的约定

- 消费 `/yolo/detections`（2D）或 `/yolo/detections_3d`（3D），消息类型 `yolo_msgs/DetectionArray`
- 3D 框的 `bbox3d.frame_id` = `target_frame`（默认 `base_link`），尺寸单位米
- 停止类逻辑参考：`eval_g1_yolo_stop.py`（用 `YOLO_STOP_CLASS=cloth` + `YOLO_STOP_AREA_THRESHOLD` 等环境变量控制），先用 `monitor_bbox.py` 实测面积再定阈值

---

## 12. 常见问题排查

### Q1 服务起不来：节点启动即报错退出

**症状**：`ros2 launch` 后终端打印 `Model file 'xxx' does not exists` 或 `ModuleNotFoundError: ultralytics`，进程退出。

**原因**：模型路径不对 / 入口脚本 shebang 指向了系统 Python。

**解决**：

```bash
# 检查模型路径
ls -l /home/unitree/YOLO_ws/models/best.pt
# 检查入口脚本 shebang（必须指向 YOLO 环境）
head -1 /home/unitree/YOLO_ws/install/yolo_ros/lib/yolo_ros/yolo_node
# 不对则改回：
sed -i '1c #!/home/unitree/miniconda3/envs/YOLO/bin/python' \
    /home/unitree/YOLO_ws/install/yolo_ros/lib/yolo_ros/{yolo_node,tracking_node,detect_3d_node,debug_node}
```

### Q2 连不上设备：`ros2 topic hz /yolo/image_raw` 为 0

**症状**：节点活着，但没有图像进来，`/yolo/detections` 一直空。

**原因**：ZMQ 图传（teleimager）没开 / 网络不通 / 端口不对。

**解决**：

```bash
# 检查图传进程
ps aux | grep -E "teleimager|image_server" | grep -v grep
# 检查端口连通性
nc -zv 192.168.123.164 55555
# 检查 ZMQ 是否收到帧（看节点日志）
grep "Failed to receive ZMQ\|undecodable" ~/.ros/log/yolo_node_*.log | tail
# 确认图传格式：应支持 JPEG / 480×1280×3 RGB / 480×1280 灰度
```

### Q3 日志没写出来：`set_classes` 报 `No module named 'clip'`

**症状**：调用 `/yolo/set_classes` 后节点日志报 `Error setting classes: No module named 'clip'`。

**原因**：YOLO-World 需要 CLIP 依赖，未安装。

**解决**：

```bash
conda activate YOLO
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY   # 有代理时先清
pip install open-clip-torch
pip install git+https://github.com/ultralytics/CLIP.git
# 重启节点
```

### Q4 配置不生效：改了 threshold 没反应

**症状**：`ros2 param set` 后行为没变；或改了 `zmq_host` 重启也没用。

**原因**：`zmq_*`/`source_*` 只在 `on_configure` 读一次，运行时 set 不生效；或改的是旧进程。

**解决**：

```bash
# 确认改的是哪个进程/命名空间
ros2 param list /yolo/yolo_node
# 运行时只改 on_configure 之后读的参数（threshold/containment_threshold 可以）
ros2 param set /yolo/yolo_node threshold 0.3
# zmq/source 参数必须在启动时传，且重启节点：
ros2 launch yolo_bringup yolo.launch.py zmq_host:=192.168.123.164 ...
# 检查是否有多个 yolo_node 实例在跑（旧实例占用端口）
ps aux | grep yolo_node | grep -v grep
```

### Q5 重启后服务没拉起来（开机场景）

**症状**：机器重启后没有检测结果，`ros2 node list` 为空。

**原因**：YOLO 节点默认无 systemd 自启；或 conda 环境没激活导致入口脚本找不到解释器。

**解决**：

```bash
# 手动拉起
bash /home/unitree/YOLO_ws/image_depth/run_stereo_yolo.sh
# 需要自启则配置 systemd（见 9.4），注意 Environment 里带上 YOLO 环境 PATH
systemctl --user status yolo 2>/dev/null || journalctl -u yolo -n 50
```

### Q6 检测速度慢

**解决**：

```bash
# 按需组合：FP16 + 小分辨率 + 小模型 + 限检测数
ros2 launch yolo_bringup yolo.launch.py \
    half:=True imgsz_height:=320 imgsz_width:=320 \
    model:=yolov8n.pt max_det:=100 augment:=False
```

### Q7 3D 结果不准

**排查顺序**：深度图质量（`ros2 topic echo /left/depth` 是否合理）→ 内参（`/left/camera_info`）→ TF（`ros2 run tf2_ros tf2_echo left_camera base_link`）→ 单位（`depth_image_units_divisor`）。

### Q8 跟踪 ID 频繁跳变

**解决**：降低 `threshold` 减少漏检；调 `bytetrack.yaml` / `botsort.yaml` 参数；保证帧率稳定。

---

## 13. 领域基础知识

| 概念 | 说明 |
|------|------|
| **YOLO 目标检测** | 一次性预测图上所有目标的位置和类别，输出 N 个 `(class, score, box)` |
| **置信度阈值 threshold** | 低于该分数的检测被丢弃。越高越准但漏检多，越低越全但误检多 |
| **IoU / NMS** | 两个框重叠程度（交集÷并集）；NMS 去掉同一个目标上的重复框。`iou` 参数控制去重严格度 |
| **开放词汇检测** | YOLO-World/YOLOE：用文本提示词（经 CLIP 编码）匹配任意目标，无需固定类别 |
| **目标跟踪** | ByteTrack/BOT-SORT：跨帧关联同一目标并分配稳定 ID（`Detection.id`） |
| **相机内参** | fx/fy（焦距像素）、cx/cy（主点）、基线：把像素坐标还原为空间射线，是 2D→3D 的前提 |
| **深度图** | 每个像素存该点到相机的距离；`sgbm_stereo_node` 由双目视差算出（米） |
| **TF 坐标系** | `left_camera → base_link` 的变换把相机坐标系下的 3D 点转到机器人本体坐标系 |
| **包含关系过滤** | 用 IoU 式重叠比例判断"目标在不在容器框内"，业务后处理 |
| **LifecycleNode** | ROS2 生命周期节点：Unconfigured→Inactive→Active 状态机，`main()` 自动触发 |

---

## 14. 第三方 SDK / 库速查表

| 库 | 版本/来源 | 用途 | 常用入口 |
|----|----------|------|---------|
| `ultralytics` | ==8.4.6（Conda YOLO） | YOLO 推理 | `YOLO(model).predict(source=, conf=, iou=, imgsz=)` |
| `torch` | CUDA 版（Conda YOLO） | 深度学习后端 | `torch.cuda.is_available()` |
| `opencv-python` | ≥4.8.1.78 | 图像编解码、画框 | `cv2.imdecode / cvtColor / rectangle` |
| `pyzmq` | 任意（Conda 两环境） | ZMQ SUB 收图 | `ctx.socket(zmq.SUB).connect("tcp://ip:port")` |
| `cv_bridge` | ROS2 Humble | ROS Image ↔ OpenCV | `CvBridge().cv2_to_imgmsg(img, "bgr8")` |
| `message_filters` | ROS2 Humble | 多话题时间同步 | `ApproximateTimeSynchronizer([...], 10, 0.5)` |
| `tf2_ros` | ROS2 Humble | 坐标系变换 | `TransformListener(buffer, node)` |
| `lap` | ≥0.5.12 | 线性分配（跟踪用） | 供 bytetrack/botsort 内部使用 |
| `open-clip-torch` / `ultralytics/CLIP` | YOLO-World 必需 | 文本-图像对齐 | 供 `model.set_classes()` 使用 |
| `rclpy` | ROS2 Humble | 节点/话题/服务框架 | `create_node / create_publisher / create_service` |
| `sensor_msgs` / `visualization_msgs` / `stereo_msgs` | ROS2 Humble | 标准消息 | Image / CameraInfo / MarkerArray / DisparityImage |

---

## 项目信息

| 项 | 值 |
|----|----|
| 项目名 | YOLO 检测节点（YOLO_ws 工作空间） |
| 仓库地址 | 本地 `/home/unitree/YOLO_ws`；上游 [mgonzs13/yolo_ros](https://github.com/mgonzs13/yolo_ros) v4.6.0 |
| 所属团队 | bsd-具身智能开发团队 |
| 一句话定位 | 为 G1 机器人提供实时目标检测与 3D 定位能力 |
| 技术栈 | ROS2 Humble + Python 3.10 + Ultralytics YOLO 8.4.6 + PyTorch CUDA + ZeroMQ + OpenCV |
| 部署环境 | Linux（Ubuntu 22.04）/ Conda（YOLO、teleimager）|
| 文档版本 | v2.0 |
| 最后更新 | 2026-08-03 |
