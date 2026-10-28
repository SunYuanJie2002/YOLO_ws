# YOLO_ws: 宇树 G1 人形机器人双目空间感知与 YOLO 目标检测工作空间

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green.svg)](https://www.python.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLOv8-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Platform](https://img.shields.io/badge/Platform-Jetson%20Orin%20NX-lightgrey.svg)](https://www.nvidia.com/zh-cn/autonomous-machines/embedded-systems/jetson-orin/)

> **项目归档与留档指南**  
> 适用硬件：Unitree G1 人形机器人（板载计算单元 Jetson Orin NX）及配套工控机 / 算力背包  
> 维护人员与后续开发者参考手册

---

## 目录
1. [项目简介与应用背景](#1-项目简介与应用背景)
2. [系统整体架构与数据流](#2-系统整体架构与数据流)
3. [工作空间目录结构解析](#3-工作空间目录结构解析)
4. [环境配置与依赖安装](#4-环境配置与依赖安装)
5. [快速启动与运行指南](#5-快速启动与运行指南)
6. [模型热切换机制 (switch_model.sh)](#6-模型热切换机制-switch_modelsh)
7. [双目相机标定与深度解算 (image_depth)](#7-双目相机标定与深度解算-image_depth)
8. [ROS2 话题与接口参考](#8-ros2-话题与接口参考)
9. [配置中心参数说明 (run_config.sh)](#9-配置中心参数说明-run_configsh)
10. [常见问题排查 (FAQ)](#10-常见问题排查-faq)

---

## 1. 项目简介与应用背景

本项目是部署在 **宇树 G1 (Unitree G1) 人形机器人** 上的核心视觉感知算法工作空间。

机器人头部搭载了双目图传相机，通过高速 ZeroMQ（ZMQ）管道以广播形式向板载计算单元推流（默认分辨率为 `480×1280` 双目拼接图）。本工程的主要职责是：
1. **实时 2D 目标检测与跟踪（核心独立能力）**：裁切并提取左目前视图像，直接运行 YOLO 模型实时识别画面中的衣物（`cloth`）、收纳框（`bin`）、机械臂末端（`arm`）等关键目标，并使用 ByteTrack 进行多目标连续追踪，**无需深度信息即可高速独立运行**。
2. **OBB 旋转矩形框与姿态解算**：支持定向旋转框（Oriented Bounding Box）检测，输出物体在视野平面的旋转偏转角，引导机械臂末端对准抓取姿态。
3. **双目空间 3D 空间定位（可选融合能力）**：当开启 3D 功能时，通过半全局块匹配（SGBM）解算视差与深度图，将 2D 目标框映射为相对机器人本体 `base_link` 坐标系的 3D 空间立方体坐标 $(X, Y, Z)$，为机械臂空间抓取分拣与避障提供三维决策依据。

> **💡 核心设计理念：完全解耦与自由选择**  
> 我们的数据流**不强依赖深度环节**！在项目中可以**完全自主选择进行纯 2D 检测还是 3D 检测**：
> * **纯 2D 模式**：只消费彩色图像，极简高速，算力开销极小，直接输出 2D 目标框、OBB 旋转角度及跟踪轨迹。
> * **3D 融合模式**：并行开启双目 SGBM 深度图解算，由 `detect_3d_node` 将 2D 结果与深度图汇聚融合成三维点位。

---

## 2. 系统整体架构与数据流

本项目的感知架构采用**双分支并行解耦设计**，开发者可自由选择只跑 2D 还是跑 2D+3D：

```text
                           ┌────────────────────────┐
                           │ teleimager 图传相机节点 │
                           │ (480×1280 双目拼接画面) │
                           └───────────┬────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                │ ZMQ PUB (tcp://192.168.123.164:55555)       │
                ▼                                             ▼
┌──────────────────────────────┐              ┌──────────────────────────────┐
│  【分支一：纯 2D 检测主干】   │              │  【分支二：双目深度旁路分支】 │
│  (核心主干，独立运行，无深度依赖)│              │  (可选开启，为 3D 空间测距服务)│
└──────────────┬───────────────┘              └──────────────┬───────────────┘
               │                                             │
               │ ① 裁切左半 480×640 (或订阅 /left/image_raw)   │ stereo_split_node (双目拆分)
               ▼                                             ▼
┌──────────────────────────────┐              ┌──────────────────────────────┐
│     yolo_node (核心检测)      │              │  /left/image_raw + /right    │
│ - 目标识别 (cloth/bin/arm)   │              └──────────────┬───────────────┘
│ - OBB 旋转矩形框 & 偏转角解算  │                             │
│ - 包含关系重叠过滤            │                             ▼
└──────────────┬───────────────┘              ┌──────────────────────────────┐
               │                              │  sgbm_stereo_node (SGBM视差)  │
               │ /yolo/detections             └──────────────┬───────────────┘
               ├──────────────────────┐                      │ /stereo/disparity
               ▼                      │                      ▼
┌──────────────────────────────┐      │       ┌──────────────────────────────┐
│  tracking_node (目标跟踪)     │      │       │  disparity_to_depth_node     │
│  (ByteTrack 跨帧 ID 维护)     │      │       │  (输出 /left/depth 绝对深度)  │
└──────────────┬───────────────┘      │       └──────────────┬───────────────┘
               │ /yolo/tracking       │                      │
               ▼                      │                      │
       【纯 2D 检测与跟踪输出】        │                      │
     ★ 可直接驱动机械臂做平面决策 ★     │                      │
               │                      │                      │
               └──────────────┬───────┴──────────────────────┘
                              │ 
                              │ 【可选汇聚节点】：当配置 YOLO_USE_3D=True 时启动
                              ▼
               ┌──────────────────────────────┐
               │       detect_3d_node         │ ◀─── TF: left_camera -> base_link
               │ (2D 检测框 + 深度图 融合解算) │
               └──────────────┬───────────────┘
                              │ /yolo/detections_3d
                              ▼
                      【3D 空间定位输出】
                    相对机器人本体 base_link
```

---

## 3. 工作空间目录结构解析

```text
YOLO_ws/
├── config/
│   └── run_config.sh          # 统一配置中心：修改模型路径、推理阈值、图像尺寸等，无需重编译
├── image_depth/               # 双目立体视觉与深度解算子系统
│   ├── calibrate_stereo.py    # 离线双目相机标定脚本
│   ├── calibration_result.yaml# 标定得到的内参矩阵与畸变参数
│   ├── stereo_split_node.py   # 将 ZMQ 双目图拆分为 /left 和 /right 话题
│   ├── sgbm_stereo_node.py    # SGBM 立体匹配生成视差图
│   ├── disparity_to_depth_node.py # 将视差转为 metric 深度图
│   ├── run_stereo_yolo.sh     # 一键启动双目深度 + YOLO 全流程脚本
│   └── view_g1_urdf.sh        # RViz2 可视化机器人模型与点云/检测框
├── models/                    # YOLO 预训练及微调权重模型
│   ├── bin_and_cloth-OBB.pt   # 收纳筐与衣物旋转框（OBB）检测模型（主力模型）
│   ├── best-OBB.pt            # 高精度备用 OBB 模型
│   └── Arm_holding.pt         # 机械臂抓持物体状态检测模型
├── src/
│   └── yolo_ros-main/         # ROS2 核心节点源码
│       ├── yolo_ros/          # YOLO 节点、3D 检测、跟踪与 Debug 可视化节点
│       ├── yolo_bringup/      # Launch 启动包（支持 v5/v8/v10/v11/World 全系列）
│       └── yolo_msgs/         # 自定义 ROS2 消息与服务定义
├── echo_angle.py              # 实时打印并订阅 OBB 旋转检测角度的工具脚本
├── monitor_bbox.py            # 实时监测目标边界框面积占比的工具脚本
├── switch_model.sh            # 利用 LifecycleNode 生命周期状态机的不停机热换模型脚本
├── yolo_node_readme.md        # 详细技术规范与节点底层接口手册
└── readme 提示词.md            # 模型训练提示词与数据标注记录
```

---

## 4. 环境配置与依赖安装

### 4.1 系统要求
* **操作系统**：Ubuntu 22.04 LTS (aarch64 / x86_64)
* **ROS2 版本**：ROS2 Humble Hawksbill
* **Python 环境**：Python 3.10 (推荐使用 Conda 隔离管理)
* **加速计算**：CUDA 11.8+ / 12.x 及配套 TensorRT / cuDNN

### 4.2 环境准备与依赖安装
在 Conda 中创建并激活 `YOLO` 运行环境：
```bash
conda activate YOLO

# 安装核心依赖
pip install ultralytics==8.4.6
pip install pyzmq opencv-python pyyaml numpy lapx
```

### 4.3 编译工作空间
工程基于 ROS2 标准构建系统构建，推荐使用 `--symlink-install` 编译，使 Python 节点的修改即时生效：
```bash
cd /home/unitree/YOLO_ws

# 激活 ROS2 环境
source /opt/ros/humble/setup.bash

# 编译
colcon build --symlink-install

# 刷新当前环境
source install/setup.bash
```

---

## 5. 快速启动与运行指南

项目中可按具体任务需求，自由选择以 **纯 2D 极速模式** 运行，还是以 **2D+3D 空间融合模式** 运行：

### 5.1 模式 A：纯 2D 检测与跟踪模式（独立运行，无深度依赖）
如果你仅需要物体的 2D 像素框、类别置信度、OBB 旋转偏转角或 2D 目标跟踪，**完全无需启动双目深度分支**：
```bash
source install/setup.bash

# 仅启动 2D YOLO 检测与跟踪（关闭 3D 模块，节省显存算力，高帧率）
ros2 launch yolo_bringup yolo.launch.py \
    model:=/home/unitree/YOLO_ws/models/bin_and_cloth-OBB.pt \
    input_image_topic:=/left/image_raw \
    use_3d:=False
```

### 5.2 模式 B：2D + 双目深度 3D 空间定位模式（全链路融合）
如果机械臂需要物体的真实 3D 空间坐标（例如三维抓取空间测距），则启动整合了双目拆分、视差深度解算与 3D 投影的全套管线：
```bash
cd /home/unitree/YOLO_ws/image_depth

# 一键启动全链路（双目拆分 + SGBM 深度 + YOLO 2D/3D + Tracking）
./run_stereo_yolo.sh
```

### 5.3 启动角度提取工具
在机械臂执行对准抓取时，运行此脚本实时监听目标旋转偏转弧度：
```bash
python3 echo_angle.py
```

### 5.4 启动面积占比监测
监测目标是否贴近镜头或占据大部分视野：
```bash
python3 monitor_bbox.py
```

---

## 6. 模型热切换机制 (switch_model.sh)

在传统 ROS 节点中，更换模型往往需要杀死进程重新启动，这会导致 TF 树中断以及下游动作节点失效。

本项目中的 `yolo_node` 继承自 `rclpy.lifecycle.LifecycleNode`，模型仅在 **`activate`** 阶段装载。我们提供了 `switch_model.sh` 脚本，通过生命周期状态机在线无感换模型：

```bash
# 语法: ./switch_model.sh <模型文件路径> [模型类型: YOLO/World/YOLOE]

# 切换为机械臂手持检测模型
./switch_model.sh /home/unitree/YOLO_ws/models/Arm_holding.pt

# 切换为框与衣物旋转检测模型
./switch_model.sh /home/unitree/YOLO_ws/models/bin_and_cloth-OBB.pt
```

**转换流程**：`deactivate` (断开推理) $\rightarrow$ `cleanup` (释放旧模型显存) $\rightarrow$ `set_parameters` (写入新模型路径) $\rightarrow$ `configure` $\rightarrow$ `activate` (加载新模型并继续推流)。**进程全程不退出**。

---

## 7. 双目相机标定与深度解算 (image_depth)

针对 G1 前视相机的深度距离解算模块位于 `image_depth/`：
1. **标定板准备**：使用标准棋盘格（例如标定包中的 `checkerboard_9x7_25mm.png`）。
2. **标定求解**：运行 `python3 calibrate_stereo.py` 计算立体校准参数，结果将输出至 `calibration_result.yaml`。
3. **视差转换原理**：
   $$Z = \frac{f \times B}{d}$$
   其中 $f$ 为焦距，$B$ 为双目基线距离（Baseline），$d$ 为匹配视差值。

---

## 8. ROS2 话题与接口参考

### 8.1 核心话题 (Topics)

| 话题名称 | 消息类型 | 作用说明 |
| :--- | :--- | :--- |
| `/yolo/detections` | `yolo_msgs/msg/DetectionArray` | 包含类别、置信度、2D BoundingBox、OBB 旋转角及关键点 |
| `/yolo/detections_3d` | `yolo_msgs/msg/DetectionArray` | 结合双目深度后计算出的目标 3D 边界框（默认在 `base_link` 坐标系） |
| `/yolo/tracking` | `yolo_msgs/msg/DetectionArray` | 带有全局追踪 ID（`id` 字段）的目标检测框 |
| `/left/image_raw` | `sensor_msgs/msg/Image` | 裁切出的左目前视彩色图像 |
| `/left/depth` | `sensor_msgs/msg/Image` | SGBM 计算出的对齐深度图（单位：米） |
| `/yolo/debug_image` | `sensor_msgs/msg/Image` | 带有绘制框、标签、置信度的可视化图（仅在开启 debug 时发布） |

### 8.2 控制服务 (Services)

* **动态调整目标类别**：`/yolo/yolo_node/set_classes` (`yolo_msgs/srv/SetClasses`)  
  针对 YOLO-World 等开放词汇模型，支持运行中修改 prompt 类别（例如从 `"cloth"` 改为 `"bottle"`）。
* **包含过滤规则配置**：`/yolo/yolo_node/set_containment_rule` (`yolo_msgs/srv/SetContainmentRule`)  
  当检测到大框与小框重叠时（例如机械臂包含衣物），自动过滤外层或内层框。

---

## 9. 配置中心参数说明 (run_config.sh)

位于 `config/run_config.sh`，可修改以下关键超参数，保存后启动脚本即刻生效：

```bash
YOLO_MODEL="/home/unitree/YOLO_ws/models/bin_and_cloth-OBB.pt"   # 启动时装载的默认模型
YOLO_THRESHOLD="0.5"             # 置信度过滤阈值 (0.0~1.0)
YOLO_IOU="0.7"                   # NMS 非极大值抑制 IoU 阈值
YOLO_HALF="False"                # 是否启用 FP16 半精度推理（Jetson 上开启可提速）
YOLO_DEVICE="cuda:0"             # 推理硬件: cuda:0 或 cpu
YOLO_USE_TRACKING="True"         # 是否启用多目标跟踪
YOLO_USE_3D="True"               # 是否开启 3D 空间坐标投影
YOLO_TARGET_FRAME="base_link"    # 3D 目标坐标转换基准系
```

---

## 10. 常见问题排查 (FAQ)

### Q1: 启动后报错提示连接不上 ZMQ 图传源？
* **原因**：机器人的图传发布服务（`teleimager`）未开启，或者图传 IP 端口配置不匹配。
* **解决**：检查 `ps aux | grep teleimager`，确认图传推流端口为 `tcp://192.168.123.164:55555`。

### Q2: 3D 检测话题 `/yolo/detections_3d` 没有输出？
* **原因**：3D 检测依赖 TF 树变换和深度图像。
* **解决**：
  1. 确认已启动 `run_stereo_yolo.sh`（深度链路在运行）；
  2. 检查当前机器人的 TF 树中是否存在从 `left_camera_optical_frame` 到 `base_link` 的变换（`ros2 run tf2_tools view_frames`）。

### Q3: 运行 `switch_model.sh` 提示服务调用超时？
* **原因**：CycloneDDS 本地回环发现存在多网卡干扰，或者 `yolo_node` 崩溃。
* **解决**：检查节点是否存活（`ros2 node list`），确认 `CYCLONEDDS_URI` 绑定的网卡正确。

---

## 11. 附录与致谢
* 核心底层框架基于 [mgonzs13/yolo_ros](https://github.com/mgonzs13/yolo_ros) v4.6.0 定制开发。
* 包含了专门针对宇树双目相机裁切、ByteTrack 跨帧 ID 维护、SGBM 视差结合以及 OBB 角度解算的定制模块。
