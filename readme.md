# 中国石油大学（北京）SPR战队26赛季自瞄算法

本项目建立在[sp_vision_25](https://github.com/TongjiSuperPower/sp_vision_25)、[SPR-Vision-2026](https://github.com/SPR-Algorithm/SPR-Vision-2026)项目的基础之上。参考了sp25的整体框架同时修改了，将打符模块进行了重构。

> 环境配置、YAML 字段说明、标定流程等请参阅 [GitHub Wiki](https://github.com/SPR-Algorithm/spr_vision_26/wiki)；调试工具与常见问题请参阅 [调试工具](https://github.com/SPR-Algorithm/spr_vision_26/wiki/调试工具) 页面。

## 功能简介

### 数据流图

视觉相关模块如图1.1所示。其中，相机线程产生图像、时间戳，通过下位机线程获取对应的云台姿态四元数；图像经过识别器，获得装甲板的四个顶点像素坐标，以及其图案类别；估计器根据装甲板信息，获得目标单位的运动状态；决策器则根据当前的目标运动状态信息，预测目标的运动轨迹，从而判断最佳瞄准位置和最佳开火时机，形成指令发送给下位机；最后控制器和执行机构则根据该指令进行执行，从而完成一个完整的自瞄流程。

![数据流图](https://github.com/user-attachments/assets/b89ce42f-a769-49c5-b82a-d69aeac02925)

图1.1 数据流图

### 软件架构

各兵种需要实现的功能往往有多个，但是每个功能不能作为独立的程序。例如步兵需要自瞄和打符，显然，这两个功能都需要从相机获取图像，但是相机只能被一个程序打开，这会导致另一个程序无法正常工作（相机被占用）。

为了解决这个问题，我们提出了视觉框架（spr_vision）：自瞄、打符等功能，被拆解为该框架下的一个个功能组，每个功能组包含多个类或函数（如识别器、解算器、预测器等等）；而运行的程序（main函数）只有一个，它负责从相机获取图像，再根据电控发来的信号（自瞄档or打符档），选择对应的功能组执行，不同兵种的执行逻辑各不相同，即不同的main函数。此外，为了方便大家合作开发，视觉框架还提供了许多常用的工具函数，减少因重复造轮子所导致的出错概率和时间成本。本框架的组成如图1.2所示。

![软件架构](https://github.com/user-attachments/assets/2603a3b3-ae1d-4fa8-afbb-490efde30d77)

图1.2 软件架构

### 文件结构

```
spr_vision_26
├── assets                          // 包含demo素材、网络权重等
│   ├── *.onnx / *.xml / *.bin      // 模型权重文件
│   ├── standard_fanblade.jpg
│   └── demo/
├── autostart.sh                    // 自动启动脚本
├── buff_layout.xml                 // 打符窗口布局
├── calibration                     // 标定相关程序
│   ├── calibrate_camera.cpp        // 相机内参标定
│   ├── calibrate_handeye.cpp       // 手眼标定
│   ├── calibrate_robotworld_handeye.cpp // 手眼标定（含标定板位置）
│   ├── capture.cpp                 // 标定数据采集
│   └── split_video.cpp             // 视频切分
├── CMakeLists.txt                  // CMake配置
├── configs                         // 每台机器人的YAML配置文件
│   ├── *.yaml                      // 各兵种/功能配置
│   └── ...
├── debug                           // web调试工具
│   └── ...
├── docker-compose.yml              // Docker编排
├── Dockerfile                      // Docker镜像构建
├── io                              // 硬件抽象层
│   └── ...                 
├── LICENSE
├── mpc_layout.xml                  // MPC调试布局
├── src                             // 应用层（main函数）
│   ├── standard.cpp / standard_mpc.cpp          // 步兵
│   ├── mt_standard.cpp / mt_auto_aim_debug.cpp  // 多线程步兵
│   ├── sentry.cpp / sentry_bp.cpp / sentry_debug.cpp // 哨兵
│   ├── sentry_multithread.cpp / ...             // 哨兵多线程
│   ├── auto_aim_debug_mpc.cpp                   // 自瞄调试
│   └── auto_buff_debug.cpp / auto_buff_debug_mpc.cpp // 打符调试
├── tasks                           // 功能层
│   ├── auto_aim/                   // 自瞄算法
│   │   ├── aimer / armor / classifier / detector / solver / tracker ...
│   │   ├── multithread/            // 多线程模块
│   │   ├── planner/tinympc/        // MPC轨迹规划
│   │   └── yolos/                  // YOLO系列推理
│   ├── auto_buff/                  // 打符算法
│   │   ├── buff_aimer / buff_detector / buff_solver ...
│   │   └── ...
│   └── omniperception/            // 全向感知
│       ├── decider / perceptron
│       └── ...
├── tests
│   ├── auto_aim_test.cpp           // 自瞄视频测试
│   ├── auto_buff_test.cpp          // 打符视频测试
│   ├── camera_detect_test.cpp      // 识别器测试（工业相机）
│   ├── camera_test.cpp             // 相机测试
│   ├── cboard_test.cpp             // C板测试
│   ├── detector_video_test.cpp     // 识别器测试（视频）
│   ├── dm_test.cpp                 // 达妙IMU测试
│   ├── fire_test.cpp               // 开火测试
│   ├── gimbal_test.cpp             // 云台通信测试
│   ├── handeye_test.cpp            // 手眼标定测试
│   ├── minimum_vision_system.cpp   // 最小视觉系统
│   ├── multi_usbcamera_test.cpp    // 多USB相机测试
│   ├── planner_test.cpp / planner_test_offline.cpp // 规划器测试
│   ├── usbcamera_test.cpp / usbcamera_detect_test.cpp
│   └── ...                         // ROS测试等
└── tools                           // 工具层
    ├── crc.cpp / .hpp              // CRC校验
    ├── exiter.cpp / .hpp           // 退出检测
    ├── extended_kalman_filter.cpp / .hpp // 扩展卡尔曼滤波器
    ├── img_tools.cpp / .hpp        // 图像处理
    ├── logger.cpp / .hpp           // 日志记录
    ├── math_tools.cpp / .hpp       // 数学工具
    ├── pid.cpp / .hpp              // PID控制器
    ├── plotter.cpp / .hpp          // 曲线图绘制
    ├── ransac_sine_fitter.cpp / .hpp // RANSAC正弦拟合
    └── ...
```

### 核心模块

#### auto_aim — 自瞄模块

自瞄模块位于 `tasks/auto_aim/`，是整个系统的核心。各组件职责如下：

| 组件 | 文件 | 功能 |
|------|------|------|
| `Armor` | `armor.hpp` | 装甲板数据结构，定义颜色枚举（红/蓝/灭/紫）、类型（大/小）、编号（1-5/哨兵/前哨站/基地） |
| `Detector` | `detector.cpp` | 检测器：先尝试 YOLO 推理，若失败或置信度不足则回退到传统视觉方法 |
| `Classifier` | `classifier.cpp` | 数字分类器：使用 `tiny_resnet.onnx` 对装甲板 ROI 进行数字识别 |
| `Solver` | `solver.cpp` | 解算器：PnP 解算装甲板在三维空间中的位置，含重投影功能 |
| `Target` | `target.cpp` | 目标状态：封装目标的位置、速度、角速度等运动状态 |
| `Tracker` | `tracker.cpp` | 跟踪器：多目标跟踪，维持目标 ID 一致性，处理丢失与重识别 |
| `Aimer` | `aimer.cpp` | 决策器：选择最优目标，计算瞄准角度，含弹道补偿 |
| `Shooter` | `shooter.cpp` | 射击控制器：根据距离与容差判断开火时机 |
| `Voter` | `voter.cpp` | 投票器：多模型/多帧结果融合，提高检测稳定性 |
| `YOLO` | `yolo*.cpp` | YOLO 推理封装：支持 v5 / v8 / v11，CPU / GPU 推理 |
| `planner/tinympc/` | — | MPC 轨迹规划器：预测目标运动，计算最优瞄准点 |

**调用流程**（每帧）：
```
相机图像 → Detector.detect() → 得到 Armor 列表
  → Classifier 识别数字
  → Solver.solve() PnP 解算位置
  → Tracker 跟踪/匹配目标
  → Aimer 选择目标 + 弹道补偿
  → Shooter 判断开火 → 发送云台指令
```

#### auto_buff — 打符模块

位于 `tasks/auto_buff/`，用于能量机关（大符/小符）的识别与打击。

| 组件 | 功能 |
|------|------|
| `BuffDetector` | 识别能量机关扇叶/旋转中心，支持 YOLO11 专用模型 |
| `BuffSolver` | 解算能量机关在三维空间中的位置与姿态 |
| `BuffAimer` | 预测扇叶旋转轨迹，计算提前量 |
| `BuffPredict` | 基于 RANSAC 正弦拟合的旋转速度预测 |

**特殊之处**：能量机关是旋转的，因此需要预测其旋转速度（角速度），并计算合适的提前瞄准点。预测器基于历史角速度数据使用 RANSAC 拟合正弦曲线来预测未来位置。

#### omniperception — 全向感知（哨兵专用）

位于 `tasks/omniperception/`，用于哨兵机器人的多方向感知与决策。

| 组件 | 功能 |
|------|------|
| `Perceptron` | 多方向感知融合，处理多个相机的检测结果 |
| `Decider` | 决策逻辑：根据当前目标分布选择最优攻击目标 |
| `Detection` | 统一的检测结果数据结构 |

#### IO — 硬件抽象层

`io/` 目录封装了所有硬件接口，上层代码不直接操作硬件。

| 子目录/文件 | 功能 |
|------|------|
| `cboard.hpp / cboard.cpp` | 下位机（C 型开发板）通信：接收 IMU 四元数、弹速、模式切换信号；发送云台控制指令（yaw/pitch/shoot） |
| `socketcan.hpp` | SocketCAN 封装，提供 CAN 总线读写接口 |
| `camera.hpp / camera.cpp` | 相机基类，定义统一接口 |
| `hikrobot/` | 海康机器人工业相机驱动 |
| `mindvision/` | 迈德威视工业相机驱动 |
| `usbcamera/` | USB 免驱相机驱动 |
| `gimbal/` | 云台控制协议封装 |
| `dm_imu/` | 达妙 IMU 驱动 |
| `serial/` | 串口通信封装 |
| `ros2/` | ROS2 集成（可选），用于哨兵多机协同 |

**通信协议（CBoard → 视觉 → CBoard）**：
```
下位机 → CAN → 视觉:
  - 0x01: IMU 四元数 (w, x, y, z)
  - 0x101: 弹速
  - 模式指令: idle / auto_aim / small_buff / big_buff

视觉 → CAN → 下位机:
  - 0xFF: 云台控制 (yaw, pitch, shoot_flag)
```

#### tools — 工具层

`tools/` 提供通用工具函数，减少重复造轮子。

| 文件 | 功能 |
|------|------|
| `pid.cpp/hpp` | PID 控制器（位置式/增量式） |
| `extended_kalman_filter.cpp/hpp` | 扩展卡尔曼滤波器（EKF），用于目标状态估计 |
| `math_tools.cpp/hpp` | 数学工具：坐标系转换、角度归一化、插值等 |
| `img_tools.cpp/hpp` | 图像处理：绘制检测框、颜色转换、透视变换等 |
| `trajectory.cpp` | 弹道模型：计算子弹飞行时间与下落量 |
| `crc.cpp/hpp` | CRC 校验 |
| `logger.cpp/hpp` | 日志封装（基于 spdlog） |
| `exiter.cpp/hpp` | 优雅退出检测（按键检测） |
| `plotter.cpp/hpp` | PlotJuggler UDP 数据推送 |
| `recorder.cpp/hpp` | 视频/数据录制 |
| `ransac_sine_fitter.cpp/hpp` | RANSAC 正弦曲线拟合（用于打符预测） |
| `thread_pool.hpp` | 线程池 |
| `thread_safe_queue.hpp` | 线程安全队列 |

## 贡献指南

### Git 工作流

`main` 是稳定分支，**禁止直接在其上修改或推送**。所有改动须在本地功能分支完成，经 Pull Request 合并回 `main`。

#### 标准流程

```bash
# 1. 同步最新 main
git checkout main
git pull origin main

# 2. 从 main 创建功能分支（命名见下文）
git checkout -b feat/tracker-improve

# 3. 本地开发、自测、提交（提交信息见下文）
git add <files>
git commit -m "feat(tracker): 优化目标丢失后的重识别逻辑"

# 4. 推送到远程并发起 PR
git push -u origin feat/tracker-improve
# 在 GitHub 上创建 Pull Request，目标分支选择 main

# 5. 代码审查通过后合并；合并后删除已合入的分支
git checkout main
git pull origin main
git branch -d feat/tracker-improve
```

#### 分支命名

| 前缀 | 用途 | 示例 |
|------|------|------|
| `feat/` | 新功能 | `feat/buff-yolo11` |
| `fix/` | Bug 修复 | `fix/can-reconnect` |
| `refactor/` | 重构（不改变外部行为） | `refactor/io-serial` |
| `docs/` | 文档变更 | `docs/readme-wiki` |
| `test/` | 测试相关 | `test/planner-offline` |
| `chore/` | 构建、依赖、杂项 | `chore/cmake-openvino` |

#### 提交信息规范

采用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<type>(<scope>): <简短描述>

[可选正文：说明改动原因、影响范围]
```

**常用 type：**

| type | 含义 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `refactor` | 重构，不改变功能行为 |
| `docs` | 仅文档变更 |
| `style` | 代码格式（空格、缩进等），不影响逻辑 |
| `test` | 新增或修改测试 |
| `chore` | 构建脚本、依赖、CI 等杂项 |
| `perf` | 性能优化 |

**示例：**

```
feat(auto_aim): 支持 YOLO11 模型推理
fix(cboard): 修复串口断连后无法重连的问题
refactor(debug): 将 WebDebugger 接入 DebugBus
docs: 将调试说明迁移至 Wiki
chore(docker): 更新 OpenVINO 基础镜像版本
```

**要求：**
- 标题使用中文或英文均可，但需简洁明确
- 一次提交只做一件事，避免将无关改动混在同一 commit
- 合并到 `main` 前确保本地可编译通过

### 如何添加新兵种

1. 在 `configs/` 下创建对应 YAML 配置文件（参考 `example.yaml`）
2. 在 `src/` 下创建新的 main 函数（参考 `standard.cpp` 的结构）
3. 在顶层 `CMakeLists.txt` 中添加可执行目标并链接所需库
4. 如有特殊硬件需求，在 `io/` 中添加对应驱动

main 函数标准结构：
```cpp
int main(int argc, char *argv[]) {
  // 1. 解析命令行参数（配置文件路径）
  // 2. 读取 YAML 配置
  // 3. 初始化相机、CBoard 通信
  // 4. 创建 Detector / Solver / Tracker / Aimer / Shooter
  // 5. 主循环：采集图像 → 检测 → 解算 → 跟踪 → 决策 → 发送
  // 6. 退出清理
}
```

### 如何添加新功能组

框架设计支持将新功能拆解为独立模块加入 `tasks/`：

1. 在 `tasks/` 下新建目录（如 `tasks/my_feature/`）
2. 实现功能类，遵循 `namespace my_feature` 命名空间
3. 在该目录下创建 `CMakeLists.txt`，编译为静态库
4. 在顶层 `CMakeLists.txt` 中添加 `add_subdirectory(tasks/my_feature)`
5. 在 main 函数中根据电控模式信号选择执行对应功能组

### 代码规范

- **语言标准**：C++17
- **命名规范**：
  - 类名/枚举：PascalCase（如 `ArmorType`、`Detector`）
  - 变量/函数：snake_case（如 `min_confidence`、`detect()`）
  - 常量/枚举值：snake_case（如 `red`、`big`）
  - 文件：snake_case（如 `auto_aim`、`extended_kalman_filter`）
- **命名空间**：每个模块使用独立 namespace（`auto_aim`、`auto_buff`、`debug`、`tools`）
- **头文件**：使用 `#ifndef` 宏防止重复包含
- **注释**：关键算法逻辑、配置文件字段含义必须加注释

### 模型训练与部署

1. **训练**：使用 RoboFlow / YOLO 训练流程，导出为 OpenVINO 格式（.xml + .bin）
2. **数字分类**：训练 `tiny_resnet.onnx` 用于装甲板数字识别（0-8 共 9 类）
3. **部署**：将模型文件放入 `assets/` 目录，更新配置文件的 `yolo*_model_path` 字段
4. **GPU 推理**：在 Intel NUC 上使用 GPU 推理需安装 Intel GPU 驱动（详见 [Wiki 配置指南](https://github.com/SPR-Algorithm/spr_vision_26/wiki/配置指南)）

### 多线程架构

部分兵种（`mt_standard`、`sentry_multithread`）使用了多线程架构：

- **相机线程**：仅负责采集图像和 IMU 数据，放入线程安全队列
- **检测线程**：从队列取图像运行 YOLO 推理（耗时最长，独立线程避免阻塞主循环）
- **主线程**：处理解算、跟踪、决策和通信

如需使用多线程，参考 `multithread/` 目录下的 `mt_detector` 实现。

## 项目成员

**SPR 战队 2026 赛季视觉组**
- 唐京
- 李家乐
- 张容溪

**特别感谢**
- [TongjiSuperPower/sp_vision_25](https://github.com/TongjiSuperPower/sp_vision_25) — 提供了优秀的视觉框架参考
- [SPR-Algorithm/SPR-Vision-2026](https://github.com/SPR-Algorithm/SPR-Vision-2026) — 赛季前期探索与积累
- Alan Day. 【RM2024赛季-识别模型】深圳大学-RobotPilots[EB/OL]. RoboMaster论坛. https://bbs.robomaster.com/article/54091, 2025.
- 陈君. rm_vision[EB/OL]. GitHub. https://github.com/chenjunnn/rm_vision, 2023.
