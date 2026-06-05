# 中国石油大学（北京）SPR战队26赛季自瞄算法

本项目建立在[sp_vision_25](https://github.com/TongjiSuperPower/sp_vision_25)、[SPR-Vision-2026](https://github.com/SPR-Algorithm/SPR-Vision-2026)项目的基础之上。参考了sp25的整体框架同时修改了，将打符模块进行了重构。

## 1 详细信息
### 1.1 项目环境
操作系统：Ubuntu 22.04\
运算平台：NUC12WSKI7（i7-1260P，16GB）\
相机型号：海康MV-CS016-10UC\
镜头型号：海康官方8mm镜头\
下位机型号：RoboMaster开发板C型（STM32F407）\
IMU型号：使用C板内置BMI088作为IMU\
通信方式：USB2CAN（旧）、MicroUSB虚拟串口（新）\
辅助工具：NoMachine（远程桌面）、PlotJuggler（绘制曲线图）

### 1.2 编译方式
1. 安装依赖项：
   - [MindVision SDK](https://mindvision.com.cn/category/software/sdk-installation-package/)或[HikRobot SDK](https://www.hikrobotics.com/cn2/source/support/software/MVS_STD_GML_V2.1.2_231116.zip)
   - [OpenVINO](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-linux.html)
   - [Ceres](http://ceres-solver.org/installation.html)
   - 其余：
    ```bash
    sudo apt install -y \
        git \
        g++ \
        cmake \
        can-utils \
        libopencv-dev \
        libfmt-dev \
        libeigen3-dev \
        libspdlog-dev \
        libyaml-cpp-dev \
        libusb-1.0-0-dev \
        nlohmann-json3-dev \
        openssh-server \
        screen
    ```

2. 编译：
    ```bash
    cmake -B build
    make -C build/ -j`nproc`
    ```

3. 运行demo:
    ```bash
    ./build/auto_aim_test
    ```

4. 注册自启：
    1. 确保已安装`screen`:
        ```
        sudo apt install screen
        ```
    2. 创建`.desktop`文件:
        ```
        mkdir ~/.config/autostart/
        touch ~/.config/autostart/sp_vision.desktop
        ```
    3. 在该文件中写入:
        ```
        [Desktop Entry]
        Type=Application
        Exec=/home/spr/Desktop/spr_vision_26/autostart.sh
        Name=spr_vision
        ```
        注: [Exec](https://specifications.freedesktop.org/desktop-entry-spec/desktop-entry-spec-latest.html)必须为绝对路径.
    4. 确保`autostart.sh`有可执行权限:
        ```
        chmod +x autostart.sh
        ```

5. USB2CAN设置（可选）
    1. 创建`.rules`文件:
        ```
        sudo touch /etc/udev/rules.d/99-can-up.rules
        ```
    2. 在该文件中写入:
        ```
        ACTION=="add", KERNEL=="can0", RUN+="/sbin/ip link set can0 up type can bitrate 1000000"
        ACTION=="add", KERNEL=="can1", RUN+="/sbin/ip link set can1 up type can bitrate 1000000"

6. 使用GPU推理（可选）
    ```
    mkdir neo  
    cd neo  

    wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.13463.18/intel-igc-core_1.0.13463.18_amd64.deb  
    wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.13463.18/intel-igc-opencl_1.0.13463.18_amd64.deb  
    wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/intel-level-zero-gpu-dbgsym_1.3.25812.14_amd64.ddeb  
    wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/intel-level-zero-gpu_1.3.25812.14_amd64.deb  
    wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/intel-opencl-icd-dbgsym_23.09.25812.14_amd64.ddeb  
    wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/intel-opencl-icd_23.09.25812.14_amd64.deb  
    wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/libigdgmm12_22.3.0_amd64.deb  
    wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/ww09.sum  

    sha256sum -c ww09.sum  
    sudo dpkg -i *.deb  
    ```
    注：如果使用 GPU 异步推理（async-infer），最高显示分辨率限制为 1920×1080 (24Hz)

7. 串口设置
    1. 授予权限
        ```
        sudo usermod -a -G dialout $USER
        ```
    2. 获取端口 ID（serial, idVendor, idProduct）
        ```
        udevadm info -a -n /dev/ttyACM0 | grep -E '({serial}|{idVendor}|{idProduct})'
        ```
        将 /dev/ttyACM0 替换为实际设备名。
    3. 创建 udev 规则文件
        ```
        sudo touch /etc/udev/rules.d/99-usb-serial.rules
        ```
        然后在文件中写入如下内容（用真实 ID 替换示例，SYMLINK 是规则应用后固定的串口名）：
        ```
        SUBSYSTEM=="tty", ATTRS{idVendor}=="1d6b", ATTRS{idProduct}=="0002", ATTRS{serial}=="0000:00:14.0", SYMLINK+="gimbal"
        ```

    4. 重新加载 udev 规则
        ```
        sudo udevadm control --reload-rules
        sudo udevadm trigger
        ```
    5. 检查结果
        ```
        ls -l /dev/gimbal
        # Expected output (example):
        # lrwxrwxrwx 1 root root 7 Jul 21 10:00 /dev/gimbal -> ttyACM0
        ```

### 1.3 数据流图
视觉相关模块如图3.1所示。其中，相机线程产生图像、时间戳，通过下位机线程获取对应的云台姿态四元数；图像经过识别器，获得装甲板的四个顶点像素坐标，以及其图案类别；估计器根据装甲板信息，获得目标单位的运动状态；决策器则根据当前的目标运动状态信息，预测目标的运动轨迹，从而判断最佳瞄准位置和最佳开火时机，形成指令发送给下位机；最后控制器和执行机构则根据该指令进行执行，从而完成一个完整的自瞄流程。
![数据流图](https://github.com/user-attachments/assets/b89ce42f-a769-49c5-b82a-d69aeac02925)
图3.1 数据流图

### 1.4 软件架构
各兵种需要实现的功能往往有多个，但是每个功能不能作为独立的程序。例如步兵需要自瞄和打符，显然，这两个功能都需要从相机获取图像，但是相机只能被一个程序打开，这会导致另一个程序无法正常工作（相机被占用）。

为了解决这个问题，我们提出了视觉框架（spr_vision）：自瞄、打符等功能，被拆解为该框架下的一个个功能组，每个功能组包含多个类或函数（如识别器、解算器、预测器等等）；而运行的程序（main函数）只有一个，它负责从相机获取图像，再根据电控发来的信号（自瞄档or打符档），选择对应的功能组执行，不同兵种的执行逻辑各不相同，即不同的main函数。此外，为了方便大家合作开发，视觉框架还提供了许多常用的工具函数，减少因重复造轮子所导致的出错概率和时间成本。本框架的组成如图3.2所示。
![软件架构](https://github.com/user-attachments/assets/2603a3b3-ae1d-4fa8-afbb-490efde30d77)
图3.2 软件架构

### 1.5 文件结构
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
├── io                              // 硬件抽象层，见3.4软件架构
│   └── ...                 
├── LICENSE
├── mpc_layout.xml                  // MPC调试布局
├── src                             // 应用层（main函数），见3.4软件架构
│   ├── standard.cpp / standard_mpc.cpp          // 步兵
│   ├── mt_standard.cpp / mt_auto_aim_debug.cpp  // 多线程步兵
│   ├── sentry.cpp / sentry_bp.cpp / sentry_debug.cpp // 哨兵
│   ├── sentry_multithread.cpp / ...             // 哨兵多线程
│   ├── auto_aim_debug_mpc.cpp                   // 自瞄调试
│   └── auto_buff_debug.cpp / auto_buff_debug_mpc.cpp // 打符调试
├── tasks                           // 功能层，见3.4软件架构
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
└── tools                           // 工具层，见3.4软件架构
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

## 2 调试工具

本框架提供了一套完整的调试工具链，位于 `debug/` 目录下，包括 **Web 可视化调试器**、**Debug 事件总线**、**动态参数调节器** 三个组件。支持运行时实时查看检测结果、调节 EKF 参数，无需重新编译。

### 2.1 WebDebugger — 浏览器可视化调试

通过 HTTP + WebSocket 在浏览器中实时展示每帧检测结果。

**启用方式**：在 main 函数中创建 `WebDebugger` 实例并启动：

```cpp
#include "debug/web_debugger.hpp"

debug::WebDebugger debugger(8080);
debugger.start();
// 浏览器访问 http://<机器人IP>:8080
```

**每帧推送检测数据**：

```cpp
void push_debug(debug::WebDebugger & dbg,
                const cv::Mat & frame,
                const std::list<auto_aim::Armor> & armors,
                const std::vector<std::vector<cv::Point2f>> & reprojected_pts,
                double latency_ms)
{
  std::vector<debug::DetectionData> dets;
  for (const auto & armor : armors) {
    debug::DetectionData d;
    d.pts    = armor.points;          // 装甲板四个顶点
    d.color  = static_cast<int>(armor.color);   // 0=蓝 1=红
    d.number = static_cast<int>(armor.name);    // 0=guard 1~5=步兵 6=前哨站...
    d.conf   = armor.confidence;
    dets.push_back(d);
  }

  std::vector<debug::ReprojectionData> reproj;
  for (const auto & pts : reprojected_pts) {
    reproj.push_back({pts});
  }

  dbg.push(frame, dets, reproj, latency_ms);
}
```

#### 2.1.1 Web 界面功能
- 显示相机实时画面，叠加检测框与重投影点
- 右侧面板展示帧率、延迟、检测数量等统计信息
- 可折叠卡片展示每条检测目标的颜色、编号、置信度

### 2.2 DebugBus — 调试事件总线

`DebugBus` 是单例模式的事件总线，将主循环中的调试数据分发给多个注册的 `IDebugSink`。支持通过 YAML 配置动态添加输出后端。

**配置示例**（在 `configs/*.yaml` 中）：

```yaml
debug_bus:
  sinks:
    web:
      enabled: true
      port: 8080
      bind: "0.0.0.0"
```

**代码中启用**：

```cpp
#include "debug/debug_bus.hpp"

auto & bus = debug::DebugBus::instance();
bus.load_config(config);              // 从 YAML 读取配置
// 或手动注册 sink
// bus.add_sink(std::make_unique<WebSink>(8080));

// 每帧推送
debug::FrameDebugData data;
data.frame      = frame;
data.armors     = &armors;
data.reprojections = reprojs;
data.latency_ms = latency;
bus.post(data);
```

### 2.3 ParamTuner — 动态参数调节

`ParamTuner` 管理多组 EKF 滤波器参数，支持运行时通过 WebSocket 动态切换和调节参数，无需重新编译。

**配置示例**（在 YAML 中定义多组参数集）：

```yaml
ekf_param_sets:
  - name: "slow"
    v1: 0.1; v2: 0.05
    R_yaw: 0.05; R_pitch: 0.05; R_distance: 0.1
    angular_velocity_threshold: 1.0     # |w| < 1.0 rad/s 时使用
  - name: "normal"
    v1: 1.0; v2: 0.5
    R_yaw: 0.1; R_pitch: 0.1; R_distance: 0.5
    angular_velocity_threshold: 3.0
  - name: "fast"
    v1: 5.0; v2: 2.0
    R_yaw: 0.2; R_pitch: 0.2; R_distance: 1.0
    angular_velocity_threshold: inf     # 高角速度时使用
```

系统根据当前目标角速度自动选择合适的参数集，也可通过 Web Debugger 界面手动调节。

### 2.4 PlotJuggler 使用

PlotJuggler 是官方推荐的实时曲线绘制工具，用于可视化 EKF 状态、预测轨迹、云台响应等数据。本项目通过 `tools::Plotter` 工具类以 UDP 协议向 PlotJuggler 发送 JSON 格式数据。

#### 2.4.1 安装 PlotJuggler

```bash
sudo apt install plotjuggler
# 或从源码编译：https://github.com/facontidavide/PlotJuggler
```

#### 2.4.2 启用方式

`Plotter` 默认向 `127.0.0.1:9870` 发送 UDP 数据，用法如下：

```cpp
#include "tools/plotter.hpp"

tools::Plotter plotter;   // 默认 127.0.0.1:9870
// tools::Plotter plotter("192.168.1.100", 9870);  // 远程地址

// 构造 JSON 数据并发送
nlohmann::json data;
data["timestamp"] = current_time;
data["yaw"]       = target_yaw;
data["pitch"]     = target_pitch;
data["distance"]  = target_distance;
data["vx"]        = target_vx;
data["vy"]        = target_vy;

plotter.plot(data);
```

已在以下兵种 main 函数中默认启用 Plotter：
- `standard.cpp` / `standard_mpc.cpp`
- `mt_standard.cpp` / `mt_auto_aim_debug.cpp`
- `sentry.cpp` / `sentry_multithread.cpp`
- `uav.cpp` / `uav_debug.cpp`

#### 2.4.3 PlotJuggler 使用步骤

1. **启动 PlotJuggler**：
   ```bash
   plotjuggler
   ```

2. **配置数据源** (Streaming)：
   - 点击菜单栏 **Streaming → Start UDP Server**
   - 端口填写 `9870`（与代码中一致）

3. **拖拽曲线**：
   - 左侧数据树中将显示收到的 JSON 字段（如 `yaw`、`pitch`、`distance`）
   - 将字段拖入右侧绘图区即可显示实时曲线
   - 按住 `Ctrl` 可多选同时绘制

4. **保存/加载布局**：
   - 配置好曲线布局后，保存为 `.xml` 文件（如项目中的 `mpc_layout.xml`、`buff_layout.xml`）
   - 下次启动时通过 **File → Load Layout** 快速恢复

5. **常见用途**：
   - 观测 EKF 收敛过程（位置/速度/角度估计值）
   - 评估弹道解算效果（pitch 角度随时间变化）
   - 对比预测轨迹与实际目标运动
   - 监控打符时的能量机关旋转速度预测

## 3 配置文件详解

`configs/` 目录下的 YAML 文件是项目唯一的配置入口，每个兵种/用途对应一个文件（如 `standard4.yaml`、`demo.yaml`）。各字段说明如下：

| 配置段 | 关键字段 | 说明 |
|--------|----------|------|
| 顶层 | `enemy_color` | 敌方颜色：`red` / `blue` |
| **神经网络** | `yolo_name` | 使用的模型：`yolov5` / `yolov8` / `yolo11` |
| | `classify_model` | 数字分类模型路径（如 `tiny_resnet.onnx`） |
| | `yolo*_model_path` | 各版本 YOLO 权重路径 |
| | `device` | 推理设备：`CPU` / `GPU` |
| | `min_confidence` | 检测最小置信度阈值 |
| | `use_traditional` | 是否启用传统视觉方法作为 fallback |
| **ROI** | `x, y, width, height` | 检测区域（感兴趣区域），减小搜索范围提升帧率 |
| **USB 相机** | `image_width/height` | 分辨率，默认 1920×1080 |
| | `fov_h / fov_v` | 水平/垂直视场角（度） |
| | `usb_exposure` | 曝光时间（1-80000），日光 250，夜间可调大 |
| | `usb_gamma` / `usb_gain` | 伽马值 / 增益 |
| **工业相机** | `camera_name` | `mindvision` / `hikrobot` |
| | `exposure_ms` | 曝光时间（毫秒） |
| | `vid_pid` | USB 设备 VID:PID |
| **传统视觉** | `threshold` 等 | 灯条/装甲板提取参数（二值化阈值、长宽比约束等） |
| **CAN 通信** | `can_interface` | CAN 接口名：`can0` / `can1` |
| | `*_canid` | 各数据项的 CAN ID |
| **Tracker** | `min_detect_count` | 确认目标所需最少检测帧数 |
| | `max_temp_lost_count` | 目标丢失后保留跟踪的最大帧数 |
| **Aimer** | `yaw_offset` / `pitch_offset` | 弹道补偿角度（度） |
| | `comming_angle` / `leaving_angle` | 接敌/离敌角度阈值 |
| **Shooter** | `auto_fire` | 是否启用自瞄自动射击 |
| | `first_tolerance` / `second_tolerance` | 近/远距离射击容差 |
| **标定参数** | `camera_matrix` / `distort_coeffs` | 相机内参和畸变系数 |
| | `R_camera2gimbal` / `t_camera2gimbal` | 相机到云台的旋转/平移矩阵 |
| | `R_gimbal2imubody` | 云台到 IMU 机体的旋转矩阵 |

**提示**：修改配置文件后无需重新编译，程序会在启动时读取。

## 4 核心模块详解

### 4.1 auto_aim — 自瞄模块

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

### 4.2 auto_buff — 打符模块

位于 `tasks/auto_buff/`，用于能量机关（大符/小符）的识别与打击。

| 组件 | 功能 |
|------|------|
| `BuffDetector` | 识别能量机关扇叶/旋转中心，支持 YOLO11 专用模型 |
| `BuffSolver` | 解算能量机关在三维空间中的位置与姿态 |
| `BuffAimer` | 预测扇叶旋转轨迹，计算提前量 |
| `BuffPredict` | 基于 RANSAC 正弦拟合的旋转速度预测 |

**特殊之处**：能量机关是旋转的，因此需要预测其旋转速度（角速度），并计算合适的提前瞄准点。预测器基于历史角速度数据使用 RANSAC 拟合正弦曲线来预测未来位置。

### 4.3 omniperception — 全向感知（哨兵专用）

位于 `tasks/omniperception/`，用于哨兵机器人的多方向感知与决策。

| 组件 | 功能 |
|------|------|
| `Perceptron` | 多方向感知融合，处理多个相机的检测结果 |
| `Decider` | 决策逻辑：根据当前目标分布选择最优攻击目标 |
| `Detection` | 统一的检测结果数据结构 |

### 4.4 IO — 硬件抽象层

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

### 4.5 tools — 工具层

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

## 5 开发指南

### 5.1 如何添加新兵种

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

### 5.2 如何添加新功能组

框架设计支持将新功能拆解为独立模块加入 `tasks/`：

1. 在 `tasks/` 下新建目录（如 `tasks/my_feature/`）
2. 实现功能类，遵循 `namespace my_feature` 命名空间
3. 在该目录下创建 `CMakeLists.txt`，编译为静态库
4. 在顶层 `CMakeLists.txt` 中添加 `add_subdirectory(tasks/my_feature)`
5. 在 main 函数中根据电控模式信号选择执行对应功能组

### 5.3 代码规范

- **语言标准**：C++17
- **命名规范**：
  - 类名/枚举：PascalCase（如 `ArmorType`、`Detector`）
  - 变量/函数：snake_case（如 `min_confidence`、`detect()`）
  - 常量/枚举值：snake_case（如 `red`、`big`）
  - 文件：snake_case（如 `auto_aim`、`extended_kalman_filter`）
- **命名空间**：每个模块使用独立 namespace（`auto_aim`、`auto_buff`、`debug`、`tools`）
- **头文件**：使用 `#ifndef` 宏防止重复包含
- **注释**：关键算法逻辑、配置文件字段含义必须加注释

### 5.4 模型训练与部署

1. **训练**：使用 RoboFlow / YOLO 训练流程，导出为 OpenVINO 格式（.xml + .bin）
2. **数字分类**：训练 `tiny_resnet.onnx` 用于装甲板数字识别（0-8 共 9 类）
3. **部署**：将模型文件放入 `assets/` 目录，更新配置文件的 `yolo*_model_path` 字段
4. **GPU 推理**：在 Intel NUC 上使用 GPU 推理需安装 Intel GPU 驱动（见 1.2 节第 6 步）

### 5.5 多线程架构

部分兵种（`mt_standard`、`sentry_multithread`）使用了多线程架构：

- **相机线程**：仅负责采集图像和 IMU 数据，放入线程安全队列
- **检测线程**：从队列取图像运行 YOLO 推理（耗时最长，独立线程避免阻塞主循环）
- **主线程**：处理解算、跟踪、决策和通信

如需使用多线程，参考 `multithread/` 目录下的 `mt_detector` 实现。

## 6 标定流程

所有标定程序位于 `calibration/`，编译后生成独立可执行文件。

### 6.1 相机内参标定

```bash
# 1. 采集标定板图像（打印棋盘格，从不同角度拍摄）
./build/capture assets/img_with_q configs/calibration.yaml

# 2. 标定相机内参
./build/calibrate_camera assets/img_with_q configs/calibration.yaml
```

标定结果会输出相机矩阵和畸变系数，更新到兵种配置文件的 `camera_matrix` 和 `distort_coeffs` 字段。

### 6.2 手眼标定

手眼标定用于确定相机坐标系到云台坐标系的外参（旋转矩阵 R + 平移向量 t）。

```bash
# 1. 采集带云台姿态的标定数据
./build/capture assets/img_with_q configs/calibration.yaml

# 2. 标准手眼标定
./build/calibrate_handeye assets/img_with_q configs/calibration.yaml

# 3. 含标定板世界坐标的手眼标定（精度更高）
./build/calibrate_robotworld_handeye assets/img_with_q configs/calibration.yaml
```

标定结果写入配置文件的 `R_camera2gimbal` 和 `t_camera2gimbal` 字段。

### 6.3 数据采集辅助

```bash
# 将长视频按帧切分为图片
./build/split_video <video_path> <output_folder>
```

### 6.4 标定配置文件

标定参数在 `configs/calibration.yaml` 中配置：

```yaml
pattern_size: [9, 6]          # 棋盘格内角点数量 (宽, 高)
center_distance: 0.025        # 棋盘格格子边长（米）
```

## 7 Docker 使用

项目提供 Docker 支持，用于 CI/CD 和快速搭建编译环境。

### 7.1 编译

```bash
# 仅编译（用于验证代码可编译）
docker compose up spr-vision-build

# 编译并运行 demo
docker compose up spr-vision
```

### 7.2 手动构建

```bash
# 指定平台构建（Intel NUC 使用 linux/amd64）
docker build --platform linux/amd64 -t spr-vision:latest .

# Apple Silicon Mac 上会自动使用 QEMU 模拟，编译速度较慢
```

### 7.3 注意事项

- Docker 镜像仅包含编译环境和 CPU 推理支持
- 如需 GPU 推理，需在宿主机上原生编译（Intel NUC）
- Apple Silicon Mac 构建的镜像无法在 x86_64 机器上运行

## 8 常见问题 (FAQ)

| 问题 | 原因 | 解决 |
|------|------|------|
| 编译时报 `OpenVINO_DIR` 未设置 | OpenVINO 未安装或路径不正确 | `export OpenVINO_DIR=/opt/intel/openvino_2024.6.0/runtime/cmake` |
| `sudo: dpkg: 未找到命令` | Docker 内缺少 dpkg | 使用 Docker 编译而非直接运行 shell 命令 |
| 相机打开失败 | 权限不足 / USB 端口被占用 | `sudo usermod -a -G dialout $USER`，重新登录；检查 `ls /dev/video*` |
| `can0: 未找到设备` | CAN 适配器未连接或驱动未加载 | 检查 USB2CAN 是否插入；`sudo modprobe can`；查看 `dmesg` |
| 自瞄不准 | 标定参数不准确 / 弹道补偿未调好 | 重新标定相机 → 手眼标定 → 调整 `yaw_offset` / `pitch_offset` |
| 帧率低 | 推理耗时过长 / 曝光时间太长 | 启用 GPU 推理；缩小 ROI；调低分辨率；使用多线程架构 |
| 程序启动后闪退 | 配置文件路径错误或模型文件缺失 | 确认 `assets/` 下有对应的 `.xml/.onnx` 文件；检查配置文件路径 |
| 串口 `gimbal` 设备不存在 | udev 规则未生效 | 重新执行 `sudo udevadm control --reload-rules && sudo udevadm trigger`；确认 VID/PID/序列号正确 |

## 项目成员

**SPR 战队 2026 赛季视觉组**
- 唐京
- 李家乐

**特别感谢**
- [TongjiSuperPower/sp_vision_25](https://github.com/TongjiSuperPower/sp_vision_25) — 提供了优秀的视觉框架参考
- [SPR-Algorithm/SPR-Vision-2026](https://github.com/SPR-Algorithm/SPR-Vision-2026) — 赛季前期探索与积累
- Alan Day. 【RM2024赛季-识别模型】深圳大学-RobotPilots[EB/OL]. RoboMaster论坛. https://bbs.robomaster.com/article/54091, 2025.
- 陈君. rm_vision[EB/OL]. GitHub. https://github.com/chenjunnn/rm_vision, 2023.