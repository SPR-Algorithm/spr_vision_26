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

### 3.5 文件结构
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

### 3.6 调试工具 (debug)

本框架提供了一套完整的调试工具链，位于 `debug/` 目录下，包括 **Web 可视化调试器**、**Debug 事件总线**、**动态参数调节器** 三个组件。支持运行时实时查看检测结果、调节 EKF 参数，无需重新编译。

#### 3.6.1 WebDebugger — 浏览器可视化调试

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

Web 界面功能：
- 显示相机实时画面，叠加检测框与重投影点
- 右侧面板展示帧率、延迟、检测数量等统计信息
- 可折叠卡片展示每条检测目标的颜色、编号、置信度

#### 3.6.2 DebugBus — 调试事件总线

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

#### 3.6.3 ParamTuner — 动态参数调节

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

### 3.7 PlotJuggler 使用

PlotJuggler 是官方推荐的实时曲线绘制工具，用于可视化 EKF 状态、预测轨迹、云台响应等数据。本项目通过 `tools::Plotter` 工具类以 UDP 协议向 PlotJuggler 发送 JSON 格式数据。

#### 3.7.1 安装 PlotJuggler

```bash
sudo apt install plotjuggler
# 或从源码编译：https://github.com/facontidavide/PlotJuggler
```

#### 3.7.2 启用方式

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

#### 3.7.3 PlotJuggler 使用步骤

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

## 项目成员
唐京 李家乐