# spr_vision_26 auto_buff 系统重构计划

> 本文档是本次重构的执行清单与阶段记录。实施过程中逐项勾选；每一阶段必须先通过本阶段验收，才能进入下一阶段。

## 当前状态

| 阶段 | 状态 | 进入条件 |
|---|---|---|
| 阶段 0：构建基线与统一入口 | 进行中 | 当前阶段 |
| 阶段 1：观测和目标数据模型 | 待开始 | 阶段 0 验收通过 |
| 阶段 2：锚点 PnP 与 Chamfer 位姿精化 | 阻塞 | 两段实测 JSONL v1 标注完成并通过校验 |
| 阶段 3：相位与运动估计 | 待开始 | 阶段 2 验收通过 |
| 阶段 4：决策、弹道与完整输出 | 待开始 | 阶段 3 验收通过 |
| 阶段 5：集成与回归 | 待开始 | 阶段 4 验收通过 |

阻塞说明：仓库中已有 `assets/test_video/buff.avi` 和 `buff_2.avi`，但尚未发现对应的实测 JSONL v1 标注。按本计划的硬约束，在标注完成并通过校验前只实施阶段 0–1。

## 总体方案

以 `BuffProcessor` 作为唯一有状态入口，形成：

```text
五点观测 → 观测精修 → 位姿解算 → 相位预测 → 决策/弹道 → VisionToGimbal
```

只迁移 `RP-26Rune` 的算法、模型点和参数思想，不引入其 TFTree、Foxglove、glog 等框架依赖。阶段 0 直接切换新接口，不长期维护新旧双轨。

## 公共接口与约束

新增 `BuffInput`，包含：

- `cv::Mat img`
- `std::chrono::steady_clock::time_point timestamp`
- 按 `[上, 右, 下, 左, R]` 排列的定长五点数组
- 激活类别和检测置信度
- IMU 四元数
- `io::GimbalState`
- `io::GimbalMode`

统一入口：

```cpp
class BuffProcessor {
public:
  BuffProcessor(
    const std::string &camera_config,
    const std::string &auto_buff_config);

  io::VisionToGimbal process(const BuffInput &input);
  void reset();
};
```

全局规则：

- 大小符以 `GimbalMode` 为权威；检测类别负责激活状态。两者冲突时拒绝该帧。
- `mode=0`：空闲、输入无效、解算失败或状态过期。
- `mode=1`：可以跟踪但禁止开火，包括 PnP 降级和预测置信度不足。
- `mode=2`：位姿、预测、弹道和开火时机全部可靠。
- mode 0 时 `yaw/yaw_vel/yaw_acc/pitch/pitch_vel/pitch_acc` 全部清零。
- 固定 `head={'S','P'}`、`tail=0xef`、`sizeof(io::VisionToGimbal)==28`。
- 模式切换、时间倒退和跟踪超时必须清空预测、目标切换及开火冷却状态。
- `Buff_Detector` 只负责“图像 → 标准五点观测”。
- 算法参数保存于独立的 `configs/auto_buff.yaml`。
- 保持现有 `R_gimbal2world` 坐标变换方向和相机—云台外参语义。
- Ceres 是构建必需依赖；精化失败时可回退到合格 PnP，但该帧最多输出 mode 1。
- 不覆盖或回滚 `configs/ascento.yaml`、`configs/demo.yaml`、`configs/sentry.yaml`、`configs/standard4.yaml`、`configs/uav.yaml` 的现有未提交改动。
- 不破坏 auto_aim、omniperception、主程序和 gimbal 协议。
- 可复用算法不得在 auto_buff 与 `tools/` 中重复实现。

## Task 1：阶段 0——构建基线与统一入口

- [ ] 为 auto_buff 相关目标定义 `OPENCV_DISABLE_EIGEN_TENSOR_SUPPORT`。
- [ ] 建立协议测试，先观察预期失败。
- [ ] 建立 `BuffInput`、`BuffProcessor` 和唯一的 `VisionToGimbal` 输出映射。
- [ ] 占位处理只做输入校验和 mode 0/1 映射，永不输出 mode 2。
- [ ] 切换 `auto_buff_debug`、`auto_buff_debug_mpc` 到 `processor.process(input)` 与 `gimbal.send(output)`。
- [ ] 删除 `Command/Plan` 作为 auto_buff 对外输出的用法。
- [ ] 验证 mode 0 六个运动字段清零、头尾及 28 字节布局。
- [ ] 构建 auto_buff 相关目标。

验收命令与结果：

```text
待记录
```

建议提交：`refactor(auto_buff): align processor and gimbal interface`

## Task 2：阶段 1——观测和目标数据模型

- [ ] 定义 `RuneKind`、`ActivationState`、`RuneState`、`TrackingQuality`。
- [ ] 定义 `RefinedObservation`、`RunePose`、`MotionState`、`RuneTarget`。
- [ ] 使用定长五点数组并校验点数、有限值、图像边界、顺序及几何退化。
- [ ] 实现 `ObservationRefiner`，生成 ROI、装甲板/R 标/灯臂轮廓和观测质量。
- [ ] 让 `Buff_Detector` 仅输出标准五点观测。
- [ ] 建立实测标注 JSONL v1 格式。
- [ ] 每帧记录视频、帧号、时间戳、五点、状态、IMU、gimbal、符心世界坐标、平面法向和连续相位。
- [ ] 增加标注校验器。
- [ ] 确定性测试覆盖四种 RuneState、模式冲突、非法点序、非有限/越界/退化点、时间异常、跟踪超时和目标切换。
- [ ] 构建 auto_buff 并执行不依赖 ROS2 的全项目回归。

验收命令与结果：

```text
待记录
```

建议提交：`refactor(auto_buff): model rune observations and targets`

## Task 3：阶段 2——锚点 PnP 与 Chamfer 位姿精化

硬前置：

- [ ] `buff.avi` 的实测 JSONL v1 标注通过校验。
- [ ] `buff_2.avi` 的实测 JSONL v1 标注通过校验。

实现清单：

- [ ] 维护 1–5 号位、72° 等分及三类模型点：未激活结构、小符激活装甲板、大符激活灯臂。
- [ ] 初次追踪枚举五个候选，按锚点重投影和 Chamfer 代价选优。
- [ ] 后续追踪使用连续相位关联位置。
- [ ] 在 `tools/` 新建 `rune_pose_tools`：多锚点 PnP、距离变换/Chamfer 残差、Ceres 位姿优化。
- [ ] 精化失败安全回退至合格 PnP，并限制输出为 mode 1。
- [ ] 统计优化收敛率、PnP 回退率和位姿跳变。
- [ ] 达到重投影误差 median≤3 px、P95≤6 px。

验收命令与结果：

```text
待记录
```

建议提交：`refactor(auto_buff): rebuild rune pose estimation`

## Task 4：阶段 3——相位与运动估计

- [ ] 在 `tools/` 新建无 Ceres 依赖的 `rune_motion_tools`。
- [ ] 小符实现方向投票、72° 相位解缠和定角速度滤波。
- [ ] 大符使用 `ransac_sine_fitter` 初始化并拟合：

  ```text
  speed = a·sin(ωt+δ) + (2.090-a)
  ```

- [ ] 维护连续相位、角速度、方向、参考时间和置信度。
- [ ] 废弃重复的旧 `buff_predict.hpp`/Target EKF 预测路径。
- [ ] 严格按真实时间推进，禁止按固定帧率外推。
- [ ] 0.5 s 相位预测 MAE≤0.08 rad，P95≤0.15 rad。
- [ ] 连续丢 5 帧后恢复首帧跳变≤0.15 rad。
- [ ] 分别验证大小符和顺/逆时针。

验收命令与结果：

```text
待记录
```

建议提交：`refactor(auto_buff): add reusable rune motion estimator`

## Task 5：阶段 4——决策、弹道与完整输出

- [ ] 将 `buff_aimer` 重构为 `DecisionModule`、`BallisticModel`、`OutputMapper`。
- [ ] 决策只使用结构化 `RuneTarget`、gimbal 状态和跟踪质量。
- [ ] 弹速 `<10 m/s` 时使用 24 m/s，标记降级并禁止首帧直接开火。
- [ ] 弹道迭代计入采集延迟、算法延迟、云台响应时间和子弹飞行时间。
- [ ] 以固定时间差中心差分生成 yaw/pitch 速度及加速度。
- [ ] 执行角度解缠、有限值检查和 YAML 限幅。
- [ ] 开火同时要求正确模式、未激活目标、完整精化位姿、运动模型收敛、弹道可解、目标未切换、冷却完成。
- [ ] 表驱动测试覆盖模式切换、大小符切换、冷却、弹速异常、弹道无解、PnP 降级、目标跳变以及 mode 0/1/2。

验收命令与结果：

```text
待记录
```

建议提交：`refactor(auto_buff): rebuild rune decision and ballistics`

## Task 6：阶段 5——集成与回归

- [ ] 建立离线回放执行器，输出逐帧 JSONL 和汇总 CSV。
- [ ] 对两段视频生成位姿、相位、yaw/pitch、动态量、mode、回退原因和耗时报告。
- [ ] 保留可配置 `plotter`，不加入实时必需路径。
- [ ] 将协议、输入校验、位姿、运动、决策和回放拆成六组独立自动测试并注册 CTest。
- [ ] 保留现有交互式测试程序。
- [ ] 构建并回归 auto_buff、auto_aim、主程序；ROS2 环境可用时验证相关目标。
- [ ] 更新 README、`../Project.md` 阶段勾选和每阶段验证记录。
- [ ] 记录 TinyMPC/Eigen 既有警告基线，确保全量构建无新增警告。

验收命令与结果：

```text
待记录
```

建议提交：`test(auto_buff): complete replay and regression validation`

## 测试与完成标准

- [ ] 每个阶段都执行：失败测试 → 最小实现 → 单目标测试 → auto_buff 构建 → 全项目回归 → 独立提交。
- [ ] 自动测试至少拆分为协议、输入校验、位姿、运动、决策和回放六组独立可执行文件。
- [ ] Ceres 优化失败、NaN/Inf、时间倒退、错误点序和模式冲突全部安全降级，不能产生开火帧。
- [ ] 全量构建不得新增警告；现有 TinyMPC/Eigen 警告单独记录为基线。
- [ ] auto_aim、omniperception 和 gimbal 协议无回归。
- [ ] 可复用算法在 `tools/` 中只有一份实现。
- [ ] 五个受保护车辆 YAML 的内容哈希与实施前一致。

## 阶段验证记录

| 日期 | 阶段 | Commit | 聚焦测试 | 构建/回归 | 备注 |
|---|---|---|---|---|---|
| 待填写 | 阶段 0 | 待填写 | 待填写 | 待填写 | 待填写 |
| 待填写 | 阶段 1 | 待填写 | 待填写 | 待填写 | 待填写 |
| 待填写 | 阶段 2 | 待填写 | 待填写 | 待填写 | 待填写 |
| 待填写 | 阶段 3 | 待填写 | 待填写 | 待填写 | 待填写 |
| 待填写 | 阶段 4 | 待填写 | 待填写 | 待填写 | 待填写 |
| 待填写 | 阶段 5 | 待填写 | 待填写 | 待填写 | 待填写 |
