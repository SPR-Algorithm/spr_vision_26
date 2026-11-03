# auto_buff 打符链路流程设计（图像采集 → 云台指令下发）

> 版本基准：`refactor(auto_buff)` 分支（HEAD 含 ObservationRefiner 实现）
> 日期：2026-09-05
> 性质：打符链路的分模块设计与实现指引，规定各模块的职责边界、输入/输出数据结构、处理流程与调用次序；与仓库 `AUTO_BUFF_REFACTOR_PLAN.md` 的 Task 分期对应；流程编排参考深大 RP-26Rune，仅移植其成熟算法，不引入其框架依赖。

---

## 1. 总体流程与数据接口

整链路按数据流分为六级处理，外加唯一编排入口：

```
图像帧/平台状态
  → S1 编排入口(BuffProcessor)
  → S2 检测(Buff_Detector / YOLO11_BUFF)
  → S3 观测精化(ObservationRefiner)
  → S4 3D 重建与位姿估计(RuneReconstructor / rune_pose_tools)
  → S5 相位与运动估计(PhaseMotionEstimator)
  → S6 决策与弹道(BuffDecisionModule)
  → VisionToGimbal → 串口 → 电控
```

模块间以结构化数据交接，接口类型定义于 `rune_types.hpp`、`buff_processor.hpp`、`io/gimbal/gimbal.hpp`：

```
cv::Mat
  → std::vector<BuffObservation>
  → std::vector<RefinedObservation>
  → RunePose
  → RuneTarget
  → io::VisionToGimbal
```

即：任一模块的输出即为下一模块的输入，无跨模块共享可变状态。

---

## 2. 模块实现状态

| 模块 | 文件 | 状态 |
|---|---|---|
| S2 检测 | `tasks/auto_buff/buff_detector.cpp`、`yolo11_buff.cpp` | ✅ 已实现 |
| S3 观测精化 | `tasks/auto_buff/observation_refiner.cpp/.hpp` | ✅ 已实现 |
| S1 编排 | `tasks/auto_buff/buff_processor.cpp/.hpp` | 已实现占位，需扩展为全链路编排 |
| S4 3D 重建与位姿 | `tasks/auto_buff/rune_reconstructor.hpp/.cpp`（新增） | 待实现 |
| S4 位姿工具 | `tasks/auto_buff/rune_pose_tools.hpp/.cpp`（新增） | 待实现 |
| S5 运动估计 | `tasks/auto_buff/phase_motion_estimator.hpp/.cpp`（新增） | 待实现 |
| S6 决策与弹道 | `tasks/auto_buff/buff_decision_module.hpp/.cpp`（新增） | 待实现 |
| 构建 | `tasks/auto_buff/CMakeLists.txt` | 需注册新增源文件 |

---

## 3. 模块详细设计

### S2 检测 `Buff_Detector`（已实现）

- 职责：由 BGR 图像生成一帧内所有扇叶的标准五点观测。
- 输入：`cv::Mat & bgr_img`。
- 处理：`YOLO11_BUFF::get_multicandidateboxes` 完成 OpenVINO 推理与后处理（letterbox → 推理 → anchor 解码 → NMS）；逐目标调用 `to_observation`，将模型关键点序 `[top,left,R,right,bottom]` 重排为公开序 `[top,right,bottom,left,R]`。
- 输出：`std::vector<BuffObservation>`（NMS 已按 `quality = conf × mean(kpt_conf)` 降序，首元素为最高置信观测）
  - `BuffObservation::points`（`BuffPoints`）：五点像素坐标
  - `BuffObservation::activation`（`BuffActivation`）：INACTIVE / SMALL_ACTIVATED / BIG_ACTIVATED
  - `BuffObservation::confidence`：检测置信度

### S3 观测精化 `ObservationRefiner`（已实现）

- 职责：以网络五点观测为先验，对每一片扇叶做传统视觉精化——语义差分提取轮廓、三类互斥语义分类（装甲板 / R标 / 灯臂）、几何校验与质量分级。产出可供 S4 位姿对齐使用的轮廓特征。
- 输入：
  - `const cv::Mat & image`：原始 BGR 图像
  - `const BuffObservation & observation`：S2 输出单片观测
  - `const RuneState & state`：大小符状态（由电控档位与激活类别解析）
- 处理（`refine()`）：
  1. `validate_observation`：图像格式、置信度、五点有限性与界内、几何退化（凸性/最短边/面积/R 标与角点不重合）逐项校验；不合格返回 `INVALID` 并附 `ObservationRejectReason`。
  2. `make_roi`：由五点外接矩形按比例外扩生成 ROI。
  3. `make_binary_mask`：通道差 `absdiff(B,R)` → 高斯滤波 → 阈值二值 → 闭运算（形态学）。
  4. `extract_semantic_contours`：findContours 后按包含/排斥约束分类：含装甲中心且面积误差最小者为装甲板；含 R 点且不含其余语义点为 R 标（取面积最小）；不含两类点且"装甲中心—R"连线穿过者为灯臂（取面积最大）。每类计算实心度等 `support`。
  5. 质量判定：三类轮廓齐备、置信度与几何分达阈值 → `GOOD`，否则 `DEGRADED`。
- 输出：`RefinedObservation`
  - `state`、`points`、`detection_confidence`
  - `roi`（`cv::Rect`）、`armor_polygon`
  - `armor_contour` / `r_mark_contour` / `light_arm_contour`（`ContourFeature`：全图坐标轮廓 + `support` + `usable()`）
  - `quality`（`TrackingQuality`）、`geometry_score`

多片场景：对观测集合循环调用 `refine`，剔除 `quality == INVALID` 的片。

### S4 3D 重建与位姿估计 `RuneReconstructor` + `rune_pose_tools`（新增）

- 职责：将多片 2D 精化观测组合为能量机关的 3D 几何——圆心、盘面法向与整体位姿。单一片无法确定圆心，必须以多片联合求解；对标深大 `PowerRunePlane`/`Projector`。
- 输入：
  - `const std::vector<RefinedObservation> & blades`：S3 有效片集合
  - `const Eigen::Quaterniond & imu_q`：机体姿态（用于 gimbal→world 坐标变换）
  - 相机内参与相机—云台外参（构造时由配置读取）
- 处理（`reconstruct()`）：
  1. 号位匹配：按各片相对圆心的极角，将扇叶分配到 1~5 号位（槽位间隔 72°=2π/5），以未激活片为 1 号位基准。
  2. 圆心像素初值：对多片 `r_mark_contour` 做 `cv::fitEllipse` 取圆心。
  3. 锚点聚合：取未激活片的装甲板/灯臂轮廓 PCA 定向外接矩形角点作为锚点。
  4. 先验位姿：锚点 2D 与对应 3D 模型锚框点做多锚点 `solvePnP`（`PoseSource::PNP`）。
  5. 位姿精修：调用 `rune_pose_tools` 执行 Chamfer 距离场 + Ceres 优化；成功置 `PoseSource::CHAMFER_REFINED`，失败回退 PnP 结果并标记该帧可信度受限。
  6. 平面估计：对重建点云去中心化 SVD 拟合盘面，最小奇异值方向为法向并按朝向约束翻转。
- 输出：`std::optional<RunePose>`
  - `position_world_m`（`Eigen::Vector3d`）：圆心世界坐标
  - `orientation_world`（`Eigen::Quaterniond`）：符整体姿态
  - `plane_normal_world`：盘面法向（单位向量）
  - `reprojection_error_px`：位姿重投影误差，供决策层判断可信度
  - `confidence`、`source`（`PoseSource`：PNP / CHAMFER_REFINED）
  - 空 optional：本帧无法重建，链路输出降级 mode。

`rune_pose_tools`（无状态工具函数）：
- `solve_pnp_multi_anchor(anchors_2d, anchors_3d, camera_matrix, dist_coeffs) → rvec, tvec`
- `make_distance_transform(contours, roi) → cv::Mat(CV_32FC1)`：轮廓边界的欧氏距离场
- `refine_by_chamfer(model_points, distance_transform, camera_params, prior_pose, plane_y_constraint_weight) → refined_pose, reprojection_error`

### S5 相位与运动估计 `PhaseMotionEstimator`（新增）

- 职责：由靶心观测与 3D 几何维护连续相位与运动模型，为命中时刻外推提供依据；对标深大 `PhaseMotionEstimator`。
- 输入：
  - `const RunePose & pose`：S4 输出的符几何
  - `const RefinedObservation & target_blade`：当前 target 片观测
  - `const std::chrono::steady_clock::time_point & timestamp`：帧时间
- 处理（按大小符分流）：
  1. 将靶心投影至盘面、绕法向求观测相位。
  2. 相位连续化：跨帧解缠；检测 target 切换，按 `k·2π/5` 平移整窗历史相位，避免切换被误判为转速突变。
  3. 小符：旋转方向投票 + 一维相位卡尔曼（角速度初始化 ±π/3）。
  4. 大符：LM-IRLS 拟合相位方程 `θ(t) = A·cos(ωt) + B·sin(ωt) + b·t + C`（外层 LM 迭代 ω，内层 IRLS 加权最小二乘），预测直接由方程外推。
- 输出：`std::optional<RuneTarget>`
  - `RefinedObservation observation`：当前观测
  - `RunePose pose`：S4 结果
  - `MotionState motion`：`continuous_phase_rad`、`angular_velocity_rad_s`、`direction`（RotationDirection）、`reference_timestamp`、`converged`
  - `TrackingQuality quality`
  - 空 optional：运动未收敛，仅可跟踪（mode 1）。

### S6 决策与弹道 `BuffDecisionModule`（新增）

- 职责：由运动模型外推命中时刻，联合弹道解算云台角，并经决策状态机输出开火许可；对标深大 `RuneDecisionModule`。
- 输入：
  - `const RuneTarget & target`：S5 输出
  - `const io::GimbalState & gimbal`：含 `bullet_speed`、云台姿态
  - `const time_point & now`
- 处理：
  1. **命中点外推**：命中点无法预先固定（扇叶旋转 + 子弹飞行时间非零），按固定点迭代求命中相位：
     - 以初始飞行时间估计 t₀ → 由 `MotionState` 外推 `phase(now + t)`；
     - 由相位求命中靶心 3D：`armor_center_at(圆心, 起始方向, 盘面法向, 半径, phase)`（起始方向绕法向旋转 phase、乘以半径 0.7 m 后叠加圆心）；
     - 以靶心解算弹道得新飞行时间，迭代至收敛（判据如 `|Δt| < 0.01 s`）。
  2. **弹道解算**：输入靶心水平距离 d、高度 h 与弹速 v0，输出 pitch 与飞行时间。先沿用现有无阻力模型 `tools/trajectory.cpp`；后续可升级为含空气阻力 + 马格努斯项的 RK4 与 Ceres 联合拦截。
  3. **决策状态机**：目标切换防抖（大符需连续多帧确认）、开火冷却计时、数据超时平滑回圆心（`recover2rune_center`）、开火许可判定。
- 输出：决策结果，映射为 `VisionToGimbal` 字段
  - `mode`：0 空闲 / 1 跟踪禁开火 / 2 可开火
  - `yaw`、`pitch`：云台期望角
  - `yaw_vel`、`yaw_acc`、`pitch_vel`、`pitch_acc`：速度/加速度前馈

### S1 编排 `BuffProcessor::process`（需扩展）

- 职责：唯一有状态入口，按 S2→S3→S4→S5→S6 顺序编排单帧处理；任一环节失败即提前返回降级 mode，不产生开火帧。
- 处理补充：mode 0 时清零全部运动字段；模式切换 / 时间倒退 / 超过 `tracking_timeout_ms` 时重置内部状态；最终把决策结果封装为 `io::VisionToGimbal`。
- 输出：`io::VisionToGimbal`（28 字节，SP 头 + mode + 7 个 double + 0xef 尾）→ `gimbal.send()`。

---

## 4. 与遗留代码的关系

`buff_solver`、`buff_target`、`buff_aimer` 及 `buff_type`(PowerRune 链路) 当前已无调用方（主程序仅经 `Buff_Detector → BuffProcessor` 执行）。本设计的替换关系：

| 环节 | 遗留实现 | 本设计 |
|---|---|---|
| 3D 位姿 | `buff_solver`：单扇叶 5 点 PnP | PnP 保留为先验 + Chamfer 精修（多片联合），S4 |
| 大符运动预测 | `buff_target`：10 维 EKF + RANSAC | LM-IRLS 相位方程拟合 + 外推，S5 |
| 小符运动预测 | 7 维 EKF | 一维相位卡尔曼 + 方向投票，S5 |
| 开火决策 | 旧 `aimer`：角度阈值 + mistake_count | 独立决策状态机（冷却/防抖/回圆心），S6 |
| 弹道 | `tools/trajectory.cpp`（无阻力两轮迭代） | 先复用；后续升级含阻力 RK4 + Ceres 联合拦截 |

大符弃用 EKF+RANSAC 的原因：角速度作状态导致噪声两次放大；RANSAC 拟合结果未接入预测；target 切换易被误判为高速跳变。

---

## 5. 与 `AUTO_BUFF_REFACTOR_PLAN.md` 对照

| 计划 Task | 计划名称 | 本设计模块 | 状态 |
|---|---|---|---|
| Task 1 | 阶段 0 构建基线/统一入口 | S1 门面骨架 | ✅ 完成 |
| Task 2 | 阶段 1 观测数据模型 | S2+S3 + `rune_types` + JSONL 标注/校验器 | 部分完成（S2/S3 完成；标注未做） |
| Task 3 | 阶段 2 锚点 PnP + Chamfer | S4 | 阻塞：缺标注真值验收 |
| Task 4 | 阶段 3 相位与运动估计 | S5 | 待实现 |
| Task 5 | 阶段 4 决策/弹道/输出 | S6 | 待实现 |
| Task 6 | 阶段 5 集成与回归 | S1 收口 + 测试 | 待实现 |

---

## 6. 深大 RP-26Rune 借鉴点

**移植保留**：R 标轮廓拟合圆心；多片锚点 PnP + Chamfer 精修并可回退；大符 LM-IRLS 相位方程拟合；target 切换的 `k·2π/5` 整窗相位补偿；小符方向投票与一维相位卡尔曼；决策状态机（冷却/切换防抖/超时回圆心）；mode 0/1/2 输出语义。

**不引入**：TFTree / Foxglove / glog 等框架依赖；插件式架构；`compensate_pitch`（对方弃用的死代码）；多帧平面滑窗（对方实测退化）。

---

## 7. 实现与验收顺序

阻塞仅存在于真值验收一环；以下工作不依赖标注，可先行：

1. S1：扩展 `BuffProcessor::process` 为逐级编排（先以桩实现接通数据流）。
2. S4：实现号位匹配、圆心拟合、多锚点 PnP 先验与 SVD 平面估计（可用合成 72° 五片数据单测）。
3. S5：实现相位解缠、target 切换补偿、小符一维卡尔曼与大符 LM-IRLS 拟合。
4. S6：实现命中点外推—弹道迭代与决策状态机（表驱动单测冷却/防抖/回圆心）。
5. JSONL v1 标注完成后，执行精度验收（如位姿重投影误差 median ≤ 3 px、P95 ≤ 6 px；大符 0.5 s 相位预测误差指标）。

落地勾选清单与各模块验收命令记录于 `AUTO_BUFF_REFACTOR_PLAN.md` 对应 Task，实现时同步勾选。
