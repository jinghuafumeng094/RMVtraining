# EKF 跟踪与预测（§3.6）代码解释

对应评分项：**EKF 跟踪与预测（20 分）**
对应规格：`轨迹预测说明.md`「旋转车体 EKF 预测要求」、`docs/大作业任务书.md` §3.6
代码位置：
- `student/include/ArmorEkf.hpp`（335 行，类声明、状态下标、全部可调常量）
- `student/src/ArmorEkf.cpp`（936 行，全部实现）
- `student/src/StudentTasks.cpp:206-246`（`ekf_predict`，跨帧状态的适配器）
- `tests/test_ekf_predict.cpp`（1482 行，22 条用例：无噪收敛、雅可比对拍、状态机时序、安全不变量、
  单板初始化回归、偏航观测、半径边界、同帧重复框、跳变闸外重建、下坠补偿回归、NIS 出口契约、
  逐帧 `best_nis` 出口；夹具只喂 80 度锥角内真实可见的板，与 `testrig/ekf_trace` 同源）

本文按「一帧数据怎么流过这几百行」的顺序讲，每个小节末尾给行号锚点。答辩要独立解释实现时，照本文的节序讲即可。

---

# 零、这一层在整个流水线里的位置

`main.cpp:566` 那一行的前后：

```
armor_solve 的产物                          EKF 的输入
observations: vector<ArmorPose>
   每块板: position_gimbal_m  云台系板心 (m)   <- EKF 读，三维位置观测
           position_camera_m  相机系板心 (m)   <- 调试用，EKF 不看
           armor_yaw          板面法向方位角    <- EKF 读，标量更新（2026-10-20 起）
           reprojection_error                  <- 当闸门用
           detection.target_id / size          <- 只记进槽位表，不参与关联
                    |
                    v
ekf_predict(observations, gimbal, time_stamp)     StudentTasks.cpp:206
    内部 static ArmorEkf filter                   跨帧状态就活在这个 static 里
                    |
                    v
PredictionResult
    state (= crc), target_id, center_x/y_m, center_velocity_x/y_m_s,
    body_yaw, body_yaw_velocity, radius_1_m, radius_2_m, distance_m,
    target_yaw, target_pitch, target_yaw_velocity, target_pitch_velocity,
    fire_allowed
                    |
                    v
applyPrediction(translator, prediction)           src/AppSupport.cpp:74
```

三条设计决定先摆在这里，后面每一节都在兑现它们：

1. **运动模型建在世界系**，不是云台系。云台会转，同一个静止目标在云台系里的坐标每帧都在变；在云台系里做运动模型会把云台的转动误当成目标的运动。
2. **观测是板心三维坐标，外加一条 `armor_yaw`**。三维位置是主观测；`armor_yaw` 2026-10-20 起作为标量观测单独更新，目的是在单板帧下把偏航从切向位置里分出来（见第十二节 12.2）。它对任务书「板心方位角」措辞的常值差已并进换算式。
3. **`ArmorEkf` 只依赖 Eigen 和标准库**，不碰 OpenCV，也不碰串口。所以它可以被 `ekf_trace` 这类仓库外工具直接构造、逐帧驱动，不需要相机。

---

# 一、三个坐标系与唯一的那个变换

## 1.1 三个系

| 记号 | 各轴 | 原点 | 从哪来 |
|---|---|---|---|
| **G 云台系** | x 前、y 左、z 上 | 云台旋转中心 | `armor_solve` 输出的 `position_gimbal_m` 就在这个系里 |
| **W 世界系** | x 前、y 左、z 上 | 云台旋转中心在初始时刻的位置 | 与 `armor/include/BulletModel.hpp` 的弹道模型一致 |
| **物体系**（装甲板） | x 板右、y 板下、z 指板背 | 板心 | `armor_solve` 里 `plateObjectPoints` |

**G 与 W 的轴向完全相同**（学长 2026-10-15：「云台系和世界系一样」），两系只差一个绕竖直轴的偏航角，再外加原点平移。依据与推导过程见 `docs/上车前问题清单.md` 第 1.1 节，图见 `docs/img/坐标系方向.png`。

两个偏航角的转正方向**故意不同**，这不是笔误（`ArmorEkf.cpp:24-27` 有原话）：

- `theta_c` 是 W 系里的方位角，从 +x 转向 +y 为正，而 W 的 +y 朝左，所以 `theta_c` **左为正**。
- `gimbal.yaw` 在 `atan2(-y, x)` 这个口径下定义成 **右为正**，与 `main.cpp` 的 `pnp_yaw_rad`、`armor_solve` 的 `armor_yaw` 同号。

两者各自只在观测模型里用一次，滤波器因此仍然自洽，不需要强行统一。与电控实际符号对不对，仓库里没有任何一处写明，只能上车确认。

> **（本节 2026-10-15 改过。）** 原先把云台系定义成与相机系各轴同向（x 右 / y 下 / z 前），G 与 W 之间因此要补一个 90° 的轴置换。学长 2026-10-15 定下云台系与世界系同向之后，那个 90° 从两系之间挪到了**相机系 -> 云台系的旋转外参 R** 上（见 `docs/答辩-位姿解算.md` §1.2），本节与 §1.3 的变换就只剩一个偏航。

## 1.2 唯一的变换函数

`ArmorEkf.cpp:122-134`：

```cpp
Eigen::Matrix3d ArmorEkf::gimbalToWorld(double gimbal_yaw) {
    const double s = std::sin(gimbal_yaw);
    const double c = std::cos(gimbal_yaw);
    Eigen::Matrix3d m;
    m << c,   s,   0.0,
         -s,   c,   0.0,
         0.0,  0.0, 1.0;
    return m;
}

Eigen::Matrix3d ArmorEkf::worldToGimbal(double gimbal_yaw) {
    return gimbalToWorld(gimbal_yaw).transpose();
}
```

**整个 `ArmorEkf.cpp` 里，G 与 W 之间的坐标变换只出现这一个函数。** 三个调用点：

| 调用点 | 方向 | 用途 |
|---|---|---|
| `tryInitialize:397` | G -> W | 把初始化的几帧观测搬进 W 系比一致性 |
| `observationModel:183` | W -> G | 预测观测值，与 `position_gimbal_m` 同系才能相减 |
| `selectArmorIndex:735` / `buildPrediction:804,832` | W -> G | 把 W 系的瞄准点搬回当前云台系算 `rel_yaw/rel_pitch` |

## 1.3 这个矩阵怎么来的

**第一步，取 `gimbal_yaw = 0`（云台正对前方）。**

两系轴向相同（前 / 左 / 上），所以这一步**一次改名都不用做**：

```
M(0) = I
```

这正是「云台系和世界系一样」的直接结果。旧口径下这里是一个 90° 的轴置换矩阵，现在它挪到了相机外参 `R` 上。

**第二步，把云台的转动叠上去。**

云台绕竖直轴转 `gimbal_yaw`。两系的竖直轴都是 z，所以转动就是绕 z 的一次旋转，唯一的自由度是符号：

```
M(θg) = R_z(-θg)
```

拿 `s, c` 代进去展开，正好得到代码里那个矩阵（第一行 `[c, s, 0]`、第二行 `[-s, c, 0]`、第三行 `[0, 0, 1]`）。

**那个负号就是「右为正」的核对**，答辩时值得当场推一遍。`s = sin θg`：取 G 系正前方 d 米的点 `p_G = (d, 0, 0)`：

```
M(θg) * (d,0,0)^T = d * (第 1 列) = d * (c, -s, 0)
```

`θg > 0` 时第二个分量是负的，而 W 的 +y 朝**左**，所以这个点在 W 里偏到了**右边** —— 云台往右转，正前方的点在世界上确实跑到右边。若写成 `R_z(+θg)`，正前方会被转到左边，与「右为正」相反。行列式恒 +1，永远是纯转动。

自检锚点（答辩时背这几条就够）：

```
θg = 0,  M = I                              两系同向，不转就重合
θg > 0（右正）,  p_G = (d,0,0)  ->  p_W = (d·cosθg, -d·sinθg, 0)   视线偏到右边
det(M(θg)) = +1                             纯转动，不翻转不缩放
worldToGimbal = transpose                   正交阵的逆就是转置，不用求逆
```

`tests/test_ekf_predict.cpp:368` 的坐标变换用例钉住的就是这几条。

`tests/test_ekf_predict.cpp:368-389` 把 `θg = 0.2` 时的方向也钉住了：转完的视线方向再绕回云台角，方位角必须精确复原。

## 1.4 角度包裹

`wrapToPi`（:146-152）：

```cpp
double ArmorEkf::wrapToPi(double angle) {
    double wrapped = std::fmod(angle + kPi, 2.0 * kPi);
    if (wrapped <= 0.0) {
        wrapped += 2.0 * kPi;
    }
    return wrapped - kPi;
}
```

把角度折算到 $(-\pi, \pi]$。`propagate` 里每推一步包一次，`associateAndUpdate` 更新完包一次。这是「用 `sin/cos` 建模型」之外的第二道防线：`h` 对 `theta_c` 天然以 $2\pi$ 为周期是主防线，包裹是防止状态值本身漂到很大之后浮点精度变差。

---

# 二、状态向量与下标

`ArmorEkf.hpp:43-60`：

```cpp
using StateVector = Eigen::Matrix<double, 10, 1>;
using Covariance = Eigen::Matrix<double, 10, 10>;
using ObservationJacobian = Eigen::Matrix<double, 3, 10>;

enum Index : int {
    Px = 0, Py = 1, Pz = 2,
    Vx = 3, Vy = 4, Vz = 5,
    Yaw = 6, YawRate = 7,
    Radius1 = 8, Radius2 = 9,
    Dimension = 10,
};
```

```
x = [ px, py, pz,  vx, vy, vz,  theta_c,  omega,  r1,   r2 ]^T
      旋转中心 (m)   中心速度 (m/s)   车体偏航 角速度  半径 (m)
```

| 分量 | 含义 | 落在哪条任务要求 |
|---|---|---|
| `px, py, pz` | 旋转中心在 W 系的位置 | 任务书 §3.6①「旋转车体中心平移」 |
| `vx, vy, vz` | 旋转中心速度 | 同上 |
| `theta_c` | 车体偏航角，W 系方位角，左为正 | 任务书「旋转车体」 |
| `omega` | 车体偏航角速度 | 同上 |
| `r1, r2` | 偶数编号板用 `r1`、奇数用 `r2` | 「交替半径」，对应 `PredictionResult::radius_1_m/radius_2_m` |

用 `enum` 而不是裸数字，是为了「下标含义」在代码里只写一次。`observationJacobian` 里 `H.col(Px)` 这种写法，比 `H.col(0)` 好审。

**高度差 `h_i` 不进状态，进常量表**（`Config::plate_height_m`，默认全零）。理由：高度与 `pz` 强相关，单块板时不可观测，两个都放进状态会把彼此带偏。前哨站不是共面的，改成 `{0, -0.102, 0.102, 0}` 即可，代码不动（来源 `armor/include/Planner.hpp`）。

---

# 三、主循环：一帧调四次

`ekf_predict` 这个适配器（`StudentTasks.cpp:206-246`）每帧固定做这几件事：

```cpp
// 两辆车同时灌进四槽关联器会互相抢槽位，滤波之前先选出一辆车（:212-213）
const armor::TargetSelection selection =
    selector.select(observations, gimbal.yaw, timestamp_seconds);
// 换簇不等于换车，只有本来就没有目标时才重置（:220-222）
if (selection.switched && filter.status() == TrackingState::NoTarget) {
    filter.reset();
}

filter.predict(dt);                                 // :239
for (int index : selection.observation_indices) {
    filter.update(observations[index], gimbal.yaw); // :241
}
filter.advanceStateMachine(timestamp_seconds);      // :244
return filter.buildPrediction(gimbal, latency_s);   // :245
```

| 调用 | 什么时候调 | 没观测时 |
|---|---|---|
| `selector.select(...)` | 每帧一次，在 `predict` 之前 | 返回空选择，后续不喂观测 |
| `predict(dt)` | 每帧，包括没有观测的帧 | 照常做纯预测 |
| `update(obs, yaw)` | 每有一条观测调一次 | 不调 |
| `advanceStateMachine(ts)` | 每帧一次 | 照常调（它负责计 miss、超时重置） |
| `buildPrediction(gimbal, latency)` | 每帧一次 | 状态 `NoTarget` 时返回全零 |

`predict` 和 `advanceStateMachine` 必须每帧都调，这是状态机能走对的前提。`ekf_predict` 的注释和 `ArmorEkf.hpp:199-204` 都写明了这一点。

`switched` 只在 `filter.status() == TrackingState::NoTarget` 时才触发 `reset()`。原因是同一辆车的两块板在筛选层被拆成两簇也会置 `switched`，此时滤波器可能正处在 `Stable`，无条件重置会丢掉收敛好的半径与速度，重建时按单板视线假设把车心放偏，第二块板从此落在 `max_radius` 之外。真换了车的情况，旧目标的观测全部落在跳变闸或 NIS 闸之外，连续 miss 到 `reinit_after_misses` 会就地重建，出口一样。用例 `switched_cluster_keeps_state` 钉住这条。

## 3.1 `predict(dt)`：先验

`ArmorEkf.cpp:342-364`：

```cpp
void ArmorEkf::predict(double dt) {
    used_indices_.fill(false);          // 每帧重置「这块板配过没」
    updated_this_frame_ = false;        // 每帧重置「这帧吃过观测没」
    init_used_this_frame_ = false;
    if (status_ == TrackingState::NoTarget) {
        return;                          // 还没建起来，不预测
    }
    if (!std::isfinite(dt)) {
        return;
    }
    dt = std::clamp(dt, 0.0, config_.max_step_s);   // dt 上限 0.5 s
    if (!(dt > 0.0)) {
        return;
    }

    propagate(x_, dt);
    Eigen::Matrix<double, 10, 10> F;
    fillTransition(dt, F);
    Covariance Q;
    fillProcessNoise(dt, Q);
    P_ = F * P_ * F.transpose() + Q;
}
```

三个 `fill(false)` 是「每帧的清零」。`used_indices_` 保证一帧内一块板不会被两条观测同时配上；`updated_this_frame_` 是状态机判命中/丢失的依据；`init_used_this_frame_` 让初始化窗口按帧计数（见第七节）。

`dt` 加 `clamp` 到 `[0, 0.5]`：时间戳异常或线程卡住时，一次大 `dt` 会把 `P` 冲爆。

## 3.2 过程模型 `propagate` 与 `F`

`ArmorEkf.cpp:243-256`：

```cpp
void ArmorEkf::propagate(StateVector& x, double dt) const {
    x[Px] += x[Vx] * dt;
    x[Py] += x[Vy] * dt;
    x[Pz] += x[Vz] * dt;
    x[Yaw] = wrapToPi(x[Yaw] + x[YawRate] * dt);
}

void ArmorEkf::fillTransition(double dt, Eigen::Matrix<double, 10, 10>& F) const {
    F.setIdentity();
    F(Px, Vx) = dt;
    F(Py, Vy) = dt;
    F(Pz, Vz) = dt;
    F(Yaw, YawRate) = dt;
}
```

匀速直线 + 匀角速度。`F` 是单位阵外加四个 `dt`。`vx/vy/vz` 和 `theta_c/omega` 自己是随机游走（`F` 里它们不动），随机性由 `Q` 给。

**注意 `theta_c` 的传播是真非线性的**（`wrapToPi`），但 `F` 用的是线性化的雅可比（`F(Yaw,YawRate) = dt`），这是 EKF 的常规做法，误差项被 `Q` 吸收。

## 3.3 过程噪声 `Q`

`ArmorEkf.cpp:258-287`。按随机加速度模型离散化，每个「位置 + 速度」轴对共享一个分块：

```
Q_pv = sigma_a^2 * [ dt^4/4   dt^3/2 ]
                   [ dt^3/2   dt^2   ]
```

```cpp
const double linear = config_.accel_noise_m_s2 * config_.accel_noise_m_s2;
const int pairs[3][2] = {{Px, Vx}, {Py, Vy}, {Pz, Vz}};
for (const auto& pair : pairs) {
    Q(pair[0], pair[0]) = linear * dt4 / 4.0;
    Q(pair[0], pair[1]) = linear * dt3 / 2.0;
    Q(pair[1], pair[0]) = linear * dt3 / 2.0;
    Q(pair[1], pair[1]) = linear * dt2;
}
```

`[theta_c, omega]` 那一块形状一样，强度换 `angular_accel_noise_rad_s2`。半径只给一个极小方差：

```cpp
const double radius_variance =
    std::max(config_.radius_drift_m * config_.radius_drift_m * dt2,
             config_.radius_variance_floor_m2);
Q(Radius1, Radius1) = radius_variance;
Q(Radius2, Radius2) = radius_variance;
```

**2026-10-21 起半径不再是待估量**：`min_radius_m = max_radius_m = initial_radius_m = 0.303`，`clampRadii` 每帧把它夹回这个常量，`initial_radius_std_m` 降到 0.003，免得协方差与钳位互相矛盾。这一段 `Q` 仍逐帧计算、仍写进协方差，但半径方向被钳位压住，不再漂。原因是单板帧里半径与车心径向严格不可辨识（`δr = ε`、`δc = -ε·r` 给出同一个板心），放开它只会让 r 滑到边界，再靠 `clampRadii` 平移车心维持自洽，车心因此每帧多出几厘米的位移。`radius_variance_floor_m2 = 9e-8`（即 0.3 mm 的平方）在 166 fps 下是主导项：`radius_drift_m² · dt²` 只有 1.4e-8，比它小一个数量级。

### 这些数从哪来

| 常量 | 值 | 依据 |
|---|---|---|
| `accel_noise_m_s2` | 6.0 | RM 步兵 0 到 20 m/s 约 1 到 2 秒，瞬时机动约 10 m/s²，取一半（`ArmorEkf.cpp:36-37`） |
| `angular_accel_noise_rad_s2` | 12.0 | 车能快速起旋停旋，同量级取整 |
| `radius_drift_m` | 0.02 | 半径近乎恒定，只给随机游走强度 0.02 m/√s |

调参顺序按博客 §3.7：**先验单位与模型 -> 固定 `R` 调 `Q` -> 最后才谈动态噪声**。`Q` 大则跟得快但抖，小则平滑但滞后。

## 3.4 `P0`

`ArmorEkf.cpp:303-319`：

```cpp
P(Px, Px) = 1.0;      P(Py, Py) = 1.0;      P(Pz, Pz) = 1.0;      // 位置 (m^2)
P(Vx, Vx) = 25.0;     P(Vy, Vy) = 25.0;     P(Vz, Vz) = 9.0;      // 速度 (m/s)^2
P(Yaw, Yaw) = 0.25;   P(YawRate, YawRate) = 9.0;                  // rad^2 / (rad/s)^2
P(Radius1, Radius1) = 0.0004;  P(Radius2, Radius2) = 0.0004;      // m^2
```

`P0` 给大一点，等于说「初始化时先信观测」，随后靠 `P_ = F P F^T + Q` 和更新自然收缩。位置给 1 m² 是「中心可能离这块板一米左右」的量级；速度给 5 m/s 的标准差是「不知道车怎么动」。

---

# 四、观测模型 `h` 与雅可比 `H`

## 4.1 `h`

`ArmorEkf.cpp:172-184`：

```cpp
Eigen::Vector3d ArmorEkf::observationModel(const StateVector& x, int armor_index,
                                           double gimbal_yaw,
                                           const std::array<double, 4>& plate_height_m,
                                           double plate_spacing_rad) {
    const int i = armor_index;
    const double psi = x[Yaw] + static_cast<double>(i) * plate_spacing_rad;
    const double radius = x[radiusIndex(i)];
    Eigen::Vector3d u;
    u << x[Px] + radius * std::cos(psi),
         x[Py] + radius * std::sin(psi),
         x[Pz] + plate_height_m[static_cast<std::size_t>(i)];
    return worldToGimbal(gimbal_yaw) * u;
}
```

写成公式：

```
psi_i  = theta_c + i * (pi/2)                     第 i 块板的方位角
r_i    = r1 若 i 偶, r2 若 i 奇                     radiusIndex(i)

u_i(x) = [ px + r_i*cos(psi_i),                    W 系里第 i 块板的位置
           py + r_i*sin(psi_i),
           pz + h_i ]

h_i(x) = M(theta_g)^T * u_i(x)                    换到 G 系, 3 维观测
```

对照 `z = position_gimbal_m`：两者同系，所以 `residual = z - h_i(x)` 直接相减，不需要任何附加变换。**这就是整条链条在 `armor_solve` 里断在云台系的好处**：EKF 这边只要把「假设的板位置」推到云台系就行。

`plate_spacing_rad` 默认 $\pi/2$（标准步兵四块 90 度），前哨站改 $2\pi/3$。

## 4.2 `H`

`ArmorEkf.cpp:186-213`。非零的四个列：

```cpp
ObservationJacobian H = ObservationJacobian::Zero();
H.col(Px) = to_gimbal * Eigen::Vector3d(1.0, 0.0, 0.0);
H.col(Py) = to_gimbal * Eigen::Vector3d(0.0, 1.0, 0.0);
H.col(Pz) = to_gimbal * Eigen::Vector3d(0.0, 0.0, 1.0);
//   u 对 theta_c 的偏导 = r * (-sin psi, cos psi, 0)   方位角转，切向
//   u 对 r 的偏导       =     ( cos psi, sin psi, 0)   半径伸长，径向
const Eigen::Vector3d tangential(-std::sin(psi), std::cos(psi), 0.0);
const Eigen::Vector3d radial(std::cos(psi), std::sin(psi), 0.0);
H.col(Yaw) = to_gimbal * (radius * tangential);
H.col(radiusIndex(i)) = to_gimbal * radial;
```

其中 `to_gimbal = worldToGimbal(gimbal_yaw)`，`psi = psi_i = theta_c + i·pi/2`。对 `vx/vy/vz/omega` 全为 0（观测模型里没有它们）。

推导要点：`h = M^T u`，`M^T` 在这一步是常量，所以每一列就是 `M^T` 乘上 `u` 对那个分量在 W 系里的偏导。**代码里一律走 `worldToGimbal`，不手抄 `M^T` 的元素** —— 手抄等于把同一个约定写两遍，换约定时容易只改一处。

展开成显式元素（`psi_plus_gimbal = psi_i + gimbal_yaw`，也是能一眼核对的形式）：

```
dh/dpx = M^T * (1,0,0)^T = ( c,  s, 0)
dh/dpy = M^T * (0,1,0)^T = (-s,  c, 0)
dh/dpz = M^T * (0,0,1)^T = ( 0,  0, 1)

H.col(Yaw)    = M^T * r_i * (-sin psi_i, cos psi_i, 0) = r_i * (-sin psi_plus_gimbal, cos psi_plus_gimbal, 0)
H.col(radius) = M^T *       ( cos psi_i, sin psi_i, 0) =       ( cos psi_plus_gimbal, sin psi_plus_gimbal, 0)
```

**注意最后两列的方向差 90°**：方位角转一格是**切向**位移，半径伸长一笔是**径向**位移，两者不能共用同一个向量。共用会把半径那一列写成切向，滤波器不报错、也不发散的迹象，只有对拍差分的用例会红。半径列**只有该板对应的那一列非零**（另一列已在 `H = Zero()` 里置零）。

**手推雅可比唯一的保险是解析对拍差分**，`tests/test_ekf_predict.cpp` 有一条用例逐元素比 `observationJacobian` 与有限差分，误差 < 1e-6。这个函数写错的话，滤波器会收敛慢或不收敛，而重投影误差之类的闸门都看不出来，是静默错误。

## 4.3 关联用的 `radiusIndex` 与槽位

`ArmorEkf.cpp:158-170`：

```cpp
int ArmorEkf::normalizeIndex(int armor_index, int plate_count) {
    if (plate_count <= 0) return 0;
    const int wrapped = armor_index % plate_count;
    return wrapped < 0 ? wrapped + plate_count : wrapped;
}

int ArmorEkf::radiusIndex(int armor_index) {
    return (armor_index % 2 == 0) ? Radius1 : Radius2;
}
```

**这里有一个必须知道的规范对称**：`r1 = r2` 时，把槽位整体挪一格（`i -> i+1`）同时 `theta_c -> theta_c - 90°`，四块板的位置角逐个不变（`psi` 只是整体重排），代入的半径又一样，所以两组状态**预测逐点完全相同**，中心也不变。也就是说「槽位编号锁在哪一支」在等半径时不影响任何可观测量。锁错的后果只在两半径不等时暴露，见第七、十二节。

---

# 五、更新：一条观测怎么被吃下或拒绝

`update`（:370-378）是分派器：

```cpp
bool ArmorEkf::update(const ArmorPose& observation, double gimbal_yaw) {
    if (!std::isfinite(gimbal_yaw)) gimbal_yaw = 0.0;
    if (status_ == TrackingState::NoTarget) {
        return tryInitialize(observation, gimbal_yaw);   // 还没建起来：攒初始化
    }
    return associateAndUpdate(observation, gimbal_yaw);   // 已建起来：关联 + 更新
}
```

## 5.1 `associateAndUpdate`：三道闸

`ArmorEkf.cpp:465-617`。进来先过两道「观测本身合不合法」的闸：

```cpp
if (!z.allFinite() || !std::isfinite(observation.reprojection_error)) {
    ++rejected_count_;  return false;
}
if (observation.reprojection_error > config_.max_reprojection_error_px) {   // 5.0 px
    ++rejected_count_;  return false;
}
```

第三道在关联循环里逐块板算：

```cpp
for (int i = 0; i < config_.plate_count; ++i) {
    if (used_indices_[i]) continue;                       // 一帧一块板只配一条观测
    const Eigen::Vector3d h = observationModel(x_, i, ...);
    const Eigen::Vector3d residual = z - h;
    if (residual.norm() > config_.jump_gate_m) {          // 物理跳变闸 0.6 m
        continue;
    }
    const ObservationJacobian H = observationJacobian(x_, i, ...);
    const Eigen::Matrix3d S = H * P_ * H.transpose() + R;
    const double nis = residual.transpose() * S.inverse() * residual;
    if (!std::isfinite(nis)) continue;
    if (nis < best_nis) { best_nis = nis; best_index = i; best_h = h; best_H = H; }
}
```

三道闸各自的角色：

| 闸 | 量 | 门限 | 挡什么 |
|---|---|---|---|
| 重投影误差 | `observation.reprojection_error` | 5.0 px | 角点被拉开、灯条配错 |
| 物理跳变闸 | `‖z - h_i‖` | 0.6 m | 6 ms 内 0.6 m 相当于 100 m/s，不可能是真目标 |
| 统计闸 NIS | `r^T S^-1 r` | 11.34（卡方 3 自由度 99%） | 统计上不合理的观测 |

关联用**马氏距离**（NIS）而不是最近欧氏距离，因为侧板的横向不确定度大，欧氏距离会把它误配进来。`S = H P H^T + R` 里已经把不确定度算进去了。

**`residual.norm() > jump_gate_m` 用的是 `continue` 而不是直接拒绝整个观测**：一块板跳得远，不代表其他几块也跳，剩下的板还有机会配上。全都过不了才 `++rejected_count_; return false;`（`continue` 在 `:523`，全部落空的分支在 `:547-562`）。

三道闸的先后顺序就是上面这张表的顺序：重投影在检测层就筛掉，跳变闸在槽位循环里逐块判，NIS 只对最后选中的那个槽位判一次。所以一帧走到 NIS 闸之前，`best_nis` 就已经有了值：取的是槽位比较里赢下来的那条观测的 NIS。这个值在闸门之前写到 `frame_best_nis_`，再由 `buildPrediction` 原样搬进 `PredictionResult::best_nis`，每帧 `predict` 开头清成 -1。它补的是 `nis_mean` 的洞：均值只统计被接受的更新，一帧被统计闸拒了，均值原地不动，现场分不清是没观测还是被拒。`best_nis` 大于 `nis_gate` 就是明确的「这一帧被统计闸挡了」。

## 5.2 吃下观测：Joseph 形式

`ArmorEkf.cpp:568-577`，槽位记账在 `:609-615`（中间省略了 `armor_yaw` 标量更新与 `clampRadii`）：

```cpp
const Eigen::Vector3d residual = z - best_h;
const Eigen::Matrix3d S = best_H * P_ * best_H.transpose() + R;
const Eigen::Matrix<double, 10, 3> K = P_ * best_H.transpose() * S.inverse();
x_ += K * residual;
const Covariance identity_minus_kh = Covariance::Identity() - K * best_H;
P_ = identity_minus_kh * P_ * identity_minus_kh.transpose() +
     K * R * K.transpose();
x_[Yaw] = wrapToPi(x_[Yaw]);

used_indices_[best_index] = true;
index_target_id_[best_index] = observation.detection.target_id;
target_id_ = observation.detection.target_id;
updated_this_frame_ = true;
recordNis(best_nis);
```

协方差用 **Joseph 形式**：

```
P = (I - K H) P (I - K H)^T + K R K^T
```

比教科书那个 `(I - K H) P` 多两次矩阵乘，但数值上永远保持正定，长时间跑下来更稳。这是工程实现的常见选择，答辩可以直接按这个理由讲。

`index_target_id_[槽位]` 记住「这个槽位最近读到的数字」。选板之后回填 `armor_flag` 要用它（第八节）。`target_id_` 存最近一次观测的编号，只做兜底。

`recordNis` 把这次的 NIS 存进环形缓冲，供状态机的「最近 NIS 均值」门用（第六节）。

---

# 六、状态机：`TrackingState` 就是 `crc`

`TrackingState` 定义在 `AutoAimTypes.hpp`，数值故意选成 `NoTarget = 0, Stable = 1, Unstable = 2`。`applyPrediction`（`AppSupport.cpp:94`）直接搬进 `message.crc`，没有一处算校验和。所以三个值：

| 值 | 状态 | `fire_allowed` |
|---|---|---|
| 0 | `NoTarget` 无目标 | 必须假 |
| 1 | `Stable` 预测稳定 | 唯一可为真 |
| 2 | `Unstable` 有目标未稳定 | 必须假 |

## 6.1 每帧的四个计数器

`ArmorEkf.cpp:623-676` 的前半段：

```cpp
if (updated_this_frame_) {
    miss_count_ = 0;
    ++hit_streak_;
    if (std::isfinite(timestamp_seconds)) last_obs_ts_ = timestamp_seconds;
} else {
    ++miss_count_;
    hit_streak_ = 0;
}
updated_this_frame_ = false;

if (last_obs_ts_ > 0.0 && std::isfinite(timestamp_seconds) &&
    timestamp_seconds - last_obs_ts_ > config_.reset_timeout_s) {   // 1.5 s
    reset();
    return;
}
```

- `hit_streak_`：连续命中帧数。
- `miss_count_`：连续丢失帧数。
- `last_obs_ts_`：上次有效观测的墙钟时间。
- **长时间丢失重置**：距上次有效观测 > `reset_timeout_s`（1.5 s）就 `reset()` 回 `NoTarget`。1.5 s 内车已可能平移数米或转好几圈，旧态外推没有意义；这个门限远大于真实遮挡（几百毫秒），不会误触发短时遮挡的恢复。

## 6.2 短时丢失的滞回

```cpp
if (status_ == TrackingState::Stable && miss_count_ >= config_.stable_hold_misses) {  // 3
    status_ = TrackingState::Unstable;
}
```

丢 3 帧以内仍维持 `Stable`。165 fps 下单帧 6 ms，3 帧 18 ms，纯预测极准。

## 6.3 进入 `Stable` 的三个条件

```cpp
if (status_ == TrackingState::Unstable) {
    const int required = ever_stable_ ? config_.recover_frames : config_.stable_frames;
    if (hit_streak_ >= required && covarianceIsStable() &&
        recentNisMean() < config_.nis_mean_gate) {
        status_ = TrackingState::Stable;
        ever_stable_ = true;
    }
}
```

三条同时满足才转 `Stable`：

| 条件 | 门限 | 含义 |
|---|---|---|
| `hit_streak_ >= required` | 首次 15 帧 / 丢失后恢复 10 帧 | 连续命中够久（15 帧约 90 ms @165 fps） |
| `covarianceIsStable()` | 见下 | 协方差确实收缩了 |
| `recentNisMean() < 8.0` | 最近 5 次 NIS 均值 | 没有持续偏离模型 |

`covarianceIsStable`（:678-686）：

```cpp
const double center_std = std::sqrt(std::max(0.0, P_(Px, Px) + P_(Py, Py)));
const double radius_std = std::sqrt(std::max(0.0, P_(Radius1, Radius1) + P_(Radius2, Radius2)));
const double yaw_std    = std::sqrt(std::max(0.0, P_(Yaw, Yaw)));
return center_std < config_.stable_center_std_m &&    // 0.10 m
       radius_std < config_.stable_radius_std_m &&    // 0.05 m
       yaw_std    < config_.stable_yaw_std_rad;       // 0.10 rad
```

三点都取下限 0，防止浮点负零开根号出 NaN。

**「首次进 Stable」和「丢失后重新进」用不同帧数**：`ever_stable_` 一旦置真就一直是真，之后从 `Unstable` 回 `Stable` 只要 10 帧（`recover_frames`），因为这时模型已经证明过自己是可信的。

---

# 七、初始化 `tryInitialize`

`ArmorEkf.cpp:380-463`。这是整份代码里最需要关注的一段，也是曾经唯一出过问题的一段：
反推中心的方位角方向反了 180 度，2026-10-13 已修（提交 `d3a7ff3`），下面 7.2 讲的就是它。

## 7.1 三帧一致门

```cpp
if (init_used_this_frame_) {
    return false;                       // 同一帧的其余观测直接跳过
}
init_used_this_frame_ = true;

const Eigen::Vector3d z(observation.position_gimbal_m[0], ...);
if (!z.allFinite() || ... || observation.reprojection_error > 5.0) {
    ++rejected_count_; init_count_ = 0; return false;
}

const Eigen::Vector3d u = gimbalToWorld(gimbal_yaw) * z;   // 转到 W 系
if (init_count_ == 0) {
    init_mean_ = u; init_count_ = 1;
} else if ((u - init_mean_).norm() <= config_.init_consistency_m) {   // 0.3 m
    init_mean_ = (init_mean_ * init_count_ + u) / (init_count_ + 1);   // 滑动均值
    ++init_count_;
} else {
    init_mean_ = u; init_count_ = 1;    // 不接近就重新开始计数
}
if (init_count_ < config_.init_frames) {    // 3 帧
    return false;
}
```

**按帧计数，不按观测条数**：`init_used_this_frame_` 保证同一帧里若有几块板，只有第一条参与计数。否则两块板之间的间距（0.5 m）会顶掉 0.3 m 的一致门，计数永远凑不满。

三帧一致门的作用是防「把两个不同目标拼成一个中心」：连续三帧板心彼此相差 < 0.3 m 才认为看的是同一辆车。

## 7.2 反推中心

`ArmorEkf.cpp:436-447`（`plate_world` 就是 `init_mean_`）：

```cpp
x_.setZero();
x_[Yaw] = wrapToPi(std::atan2(-plate_world.y(), -plate_world.x()));
// 先夹半径再反推中心，否则滤波器一建立就与这条观测矛盾，下一帧起全被拒。
const double initial_radius =
    std::min(config_.max_radius_m, std::max(config_.min_radius_m,
                                            config_.initial_radius_m));
x_[Radius1] = initial_radius;
x_[Radius2] = initial_radius;
x_[Px] = plate_world.x() - initial_radius * std::cos(x_[Yaw]);
x_[Py] = plate_world.y() - initial_radius * std::sin(x_[Yaw]);
x_[Pz] = plate_world.z() - config_.plate_height_m[0];
```

逻辑是：把观测到的板心当作 0 号板，编号取 0（任意全局相位，不影响物理，后续随观测自动纠正），用它减掉一个半径向量得到中心。

```
theta_c = atan2(-板心y, -板心x)            板心【视线】的反方位角 = 中心指向板心的径向方位
中心     = 板心 - r * (cos theta_c, sin theta_c)
```

**那个负号是 2026-10-13 修掉的一个真 bug，答辩要能讲**（提交 `d3a7ff3`）。
这里要的是「板心相对**旋转中心**的方位角」（径向），而观测直接给的是「板心相对**世界原点**」的方位角（视线）。两者的关系由检测条件决定：能被检出的板必须**板面朝相机**，而板面法向就是径向，所以

```
r_hat 约等于 -p_hat          （r_hat: 中心指向板心的径向；p_hat: 原点指向板心的视线）
```

也就是**径向是视线的反方向**。改之前代码假设 `r_hat = p_hat`，方向整整差了 180 度。初始中心误差可以直接写出来（`delta` = 这块板被看到的斜角）：

| 假设 | 中心误差 | `delta` 在可见范围内（锥角 80 度） |
|---|---|---|
| 改前 `r_hat = p_hat` | `2r·cos(delta/2)` | `(0.383, 0.50]` m |
| 改后 `r_hat = -p_hat` | `2r·sin(delta/2)` | `[0, 0.321)` m |

`delta < 90°` 时 `sin(delta/2) < cos(delta/2)` 恒成立，所以**凡是能被检出的姿态，取反都严格更优** —— 这不是调参，是把一个符号改回来。

实测（合成旋转车体，`testrig/ekf_trace --direct`）：

| | frame 7 | frame 120 | frame 189 |
|---|---|---|---|
| 中心误差（改前） | 0.501 m | 0.180 m | 0.188 m |
| 中心误差（改后） | 0.021 m | 0.025 m | **0.003 m** |
| 内部半径（改前） | (0.250, 0.250) | **(-0.083, 0.177)** | (-0.083, 0.175) |
| 内部半径（改后） | (0.250, 0.250) | (0.232, 0.267) | (0.248, 0.252) |

改前第 120 帧的内部半径是**负的**，物理上不可能，是锁到错解的指纹；而且它在第 21 帧就报了 `Stable`，
改后推到第 164 帧（真正收敛之后）。完整推导见 `docs/EKF初始化调整方案.md`。

**几何退化**：可见的板永远在车的**近侧**（`r_hat` 与 `-p_hat` 夹角 ≤ 锥角），所以"看不见的板"和"看得见的板"给出的信息是两种，一份四板车在 80 度锥角下每帧最多同时看到 2 块、多数只有 1 块。这一条同时解释了**为什么这套初始化够用**（近侧视线方向是径向的良好近似）与**为什么 `r_hat = p_hat` 这种写法在合成夹具里能骗过测试**（夹具喂了背对相机的板，那种板恰好满足 `r_hat = p_hat`）。

## 7.3 建好之后

```cpp
P_ = initialCovariance();
status_ = TrackingState::Unstable;
target_id_ = observation.detection.target_id;
index_target_id_.fill(0);
index_target_id_[0] = observation.detection.target_id;
hit_streak_ = 0;
miss_count_ = 0;
ever_stable_ = false;
...
init_count_ = 0;
init_mean_.setZero();
return true;
```

初始化只把自己置成 `Unstable`，不直接进 `Stable`：还要经过第六节那 15 帧和协方差门。

---

# 八、输出 `buildPrediction`

`ArmorEkf.cpp:784-936`。这是把状态向量变成云台指令的一段，顺序上分六步。

## 8.1 取云台角、bias，算 `t0`

```cpp
const double gimbal_yaw   = std::isfinite(gimbal.yaw)   ? gimbal.yaw   : 0.0;
const double gimbal_pitch = std::isfinite(gimbal.pitch) ? gimbal.pitch : 0.0;
const double bias = std::isfinite(gimbal.prediction_bias_s) ? gimbal.prediction_bias_s : 0.0;

const double latency = smoothedLatency(latency_s);
const double t0 = std::clamp(latency + bias, 0.0, config_.max_prediction_s);
```

**bias 允许为负**（:794-795 有注释）：电控若想「少预测一点」会下发负值，单边截正会把它吃掉。总时长 `t0` 仍然截在 `[0, 0.5]` 里，异常值照样挡得住。

`smoothedLatency`（:709-728）把 `latency_s` 过 20 帧滑动均值，抑制单帧抖动。

## 8.2 弹道飞行时间：两次迭代解开循环

`ArmorEkf.cpp:806-822`：

```cpp
const Eigen::Matrix3d to_gimbal = worldToGimbal(gimbal_yaw);   // 飞行迭代、瞄准、中心差分共用

double flight_s = 0.0;
for (int iteration = 0; iteration < 2; ++iteration) {
    StateVector x_step = x_;
    propagate(x_step, t0 + flight_s);
    const int index = selectArmorIndex(x_step, gimbal_yaw, false);   // commit=false
    if (index < 0) break;
    const Eigen::Vector3d p_world = platePositionWorld(x_step, index);
    // 弹道从枪口量起。先在云台系里减掉枪口常向量，再取水平距离与高度差。
    const BallisticInputs inputs =
        ballisticInputsFromMuzzle(to_gimbal * p_world, config_.muzzle_offset_m);
    // 这一轮只取飞行时间，第三个参数是回退仰角，用不到。
    flight_s = solveBallistics(inputs.horizontal_m, inputs.vertical_m, 0.0).flight_time_s;
}
const double t_pred = std::clamp(t0 + flight_s, 0.0, config_.max_prediction_s);
```

飞行时间依赖距离，距离又依赖预测时长，所以迭代两次。`selectArmorIndex` 第三个参数传 `false`（`commit=false`）表示**只试算，不写 `aim_index_`**，否则迟滞状态会被试算值写脏。

### 8.2.1 弹道输入从枪口量起，减法做在云台系里

`solveBallistics` 的两个输入是**目标相对枪口**的水平距离与高度差，不是相对云台 P 轴原点。枪口 = 云台原点 + `(muzzle_offset_m, 0, 0)`，74.1 mm，来源是外参 handout 的「云台 P 轴到摩擦轮」。

**这个向量在云台系里是常量，在世界系里不是。** 相机、云台 P 轴、摩擦轮三者刚体固定在同一个俯仰层上，云台怎么转、俯仰怎么动，向量在 G 里恒为 `(0.0741, 0, 0)`。而 W 与 G 只差一个绕 z 的 yaw，同一个向量在 W 里是 `0.0741·(cos θg, −sin θg, 0)`，跟着 yaw 转。

所以在 `to_gimbal * p_world` 之后减，不是减 `p_world`：

| 目标在云台系正前方 5 m | yaw 0° | 30° | 60° | 90° |
|---|---|---|---|---|
| 在 W 里减常向量的水平距离误差 | 0 mm | 10.07 mm | 37.46 mm | 74.65 mm |

90 度时误差正好等于一整个偏移量，因为那时枪口在世界系里跑到了 `(0, −0.0741, 0)`。**在 G 里减与在 W 里减掉旋转后的那个向量逐位相同**：`M(θg) = Rz(−θg)` 只绕 z，`hypot(x, y)` 与 z 都不过 yaw。

实测影响分两半。闭式可算的两个：**飞行时间恒定 −3.088 ms**（`t = δ/v`，与距离无关）、**发射仰角差 0.631 mrad = 0.0362°**（`= g·δ/(2v²)`）。端到端由 `testMuzzleOffsetIsSubtractedInGimbalFrame` 打出来：同一场景只差这一项跑两遍，**指令仰角差 6.467e-04 rad**，对上闭式 6.310e-04，落在 ±20% 内。

**`distance_m` 没有跟着挪。** 它仍然是 `p_gimbal_aim.norm()`，留在云台 P 轴口径上，与 `pnp_distance_m` 同口径；它随偏移只漂 **0.3753 mm**，因为 `distance_m` 量的是预测瞄准点，而 `t_pred` 里已经含了飞行时间。串口协议头把 `distance` 和 `coo_x`/`coo_y` 排在一起，那两个是 P 轴系的量，所以保持原样。

这一项**只进输出层**，不进状态、不进观测，所以补它不用重扫 `Q`/`R`。

`solveBallistics`（`:73-93`）复用仓库里的 `Bullet_Model`（`armor/include/BulletModel.hpp`）：默认初速 24 m/s、`Kf = 0.08`、`gravity = 9.81`。它的坐标系就是 W 系。带空气阻力没有解析解，`Bullet_Model` 用 `do-while` 迭代最多 100 次、判据 `|z_diff| < 1e-4`。

它一次返回三样东西：

```cpp
struct BallisticSolution {
    double flight_time_s{0.0};     // 用来定 t_pred
    double launch_pitch_rad{0.0};  // 炮口要抬到多少度，见 8.4
    bool   converged{false};
};
```

**不收敛时回落到调用方传进来的几何仰角**，等价于不作补偿。这里有个静默陷阱：撞上 100 次迭代上限时 `beta` 仍然是个有限数、不报错，直接发出去就是一个没落到位的中途值。所以判据必须是 `|model.z_diff| < model.err_max`，不能只看 `beta` 是不是有限。宁可少补，也不能发一个假解。

## 8.3 选板与迟滞

`selectArmorIndex`（:734-782）：

```cpp
for (int i = 0; i < config_.plate_count; ++i) {
    const Eigen::Vector3d p_world = platePositionWorld(x, i);
    const double norm = p_world.norm();
    if (norm < 1e-6) continue;
    const double psi = x[Yaw] + i * config_.plate_spacing_rad;
    const Eigen::Vector3d normal(std::cos(psi), std::sin(psi), 0.0);
    if (normal.dot(-p_world / norm) < facing_threshold) continue;   // 剔侧板
    if (i == aim_index_) current_facing = true;
    const Eigen::Vector3d p_gimbal = to_gimbal * p_world;
    const double abs_yaw = std::fabs(gimbalAzimuth(p_gimbal));
    if (abs_yaw < best_abs_yaw) { best_abs_yaw = abs_yaw; best = i; }
}

if (aim_index_ >= 0 && current_facing && aim_index_ != best) {
    const Eigen::Vector3d p_current = to_gimbal * platePositionWorld(x, aim_index_);
    const double current_abs_yaw = std::fabs(gimbalAzimuth(p_current));
    if (current_abs_yaw - best_abs_yaw < config_.aim_switch_margin_rad) {   // 5 度
        best = aim_index_;
    }
}
```

三个要点：

1. **剔侧板**：板面法向（朝外）与「板心指向云台」的夹角超过 60 度就跳过。侧板看不见，不该拿来瞄。
2. **取 `|rel_yaw|` 最小的板**，也就是最正对的那块。
3. **迟滞**：当前瞄的板仍然可见，且新板只比它好不到 5 度，就**不切**。避免在两块板之间来回跳。

迟滞状态 `aim_index_` 只在 `commit=true` 时更新，也就是只在最后那次正式调用里。

## 8.4 `target_yaw/target_pitch`

`ArmorEkf.cpp:825-836`：

```cpp
StateVector x_pred = x_;
propagate(x_pred, t_pred);
const int aim_index = selectArmorIndex(x_pred, gimbal_yaw, true);
if (aim_index < 0) return PredictionResult{};
// to_gimbal 在 8.2 已经算过，这里直接用。
const Eigen::Vector3d p_world_aim = platePositionWorld(x_pred, aim_index);
const Eigen::Vector3d p_gimbal_aim = to_gimbal * p_world_aim;

// 与 main.cpp 里 pnp_yaw_rad / pnp_pitch_rad 同式：yaw 右为正，pitch 抬头为正。
const double rel_yaw = gimbalAzimuth(p_gimbal_aim);
const double rel_pitch = gimbalElevation(p_gimbal_aim);
```

**这两个式子在 `ArmorEkf.cpp:53-59` 集中定义，与 `main.cpp:548-550` 的 `pnp_yaw_rad/pnp_pitch_rad`、与 `armor_solve` 的 `armor_yaw` 逐字同式**：云台系是 前 / 左 / 上，所以

```
rel_yaw   = atan2(-y, x)                  右为正，正前方是 0
rel_pitch = atan2( z, hypot(x, y))        抬头为正，正前方是 0
```

yaw 取 `-y` 是因为云台系的 +y 朝**左**，取负才得到右为正。差别只是「单帧原始观测」对「滤波 + 预测」。

### yaw 输出绝对角，pitch 输出**弹道发射角**

输出（:915-919）：

```cpp
// 输出绝对云台指令，不是相对角。依据是框架的 setSafeOutput：
// 安全帧写成 target_yaw = message.yaw（保持当前），只有绝对角才自洽。
result.target_yaw = static_cast<float>(gimbal_yaw + rel_yaw);
// 下坠补偿在算法侧做，这里发的是弹道发射仰角，不是瞄准点的几何仰角。
result.target_pitch = static_cast<float>(command_pitch);
```

`command_pitch` 由 8.2 那个 `solveBallistics` 在瞄准点上再解一次得到（:840-845）：

```cpp
const BallisticInputs aim_inputs =
    ballisticInputsFromMuzzle(p_gimbal_aim, config_.muzzle_offset_m);
const double command_pitch =
    solveBallistics(aim_inputs.horizontal_m, aim_inputs.vertical_m,
                    gimbal_pitch + rel_pitch)
        .launch_pitch_rad;
```

**yaw 和 pitch 的构造方式不一样，这是故意的。**

yaw 是绕竖直轴的角度，跟重力无关，`gimbal_yaw + rel_yaw` 就是绝对方位角，直接发。

pitch 不一样。`rel_pitch` 是**目标在哪**（几何视线仰角 $\eta$），而炮口要抬到的是 $\beta$，因为弹丸要飞一段水平距离 $d$ 之后落下 $\frac{1}{2}gt^2$。对打到 $(d, h)$：

$$\tan\beta = \frac{h}{d} + \frac{g\,d}{2v^2\cos^2\beta}, \qquad \tan\eta = \frac{h}{d}$$

两式相减得 $\tan\beta - \tan\eta = \dfrac{g\,d}{2v^2\cos^2\beta} > 0$，**$\beta$ 恒大于 $\eta$**，只有 $g = K_f = 0$ 时相等。小角近似下差值就是

$$\beta - \eta \approx \frac{g\,d}{2v^2}$$

**只跟距离和弹速有关，一阶与高度无关。** 初速 24 m/s 时每米约 0.00852 rad：3 m → 1.46°、5 m → 2.44°、10 m → 4.88°。空气阻力让它再大一点，5 m 处约多 0.03°，随 $d^3$ 增长，短距离可以忽略。

这不是新引入的量，`Bullet_Model` 每次迭代都在算它，只是原先把收敛完的 `beta` 随局部对象丢掉了、只留了飞行时间。**接上它不引入任何新未知量**：$\beta$ 是 $\eta$ 与 $d$ 的函数，而 $d$、$h$ 在算 `p_world_aim` 时已经有了；唯一新增的依赖是弹速 $v$，而飞行时间那一步本来就在用同一个默认值 24 m/s。

**分工的结论来自外部**：下坠补偿由算法侧做（学长 2026-10-16），所以发的是 $\beta$，不是 $\eta$。旁证是仓库参考实现 `armor/src/Planner.cpp:124` 的 `result(1) = final_bullet_model.beta` 也是这么做的。`Bullet_Model` 的初速仍吃构造函数的硬编码 24 m/s、不是实测值，这条挂在问题清单 §2.3。

**顺带被改掉的一个真 bug**：开火门原来用 `|rel_pitch|` 判对准，那是「到几何瞄点」的偏差。命令既然改成 $\beta$，门也必须跟着改成对命令求差，否则门会在云台还差一整个补偿量时就放行 —— 5 m 处 2.44°，是 `fire_pitch_tolerance_rad = 0.02` 的 2.1 倍，落点低 `5·tan(2.44°) ≈ 21 cm`。见 8.6。

**为什么 yaw 输出绝对角**：框架的 `setSafeOutput`（`AppSupport.cpp:66-67`）把安全帧写成 `target_yaw = message.yaw`（保持当前，误差 0 等于不动）。只有 `target` 是绝对角时这句才自洽。这一点已按学长口径（问题清单 §1.2）定案，不再加开关。

（pitch 那边 `gimbal_pitch` 这一项恰好消失了，不是同一个理由：$\beta$ 本身就是相对水平面的绝对发射仰角，不需要再加云台当前角。回退分支除外，见 8.2。）

## 8.5 角速度：中心差分

`ArmorEkf.cpp:847-879`。在 `t_pred ± 5 ms` 各推一次，两个 `rel_yaw/rel_pitch` 相减除以窗口：

```cpp
const double yaw_velocity = wrapToPi(yaw_late - yaw_early) / dt_window;
const double pitch_velocity = (pitch_late - pitch_early) / dt_window;
```

`yaw` 用 `wrapToPi` 包一次差值，防止在 $\pi$ 附近跨号；`pitch` 不会跨，直接相减。

**`pitch` 的两端不是 `rel_pitch`，是补偿后的角**（:863-874）：

```cpp
// target_pitch 是补偿后的角，它的速率也必须来自补偿后的角，
// 否则同一个上报结构里两个量对不上。
const BallisticInputs early_inputs =
    ballisticInputsFromMuzzle(p_early, config_.muzzle_offset_m);
const double pitch_early =
    solveBallistics(early_inputs.horizontal_m, early_inputs.vertical_m,
                    gimbal_pitch + gimbalElevation(p_early))
        .launch_pitch_rad;
```

同一个 `PredictionResult` 里 `target_pitch` 是 $\beta$，它的速率就得是 $\dot\beta$。若这里仍用 $\eta$ 做差分，报给 Web Debug 的一条曲线是补偿后的角、它的导数是补偿前的角，答辩时对不上。

## 8.6 填 `PredictionResult` 与 `fire_allowed`

`ArmorEkf.cpp:881-931`。要点：

```cpp
result.state = status_;                                   // 就是 crc
const std::uint8_t aim_target_id = index_target_id_[aim_index];
result.target_id = aim_target_id != 0 ? aim_target_id : target_id_;

result.center_x_m = x_[Px];                               // 车体状态报在 W 系
result.center_y_m = x_[Py];
result.radius_1_m = std::max(0.0, x_[Radius1]);           // 负半径不往外报
result.radius_2_m = std::max(0.0, x_[Radius2]);
result.distance_m = p_gimbal_aim.norm();                  // 与 pnp_distance_m 同口径

// yaw 的指令是 gimbal_yaw + rel_yaw，误差就是 rel_yaw；pitch 的指令是
// command_pitch，误差是它与当前绝对仰角 gimbal_pitch 之差。两者口径一致。
const bool aligned =
    std::fabs(rel_yaw) < config_.fire_yaw_tolerance_rad &&                    // 0.02
    std::fabs(command_pitch - gimbal_pitch) < config_.fire_pitch_tolerance_rad; // 0.02
const bool spin_ok = std::fabs(x_[YawRate]) < config_.fire_spin_tolerance_rad_s; // 1.0
result.fire_allowed = status_ == TrackingState::Stable && aligned && spin_ok;
```

四个细节：

1. **`armor_flag` 描述「正瞄着的那块板」**，不是「最后看到的那块」。多板同帧时两者不是同一块，所以从 `index_target_id_[aim_index]` 取；纯预测到一块没直接观测过的板时才退回 `target_id_`。
2. **负半径不往外报**（`std::max(0.0, ...)`）。内部 `x_` 可能是负的（第七节那个符号错没修之前就会），但对外统一截成 0。这个截断本身会掩盖问题，也是那个 bug 靠日志看不出来、只能靠读内部状态才发现的原因，见第七节与第十二节。
3. **仰角那个门的被减数是 `command_pitch`，不是 `rel_pitch`**。门问的是「云台到位了没有」，而云台要去的是 $\beta$，所以拿 $\beta - $ 当前绝对仰角。用 `rel_pitch` 的话，门在云台还差一整个补偿量时就会放行。对比一下两边的结构：yaw 的指令 `gimbal_yaw + rel_yaw` 减去当前 `gimbal_yaw` 正好剩 `rel_yaw`，所以那一项没变；pitch 的指令 `command_pitch` 减当前 `gimbal_pitch`，写出来就是上面那行。**两项现在口径一致了**，都是「指令减当前」。这条 2026-10-16 与仰角指令同步改，见 8.4。
4. **`fire_allowed` 只在 `Stable` 且对准且车体别转太快时为真**。这是红线的第一道保险；第二道在 `applyPrediction`（`AppSupport.cpp:95-98`）。

最后一道防线：

```cpp
if (!predictionIsFinite(result)) {
    return PredictionResult{};
}
```

`predictionIsFinite`（`:97-110`）把每个字段都查一遍，还额外查了 `center_x_m * 1000.0F`（因为通信帧里米转毫米还要再乘一次 1000，这一步也可能溢出成非有限值）。任一非有限就返回全零，交给框架降级成安全帧。

---

# 九、`Config` 常量速查

`ArmorEkf.hpp:62-191`。按用途分组，改哪个先看这一列。

| 常量 | 值 | 用途 / 依据 |
|---|---|---|
| `plate_height_m` | `{0,0,0,0}` | 每块板相对中心的高度差，前哨站换 `{0,-0.102,0.102,0}` |
| `plate_count` | 4 | 车体周向板数，前哨站 3 |
| `plate_spacing_rad` | 90° | 相邻板夹角，前哨站 120° |
| `accel_noise_m_s2` | 6.0 | `Q` 的中心三轴加速度强度 |
| `angular_accel_noise_rad_s2` | 12.0 | `Q` 的车体角加速度强度 |
| `radius_drift_m` | 0.02 | 半径随机游走强度，2026-10-21 起被钳位压住，实际不起作用 |
| `radius_variance_floor_m2` | 9e-8 | 半径过程噪声下限，(0.3 mm)²，166 fps 下的主导项 |
| `lateral_noise_scale` / `base_m` | 0.002 / 0.005 | 观测横向噪声 `sigma = 0.002 d + 0.005` |
| `depth_noise_scale` / `base_m` | 0.010 / 0.010 | 观测深度噪声 `sigma = 0.010 d + 0.010` |
| `jump_gate_m` | 0.6 | 物理跳变闸 |
| `nis_gate` | 11.34 | 卡方 (3, 99%) 分位点 |
| `max_reprojection_error_px` | 5.0 | 单次观测重投影误差上限 |
| `init_frames` | 3 | 建滤波器前的连续一致帧数 |
| `init_consistency_m` | 0.3 | 初始化窗口内板心最大离散 |
| `stable_frames` | 15 | 首次进 Stable 需要的连续命中帧 |
| `recover_frames` | 10 | 丢失后回 Stable 需要的连续命中帧 |
| `stable_hold_misses` | 3 | 丢这么多帧以内仍维持 Stable |
| `reset_timeout_s` | 1.5 | 超过它没观测就重置 |
| `nis_window` | 5 | 最近 NIS 均值窗口 |
| `nis_mean_gate` | 8.0 | 进 Stable 的门 |
| `stable_center_std_m` | 0.10 | 位置标准差上限 |
| `stable_radius_std_m` | 0.05 | 半径标准差上限 |
| `stable_yaw_std_rad` | 0.10 | yaw 标准差上限 |
| `initial_radius_m` | 0.303 | 初始化半径，靶车 Video_1 两板心弦长 0.428 m 按方阵算出（2026-10-21）。2026-10-20 卷尺量得 27.3 cm，小 10.9%，判定为读数偏小 |
| `initial_radius_std_m` | 0.003 | 初始半径标准差，半径已钳成常量，这个数只保证协方差与钳位不矛盾 |
| `min_radius_m` / `max_radius_m` | 0.303 / 0.303 | 上下界相等，半径是常量而不是待估量（2026-10-21） |
| `muzzle_offset_m` | 0.0741 | 枪口相对云台 P 轴原点的前移量，弹道输入从枪口量起。来源外参 handout，**不是调参旋钮** |
| `max_step_s` | 0.5 | `predict` 的 `dt` 上限 |
| `max_latency_s` | 0.2 | 单帧延迟上限 |
| `latency_window` | 20 | 延迟滑动均值窗口 |
| `max_prediction_s` | 0.5 | 总预测时长上限 |
| `side_plate_angle_rad` | 60° | 剔侧板的夹角 |
| `aim_switch_margin_rad` | 5° | 选板迟滞 |
| `fire_yaw_tolerance_rad` | 0.02 | 开火对准容差 |
| `fire_pitch_tolerance_rad` | 0.02 | 同上 |
| `fire_spin_tolerance_rad_s` | 1.0 | 开火时车体自转上限 |

### 观测噪声 `R` 的数从哪来

`ArmorEkf.cpp:39-42` 记了依据：仓库已实测的角点抖动是 1.08 m 处单次重投影 0.70~0.79 px、同一角点两帧间最多动 1.41 px。按 `delta_px * d / f` 估横向约 `0.0006 * d`，再留约 3 倍余量取 `0.002 d + 0.005`。深度误差按同样比例放大取 `0.010 d + 0.010`。这里用的 `f` 是 1805 px，**量那批角点时的本机标定**，不是现在 `config/camera.yaml` 里车上那台的 1783 px；差 1% 量级，落在余量里，`scale` 不用改。

**两个数放到哪一维，跟着云台系走**（`observationNoise`，`ArmorEkf.cpp:289-301`）：

```
R = diag(depth², lateral², lateral²)
```

云台系是 前 / 左 / 上，**第 0 个分量正好沿视线**，所以深度进 `(0,0)`、横向进 `(1,1)` 与 `(2,2)`。
旧口径（x 右 / y 下 / z 前）下第 0 维是「右」，这一格是反的。**放错轴不会报错，只是让滤波器
该信的不信、该松的不松**，对拍差分也看不出来（`H` 是对的），只能靠人工核对。

---

# 十、函数清单（找代码用）

`ArmorEkf.hpp`：

| 行 | 内容 |
|---|---|
| `:43-45` | 三个 `using`：`StateVector` / `Covariance` / `ObservationJacobian` |
| `:48-60` | `Index` 枚举 |
| :62-191 | `Config` |
| :193-207 | 构造函数、`reset`、主循环四个方法 |
| :209-225 | 只读访问器（测试与 Web Debug 用） |
| :227-260 | 两个变换 + 观测模型/雅可比 + `wrapToPi` 的声明 |
| :262-289 | 私有辅助函数声明 |
| :291-335 | 成员状态（`x_`、`P_`、计数器、环形缓冲、`frame_best_nis_`、`aim_index_`） |

`ArmorEkf.cpp`：

| 行 | 函数 |
|---|---|
| `:53-59` | `gimbalAzimuth` / `gimbalElevation`（匿名命名空间，右正 / 抬头正） |
| `:67-93` | `BallisticSolution` + `solveBallistics`（匿名命名空间，包 `Bullet_Model`，出时间与发射仰角） |
| `:97-110` | `predictionIsFinite`（匿名命名空间） |
| `:122-130` | `gimbalToWorld` |
| `:132-134` | `worldToGimbal` |
| `:136-144` | `ballisticInputsFromMuzzle`（在云台系里减枪口偏移） |
| :146-152 | `wrapToPi` |
| :158-164 | `normalizeIndex` |
| :166-170 | `radiusIndex` |
| :172-184 | `observationModel` |
| :186-213 | `observationJacobian` |
| :215-222 | `platePositionWorld` |
| :228-237 | `clampRadii`（夹半径并平移车心） |
| :243-248 | `propagate` |
| :250-256 | `fillTransition` |
| :258-287 | `fillProcessNoise` |
| :289-301 | `observationNoise` |
| :303-319 | `initialCovariance` |
| :321-340 | `reset` |
| :342-364 | `predict`（开头把 `frame_best_nis_` 清成 -1） |
| :370-378 | `update`（分派器） |
| :380-403 | `tryInitialize` |
| :407-420 | `accumulateInitWindow`（攒初始化窗口） |
| :436-463 | `startTracking`（建状态与协方差） |
| :465-617 | `associateAndUpdate`（`:540-545` 记 `frame_best_nis_`） |
| :623-676 | `advanceStateMachine` |
| :678-686 | `covarianceIsStable` |
| :688-694 | `recordNis` |
| :696-707 | `recentNisMean` |
| :709-728 | `smoothedLatency` |
| :734-782 | `selectArmorIndex` |
| :784-936 | `buildPrediction`（`:900-902` 写出口 `best_nis`） |

---

# 十一、测试怎么钉住它

`tests/test_ekf_predict.cpp` 有 **22 条用例**，都不需要相机、不需要真车，跑 `ctest` 就行（`ctest` 里是 10 个可执行目标，`ekf_predict` 是其中之一）。每条都直接对应上面某一节的一个具体错误模式。

**夹具只喂真实可见的板**（很关键，答辩时值得主动讲）：仿真器推进一份真值车体，每帧按 80 度可见锥角判据筛出「板面朝相机」的板再喂进去，与 `testrig/ekf_trace` 的 `visiblePlates` 同源。改之前夹具是无条件发全部四块板，等于每帧凭空补两块背对相机的板 —— 真实检测器永远给不出这种输入，而且它恰好让「径向 = 视线」这条写反的假设成立，于是**方向写反的那一版在测试里反而全绿**。这条 bug 就是靠改夹具口径才暴露出来的。

| # | 用例 | 钉住什么 | 写错会怎样 |
|---|---|---|---|
| 1 | `testCoordinateTransforms` | `gimbalToWorld(0) = I`（两系同向，若又冒出一个 90° 置换就红）；三轴一一对应；`θg = 0.2` 时视线朝 −y，把「右为正」钉死；竖直轴不参与偏航；往返恒等；`wrapToPi` 边界 | G/W 变换符号写反，位置对方向错，静默 |
| 2 | `testPlateAzimuthConvention` | `theta_c` 左为正、`gimbal.yaw` 右为正，两者确实相反 | 日后有人把两个符号「统一」掉 |
| 3 | `testAnalyticJacobian` | 解析 `observationJacobian` 与中心差分逐元素差 < 1e-6，10 列 × 4 块板全查 | 手推雅可比错，滤波器收敛慢，闸门都看不出来 |
| 4 | `testNoiselessConvergence` | 观测由 `observationModel` 自己生成（**只喂可见板**），**2400 帧**后中心/速度/角速度/板位四项 < 1e-6，且进了 `Stable` | 单位错、`h`/`H` 错，现场怎么调都救不回来 |
| 5 | `testTwoPlateVehicleObservesRadiusTwo` | 两板对置（180°）时**单板轮换观测**下半径仍可辨识：**24000 帧**后中心/速度/角速度/半径和/半径差/板位六项 < 1e-4 | 间距仍按 90° 排，两块板落进同奇偶槽位，`r2` 永远没观测 |
| 6 | `testRadiusStaysWithinPhysicalBounds` | 先验给 0.9 m 与 0.01 m 两个离谱初值，初始化后半径都必须落在 `[min_radius_m, max_radius_m]`，且夹过之后模型仍复现刚吃下的那条观测（中心按夹过的半径反推）；再以 2 rad/s 自转跑 800 帧，全程不越界 | 没有边界时更新能把 `r2` 拉到 -0.14，回放里 r=0.05 / -0.03 就是它；只夹半径不反推中心，滤波器一建立就与观测矛盾，下一帧起全被拒 |
| 7 | `testGimbalRotationIsNotTargetMotion` | 目标在世界系里静止、云台原地扫，滤波器必须给零速度零角速度，中心到 1e-6 | 运动模型误建在云台系，把云台转动读成目标运动 |
| 8 | `testStateMachineTiming` | `crc` 序列走完整 `1 -> 2 -> 1 -> 0`；丢 2 帧仍稳定、第 3 帧掉；末次后 > 1.5 s 回 `NoTarget` | 状态机时序错，遮挡恢复或超时重置不对 |
| 9 | `testRejectedObservationsInflateCovarianceAndRecover` | 收敛后连续拒收达到 `reject_recovery_frames` 帧，协方差按 `reject_inflation_factor` 放大，之后一条偏差半米的正确观测必须吃得进去 | 拒收不改 P，收敛后的 P 又极小，马氏距离永远过不了 `nis_gate`：v4 回放里连续 115 帧 no-target、f=840~980 每帧全拒都是这条 |
| 10 | `testDivergedFilterReinitializesFromConsistentObservations` | 目标在两帧间瞬移到 3 m 外，全部观测都在跳变闸之外；连续丢 `reinit_after_misses` 帧后按新观测就地重建，10 帧内模型回到 0.1 m 内，之后继续正常吃观测 | 只剩 `reset_timeout_s` 一条出口，跳变闸外的目标永远等不回来（回放里 493 条观测被跳变闸挡掉，车在视野里却一直 no-target） |
| 11 | `testInitializationTakesOneObservationPerFrame` | 一帧里多发几块板仍能初始化成功 | 一致门被板间距顶掉，计数永远凑不满 |
| 12 | `testSinglePlateInitializationIsOnTheFarSide` | **只有一块板可见**时，初始化后中心误差 < 0.05 m、偏航误差 < 1e-6 rad，**且全程不许报 `Stable`** | 第七节那个 180 度符号错（改前这里是 0.5000 m 与 3.1416 rad） |
| 13 | `testArmorYawObservationPinsBodyYaw` | 板心固定不动，只靠位置观测分不出偏航；先连喂 50 帧错 1.5 rad 的 `armor_yaw` 确认 `armor_yaw_gate_rad` 一条都不吃，再喂正确值，偏航误差收到 0.03 rad 内 | 单板帧下偏航与车心切向分量简并，偏航和半径都由协方差编出来（v4 里 yaw 跑到 7~8 rad/s、半径塌到几厘米） |
| 14 | `testSameFrameDuplicatePlateIsRejectedByYaw` | 同一帧同一块板连喂两条，第二条按 `armor_yaw` 解出的槽位本帧已被占用，当场丢掉：`rejected_count` 加一、状态最大变化 < 1e-9 | 第二条被配到对面槽位（两板时相距 2r = 0.5 m，落在跳变闸 0.6 m 内），一次更新把两个半径拉到边界（v4 f=137，r 0.234 -> 0.100/0.100） |
| 15 | `testNegativePredictionBiasIsHonored` | 正 bias 与负 bias 把指令往相反方向推（乘积 < 0） | `bias` 被单边截正，电控想「少预测一点」时失效 |
| 16 | `testTrackingEvidenceFieldsReachPrediction` | 自转 2 rad/s 八百帧跑进 `Stable` 后，`aim_index` 确实换过板、`hit_streak` 与 `miss_count` 与 `rejected_count` 都从内部漏到了 `PredictionResult` 里 | 这三个诊断量漏不出来，Web Debug 上就看不见换板和遮挡 |
| 17 | `testBallisticDropCompensation` | 目标与云台同高时几何仰角恰为 0，**指令仰角就等于补偿量本身**，落在闭式 `g·d/(2v²)` 的 ±15% 内；再断言云台只转方位、仰角留平时开火门必须**拒绝**，云台跟到指令时必须**放行** | 只改指令不改门：命令抬了门没抬，5 m 外永远不开火；只改门不改指令：放行早 2.44°，落点低 21 cm |
| 18 | `testMuzzleOffsetIsSubtractedInGimbalFrame` | 正前方 5 m 处，弹道输入的水平距离恰少一个 `muzzle_offset_m`；同一场景只差这一项跑两遍，指令仰角差落在闭式 `g·δ/(2v²)` 的 ±20% 内；**并断言 `distance_m` 没有跟着挪** | 在 W 里减常向量而不是在云台系里减，误差随 yaw 走，yaw 90 度时正好错一个偏移量 |
| 19 | `testSafetyInvariants` | `crc != 1` 时 `fire_allowed` 恒假；空观测连调 100 帧不崩、输出有限；观测不够时不建目标 | 踩红线 |
| 20 | `testNisMeanReachesPrediction` | 窗口里没有样本时 `PredictionResult::nis_mean` 是负的哨兵值而不是无穷大；同一段真值只改观测扰动的大小跑两遍，扰动版必须真的比干净版大（实测 0.000000 对 3.7019）。**故意不断言「均值等于 3」**：夹具的观测由真值加扰动造，`R` 用的是滤波器自己那套，等于用同一把尺子量自己，`R` 整体系数填错均值照样是 3。这条标定只能上车做（`docs/上车手册.md` 第 4 节） | 哨兵换成 -1 会让「没有 NIS 历史」也满足进 Stable 的门；接成常数或错接到 `P` 的对角线都倒在这条上 |
| 21 | `testNisSentinelBeforeFirstUpdate` | 刚初始化、还没接受过更新这一刻：内部 `recentNisMean()` 必须是正无穷、不满足 Stable 门；出口 `PredictionResult::nis_mean` 必须是负值，且预测确实走到了 `NoTarget` 提前返回之外。这一时刻真车上必然出现（`tryInitialize` 把状态置成 Unstable 并清零 NIS 窗口） | 内部改成 -1 会让「没有 NIS 历史」直接过 Stable 门；出口漏转换会把无穷大一路打到曲线上 |
| 22 | `testBestNisReachesPrediction` | 收敛后干净帧 `best_nis` 在 1 以下（实测 0.000564），钉住它不是常量占位；把 0 号板横向推开 0.12 m，NIS 约 50.9，跨过 `nis_gate` 11.34 而远在 `jump_gate_m` 0.6 m 之内，更新被拒后出口 `best_nis` 必须大于 `nis_gate`，同一帧 `nis_mean` 与干净帧逐位相同；再走一帧无观测，出口必须是 -1，不许把上一帧的数留在那里 | 把取值放在闸门之后，被拒的帧仍然什么都留不下，这个字段等于白加；内部哨兵用正无穷再取 `max`，从第一帧起就永远钉在无穷，而 `best_nis` 与 `nis_mean` 不同，它每帧都要能回到「本帧没有数」 |

用例 4 的时序有个坑（文件里也写了）：真值必须先推进 `dt` 再让滤波器 `predict` 同样步长。顺序反了，滤波器会比真值落后一个 `v·dt`，误差恰好是 `0.5/165` 这种量级，看着像「没收敛」，其实是测试写错了。

用例 5 与 12 的两条几何断言是**规范无关**的，这一点答辩时要能解释：四块板等高时，「槽位整体挪两格、`theta_c` 加 pi」与「挪一格、`theta_c` 减 90°、`r1` 与 `r2` 互换」都是精确的等价变换，中心、速度、角速度、半径多重集在等价轨道上完全一样，只有槽位标签会换。所以断言只对这几个量提，板位按**集合**比（`worstPlateMatch`，逐块板找最近的真值板），不对标签提 —— 旧版那句「`r1` 一定大于 `r2`」是在断言一个约定，不是在断言物理。

**注意构建类型**：`tests/` 里是裸 `assert`，`-DNDEBUG`（Release）会把它们**全部编译掉**，`ctest` 照样全绿但那是假绿。所以**不要用 Release 构建**，要优化用 `-DCMAKE_CXX_FLAGS=-O2`，永远不动 `CMAKE_BUILD_TYPE`。

---

# 十二、已知问题与未闭环项

这一节是诚实清单，答辩时主动讲，比被追问好。

## 12.1 初始化在少板时会锁到错分支（**已修，提交 `d3a7ff3`**）

现象：离线跑 `ekf_trace`（仓库外自测工具，合成旋转车体）时，**每帧只喂真实看得见的 1 到 2 块板**，滤波器会收敛到错的地方：内部半径跑成负数（`r1 = -0.083`，真值 0.25），中心差 0.18 m，但仍报 `Stable`、拒绝计数为 0。**危险的不是「偏了 0.18 m」，是「偏了还报 Stable 并给火控指令」。**

用户答复（2026-10-13，问题清单 §2.7a）：同一辆敌车**最多同时看到 2 块，大部分时候只有 1 块** ⇒ 这个分支实车必发作。

根因与那一行的修法见第七节：初始化反推中心时方位角方向反了 180 度。这条线上走过并作废的路子（都实测过）：

| 路线 | 结论 |
|---|---|
| 要求建滤波器时至少两板可见 | 作废（实车大部分只有 1 块） |
| 约束半径（非负夹、夹区间、两半径合成一个、非均匀板高） | 试过、无效，一个都没把中心拉回真值 |
| 用 `armor_yaw` 定中心 | Δ 扫过：Δ ≤ 20° 大赚、Δ = 60~90° 退回现状、Δ = 45°/100° 反而更差；真帧上平板 PnP 双解的方位角中位差 52° ⇒ 不能无条件换。取反已把这块收益吃掉（frame 189 的 0.003 m 对 F 变体 Δ=0 的 0.001 m） |
| **取反（`atan2(-y,-x)`）** | **采用**。改后 frame 189 误差 0.003 m、半径 (0.248, 0.252)、`Stable` 从第 21 帧推到第 164 帧 |

**与"车转得慢"有关的未闭环项，三条**（同一根因，要不要处理待定，实测见 `docs/EKF初始化调整方案.md` 第五节与本文 12.6）：

- **慢转稳态残差**：0.4 rad/s 跑约 9 s，收敛到中心偏差 **0.052 m**、半径分成 0.309/0.177（真值 0.28/0.22），末态仍报 `Stable`；转速提到 2.0 rad/s 时同样几行收敛到 0.0000。这是「车转得慢 + 每次只看一块板」下的几何精度上限，不是实现错误。
- **完全不转的目标永远不报 `Stable`**：`omega = 0` 时 yaw 严格不可观测，`covarianceIsStable()` 的 `yaw_std < 0.10` 永远过不去（6000 帧与 20000 帧都没进过 `Stable`）。代价是不开火，偏安全侧。
- **慢转的目标同样进不了 `Stable`**：这条比上一条更值钱，因为实车不会是精确的 0。实测门槛约 `omega = 0.8 rad/s`，瓶颈全程是 `yaw_std`，不是半径也不是中心。转速落在 0.3 到 0.7 rad/s 的目标，`fire_allowed` 永远为假，哪怕瞄准几何完全正确。区分"信号弱"与"没有信号"的扫描见 12.6。

另外，「报 `Stable` 的那一刻中心还差 10 cm」这条缝仍在：27 组 `omega` × 初始朝向扫下来，改后最差 0.108 m（随后收敛到 0），旧版最差 0.582 m 且半径出现负值。要收这条缝的杠杆是**协方差阈值**，不是加"槽位奇偶门"—— 后者实测**免费但堵不住**（误差最大的那 9 行，两个奇偶的板从第 0 帧起就同时可见）。

## 12.2 `ArmorEkf` 读 `armor_yaw`（2026-10-20 起）

`ArmorEkf.cpp:9-22` 写明：观测是板心三维坐标，外加一条板面偏航角 `armor_yaw`。加它的原因是单板帧下偏航与车心切向分量简并，两个位置约束解不出四个未知量。这条观测按标量更新单独做，闸门是 `armor_yaw_gate_rad`，不记入 NIS 窗口；`armor_yaw` 与任务书「板心方位角」措辞相差的常值已并进换算式，所以它准不准直接影响偏航状态，不再是不影响跟踪的旁观量。

## 12.3 `gimbal.yaw` 符号未上车确认

`theta_c` 左为正、`gimbal.yaw` 右为正，是各按各的坐标系定义来的，滤波器自洽。但「电控下发的 `gimbal.yaw` 是不是右正」仓库里没有任何一处写明（问题清单 §1.5）。若反了，把 `gimbalToWorld` 换成它的转置即可，只有那一处。答错的影响是静默的：跟踪位置对、指令方向反。

## 12.4 EKF 之前的目标筛选（问题清单 §2.7b 仍开）

`main.cpp:534` 的 `armor_solve` 把所有同色检测解算完，`main.cpp:566` 把这一整帧观测交给 `ekf_predict`。筛选发生在适配器内部（`StudentTasks.cpp:212-222`）：先过 `TargetSelector`，只把选中那一簇的观测下标喂给滤波器。

筛选层的规则（`TargetSelector.hpp`）：按板心世界系单链接聚簇，链接距离 `cluster_link_m` 0.55 m、簇直径上限 `cluster_max_extent_m` 0.75 m，编号相同的一对在合并顺序上提前 `id_agreement_bias_m` 0.10 m；簇内打分按距离、目标类型、画面偏置三项加权（0.35 / 0.35 / 0.30，按总和归一化）。跨帧认同看新簇心与上次选中簇心的距离：编号相同比 `identity_match_m` 1.20 m，对不上或读不出时先加 `identity_sticker_penalty_m` 0.65 m 再比，等价于只认 0.55 m。挑战者要高出 `switch_margin` 0.08 分并连续 `switch_confirm_frames` 3 帧才切换；切换时 `switched` 置位，适配器只在滤波器本来就没有目标时才 `filter.reset()`，`Stable` 下换簇保留状态。目标整帧没观测时保持 `coast_timeout_s` 0.5 s。

`ArmorEkf` 本身仍然只认一辆车、一套四槽关联器；把两辆车同时灌进去会在槽位上互相抢。这与 12.1 是两回事，`ekf_trace` 也没有建这个模。
**仍开**：2.7b 问的是整帧通常有几辆敌车，这个问题没有答复，所以筛选层的参数全部来自 v4 回放标定，实车场景（哨兵与步兵同时在视野）没有验证过。已知遗留是 `identity_match_m` 1.20 m 偏宽。

## 12.5 弹速是默认值

用仓库默认初速 24 m/s（`BulletModel.hpp`）。这不只影响飞行时间，也**直接影响补偿量**：`beta - eta` 约等于 `g·d/(2v²)`，与 v 的平方成反比。5 m 处 v = 20 时是 3.57°、23 → 2.69°、24 → 2.47°、25 → 2.27°。上车要确认实际弹速（问题清单 §2.3）。

## 12.6 只看得到一块板时的「半径」与「进 Stable 的三条闸」（2026-10-14 实测）

要回答的问题：实车大部分时间只看得见一块装甲板，`ekf_radius_1_m` / `ekf_radius_2_m` 会不会有一个永远是假数？

工具 `testrig/radius_parity_probe.cpp`。它**直接驱动 `ArmorEkf`**（不隔 `ekf_predict`），所以能打出协方差对角线；可见性判据与 `ekf_trace.cpp:69-85` 逐字同一套（板面法向 = 径向，与"板心指向相机"的夹角小于锥角才算看得见）。每条汇总同时打 `covarianceIsStable()` 的三条闸，否则只知道"没到 `Stable`"，不知道被谁挡住。

### 12.6.1 「一次只看得到一块」是常态，不是边角

默认锥角 80 度（半角 40 度），标准车相邻板法向差 90 度，40 + 40 < 90 ⇒ **任何车体朝向下最多一块板可见**。

这句话的适用边界用纯几何量过（`testrig/plate_visibility_geometry.cpp`，扫 20000 个朝向、距离 1 到 8 m）：锥角 92 度以内，任何距离都只有一块；锥角到 94 度时，8 m 处有 1.5% 的朝向能同时看到两块。所以「最多一块」是**锥角不超过约 93 度**时的性质，不是无条件成立。

### 12.6.2 两个半径只要目标在转，都会收敛

**这一小节是 2026-10-14 放开半径时测的，2026-10-21 起半径已钳成常量（`min_radius_m = max_radius_m = 0.303`），下面这张表描述的现象不再发生，保留它只为说明「半径可辨识」需要什么条件。**

`H.col(radiusIndex(i))`（`ArmorEkf.cpp:211`）是雅可比里唯一非零的半径列，一条观测只修得动一个奇偶的半径。但可见槽位随车体旋转轮换，两个半径都会陆续拿到观测：

| 目标自转 omega (rad/s) | 末态 r1 / r2 | 真值 | 末态中心误差 |
|---|---|---|---|
| 0 | 0.2525 / 0.2474 | 0.25 / 0.25 | 0.0037 m |
| 0.3 | 0.2470 / 0.2530 | 0.25 / 0.25 | 0.0043 m |
| 0.7 | 0.2502 / 0.2497 | 0.25 / 0.25 | 0.0003 m |
| 2.0 | 0.2498 / 0.2503 | 0.25 / 0.25 | 0.0005 m |

只有 `omega = 0`（目标完全不转）时另一个半径才拿不到观测：它是初始化先验（现行配置 0.303，上表那一跑为 0.25）靠交叉协方差漂出来的数，sd 单调增长。此时 `ekf_radius_2_m` 报的是先验，不是测量。

### 12.6.3 卡住 `Stable` 的是 yaw 闸，不是半径闸

`covarianceIsStable()` 三条闸逐档实测（每档 1600 或 6000 帧）：

| omega (rad/s) | 报过 Stable | 半径 sd 合计 < 0.05 | 中心 sd 合计 < 0.10 | yaw sd < 0.10 |
|---|---|---|---|---|
| 0 | 没有 | 0.0873 涨到 0.1175 | 0.0898 涨到 0.1256 | 0.2495 涨到 0.3399 |
| 0.3 | 没有 | 0.0535 / 0.0623 平台 | 0.0581 / 0.0768 过 | 0.1608 / 0.1939 卡 |
| 0.4 | 没有 | 0.0549 卡 | 0.0697 过 | 0.1591 卡 |
| 0.5 | 没有 | 0.0514 卡 | 0.0848 过 | 0.2175 卡 |
| 0.7 | 没有 | 0.0394 过 | 0.0583 过 | 0.1188 卡 |
| 1.0 | 是（首帧 704） | 0.0343 过 | 0.0611 过 | 0.1750，见 12.6.5 |
| 1.5 | 是（首帧 213） | 0.0264 过 | 0.0385 过 | 0.0761 过 |
| 2.0 | 是（首帧 157） | 0.0223 过 | 0.0340 过 | 0.0720 过 |

⇒ 进 `Stable` 的门槛约在 `omega = 0.8 rad/s`，瓶颈全程是 yaw。慢于此的目标 `fire_allowed` 永远为假，**哪怕瞄准几何完全正确**（`omega = 0.6` 时中心误差只有 0.0006 m）。

### 12.6.4 这条闸是「几何墙」还是「调参」：扫角加速噪声定论

yaw 为什么不可观测：相机只看得到板，看不到车心。车心由「板心减 r 乘径向」反推，于是"车心沿半径方向平移一点"和"车体转一点"给出同一个板心观测；`omega = 0` 时两者精确等价，yaw 严格不可观测。

要区分"信号弱"与"没有信号"，扫 `angular_accel_noise_rad_s2`（当前默认 12.0）。

`omega = 0.3 rad/s`，4000 帧：

| 角加速噪声 alpha | 报过 Stable | yaw sd | 半径 sd | 中心 sd |
|---|---|---|---|---|
| 12 | 没有 | 0.7099 | 0.0702 | 0.1815 |
| 6 | 没有 | 0.4329 | 0.0627 | 0.1228 |
| 3 | 没有 | 0.2774 | 0.0557 | 0.0910 |
| 1 | 是（首帧 1296） | 0.1692 | 0.0495 | 0.0709 |

⇒ `omega > 0` 时 yaw 方差的平台高度大致按 alpha 的平方根缩放，是**可调的量**，不是几何硬限。

`omega = 0`，20000 帧：

| alpha | 末态 yaw sd | 末态半径 sd 合计 |
|---|---|---|
| 1 | 0.5521 | 0.1928 |
| 0.01 | 0.5658 | 0.2000 |

⇒ `omega = 0` 时怎么降 alpha 都不收敛：alpha 差 100 倍，yaw sd 反而略大，方差单调增长。这就是真不可观测，增长由**平移过程噪声**（`accel_noise_m_s2 = 6.0`）经交叉协方差灌进 yaw 的零空间，与 alpha 无关。

结论：**慢转（omega > 0）是调参问题，完全不转是几何问题。** 0.8 rad/s 这条门可以用 alpha 挪，代价是目标真正改变转速时滤波器跟得慢。

### 12.6.5 顺带查出：`covarianceIsStable()` 是单向门

它只在 Unstable 到 Stable 那一步被查（`ArmorEkf.cpp:668-675`）。进了 `Stable` 之后，只有连续丢 `stable_hold_misses = 3` 帧能降回 `Unstable`。所以 **`Stable` 可以和远超门槛的协方差共存**：`omega = 1.0` 第 704 帧进入 `Stable`，跑到第 2999 帧状态仍是 `Stable`，而末态 yaw sd 已经 0.1750，是门槛 0.10 的 3.5 倍。是刻意留的滞回还是漏了回退闸，需要定。

### 12.6.6 待定：整状态协方差闸换成「瞄准点自身的不确定度闸」

现状这条闸判的是整个状态的协方差，「yaw 不可观测」会连带把「其实能打的目标」也否掉。真正决定打不打的是**瞄准点**的不确定度，不是车体 yaw 的不确定度。不过 12.6.4 说明 alpha 本身也是这条闸的杠杆，所以在动代码之前要先定 alpha 的取舍，再定阈值。**未改。**

### 12.6.7 夹具口径提示（又踩一次）

探针第一版按 `--slots 0` / `--slots 0,1` 这类**固定槽位**发板，喂进去的是背对相机、真实系统检不到的那些板，于是得到"r2 永远冻在 0.25、sd 只涨"的错结论。改成按可见性发板后，旋转目标上两个半径都正常收敛。教训同 `99-踩坑记录`：**夹具口径错会把结论判反**。`--allplates`（旧口径）更离谱：16 帧就报 `Stable`，半径却收敛成 0.11 / 0.38、中心误差 0.18 m。

### 12.7 74.1 mm 枪口偏移（2026-10-16 **已接上**）

欠账已清。`solveBallistics` 的两个输入改成从枪口量起，见 `ballisticInputsFromMuzzle`（`ArmorEkf.cpp:136-144`）。三件事值得单独记：

1. **它在云台系里是常向量。** 相机、云台 P 轴、摩擦轮刚体固定在同一个俯仰层上，向量恒为 `(muzzle_offset_m, 0, 0)`，不随 yaw 也不随 pitch 变。先前记的「它跟着 pitch 走、抬 10 度差 12.87 mm」是错的：那 12.87 mm 是**地面系里枪口的位置**，不是「相对 P 轴的向量」。
2. **减法必须做在 G 里。** 在 W 里直接减常向量，误差随 yaw 走：5 m 处 yaw 30° 差 10.07 mm、60° 差 37.46 mm、90° 差 74.65 mm。90° 时正好错一个偏移量。在 G 里减与在 W 里减掉旋转后的那个向量逐位相同，因为两系只差一个绕 z 的 yaw。
3. **`distance_m` 没有跟着挪**，仍留在云台 P 轴口径上，与 `pnp_distance_m` 同口径。

它只进输出层，不进状态也不进观测，所以补它不牵动 `Q`/`R`。数量级见 8.2.1：指令仰角差 6.467e-04 rad，是**弹速误差的 1/5 都不到**——弹速 24 m/s 没人实测过，错 1 m/s 在 5 m 处就是 0.203°。所以枪口偏移不是精度瓶颈，弹速才是（§12.5）。

---

# 十三、复现与自测工具

`ArmorEkf` 不依赖相机和串口，可以被仓库外的工具直接构造驱动。现有工具在 `~/workplace/rm_autoaim_testrig/`（不进提交）：

| 工具 | 用途 |
|---|---|
| `ekf_trace.cpp` | 合成旋转车体喂 `ekf_predict`，打状态机与预测时间线。开关 `--direct`（额外打内部 `x`/`P`）、`--allplates`、`--initall`、`--asym`、`--theta45`、`--cone N` |
| `ekf_trace_yaw.cpp` | 同上，但给观测填 `armor_yaw`。开关 `--yawerr N`（给方位角加固定偏差） |
| `yaw_probe.cpp` | 对真实整图量每块板的 `armor_yaw` 与平板 PnP 双解差 |
| `radius_parity_probe.cpp` | 直接驱动 `ArmorEkf`（不隔 `ekf_predict`）打协方差对角线，只喂看得见的板。开关 `--omega` / `--cone` / `--alpha` / `--frames` / `--allplates`。回答「只看到一块板时两个半径怎么办」与「进 Stable 的门槛」，见 12.6 |
| `plate_visibility_geometry.cpp` | 纯几何，不含滤波器：扫车体朝向统计 4 板标准车同时可见几块，给出「最多一块」成立的锥角上限 |

**`ekf_trace` 必须真睡够帧间隔**，没有 `--fast`：`ekf_predict` 内部用 `steady_clock` 现取「现在」减帧时间戳当处理延迟，伪造时间戳会让它量出 0.2 秒（clamp 上限），预测时长整条时间线都是错的。

```bash
bash ~/workplace/rm_autoaim_testrig/build_ekf_trace.sh
./ekf_trace --direct
```

自测工具不提交。作业要的是结果。
