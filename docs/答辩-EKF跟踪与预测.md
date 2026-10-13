# EKF 跟踪与预测（§3.6）代码解释

对应评分项：**EKF 跟踪与预测（20 分）**
对应规格：`轨迹预测说明.md`「旋转车体 EKF 预测要求」、`docs/大作业任务书.md` §3.6
代码位置：
- `student/include/ArmorEkf.hpp`（228 行，类声明、状态下标、全部可调常量）
- `student/src/ArmorEkf.cpp`（697 行，全部实现）
- `student/src/StudentTasks.cpp:186-212`（`ekf_predict`，跨帧状态的适配器）
- `tests/test_ekf_predict.cpp`（无噪收敛、雅可比对拍、状态机时序、安全不变量）

本文按「一帧数据怎么流过这几百行」的顺序讲，每个小节末尾给行号锚点。答辩要独立解释实现时，照本文的节序讲即可。

---

# 零、这一层在整个流水线里的位置

`main.cpp:465` 那一行的前后：

```
armor_solve 的产物                          EKF 的输入
observations: vector<ArmorPose>
   每块板: position_gimbal_m  云台系板心 (m)   <- 唯一被 EKF 读取的量
           position_camera_m  相机系板心 (m)   <- 调试用，EKF 不看
           armor_yaw          板面法向方位角    <- EKF 不看
           reprojection_error                  <- 当闸门用
           detection.target_id / size          <- 只记进槽位表，不参与关联
                    |
                    v
ekf_predict(observations, gimbal, time_stamp)     StudentTasks.cpp:186
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
2. **观测量只用板心的三维坐标**，不用任何角度。这样绕开 `armor_yaw` 的语义歧义（它实现成「板面法向方位角」，任务书三处措辞指向「板心方位角」，两者差 90 度）。
3. **`ArmorEkf` 只依赖 Eigen 和标准库**，不碰 OpenCV，也不碰串口。所以它可以被 `ekf_trace` 这类仓库外工具直接构造、逐帧驱动，不需要相机。

---

# 一、三个坐标系与唯一的那个变换

## 1.1 三个系

| 记号 | 各轴 | 原点 | 从哪来 |
|---|---|---|---|
| **G 云台系** | x 右、y 下、z 前 | 云台旋转中心 | `armor_solve` 输出的 `position_gimbal_m` 就在这个系里 |
| **W 世界系** | x 前、y 左、z 上 | 云台旋转中心在初始时刻的位置 | 与 `armor/include/BulletModel.hpp` 的弹道模型一致 |
| **物体系**（装甲板） | x 板右、y 板下、z 指板背 | 板心 | `armor_solve` 里 `plateObjectPoints` |

两个偏航角的转正方向**故意不同**，这不是笔误（`ArmorEkf.cpp:15-22` 有原话）：

- `theta_c` 是 W 系里的方位角，从 +x 转向 +y 为正，而 W 的 +y 朝左，所以 `theta_c` **左为正**。
- `gimbal.yaw` 经 G 系（x 右、y 下、z 前）定义成 **右为正**。

两者各自只在观测模型里用一次，滤波器因此仍然自洽，不需要强行统一。与电控实际符号对不对，仓库里没有任何一处写明，只能上车确认。

## 1.2 唯一的变换函数

`ArmorEkf.cpp:78-90`：

```cpp
Eigen::Matrix3d ArmorEkf::gimbalToWorld(double gimbal_yaw) {
    const double s = std::sin(gimbal_yaw);
    const double c = std::cos(gimbal_yaw);
    Eigen::Matrix3d m;
    m << -s,   0.0,  c,
         -c,   0.0, -s,
          0.0, -1.0, 0.0;
    return m;
}

Eigen::Matrix3d ArmorEkf::worldToGimbal(double gimbal_yaw) {
    return gimbalToWorld(gimbal_yaw).transpose();
}
```

**整个 `ArmorEkf.cpp` 里，G 与 W 之间的坐标变换只出现这一个函数。** 三个调用点：

| 调用点 | 方向 | 用途 |
|---|---|---|
| `tryInitialize:319` | G -> W | 把初始化的几帧观测搬进 W 系比一致性 |
| `observationModel:129` | W -> G | 预测观测值，与 `position_gimbal_m` 同系才能相减 |
| `selectArmorIndex:542,563` / `buildPrediction:632,650-651` | W -> G | 把 W 系的瞄准点搬回当前云台系算 `rel_yaw/rel_pitch` |

## 1.3 这个矩阵怎么来的

拆成两步，能一步步手算核对。

**第一步，取 `gimbal_yaw = 0`（云台正对前方），只做轴改名。**

`gimbalToWorld(0)` 把 `s=0, c=1` 代进去：

```
M(0) = [ 0  0  1 ]
       [-1  0  0 ]
       [ 0 -1  0 ]
```

读这个矩阵用「第 j 列 = 第 j 个基向量的像」：`M(0)` 的第 1 列 `(0,-1,0)` 就是 G 的 +x（右）在 W 系里的坐标，第 2 列第 3 列同理。注意 Eigen 的 `<<` 是**按行填**，写代码和读矩阵要对齐。

直接代三个基向量核对（`d > 0`）。取 `p_G = (0, 0, d)`，即 G 系正前方 d 米：

```
M(0) * (0,0,d)^T = d * (第 3 列) = d * (1, 0, 0) = (d, 0, 0)     W 的 +x = 前
```

正前方在 W 里就是前方，对。取 `p_G = (d, 0, 0)`（G 系右侧 d 米）：

```
M(0) * (d,0,0)^T = d * (第 1 列) = d * (0, -1, 0) = (0, -d, 0)   W 的 -y = 右
```

G 的右边是 W 的右边，对。取 `p_G = (0, d, 0)`（G 系下方 d 米）：

```
M(0) * (0,d,0)^T = d * (第 2 列) = d * (0, 0, -1) = (0, 0, -d)   W 的 -z = 下
```

G 的下方是 W 的下方，对。所以 `M(0)` 就是一个纯轴改名的置换矩阵，行列式 +1。这三条正是 `tests/test_ekf_predict.cpp:102` 的坐标变换用例钉住的三个断言。

**第二步，把云台的转动叠上去。**

云台绕**自己的竖直轴**转 `gimbal_yaw`。关键是先分清「在哪个系里转」：

- 物理上的竖直方向，在 W 系里是 z 轴，在 G 系里是 y 轴。云台是绕 G 系的 y 轴转的。
- 所以正确的写法是：先在 G 系里做一个绕 y 的转动 `R_y(θg)`，再用 `M(0)` 把结果搬到 W 系。写成一个乘积：

```
p_W = M(0) · R_y(θg) · p_G       即   M(θg) = M(0) · R_y(θg)
```

拿 `s, c` 代进去展开，正好得到代码里那个矩阵（第一行 `[-s, 0, c]`、第二行 `[-c, 0, -s]`、第三行 `[0, -1, 0]`）。

**这里最容易看错的是把 `R_y` 写成 `R_z`。** 记法：「云台绕自己的 y 轴转，而 G 系的 y 是向下的竖直轴」，转动写在 G 系那一侧，`M(0)` 只负责最后改一次名。

自检锚点（答辩时背这两条就够）：

```
θg = 0,  正前方板 p_G = (0,0,d)  ->  p_W = (d,0,0)
det(M(θg)) = +1（永远是纯转动，不翻转不缩放）
worldToGimbal =  transpose（正交阵的逆就是转置，不用求逆）
```

`tests/test_ekf_predict.cpp:114-119` 把 `θg = 0.2` 时的方向也钉住了：转完的视线方向再绕回云台角，方位角必须精确复原。

## 1.4 角度包裹

`wrapToPi`（`:92-98`）：

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

`ArmorEkf.hpp:38-55`：

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

`ekf_predict` 这个适配器（`StudentTasks.cpp:186-212`）每帧固定做四件事：

```cpp
filter.predict(dt);                                 // :206
for (const ArmorPose& observation : observations) {
    filter.update(observation, static_cast<double>(gimbal.yaw));   // :208
}
filter.advanceStateMachine(timestamp_seconds);      // :210
return filter.buildPrediction(gimbal, latency_s);   // :211
```

| 调用 | 什么时候调 | 没观测时 |
|---|---|---|
| `predict(dt)` | 每帧，包括没有观测的帧 | 照常做纯预测 |
| `update(obs, yaw)` | 每有一条观测调一次 | 不调 |
| `advanceStateMachine(ts)` | 每帧一次 | 照常调（它负责计 miss、超时重置） |
| `buildPrediction(gimbal, latency)` | 每帧一次 | 状态 `NoTarget` 时返回全零 |

`predict` 和 `advanceStateMachine` 必须每帧都调，这是状态机能走对的前提。`ekf_predict` 的注释和 `ArmorEkf.hpp:127` 都写明了这一点。

## 3.1 `predict(dt)`：先验

`ArmorEkf.cpp:263-284`：

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

`ArmorEkf.cpp:170-183`：

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

`ArmorEkf.cpp:185-214`。按随机加速度模型离散化，每个「位置 + 速度」轴对共享一个分块：

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

给半径一点自由度是为了防止滤波器把它「冻死」在初值上；`radius_variance_floor_m2 = 1e-6`（即 1 mm 的平方）是下限，`dt` 很小时靠它兜底。

### 这些数从哪来

| 常量 | 值 | 依据 |
|---|---|---|
| `accel_noise_m_s2` | 6.0 | RM 步兵 0 到 20 m/s 约 1 到 2 秒，瞬时机动约 10 m/s²，取一半（`ArmorEkf.cpp:24-25`） |
| `angular_accel_noise_rad_s2` | 12.0 | 车能快速起旋停旋，同量级取整 |
| `radius_drift_m` | 0.2 | 半径近乎恒定，只给随机游走强度 0.2 m/√s |

调参顺序按博客 §3.7：**先验单位与模型 -> 固定 `R` 调 `Q` -> 最后才谈动态噪声**。`Q` 大则跟得快但抖，小则平滑但滞后。

## 3.4 `P0`

`ArmorEkf.cpp:227-240`：

```cpp
P(Px, Px) = 1.0;      P(Py, Py) = 1.0;      P(Pz, Pz) = 1.0;      // 位置 (m^2)
P(Vx, Vx) = 25.0;     P(Vy, Vy) = 25.0;     P(Vz, Vz) = 9.0;      // 速度 (m/s)^2
P(Yaw, Yaw) = 0.25;   P(YawRate, YawRate) = 9.0;                  // rad^2 / (rad/s)^2
P(Radius1, Radius1) = 0.0225;  P(Radius2, Radius2) = 0.0225;      // m^2
```

`P0` 给大一点，等于说「初始化时先信观测」，随后靠 `P_ = F P F^T + Q` 和更新自然收缩。位置给 1 m² 是「中心可能离这块板一米左右」的量级；速度给 5 m/s 的标准差是「不知道车怎么动」。

---

# 四、观测模型 `h` 与雅可比 `H`

## 4.1 `h`

`ArmorEkf.cpp:118-130`：

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

`ArmorEkf.cpp:132-155`。非零的四个列：

```cpp
H.col(Px) << -s, 0.0, c;          // 平移
H.col(Py) << -c, 0.0, -s;
H.col(Pz) << 0.0, -1.0, 0.0;
H.col(Yaw) << -radius * std::cos(psi_plus_gimbal), 0.0,
              -radius * std::sin(psi_plus_gimbal);
H.col(radiusIndex(i)) << -std::sin(psi_plus_gimbal), 0.0,
                          std::cos(psi_plus_gimbal);
```

其中 `psi_plus_gimbal = psi_i + gimbal_yaw`。对 `vx/vy/vz/omega` 全为 0（观测模型里没有它们）。

推导要点：`h = M^T u`，`M^T` 是常量，所以

```
dh/dpx = M^T * (1,0,0)^T = M 的第 1 行 = (-s, 0, c)      <- 与代码逐元素一致
dh/dpy = M^T * (0,1,0)^T = (-c, 0, -s)
dh/dpz = M^T * (0,0,1)^T = (0, -1, 0)
```

`u` 对 `theta_c` 的偏导是 `r_i * (-sin psi_i, cos psi_i, 0)`，左乘 `M^T` 后就是上面 `H.col(Yaw)` 那个式子。半径那一列同理，而且**只有该板对应的那一列非零**（另一列已在 `H = Zero()` 里置零）。

**手推雅可比唯一的保险是解析对拍差分**，`tests/test_ekf_predict.cpp` 有一条用例逐元素比 `observationJacobian` 与有限差分，误差 < 1e-6。这个函数写错的话，滤波器会收敛慢或不收敛，而重投影误差之类的闸门都看不出来，是静默错误。

## 4.3 关联用的 `radiusIndex` 与槽位

`ArmorEkf.cpp:104-116`：

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

`update`（`:290-298`）是分派器：

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

`ArmorEkf.cpp:363-438`。进来先过两道「观测本身合不合法」的闸：

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

**`residual.norm() > jump_gate_m` 用的是 `continue` 而不是直接拒绝整个观测**：一块板跳得远，不代表其他几块也跳，剩下的板还有机会配上。全都过不了才 `++rejected_count_; return false;`（`:410-417`）。

## 5.2 吃下观测：Joseph 形式

`ArmorEkf.cpp:419-437`：

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

`ArmorEkf.cpp:444-483` 的前半段：

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

`covarianceIsStable`（`:485-493`）：

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

`ArmorEkf.cpp:300-361`。这是整份代码里最需要关注的一段，也是目前唯一已知会出问题的一段。

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

`ArmorEkf.cpp:335-343`：

```cpp
x_.setZero();
x_[Yaw] = wrapToPi(std::atan2(init_mean_.y(), init_mean_.x()));
x_[Radius1] = config_.initial_radius_m;    // 0.25
x_[Radius2] = config_.initial_radius_m;
x_[Px] = init_mean_.x() - config_.initial_radius_m * std::cos(x_[Yaw]);
x_[Py] = init_mean_.y() - config_.initial_radius_m * std::sin(x_[Yaw]);
x_[Pz] = init_mean_.z() - config_.plate_height_m[0];
```

逻辑是：把观测到的板心当作 0 号板，编号取 0（任意全局相位，不影响物理，后续随观测自动纠正），用它减掉一个半径向量得到中心。

```
theta_c = atan2(板心y, 板心x)              板心相对【世界原点】的方位角
中心     = 板心 - r * (cos theta_c, sin theta_c)
```

**这里有一个已知的近似**：第二个量本该是「板心相对**旋转中心**的方位角」，代码取的是「板心相对**世界原点**」的方位角，两者只有车心恰在世界原点时才相等。板装在前方时，视线方向与径向方向近似重合，所以误差不大；但随着车转过来，两者的夹角可以变得很大。这是当前初始化在少板时锁错分支的一个来源，详见 `docs/上车前问题清单.md` §2.7。

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

`ArmorEkf.cpp:592-697`。这是把状态向量变成云台指令的一段，顺序上分六步。

## 8.1 取云台角、bias，算 `t0`

```cpp
const double gimbal_yaw   = std::isfinite(gimbal.yaw)   ? gimbal.yaw   : 0.0;
const double gimbal_pitch = std::isfinite(gimbal.pitch) ? gimbal.pitch : 0.0;
const double bias = std::isfinite(gimbal.prediction_bias_s) ? gimbal.prediction_bias_s : 0.0;

const double latency = smoothedLatency(latency_s);
const double t0 = std::clamp(latency + bias, 0.0, config_.max_prediction_s);
```

**bias 允许为负**（`:602-603` 有注释）：电控若想「少预测一点」会下发负值，单边截正会把它吃掉。总时长 `t0` 仍然截在 `[0, 0.5]` 里，异常值照样挡得住。

`smoothedLatency`（`:516-535`）把 `latency_s` 过 20 帧滑动均值，抑制单帧抖动。

## 8.2 弹道飞行时间：两次迭代解开循环

`ArmorEkf.cpp:611-624`：

```cpp
double flight_s = 0.0;
for (int iteration = 0; iteration < 2; ++iteration) {
    StateVector x_step = x_;
    propagate(x_step, t0 + flight_s);
    const int index = selectArmorIndex(x_step, gimbal_yaw, false);   // commit=false
    if (index < 0) break;
    const Eigen::Vector3d p_world = platePositionWorld(x_step, index);
    flight_s = flightTimeSeconds(std::hypot(p_world.x(), p_world.y()), p_world.z());
}
const double t_pred = std::clamp(t0 + flight_s, 0.0, config_.max_prediction_s);
```

飞行时间依赖距离，距离又依赖预测时长，所以迭代两次。`selectArmorIndex` 第三个参数传 `false`（`commit=false`）表示**只试算，不写 `aim_index_`**，否则迟滞状态会被试算值写脏。

`flightTimeSeconds`（`:40-51`）复用仓库里的 `Bullet_Model`（`armor/include/BulletModel.hpp`）：默认初速 24 m/s、`Kf = 0.08`、`gravity = 9.81`。它的坐标系就是 W 系。

## 8.3 选板与迟滞

`selectArmorIndex`（`:541-590`）：

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
    const double abs_yaw = std::fabs(std::atan2(p_gimbal.x(), p_gimbal.z()));
    if (abs_yaw < best_abs_yaw) { best_abs_yaw = abs_yaw; best = i; }
}

if (aim_index_ >= 0 && current_facing && aim_index_ != best) {
    const Eigen::Vector3d p_current = to_gimbal * platePositionWorld(x, aim_index_);
    const double current_abs_yaw = std::fabs(std::atan2(p_current.x(), p_current.z()));
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

`ArmorEkf.cpp:626-640`：

```cpp
StateVector x_pred = x_;
propagate(x_pred, t_pred);
const int aim_index = selectArmorIndex(x_pred, gimbal_yaw, true);
if (aim_index < 0) return PredictionResult{};
const Eigen::Matrix3d to_gimbal = worldToGimbal(gimbal_yaw);
const Eigen::Vector3d p_gimbal_aim = to_gimbal * platePositionWorld(x_pred, aim_index);

const double rel_yaw = std::atan2(p_gimbal_aim.x(), p_gimbal_aim.z());
const double rel_pitch = -std::atan2(p_gimbal_aim.y(),
                                     std::hypot(p_gimbal_aim.x(), p_gimbal_aim.z()));
```

**这两个式子与 `main.cpp:442-444` 的 `pnp_yaw_rad/pnp_pitch_rad` 逐字同式**：yaw 右为正（`atan2(x,z)`），pitch 抬头为正（因为 G 系 y 向下，取负才得到抬头为正）。差别只是「单帧原始观测」对「滤波 + 预测」。

输出绝对角（`:680-683`）：

```cpp
result.target_yaw   = gimbal_yaw   + rel_yaw;
result.target_pitch = gimbal_pitch + rel_pitch;
```

**为什么输出绝对角**：框架的 `setSafeOutput`（`AppSupport.cpp:66-67`）把安全帧写成 `target_yaw = message.yaw`（保持当前，误差 0 等于不动）。只有 `target` 是绝对角时这句才自洽。这一点已按学长口径（问题清单 §1.2）定案，不再加开关。

## 8.5 角速度：中心差分

`ArmorEkf.cpp:642-662`。在 `t_pred ± 5 ms` 各推一次，两个 `rel_yaw/rel_pitch` 相减除以窗口：

```cpp
const double yaw_velocity = wrapToPi(yaw_late - yaw_early) / dt_window;
const double pitch_velocity = (pitch_late - pitch_early) / dt_window;
```

`yaw` 用 `wrapToPi` 包一次差值，防止在 $\pi$ 附近跨号；`pitch` 不会跨，直接相减。

## 8.6 填 `PredictionResult` 与 `fire_allowed`

`ArmorEkf.cpp:664-696`。要点：

```cpp
result.state = status_;                                   // 就是 crc
const std::uint8_t aim_target_id = index_target_id_[aim_index];
result.target_id = aim_target_id != 0 ? aim_target_id : target_id_;

result.center_x_m = x_[Px];                               // 车体状态报在 W 系
result.center_y_m = x_[Py];
result.radius_1_m = std::max(0.0, x_[Radius1]);           // 负半径不往外报
result.radius_2_m = std::max(0.0, x_[Radius2]);
result.distance_m = p_gimbal_aim.norm();                  // 与 pnp_distance_m 同口径

const bool aligned = std::fabs(rel_yaw)   < config_.fire_yaw_tolerance_rad &&    // 0.02
                     std::fabs(rel_pitch) < config_.fire_pitch_tolerance_rad;    // 0.02
const bool spin_ok = std::fabs(x_[YawRate]) < config_.fire_spin_tolerance_rad_s; // 1.0
result.fire_allowed = status_ == TrackingState::Stable && aligned && spin_ok;
```

三个细节：

1. **`armor_flag` 描述「正瞄着的那块板」**，不是「最后看到的那块」。多板同帧时两者不是同一块，所以从 `index_target_id_[aim_index]` 取；纯预测到一块没直接观测过的板时才退回 `target_id_`。
2. **负半径不往外报**（`std::max(0.0, ...)`）。内部 `x_` 可能是负的（初始化锁错分支时会），但对外统一截成 0。这个截断本身会掩盖问题，见第十二节。
3. **`fire_allowed` 只在 `Stable` 且对准且车体别转太快时为真**。这是红线的第一道保险；第二道在 `applyPrediction`（`AppSupport.cpp:95-98`）。

最后一道防线：

```cpp
if (!predictionIsFinite(result)) {
    return PredictionResult{};
}
```

`predictionIsFinite`（`:55-66`）把每个字段都查一遍，还额外查了 `center_x_m * 1000.0F`（因为通信帧里米转毫米还要再乘一次 1000，这一步也可能溢出成非有限值）。任一非有限就返回全零，交给框架降级成安全帧。

---

# 九、`Config` 常量速查

`ArmorEkf.hpp:57-115`。按用途分组，改哪个先看这一列。

| 常量 | 值 | 用途 / 依据 |
|---|---|---|
| `plate_height_m` | `{0,0,0,0}` | 每块板相对中心的高度差，前哨站换 `{0,-0.102,0.102,0}` |
| `plate_count` | 4 | 车体周向板数，前哨站 3 |
| `plate_spacing_rad` | 90° | 相邻板夹角，前哨站 120° |
| `accel_noise_m_s2` | 6.0 | `Q` 的中心三轴加速度强度 |
| `angular_accel_noise_rad_s2` | 12.0 | `Q` 的车体角加速度强度 |
| `radius_drift_m` | 0.2 | 半径随机游走强度 |
| `radius_variance_floor_m2` | 1e-6 | 半径过程噪声下限，(1 mm)² |
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
| `initial_radius_m` | 0.25 | 初始化假定的半径 |
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

`ArmorEkf.cpp:27-30` 记了依据：仓库已实测的角点抖动是 1.08 m 处单次重投影 0.70~0.79 px、同一角点两帧间最多动 1.41 px。按 `delta_px * d / f`（`f` 约 1805 px）估横向约 `0.0006 * d`，再留约 3 倍余量取 `0.002 d + 0.005`。深度误差按同样比例放大取 `0.010 d + 0.010`。

---

# 十、函数清单（找代码用）

`ArmorEkf.hpp`：

| 行 | 内容 |
|---|---|
| `:38-40` | 三个 `using`：`StateVector` / `Covariance` / `ObservationJacobian` |
| `:43-55` | `Index` 枚举 |
| `:57-115` | `Config` |
| `:117-131` | 构造函数、`reset`、主循环四个方法 |
| `:134-142` | 只读访问器（测试与 Web Debug 用） |
| `:149-164` | 两个变换 + 观测模型/雅可比 + `wrapToPi` 的声明 |
| `:168-187` | 私有辅助函数声明 |
| `:189-226` | 成员状态（`x_`、`P_`、计数器、环形缓冲、`aim_index_`） |

`ArmorEkf.cpp`：

| 行 | 函数 |
|---|---|
| `:40-51` | `flightTimeSeconds`（匿名命名空间，包 `Bullet_Model`） |
| `:55-66` | `predictionIsFinite`（匿名命名空间） |
| `:78-90` | `gimbalToWorld` / `worldToGimbal` |
| `:92-98` | `wrapToPi` |
| `:104-116` | `normalizeIndex` / `radiusIndex` |
| `:118-130` | `observationModel` |
| `:132-155` | `observationJacobian` |
| `:157-164` | `platePositionWorld` |
| `:170-183` | `propagate` / `fillTransition` |
| `:185-214` | `fillProcessNoise` |
| `:216-225` | `observationNoise` |
| `:227-240` | `initialCovariance` |
| `:242-261` | `reset` |
| `:263-284` | `predict` |
| `:290-298` | `update`（分派器） |
| `:300-361` | `tryInitialize` |
| `:363-438` | `associateAndUpdate` |
| `:444-483` | `advanceStateMachine` |
| `:485-493` | `covarianceIsStable` |
| `:495-514` | `recordNis` / `recentNisMean` |
| `:516-535` | `smoothedLatency` |
| `:541-590` | `selectArmorIndex` |
| `:592-697` | `buildPrediction` |

---

# 十一、测试怎么钉住它

`tests/test_ekf_predict.cpp` 有 **10 条用例**，都不需要相机、不需要真车，跑 `ctest` 就行。每条都直接对应上面某一节的一个具体错误模式。

| # | 用例 | 钉住什么 | 写错会怎样 |
|---|---|---|---|
| 1 | `testCoordinateTransforms` | `det(gimbalToWorld(0)) = +1`；三个基向量的像；`θg = 0.2` 时方位角复原；往返恒等；`wrapToPi` 边界 | G/W 变换符号写反，位置对方向错，静默 |
| 2 | `testPlateAzimuthConvention` | `theta_c` 左为正、`gimbal.yaw` 右为正，两者确实相反 | 日后有人把两个符号「统一」掉 |
| 3 | `testAnalyticJacobian` | 解析 `observationJacobian` 与中心差分逐元素差 < 1e-6，10 列 × 4 块板全查 | 手推雅可比错，滤波器收敛慢，闸门都看不出来 |
| 4 | `testNoiselessConvergence` | 观测由 `observationModel` 自己生成，**600 帧**后 `state - 真值` 逐分量 < 1e-6，且中途进了 `Stable` | 单位错、`h`/`H` 错，现场怎么调都救不回来 |
| 5 | `testTwoPlateVehicleObservesRadiusTwo` | 两块板间距 180° 时能把 `r1 > r2` 辨识出来（差值 0.04 以内） | 间距仍按 90° 排，两块板落进同奇偶槽位，`r2` 永远没观测 |
| 6 | `testGimbalRotationIsNotTargetMotion` | 目标在世界系里静止、云台原地扫，滤波器必须给零速度零角速度，中心到 1e-6 | 运动模型误建在云台系，把云台转动读成目标运动 |
| 7 | `testStateMachineTiming` | `crc` 序列走完整 `1 -> 2 -> 1 -> 0`；丢 2 帧仍稳定、第 3 帧掉；末次后 > 1.5 s 回 `NoTarget` | 状态机时序错，遮挡恢复或超时重置不对 |
| 8 | `testInitializationTakesOneObservationPerFrame` | 一帧里多发几块板仍能初始化成功 | 一致门被板间距顶掉，计数永远凑不满 |
| 9 | `testNegativePredictionBiasIsHonored` | 正 bias 与负 bias 把指令往相反方向推（乘积 < 0） | `bias` 被单边截正，电控想「少预测一点」时失效 |
| 10 | `testSafetyInvariants` | `crc != 1` 时 `fire_allowed` 恒假；空观测连调 100 帧不崩、输出有限；观测不够时不建目标 | 踩红线 |

用例 4 的时序有个坑（文件里也写了）：真值必须先推进 `dt` 再让滤波器 `predict` 同样步长。顺序反了，滤波器会比真值落后一个 `v·dt`，误差恰好是 `0.5/165` 这种量级，看着像「没收敛」，其实是测试写错了。

**注意构建类型**：`tests/` 里是裸 `assert`，`-DNDEBUG`（Release）会把它们**全部编译掉**，`ctest` 照样全绿但那是假绿。所以**不要用 Release 构建**，要优化用 `-DCMAKE_CXX_FLAGS=-O2`，永远不动 `CMAKE_BUILD_TYPE`。

---

# 十二、已知问题与未闭环项

这一节是诚实清单，答辩时主动讲，比被追问好。

## 12.1 初始化在少板时会锁到错分支（最要紧）

现象：离线跑 `ekf_trace`（仓库外自测工具，合成旋转车体）时，**每帧只喂真实看得见的 1 到 2 块板**，滤波器会收敛到错的地方：内部半径跑成负数（`r1 = -0.083`，真值 0.25），中心差 0.18 m，但仍报 `Stable`、拒绝计数为 0。**危险的不是「偏了 0.18 m」，是「偏了还报 Stable 并给火控指令」。**

- 根因是**初始化那 3 帧**，不是半径状态：约束半径（非负夹、夹区间、两半径合成一个）全都试过无效，只有把开头的帧喂成四块板才拉得回真值。
- 用户答复（2026-10-13，问题清单 §2.7a）：同一辆敌车**最多同时看到 2 块，大部分时候只有 1 块**。所以这个分支实车必发作。

修法路线（已实测，见 `docs/上车前问题清单.md` §2.7 与仓库外证据目录）：

| 路线 | 结论 |
|---|---|
| 要求建滤波器时至少两板可见 | 作废（实车大部分只有 1 块） |
| 约束半径 | 试过、无效 |
| 用 `armor_yaw` 定中心 | Δ 扫过：Δ ≤ 20° 大赚、Δ = 60~90° 退回现状、Δ = 45°/100° 反而更差 ⇒ 不能无条件换 |
| **兜底：可见板数长期 ≤ 1 且半径/偏航标准差没收敛时不许报 `Stable`** | 推荐先做，不依赖 `armor_yaw` |
| 只在两个 PnP 解的方位角接近时才用 `armor_yaw` | 需要 `armor_solve` 多给一个字段 |

**当前 `ArmorEkf.cpp` 一个字没改**，等路线定下来。

## 12.2 `ArmorEkf` 不读 `armor_yaw`（设计如此）

`ArmorEkf.cpp:9-13` 写明：观测量只用板心三维坐标，绕开 `armor_yaw` 的语义歧义。所以这个量准不准不影响跟踪。它影响的是 Web Debug 的解释和实验报告怎么写（问题清单 §1.4 存疑）。

## 12.3 `gimbal.yaw` 符号未上车确认

`theta_c` 左为正、`gimbal.yaw` 右为正，是各按各的坐标系定义来的，滤波器自洽。但「电控下发的 `gimbal.yaw` 是不是右正」仓库里没有任何一处写明（问题清单 §1.5）。若反了，把 `gimbalToWorld` 换成它的转置即可，只有那一处。答错的影响是静默的：跟踪位置对、指令方向反。

## 12.4 EKF 之前没有目标筛选（问题清单 §2.7b，未答）

`main.cpp:428` 的 `armor_solve` 把所有同色检测解算完，`main.cpp:465` 整体喂给 `ekf_predict`，EKF 之前**没有目标筛选**。一帧里若有两辆敌车，八块板会一起进同一个「单目标 4 槽」模型。这与 12.1 是两回事，`ekf_trace` 也没有建这个模。

## 12.5 弹速是默认值

用仓库默认初速 24 m/s（`BulletModel.hpp`）。弹道飞行时间对初速极敏感，远距离提前量直接受影响。上车要确认实际弹速（问题清单 §2.3）。

---

# 十三、复现与自测工具

`ArmorEkf` 不依赖相机和串口，可以被仓库外的工具直接构造驱动。现有工具在 `~/workplace/rm_autoaim_testrig/`（不进提交）：

| 工具 | 用途 |
|---|---|
| `ekf_trace.cpp` | 合成旋转车体喂 `ekf_predict`，打状态机与预测时间线。开关 `--direct`（额外打内部 `x`/`P`）、`--allplates`、`--initall`、`--asym`、`--theta45`、`--cone N` |
| `ekf_trace_yaw.cpp` | 同上，但给观测填 `armor_yaw`。开关 `--yawerr N`（给方位角加固定偏差） |
| `yaw_probe.cpp` | 对真实整图量每块板的 `armor_yaw` 与平板 PnP 双解差 |

**`ekf_trace` 必须真睡够帧间隔**，没有 `--fast`：`ekf_predict` 内部用 `steady_clock` 现取「现在」减帧时间戳当处理延迟，伪造时间戳会让它量出 0.2 秒（clamp 上限），预测时长整条时间线都是错的。

```bash
bash ~/workplace/rm_autoaim_testrig/build_ekf_trace.sh
./ekf_trace --direct
```

自测工具不提交。作业要的是结果。
