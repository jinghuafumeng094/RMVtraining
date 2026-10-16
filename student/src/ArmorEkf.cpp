#include "ArmorEkf.hpp"

#include "BulletModel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

// 观测量只用板心的三维坐标，不用任何角度。
//
// 这样绕开了 armor_yaw 的语义歧义：它实现成"板面法向的偏航角"
// （见 StudentTasks.cpp 的 armor_solve），而任务书三处措辞指的是"板心方位角"。
// 两者差 90 度，当作观测量会把整个滤波带偏。三维板心没有这个问题。
//
// 符号约定：两个偏航角转正的方向并不相同，这不是笔误。
// theta_c 是 W 系（x 前、y 左）里的方位角，从 +x 转向 +y 为正，而 +y 朝左，
// 所以 theta_c 左为正。gimbal.yaw 定义成右为正。
// 两者各自只在一个地方用一次，滤波器因此仍然自洽，不需要强行统一。
//
// G 云台系与 W 世界系轴向相同（前 / 左 / 上），只差绕竖直轴的偏航角。依据是
// 学长 2026-10-15「云台系和世界系一样」，见 docs/上车前问题清单.md 第 1.1 节。
// tests/test_ekf_predict.cpp 把两处都钉住了：坐标变换用例钉 gimbalToWorld 的
// 方向，无噪收敛用例逐分量比对真值（真值的 psi 就用左为正），把 theta_c 钉住。
// 但"是否符合电控下发的符号"仓库里没有任何一处写明，只能上车确认。若发现
// gimbal 方向反了，把 gimbalToWorld 换成它的转置即可，只有这一处。
//
// 过程噪声 sigma_a = 6 m/s^2：RM 步兵 0 到 20 m/s 约 1 到 2 秒，
// 瞬时机动约 10 m/s^2，取一半。角加速度 12 rad/s^2 同理。
//
// 观测噪声来自仓库已实测的角点抖动：1.08 m 处单次重投影 0.70~0.79 px、
// 同一角点两帧间最多动 1.41 px。按 delta_px * d / f 估横向约 0.0006 * d，
// 再留约 3 倍余量取 0.002 * d + 0.005，深度按同比例放大取 0.010 * d + 0.010。
// 那批 px 量于 2026-10-11，f 约 1805 px 是当时本机标定；车上那台约 1783 px。

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kNisRingSize = 8;
constexpr int kLatencyRingSize = 32;

// 云台系（前 / 左 / 上）里，一个点相对云台的两个视线角。
// 这两个式子与 main.cpp 的 pnp_yaw_rad / pnp_pitch_rad 同式，与 armor_solve 的
// armor_yaw 同口径：右为正、抬头为正，正前方是 0。集中在这里，别在调用处散抄。
double gimbalAzimuth(const Eigen::Vector3d& p) {
    return std::atan2(-p.y(), p.x());
}

double gimbalElevation(const Eigen::Vector3d& p) {
    return std::atan2(p.z(), std::hypot(p.x(), p.y()));
}

// 弹道解：飞完 (horizontal, vertical) 要多久，以及炮口要抬到多少度。
// beta 是绝对发射仰角，高于水平面，恒大于瞄准点的几何仰角，差值就是重力与
// 空气阻力造成的下坠补偿。复用仓库里的 Bullet_Model
// (armor/include/BulletModel.hpp)，它的坐标系就是 W（前 x、左 y、上 z），
// 默认初速 24 m/s。不收敛时 launch_pitch_rad 回落到调用方给的几何仰角，
// 此时等价于不作补偿；宁可少补，也不能把一个没落到位的中途值发出去。
struct BallisticSolution {
    double flight_time_s{0.0};
    double launch_pitch_rad{0.0};
    bool converged{false};
};

BallisticSolution solveBallistics(double horizontal_m, double vertical_m,
                                  double geometric_pitch_rad) {
    BallisticSolution solution;
    solution.launch_pitch_rad = geometric_pitch_rad;
    if (!std::isfinite(horizontal_m) || !std::isfinite(vertical_m)) {
        return solution;
    }
    Bullet_Model model(static_cast<float>(std::max(0.0, horizontal_m)),
                       static_cast<float>(vertical_m));
    const float time = model.bullet_iteration_cal();
    if (!std::isfinite(time) || time <= 0.0F) {
        return solution;
    }
    solution.flight_time_s = static_cast<double>(time);
    // 撞上 100 次迭代上限时 z_diff 仍然很大，此时 beta 是没落到位的中途值。
    if (std::isfinite(model.beta) && std::fabs(model.z_diff) < model.err_max) {
        solution.launch_pitch_rad = static_cast<double>(model.beta);
        solution.converged = true;
    }
    return solution;
}

// 通信帧里是 float，米转毫米还要再乘一次 1000，两处都得是有限值，
// 否则框架会把它降级成安全帧，不如在这里就先判掉。
bool predictionIsFinite(const PredictionResult& p) {
    return std::isfinite(p.center_x_m) && std::isfinite(p.center_y_m) &&
           std::isfinite(p.center_z_m) &&
           std::isfinite(p.center_x_m * 1000.0F) &&
           std::isfinite(p.center_y_m * 1000.0F) &&
           std::isfinite(p.center_velocity_x_m_s) &&
           std::isfinite(p.center_velocity_y_m_s) &&
           std::isfinite(p.center_velocity_z_m_s) && std::isfinite(p.body_yaw) &&
           std::isfinite(p.body_yaw_velocity) && std::isfinite(p.radius_1_m) &&
           std::isfinite(p.radius_2_m) && std::isfinite(p.distance_m) &&
           std::isfinite(p.target_yaw) && std::isfinite(p.target_pitch) &&
           std::isfinite(p.target_yaw_velocity) &&
           std::isfinite(p.target_pitch_velocity);
}

}  // namespace

ArmorEkf::ArmorEkf() = default;

ArmorEkf::ArmorEkf(const Config& config) : config_(config) {}

// ---------------------------------------------------------------------------
// 坐标系变换
// ---------------------------------------------------------------------------

Eigen::Matrix3d ArmorEkf::gimbalToWorld(double gimbal_yaw) {
    const double s = std::sin(gimbal_yaw);
    const double c = std::cos(gimbal_yaw);
    Eigen::Matrix3d m;
    m << c, s, 0.0,
         -s, c, 0.0,
         0.0, 0.0, 1.0;
    return m;
}

Eigen::Matrix3d ArmorEkf::worldToGimbal(double gimbal_yaw) {
    return gimbalToWorld(gimbal_yaw).transpose();
}

ArmorEkf::BallisticInputs ArmorEkf::ballisticInputsFromMuzzle(
    const Eigen::Vector3d& p_gimbal, double muzzle_offset_m) {
    const Eigen::Vector3d from_muzzle =
        p_gimbal - Eigen::Vector3d(muzzle_offset_m, 0.0, 0.0);
    BallisticInputs inputs;
    inputs.horizontal_m = std::hypot(from_muzzle.x(), from_muzzle.y());
    inputs.vertical_m = from_muzzle.z();
    return inputs;
}

double ArmorEkf::wrapToPi(double angle) {
    double wrapped = std::fmod(angle + kPi, 2.0 * kPi);
    if (wrapped <= 0.0) {
        wrapped += 2.0 * kPi;
    }
    return wrapped - kPi;
}

// ---------------------------------------------------------------------------
// 观测模型与雅可比
// ---------------------------------------------------------------------------

int ArmorEkf::normalizeIndex(int armor_index, int plate_count) {
    if (plate_count <= 0) {
        return 0;
    }
    const int wrapped = armor_index % plate_count;
    return wrapped < 0 ? wrapped + plate_count : wrapped;
}

int ArmorEkf::radiusIndex(int armor_index) {
    // 偶数槽位用 r1，奇数用 r2。标准步兵四块板等半径，前哨站三块是交替半径。
    // 调用方传进来的下标必须已经落在 [0, plate_count) 里。
    return (armor_index % 2 == 0) ? Radius1 : Radius2;
}

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

ArmorEkf::ObservationJacobian ArmorEkf::observationJacobian(
    const StateVector& x, int armor_index, double gimbal_yaw,
    const std::array<double, 4>& plate_height_m, double plate_spacing_rad) {
    (void)plate_height_m;  // 高度是常量，不进状态，因此不产生列
    const int i = armor_index;
    const double psi = x[Yaw] + static_cast<double>(i) * plate_spacing_rad;
    const double radius = x[radiusIndex(i)];
    const Eigen::Matrix3d to_gimbal = worldToGimbal(gimbal_yaw);

    // h(x) = M^T * u(x)，所以每列就是 M^T 乘上 u 对那个状态分量在 W 系里的偏导。
    // 一律走 worldToGimbal，不在这里手抄 M^T 的元素。手抄等于把同一个约定写两遍，
    // 换约定时容易只改一处；测试的有限差分对拍能抓到，但不该给它这个机会。
    ObservationJacobian H = ObservationJacobian::Zero();
    // 平移：u 的三个平移项分别是 px/py/pz 加一个与状态无关的偏移，偏导就是坐标轴。
    H.col(Px) = to_gimbal * Eigen::Vector3d(1.0, 0.0, 0.0);
    H.col(Py) = to_gimbal * Eigen::Vector3d(0.0, 1.0, 0.0);
    H.col(Pz) = to_gimbal * Eigen::Vector3d(0.0, 0.0, 1.0);
    // yaw 与半径各自在 W 系里有一个方向，再统一换到云台系。两个方向差 90 度，
    // 别图省事共用同一个向量：
    //   u 对 theta_c 的偏导 = r * (-sin psi, cos psi, 0)   方位角转，切向
    //   u 对 r 的偏导       =     ( cos psi, sin psi, 0)   半径伸长，径向
    const Eigen::Vector3d tangential(-std::sin(psi), std::cos(psi), 0.0);
    const Eigen::Vector3d radial(std::cos(psi), std::sin(psi), 0.0);
    H.col(Yaw) = to_gimbal * (radius * tangential);
    // 半径只有该板对应的那一列非零，另一列在上面已被置零。
    H.col(radiusIndex(i)) = to_gimbal * radial;
    return H;
}

Eigen::Vector3d ArmorEkf::platePositionWorld(const StateVector& x, int armor_index) const {
    const int i = normalizeIndex(armor_index, config_.plate_count);
    const double psi = x[Yaw] + static_cast<double>(i) * config_.plate_spacing_rad;
    const double radius = x[radiusIndex(i)];
    return Eigen::Vector3d(x[Px] + radius * std::cos(psi),
                           x[Py] + radius * std::sin(psi),
                           x[Pz] + config_.plate_height_m[static_cast<std::size_t>(i)]);
}

// ---------------------------------------------------------------------------
// 预测
// ---------------------------------------------------------------------------

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

void ArmorEkf::fillProcessNoise(double dt, Covariance& Q) const {
    Q.setZero();
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt2 * dt2;

    // 随机加速度模型：每个 [位置, 速度] 轴对共享同一个分块。
    const double linear = config_.accel_noise_m_s2 * config_.accel_noise_m_s2;
    const int pairs[3][2] = {{Px, Vx}, {Py, Vy}, {Pz, Vz}};
    for (const auto& pair : pairs) {
        Q(pair[0], pair[0]) = linear * dt4 / 4.0;
        Q(pair[0], pair[1]) = linear * dt3 / 2.0;
        Q(pair[1], pair[0]) = linear * dt3 / 2.0;
        Q(pair[1], pair[1]) = linear * dt2;
    }

    const double angular =
        config_.angular_accel_noise_rad_s2 * config_.angular_accel_noise_rad_s2;
    Q(Yaw, Yaw) = angular * dt4 / 4.0;
    Q(Yaw, YawRate) = angular * dt3 / 2.0;
    Q(YawRate, Yaw) = angular * dt3 / 2.0;
    Q(YawRate, YawRate) = angular * dt2;

    // 半径几乎恒定，只给一点自由度防止滤波器把它冻死。dt 很小时由下限兜底。
    const double radius_variance =
        std::max(config_.radius_drift_m * config_.radius_drift_m * dt2,
                 config_.radius_variance_floor_m2);
    Q(Radius1, Radius1) = radius_variance;
    Q(Radius2, Radius2) = radius_variance;
}

Eigen::Matrix3d ArmorEkf::observationNoise(const Eigen::Vector3d& z) const {
    // 云台系是 前 / 左 / 上，第 0 个分量沿视线，也就是深度；1、2 两个是横向。
    // 深度方向的不确定度比横向大好几倍（单目深度靠板宽反推），别把两者放错轴：
    // 放错不会报错，只会让滤波器该信的不信、该松的不松。
    const double d = z.norm();
    const double lateral = config_.lateral_noise_scale * d + config_.lateral_noise_base_m;
    const double depth = config_.depth_noise_scale * d + config_.depth_noise_base_m;
    Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
    R(0, 0) = depth * depth;
    R(1, 1) = lateral * lateral;
    R(2, 2) = lateral * lateral;
    return R;
}

ArmorEkf::Covariance ArmorEkf::initialCovariance() {
    Covariance P = Covariance::Zero();
    P(Px, Px) = 1.0;
    P(Py, Py) = 1.0;
    P(Pz, Pz) = 1.0;
    P(Vx, Vx) = 25.0;
    P(Vy, Vy) = 25.0;
    P(Vz, Vz) = 9.0;
    P(Yaw, Yaw) = 0.25;
    P(YawRate, YawRate) = 9.0;
    P(Radius1, Radius1) = 0.0225;
    P(Radius2, Radius2) = 0.0225;
    return P;
}

void ArmorEkf::reset() {
    x_.setZero();
    P_.setZero();
    status_ = TrackingState::NoTarget;
    target_id_ = 0;
    hit_streak_ = 0;
    miss_count_ = 0;
    updated_this_frame_ = false;
    ever_stable_ = false;
    last_obs_ts_ = 0.0;
    used_indices_.fill(false);
    index_target_id_.fill(0);
    init_mean_.setZero();
    init_count_ = 0;
    init_used_this_frame_ = false;
    nis_count_ = 0;
    nis_head_ = 0;
    aim_index_ = -1;
    // rejected_count_ 与延迟窗口是诊断量，跨目标保留。
}

void ArmorEkf::predict(double dt) {
    used_indices_.fill(false);
    updated_this_frame_ = false;
    init_used_this_frame_ = false;
    if (status_ == TrackingState::NoTarget) {
        return;
    }
    if (!std::isfinite(dt)) {
        return;
    }
    dt = std::clamp(dt, 0.0, config_.max_step_s);
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

// ---------------------------------------------------------------------------
// 初始化与更新
// ---------------------------------------------------------------------------

bool ArmorEkf::update(const ArmorPose& observation, double gimbal_yaw) {
    if (!std::isfinite(gimbal_yaw)) {
        gimbal_yaw = 0.0;
    }
    if (status_ == TrackingState::NoTarget) {
        return tryInitialize(observation, gimbal_yaw);
    }
    return associateAndUpdate(observation, gimbal_yaw);
}

bool ArmorEkf::tryInitialize(const ArmorPose& observation, double gimbal_yaw) {
    if (init_used_this_frame_) {
        // 本帧已经有一条观测参与了初始化计数，同帧的其余观测直接跳过。
        return false;
    }
    init_used_this_frame_ = true;

    const Eigen::Vector3d z(observation.position_gimbal_m[0],
                            observation.position_gimbal_m[1],
                            observation.position_gimbal_m[2]);
    if (!z.allFinite() || !std::isfinite(observation.reprojection_error) ||
        observation.reprojection_error > config_.max_reprojection_error_px) {
        ++rejected_count_;
        init_count_ = 0;
        return false;
    }

    // 先把观测板心转到世界系，再检查连续若干帧是否彼此接近。
    // 不接近说明这几帧看到的不是同一块板（或不是同一个目标），重新开始计数。
    const Eigen::Vector3d u = gimbalToWorld(gimbal_yaw) * z;
    if (init_count_ == 0) {
        init_mean_ = u;
        init_count_ = 1;
    } else if ((u - init_mean_).norm() <= config_.init_consistency_m) {
        init_mean_ = (init_mean_ * static_cast<double>(init_count_) + u) /
                     static_cast<double>(init_count_ + 1);
        ++init_count_;
    } else {
        init_mean_ = u;
        init_count_ = 1;
    }
    if (init_count_ < config_.init_frames) {
        return false;
    }

    // 把观测到的板心当作 0 号板，反推旋转中心。编号是任意的全局相位，
    // 取 0 号不影响物理，后续会随观测自动纠正。
    //
    // 反推要用的方位角是"旋转中心指向板心"的径向方位。板装在外侧、板面朝外，
    // 而检测条件是板面法向朝向相机，所以这块板的径向方向与"板心指向相机"
    // 同向，也就是板心视线的反方位。取板心相对世界原点的方位角是反的 180 度，
    // 只有车心恰在世界原点时才碰巧相等。
    // 初始中心误差由此从 2r*cos(delta/2) 降到 2r*sin(delta/2)，
    // delta 是该板被看到的斜角，可见范围内不超过锥角。
    x_.setZero();
    x_[Yaw] = wrapToPi(std::atan2(-init_mean_.y(), -init_mean_.x()));
    x_[Radius1] = config_.initial_radius_m;
    x_[Radius2] = config_.initial_radius_m;
    x_[Px] = init_mean_.x() - config_.initial_radius_m * std::cos(x_[Yaw]);
    x_[Py] = init_mean_.y() - config_.initial_radius_m * std::sin(x_[Yaw]);
    x_[Pz] = init_mean_.z() - config_.plate_height_m[0];
    P_ = initialCovariance();

    status_ = TrackingState::Unstable;
    target_id_ = observation.detection.target_id;
    index_target_id_.fill(0);
    index_target_id_[0] = observation.detection.target_id;
    hit_streak_ = 0;
    miss_count_ = 0;
    ever_stable_ = false;
    updated_this_frame_ = true;
    nis_count_ = 0;
    nis_head_ = 0;
    aim_index_ = -1;
    used_indices_.fill(false);
    init_count_ = 0;
    init_mean_.setZero();
    return true;
}

bool ArmorEkf::associateAndUpdate(const ArmorPose& observation, double gimbal_yaw) {
    const Eigen::Vector3d z(observation.position_gimbal_m[0],
                            observation.position_gimbal_m[1],
                            observation.position_gimbal_m[2]);
    if (!z.allFinite() || !std::isfinite(observation.reprojection_error)) {
        ++rejected_count_;
        return false;
    }
    if (observation.reprojection_error > config_.max_reprojection_error_px) {
        ++rejected_count_;
        return false;
    }

    const Eigen::Matrix3d R = observationNoise(z);

    // 在四块板里挑一块来配这条观测。用马氏距离而不是最近欧氏距离：
    // 侧板的横向不确定度大，直接用欧氏距离会把它误配进来。
    int best_index = -1;
    double best_nis = std::numeric_limits<double>::infinity();
    Eigen::Vector3d best_h = Eigen::Vector3d::Zero();
    ObservationJacobian best_H = ObservationJacobian::Zero();

    for (int i = 0; i < config_.plate_count; ++i) {
        if (used_indices_[static_cast<std::size_t>(i)]) {
            continue;  // 一帧内一块板只配一条观测
        }
        const Eigen::Vector3d h = observationModel(
            x_, i, gimbal_yaw, config_.plate_height_m, config_.plate_spacing_rad);
        const Eigen::Vector3d residual = z - h;
        if (residual.norm() > config_.jump_gate_m) {
            continue;  // 物理跳变闸：6 ms 内 0.6 m 相当于 100 m/s，不可能是真目标
        }
        const ObservationJacobian H = observationJacobian(
            x_, i, gimbal_yaw, config_.plate_height_m, config_.plate_spacing_rad);
        const Eigen::Matrix3d S = H * P_ * H.transpose() + R;
        const double nis = residual.transpose() * S.inverse() * residual;
        if (!std::isfinite(nis)) {
            continue;
        }
        if (nis < best_nis) {
            best_nis = nis;
            best_index = i;
            best_h = h;
            best_H = H;
        }
    }

    if (best_index < 0) {
        ++rejected_count_;
        return false;
    }
    if (best_nis > config_.nis_gate) {
        ++rejected_count_;  // 统计闸：卡方 (3, 99%) 分位点
        return false;
    }

    const Eigen::Vector3d residual = z - best_h;
    const Eigen::Matrix3d S = best_H * P_ * best_H.transpose() + R;
    const Eigen::Matrix<double, 10, 3> K = P_ * best_H.transpose() * S.inverse();
    x_ += K * residual;
    // Joseph 形式。比 (I - K H) P 多一次乘法，但数值上不会失去正定性，
    // 长时间跑下来更稳。
    const Covariance identity_minus_kh = Covariance::Identity() - K * best_H;
    P_ = identity_minus_kh * P_ * identity_minus_kh.transpose() +
         K * R * K.transpose();
    x_[Yaw] = wrapToPi(x_[Yaw]);

    used_indices_[static_cast<std::size_t>(best_index)] = true;
    // 记住这一槽位读到的编号，供选板后回填 armor_flag。target_id_ 另存
    // 最近一次观测的编号，仅用于槽位编号未知时兜底。
    index_target_id_[static_cast<std::size_t>(best_index)] = observation.detection.target_id;
    target_id_ = observation.detection.target_id;
    updated_this_frame_ = true;
    recordNis(best_nis);
    return true;
}

// ---------------------------------------------------------------------------
// 状态机
// ---------------------------------------------------------------------------

void ArmorEkf::advanceStateMachine(double timestamp_seconds) {
    if (status_ == TrackingState::NoTarget) {
        updated_this_frame_ = false;
        return;
    }

    if (updated_this_frame_) {
        miss_count_ = 0;
        ++hit_streak_;
        if (std::isfinite(timestamp_seconds)) {
            last_obs_ts_ = timestamp_seconds;
        }
    } else {
        ++miss_count_;
        hit_streak_ = 0;
    }
    updated_this_frame_ = false;

    // 长时间丢失：1.5 秒内车已经可能平移数米或转好几圈，旧态外推没有意义。
    // 这个门限远大于真实遮挡（几百毫秒），不会误触发短时遮挡的恢复。
    if (last_obs_ts_ > 0.0 && std::isfinite(timestamp_seconds) &&
        timestamp_seconds - last_obs_ts_ > config_.reset_timeout_s) {
        reset();
        return;
    }

    // 短时丢失的滞回：丢几帧以内仍按 Stable 处理，纯预测足够准。
    if (status_ == TrackingState::Stable && miss_count_ >= config_.stable_hold_misses) {
        status_ = TrackingState::Unstable;
    }

    if (status_ == TrackingState::Unstable) {
        const int required = ever_stable_ ? config_.recover_frames : config_.stable_frames;
        if (hit_streak_ >= required && covarianceIsStable() &&
            recentNisMean() < config_.nis_mean_gate) {
            status_ = TrackingState::Stable;
            ever_stable_ = true;
        }
    }
}

bool ArmorEkf::covarianceIsStable() const {
    const double center_std = std::sqrt(std::max(0.0, P_(Px, Px) + P_(Py, Py)));
    const double radius_std =
        std::sqrt(std::max(0.0, P_(Radius1, Radius1) + P_(Radius2, Radius2)));
    const double yaw_std = std::sqrt(std::max(0.0, P_(Yaw, Yaw)));
    return center_std < config_.stable_center_std_m &&
           radius_std < config_.stable_radius_std_m &&
           yaw_std < config_.stable_yaw_std_rad;
}

void ArmorEkf::recordNis(double nis) {
    nis_history_[static_cast<std::size_t>(nis_head_)] = nis;
    nis_head_ = (nis_head_ + 1) % kNisRingSize;
    if (nis_count_ < kNisRingSize) {
        ++nis_count_;
    }
}

double ArmorEkf::recentNisMean() const {
    const int n = std::min(nis_count_, config_.nis_window);
    if (n <= 0) {
        return std::numeric_limits<double>::infinity();
    }
    double sum = 0.0;
    for (int k = 0; k < n; ++k) {
        const int index = ((nis_head_ - 1 - k) % kNisRingSize + kNisRingSize) % kNisRingSize;
        sum += nis_history_[static_cast<std::size_t>(index)];
    }
    return sum / static_cast<double>(n);
}

double ArmorEkf::smoothedLatency(double raw_latency_s) {
    if (!std::isfinite(raw_latency_s)) {
        raw_latency_s = 0.0;
    }
    raw_latency_s = std::clamp(raw_latency_s, 0.0, config_.max_latency_s);
    latency_history_[static_cast<std::size_t>(latency_head_)] = raw_latency_s;
    latency_head_ = (latency_head_ + 1) % kLatencyRingSize;
    if (latency_count_ < kLatencyRingSize) {
        ++latency_count_;
    }

    const int n = std::min(latency_count_, std::max(1, config_.latency_window));
    double sum = 0.0;
    for (int k = 0; k < n; ++k) {
        const int index =
            ((latency_head_ - 1 - k) % kLatencyRingSize + kLatencyRingSize) % kLatencyRingSize;
        sum += latency_history_[static_cast<std::size_t>(index)];
    }
    return sum / static_cast<double>(n);
}

// ---------------------------------------------------------------------------
// 选板与输出
// ---------------------------------------------------------------------------

int ArmorEkf::selectArmorIndex(const StateVector& x, double gimbal_yaw, bool commit) {
    const Eigen::Matrix3d to_gimbal = worldToGimbal(gimbal_yaw);
    const double facing_threshold = std::cos(config_.side_plate_angle_rad);
    double best_abs_yaw = std::numeric_limits<double>::infinity();
    int best = -1;
    bool current_facing = false;

    for (int i = 0; i < config_.plate_count; ++i) {
        const Eigen::Vector3d p_world = platePositionWorld(x, i);
        const double norm = p_world.norm();
        if (norm < 1e-6) {
            continue;
        }
        // 板面法向朝外。它要大致指着云台，这块板才看得见，否则是侧板。
        const double psi = x[Yaw] + static_cast<double>(i) * config_.plate_spacing_rad;
        const Eigen::Vector3d normal(std::cos(psi), std::sin(psi), 0.0);
        if (normal.dot(-p_world / norm) < facing_threshold) {
            continue;
        }
        if (i == aim_index_) {
            current_facing = true;
        }
        const Eigen::Vector3d p_gimbal = to_gimbal * p_world;
        const double abs_yaw = std::fabs(gimbalAzimuth(p_gimbal));
        if (abs_yaw < best_abs_yaw) {
            best_abs_yaw = abs_yaw;
            best = i;
        }
    }

    if (best < 0) {
        if (commit) {
            aim_index_ = -1;
        }
        return -1;
    }
    // 迟滞：新板要好过当前瞄准板一个余量才切，避免在两块板之间来回跳。
    if (aim_index_ >= 0 && current_facing && aim_index_ != best) {
        const Eigen::Vector3d p_current = to_gimbal * platePositionWorld(x, aim_index_);
        const double current_abs_yaw = std::fabs(gimbalAzimuth(p_current));
        if (current_abs_yaw - best_abs_yaw < config_.aim_switch_margin_rad) {
            best = aim_index_;
        }
    }
    if (commit) {
        aim_index_ = best;
    }
    return best;
}

PredictionResult ArmorEkf::buildPrediction(const GimbalState& gimbal, double latency_s) {
    PredictionResult result;
    if (status_ == TrackingState::NoTarget) {
        return result;
    }

    const double gimbal_yaw =
        std::isfinite(gimbal.yaw) ? static_cast<double>(gimbal.yaw) : 0.0;
    const double gimbal_pitch =
        std::isfinite(gimbal.pitch) ? static_cast<double>(gimbal.pitch) : 0.0;
    // bias 允许为负：电控若想"少预测一点"就会下发负值，单边截正会把它吃掉。
    // 总时长 t0 仍然截在 [0, max] 里，异常值照样挡得住。
    const double bias = std::isfinite(gimbal.prediction_bias_s)
                            ? static_cast<double>(gimbal.prediction_bias_s)
                            : 0.0;

    const double latency = smoothedLatency(latency_s);
    const double t0 = std::clamp(latency + bias, 0.0, config_.max_prediction_s);

    // 云台系的点后面多处要用，先算一次，飞行时间迭代、瞄准、中心差分共用。
    const Eigen::Matrix3d to_gimbal = worldToGimbal(gimbal_yaw);

    // 飞行时间依赖距离，距离又依赖预测时长。两次迭代解开这个循环。
    // 迭代里只试算，不提交 aim_index_，否则迟滞状态会被试算值写脏。
    double flight_s = 0.0;
    for (int iteration = 0; iteration < 2; ++iteration) {
        StateVector x_step = x_;
        propagate(x_step, t0 + flight_s);
        const int index = selectArmorIndex(x_step, gimbal_yaw, false);
        if (index < 0) {
            break;
        }
        const Eigen::Vector3d p_world = platePositionWorld(x_step, index);
        // 这一轮只取飞行时间，第三个参数是回退仰角，用不到。
        const BallisticInputs inputs =
            ballisticInputsFromMuzzle(to_gimbal * p_world, config_.muzzle_offset_m);
        flight_s = solveBallistics(inputs.horizontal_m, inputs.vertical_m, 0.0)
                       .flight_time_s;
    }
    const double t_pred = std::clamp(t0 + flight_s, 0.0, config_.max_prediction_s);

    StateVector x_pred = x_;
    propagate(x_pred, t_pred);
    const int aim_index = selectArmorIndex(x_pred, gimbal_yaw, true);
    if (aim_index < 0) {
        return PredictionResult{};
    }
    const Eigen::Vector3d p_world_aim = platePositionWorld(x_pred, aim_index);
    const Eigen::Vector3d p_gimbal_aim = to_gimbal * p_world_aim;

    // 与 main.cpp 里 pnp_yaw_rad / pnp_pitch_rad 同式：yaw 右为正，pitch 抬头为正。
    const double rel_yaw = gimbalAzimuth(p_gimbal_aim);
    const double rel_pitch = gimbalElevation(p_gimbal_aim);

    // 指令仰角取弹道发射仰角。下坠补偿在算法侧做，发的不是瞄准点的几何仰角；
    // 第三个参数是弹道解不收敛时的回退值，即不作补偿的几何绝对仰角。
    const BallisticInputs aim_inputs =
        ballisticInputsFromMuzzle(p_gimbal_aim, config_.muzzle_offset_m);
    const double command_pitch =
        solveBallistics(aim_inputs.horizontal_m, aim_inputs.vertical_m,
                        gimbal_pitch + rel_pitch)
            .launch_pitch_rad;

    // 角速度用中心差分，窗口 +/-5 ms。
    constexpr double kRateHalfWindowS = 0.005;
    const double t_early = std::max(0.0, t_pred - kRateHalfWindowS);
    const double t_late = t_pred + kRateHalfWindowS;
    StateVector x_early = x_;
    StateVector x_late = x_;
    propagate(x_early, t_early);
    propagate(x_late, t_late);
    const Eigen::Vector3d p_world_early = platePositionWorld(x_early, aim_index);
    const Eigen::Vector3d p_world_late = platePositionWorld(x_late, aim_index);
    const Eigen::Vector3d p_early = to_gimbal * p_world_early;
    const Eigen::Vector3d p_late = to_gimbal * p_world_late;
    const double yaw_early = gimbalAzimuth(p_early);
    const double yaw_late = gimbalAzimuth(p_late);
    // target_pitch 是补偿后的角，它的速率也必须来自补偿后的角，
    // 否则同一个上报结构里两个量对不上。
    const BallisticInputs early_inputs =
        ballisticInputsFromMuzzle(p_early, config_.muzzle_offset_m);
    const BallisticInputs late_inputs =
        ballisticInputsFromMuzzle(p_late, config_.muzzle_offset_m);
    const double pitch_early =
        solveBallistics(early_inputs.horizontal_m, early_inputs.vertical_m,
                        gimbal_pitch + gimbalElevation(p_early))
            .launch_pitch_rad;
    const double pitch_late =
        solveBallistics(late_inputs.horizontal_m, late_inputs.vertical_m,
                        gimbal_pitch + gimbalElevation(p_late))
            .launch_pitch_rad;
    const double dt_window = t_late - t_early;
    const double yaw_velocity =
        dt_window > 0.0 ? wrapToPi(yaw_late - yaw_early) / dt_window : 0.0;
    const double pitch_velocity =
        dt_window > 0.0 ? (pitch_late - pitch_early) / dt_window : 0.0;

    result.state = status_;
    // armor_flag 描述"正瞄着的那块板"。aim_index 槽位读到过编号就用它，
    // 没读到过（纯预测到一块没直接观测过的板）才退回最近一次观测的编号。
    const std::uint8_t aim_target_id =
        index_target_id_[static_cast<std::size_t>(aim_index)];
    result.target_id = aim_target_id != 0 ? aim_target_id : target_id_;
    result.aim_index = aim_index;
    result.hit_streak = hit_streak_;
    result.miss_count = miss_count_;
    result.rejected_count = rejected_count_;
    // 车体状态报在跟踪系（W：x 前、y 左、z 上）里，与 PredictionResult 的注释一致。
    result.center_x_m = static_cast<float>(x_[Px]);
    result.center_y_m = static_cast<float>(x_[Py]);
    result.center_z_m = static_cast<float>(x_[Pz]);
    result.center_velocity_x_m_s = static_cast<float>(x_[Vx]);
    result.center_velocity_y_m_s = static_cast<float>(x_[Vy]);
    result.center_velocity_z_m_s = static_cast<float>(x_[Vz]);
    result.body_yaw = static_cast<float>(x_[Yaw]);
    result.body_yaw_velocity = static_cast<float>(x_[YawRate]);
    result.radius_1_m = static_cast<float>(std::max(0.0, x_[Radius1]));
    result.radius_2_m = static_cast<float>(std::max(0.0, x_[Radius2]));
    result.distance_m = static_cast<float>(p_gimbal_aim.norm());
    // 输出绝对云台指令，不是相对角。依据是框架的 setSafeOutput：
    // 安全帧写成 target_yaw = message.yaw（保持当前），只有绝对角才自洽。
    result.target_yaw = static_cast<float>(gimbal_yaw + rel_yaw);
    // 下坠补偿在算法侧做，这里发的是弹道发射仰角，不是瞄准点的几何仰角。
    result.target_pitch = static_cast<float>(command_pitch);
    result.target_yaw_velocity = static_cast<float>(yaw_velocity);
    result.target_pitch_velocity = static_cast<float>(pitch_velocity);

    // yaw 的指令是 gimbal_yaw + rel_yaw，误差就是 rel_yaw；pitch 的指令是
    // command_pitch，误差是它与当前绝对仰角 gimbal_pitch 之差。两者口径一致。
    const bool aligned =
        std::fabs(rel_yaw) < config_.fire_yaw_tolerance_rad &&
        std::fabs(command_pitch - gimbal_pitch) < config_.fire_pitch_tolerance_rad;
    const bool spin_ok = std::fabs(x_[YawRate]) < config_.fire_spin_tolerance_rad_s;
    result.fire_allowed =
        status_ == TrackingState::Stable && aligned && spin_ok;

    if (!predictionIsFinite(result)) {
        return PredictionResult{};
    }
    return result;
}
