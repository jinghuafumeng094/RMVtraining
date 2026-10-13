// ArmorEkf 的离线烟测。
//
// 这不是验收手段，验收以上车实测为准。它只负责在上车前把低层错误挡掉：
// 坐标变换的符号、解析雅可比、状态机时序、安全不变量。
// 运行：ctest -R ekf_predict

#include "ArmorEkf.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>

namespace {

constexpr double kPi = 3.14159265358979323846;

// 测试自己维护一份真值状态并自己推进，不复用被测代码的 propagate，
// 这样自查才有可能发现 propagate 本身写错了。
struct Truth {
    Eigen::Vector3d center{0.0, 0.0, 0.0};
    Eigen::Vector3d velocity{0.0, 0.0, 0.0};
    double yaw{0.0};
    double yaw_rate{0.0};
    double radius1{0.28};
    double radius2{0.22};
};

// 四块板的高度刻意取得不完全相同。等半径等高时，把整块板的编号错开两位、
// 同时把 theta_c 偏移 pi，会得到一模一样的一组板心，状态不可辨识，
// 收敛断言就没有意义了。半径也刻意取 r1 不等于 r2，把奇数编号错开一位同理。
std::array<double, 4> testHeights() {
    return {0.0, 0.0, 0.15, 0.0};
}

Eigen::Vector3d truePlateWorld(const Truth& truth, int index,
                               const ArmorEkf::Config& config) {
    const double psi =
        truth.yaw + static_cast<double>(index) * config.plate_spacing_rad;
    const double radius = (index % 2 == 0) ? truth.radius1 : truth.radius2;
    return Eigen::Vector3d(
        truth.center.x() + radius * std::cos(psi),
        truth.center.y() + radius * std::sin(psi),
        truth.center.z() +
            config.plate_height_m[static_cast<std::size_t>(index)]);
}

void trueStep(Truth& truth, double dt) {
    truth.center += truth.velocity * dt;
    truth.yaw += truth.yaw_rate * dt;
}

ArmorEkf::StateVector truthVector(const Truth& truth) {
    ArmorEkf::StateVector x = ArmorEkf::StateVector::Zero();
    x[ArmorEkf::Px] = truth.center.x();
    x[ArmorEkf::Py] = truth.center.y();
    x[ArmorEkf::Pz] = truth.center.z();
    x[ArmorEkf::Vx] = truth.velocity.x();
    x[ArmorEkf::Vy] = truth.velocity.y();
    x[ArmorEkf::Vz] = truth.velocity.z();
    x[ArmorEkf::Yaw] = truth.yaw;
    x[ArmorEkf::YawRate] = truth.yaw_rate;
    x[ArmorEkf::Radius1] = truth.radius1;
    x[ArmorEkf::Radius2] = truth.radius2;
    return x;
}

// 一帧喂满 config.plate_count 块板，全部由真值经观测模型生成，无噪声。
// 板数取配置值而不是写死 4：两块板的车与前哨站都要能走同一条收敛路径。
void feedFrame(ArmorEkf& filter, const ArmorEkf::Config& config, const Truth& truth,
               double gimbal_yaw) {
    const Eigen::Matrix3d to_gimbal = ArmorEkf::worldToGimbal(gimbal_yaw);
    for (int index = 0; index < config.plate_count; ++index) {
        const Eigen::Vector3d z = to_gimbal * truePlateWorld(truth, index, config);
        ArmorPose observation;
        observation.position_gimbal_m = cv::Vec3d(z.x(), z.y(), z.z());
        observation.position_camera_m = observation.position_gimbal_m;
        observation.armor_yaw = 0.0;
        observation.reprojection_error = 0.3;
        observation.detection.target_id = index + 1;
        filter.update(observation, gimbal_yaw);
    }
}

GimbalState makeGimbal(double yaw, double pitch, double bias = 0.0) {
    GimbalState gimbal;
    gimbal.yaw = static_cast<float>(yaw);
    gimbal.pitch = static_cast<float>(pitch);
    gimbal.roll = 0.0F;
    gimbal.prediction_bias_s = static_cast<float>(bias);
    return gimbal;
}

ArmorEkf::Config testConfig() {
    ArmorEkf::Config config;
    config.plate_height_m = testHeights();
    return config;
}

// ---------------------------------------------------------------------------

void testCoordinateTransforms() {
    const Eigen::Matrix3d m = ArmorEkf::gimbalToWorld(0.0);
    assert(std::fabs(m.determinant() - 1.0) < 1e-12);

    // 云台系正前方 +z 对应世界系 +x，云台系右侧 +x 对应世界系 -y。
    assert((m * Eigen::Vector3d(0.0, 0.0, 5.0) - Eigen::Vector3d(5.0, 0.0, 0.0)).norm() <
           1e-12);
    assert((m * Eigen::Vector3d(5.0, 0.0, 0.0) - Eigen::Vector3d(0.0, -5.0, 0.0)).norm() <
           1e-12);
    assert((m * Eigen::Vector3d(0.0, 5.0, 0.0) - Eigen::Vector3d(0.0, 0.0, -5.0)).norm() <
           1e-12);

    // 云台向右转，视线在世界系里朝 -y 走。这条把 gimbal.yaw 的方向钉死：
    // 右为正是它的定义，符号若反了这里立刻红。
    const double yaw = 0.2;
    const Eigen::Vector3d boresight =
        ArmorEkf::gimbalToWorld(yaw) * Eigen::Vector3d(0.0, 0.0, 1.0);
    assert(std::fabs(std::atan2(boresight.y(), boresight.x()) + yaw) < 1e-12);

    const Eigen::Matrix3d round_trip =
        ArmorEkf::worldToGimbal(yaw) * ArmorEkf::gimbalToWorld(yaw);
    assert((round_trip - Eigen::Matrix3d::Identity()).norm() < 1e-12);

    assert(std::fabs(ArmorEkf::wrapToPi(0.0)) < 1e-15);
    assert(std::fabs(ArmorEkf::wrapToPi(3.0 + 2.0 * kPi) - 3.0) < 1e-12);
    assert(std::fabs(ArmorEkf::wrapToPi(6.0 * kPi)) < 1e-12);
    assert(ArmorEkf::wrapToPi(kPi) > 0.0);
    assert(std::fabs(ArmorEkf::wrapToPi(-kPi) - kPi) < 1e-12);

    std::printf("[ekf] 坐标变换 通过\n");
}

// theta_c 是 W 系里的方位角，左为正；gimbal.yaw 是右为正。两者方向相反，
// 这条把 theta_c 的方向单独钉出来，免得日后被人"统一"成同一个符号。
// 收敛用例虽然也能抓到符号翻转，但它是间接的，这里给一条直读的。
void testPlateAzimuthConvention() {
    const ArmorEkf::Config config = testConfig();
    const double theta_c = 0.3;
    Truth truth;
    truth.center = Eigen::Vector3d(4.0, 0.0, 0.0);
    truth.yaw = theta_c;
    const ArmorEkf::StateVector x = truthVector(truth);

    // observationModel 返回的是 G 系里的预测值 h = M^T * u。
    // 再左乘 M 换回 W 系，才拿到板在 W 系里的位置。
    const Eigen::Vector3d plate_gimbal =
        ArmorEkf::observationModel(x, 0, 0.0, config.plate_height_m,
                                   config.plate_spacing_rad);
    const Eigen::Vector3d plate = ArmorEkf::gimbalToWorld(0.0) * plate_gimbal;
    // 0 号板在 W 系里的方位角就是 theta_c，且取正值时落在 +y 一侧（左）。
    const Eigen::Vector3d relative = plate - Eigen::Vector3d(4.0, 0.0, 0.0);
    assert(std::fabs(std::atan2(relative.y(), relative.x()) - theta_c) < 1e-12);
    assert(relative.y() > 0.0);  // 左为正

    // 同一时刻云台 yaw 取同一个正值，视线却朝 -y（右）。方向确实相反。
    const Eigen::Vector3d boresight =
        ArmorEkf::gimbalToWorld(theta_c) * Eigen::Vector3d(0.0, 0.0, 1.0);
    assert(boresight.y() < 0.0);

    std::printf("[ekf] theta_c 左为正 / gimbal.yaw 右为正 通过\n");
}

void testAnalyticJacobian() {
    const ArmorEkf::Config config = testConfig();
    ArmorEkf::StateVector x = ArmorEkf::StateVector::Zero();
    x[ArmorEkf::Px] = 3.0;
    x[ArmorEkf::Py] = -1.2;
    x[ArmorEkf::Pz] = 0.4;
    x[ArmorEkf::Vx] = 0.7;
    x[ArmorEkf::Vy] = -0.3;
    x[ArmorEkf::Vz] = 0.1;
    x[ArmorEkf::Yaw] = 0.6;
    x[ArmorEkf::YawRate] = 0.5;
    x[ArmorEkf::Radius1] = 0.27;
    x[ArmorEkf::Radius2] = 0.21;

    const double gimbal_yaw = 0.31;

    for (int index = 0; index < config.plate_count; ++index) {
        const ArmorEkf::ObservationJacobian analytic = ArmorEkf::observationJacobian(
            x, index, gimbal_yaw, config.plate_height_m, config.plate_spacing_rad);
        for (int column = 0; column < ArmorEkf::Dimension; ++column) {
            const double step = 1e-6 * std::max(1.0, std::fabs(x[column]));
            ArmorEkf::StateVector forward = x;
            ArmorEkf::StateVector backward = x;
            forward[column] += step;
            backward[column] -= step;
            const Eigen::Vector3d numeric =
                (ArmorEkf::observationModel(forward, index, gimbal_yaw,
                                            config.plate_height_m,
                                            config.plate_spacing_rad) -
                 ArmorEkf::observationModel(backward, index, gimbal_yaw,
                                            config.plate_height_m,
                                            config.plate_spacing_rad)) /
                (2.0 * step);
            for (int row = 0; row < 3; ++row) {
                assert(std::fabs(numeric[row] - analytic(row, column)) < 1e-6);
            }
        }
    }
    std::printf("[ekf] 解析雅可比对拍差分 通过\n");
}

void testNoiselessConvergence() {
    const ArmorEkf::Config config = testConfig();
    ArmorEkf filter(config);

    Truth truth;
    truth.center = Eigen::Vector3d(4.0, 0.3, 0.6);
    truth.velocity = Eigen::Vector3d(0.5, -0.2, 0.0);
    truth.yaw = 0.4;
    truth.yaw_rate = 0.4;

    const double dt = 1.0 / 165.0;
    for (int frame = 0; frame < 600; ++frame) {
        // 真值先推进到本帧的观测时刻，再让滤波器预测同样的步长并吃下这条观测。
        // 顺序反过来，滤波器收敛到的状态会整整落后一个 v*dt，
        // 误差恰好是 0.5/165 这类量级，看着像"没收敛"，其实是测试的时序错了。
        trueStep(truth, dt);
        filter.predict(dt);
        feedFrame(filter, config, truth, 0.0);
        filter.advanceStateMachine(static_cast<double>(frame + 1) * dt);
    }

    assert(filter.status() == TrackingState::Stable);
    const ArmorEkf::StateVector error = filter.state() - truthVector(truth);
    assert(error.cwiseAbs().maxCoeff() < 1e-6);
    std::printf("[ekf] 无噪自洽收敛 %.3e 通过\n", error.cwiseAbs().maxCoeff());
}

// 两块板的车：板间夹角是 pi 而不是 90 度。间距若仍按 90 度排，两块板会落在
// 同奇偶的槽位里，r2 拿不到任何观测，报出去的是先验假值。这条盯着 r2 收敛。
//
// 两块板等高时状态不可辨识：把两块板对调、theta_c 加 pi、r1 与 r2 互换，
// 会得到一模一样的一组板心。所以这里给一个高度差把对称性打破，
// 做法与四块板用例里那个 {0,0,0.15,0} 同源。
//
// 半径还只是弱可观测：两块板对置时，中心落在连心线上，位置只由比值
// r1/(r1+r2) 决定，而这个比值只能靠车自己转出来。所以这里让车快转
// 好几圈，观测窗口比四块板那条用例长得多。转速低时半径收敛得慢是
// 几何决定的，不是滤波器的问题。
void testTwoPlateVehicleObservesRadiusTwo() {
    ArmorEkf::Config config = testConfig();
    config.plate_height_m = {0.0, 0.12, 0.0, 0.0};
    config.plate_count = 2;
    config.plate_spacing_rad = kPi;
    ArmorEkf filter(config);

    Truth truth;
    truth.center = Eigen::Vector3d(4.0, -0.4, 0.5);
    truth.velocity = Eigen::Vector3d(0.3, 0.2, 0.0);
    truth.yaw = 0.5;
    truth.yaw_rate = 2.0;
    truth.radius1 = 0.30;
    truth.radius2 = 0.19;

    const double dt = 1.0 / 165.0;
    for (int frame = 0; frame < 3000; ++frame) {
        trueStep(truth, dt);
        filter.predict(dt);
        feedFrame(filter, config, truth, 0.0);
        filter.advanceStateMachine(static_cast<double>(frame + 1) * dt);
    }

    const ArmorEkf::StateVector x = filter.state();
    // theta_c 被裹到 (-pi, pi]，真值一路累加没有裹，比较前先裹一遍。
    assert(std::fabs(ArmorEkf::wrapToPi(x[ArmorEkf::Yaw] - truth.yaw)) < 5e-3);
    assert(std::fabs(x[ArmorEkf::YawRate] - truth.yaw_rate) < 5e-3);
    // 两半径之和就是两板间距，是强观测量，先把它钉死。
    assert(std::fabs(x[ArmorEkf::Radius1] + x[ArmorEkf::Radius2] -
                     (truth.radius1 + truth.radius2)) < 1e-3);
    // 关键判据在这里。槽位若按 90 度排，两块板会落进同奇偶的槽位，
    // 模型被迫认为 r1 == r2，这个差就停在 0 附近收敛不出来。
    assert(x[ArmorEkf::Radius1] > x[ArmorEkf::Radius2]);
    assert(std::fabs((x[ArmorEkf::Radius1] - x[ArmorEkf::Radius2]) -
                     (truth.radius1 - truth.radius2)) < 0.04);
    std::printf("[ekf] 两块板 180 度 半径 r1>r2 可辨识 通过\n");
}

// 目标在世界系里完全静止，只让云台原地扫。运动模型建在世界系，
// 所以滤波器必须给出零速度、零角速度。G 到 W 的变换若符号写反，
// 云台的转动会被读成目标的转动，这条会红。
void testGimbalRotationIsNotTargetMotion() {
    const ArmorEkf::Config config = testConfig();
    ArmorEkf filter(config);

    Truth truth;
    truth.center = Eigen::Vector3d(3.0, 1.0, 0.5);
    truth.velocity = Eigen::Vector3d::Zero();
    truth.yaw = 0.7;
    truth.yaw_rate = 0.0;

    const double dt = 1.0 / 165.0;
    for (int frame = 0; frame < 500; ++frame) {
        const double t = static_cast<double>(frame) * dt;
        const double gimbal_yaw = 0.35 * std::sin(2.0 * kPi * 0.7 * t);
        filter.predict(dt);
        feedFrame(filter, config, truth, gimbal_yaw);
        filter.advanceStateMachine(static_cast<double>(frame + 1) * dt);
    }

    const ArmorEkf::StateVector x = filter.state();
    assert(std::fabs(x[ArmorEkf::YawRate]) < 1e-6);
    assert(std::fabs(x[ArmorEkf::Vx]) < 1e-6);
    assert(std::fabs(x[ArmorEkf::Vy]) < 1e-6);
    assert(std::fabs(x[ArmorEkf::Vz]) < 1e-6);
    const Eigen::Vector3d center(x[ArmorEkf::Px], x[ArmorEkf::Py], x[ArmorEkf::Pz]);
    assert((center - truth.center).norm() < 1e-6);
    std::printf("[ekf] 云台转动不产生目标运动 通过\n");
}

void testStateMachineTiming() {
    const ArmorEkf::Config config = testConfig();
    ArmorEkf filter(config);

    Truth truth;
    truth.center = Eigen::Vector3d(4.0, 0.0, 0.4);
    truth.velocity = Eigen::Vector3d(0.3, 0.1, 0.0);
    truth.yaw = 0.2;
    truth.yaw_rate = 0.6;

    const double dt = 1.0 / 165.0;
    double timestamp = 0.0;

    for (int frame = 0; frame < 60; ++frame) {
        timestamp += dt;
        filter.predict(dt);
        feedFrame(filter, config, truth, 0.0);
        filter.advanceStateMachine(timestamp);
        trueStep(truth, dt);
    }
    assert(filter.status() == TrackingState::Stable);
    // crc 的取值就是 TrackingState 的枚举值，Stable 必须是 1。
    assert(static_cast<std::uint16_t>(filter.status()) == 1);

    // 丢两帧以内仍维持稳定，第三帧掉出稳定态。
    for (int frame = 0; frame < 2; ++frame) {
        timestamp += dt;
        filter.predict(dt);
        filter.advanceStateMachine(timestamp);
        trueStep(truth, dt);
    }
    assert(filter.status() == TrackingState::Stable);

    timestamp += dt;
    filter.predict(dt);
    filter.advanceStateMachine(timestamp);
    trueStep(truth, dt);
    assert(filter.status() == TrackingState::Unstable);
    assert(static_cast<std::uint16_t>(filter.status()) == 2);
    assert(filter.missCount() == 3);

    // 观测回来，连续命中到恢复门限后回到稳定态。
    for (int frame = 0; frame < 10; ++frame) {
        timestamp += dt;
        filter.predict(dt);
        feedFrame(filter, config, truth, 0.0);
        filter.advanceStateMachine(timestamp);
        trueStep(truth, dt);
    }
    assert(filter.status() == TrackingState::Stable);

    // 连续丢失超过超时门限，整个目标被丢弃。
    for (int frame = 0; frame < 300; ++frame) {
        timestamp += dt;
        filter.predict(dt);
        filter.advanceStateMachine(timestamp);
        trueStep(truth, dt);
    }
    assert(filter.status() == TrackingState::NoTarget);
    assert(static_cast<std::uint16_t>(filter.status()) == 0);
    assert(filter.targetId() == 0);
    std::printf("[ekf] 状态机时序 1->2->1->0 通过\n");
}

// 一帧里有多块板时，初始化计数只取第一条观测，否则板与板之间的间距
// 会顶掉一致门，计数永远凑不满。
void testInitializationTakesOneObservationPerFrame() {
    const ArmorEkf::Config config = testConfig();
    ArmorEkf filter(config);

    Truth truth;
    truth.center = Eigen::Vector3d(5.0, 0.0, 0.0);
    truth.yaw = 0.3;

    const double dt = 1.0 / 165.0;
    for (int frame = 0; frame < 10; ++frame) {
        filter.predict(dt);
        feedFrame(filter, config, truth, 0.0);
        filter.advanceStateMachine(static_cast<double>(frame + 1) * dt);
    }
    assert(filter.status() != TrackingState::NoTarget);
    std::printf("[ekf] 一帧多板仍能初始化 通过\n");
}

// 电控下发负的 prediction_bias_s 表示"少预测一点"，单边截正会把它吃掉。
//
// latency 特意取 50 ms：正负 bias 都要落在 (0, max) 里才不会被总时长那道
// clamp 截断。若只给 1 ms 的 latency，负 bias 会被总时长夹到 0，
// 这时测到的是"被夹住"而不是"被吃掉"，分不清是哪种毛病。
void testNegativePredictionBiasIsHonored() {
    const ArmorEkf::Config config = testConfig();

    auto convergeAndPredict = [&config](double bias) {
        ArmorEkf filter(config);
        Truth truth;
        truth.center = Eigen::Vector3d(4.0, 0.5, 0.4);
        truth.velocity = Eigen::Vector3d(0.6, 0.0, 0.0);
        truth.yaw = 0.3;
        truth.yaw_rate = 0.8;
        const double dt = 1.0 / 165.0;
        for (int frame = 0; frame < 60; ++frame) {
            trueStep(truth, dt);
            filter.predict(dt);
            feedFrame(filter, config, truth, 0.0);
            filter.advanceStateMachine(static_cast<double>(frame + 1) * dt);
        }
        assert(filter.status() == TrackingState::Stable);
        return filter.buildPrediction(makeGimbal(0.0, 0.0, bias), 0.05);
    };

    const PredictionResult none = convergeAndPredict(0.0);
    const PredictionResult later = convergeAndPredict(0.03);
    const PredictionResult earlier = convergeAndPredict(-0.03);
    const double d_later = later.target_yaw - none.target_yaw;
    const double d_earlier = earlier.target_yaw - none.target_yaw;
    assert(std::fabs(d_later) > 1e-3);
    assert(std::fabs(d_earlier) > 1e-3);
    // 正 bias 把事情往前推，负 bias 往后推，两个方向必须相反。
    // 负值若被单边截成 0，d_earlier 会退化成 0，这条立刻红。
    assert(d_later * d_earlier < 0.0);
    std::printf("[ekf] 负 prediction_bias 不被截正 通过\n");
}

void testSafetyInvariants() {
    const ArmorEkf::Config config = testConfig();
    ArmorEkf filter(config);
    const GimbalState gimbal = makeGimbal(0.0, 0.0);

    const double dt = 1.0 / 165.0;
    double timestamp = 0.0;

    // 无观测空跑：不崩、输出全有限、绝不允许开火。
    for (int frame = 0; frame < 100; ++frame) {
        timestamp += dt;
        filter.predict(dt);
        filter.advanceStateMachine(timestamp);
        const PredictionResult result = filter.buildPrediction(gimbal, 0.001);
        assert(result.state == TrackingState::NoTarget);
        assert(!result.fire_allowed);
        assert(std::isfinite(result.target_yaw));
        assert(std::isfinite(result.target_pitch));
        assert(std::isfinite(result.center_x_m));
        assert(std::isfinite(result.distance_m));
    }

    // 观测不足初始化门限，不许建立目标。
    Truth truth;
    truth.center = Eigen::Vector3d(4.0, 0.3, 0.6);
    truth.yaw = 0.4;
    truth.yaw_rate = 0.4;

    timestamp += dt;
    filter.predict(dt);
    feedFrame(filter, config, truth, 0.0);
    filter.advanceStateMachine(timestamp);
    assert(filter.status() == TrackingState::NoTarget);

    // 跑够帧数进入稳定态。云台没有对准目标，稳定但不许开火。
    for (int frame = 0; frame < 60; ++frame) {
        timestamp += dt;
        filter.predict(dt);
        feedFrame(filter, config, truth, 0.0);
        filter.advanceStateMachine(timestamp);
        trueStep(truth, dt);
        const PredictionResult result = filter.buildPrediction(gimbal, 0.001);
        // 只有稳定态才可能开火，这是 crc 语义的硬约束。
        assert(!result.fire_allowed || result.state == TrackingState::Stable);
    }
    assert(filter.status() == TrackingState::Stable);
    const PredictionResult misaligned = filter.buildPrediction(gimbal, 0.001);
    assert(misaligned.state == TrackingState::Stable);
    assert(!misaligned.fire_allowed);
    std::printf("[ekf] 安全不变量 通过\n");
}

}  // namespace

int main() {
    testCoordinateTransforms();
    testPlateAzimuthConvention();
    testAnalyticJacobian();
    testNoiselessConvergence();
    testTwoPlateVehicleObservesRadiusTwo();
    testGimbalRotationIsNotTargetMotion();
    testStateMachineTiming();
    testInitializationTakesOneObservationPerFrame();
    testNegativePredictionBiasIsHonored();
    testSafetyInvariants();
    std::printf("[ekf] 全部通过\n");
    return 0;
}
