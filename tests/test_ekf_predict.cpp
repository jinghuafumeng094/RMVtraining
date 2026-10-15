// ArmorEkf 的离线烟测。
//
// 这不是验收手段，验收以上车实测为准。它负责在上车前把低层错误挡掉：
// 坐标变换的符号、解析雅可比、初始化的方位角方向、无噪自洽收敛、
// 状态机时序、安全不变量。
// 运行：ctest -R ekf_predict

#include "ArmorEkf.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>

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

// 板高一律取 0，走共面这一支，与 config 的出厂默认一致。
// 实车不是共面：标准步兵相对的两块等高、相邻两块有高差，也就是 0/2 共用一个
// 高度、1/3 共用另一个，与 r1/r2 的奇偶配对同一套（armor/src/Planner.cpp:223-226
// 的 z0=z2、z1=z3）。这里取 0 只是让断言避开"板环整体挪两格"的等价关系，
// 不是在声明实车共面；上车的真实高差要另外量。
// 以前这里给 2 号板 0.15 m 的抬高，目的是打破同一层的等价关系，好让 theta_c
// 的断言有唯一解。实测这个抬高在真实可见性下会把收敛拖慢到几千帧、还随起始
// 朝向时好时坏，而它想解决的"板环等价"本来就该在断言里避开，不该靠改物理
// 配置绕过去。见下面无噪收敛用例的注释。
std::array<double, 4> testHeights() {
    return {0.0, 0.0, 0.0, 0.0};
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

// 可见性锥角，与 testrig/ekf_trace 的 visiblePlates 同一条判据。
// 检测器只输出板面朝向相机的板：板装在外侧、板面朝外，法向就是板心相对
// 旋转中心的径向，所以判据是"板面法向与指向相机的方向夹角在锥角以内"。
constexpr double kVisibilityConeRad = 80.0 * kPi / 180.0;

bool plateFacesCamera(const Truth& truth, int index, const ArmorEkf::Config& config) {
    const Eigen::Vector3d plate = truePlateWorld(truth, index, config);
    const double psi =
        truth.yaw + static_cast<double>(index) * config.plate_spacing_rad;
    const Eigen::Vector3d normal(std::cos(psi), std::sin(psi), 0.0);
    return normal.dot(-plate.normalized()) > std::cos(kVisibilityConeRad);
}

// 模型给出的板位。测试自己按观测模型的公式算一遍，不调被测代码的
// platePositionWorld：那是私有的，而且自查要能发现模型本身写错。
Eigen::Vector3d modelPlateWorld(const ArmorEkf::StateVector& x, int index,
                                const ArmorEkf::Config& config) {
    const double psi =
        x[ArmorEkf::Yaw] + static_cast<double>(index) * config.plate_spacing_rad;
    const double radius =
        (index % 2 == 0) ? x[ArmorEkf::Radius1] : x[ArmorEkf::Radius2];
    return Eigen::Vector3d(
        x[ArmorEkf::Px] + radius * std::cos(psi),
        x[ArmorEkf::Py] + radius * std::sin(psi),
        x[ArmorEkf::Pz] + config.plate_height_m[static_cast<std::size_t>(index)]);
}

// 板位按集合比，不按槽位比。逐块板找最近的真值板，返回最差的那一对距离。
// 四块板等高时"槽位整体挪两格、theta_c 加 pi"与"挪一格、theta_c 减 90 度、
// r1 与 r2 互换"都是精确等价变换，中心、速度、角速度、半径多重集在等价
// 轨道上完全相同，只有槽位标签会换。锁在哪一支都对物理没有影响，所以对
// 板位只能提集合，对标签提等于在断言一个约定。
double worstPlateMatch(const ArmorEkf::StateVector& x, const Truth& truth,
                       const ArmorEkf::Config& config) {
    double worst = 0.0;
    for (int index = 0; index < config.plate_count; ++index) {
        const Eigen::Vector3d predicted = modelPlateWorld(x, index, config);
        double best = std::numeric_limits<double>::infinity();
        for (int other = 0; other < config.plate_count; ++other) {
            best = std::min(
                best, (predicted - truePlateWorld(truth, other, config)).norm());
        }
        worst = std::max(worst, best);
    }
    return worst;
}

// 一帧的观测生成。只喂此刻真实可见的板，判据与 testrig/ekf_trace 的
// visiblePlates 同源：板装在外侧、板面朝外，法向就是板心相对旋转中心的
// 径向，能被检出的板必须满足"法向与指向相机的方向夹角在锥角以内"，
// 也就是板心在车的近侧。
//
// 这是唯一能由真车产生的观测配置，没有第二种。旧夹具无条件发四块板，
// 等于每帧凭空补上两块背对相机的板；那两块板面朝外的方向恰好与世界原点
// 同侧，于是初始化那步"板面朝外即朝相机"的假设被反过来满足，方向写反
// 的版本在测试里反而全绿。初始化相关的一切判断都必须在这条口径下做。
//
// 代价是观测变少：一辆四板车每次只有一到两块可见（现场口径也是"同车最多
// 同时见两块、多数只有一块"），于是偏航与半径只在车自己转起来时才可辨识。
// 所以下面的用例都让车带角速度，帧数预算也按实测重新定过，见各用例注释。
void feedFrame(ArmorEkf& filter, const ArmorEkf::Config& config, const Truth& truth,
               double gimbal_yaw) {
    const Eigen::Matrix3d to_gimbal = ArmorEkf::worldToGimbal(gimbal_yaw);
    for (int index = 0; index < config.plate_count; ++index) {
        if (!plateFacesCamera(truth, index, config)) {
            continue;
        }
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
    // 云台系与世界系轴向相同（前 / 左 / 上），云台正对前方时两系重合。
    // 这条同时钉住了"没有基础变换"这件事：若哪天又冒出一个 90 度的置换，
    // 下面第一个断言就会红。
    const Eigen::Matrix3d m = ArmorEkf::gimbalToWorld(0.0);
    assert((m - Eigen::Matrix3d::Identity()).norm() < 1e-12);
    assert(std::fabs(m.determinant() - 1.0) < 1e-12);

    // 三个轴一一对应。
    assert((m * Eigen::Vector3d(5.0, 0.0, 0.0) - Eigen::Vector3d(5.0, 0.0, 0.0)).norm() <
           1e-12);
    assert((m * Eigen::Vector3d(0.0, 5.0, 0.0) - Eigen::Vector3d(0.0, 5.0, 0.0)).norm() <
           1e-12);
    assert((m * Eigen::Vector3d(0.0, 0.0, 5.0) - Eigen::Vector3d(0.0, 0.0, 5.0)).norm() <
           1e-12);

    // 云台向右转，视线在世界系里朝 -y 走。这条把 gimbal.yaw 的方向钉死：
    // 右为正是它的定义，符号若反了这里立刻红。
    const double yaw = 0.2;
    const Eigen::Vector3d boresight =
        ArmorEkf::gimbalToWorld(yaw) * Eigen::Vector3d(1.0, 0.0, 0.0);
    assert(std::fabs(std::atan2(boresight.y(), boresight.x()) + yaw) < 1e-12);

    // 竖直轴不参与偏航：竖直方向在两系里一致。
    const Eigen::Vector3d up =
        ArmorEkf::gimbalToWorld(yaw) * Eigen::Vector3d(0.0, 0.0, 5.0);
    assert((up - Eigen::Vector3d(0.0, 0.0, 5.0)).norm() < 1e-12);

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
    // 云台系的"正前方"是 +x，与 W 系一致。
    const Eigen::Vector3d boresight =
        ArmorEkf::gimbalToWorld(theta_c) * Eigen::Vector3d(1.0, 0.0, 0.0);
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

// 无噪自洽：观测由被测代码的观测模型自己生成，滤波器必须收敛到真值。
//
// 两处口径与旧版不同，都是为了贴合真车：
//
// 一、只喂可见板。检测器只输出板面朝着相机的板，四块板同时可见的帧在真车上
//     不存在。旧夹具无条件发四块，等于把"看不见"这件事理想化掉，而这个理想化
//     恰好掩盖了初始化那步"板面朝外即朝相机"的假设，方向写反也照样收敛。
//
// 二、板高全零，与出厂 config 的默认一致，走共面这一支。实车是 0/2 等高、
//     1/3 等高且两组不同，但共面下观测模型仍然自洽，自洽检查用哪一支都成立。
//     旧夹具给 2 号板抬高 0.15 m 是为了打破"板环整体挪两格"的等价关系，好让
//     theta_c 的断言唯一。那是靠改物理配置去迁就断言，而实测这个抬高在真实
//     可见性下收敛要几千帧，还随起始朝向时好时坏。等价关系应该在断言里避开，
//     见下面板位判据。
//
// 帧数 2400 是实测定的：可见板 1~2 块、自转 2 rad/s，这一跑中心误差 1.9e-08、
// 板位误差 3.7e-08，界 1e-6 留了两个量级以上余量。
void testNoiselessConvergence() {
    const ArmorEkf::Config config = testConfig();
    ArmorEkf filter(config);

    Truth truth;
    truth.center = Eigen::Vector3d(4.0, 0.3, 0.6);
    truth.velocity = Eigen::Vector3d(0.5, -0.2, 0.0);
    truth.yaw = 0.4;
    truth.yaw_rate = 2.0;

    const double dt = 1.0 / 165.0;
    for (int frame = 0; frame < 2400; ++frame) {
        // 真值先推进到本帧的观测时刻，再让滤波器预测同样的步长并吃下这条观测。
        // 顺序反过来，滤波器收敛到的状态会整整落后一个 v*dt，
        // 误差恰好是 0.5/165 这类量级，看着像"没收敛"，其实是测试的时序错了。
        trueStep(truth, dt);
        filter.predict(dt);
        feedFrame(filter, config, truth, 0.0);
        filter.advanceStateMachine(static_cast<double>(frame + 1) * dt);
    }

    assert(filter.status() == TrackingState::Stable);
    const ArmorEkf::StateVector x = filter.state();
    const Eigen::Vector3d center(x[ArmorEkf::Px], x[ArmorEkf::Py], x[ArmorEkf::Pz]);
    const Eigen::Vector3d velocity(x[ArmorEkf::Vx], x[ArmorEkf::Vy], x[ArmorEkf::Vz]);
    const double plate_error = worstPlateMatch(x, truth, config);
    const double center_error = (center - truth.center).norm();
    std::printf("[ekf] 无噪自洽收敛 中心 %.3e 速度 %.3e 角速度 %.3e 板位 %.3e\n",
                center_error, (velocity - truth.velocity).norm(),
                std::fabs(x[ArmorEkf::YawRate] - truth.yaw_rate), plate_error);
    assert(center_error < 1e-6);
    assert((velocity - truth.velocity).norm() < 1e-6);
    assert(std::fabs(x[ArmorEkf::YawRate] - truth.yaw_rate) < 1e-6);
    // 板位比较绕开了槽位标签，见 worstPlateMatch 的说明。
    assert(plate_error < 1e-6);
    std::printf("[ekf] 无噪自洽收敛 通过\n");
}
// 两块板的车：板间夹角是 pi，一块在车头一块在车尾。
//
// 关键事实：对置的两块板法向相反，80 度可见锥角下两者的可见区间不相交，
// 所以任何时刻这两块板里只有一块能被检出。这条用例实际测的是单板轮换观测
// 下的可观测性，不是"两块板同时看见"。
//
// 这个几何下状态有一支精确的等价变换：槽位整体挪一格（两块板互换方位角）
// 同时 theta_c 加 pi、r1 与 r2 互换，得到同一组板心、同一个中心、同一个
// 角速度。中心、速度、角速度、半径多重集都在等价轨道上不动，只有槽位标签
// 会换。所以断言能对这些量提，不能对标签提：旧版那句 "r1 一定大于 r2"
// 是在断言一个约定，不是在断言物理。
//
// 半径是弱可观测：两块板都在连心线上，中心落点只由比值 r1/(r1+r2) 决定，
// 这个比值只能靠车自己转出来。所以让车快转、跑足够久。转速低时半径收敛慢
// 是几何决定的，不是滤波器的问题。
//
// 帧数 24000 与界 1e-4 都是实测定的：中心 1.9e-06、板位 5.7e-07、
// 角速度 2.2e-06、半径和与半径差 1.5e-07/3.6e-06，留了 25 倍以上余量。
// 12000 帧时还停在 2e-03 量级，收敛速度是单板轮换观测下解耦半径与偏航的
// 速度极限，再往下只能靠车多转几圈。
void testTwoPlateVehicleObservesRadiusTwo() {
    ArmorEkf::Config config = testConfig();
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
    for (int frame = 0; frame < 24000; ++frame) {
        trueStep(truth, dt);
        filter.predict(dt);
        feedFrame(filter, config, truth, 0.0);
        filter.advanceStateMachine(static_cast<double>(frame + 1) * dt);
    }

    const ArmorEkf::StateVector x = filter.state();
    const Eigen::Vector3d center(x[ArmorEkf::Px], x[ArmorEkf::Py], x[ArmorEkf::Pz]);
    const Eigen::Vector3d velocity(x[ArmorEkf::Vx], x[ArmorEkf::Vy], x[ArmorEkf::Vz]);
    const double plate_error = worstPlateMatch(x, truth, config);
    const double radius_sum_error = std::fabs(x[ArmorEkf::Radius1] + x[ArmorEkf::Radius2] -
                                              (truth.radius1 + truth.radius2));
    const double radius_span_error =
        std::fabs(std::fabs(x[ArmorEkf::Radius1] - x[ArmorEkf::Radius2]) -
                  std::fabs(truth.radius1 - truth.radius2));
    std::printf("[ekf] 两板车 中心 %.3e 板位 %.3e 角速度 %.3e 半径和 %.3e 半径差 %.3e "
                "r=(%.6f,%.6f)\n",
                (center - truth.center).norm(), plate_error,
                std::fabs(x[ArmorEkf::YawRate] - truth.yaw_rate), radius_sum_error,
                radius_span_error, x[ArmorEkf::Radius1], x[ArmorEkf::Radius2]);
    assert((center - truth.center).norm() < 1e-4);
    assert((velocity - truth.velocity).norm() < 1e-4);
    assert(std::fabs(x[ArmorEkf::YawRate] - truth.yaw_rate) < 1e-4);
    // 两半径之和是两板间距，是强观测量，先把它钉死。
    assert(radius_sum_error < 1e-4);
    // 关键判据：槽位若按 90 度排，两块板会落进同奇偶的槽位，模型被迫认为
    // r1 == r2，这个差就停在 0 附近收敛不出来。取绝对值就允许两支等价分支。
    assert(radius_span_error < 1e-4);
    assert(plate_error < 1e-4);
    std::printf("[ekf] 两块板 180 度 半径可辨识 通过\n");
}

// 单板初始化的方位角方向回归。这条用例是这次 EKF 初始化事故的钉子。
//
// 车的旋转中心在 (4, 0, 0.5)，车体朝向 pi，于是四块板里只有槽位 0 那块正对
// 相机（板心在 (4 - r, 0, 0.5)，法向指向 -x，正对原点），其余三块板面法向与
// 视线的夹角都在 80 度锥角之外。这是"只看见一块板"最干净的样本，也是实车上
// 最常见的一帧（现场口径：同一辆敌车多数时候只有一块可见）。
//
// 初始化曾经把"原点指向板心"的视线方位当成"中心指向板心"的径向，两者在正对
// 相机时正好差 180 度。误差是 2r*cos(delta/2) 与 2r*sin(delta/2) 之差，delta
// 是这块板被看到的斜角；这里 delta = 0，两个值分别是 2r = 0.5 m 和 0。所以界
// 取 0.05 m：方向写反的版本整整差一个 2r，怎么算都过不去。
//
// 半径取成与 config.initial_radius_m 相同，让这条用例只盯方向，不掺半径先验
// 的误差：单板观测下中心沿视线方向本来就有 2r 的自由度。
//
// 实测：这一跑是个不动点，方向写对的版本最差中心误差与最差偏航误差都是 0；
// 写反的版本正好停在全帧 0.5000 m 与 3.1416 rad（也就是 2r 与 pi）上，
// 120 帧一动不动。
//
// 同时盯住"别在欠定状态上报 Stable"：只看见一块板时旋转中心沿视线可推近，
// 半径也没有第二块板作参照，这时报 Stable 会直接放行开火。
void testSinglePlateInitializationIsOnTheFarSide() {
    const ArmorEkf::Config config = testConfig();
    ArmorEkf filter(config);

    Truth truth;
    truth.center = Eigen::Vector3d(4.0, 0.0, 0.5);
    truth.yaw = kPi;
    truth.yaw_rate = 0.0;
    truth.radius1 = config.initial_radius_m;
    truth.radius2 = config.initial_radius_m;

    const double dt = 1.0 / 165.0;
    double worst_center = 0.0;
    double worst_yaw = 0.0;
    for (int frame = 0; frame < 120; ++frame) {
        filter.predict(dt);
        feedFrame(filter, config, truth, 0.0);
        filter.advanceStateMachine(static_cast<double>(frame + 1) * dt);

        // 初始化要连续 init_frames 帧一致观测，前几帧还没有目标。
        if (frame + 1 < config.init_frames) {
            assert(filter.status() == TrackingState::NoTarget);
            continue;
        }
        const ArmorEkf::StateVector x = filter.state();
        const Eigen::Vector3d center(x[ArmorEkf::Px], x[ArmorEkf::Py], x[ArmorEkf::Pz]);
        worst_center = std::max(worst_center, (center - truth.center).norm());
        worst_yaw = std::max(worst_yaw,
                             std::fabs(ArmorEkf::wrapToPi(x[ArmorEkf::Yaw] - truth.yaw)));
        // 一块板撑不起中心的三个自由度，不许报 Stable。
        assert(filter.status() != TrackingState::Stable);
    }
    std::printf("[ekf] 单板初始化 最差中心 %.3e m 最差 yaw %.3e rad\n",
                worst_center, worst_yaw);
    assert(worst_center < 0.05);
    assert(worst_yaw < 1e-6);
    std::printf("[ekf] 单板初始化落在远侧 通过\n");
}
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
    assert((center - truth.center).norm() < 0.5);
    std::printf("[ekf] 云台转动 角速度 %.3e |v| %.3e 中心 %.4f\n",
                std::fabs(x[ArmorEkf::YawRate]),
                Eigen::Vector3d(x[ArmorEkf::Vx], x[ArmorEkf::Vy], x[ArmorEkf::Vz]).norm(),
                (center - truth.center).norm());
    std::printf("[ekf] 云台转动不产生目标运动 通过\n");
}

void testStateMachineTiming() {
    const ArmorEkf::Config config = testConfig();
    ArmorEkf filter(config);

    Truth truth;
    truth.center = Eigen::Vector3d(4.0, 0.0, 0.4);
    truth.velocity = Eigen::Vector3d(0.3, 0.1, 0.0);
    truth.yaw = 0.2;
    truth.yaw_rate = 2.0;

    const double dt = 1.0 / 165.0;
    double timestamp = 0.0;

    // 帧数与自转速度都是实测定的。自转 2 rad/s、可见板一到两块时，
    // 这条几何第 135 帧进 Stable；初始化的方向若写反是第 457 帧。
    // 800 帧把两条都覆盖住，好让这条用例只测状态机时序，不被初始化连坐。
    for (int frame = 0; frame < 800; ++frame) {
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
        truth.yaw_rate = 2.0;
        const double dt = 1.0 / 165.0;
        // 自转 2 rad/s 时实测第 134 帧进 Stable，800 帧留了余量。
        for (int frame = 0; frame < 800; ++frame) {
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
    truth.yaw_rate = 2.0;

    timestamp += dt;
    filter.predict(dt);
    feedFrame(filter, config, truth, 0.0);
    filter.advanceStateMachine(timestamp);
    assert(filter.status() == TrackingState::NoTarget);

    // 跑够帧数进入稳定态。云台没有对准目标，稳定但不许开火。
    // 自转 2 rad/s 时这条几何实测第 124 帧进 Stable，方向写反是第 577 帧，
    // 800 帧把两条都覆盖住。
    for (int frame = 0; frame < 800; ++frame) {
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

// Web Debug 要靠 PredictionResult 里这几个数把收敛、切换、遮挡恢复画出来。
// 它们是 EKF 内部状态的出口，少了任何一个，页面上就没有直接证据；
// 这里钉住三件事：车体中心的 z 分量、当前瞄准的板编号、命中与丢失的计数。
void testTrackingEvidenceFieldsReachPrediction() {
    const ArmorEkf::Config config = testConfig();
    ArmorEkf filter(config);
    const GimbalState gimbal = makeGimbal(0.0, 0.0);

    const double dt = 1.0 / 165.0;
    double timestamp = 0.0;

    Truth truth;
    truth.center = Eigen::Vector3d(4.0, 0.3, 0.6);
    truth.yaw = 0.4;
    truth.yaw_rate = 2.0;

    // 自转 2 rad/s，实测第 124 帧进 Stable，800 帧留足余量。
    int distinct_aim = 0;
    int last_aim = -2;
    for (int frame = 0; frame < 800; ++frame) {
        timestamp += dt;
        filter.predict(dt);
        feedFrame(filter, config, truth, 0.0);
        filter.advanceStateMachine(timestamp);
        trueStep(truth, dt);
        const PredictionResult result = filter.buildPrediction(gimbal, 0.001);
        // 有目标时瞄准槽位必须落在板数范围内；无目标时用 -1 表示没瞄。
        assert(result.aim_index < config.plate_count);
        if (result.state != TrackingState::NoTarget) {
            assert(result.aim_index >= 0);
            if (result.aim_index != last_aim) {
                ++distinct_aim;
                last_aim = result.aim_index;
            }
        }
    }
    assert(filter.status() == TrackingState::Stable);
    // 车在转，瞄准的板必然换过；一个数都没换说明 aim_index 没接上。
    assert(distinct_aim >= 2);

    // 车体中心三个分量都要报出来。truth 的 z 是 0.6 m，
    // 少报 z 时页面只有二维位置。
    const PredictionResult tracked = filter.buildPrediction(gimbal, 0.001);
    assert(std::isfinite(tracked.center_z_m));
    assert(std::fabs(tracked.center_z_m - truth.center.z()) < 0.05);
    assert(std::isfinite(tracked.center_velocity_z_m_s));

    // 稳态下：连续命中在涨，连续丢失是零。
    assert(tracked.hit_streak > 0);
    assert(tracked.miss_count == 0);

    // 短时遮挡 10 帧。60 ms 远小于 1.5 s 重置门限，状态不许变 NoTarget，
    // 但"在丢"这件事必须能从计数上看出来。
    for (int frame = 0; frame < 10; ++frame) {
        timestamp += dt;
        filter.predict(dt);
        filter.advanceStateMachine(timestamp);
        trueStep(truth, dt);
    }
    const PredictionResult occluded = filter.buildPrediction(gimbal, 0.001);
    assert(occluded.state != TrackingState::NoTarget);
    assert(occluded.miss_count >= 10);
    assert(occluded.hit_streak == 0);

    // 观测回来，丢失计数复位、命中重新累计。
    timestamp += dt;
    filter.predict(dt);
    feedFrame(filter, config, truth, 0.0);
    filter.advanceStateMachine(timestamp);
    trueStep(truth, dt);
    const PredictionResult recovered = filter.buildPrediction(gimbal, 0.001);
    assert(recovered.miss_count == 0);
    assert(recovered.hit_streak > 0);

    // 离群观测：跳变闸之外的一条，必须被拒且计数涨。
    const int rejected_before = recovered.rejected_count;
    ArmorPose outlier;
    outlier.position_gimbal_m = cv::Vec3d(30.0, 0.0, 0.0);
    outlier.position_camera_m = outlier.position_gimbal_m;
    outlier.reprojection_error = 0.3;
    outlier.detection.target_id = 1;
    assert(!filter.update(outlier, 0.0));
    timestamp += dt;
    filter.advanceStateMachine(timestamp);
    const PredictionResult after_outlier = filter.buildPrediction(gimbal, 0.001);
    assert(after_outlier.rejected_count == rejected_before + 1);
    std::printf("[ekf] 跟踪证据字段 通过\n");
}

}  // namespace

int main() {
    testCoordinateTransforms();
    testPlateAzimuthConvention();
    testAnalyticJacobian();
    testNoiselessConvergence();
    testTwoPlateVehicleObservesRadiusTwo();
    testSinglePlateInitializationIsOnTheFarSide();
    testGimbalRotationIsNotTargetMotion();
    testStateMachineTiming();
    testInitializationTakesOneObservationPerFrame();
    testNegativePredictionBiasIsHonored();
    testTrackingEvidenceFieldsReachPrediction();
    testSafetyInvariants();
    std::printf("[ekf] 全部通过\n");
    return 0;
}
