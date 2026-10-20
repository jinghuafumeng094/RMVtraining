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
    // 2026-10-20 靶车实测旋转中心到板心 27.3 cm，四块板近似等大。夹具留一点
    // 不等，是为了让两个半径各自的观测性在这些用例里也能被测到；两个数都必须
    // 落在 Config 的物理范围内，否则 clamp 会先把真值吃掉。
    double radius1{0.30};
    double radius2{0.285};
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

// 真值板面法向在云台系里的偏航角，右为正，与 armor_solve 的 armor_yaw 同口径。
//
// 物体系 +z 指板背（背离相机），板面朝外，所以它就是板心径向 u(psi) 的反向。
// 云台系里 u(psi) 的方位角是 -(psi + gimbal_yaw)，反向再加 pi，
// atan2(-y, x) 读出来正好是 pi - (psi + gimbal_yaw)。
// 这条式子与 tests/test_armor_solve.cpp 第 5 节同源：那里相机转 phi 时
// armor_yaw 必须等于 phi，把置换方向与 atan2 的两个符号都钉住了。
double trueArmorYaw(const Truth& truth, int index, const ArmorEkf::Config& config,
                    double gimbal_yaw) {
    const double psi =
        truth.yaw + static_cast<double>(index) * config.plate_spacing_rad;
    return ArmorEkf::wrapToPi(kPi - (psi + gimbal_yaw));
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
// noise_sigma_m 大于零时给板心叠一层扰动，用来检验 NIS 这类残差统计量是不是活的。
// 扰动用确定性的三角叠加而不是随机数：同一个 frame 永远得到同一个值，
// 跑多少遍结果都一样，断言不会偶发。三个轴取不成比例的角频率,看上去就是噪声。
ArmorPose makeObservation(const Eigen::Vector3d& z, int target_id, double armor_yaw) {
    ArmorPose observation;
    observation.position_gimbal_m = cv::Vec3d(z.x(), z.y(), z.z());
    observation.position_camera_m = observation.position_gimbal_m;
    observation.armor_yaw = armor_yaw;
    observation.reprojection_error = 0.3;
    observation.detection.target_id = target_id;
    return observation;
}

void feedFrame(ArmorEkf& filter, const ArmorEkf::Config& config, const Truth& truth,
               double gimbal_yaw, double noise_sigma_m = 0.0, int frame = 0) {
    const Eigen::Matrix3d to_gimbal = ArmorEkf::worldToGimbal(gimbal_yaw);
    for (int index = 0; index < config.plate_count; ++index) {
        if (!plateFacesCamera(truth, index, config)) {
            continue;
        }
        Eigen::Vector3d z = to_gimbal * truePlateWorld(truth, index, config);
        if (noise_sigma_m > 0.0) {
            const double n = static_cast<double>(frame) + 0.37 * static_cast<double>(index);
            z.x() += noise_sigma_m * std::sin(1.7 * n + 0.3);
            z.y() += noise_sigma_m * std::sin(2.3 * n + 1.1);
            z.z() += noise_sigma_m * std::sin(3.1 * n + 2.7);
        }
        filter.update(makeObservation(z, index + 1, trueArmorYaw(truth, index, config, gimbal_yaw)),
                      gimbal_yaw);
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

// NIS 是残差相对「H P H^T + R」的马氏距离平方，3 维观测下理论均值等于观测维数 3，
// 是唯一能检验 Q/R 填得准不准的量。这里钉住"有样本时这个数必须真的随观测残差动"。
// 窗口为空时的哨兵语义由 testNisSentinelBeforeFirstUpdate 单独负责：本用例开头
// 那个未初始化的滤波器走的是 buildPrediction 的 NoTarget 提前返回，拿到的只是
// 结构体默认值，钉不住任何一行转换。
//
// 不拿「均值等于 3」当断言。夹具的观测是真值加扰动造的，R 用的是滤波器自己那套，
// 等于用同一把尺子量自己：R 整体系数填错，NIS 均值照样是 3。那个校准只能上车实测，
// 见 docs/上车手册.md 第六节。
void testNisMeanReachesPrediction() {
    const ArmorEkf::Config config = testConfig();
    const GimbalState gimbal = makeGimbal(0.0, 0.0);
    const double dt = 1.0 / 165.0;

    // 还没建起滤波器时窗口里没有样本，出口必须是负的哨兵值。
    ArmorEkf idle(config);
    assert(idle.buildPrediction(gimbal, 0.001).nis_mean < 0.0);

    // 同一段真值跑两遍，只差观测扰动的大小，其余完全一样，这样比较才有意义。
    const auto run = [&](double noise_sigma_m) {
        ArmorEkf filter(config);
        Truth truth;
        truth.center = Eigen::Vector3d(4.0, 0.3, 0.6);
        truth.yaw = 0.4;
        truth.yaw_rate = 2.0;
        double timestamp = 0.0;
        for (int frame = 0; frame < 800; ++frame) {
            timestamp += dt;
            filter.predict(dt);
            feedFrame(filter, config, truth, 0.0, noise_sigma_m, frame);
            filter.advanceStateMachine(timestamp);
            trueStep(truth, dt);
        }
        assert(filter.status() == TrackingState::Stable);
        return filter.buildPrediction(gimbal, 0.001).nis_mean;
    };

    const float clean = run(0.0);
    const float perturbed = run(0.02);

    // 干净观测下残差只剩线性化误差，远小于滤波器自己声明的 R；
    // 实测这个量在 1e-6 量级，界留得很宽。
    assert(clean >= 0.0F);
    assert(clean < 1.0F);
    // 换上真实存在的扰动必须涨上来。恒等于常数、或者接错成 P 的对角线，都会倒在这条上。
    assert(std::isfinite(perturbed));
    assert(perturbed > clean);
    assert(perturbed < 50.0F);
    std::printf("[ekf] NIS 均值 干净 %.6f 扰动 %.4f 通过\n", clean, perturbed);
}

// NIS 的两个哨兵值服务两处语义不同的判据，任何一个写成另一个都会静默失效。
//
// 内部：recentNisMean() 在窗口为空时返回正无穷，使「< nis_mean_gate」为假，
// 也就是没有 NIS 历史就不许进 Stable。改成 -1 会让这道门恒真，整组闸门白设。
// 出口：buildPrediction() 把正无穷换成 -1，页面据此跳过这一帧。
// predictionIsFinite 不查 nis_mean，漏了转换会把无穷大一路打到曲线上。
//
// 必须用「刚初始化、还没接受过更新」这一刻来测。未初始化的滤波器在
// buildPrediction 里对 NoTarget 提前返回，拿到的是结构体默认值，两条契约都走不到。
// 这一时刻在真实运行里必然出现：tryInitialize 把状态置成 Unstable 并清零 NIS 窗口。
void testNisSentinelBeforeFirstUpdate() {
    const ArmorEkf::Config config = testConfig();
    const GimbalState gimbal = makeGimbal(0.0, 0.0);
    const double dt = 1.0 / 165.0;

    ArmorEkf filter(config);
    Truth truth;
    truth.center = Eigen::Vector3d(4.0, 0.3, 0.6);
    truth.yaw = 0.4;

    // 每帧只喂一块板。同一帧的第二条观测会走关联更新，把 NIS 写进窗口，
    // 那样就测不到"窗口为空"这一刻了。
    const Eigen::Vector3d plate = truePlateWorld(truth, 0, config);
    double timestamp = 0.0;
    for (int frame = 0; frame < config.init_frames; ++frame) {
        timestamp += dt;
        filter.predict(dt);
        filter.update(makeObservation(plate, 1, trueArmorYaw(truth, 0, config, 0.0)), 0.0);
        filter.advanceStateMachine(timestamp);
    }
    assert(filter.status() == TrackingState::Unstable);

    // 内部哨兵：窗口为空必须是正无穷，绝不满足 Stable 门。
    const double nis_mean = filter.recentNisMean();
    assert(std::isinf(nis_mean));
    assert(nis_mean > 0.0);
    assert(!(nis_mean < config.nis_mean_gate));

    // 出口哨兵：同一时刻打进 PredictionResult 的必须是负值。
    // 顺带断言这条预测确实走到了提前返回之外，否则又是在默认值上打转。
    const PredictionResult prediction = filter.buildPrediction(gimbal, 0.001);
    assert(prediction.state == TrackingState::Unstable);
    assert(prediction.nis_mean < 0.0F);
    assert(std::isfinite(prediction.distance_m));
    assert(prediction.distance_m > 0.0F);
    std::printf("[ekf] NIS 哨兵 窗口空 %.1f 出口 %.1f 通过\n", nis_mean,
                static_cast<double>(prediction.nis_mean));
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
    // 本用例验两个半径各自的观测性。半径差是弱可观测方向，信噪比靠差值本身
    // 撑起来：产线真车 27.3 cm 的半径差只有几毫米，比 5 cm 量级的深度噪声
    // 还小，24000 帧内学不出来。这里用一辆 0.30/0.19 的大车，先验和过程
    // 噪声也放回旧值，保持原有帧数与 1e-4 的收敛预算。
    config.initial_radius_m = 0.25;
    config.initial_radius_std_m = 0.15;
    config.min_radius_m = 0.05;
    config.max_radius_m = 0.50;
    config.radius_drift_m = 0.2;
    config.radius_variance_floor_m2 = 1.0e-6;
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
    ArmorEkf::Config config = testConfig();
    // 本用例盯的是"半径只有先验支撑时不许报 Stable"。生产配置把半径按实测值
    // 钉到 2 cm，P 的半径门会直接放开，单板也会进 Stable；那不是这条用例要
    // 证明的欠定场景，所以这里保留旧版的松先验。
    config.initial_radius_std_m = 0.15;
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

// armor_yaw 是板面法向在云台系里的偏航角，板面朝外，所以它就是车体偏航的
// 直接读数：h = pi - (Yaw + i * spacing + gimbal_yaw)，对 Yaw 的偏导恒为 -1。
//
// 为什么必须要它：一帧一块板只给两个位置约束，车心 (2 个) 加偏航加半径共
// 四个未知量。偏航与车心沿切向的分量在单板下完全混在一起，位置残差按方差
// 分配，谁方差大谁吃，于是滤出的偏航和半径都是编的。v4 回放里半径塌到几
// 厘米、yaw 跑飞就是这条。接上第四维观测后偏航单独被钉住，位置观测只剩
// 车心与半径的径向分量要分。
//
// 用例：板心固定不动，位置观测对任何偏航都成立，所以只靠位置的旧版会一直
// 停在初始值。先喂一串错的 armor_yaw，确认闸门挡得住；再喂正确的那条，
// 偏航必须转过去。
void testArmorYawObservationPinsBodyYaw() {
    const ArmorEkf::Config config = testConfig();
    ArmorEkf filter(config);

    Truth truth;
    truth.center = Eigen::Vector3d(4.0, 0.0, 0.5);
    truth.yaw = kPi;
    truth.yaw_rate = 0.0;
    truth.radius1 = config.initial_radius_m;
    truth.radius2 = config.initial_radius_m;

    const Eigen::Vector3d plate = truePlateWorld(truth, 0, config);
    const double dt = 1.0 / 165.0;
    double timestamp = 0.0;
    const auto step = [&](double armor_yaw) {
        timestamp += dt;
        filter.predict(dt);
        filter.update(makeObservation(plate, 1, armor_yaw), 0.0);
        filter.advanceStateMachine(timestamp);
    };

    for (int frame = 0; frame < config.init_frames; ++frame) {
        step(trueArmorYaw(truth, 0, config, 0.0));
    }
    assert(filter.status() != TrackingState::NoTarget);
    assert(std::fabs(ArmorEkf::wrapToPi(filter.state()[ArmorEkf::Yaw] - kPi)) < 1e-6);

    // 错 1.5 rad，远在 armor_yaw_gate_rad 之外，一条都不许吃。
    for (int frame = 0; frame < 50; ++frame) {
        step(-1.5);
    }
    assert(std::fabs(ArmorEkf::wrapToPi(filter.state()[ArmorEkf::Yaw] - kPi)) < 0.02);

    // 板面法向对应 Yaw = pi + 0.2，板心位置不变，位置观测分不出这两种偏航。
    const double target_yaw = kPi + 0.2;
    const double measurement = ArmorEkf::wrapToPi(kPi - target_yaw);
    for (int frame = 0; frame < 150; ++frame) {
        step(measurement);
    }
    const double settled = filter.state()[ArmorEkf::Yaw];
    const double error = std::fabs(ArmorEkf::wrapToPi(settled - target_yaw));
    std::printf("[ekf] armor_yaw 钉偏航 误差 %.4f rad\n", error);
    assert(error < 0.03);
    std::printf("[ekf] armor_yaw 钉住车体偏航 通过\n");
}

// 半径是估计量，不是连续量里的普通状态：它必须落在车体几何可能的范围里。
// 旧版没有任何边界，初始化给 0.9 m 就一路 0.9 m，更新能把 r2 拉到 -0.14，
// 回放里 r=0.05 / -0.03 那次就是它。夹回来的时候中心要按夹过的半径反推，
// 否则滤波器一建立就与刚吃下的那条观测矛盾，下一帧起全被 NIS 拒掉。
void testRadiusStaysWithinPhysicalBounds() {
    const double dt = 1.0 / 165.0;
    for (const double initial : {0.9, 0.01}) {
        ArmorEkf::Config config = testConfig();
        config.initial_radius_m = initial;
        ArmorEkf filter(config);

        Truth truth;
        truth.center = Eigen::Vector3d(4.0, 0.0, 0.5);
        truth.yaw = kPi;
        truth.yaw_rate = 0.0;
        truth.radius1 = 0.25;
        truth.radius2 = 0.25;

        const Eigen::Vector3d plate = truePlateWorld(truth, 0, config);
        double timestamp = 0.0;
        for (int frame = 0; frame < config.init_frames; ++frame) {
            timestamp += dt;
            filter.predict(dt);
            filter.update(makeObservation(plate, 1, trueArmorYaw(truth, 0, config, 0.0)), 0.0);
            filter.advanceStateMachine(timestamp);
        }
        const ArmorEkf::StateVector x = filter.state();
        assert(x[ArmorEkf::Radius1] >= config.min_radius_m - 1e-12);
        assert(x[ArmorEkf::Radius1] <= config.max_radius_m + 1e-12);
        assert(x[ArmorEkf::Radius2] >= config.min_radius_m - 1e-12);
        assert(x[ArmorEkf::Radius2] <= config.max_radius_m + 1e-12);
        // 夹过之后模型仍要复现刚吃下的那条观测。
        const Eigen::Vector3d model = modelPlateWorld(x, 0, config);
        assert((model - plate).norm() < 1e-9);
    }

    // 跑起来之后也不许越界。
    const ArmorEkf::Config config = testConfig();
    ArmorEkf filter(config);
    Truth truth;
    truth.center = Eigen::Vector3d(4.0, 0.3, 0.6);
    truth.yaw = 0.4;
    truth.yaw_rate = 2.0;
    double timestamp = 0.0;
    double lowest = std::numeric_limits<double>::infinity();
    double highest = -std::numeric_limits<double>::infinity();
    for (int frame = 0; frame < 800; ++frame) {
        timestamp += dt;
        filter.predict(dt);
        feedFrame(filter, config, truth, 0.0);
        filter.advanceStateMachine(timestamp);
        trueStep(truth, dt);
        const ArmorEkf::StateVector x = filter.state();
        if (filter.status() == TrackingState::NoTarget) {
            continue;
        }
        lowest = std::min({lowest, x[ArmorEkf::Radius1], x[ArmorEkf::Radius2]});
        highest = std::max({highest, x[ArmorEkf::Radius1], x[ArmorEkf::Radius2]});
        assert(x[ArmorEkf::Radius1] >= config.min_radius_m - 1e-12);
        assert(x[ArmorEkf::Radius1] <= config.max_radius_m + 1e-12);
        assert(x[ArmorEkf::Radius2] >= config.min_radius_m - 1e-12);
        assert(x[ArmorEkf::Radius2] <= config.max_radius_m + 1e-12);
    }
    std::printf("[ekf] 半径边界 全程 [%.3f, %.3f]\n", lowest, highest);
    std::printf("[ekf] 半径不越界 通过\n");
}

// 连续拒收必须能自己爬出来，不能只等 reset_timeout_s 的整段重置。
//
// 拒收不改状态也不改协方差，收敛之后的 P 又被压得极小，一条偏差半米的
// 正确观测算出的马氏距离远超 nis_gate，于是永远进不去：v4 回放里连续 115
// 帧 no-target、f=840-980 每帧全拒都是这个自锁。修法是在连拒达到
// reject_recovery_frames 帧后按 reject_inflation_factor 放大 P 的对角线，
// 把闸门重新撑开。
void testRejectedObservationsInflateCovarianceAndRecover() {
    // 两板模型：四板时把板心横向挪开半米，最近的那一槽就换人了，关联会跳到
    // 相邻槽位上，测到的是换板而不是自锁。两板间距 pi，挪 0.3 m 之后 0 号槽
    // 仍是最近的一槽。
    ArmorEkf::Config config = testConfig();
    config.plate_count = 2;
    config.plate_spacing_rad = kPi;
    ArmorEkf filter(config);

    Truth truth;
    truth.center = Eigen::Vector3d(4.0, 0.0, 0.5);
    truth.yaw = kPi;
    truth.yaw_rate = 0.0;
    truth.radius1 = config.initial_radius_m;
    truth.radius2 = config.initial_radius_m;

    const Eigen::Vector3d plate = truePlateWorld(truth, 0, config);
    const double armor_yaw = trueArmorYaw(truth, 0, config, 0.0);
    const double dt = 1.0 / 165.0;
    double timestamp = 0.0;
    const auto feed = [&](const Eigen::Vector3d& p) {
        timestamp += dt;
        filter.predict(dt);
        filter.update(makeObservation(p, 1, armor_yaw), 0.0);
        filter.advanceStateMachine(timestamp);
    };

    // 收敛。静态单板跑 200 帧后 P 的横向分量已经很小。
    for (int frame = 0; frame < 200; ++frame) {
        feed(plate);
    }
    assert(filter.hitStreak() > 0);

    // 每帧横向挪 3 mm，把状态带到偏 0.3 m 的地方。每步都在闸内，滤波器会
    // 老实跟过去。偏移量取 0.3 而不是 0.18：收敛后的横向方差很小，0.18 的
    // 马氏距离仍可能落在 nis_gate 里，那样测的就不是自锁。上限是 jump_gate_m
    // 的 0.6，超过它走的是另一条分支。
    const double offset_m = 0.3;
    Eigen::Vector3d shifted = plate;
    const int ramp_frames = static_cast<int>(offset_m / 0.003);
    for (int frame = 0; frame < ramp_frames; ++frame) {
        shifted.y() += 0.003;
        feed(shifted);
    }
    // 再原地停住，让横向方差收回去。斜坡本身是持续机动，方差会一直撑着，
    // 不等它收下来就切回真值，这条观测会被正常吃下，测不到任何东西。
    for (int frame = 0; frame < 200; ++frame) {
        feed(shifted);
    }
    assert(std::fabs(filter.state()[ArmorEkf::Py] - offset_m) < 0.06);
    const double locked_std =
        std::sqrt(std::max(0.0, filter.covariance()(ArmorEkf::Py, ArmorEkf::Py)));

    // 真值回到原位。偏差 0.45 m 对已经收小的 P 是远超 nis_gate 的马氏距离，
    // 头几帧必须被拒；连拒之后 P 被放大，同一批观测必须重新被吃下。
    const int rejected_before = filter.rejectedCount();
    int recovered_at = -1;
    for (int frame = 0; frame < 60; ++frame) {
        feed(plate);
        if (recovered_at < 0 && filter.hitStreak() > 0) {
            recovered_at = frame;
        }
        if (frame == 10) {
            assert(filter.rejectedCount() > rejected_before);
        }
        assert(filter.status() != TrackingState::NoTarget);
    }
    std::printf("[ekf] 连续拒收恢复 锁定横向 %.3f m 标准差 %.2e 第 %d 帧重新吃下 "
                "横向 %.4f m\n",
                offset_m, locked_std, recovered_at, filter.state()[ArmorEkf::Py]);
    assert(recovered_at >= 0);
    assert(recovered_at < 60);
    assert(std::fabs(filter.state()[ArmorEkf::Py]) < 0.03);
    std::printf("[ekf] 连续拒收后自行恢复 通过\n");
}

// 同一帧里同一块板出两个检测框，第二条必须认出来是同一块板。
//
// 一帧内一个槽位只配一条观测，第二条只能去找剩下的槽位。两板间距 pi 时
// 对面那一槽就在 2r = 0.5 m 外，落在跳变闸 0.6 m 以内；协方差被放大之后
// 马氏距离也进得了 nis_gate，于是旧版把它配到对面那一槽，一次更新就把
// 两个半径拉到边界。v4 回放 f=137 那次就是这条，r=0.234 -> 0.100/0.100。
// armor_yaw 直接读出是第几块板：它解出来的槽位本帧已经被占用，就说明这条
// 是同一块板的重复框，当场丢掉。
void testSameFrameDuplicatePlateIsRejectedByYaw() {
    ArmorEkf::Config config = testConfig();
    config.plate_count = 2;
    config.plate_spacing_rad = kPi;

    Truth truth;
    truth.center = Eigen::Vector3d(4.0, 0.0, 0.5);
    truth.yaw = kPi;
    truth.radius1 = config.initial_radius_m;
    truth.radius2 = config.initial_radius_m;

    const Eigen::Vector3d plate = truePlateWorld(truth, 0, config);
    const double armor_yaw = trueArmorYaw(truth, 0, config, 0.0);
    const double dt = 1.0 / 165.0;

    ArmorEkf settled(config);
    double timestamp = 0.0;
    for (int frame = 0; frame < 200; ++frame) {
        timestamp += dt;
        settled.predict(dt);
        settled.update(makeObservation(plate, 1, armor_yaw), 0.0);
        settled.advanceStateMachine(timestamp);
    }
    assert(settled.hitStreak() > 0);

    // 丢一段时间，让协方差按拒收放大到初始量级。这正是同一块板出两个框
    // 时能穿过 nis_gate 的条件：收敛后的方差太小，对面那一槽算出来的马氏
    // 距离大到根本进不来，测不到本条要测的东西。
    for (int frame = 0; frame < 12; ++frame) {
        timestamp += dt;
        settled.predict(dt);
        settled.advanceStateMachine(timestamp);
    }

    timestamp += dt;
    settled.predict(dt);
    assert(settled.update(makeObservation(plate, 1, armor_yaw), 0.0));
    const ArmorEkf::StateVector after_first = settled.state();
    const int rejected_before = settled.rejectedCount();

    assert(!settled.update(makeObservation(plate, 1, armor_yaw), 0.0));
    assert(settled.rejectedCount() == rejected_before + 1);
    const ArmorEkf::StateVector after_second = settled.state();
    double worst = 0.0;
    for (int i = 0; i < ArmorEkf::Dimension; ++i) {
        worst = std::max(worst, std::fabs(after_second[i] - after_first[i]));
    }
    std::printf("[ekf] 同帧重复框 第二框拒收 状态最大变化 %.2e\n", worst);
    assert(worst < 1e-9);
    assert(after_second[ArmorEkf::Radius1] >= config.min_radius_m - 1e-12);
    assert(after_second[ArmorEkf::Radius2] >= config.min_radius_m - 1e-12);
    assert(after_second[ArmorEkf::Radius1] <= config.max_radius_m + 1e-12);
    assert(after_second[ArmorEkf::Radius2] <= config.max_radius_m + 1e-12);
    std::printf("[ekf] 同帧重复框按 armor_yaw 判重 通过\n");
}

// 模型离每一块板都超过 jump_gate_m 时，旧状态已经和观测无关了。
// 旧版只有 reset_timeout_s = 1.5 s 一条出口，回放里 493 条观测就这么被
// 跳变闸挡掉，车明明在视野里却一直 no-target。新行为：连续丢
// reinit_after_misses 帧之后，把新观测喂进初始化窗口，攒够 init_frames 条
// 彼此一致的观测就地重建滤波器。
void testDivergedFilterReinitializesFromConsistentObservations() {
    ArmorEkf::Config config = testConfig();
    config.plate_count = 2;
    config.plate_spacing_rad = kPi;

    Truth truth;
    truth.center = Eigen::Vector3d(4.0, 0.0, 0.5);
    truth.yaw = kPi;
    truth.radius1 = config.initial_radius_m;
    truth.radius2 = config.initial_radius_m;

    const Eigen::Vector3d plate = truePlateWorld(truth, 0, config);
    const double armor_yaw = trueArmorYaw(truth, 0, config, 0.0);
    const Eigen::Vector3d far_plate = plate + Eigen::Vector3d(0.0, 3.0, 0.0);

    ArmorEkf filter(config);
    const double dt = 1.0 / 165.0;
    double timestamp = 0.0;
    const auto feed = [&](const Eigen::Vector3d& p) {
        timestamp += dt;
        filter.predict(dt);
        const bool taken = filter.update(makeObservation(p, 1, armor_yaw), 0.0);
        filter.advanceStateMachine(timestamp);
        return taken;
    };
    for (int frame = 0; frame < 200; ++frame) {
        feed(plate);
    }
    assert(filter.hitStreak() > 0);

    // 车在两帧之间瞬移到 3 m 外。物理上不可能，但它是"模型彻底错了"的最小
    // 复现：每一条观测都在所有槽位的跳变闸之外。真实回放里它由换车或
    // 回放循环造成。
    const int rejected_before = filter.rejectedCount();
    int rebuilt_at = -1;
    for (int frame = 0; frame < 10; ++frame) {
        feed(far_plate);
        if (rebuilt_at < 0) {
            const Eigen::Vector3d model =
                modelPlateWorld(filter.state(), 0, config);
            if ((model - far_plate).norm() < 0.1) {
                rebuilt_at = frame;
            }
        }
        assert(filter.status() != TrackingState::NoTarget);
    }
    std::printf("[ekf] 跳闸外重建 第 %d 帧 拒收 %d 条\n", rebuilt_at,
                filter.rejectedCount() - rejected_before);
    assert(rebuilt_at >= 0);
    assert(rebuilt_at < 10);
    // 重建之后必须继续正常吃观测。
    assert(feed(far_plate));
    std::printf("[ekf] 模型偏离跳闸外时按新观测重建 通过\n");
}

void testGimbalRotationIsNotTargetMotion() {
    const ArmorEkf::Config config = testConfig();
    ArmorEkf filter(config);

    Truth truth;
    truth.center = Eigen::Vector3d(3.0, 1.0, 0.5);
    truth.velocity = Eigen::Vector3d::Zero();
    truth.yaw = 0.7;
    truth.yaw_rate = 0.0;
    // 实车四块板近似等大。这条用例只盯云台转动，真值半径与先验取同一个数。
    truth.radius1 = config.initial_radius_m;
    truth.radius2 = config.initial_radius_m;

    const double dt = 1.0 / 165.0;
    // 半径与偏航是弱可观测方向，收敛速度随半径变大而变慢。半径 9.75 cm 时
    // 500 帧的 yaw_rate 残差已经远低于 1e-6；半径按实测改成 27.3 cm 后，
    // 500 帧只到 4.2e-6，1500 帧降到 6.6e-9，留出三个数量级余量。
    for (int frame = 0; frame < 1500; ++frame) {
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
    // 同样 30 ms 的 bias 推出的角度正比于瞄准点的切向速度。半径按实测
    // 27.3 cm 取值后，板随车转的那一份速度约为旧几何 9.75 cm 时的三倍，
    // 角度差仍远大于数值噪声。阈值保持 5e-4。
    assert(std::fabs(d_later) > 5e-4);
    assert(std::fabs(d_earlier) > 5e-4);
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

// 下坠补偿在算法侧做，上报的 target_pitch 必须是弹道发射仰角 beta，
// 不是瞄准点的几何仰角。
//
// beta 与几何仰角之差有闭式：目标与云台同高、水平距离 d 时约等于
// g * d / (2 * v^2)，v = 24 时每米约 0.0085 弧度，5 m 处约 0.043 弧度。
// 空气阻力让飞行时间比无阻力长，补偿再大一点，5 m 处约 0.0006 弧度。
// 所以带宽取 0.85 到 1.20 倍。
void testBallisticDropCompensation() {
    ArmorEkf::Config config = testConfig();
    // 本条只验仰角门，把自转门放到不碍事，好让稳定态能进。
    config.fire_spin_tolerance_rad_s = 100.0;

    ArmorEkf filter(config);
    const GimbalState level = makeGimbal(0.0, 0.0);

    const double dt = 1.0 / 165.0;
    double timestamp = 0.0;

    // 目标与云台同高，于是到任何一块板的几何仰角都是 0，
    // 上报的 target_pitch 全部来自下坠补偿。
    Truth truth;
    truth.center = Eigen::Vector3d(5.0, 0.3, 0.0);
    truth.yaw = 0.4;
    truth.yaw_rate = 2.0;

    for (int frame = 0; frame < 800; ++frame) {
        timestamp += dt;
        filter.predict(dt);
        feedFrame(filter, config, truth, 0.0);
        filter.advanceStateMachine(timestamp);
        trueStep(truth, dt);
    }
    assert(filter.status() == TrackingState::Stable);

    const PredictionResult result = filter.buildPrediction(level, 0.001);
    const double expected = 9.81 * result.distance_m / (2.0 * 24.0 * 24.0);
    // 几何仰角是 0，所以指令仰角就是补偿量本身，不能用小于容差的数糊过去。
    assert(result.target_pitch > 0.85 * expected);
    assert(result.target_pitch < 1.20 * expected);

    // 开火门必须与上报的指令同一个口径。云台先转到目标方位、仰角留在水平，
    // 此时"到几何瞄点"的仰角误差是 0，而指令仰角还差一整个补偿量。
    // 门若量的是前者，这一条会放行。
    const GimbalState aimed = makeGimbal(result.target_yaw, 0.0);
    const PredictionResult probe = filter.buildPrediction(aimed, 0.001);
    assert(std::fabs(probe.target_pitch - aimed.pitch) >
           config.fire_pitch_tolerance_rad);
    assert(!probe.fire_allowed);

    // 云台跟到指令上就该放行。门若量的是补偿前的几何角，这里永远进不去。
    const GimbalState served = makeGimbal(probe.target_yaw, probe.target_pitch);
    const PredictionResult settled = filter.buildPrediction(served, 0.001);
    assert(std::fabs(settled.target_pitch - served.pitch) <
           config.fire_pitch_tolerance_rad);
    assert(settled.fire_allowed);
    std::printf("[ekf] 下坠补偿 指令仰角 %.5f 闭式 %.5f 距离 %.3f 通过\n",
                static_cast<double>(result.target_pitch), expected,
                static_cast<double>(result.distance_m));
}

void testMuzzleOffsetIsSubtractedInGimbalFrame() {
    ArmorEkf::Config config = testConfig();
    const double offset = config.muzzle_offset_m;
    assert(offset > 0.0);

    // 接缝层：弹道输入从枪口量起。云台系的正前方是 +x，枪口在 P 轴前 offset。
    const ArmorEkf::BallisticInputs forward =
        ArmorEkf::ballisticInputsFromMuzzle(Eigen::Vector3d(5.0, 0.0, 0.0), offset);
    assert(std::fabs(forward.horizontal_m - (5.0 - offset)) < 1e-12);
    assert(std::fabs(forward.vertical_m) < 1e-12);

    // 枪口与云台原点等高，所以高度差原样通过，只有水平距离少一个偏移。
    const ArmorEkf::BallisticInputs raised =
        ArmorEkf::ballisticInputsFromMuzzle(Eigen::Vector3d(5.0, 3.0, 0.4), offset);
    assert(std::fabs(raised.horizontal_m - std::hypot(5.0 - offset, 3.0)) < 1e-12);
    assert(std::fabs(raised.vertical_m - 0.4) < 1e-12);

    // 同一个物理点在云台系里只有一个坐标，与云台转到哪无关。换成在 W 里减常向量
    // (offset, 0, 0) 就随 yaw 漂：目标在云台系正前方 5 m、yaw = 90 度时它在 W 里
    // 是 (0, -5, 0)，减出来 5.000549 m，比对的 4.9259 m 多出 74.649 mm。误差比
    // 一个 offset 还多 0.55 mm，因为那个偏移在 W 里转成了垂直分量，hypot 会把
    // 它的平方项加进来，约 offset^2/(2*5)。
    const double subtracted_in_world = std::hypot(0.0 - offset, -5.0);
    const double world_error = subtracted_in_world - forward.horizontal_m;
    assert(world_error > offset);
    assert(std::fabs(world_error - offset - offset * offset / (2.0 * 5.0)) < 1e-6);

    // 端到端：同一个场景跑两遍，只差枪口偏移这一个数。偏移沿炮管，把弹道的
    // 水平距离缩短一个 offset，指令仰角就少约 g*offset/(2v^2)，与距离无关。
    const auto run = [](double muzzle_offset) {
        ArmorEkf::Config run_config = testConfig();
        run_config.fire_spin_tolerance_rad_s = 100.0;
        run_config.muzzle_offset_m = muzzle_offset;
        ArmorEkf filter(run_config);
        const double dt = 1.0 / 165.0;
        double timestamp = 0.0;
        Truth truth;
        truth.center = Eigen::Vector3d(5.0, 0.3, 0.0);
        truth.yaw = 0.4;
        truth.yaw_rate = 2.0;
        for (int frame = 0; frame < 800; ++frame) {
            timestamp += dt;
            filter.predict(dt);
            feedFrame(filter, run_config, truth, 0.0);
            filter.advanceStateMachine(timestamp);
            trueStep(truth, dt);
        }
        return filter.buildPrediction(makeGimbal(0.0, 0.0), 0.001);
    };

    const PredictionResult with_offset = run(offset);
    const PredictionResult without_offset = run(0.0);

    // 偏移只进弹道解，不进状态、不进观测：两遍的低层状态逐位相同。
    // distance_m 却不完全一样，因为它量的是预测后的瞄点，而预测时长里含着
    // 飞行时间，偏移让飞行时间少了约 3.2 ms，瞄点在这段时间里挪了 0.38 mm。
    // 这一条同时钉住"distance_m 没有跟着挪到枪口"：若挪了，这里会差出
    // 一个整偏移 74 mm。
    const double distance_shift =
        std::fabs(static_cast<double>(with_offset.distance_m) -
                  static_cast<double>(without_offset.distance_m));
    assert(distance_shift < 0.005);
    assert(distance_shift > 0.0);

    const double expected_drop = 9.81 * offset / (2.0 * 24.0 * 24.0);
    const double drop = static_cast<double>(without_offset.target_pitch) -
                        static_cast<double>(with_offset.target_pitch);
    assert(drop > 0.85 * expected_drop);
    assert(drop < 1.20 * expected_drop);
    std::printf("[ekf] 枪口偏移 %.4f m 指令仰角差 %.3e rad 闭式 %.3e "
                "距离漂移 %.4f mm 通过\n",
                offset, drop, expected_drop, distance_shift * 1000.0);
}

}  // namespace

int main() {
    testCoordinateTransforms();
    testPlateAzimuthConvention();
    testAnalyticJacobian();
    testNoiselessConvergence();
    testTwoPlateVehicleObservesRadiusTwo();
    testSinglePlateInitializationIsOnTheFarSide();
    testArmorYawObservationPinsBodyYaw();
    testRadiusStaysWithinPhysicalBounds();
    testRejectedObservationsInflateCovarianceAndRecover();
    testSameFrameDuplicatePlateIsRejectedByYaw();
    testDivergedFilterReinitializesFromConsistentObservations();
    testGimbalRotationIsNotTargetMotion();
    testStateMachineTiming();
    testInitializationTakesOneObservationPerFrame();
    testNegativePredictionBiasIsHonored();
    testTrackingEvidenceFieldsReachPrediction();
    testNisMeanReachesPrediction();
    testNisSentinelBeforeFirstUpdate();
    testBallisticDropCompensation();
    testMuzzleOffsetIsSubtractedInGimbalFrame();
    testSafetyInvariants();
    std::printf("[ekf] 全部通过\n");
    return 0;
}
