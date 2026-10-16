// MeasYawRate 的离线烟测。
//
// 这不是验收手段，验收以上车实测为准。它负责在上车前把低层错误挡掉：
// 差分值算错、缺帧时把分母算错、换板时蹦出假尖刺、云台自己转时误判成车在转。
// 运行：ctest -R meas_yaw_rate

#include "MeasYawRate.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <initializer_list>

namespace {

constexpr double kPi = 3.14159265358979323846;

// 实测满帧率，见 docs/上车手册.md。
constexpr double kDt = 1.0 / 165.0;

// 造一帧的 armor_orientation_yaw。
//
// 被测量的推导见 MeasYawRate.hpp 顶部：世界系方位角 theta_world 的下降速率
// 就是车体自转速率，所以
//     theta_world = armor_yaw + gimbal_yaw
//     omega_car   = -d(theta_world)/dt
// 反过来，要造一个"车体以 omega_car 自转"的场景，就让 theta_world 以
// -omega_car 的速度变化，再按上面的式子反解出每一帧该喂什么 armor_yaw。
//
// 取 remainder 是为了复现真值来源：armor_yaw 在 armor_solve 里由 atan2 算出，
// 天然落在 (-pi, pi]，会跨界。不取的话测试就绕过了跨界那条路径。
double armorYawOf(double theta_world, double gimbal_yaw) {
    return std::remainder(theta_world - gimbal_yaw, 2.0 * kPi);
}

// 检查 1：同一块板恒速自转，差分值等于真值。
// 正负两个方向都走一遍，把符号钉死。
void testSamePlateConstantRate() {
    for (const double omega_car : {2.0, -3.0}) {
        MeasYawRate meas;
        double theta_world = 0.4;
        double t = 0.0;
        double rate = 0.0;
        // 500 帧约 3 秒，中途 theta_world 会跨过 -pi 和 +pi 各一次。
        for (int frame = 0; frame < 500; ++frame) {
            rate = meas.update(armorYawOf(theta_world, 0.0), 0.0, t);
            theta_world -= omega_car * kDt;
            t += kDt;
        }
        assert(std::fabs(rate - omega_car) < 1e-9);
        std::printf("[meas] 恒速同板 omega=%.2f rad/s 差分=%.9f 通过\n",
                    omega_car, rate);
    }
}

// 检查 2：换板那一帧输出不跳。
// 换板在数据上就是世界方位角整体平移 90 度，车体本身没有任何变化。
// 平板 PnP 两解翻转同理，平移 180 度。
// 两者都必须被闸门挡住，输出保持上一帧的值，且下一帧立刻恢复正常。
void testPlateSwitchDoesNotSpike() {
    for (const double phase_jump : {0.5 * kPi, kPi}) {
        MeasYawRate meas;
        const double omega_car = 1.5;
        double theta_world = 0.0;
        double t = 0.0;
        double rate = 0.0;

        // 先跑 50 帧把输出建立起来。
        for (int frame = 0; frame < 50; ++frame) {
            rate = meas.update(armorYawOf(theta_world, 0.0), 0.0, t);
            theta_world -= omega_car * kDt;
            t += kDt;
        }
        const double before = rate;
        assert(std::fabs(before - omega_car) < 1e-9);

        // 跳变帧：相位平移，车没动。
        theta_world += phase_jump;
        const double on_jump = meas.update(armorYawOf(theta_world, 0.0), 0.0, t);
        assert(std::fabs(on_jump - before) < 1e-12);

        // 跳变帧一过就该恢复。若闸门漏了自己挪参考点，这里会连错两帧。
        theta_world -= omega_car * kDt;
        t += kDt;
        const double after = meas.update(armorYawOf(theta_world, 0.0), 0.0, t);
        assert(std::fabs(after - omega_car) < 1e-9);

        std::printf("[meas] 换板跳 %.0f 度 输出保持 %.9f 下一帧恢复 %.9f 通过\n",
                    phase_jump * 180.0 / kPi, on_jump, after);
    }
}

// 检查 3：车不动、只有云台在转，输出必须是零。
// 这条专钉符号。少了 gimbal_yaw 会得到 +1.0，符号写反会得到 +2.0。
void testGimbalSlewAloneGivesZero() {
    MeasYawRate meas;
    const double theta_world = 0.7;      // 板在世界系里纹丝不动
    const double omega_gimbal = 1.0;     // 云台以 1 rad/s 向右转
    double gimbal_yaw = 0.0;
    double t = 0.0;
    double rate = 0.0;
    for (int frame = 0; frame < 300; ++frame) {
        rate = meas.update(armorYawOf(theta_world, gimbal_yaw), gimbal_yaw, t);
        gimbal_yaw += omega_gimbal * kDt;
        t += kDt;
    }
    assert(std::fabs(rate) < 1e-9);
    std::printf("[meas] 云台自转 %.2f rad/s 车不动 差分=%.3e 通过\n",
                omega_gimbal, rate);
}

// 检查 4：缺帧时，分母是参考帧到本帧真实经过的时间，不是一帧的长度。
// 遮挡是常态，这条错了会让速率为遮挡帧数倍地放大，看起来像车突然抽了一下。
void testSkippedFramesUseRealElapsedTime() {
    MeasYawRate meas;
    // 车以 1 rad/s 自转，每帧 10 ms，所以每帧方位角变化 -0.01 rad。
    // 第 1 帧只立参考点，第 2 帧有观测，第 3 帧没有，第 4 帧又有。
    // 分母必须是第 2 帧到第 4 帧的 20 ms。若错用一帧的 10 ms，会得到 2.0。
    constexpr double kStep = 0.01;
    double t = 0.0;
    assert(std::fabs(meas.update(0.0, 0.0, t)) < 1e-15);
    t += kStep;
    meas.update(-0.01, 0.0, t);
    t += kStep;
    meas.update(std::nan(""), 0.0, t);
    t += kStep;
    const double rate = meas.update(-0.03, 0.0, t);
    assert(std::fabs(rate - 1.0) < 1e-9);
    std::printf("[meas] 缺 1 帧 分母用 20 ms 差分=%.9f 通过\n", rate);
}

// 检查 5：参考帧到本帧的间隔过大时不出值。
// 丢失一秒钟之后，前后两点之间车可能已经转过好几圈，做差没有意义。
void testLongGapHoldsLastValue() {
    MeasYawRate meas;
    double t = 0.0;
    meas.update(0.0, 0.0, t);
    t += kDt;
    const double held = meas.update(-0.01, 0.0, t);
    assert(std::fabs(held - 0.01 / kDt) < 1e-9);

    t += 1.0;
    const double after_gap = meas.update(1.0, 0.0, t);
    assert(std::fabs(after_gap - held) < 1e-12);

    // 参考点要跟着挪，否则下一帧会拿一秒前的点再跳一次。
    t += kDt;
    const double resumed = meas.update(1.0 - 0.01, 0.0, t);
    assert(std::fabs(resumed - 0.01 / kDt) < 1e-9);
    std::printf("[meas] 间隔 1 s 输出保持 %.6f 下一帧恢复 %.6f 通过\n",
                after_gap, resumed);
}

// 检查 6：首帧、reset、非法输入都不产生假值。
void testFirstFrameResetAndBadInput() {
    MeasYawRate meas;
    double t = 0.0;
    assert(std::fabs(meas.update(1.0, 0.5, t)) < 1e-15);

    t += kDt;
    const double rate = meas.update(1.02, 0.5, t);
    assert(std::fabs(rate + 0.02 / kDt) < 1e-9);

    // 非有限输入既不挪参考角，也不挪参考时刻，所以这一帧缺失的时间
    // 会摊进下一帧的分母里，速率不被放大。见检查 4。
    t += kDt;
    assert(std::fabs(meas.update(std::nan(""), 0.5, t) - rate) < 1e-12);
    t += kDt;
    assert(std::fabs(meas.update(1.04, 0.5, t) + 0.02 / (2.0 * kDt)) < 1e-9);

    meas.reset();
    t += kDt;
    assert(std::fabs(meas.update(9.0, 0.5, t)) < 1e-15);
    std::printf("[meas] 首帧 reset 非法输入 通过\n");
}

}  // namespace

int main() {
    testSamePlateConstantRate();
    testPlateSwitchDoesNotSpike();
    testGimbalSlewAloneGivesZero();
    testSkippedFramesUseRealElapsedTime();
    testLongGapHoldsLastValue();
    testFirstFrameResetAndBadInput();
    std::printf("[meas] 全部通过\n");
    return 0;
}
