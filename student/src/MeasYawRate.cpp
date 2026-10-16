#include "MeasYawRate.hpp"

#include <cmath>

namespace {

constexpr double kTwoPi = 6.28318530717958647692;

}  // namespace

MeasYawRate::MeasYawRate(const Config& config) : config_(config) {
    // 平滑系数落在 (0, 1] 之外就没有意义：0 会让输出永远停在初值，
    // 负数和无理数直接污染输出。越界一律按不平滑处理，不在这里报错。
    if (!std::isfinite(config_.smoothing_alpha) ||
        config_.smoothing_alpha <= 0.0 || config_.smoothing_alpha > 1.0) {
        config_.smoothing_alpha = 1.0;
    }
}

double MeasYawRate::update(double armor_orientation_yaw, double gimbal_yaw,
                           double timestamp_s) {
    // 非有限输入整帧作废：参考角和参考时刻都不动，缺的这一帧的时间
    // 会摊进下一帧的分母里。见文件头。
    if (!std::isfinite(armor_orientation_yaw) || !std::isfinite(gimbal_yaw) ||
        !std::isfinite(timestamp_s)) {
        return rate_;
    }

    const double world_yaw = armor_orientation_yaw + gimbal_yaw;
    if (!has_previous_) {
        has_previous_ = true;
        previous_world_yaw_ = world_yaw;
        previous_timestamp_s_ = timestamp_s;
        return rate_;
    }

    // 两个方位角各自在 (-pi, pi] 附近，直接相减会在跨界处得到约 2 pi 的假跳变，
    // 先折到 (-pi, pi] 再判断。
    const double delta = std::remainder(world_yaw - previous_world_yaw_, kTwoPi);
    const double elapsed_s = timestamp_s - previous_timestamp_s_;
    // 参考点无条件往前挪，角和时刻一起。跳变帧、间隔过大帧也一样，只是不出值。
    previous_world_yaw_ = world_yaw;
    previous_timestamp_s_ = timestamp_s;

    if (elapsed_s <= 0.0 || elapsed_s > config_.max_dt_s) {
        return rate_;
    }
    if (std::fabs(delta) > config_.jump_gate_rad) {
        return rate_;
    }

    const double raw = -delta / elapsed_s;
    if (config_.smoothing_alpha >= 1.0) {
        // 单独走这一支：rate += 1.0 * (raw - rate) 有浮点往返，
        // 恒速场景下会让输出与真值差几个最低位。这里直接赋值。
        rate_ = raw;
    } else {
        rate_ += config_.smoothing_alpha * (raw - rate_);
    }
    return rate_;
}

void MeasYawRate::reset() {
    has_previous_ = false;
    previous_world_yaw_ = 0.0;
    previous_timestamp_s_ = 0.0;
    rate_ = 0.0;
}
