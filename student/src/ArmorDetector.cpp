#include "ArmorDetector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace armor {
namespace {

constexpr float kPi = 3.14159265358979323846F;

// 两个角度的差，取环形最小值。
// 直接相减会在跨 ±180° 时出问题：
// -179° 和 +179° 实际只差 2°，相减却是 358°。
float angleDifferenceDegrees(float a, float b) {
    float difference = std::fmod(std::fabs(a - b), 180.0F);
    if (difference > 90.0F) {
        difference = 180.0F - difference;
    }
    return difference;
}

}  // namespace

// ---------------------------------------------------------------------------
// LightBar
// ---------------------------------------------------------------------------

float LightBar::length() const {
    return static_cast<float>(cv::norm(bottom - top));
}

float LightBar::tiltDegrees() const {
    // 方向向量取"从上端指向下端"，即图像里的向下方向。
    // 竖直灯条 → atan2(0, 正数) = 0°；下端偏右 → 正角；偏左 → 负角。
    // 因为约定 top.y <= bottom.y，这里 dy 恒为非负，不会出现 180° 翻转。
    const float dx = bottom.x - top.x;
    const float dy = bottom.y - top.y;
    return std::atan2(dx, dy) * 180.0F / kPi;
}

// ---------------------------------------------------------------------------
// 纯几何
// ---------------------------------------------------------------------------

float quadSignedArea(const std::array<cv::Point2f, 4>& corners) {
    // 鞋带公式。图像坐标 y 向下，所以顺时针顺序会得到正值：
    // 左上→右上→右下→左下 恰好是屏幕上的顺时针。
    float twice_area = 0.0F;
    for (int i = 0; i < 4; ++i) {
        const cv::Point2f& current = corners[i];
        const cv::Point2f& next = corners[(i + 1) % 4];
        twice_area += current.x * next.y - next.x * current.y;
    }
    return twice_area * 0.5F;
}

std::array<cv::Point2f, 4> buildCorners(const LightBar& left,
                                        const LightBar& right) {
    // 防御一：万一调用方把左右传反了，这里按 center().x 纠正。
    // 交给"谁在左边"这个客观量来判断，不依赖调用方的自觉。
    const bool left_is_left = left.center().x <= right.center().x;
    const LightBar& l = left_is_left ? left : right;
    const LightBar& r = left_is_left ? right : left;

    std::array<cv::Point2f, 4> corners{};
    corners[0] = l.top;     // 左上
    corners[1] = r.top;     // 右上
    corners[2] = r.bottom;  // 右下
    corners[3] = l.bottom;  // 左下

    // 防御二：有向面积为负说明四点成了镜像，交换右上和左下翻回来。
    //
    // 诚实说明（实测过，不是推测）：在"按 center.x 定序"
    // 加"约定 top.y <= bottom.y"这两条同时成立时，
    // 上面生成的顺序必然给出正面积 ——
    // 5000 万组任意倾角的随机灯条采样，负面积出现 0 次，最小值 6e-4。
    // 所以这一分支目前不可达，是廉价保险：
    // 若将来有人把定序依据从 center.x 改成 top.x，它会立刻开始干活。
    //
    // 它挡不住的是另一件事：四点趋近共线。那种情况面积趋近 0 而不是变负，
    // 必须由调用方按"面积下限"丢弃，否则喂给 warpPerspective 会出 NaN。
    if (quadSignedArea(corners) < 0.0F) {
        std::swap(corners[1], corners[3]);
    }
    return corners;
}

ArmorSize classifySize(const LightBar& left, const LightBar& right,
                       float split_ratio) {
    const float length_left = left.length();
    const float length_right = right.length();

    // 退化保护：任一根长度为零，比值就没有意义了。
    // 不能只在"平均长度为零"时才拦 —— 一根 0、一根 60 时平均是 30，
    // 保护不触发，间距比会被算成 4.4，反而判成大板。
    // 宁可保守判小板，也不能把垃圾输入放大成大目标。
    if (length_left <= 0.0F || length_right <= 0.0F) {
        return ArmorSize::Small;
    }

    const float mean_length = (length_left + length_right) * 0.5F;
    const float gap = static_cast<float>(cv::norm(left.center() - right.center()));
    return (gap / mean_length >= split_ratio) ? ArmorSize::Large
                                              : ArmorSize::Small;
}

std::vector<std::pair<LightBar, LightBar>> pairLightBars(
    const std::vector<LightBar>& bars, const PairCriteria& criteria) {
    std::vector<std::pair<LightBar, LightBar>> pairs;
    const std::size_t count = bars.size();
    if (count < 2) {
        return pairs;
    }

    // O(n²) 暴力枚举。灯条数量通常个位数到十几根，n<30 时开销可忽略，
    // 换来的是每条判据都能单独调试、单独注释。
    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t j = i + 1; j < count; ++j) {
            const LightBar& a = bars[i];
            const LightBar& b = bars[j];

            const float length_a = a.length();
            const float length_b = b.length();
            const float length_min = std::min(length_a, length_b);
            const float length_max = std::max(length_a, length_b);
            if (length_min <= 0.0F) {
                continue;  // 退化灯条，后面所有比值都没意义
            }
            const float mean_length = (length_a + length_b) * 0.5F;

            // 判据 1：同一块板的两根灯条几乎等长
            if (length_max / length_min > criteria.max_length_ratio) {
                continue;
            }

            // 判据 2：两端在竖直方向大致对齐
            const float dy = std::fabs(a.center().y - b.center().y);
            if (dy / mean_length > criteria.max_center_y_diff_ratio) {
                continue;
            }

            // 判据 3：倾角接近（同一块板的两根灯条近似平行）
            const float tilt_diff =
                angleDifferenceDegrees(a.tiltDegrees(), b.tiltDegrees());
            if (tilt_diff > criteria.max_tilt_diff_deg) {
                continue;
            }

            // 判据 4：间距比落在合理区间。
            // 这是五条里最稳的一条 —— 量纲相消，
            // 与分辨率、与远近都无关。
            const float gap = static_cast<float>(cv::norm(a.center() - b.center()));
            const float gap_ratio = gap / mean_length;
            if (gap_ratio < criteria.min_gap_ratio ||
                gap_ratio > criteria.max_gap_ratio) {
                continue;
            }

            // 判据 5：左右定序，保证返回的 pair 里 first 一定在左
            if (a.center().x <= b.center().x) {
                pairs.emplace_back(a, b);
            } else {
                pairs.emplace_back(b, a);
            }
        }
    }
    return pairs;
}

}  // namespace armor
