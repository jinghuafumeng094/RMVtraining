#pragma once

#include "AutoAimTypes.hpp"

#include <array>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

// 传统视觉装甲板检测管线。
// 拆成"纯几何"和"依赖图像"两层：纯几何部分不碰任何像素，
// 可以脱离相机、脱离图片单独测试。
namespace armor {

// ---------------------------------------------------------------------------
// 灯条：用"上端中心 + 下端中心"两个点描述，而不是 cv::RotatedRect。
//
// 为什么不用 RotatedRect：OpenCV 4.5 起 RotatedRect::angle 的取值区间
// 和符号语义都变了（3.x 是 [-90,0)，4.5 是 [0,90)），
// 靠 angle 判断"哪头朝上"会踩坑。
// 两个端点自身就带方向，没有歧义。
// 约定：top.y <= bottom.y（top 是画面里在上面的那一端）。
// ---------------------------------------------------------------------------
struct LightBar {
    cv::Point2f top{};
    cv::Point2f bottom{};

    cv::Point2f center() const { return (top + bottom) * 0.5F; }
    float length() const;       // top 到 bottom 的欧氏距离
    float tiltDegrees() const;  // 与竖直方向的夹角：底端偏右为正
};

// 配对判据。全部是无量纲的比值或角度 —— 与分辨率和距离都无关，
// 所以不需要随画面大小缩放（"参数分三类"里的第一类）。
struct PairCriteria {
    float max_length_ratio{1.8F};         // 长灯条 / 短灯条 的上限
    float max_center_y_diff_ratio{0.6F};  // |Δy| / 平均灯条长度 的上限
    float max_tilt_diff_deg{15.0F};       // 两根灯条倾角差的上限
    float min_gap_ratio{1.2F};            // 中心距 / 平均灯条长度 的下限
    float max_gap_ratio{6.0F};            // 同上，上限
};

// 大小装甲板分界：间距比 ≥ 此值为大装甲板。
// 小装甲板物理 135×57 → ≈2.37；大装甲板 230×57 → ≈4.04。
// 取几何平均 √(2.37×4.04) ≈ 3.09，圆整到 3.0。
constexpr float kGapRatioSplit = 3.0F;

// ---------------------------------------------------------------------------
// 纯几何部分：输入输出都是点，不读图像。
// ---------------------------------------------------------------------------

// 把两根灯条配成装甲板。返回的每一对都保证 first 在左、second 在右。
// 注意：本函数只做"这两根灯条能不能构成一块板"的局部判断。
// 三根等距灯条会产生 3 个合法配对（左中 / 中右 / 左右），
// 去重留给后面的 NMS。
std::vector<std::pair<LightBar, LightBar>> pairLightBars(
    const std::vector<LightBar>& bars, const PairCriteria& criteria = {});

// 由一对灯条生成装甲板的四个角点。顺序是硬约定，全链路都依赖它：
//   c[0] 左上 = 左灯条 top      c[1] 右上 = 右灯条 top
//   c[2] 右下 = 右灯条 bottom   c[3] 左下 = 左灯条 bottom
// 内部有防御：若传入顺序左右颠倒，会自动交换后再生成。
std::array<cv::Point2f, 4> buildCorners(const LightBar& left,
                                        const LightBar& right);

// 四边形有向面积（鞋带公式，图像坐标 y 向下）。
// 顺序正确时为正；接近 0 说明四点退化，
// 这种角点喂给 warpPerspective 会出 NaN。
float quadSignedArea(const std::array<cv::Point2f, 4>& corners);

// 用"灯条中心距 / 灯条长度"判定大小装甲板。
// 不用角点外接矩形的长宽比：板子一倾斜外接矩形就变大，
// 比值被压低，大板会被误判成小板。
ArmorSize classifySize(const LightBar& left, const LightBar& right,
                       float split_ratio = kGapRatioSplit);

}  // namespace armor
