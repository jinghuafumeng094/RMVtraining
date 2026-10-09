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

// ---------------------------------------------------------------------------
// 依赖图像的部分：把装甲板的四个角点变成数字分类器要吃的 28x20 补丁。
// ---------------------------------------------------------------------------

// 裁剪参数。全部带默认值，现场要调的就是这几个数。
struct PatchParams {
    // 第一级：把角点四边形正射到"保物理比例的规范板"时，每毫米画几个像素。
    // 2.0 是起步值：再大只是让中间那张图更贵，对最终 20x28 没有增益；
    // 再小（比如 1.0）会让 114 像素高的板降到 57，数字笔画开始丢。
    float px_per_mm{2.0F};
    // 装甲板物理尺寸，单位 mm。大小板高度相同，只有宽度不同。
    // 依据见仓库根目录 装甲板尺寸.txt。
    float plate_height_mm{57.0F};
    float small_width_mm{135.0F};
    float large_width_mm{230.0F};
    // 数字框的宽高比，必须等于数据集补丁的比例 20:28。
    // 这个数字不是拍脑袋：它和"框高 = 整块板高"一起，
    // 把框宽唯一定了出来 —— 57 * 20/28 ≈ 40.7mm。
    float digit_aspect{20.0F / 28.0F};
    int out_width{20};
    int out_height{28};
    // 极性开关。数据集是"白数字 + 黑底"，正常运行不需要动它；
    // 万一场上抠出来是反的（黑数字白底），翻这个开关比重新训练快得多。
    bool invert{false};
};

// 从整图里抠出数字补丁。
//
//   image   : 原图，CV_8UC1 / 8UC3 / 8UC4 都收，内部转灰度
//   corners : 逆时针/顺时针都行，但顺序必须是 左上、右上、右下、左下
//             （buildCorners 就是这个顺序）
//   size    : 大板还是小板。它决定第一级正射用的物理宽高比。
//
// 成功时 patch 是 cv::Mat 的 28 行 x 20 列、CV_8UC1，取值只可能是 {0, 255}。
// 失败（图空、类型不支持、四点退化）返回 false，并把 patch 置空。
//
// 为什么是两级变换而不是一次 getPerspectiveTransform 到 20x28：
// 装甲板物理是 135:57 ≈ 2.37（横的），数据集补丁是 20:28 ≈ 0.71（竖的）。
// 一步到位会把数字横向压扁约 3 倍，而且不会报错，只会准确率掉下来。
bool extractDigitPatch(const cv::Mat& image,
                       const std::array<cv::Point2f, 4>& corners, ArmorSize size,
                       cv::Mat& patch, const PatchParams& params = {});

}  // namespace armor
