#include "ArmorDetector.hpp"

#include "DigitClassifier.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <opencv2/imgproc.hpp>

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

    // 防御二：四点成了镜像时（有向面积为负）交换右上/左下翻回来。
    // 按上面的定序这条不会触发，是收便宜的保险 —— 但要是哪天定序依据
    // 从 center.x 改成 top.x，没有它就会安静地给出镜像角点。
    // 它挡不住"四点趋近共线"：那种情况面积趋近 0 而不是变负。
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

            // 判据 5：这一对之间不能夹着第三根灯条。
            //
            // 真实装甲板中间不会竖着第三根灯条；夹着的那一对是**跨了两块板**。
            // 光靠间距比挡不住它：在实测的 20261008231729930 那帧上，
            // 跨板那对 (514, 909) 的间距比是 5.05，正好落在合法区间 [1.2, 6.0] 里，
            // 五条判据全过。而它的外接矩形最大，NMS 按面积取大正好留它下车 ——
            // 结果裁剪框落到了中间那根灯条上，"数字补丁"其实是灯条本身。
            //
            // 纵向只在两根灯条**共同覆盖的高度带**内判，避免把背景里
            // 远处、纵向错开的目标误当成"夹在中间"。
            const float pair_top = std::min(a.top.y, b.top.y);
            const float pair_bottom = std::max(a.bottom.y, b.bottom.y);
            const float x_lo = std::min(a.center().x, b.center().x);
            const float x_hi = std::max(a.center().x, b.center().x);
            bool has_bar_inside = false;
            for (std::size_t k = 0; k < count && !has_bar_inside; ++k) {
                if (k == i || k == j) {
                    continue;
                }
                const cv::Point2f c = bars[k].center();
                if (c.x <= x_lo || c.x >= x_hi) {
                    continue;  // 不在这一对的横向区间内
                }
                if (c.y <= pair_top || c.y >= pair_bottom) {
                    continue;  // 纵向错开，不算夹在中间
                }
                has_bar_inside = true;
            }
            if (has_bar_inside) {
                continue;
            }

            // 判据 6：左右定序，保证返回的 pair 里 first 一定在左
            if (a.center().x <= b.center().x) {
                pairs.emplace_back(a, b);
            } else {
                pairs.emplace_back(b, a);
            }
        }
    }
    return pairs;
}

// ---------------------------------------------------------------------------
// 依赖图像：数字补丁裁剪
// ---------------------------------------------------------------------------

namespace {

// 四点退化到多小就不再可用，单位是平方像素。
// 取 1.0 而不是 0.0：面积趋近 0 时 getPerspectiveTransform 的解已经不可信，
// 算出来的矩阵可能含 inf/NaN，warpPerspective 会把 NaN 抹到整幅输出上。
// 直接拒绝，比让 NaN 流到后面去炸要容易查得多。
constexpr float kMinQuadAreaPx = 1.0F;

// 转单通道灰度。只认 8 位的两种常见布局，别的一律拒绝。
// 猜一种错误的转换方式，比明确拒绝更难查。
bool toGray(const cv::Mat& image, cv::Mat& gray) {
    switch (image.type()) {
        case CV_8UC1:
            gray = image;  // 不拷贝，只共享数据
            return true;
        case CV_8UC3:
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            return true;
        default:
            return false;
    }
}

}  // namespace

bool extractDigitPatch(const cv::Mat& image,
                       const std::array<cv::Point2f, 4>& corners, ArmorSize size,
                       cv::Mat& patch, const PatchParams& params) {
    // 先清空。无论成功还是失败，调用方拿到的都是干净状态 ——
    // 和 get_pic 用同一个约定，免得调用方误用上一帧的补丁。
    patch.release();

    if (image.empty() || params.px_per_mm <= 0.0F || params.out_width <= 0 ||
        params.out_height <= 0) {
        return false;
    }
    if (std::fabs(quadSignedArea(corners)) < kMinQuadAreaPx) {
        return false;
    }

    cv::Mat gray;
    if (!toGray(image, gray)) {
        return false;
    }

    // ---- 第一级：把角点四边形正射到"保物理比例的规范板" ----
    // 目标尺寸 = 物理毫米数 x 像素密度。大小板只有宽度不同，高度都是 57mm。
    // 注意这一步同时把透视畸变也一起去掉了：四边形被拉成矩形，
    // 所以斜着看板子也能得到端正的数字。
    const float plate_width_mm = (size == ArmorSize::Large)
                                     ? params.large_width_mm
                                     : params.small_width_mm;
    const int canonical_w = std::max(
        1, static_cast<int>(std::lround(plate_width_mm * params.px_per_mm)));
    const int canonical_h = std::max(
        1, static_cast<int>(
               std::lround(params.plate_height_mm * params.px_per_mm)));

    const std::vector<cv::Point2f> source(corners.begin(), corners.end());
    const std::vector<cv::Point2f> destination{
        {0.0F, 0.0F},
        {static_cast<float>(canonical_w - 1), 0.0F},
        {static_cast<float>(canonical_w - 1), static_cast<float>(canonical_h - 1)},
        {0.0F, static_cast<float>(canonical_h - 1)}};

    const cv::Mat warp = cv::getPerspectiveTransform(source, destination);
    cv::Mat canonical;
    cv::warpPerspective(gray, canonical, warp,
                        cv::Size(canonical_w, canonical_h), cv::INTER_LINEAR);

    // ---- 第二级：从规范板中央裁出数字框 ----
    // 框高 = 整块板高。为什么不按字形的实际大小裁：数据集里量过，
    // 墨迹高度只占补丁高的 0.68~0.95 倍，说明人家裁的时候就留了余量。
    // 我们按字形裁会得到一个比训练数据更紧的裁剪，分布对不上。
    //
    // 框宽不是独立参数：它和框高、纵横比三者只有两个自由度。
    // 纵横比定成 20:28（= 数据集补丁的比例），框宽就唯一确定了：
    //   57mm * 20/28 ≈ 40.7mm
    // 这个 40.7 不是我拍的，是算出来的。
    const int box_h = canonical_h;
    const int box_w =
        std::max(1, static_cast<int>(std::lround(box_h * params.digit_aspect)));
    // 理论上板子总比数字框宽。万一日后有人把参数调反了，
    // 夹回图内，宁可裁小也不要越界访问。
    const int safe_w = std::min(box_w, canonical_w);
    const int box_x = (canonical_w - safe_w) / 2;  // 居中

    const cv::Mat box = canonical(cv::Rect(box_x, 0, safe_w, box_h));

    // ---- 缩放 + 二值化 ----
    // 顺序很重要：必须"先缩放、再阈值"。
    // 反过来的话，INTER_AREA 会把 0/255 平均成中间灰度，输出就不再是
    // 训练数据那种 {0,255} 二值图 —— 而且不会报任何错，只是准确率悄悄掉。
    cv::Mat resized;
    cv::resize(box, resized, cv::Size(params.out_width, params.out_height), 0.0,
               0.0, cv::INTER_AREA);

    // Otsu：阈值由这张小图自己的直方图算出来，不用手调，
    // 所以对现场的光照变化比固定阈值皮实。
    // 数据集是"白数字 + 黑底"，所以用 THRESH_BINARY。
    // 万一现场抠出来极性相反，把下面这个常量换成 THRESH_BINARY_INV。
    cv::threshold(resized, patch, 0.0, 255.0,
                  cv::THRESH_BINARY | cv::THRESH_OTSU);
    return true;
}

// ---------------------------------------------------------------------------
// 颜色掩膜
// ---------------------------------------------------------------------------

cv::Mat buildColorMask(const cv::Mat& bgr, TeamColor team,
                       const MaskParams& params) {
    // 只收三通道 BGR。灰度图没有颜色、BGRA 该由调用方先转掉
    // （detectArmors 在入口统一做这件事）。
    // 这里直接返回空矩阵，而不是内部猜一个通道顺序 ——
    // 猜错不会有任何提示，整条管线会安静地全错。
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        return cv::Mat();
    }

    cv::Mat channels[3];
    cv::split(bgr, channels);  // 0 = B，1 = G，2 = R

    const bool is_blue = (team == TeamColor::Blue);
    const cv::Mat& main = is_blue ? channels[0] : channels[2];
    const cv::Mat& other = is_blue ? channels[2] : channels[0];

    // 有符号的差。cv::subtract 对 CV_8U 是**饱和**减法：
    // 负数截成 0，所以蓝队这里等价于 max(B - R, 0)，正是想要的"只留本队颜色"。
    //
    // 换成 cv::absdiff 就是 |B - R| —— 红队和蓝队看到一模一样的结果，
    // 对面队伍的车全都变成目标。这是本函数唯一的坑，而且它不报错。
    cv::Mat diff;
    cv::subtract(main, other, diff);

    cv::Mat strong;  // 通道差够大
    cv::Mat bright;  // 主通道本身够亮
    cv::threshold(diff, strong, params.min_channel_diff, 255.0, cv::THRESH_BINARY);
    cv::threshold(main, bright, params.min_channel_value, 255.0, cv::THRESH_BINARY);

    // 两条判据是"与"关系。只用差值的话，一个通道差 70 的暗灰蓝像素
    // （比如 80/60/10）也会进来；只用亮度的话，白光（255/255/255）
    // 差值只有 0 倒是不进，可偏色的高光就会漏进来。
    cv::Mat mask;
    cv::bitwise_and(strong, bright, mask);
    return mask;
}

// ---------------------------------------------------------------------------
// 灯条提取
// ---------------------------------------------------------------------------

std::vector<LightBar> extractLightBars(const cv::Mat& mask,
                                       const BarParams& params) {
    std::vector<LightBar> bars;
    if (mask.empty() || mask.type() != CV_8UC1) {
        return bars;
    }

    std::vector<std::vector<cv::Point>> contours;
    // RETR_EXTERNAL：只要最外层轮廓。
    // 灯条过曝时中间会烧成白、边缘反而偏暗，用 RETR_LIST / RETR_TREE
    // 会把同一根灯条拆成里外两三个轮廓，每一圈都被当成独立灯条去配对，
    // 于是同一块板配出一堆假目标 —— 而且看起来"检测很灵"。
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    bars.reserve(contours.size());
    for (const std::vector<cv::Point>& contour : contours) {
        if (cv::contourArea(contour) < params.min_area) {
            continue;
        }

        const cv::RotatedRect rect = cv::minAreaRect(contour);

        // 用四条边的长度分长短边，**刻意不看 rect.angle**。
        // OpenCV 4.5 起 angle 的取值区间和符号语义都改过（3.x 是 [-90,0)，
        // 4.5 是 [0,90)），靠它判断哪条边是长边会随版本变；
        // 直接比边长没有任何歧义。
        cv::Point2f pts[4];
        rect.points(pts);
        const float edge0 = static_cast<float>(cv::norm(pts[1] - pts[0]));
        const float edge1 = static_cast<float>(cv::norm(pts[2] - pts[1]));
        const bool edge0_is_short = edge0 <= edge1;
        const float short_side = edge0_is_short ? edge0 : edge1;
        const float long_side = edge0_is_short ? edge1 : edge0;

        // 长边门槛。它比的是**灯条自己的长度**（沿灯条方向），
        // 不是画面里的竖直高度 —— 斜着的灯条竖直跨度可以远小于长度。
        // 这个参数原来叫 min_height，那个名字已经误导过一次讨论，别再改回去。
        // 它同时是"最远能看多远"的那个闸，理由写在 BarParams 的注释里。
        if (short_side <= 0.0F || long_side < params.min_long_side) {
            continue;
        }
        const float aspect = long_side / short_side;
        if (aspect < params.min_aspect || aspect > params.max_aspect) {
            continue;
        }

        // 灯条的两端 = 两条**短边**的中点。
        // 矩形四点是按顺序绕一圈的，所以短边要么是 (0,1) 和 (2,3)，
        // 要么是 (1,2) 和 (3,0) —— 由上面比出来的结果选一组。
        LightBar bar;
        if (edge0_is_short) {
            bar.top = (pts[0] + pts[1]) * 0.5F;
            bar.bottom = (pts[2] + pts[3]) * 0.5F;
        } else {
            bar.top = (pts[1] + pts[2]) * 0.5F;
            bar.bottom = (pts[3] + pts[0]) * 0.5F;
        }
        // 落到约定的 top.y <= bottom.y。minAreaRect 给的点序
        // 跟朝向有关，不保证哪一头在上面，所以必须显式对齐一次 ——
        // tiltDegrees() 的正负号全押在这个约定上。
        if (bar.top.y > bar.bottom.y) {
            std::swap(bar.top, bar.bottom);
        }
        bars.push_back(bar);
    }
    return bars;
}

// ---------------------------------------------------------------------------
// 去重
// ---------------------------------------------------------------------------

namespace {

// 四边形角点的轴对齐外接矩形。
//
// 注意 cv::boundingRect 对**点集**用的是含两端的约定：
// 宽 = x_max - x_min + 1，高 = y_max - y_min + 1。
// 所以四点跨 200 像素时给出的宽是 201，不是一个笔误。
// 后果是这里的 IoU 会比"按坐标区间手算"略大一点点
// （实测 201x61 与 101x61 的重叠是 0.5025 而不是 0.5）。
// 对 NMS 这种只要相对大小的判据没有影响，但对照手算数字时要记得这件事。
cv::Rect2f boundsOf(const std::array<cv::Point2f, 4>& corners) {
    return cv::boundingRect(std::vector<cv::Point2f>(corners.begin(),
                                                     corners.end()));
}

float intersectionOverUnion(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float intersection = (a & b).area();  
    const float total = a.area() + b.area() - intersection;
    return total > 0.0F ? intersection / total : 0.0F;
}

}  // namespace

std::vector<ArmorDetection> suppressOverlaps(
    const std::vector<ArmorDetection>& detections, float iou_threshold) {
    std::vector<ArmorDetection> kept;
    if (detections.empty()) {
        return kept;
    }

    std::vector<cv::Rect2f> boxes;
    boxes.reserve(detections.size());
    for (const ArmorDetection& detection : detections) {
        boxes.push_back(boundsOf(detection.corners));
    }

    // 按面积从大到小排。
    //
    // 为什么不是按置信度：置信度回答的是"有多确定这是某个数字"，
    // 不是"这是不是一块真板"。板子上没有数字时分类器会很有把握地
    // 报"9neg"，照样给出高置信度。面积大则同时说明离得近、像素多、
    // 数字更好读，是更强的"这是个真目标"的证据。
    std::vector<std::size_t> order(detections.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(),
              [&boxes](std::size_t a, std::size_t b) {
                  return boxes[a].area() > boxes[b].area();
              });

    std::vector<bool> suppressed(detections.size(), false);
    for (std::size_t i = 0; i < order.size(); ++i) {
        const std::size_t candidate = order[i];
        if (suppressed[candidate]) {
            continue;
        }
        kept.push_back(detections[candidate]);
        for (std::size_t j = i + 1; j < order.size(); ++j) {
            const std::size_t other = order[j];
            if (suppressed[other]) {
                continue;
            }
            if (intersectionOverUnion(boxes[candidate], boxes[other]) >
                iou_threshold) {
                suppressed[other] = true;
            }
        }
    }
    return kept;
}

// ---------------------------------------------------------------------------
// 检测串联
// ---------------------------------------------------------------------------

std::vector<ArmorDetection> detectArmors(const cv::Mat& bgr, TeamColor team,
                                         const DetectParams& params) {
    std::vector<ArmorDetection> detections;

    // 通道格式在**入口归一化一次**，后面每一级都只面对三通道 BGR。
    // 已经是三通道时不复制，只是让两个名字指向同一块数据。
    // 归一化只做这一处：分散到各级去做的话，多一处就多一处漏掉的可能。
    cv::Mat bgr_only;
    if (bgr.type() == CV_8UC4) {
        cv::cvtColor(bgr, bgr_only, cv::COLOR_BGRA2BGR);
    } else if (bgr.type() == CV_8UC3) {
        bgr_only = bgr;
    }
    if (bgr_only.empty()) {
        return detections;  // 空图、灰度图、其它位深：明确拒绝
    }

    const cv::Mat mask = buildColorMask(bgr_only, team, params.mask);
    if (mask.empty()) {
        return detections;
    }

    const std::vector<LightBar> bars = extractLightBars(mask, params.bar);
    if (bars.size() < 2) {
        return detections;  // 一根灯条配不出板
    }

    const std::vector<std::pair<LightBar, LightBar>> pairs =
        pairLightBars(bars, params.pair);

    detections.reserve(pairs.size());
    for (const std::pair<LightBar, LightBar>& pair : pairs) {
        const std::array<cv::Point2f, 4> corners =
            buildCorners(pair.first, pair.second);
        // 四点退化：面积趋近 0 时透视变换的解已经不可信，
        // 硬算下去会把 inf/NaN 抹到整幅输出上。提前丢掉。
        if (std::fabs(quadSignedArea(corners)) < params.min_quad_area_px) {
            continue;
        }

        ArmorDetection detection;
        detection.corners = corners;
        detection.size = classifySize(pair.first, pair.second);

        cv::Mat patch;
        if (extractDigitPatch(bgr_only, corners, detection.size, patch,
                              params.patch)) {
            const DigitResult digit = DigitClassifier::classify(patch);
            detection.target_id = digit.target_id;
            detection.confidence = digit.confidence;
        }
        // 抠补丁失败（或分类器不可用）时**不 continue**。
        // 角点是对的，PnP 还能用；target_id 保持默认的 0，
        // 由下游把开火闸住。把这块板丢掉的话，"检测坏了"和"分类坏了"
        // 在 Web Debug 上会变得一模一样（armor_count 都是 0）。
        detections.push_back(detection);
    }

    return suppressOverlaps(detections, params.nms_iou);
}

}  // namespace armor
