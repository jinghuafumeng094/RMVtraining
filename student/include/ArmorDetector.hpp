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
//
// 一对灯条之间夹着第三根灯条时会被拒绝 —— 那说明这一对跨了两块板。
// 只靠间距比挡不住：实测帧上跨板那对的间距比是 5.05，落在合法区间内。
// 三根等距灯条因此只产生 2 对（左中 / 中右）。剩下的重叠交给 NMS。
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

// ---------------------------------------------------------------------------
// 颜色掩膜 —— 管线第一级
// ---------------------------------------------------------------------------

// 阈值。两个数都是从海康那 69 张整图上量出来的，不是拍的：
//   |B-R| < 20 的像素占 99.78%（背景、场地、地板全在这里）
//   |B-R| ≥ 120 的只占 0.13%（灯条本体）
//   >60 且主通道 >100 之后，一帧只剩 0~40 个斑块
// 60 落在"背景极密、灯条极疏"中间的空档里，两边都有近 5 倍余量，
// 所以在 60 附近挪动不会突然崩，这一点对现场调参很重要。
//
// 差值必须是**带符号**的：蓝队找 B>R，红队找 R>B。
// 用 |B-R| 会把对面颜色的灯条一并收进来，红蓝两队看到一样多的目标 ——
// 而且不报任何错。这是本函数最容易踩的坑。
struct MaskParams {
    int min_channel_diff{60};   // 主通道 - 副通道 的下限
    int min_channel_value{100}; // 主通道自身的下限
};

// 生成二值掩膜：255 = 属于本队颜色的亮像素，0 = 其它。
// 只接受 CV_8UC3 / CV_8UC4（BGR / BGRA）。灰度图**没有颜色**，
// 这里直接返回空矩阵，而不是猜一个通道顺序 —— 猜错不会有任何提示。
cv::Mat buildColorMask(const cv::Mat& bgr, TeamColor team,
                       const MaskParams& params = {});

// ---------------------------------------------------------------------------
// 灯条提取 —— 管线第二级
// ---------------------------------------------------------------------------

// 筛选门槛。单位是**像素** —— 这一组和 PairCriteria 那组恰好相反，
// 是尺度相关的。换分辨率（比如 1440x1080 → 640x480）必须按比例重调，
// 而 PairCriteria 里那些比值一个都不用动。
//
// 门槛同样来自同一批 69 帧的实测：
//   高/宽 的分布 p5=1.79 p25=2.64 p50=3.53 p90=6.25 max=16.0，
//   只有一个斑块不满足 1.5 —— 所以 1.5 这个下限几乎不误杀真灯条；
//   面积 p10=83 p50=364，30 是很松的地板，只负责滤掉噪点碎渣。
struct BarParams {
    float min_area{30.0F};    // 轮廓面积下限，单位平方像素
    float min_height{25.0F};  // 长边下限，单位像素
    float min_aspect{1.5F};   // 长边 / 短边 的下限
    float max_aspect{20.0F};  // 同上，上限。防止一根细线冒充灯条
};

// 从掩膜里提灯条。输入是 buildColorMask 产出的单通道图
// （按非零判断前景，所以 0/1 或 0/255 两种掩膜都收）。
// 返回的每根灯条都满足 top.y <= bottom.y 的约定。
std::vector<LightBar> extractLightBars(const cv::Mat& mask,
                                       const BarParams& params = {});

// ---------------------------------------------------------------------------
// 检测串联 + 去重
// ---------------------------------------------------------------------------

// 整条检测管线的参数。分组是为了现场能只调坏掉的那一组，
// 不用在一堆数字里找是哪个起了作用。
struct DetectParams {
    MaskParams mask{};
    BarParams bar{};
    PairCriteria pair{};
    PatchParams patch{};
    // 角点四边形面积下限，单位平方像素。低于它认为四点已经退化，
    // 直接丢掉。extractDigitPatch 内部还有一道同样的检查，
    // 这里提前拦是为了不在退化角点上白跑一次透视变换。
    float min_quad_area_px{100.0F};
    // NMS 的 IoU 上限：两块板的外接矩形重叠超过它就算同一块。
    // 三根等距灯条会配出 3 对（左中 / 中右 / 左右），这三对互相重叠，
    // 不合并的话同一块板会报三次，下游 EKF 会当成三个目标去跟踪。
    float nms_iou{0.3F};
};

// 非极大值抑制：按外接矩形面积从大到小，贪心保留互不重叠的。
//
// 为什么按面积而不是按置信度排序：面积大说明离得近、像素多，
// 数字本来就更好读；而置信度回答的是"有多确定是某个数字"，
// 跟"这是不是一块真板"是两回事 —— 板子上什么都没有时，
// 分类器会很有把握地说"这是 9neg"，置信度一样可以很高。
std::vector<ArmorDetection> suppressOverlaps(
    const std::vector<ArmorDetection>& detections, float iou_threshold);

// 检测主入口：
//   颜色掩膜 → 灯条提取 → 灯条配对 → 角点 → 大小分类 → NMS → 数字分类
//
// 关于「目标丢失处理」（评分表明确列的一项）：
// 板子检测到了、但中间没有数字时，这块板**仍然返回**（角点有效，PnP 能用），
// 只是 target_id = 0。下游 applyPrediction 用 target_id != 0 把开火闸住。
// 反过来的做法（找不到数字就整块丢掉）会让"检测坏了"和"分类坏了"
// 在 Web Debug 上长得一模一样 —— armor_count 都是 0，没法区分。
//
// team 传本队颜色：只看自己颜色的灯条，对面队伍的车不会变成目标。
std::vector<ArmorDetection> detectArmors(const cv::Mat& bgr, TeamColor team,
                                         const DetectParams& params = {});

}  // namespace armor
