#include "ArmorDetector.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace {

// 造一根竖直灯条：中心 x 给定，y 从 top_y 到 bottom_y。
armor::LightBar bar(float cx, float top_y, float bottom_y) {
    return armor::LightBar{{cx, top_y}, {cx, bottom_y}};
}

// 造一根倾斜灯条：从上端 (top_x, top_y) 画到给定的偏移处。
// 用来构造"两根灯条不平行"的用例 —— 判据 3 只有靠它才测得到。
armor::LightBar tilted(float top_x, float top_y, float dx, float dy) {
    return armor::LightBar{{top_x, top_y}, {top_x + dx, top_y + dy}};
}

// 按"中心 + 长度 + 倾角"造灯条。tiltDegrees() 的逆运算，
// 用来把真实帧上量到的几何原样搬进用例。
armor::LightBar barFrom(float cx, float cy, float length, float tilt_deg) {
    const float rad = tilt_deg * 3.14159265F / 180.0F;
    const float dx = std::sin(rad) * length * 0.5F;
    const float dy = std::cos(rad) * length * 0.5F;
    return armor::LightBar{{cx - dx, cy - dy}, {cx + dx, cy + dy}};
}

bool near(float a, float b, float eps = 1e-3F) {
    return std::fabs(a - b) < eps;
}

// 造一块"物理比例正确"的假装甲板，正中间画一个正方形亮块，然后抠补丁。
//   plate_width_mm : 135（小板）或 230（大板）；高度按 57mm 走
//   marker_mm      : 中间那个方块的边长，单位 mm
//   bg / fg        : 背景与亮块的灰度。默认 0/255（数据集就是这两个值）；
//                    压低对比度用来验证 Otsu 真的在自适应光照。
// 方块是正方形，所以抠出来也必须接近正方形 —— 这是纵横比的判据。
cv::Mat makePlateAndExtract(float px_per_mm, float plate_width_mm, int marker_mm,
                           int bg = 0, int fg = 255) {
    const int plate_w = static_cast<int>(plate_width_mm * px_per_mm);
    const int plate_h = static_cast<int>(57.0F * px_per_mm);
    const cv::Point2f o{80.0F, 80.0F};
    const std::array<cv::Point2f, 4> corners{o,
                                             {o.x + plate_w, o.y},
                                             {o.x + plate_w, o.y + plate_h},
                                             {o.x, o.y + plate_h}};

    const int marker = static_cast<int>(marker_mm * px_per_mm);
    cv::Mat image(400, 900, CV_8UC1, cv::Scalar(bg));
    cv::rectangle(image,
                  cv::Rect(static_cast<int>(o.x) + (plate_w - marker) / 2,
                           static_cast<int>(o.y) + (plate_h - marker) / 2, marker,
                           marker),
                  cv::Scalar(fg), cv::FILLED);

    const ArmorSize size =
        (plate_width_mm > 180.0F) ? ArmorSize::Large : ArmorSize::Small;
    cv::Mat patch;
    if (!armor::extractDigitPatch(image, corners, size, patch)) {
        return cv::Mat();
    }
    return patch;
}

// 补丁里白像素的外接矩形：返回 {宽, 高, 白像素个数}
std::array<int, 3> whiteBox(const cv::Mat& patch) {
    int x0 = patch.cols, x1 = -1, y0 = patch.rows, y1 = -1, count = 0;
    for (int r = 0; r < patch.rows; ++r) {
        for (int c = 0; c < patch.cols; ++c) {
            if (patch.at<unsigned char>(r, c) != 0) {
                x0 = std::min(x0, c);
                x1 = std::max(x1, c);
                y0 = std::min(y0, r);
                y1 = std::max(y1, r);
                ++count;
            }
        }
    }
    if (count == 0) {
        return {0, 0, 0};
    }
    return {x1 - x0 + 1, y1 - y0 + 1, count};
}

}  // namespace

int main() {
    // =======================================================================
    // 角点顺序 —— 全场最高危的东西，先用纯构造的点焊死
    // =======================================================================
    // 两根竖直灯条，右边那根整体低 5px，模拟板子轻微倾斜。
    const armor::LightBar left = bar(100.0F, 100.0F, 160.0F);
    const armor::LightBar right = bar(200.0F, 105.0F, 165.0F);

    const std::array<cv::Point2f, 4> corners = armor::buildCorners(left, right);

    // 左上 → 右上 → 右下 → 左下
    assert(corners[0] == left.top);
    assert(corners[1] == right.top);
    assert(corners[2] == right.bottom);
    assert(corners[3] == left.bottom);

    // 逐条把"左在右左边、上在下上面"钉死
    assert(corners[0].x <= corners[1].x);
    assert(corners[3].x <= corners[2].x);
    assert(corners[0].y <= corners[3].y);
    assert(corners[1].y <= corners[2].y);

    // 传参左右颠倒时，内部必须自动纠正，结果一模一样
    const std::array<cv::Point2f, 4> swapped = armor::buildCorners(right, left);
    for (int i = 0; i < 4; ++i) {
        assert(swapped[i] == corners[i]);
    }

    // =======================================================================
    // 有向面积判据本身 —— 单独测，别只依赖 buildCorners 间接覆盖
    // =======================================================================
    // 屏幕顺时针（y 向下）：左上→右上→右下→左下
    const std::array<cv::Point2f, 4> cw{{{0, 0}, {10, 0}, {10, 10}, {0, 10}}};
    // 屏幕逆时针：左上→左下→右下→右上
    const std::array<cv::Point2f, 4> ccw{{{0, 0}, {0, 10}, {10, 10}, {10, 0}}};

    assert(armor::quadSignedArea(cw) > 0.0F);
    assert(armor::quadSignedArea(ccw) < 0.0F);
    assert(near(armor::quadSignedArea(cw), 100.0F));   // 10×10 的正方形
    assert(near(armor::quadSignedArea(ccw), -100.0F));  // 反过来是负的

    // 四点共线 → 面积 0。这正是"退化四边形"在数值上的样子。
    const std::array<cv::Point2f, 4> flat{{{0, 0}, {5, 0}, {10, 0}, {15, 0}}};
    assert(near(armor::quadSignedArea(flat), 0.0F));

    // =======================================================================
    // 灯条倾角 —— 直接断言数值，给判据 3 打地基
    // =======================================================================
    assert(near(bar(0.0F, 0.0F, 100.0F).tiltDegrees(), 0.0F));  // 竖直 = 0°
    // 下端偏右 30°：长 100 → dx = 50, dy = 86.60
    assert(near(tilted(0.0F, 0.0F, 50.0F, 86.60254F).tiltDegrees(), 30.0F));
    // 下端偏左 → 负角
    assert(near(tilted(0.0F, 0.0F, -50.0F, 86.60254F).tiltDegrees(), -30.0F));

    // =======================================================================
    // 大小装甲板分类 —— 比值与尺度无关
    // =======================================================================
    const armor::LightBar small_left = bar(100.0F, 100.0F, 160.0F);  // 长 60

    // 中心距 132，132 / 60 = 2.2 → 小板
    const armor::LightBar small_right = bar(232.0F, 100.0F, 160.0F);
    assert(armor::classifySize(small_left, small_right) == ArmorSize::Small);

    // 中心距 240，240 / 60 = 4.0 → 大板
    const armor::LightBar large_right = bar(340.0F, 100.0F, 160.0F);
    assert(armor::classifySize(small_left, large_right) == ArmorSize::Large);

    // 边界本身：中心距 180，180 / 60 = 3.0 恰好等于分界值。
    // 约定的语义是"≥ 分界即大板"，所以这里必须判大板。
    // 不测这一条，把 >= 写成 > 也没人发现。
    const armor::LightBar edge_right = bar(280.0F, 100.0F, 160.0F);
    assert(armor::classifySize(small_left, edge_right) == ArmorSize::Large);
    // 分界再抬高一点点就回落成小板。
    // 注意别写 3.0000001F —— float 在 3.0 附近的分辨率约 2.4e-7，
    // 那个字面量会被舍入回 3.0f，断言就假失败了。
    assert(armor::classifySize(small_left, edge_right, 3.0001F) ==
           ArmorSize::Small);

    // 退化保护：长度 0 的灯条不能吐出 inf 或 NaN
    const armor::LightBar degenerate = bar(100.0F, 100.0F, 100.0F);  // 长 0
    assert(armor::classifySize(degenerate, small_right) == ArmorSize::Small);
    assert(armor::classifySize(degenerate, degenerate) == ArmorSize::Small);

    // 同一个 2.2 的比值，整体放大 3 倍后仍判为小板
    // —— 这就是"与分辨率无关"
    const armor::LightBar scaled_left = bar(100.0F, 100.0F, 280.0F);   // 长 180
    const armor::LightBar scaled_right = bar(496.0F, 100.0F, 280.0F);  // 距 396
    assert(armor::classifySize(scaled_left, scaled_right) == ArmorSize::Small);

    // =======================================================================
    // 灯条配对 —— 判据 1~5 各自都要有一个"只靠它才会被拒"的用例，
    // 判据 6（左右定序）要有一个"把顺序传反"的用例；
    // 否则那条判据删掉了测试也照样绿（变异测试实测过）
    // =======================================================================
    // 一对正常的小装甲板：六条判据全过
    const std::vector<armor::LightBar> ok{small_left, small_right};
    const auto pairs = armor::pairLightBars(ok);
    assert(pairs.size() == 1);
    assert(pairs[0].first.center().x < pairs[0].second.center().x);  // 左在前

    // 判据 1：长度差 2 倍 → 不是同一块板，拒绝
    const std::vector<armor::LightBar> uneven{small_left,
                                              bar(232.0F, 100.0F, 130.0F)};
    assert(armor::pairLightBars(uneven).empty());

    // 判据 2：两根灯条等长、平行，但右灯条整体下移 60px。
    // |Δy| / 平均长度 = 60 / 60 = 1.0 > 0.6 → 拒绝。
    // 其余五条判据在这组输入下全部通过，所以拒掉的原因只能是判据 2。
    const std::vector<armor::LightBar> misaligned{small_left,
                                                  bar(232.0F, 160.0F, 220.0F)};
    assert(armor::pairLightBars(misaligned).empty());

    // 判据 3：右灯条倾斜 30°，左灯条竖直 → 倾角差 30° > 15°。
    // 长度、y 中心、间距比都仍然合格，所以拒掉的原因只能是判据 3。
    const std::vector<armor::LightBar> not_parallel{
        small_left, tilted(232.0F - 15.0F, 100.0F, 30.0F, 51.96152F)};
    assert(armor::pairLightBars(not_parallel).empty());

    // 判据 4：间距比 480 / 60 = 8.0 → 太远，拒绝
    const std::vector<armor::LightBar> too_far{small_left,
                                               bar(580.0F, 100.0F, 160.0F)};
    assert(armor::pairLightBars(too_far).empty());

    // 判据 6：把顺序传反，返回的 pair 里 first 仍然必须在左。
    // 少了这个用例，"左右定序"那两行删掉测试也不会红。
    const std::vector<armor::LightBar> reversed{small_right, small_left};
    const auto reversed_pairs = armor::pairLightBars(reversed);
    assert(reversed_pairs.size() == 1);
    assert(reversed_pairs[0].first.center().x <
           reversed_pairs[0].second.center().x);

    // 退化灯条（长 0）不能参与配对，否则后面的比值全变成 inf
    const std::vector<armor::LightBar> with_degenerate{degenerate, small_right};
    assert(armor::pairLightBars(with_degenerate).empty());

    // 两根都退化、且中心重合 —— 最阴的一种输入：
    // 长度比 0/0 = NaN、间距比 0/0 = NaN，
    // 而 NaN 与任何数比较都是 false（既不 > 也不 <），
    // 六条判据会全部放行，配出一块四个角点全假的"装甲板"。
    // 只放一根退化灯条是测不到这个的 —— 那种情况会被间距比 inf 挡下。
    assert(armor::pairLightBars({degenerate, degenerate}).empty());

    // 三根等距灯条 → **只有 2 对**（左中 / 中右），最外侧那对必须被拒。
    //
    // 这一条是从真实帧上抓回来的：海康整图 20261008231729930 上测到四根灯条
    // 在 x = 263 / 514 / 721 / 909，其中 (514, 909) 这一对间距比 5.05 落在
    // 合法区间里，能过除判据 5 以外的全部判据 —— 但它中间夹着 x=721 那根灯条，
    // 是个跨两块板的假配对。它的外接矩形最大，NMS 按面积取大正好留下它，
    // 于是裁剪框落在中间那根灯条上，抠出来的"数字补丁"其实是灯条。
    //
    // 真实装甲板中间不会竖着第三根灯条。所以：一对灯条之间若夹着别的灯条，
    // 这对就不是一块板。
    const std::vector<armor::LightBar> three{
        small_left, bar(232.0F, 100.0F, 160.0F), bar(364.0F, 100.0F, 160.0F)};
    const auto three_pairs = armor::pairLightBars(three);
    assert(three_pairs.size() == 2);
    for (const auto& p : three_pairs) {
        // 最外侧那对（左灯条 100 → 右灯条 364）跨度会盖住中间那根的中心 232
        assert(!(p.first.center().x < 232.0F && p.second.center().x > 232.0F));
    }

    // =======================================================================
    // 一对一选择 —— 配对层之上的全局约束
    // =======================================================================
    // 上面那组测的是 pairLightBars 的**局部**判据："这两根灯条能不能构成一块板"。
    // 它管不住**全局**的事。四根灯条同框时，(b0,b1)(b1,b2)(b2,b3) 三对各自都能
    // 通过全部判据，而 b1、b2 被两对同时用着 —— 物理上一根灯条只装在一块装甲板上。
    //
    // NMS 救不了这个：相邻两对只共用一个灯条宽，外接矩形的 IoU 远远够不到 0.3，
    // 根本不会进合并。上面判据 5 也救不了它 —— 判据 5 只挡住"跨过第三根灯条"的对，
    // 相邻两对中间没有第三根。
    //
    // 所以选择交给 choosePairs：按分数从高到低贪心，一根灯条只能用一次。
    // 分数 = 几何自洽度(负分) + 板面对比度 + 数字奖励。
    // 三项都**不是硬门** —— 没有竞争时任何一对都要照常返回，见下面那条契约。

    // 真实帧几何，从海康整图 hik_capture/barscene/hik_00.png 上量下来的（不是编的）：
    //   b0 中心(384.0,470.7) 长58.68 倾角+12.53   ← 左边那块看不见的板的右灯条
    //   b1 中心(545.3,478.0) 长62.12 倾角 +1.27   ┐ 真正那块板
    //   b2 中心(709.0,477.5) 长65.00 倾角  0.00   ┘
    //   b3 中心(900.3,479.9) 长88.53 倾角-14.04   ← 右边那块看不见的板的左灯条
    // 三对的间距比是 2.672 / 2.576 / 2.492 —— **真板拿的是中间那个值**。
    // 所以"取最紧的一对"和"取最像小板的一对"这两条捷径都会选错（实测都选错）。
    // 能选对的只有"三根灯条自洽"：真板那两根既等长、又等高、又平行。
    const std::vector<armor::LightBar> scene{
        barFrom(384.0F, 470.7F, 58.68F, 12.53F),
        barFrom(545.3F, 478.0F, 62.12F, 1.27F),
        barFrom(709.0F, 477.5F, 65.00F, 0.00F),
        barFrom(900.3F, 479.9F, 88.53F, -14.04F)};
    const auto scene_candidates = armor::pairLightBars(scene);
    assert(scene_candidates.size() == 3);  // 四根灯条 → 三个相邻候选，全部合法
    const auto scene_chosen = armor::choosePairs(
        scene_candidates, std::vector<armor::PairEvidence>(scene_candidates.size()));
    assert(scene_chosen.size() == 1);  // 三对共用灯条，只能留一对
    assert(near(scene_chosen[0].first.center().x, 545.3F, 0.5F));
    assert(near(scene_chosen[0].second.center().x, 709.0F, 0.5F));

    // 契约（对应评分表里的「目标丢失处理」）：**没有竞争时，一对都不能少**。
    // 板面上什么都没有（对比度 1.0）、数字也没读出来（target_id = 0）的一对，
    // 仍然是块真板 —— 角点有效、PnP 能用，只是 target_id = 0。
    // 这里要是把它筛掉，"检测坏了"（armors = 0）和"分类坏了"（armors > 0 但 id = 0）
    // 在 Web Debug 上就长得一模一样了。这是本次改动最容易踩的坑。
    const auto lone = armor::pairLightBars({small_left, small_right});
    assert(lone.size() == 1);
    const auto lone_chosen =
        armor::choosePairs(lone, std::vector<armor::PairEvidence>(1));
    assert(lone_chosen.size() == 1);

    // ---- 数字奖励 ----------------------------------------------------------
    // 两条规则，一起决定了数字对"谁赢"的影响有多大：
    //  ① **前提**：这一对必须确实有板面（板面对比度 > 1，即 pairScore 里的
    //     face_term > 0）。数字是印在牌面上的；牌面都没有的地方读出 1~8，
    //     只能是分类器在背景上误报，给它加分等于奖励噪声。
    //  ② **幅度**：只加 2 分（kDigitScoreBonus），不是压过一切的 10 分。
    //     几何自洽度本身有 3 分的跨度，一块读得出数字但歪得厉害的候选，
    //     不该挤掉一块几何干净、牌面清清楚楚的板。
    //
    // 下面三段用同一批灯条、只改证据，把这两条规则的三面钉住：
    //   A 没板面 + 有数字 → 数字不计分，几何说了算
    //   B 有板面 + 有数字 → 数字计分，能压过 0.83 的几何劣势
    //   C 有板面 + 有数字，但几何差 2.45 → 数字**不许**翻盘
    //
    // 右边那对两根灯条歪了 12.1°（几何明显更差）却读出了 7，左边那对几何完美。
    // 现场就是这样：几块板里只有一块的贴纸能读出来，而它未必是几何最好的那块。
    const std::vector<armor::LightBar> two_groups{
        small_left, small_right,
        tilted(900.0F - 6.3F, 100.0F, 12.6F, 58.66F),  // 倾角 +12.1°，长 60
        bar(1032.0F, 100.0F, 160.0F)};                 // 长 60
    const auto two_candidates = armor::pairLightBars(two_groups);
    assert(two_candidates.size() == 2);  // 互不共用 → 两对都会被留下，比的是先后
    // 右边那对的自洽度 ≈ 0.83（倾角项 12.1/15 占大头），左边是 0。

    // A：对比度留默认的 1.0 = 中间什么都没有 ⇒ 数字不该计分。
    // 于是右边只剩几何的 −0.83，左边是 0，左边排前面。
    // 把 pairScore 里 `face_term > 0.0F` 那道闸拆掉，右边会凭空多 2 分翻到前面。
    std::vector<armor::PairEvidence> no_face(two_candidates.size());
    for (std::size_t k = 0; k < two_candidates.size(); ++k) {
        no_face[k].plate_face_contrast = 1.0F;
        no_face[k].target_id = (two_candidates[k].first.center().x > 800.0F) ? 7 : 0;
    }
    const auto no_face_chosen = armor::choosePairs(two_candidates, no_face);
    assert(no_face_chosen.size() == 2);
    assert(no_face_chosen[0].first.center().x < 800.0F);

    // B：两对都有板面（对比度 2.0 ⇒ 各 +1）⇒ 右边的数字生效，
    // 2.17 反超左边的 1.0。拿掉数字奖励，右边掉到 0.17，又排回左边后面。
    std::vector<armor::PairEvidence> digit_evidence(two_candidates.size());
    for (std::size_t k = 0; k < two_candidates.size(); ++k) {
        digit_evidence[k].plate_face_contrast = 2.0F;
        digit_evidence[k].target_id =
            (two_candidates[k].first.center().x > 800.0F) ? 7 : 0;
    }
    const auto digit_chosen = armor::choosePairs(two_candidates, digit_evidence);
    assert(digit_chosen.size() == 2);
    assert(digit_chosen[0].first.center().x > 800.0F);

    // C：奖励的**幅度**上限。右边那对几何自洽度 ≈ 2.45 —— 长度比 1.67 占 0.83、
    // 中心高差 30px（均值 80）占 0.63、倾角 14.9° 占 0.99。左边是完美的 0，
    // 两边板面一样（都 +1），所以右边要翻盘，奖励得**超过 2.45**。
    // 奖励一旦被调回 3 以上，这条断言立刻变红 —— 这就是它存在的理由。
    const std::vector<armor::LightBar> wide_geometry{
        small_left, small_right,
        bar(900.0F, 100.0F, 160.0F),                // 中心 (900,130) 长 60
        tilted(987.15F, 51.68F, 25.71F, 96.64F)};   // 中心 (1000,100) 长 100 倾角 14.9°
    const auto wide_candidates = armor::pairLightBars(wide_geometry);
    assert(wide_candidates.size() == 2);
    std::vector<armor::PairEvidence> wide_evidence(wide_candidates.size());
    for (std::size_t k = 0; k < wide_candidates.size(); ++k) {
        wide_evidence[k].plate_face_contrast = 2.0F;
        wide_evidence[k].target_id =
            (wide_candidates[k].first.center().x > 800.0F) ? 7 : 0;
    }
    const auto wide_chosen = armor::choosePairs(wide_candidates, wide_evidence);
    assert(wide_chosen.size() == 2);
    assert(wide_chosen[0].first.center().x < 800.0F);  // 几何干净的那块排前面

    // D：板面项的**封顶**。同一批灯条，把右边那对的对比度抬到 5.0（一块亮得刺眼的
    // 牌面），左边只给 1.5。封顶之后右边拿到 +1、左边 +0.5，右边仍然输在几何上
    // （−2.45+1 = −1.45 < 0.5）。
    // 要是把 `min(1.0F, …)` 那个封顶去掉，右边变成 +4 ⇒ 1.55 反超 0.5，
    // 一块亮牌面就能把几何证据整个买通 —— 封顶就是为这个加的。
    std::vector<armor::PairEvidence> bright_evidence(wide_candidates.size());
    for (std::size_t k = 0; k < wide_candidates.size(); ++k) {
        bright_evidence[k].plate_face_contrast =
            (wide_candidates[k].first.center().x > 800.0F) ? 5.0F : 1.5F;
    }
    const auto bright_chosen = armor::choosePairs(wide_candidates, bright_evidence);
    assert(bright_chosen.size() == 2);
    assert(bright_chosen[0].first.center().x < 800.0F);

    // 几何打平时由板面对比度决定先后。四根灯条形状完全一样，
    // 两对的几何自洽度一模一样，唯一的差别是中间那块牌面亮不亮。
    // 数字用的是真实帧上量到的 2.065 / 0.316。
    const std::vector<armor::LightBar> flat_groups{small_left, small_right,
                                                   bar(900.0F, 100.0F, 160.0F),
                                                   bar(1032.0F, 100.0F, 160.0F)};
    const auto flat_candidates = armor::pairLightBars(flat_groups);
    assert(flat_candidates.size() == 2);
    std::vector<armor::PairEvidence> flat_evidence(flat_candidates.size());
    for (std::size_t k = 0; k < flat_candidates.size(); ++k) {
        flat_evidence[k].plate_face_contrast =
            (flat_candidates[k].first.center().x > 800.0F) ? 2.065F : 0.316F;
    }
    const auto flat_chosen = armor::choosePairs(flat_candidates, flat_evidence);
    assert(flat_chosen.size() == 2);
    assert(flat_chosen[0].first.center().x > 800.0F);

    // 分数完全打平时按候选顺序定先后。贪心的结果必须是确定的 ——
    // 否则同一帧隔一次跑给出不同的框，现场没法复现。
    const auto tie_chosen = armor::choosePairs(
        flat_candidates, std::vector<armor::PairEvidence>(flat_candidates.size()));
    assert(tie_chosen.size() == 2);
    assert(tie_chosen[0].first.center().x < tie_chosen[1].first.center().x);

    // =======================================================================
    // 数字补丁裁剪 —— 本函数最危险的地方是纵横比
    // =======================================================================
    const cv::Mat small_patch = makePlateAndExtract(3.0F, 135.0F, 20);
    assert(!small_patch.empty());

    // 契约：28 行 x 20 列、单通道、取值只可能是 0 或 255。
    // 数据集本身就是 {0,255}，喂灰度模糊图进去 HOG 分布会完全不同 ——
    // 这种事不会报错，只会让准确率悄悄掉下去，所以必须在这里焊死。
    assert(small_patch.type() == CV_8UC1);
    assert(small_patch.cols == 20);
    assert(small_patch.rows == 28);
    for (int r = 0; r < small_patch.rows; ++r) {
        for (int c = 0; c < small_patch.cols; ++c) {
            const int v = small_patch.at<unsigned char>(r, c);
            assert(v == 0 || v == 255);
        }
    }

    // 20mm 的方块，落在数字窗里，映射到 20x28 上应该 ≈ (20/81)*20 ≈ 5 宽、
    // (20/114)*28 ≈ 5 高 —— 接近正方形。
    // 窗是 ≈81mm x 114mm，比物理板(40.7mm x 57mm)大，因为 patch_quad_scale=2：
    // 灯条只占牌面一部分，抠数字得把图案上下缘一起要进来。
    // 见 PatchParams::patch_quad_scale 的注释。
    const std::array<int, 3> box = whiteBox(small_patch);
    assert(box[2] > 0);
    assert(box[0] >= 3 && box[0] <= 8);
    assert(box[1] >= 3 && box[1] <= 8);
    // ↓ 这一条是关键。把四个角点一步 getPerspectiveTransform 到 20x28 的
    //   实现，会让宽度压成 ≈3、高度还是 ≈10，宽高差 7 > 3，立刻红。
    assert(std::abs(box[0] - box[1]) <= 3);

    // 大板同样要方 —— 这条同时钉住"size 参数真的被用上了"
    const cv::Mat large_patch = makePlateAndExtract(2.0F, 230.0F, 20);
    assert(!large_patch.empty());
    const std::array<int, 3> large_box = whiteBox(large_patch);
    assert(large_box[2] > 0);
    assert(large_box[0] >= 3 && large_box[0] <= 8);
    assert(std::abs(large_box[0] - large_box[1]) <= 3);

    // 窗口必须在板子中央，而不是贴着左缘。
    // 只在最左侧 10mm 涂白（30px），数字框从 47mm 才开始，应该完全看不到。
    // 少了这一条，"从 x=0 开始裁"的实现只会让补丁整体偏一点，很难被发现。
    {
        const int plate_w = 405;
        const int plate_h = 171;
        const cv::Point2f o{100.0F, 100.0F};
        const std::array<cv::Point2f, 4> corners{
            o,
            {o.x + plate_w, o.y},
            {o.x + plate_w, o.y + plate_h},
            {o.x, o.y + plate_h}};
        cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC1);
        cv::rectangle(image, cv::Rect(100, 100, 30, plate_h), cv::Scalar(255),
                      cv::FILLED);
        cv::Mat patch;
        assert(armor::extractDigitPatch(image, corners, ArmorSize::Small, patch));
        assert(whiteBox(patch)[2] <= 5);
    }

    // 低对比度：背景 30、亮块 100，整体压暗压窄。
    // 这不是刁难 —— 现场光照一变，ROI 里的灰度范围就可能被压成这样。
    // Otsu 会自己从直方图里找刀口（≈65），固定阈值 128 会把整幅判成黑，
    // 数字直接丢干净。少了这一条，"把 Otsu 换成固定阈值"这种改动
    // 在两个 0/255 的极端之间看不出任何差别，测试抓不住。
    {
        const cv::Mat low = makePlateAndExtract(3.0F, 135.0F, 20, 30, 100);
        assert(!low.empty());
        assert(whiteBox(low)[2] > 0);  // 亮块必须还在
        assert(low.type() == CV_8UC1);
        assert(low.cols == 20);
        assert(low.rows == 28);
        for (int r = 0; r < low.rows; ++r) {
            for (int c = 0; c < low.cols; ++c) {
                const int v = low.at<unsigned char>(r, c);
                assert(v == 0 || v == 255);  // 阈值化之后仍是二值
            }
        }
    }

    // 退化输入必须被拒，而不是吐出一张全黑的图冒充成功。
    // 四点共线时 getPerspectiveTransform 会给出退化矩阵，
    // warpPerspective 的结果是垃圾（可能含 NaN），必须提前拦掉。
    {
        cv::Mat patch;
        cv::Mat image = cv::Mat::zeros(64, 64, CV_8UC1);
        const std::array<cv::Point2f, 4> collinear{
            {{0, 0}, {5, 0}, {10, 0}, {15, 0}}};
        assert(!armor::extractDigitPatch(image, collinear, ArmorSize::Small,
                                         patch));
        assert(patch.empty());

        // 空图也要拒
        cv::Mat empty_image;
        const std::array<cv::Point2f, 4> fine_corners{
            {{0, 0}, {10, 0}, {10, 10}, {0, 10}}};
        assert(!armor::extractDigitPatch(empty_image, fine_corners,
                                         ArmorSize::Small, patch));
        assert(patch.empty());

        // 参数本身非法同样要拒，不能"尽力而为"。
        // px_per_mm = 0 会把规范板缩成 1x1，再放大回 20x28 ——
        // 得到一张糊成一片的"补丁"，看着像成功，实际全是垃圾。
        armor::PatchParams zero_scale;
        zero_scale.px_per_mm = 0.0F;
        assert(!armor::extractDigitPatch(image, fine_corners, ArmorSize::Small,
                                         patch, zero_scale));
        assert(patch.empty());

        // 输出尺寸为 0 会直接喂给 resize 一个空尺寸
        armor::PatchParams zero_out;
        zero_out.out_width = 0;
        assert(!armor::extractDigitPatch(image, fine_corners, ArmorSize::Small,
                                         patch, zero_out));
        assert(patch.empty());
    }

    return 0;
}
