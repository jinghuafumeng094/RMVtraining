#include "ArmorDetector.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <vector>

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

bool near(float a, float b, float eps = 1e-3F) {
    return std::fabs(a - b) < eps;
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
    // 灯条配对 —— 五条判据各自都要有一个"只靠它才会被拒"的用例，
    // 否则那条判据删掉了测试也照样绿（变异测试实测过）
    // =======================================================================
    // 一对正常的小装甲板：五条判据全过
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
    // 其余四条判据在这组输入下全部通过，所以拒掉的原因只能是判据 2。
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

    // 判据 5：把顺序传反，返回的 pair 里 first 仍然必须在左。
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
    // 五条判据会全部放行，配出一块四个角点全假的"装甲板"。
    // 只放一根退化灯条是测不到这个的 —— 那种情况会被间距比 inf 挡下。
    assert(armor::pairLightBars({degenerate, degenerate}).empty());

    // 三根等距灯条 → 3 个合法配对（左中 / 中右 / 左右）。
    // 这正是后面必须做 NMS 的原因：一根灯条会和左右两根都配上。
    const std::vector<armor::LightBar> three{
        small_left, bar(232.0F, 100.0F, 160.0F), bar(364.0F, 100.0F, 160.0F)};
    assert(armor::pairLightBars(three).size() == 3);

    return 0;
}
