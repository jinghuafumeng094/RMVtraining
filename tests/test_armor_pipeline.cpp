// 配色掩膜 → 灯条提取 → 配对去重 → 数字分类 这一整条管线的测试。
//
// 和 test_armor_detect.cpp 的分工：那个文件测的是**纯几何**
// （给了灯条，角点/大小分类对不对），不碰像素；
// 这个文件测的是往两头延伸的部分 —— 前端的颜色掩膜与灯条提取，
// 后端的串联与去重。两个文件都不需要相机、不需要数据集。
#include "ArmorDetector.hpp"
#include "DigitClassifier.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace {

// 海康整图上**实测**到的灯条像素值（BGR 顺序），不是"理想的红 (0,0,255)"。
// 阈值就是在这些真值上调出来的，拿纯色去测等于绕开了阈值正要验的那段灰阶。
const cv::Scalar kBlueBar(254.0, 245.0, 74.0);
const cv::Scalar kRedBar(42.0, 172.0, 254.0);
// 背景取 20：低于 min_channel_value(100)，通道差也是 0，
// 所以无论掩膜阈值怎么挪，背景都不该进来。
const cv::Scalar kBackground(20.0, 20.0, 20.0);

// 画一根竖直灯条。center_x 是中心列，top_y / bottom_y 是上下端。
void drawBar(cv::Mat& image, float center_x, float top_y, float bottom_y,
             const cv::Scalar& color, int width = 12) {
    cv::rectangle(image,
                  cv::Rect(static_cast<int>(center_x - width * 0.5F),
                           static_cast<int>(top_y), width,
                           static_cast<int>(bottom_y - top_y)),
                  color, cv::FILLED);
}

bool near(float a, float b, float eps = 1.0F) {
    return std::fabs(a - b) < eps;
}

// 造一块"两根蓝灯条、中间什么都没有"的板。小板：灯条长 120，
// 中心距 264 → 264/120 = 2.2，落在 kGapRatioSplit(3.0) 的小板一侧。
cv::Mat makeBluePlate() {
    cv::Mat plate(240, 420, CV_8UC3, kBackground);
    drawBar(plate, 70.0F, 60.0F, 180.0F, kBlueBar);
    drawBar(plate, 334.0F, 60.0F, 180.0F, kBlueBar);
    return plate;
}

}  // namespace

int main() {
    // =======================================================================
    // 颜色掩膜 —— 带符号的通道差
    // =======================================================================
    cv::Mat image(240, 320, CV_8UC3, kBackground);
    drawBar(image, 100.0F, 40.0F, 160.0F, kBlueBar);
    drawBar(image, 220.0F, 40.0F, 160.0F, kRedBar);

    const cv::Mat blue_mask = armor::buildColorMask(image, TeamColor::Blue);
    assert(!blue_mask.empty());
    assert(blue_mask.type() == CV_8UC1);
    assert(blue_mask.size() == image.size());

    // 蓝队掩膜只认蓝灯条：蓝灯条中心是前景，红灯条中心必须是背景。
    //
    // ↓↓ 这两条是整个文件里最关键的一对断言。
    //    把实现里的 subtract 换成 absdiff（也就是用 |B-R| 而不是 B-R），
    //    两个位置都会变成前景 —— 红蓝两队看到一样多的目标，
    //    而且一个错误都不报。只有这一对断言能抓住它。
    assert(blue_mask.at<unsigned char>(100, 100) == 255);
    assert(blue_mask.at<unsigned char>(100, 220) == 0);

    const cv::Mat red_mask = armor::buildColorMask(image, TeamColor::Red);
    assert(red_mask.at<unsigned char>(100, 220) == 255);
    assert(red_mask.at<unsigned char>(100, 100) == 0);

    // 阈值语义逐条钉死。两条判据各自都要有一个"只靠它才会被拒"的用例，
    // 否则删掉任意一条，测试照样全绿。
    {
        // 通道差不够：B=110 R=90，差 20 < 60。形状跟灯条一模一样，
        // 唯一挡住它的是差值 —— 所以这条只测得到 min_channel_diff。
        cv::Mat dim(240, 320, CV_8UC3, kBackground);
        drawBar(dim, 100.0F, 40.0F, 160.0F, cv::Scalar(110.0, 100.0, 90.0));
        assert(cv::countNonZero(armor::buildColorMask(dim, TeamColor::Blue)) == 0);

        // 亮度不够：B=80 R=10，差 70 已经合格，但主通道 80 < 100。
        // 少这一条，把 min_channel_value 整个删掉也没人发现。
        cv::Mat dark(240, 320, CV_8UC3, kBackground);
        drawBar(dark, 100.0F, 40.0F, 160.0F, cv::Scalar(80.0, 60.0, 10.0));
        assert(cv::countNonZero(armor::buildColorMask(dark, TeamColor::Blue)) == 0);
    }

    // 没有颜色信息的输入必须被拒，而不是猜一个通道顺序。
    // 猜错的话整条管线会安静地全错，比直接失败难查得多。
    {
        const cv::Mat gray = cv::Mat::zeros(64, 64, CV_8UC1);
        assert(armor::buildColorMask(gray, TeamColor::Blue).empty());
        const cv::Mat empty;
        assert(armor::buildColorMask(empty, TeamColor::Blue).empty());
    }

    // BGRA 也要收：有些采集后端给的是四通道，不该让调用方先去转一道。
    {
        cv::Mat bgra;
        cv::cvtColor(image, bgra, cv::COLOR_BGR2BGRA);
        const cv::Mat mask = armor::buildColorMask(bgra, TeamColor::Blue);
        assert(!mask.empty());
        assert(mask.at<unsigned char>(100, 100) == 255);
        assert(mask.at<unsigned char>(100, 220) == 0);
    }

    // =======================================================================
    // 灯条提取 —— 掩膜到灯条
    // =======================================================================
    // 这一段只关心几何，所以直接喂单通道的"白条"掩膜，
    // 省得把颜色判据和形状判据混在一起，出问题时说不清是哪一级坏的。
    {
        cv::Mat geometric(240, 320, CV_8UC1, cv::Scalar(0));
        cv::rectangle(geometric, cv::Rect(94, 40, 12, 120), cv::Scalar(255),
                      cv::FILLED);

        const std::vector<armor::LightBar> bars = armor::extractLightBars(geometric);
        assert(bars.size() == 1);
        // 约定：top 是画面里在上面的那一端
        assert(bars[0].top.y <= bars[0].bottom.y);
        assert(near(bars[0].length(), 120.0F, 2.0F));
        assert(near(bars[0].top.x, 100.0F, 1.0F));
        assert(near(bars[0].bottom.x, 100.0F, 1.0F));
        assert(near(bars[0].tiltDegrees(), 0.0F, 1.0F));
    }

    // 倾斜灯条：倾角必须带正确的符号，端点必须真的落在两端
    {
        // 用实心的旋转矩形，**不用 cv::line** 画斜灯条。
        // cv::line 的粗线是圆帽的：两端各多出半个线宽，
        // 灯条长度会变成 145.6 + 12 ≈ 157.6 而不是 145.6。
        // 那不是被测代码的问题，但会让"长度对不对"这条断言失去意义。
        const cv::Point2f head(100.0F, 40.0F);
        const cv::Point2f tail(140.0F, 180.0F);
        const float dx = tail.x - head.x;
        const float dy = tail.y - head.y;
        const float span = std::sqrt(dx * dx + dy * dy);          // ≈ 145.60
        // 单位法向量 × 半宽：把线段撑成一条 12 像素宽的实心灯条
        const cv::Point2f half_normal(-dy / span * 6.0F, dx / span * 6.0F);
        // fillConvexPoly 在 OpenCV 4.5 只收整数点（CV_32S），
        // 所以这里四舍五入一下 —— 代价是每个角最多 0.5 像素的误差，
        // 下面的容差就是照这个留的。
        const auto rounded = [](const cv::Point2f& p) {
            return cv::Point(cvRound(p.x), cvRound(p.y));
        };
        const std::vector<cv::Point> quad{rounded(head + half_normal),
                                          rounded(head - half_normal),
                                          rounded(tail - half_normal),
                                          rounded(tail + half_normal)};

        cv::Mat slanted(240, 320, CV_8UC1, cv::Scalar(0));
        cv::fillConvexPoly(slanted, quad, cv::Scalar(255));

        const std::vector<armor::LightBar> bars = armor::extractLightBars(slanted);
        assert(bars.size() == 1);
        assert(bars[0].top.y <= bars[0].bottom.y);
        assert(bars[0].tiltDegrees() > 0.0F);        // 下端偏右 = 正
        assert(near(bars[0].tiltDegrees(), 15.95F, 1.0F));
        // 长度必须等于线段的真实长度，而不是外接矩形的对角线之类
        assert(near(bars[0].length(), 145.6F, 2.0F));
        // 端点必须真的落在两端，容差 1 像素
        assert(near(bars[0].top.x, 100.0F) && near(bars[0].top.y, 40.0F));
        assert(near(bars[0].bottom.x, 140.0F) && near(bars[0].bottom.y, 180.0F));
    }

    // 三条形状门槛各要有一个"只靠它才会被滤掉"的用例
    {
        cv::Mat tiny(240, 320, CV_8UC1, cv::Scalar(0));
        cv::rectangle(tiny, cv::Rect(100, 100, 3, 6), cv::Scalar(255), cv::FILLED);
        assert(armor::extractLightBars(tiny).empty());  // 面积 18 < 30

        cv::Mat stub(240, 320, CV_8UC1, cv::Scalar(0));
        cv::rectangle(stub, cv::Rect(100, 100, 4, 20), cv::Scalar(255), cv::FILLED);
        assert(armor::extractLightBars(stub).empty());  // 长边 20 < 25

        // 正方形亮块不是灯条。面积 900、长边 30 都合格，
        // 唯一挡住它的是长宽比 1.0 < 1.5 —— 场地上的方白标记全靠这一条滤掉。
        cv::Mat square(240, 320, CV_8UC1, cv::Scalar(0));
        cv::rectangle(square, cv::Rect(100, 100, 30, 30), cv::Scalar(255),
                      cv::FILLED);
        assert(armor::extractLightBars(square).empty());
    }

    // 两根灯条要各自成一根，不能被合并
    {
        cv::Mat two(240, 320, CV_8UC1, cv::Scalar(0));
        cv::rectangle(two, cv::Rect(94, 40, 12, 120), cv::Scalar(255), cv::FILLED);
        cv::rectangle(two, cv::Rect(226, 40, 12, 120), cv::Scalar(255), cv::FILLED);

        const std::vector<armor::LightBar> bars = armor::extractLightBars(two);
        assert(bars.size() == 2);
        // 不假设 findContours 的返回顺序，只比较两根中心 x 的集合
        const float left = std::min(bars[0].center().x, bars[1].center().x);
        const float right = std::max(bars[0].center().x, bars[1].center().x);
        assert(near(left, 100.0F, 1.0F));
        assert(near(right, 232.0F, 1.0F));
    }

    // 输入类型不对 → 返回空，而不是崩
    {
        const cv::Mat color = cv::Mat::zeros(64, 64, CV_8UC3);
        assert(armor::extractLightBars(color).empty());
        const cv::Mat empty;
        assert(armor::extractLightBars(empty).empty());
    }

    // =======================================================================
    // 去重：三根等距灯条配出的重叠框必须合并
    // =======================================================================
    {
        ArmorDetection far_a;
        far_a.corners = {{{0, 0}, {100, 0}, {100, 50}, {0, 50}}};
        ArmorDetection far_b;
        far_b.corners = {{{300, 300}, {400, 300}, {400, 350}, {300, 350}}};
        assert(armor::suppressOverlaps({far_a, far_b}, 0.3F).size() == 2);
        assert(armor::suppressOverlaps({}, 0.3F).empty());

        // 小框整个落在大框里，IoU ≈ 0.50 > 0.3 → 合并。
        // **故意把小的放在前面**：实现如果按输入顺序取而不是按面积取，
        // 留下的就是小框，下面这条断言立刻红。
        ArmorDetection big;
        big.corners = {{{0, 0}, {200, 0}, {200, 60}, {0, 60}}};
        ArmorDetection small;
        small.corners = {{{0, 0}, {100, 0}, {100, 60}, {0, 60}}};

        const std::vector<ArmorDetection> kept =
            armor::suppressOverlaps({small, big}, 0.3F);
        assert(kept.size() == 1);
        assert(near(kept[0].corners[1].x, 200.0F));  // 留下的是大的那个

        // 同一个重叠，门限抬到 0.7 就该分开保留两个。
        //
        // 这里**不能**把门限设成 0.5 来测"恰好等于时是否合并"：
        // cv::boundingRect 对点集用的是含两端的约定（宽 = x_max - x_min + 1），
        // 上面这两个框实际是 201x61 和 101x61，IoU = 101/201 ≈ 0.5025，
        // 不是 0.5。拿整数角点凑不出正好 0.5 的重叠，
        // 所以边界那一个 ULP 的差别不在这里测，改测"明显大于 / 明显小于"。
        assert(armor::suppressOverlaps({small, big}, 0.7F).size() == 2);
    }

    // =======================================================================
    // 端到端：一块蓝队小板
    // =======================================================================
    {
        const cv::Mat plate = makeBluePlate();

        const std::vector<ArmorDetection> found =
            armor::detectArmors(plate, TeamColor::Blue);
        assert(found.size() == 1);
        assert(found[0].size == ArmorSize::Small);

        // 角点顺序约定（全链路都依赖它）：左上 右上 右下 左下
        assert(found[0].corners[0].x <= found[0].corners[1].x);
        assert(found[0].corners[0].y <= found[0].corners[3].y);
        assert(armor::quadSignedArea(found[0].corners) > 0.0F);

        // ↓ 颜色过滤必须在**端到端**这一层也起作用：
        //   同一张图让红队去看，一个目标都不该有。
        //   这一条比上面那两条掩膜断言更强 —— 它证明 team 参数
        //   真的贯穿到了最后的输出，而不是在中间某一级被丢掉了。
        assert(armor::detectArmors(plate, TeamColor::Red).empty());

        // 空图 / 类型不对，都返回空而不是崩
        const cv::Mat empty;
        assert(armor::detectArmors(empty, TeamColor::Blue).empty());
    }

    // =======================================================================
    // 跨板配对 —— 一根灯条不许夹在另一对的中间
    // =======================================================================
    // 用海康整图 20261008231729930 上**实测**的四根灯条横向位置。
    // 那一帧上 (514, 909) 这一对的间距比 5.05 能过全部五条配对判据，
    // 但它跨了两块板，中间夹着 x=721 的灯条；而它的外接矩形最大，
    // NMS 按面积取大正好把它留下 —— 裁剪框于是落在中间那根灯条上。
    // 这就是"数字全是 0"那批结果的来源。
    //
    // 判据本身与真实帧无关：真实装甲板中间不会竖着第三根灯条。
    // 这里断言的是**不变式**而不是具体数量 —— 数量随判据调整会变，
    // "检测框里不许夹着别的灯条"不会变。
    {
        const int xs[4] = {263, 514, 721, 909};
        cv::Mat row(300, 1000, CV_8UC3, kBackground);
        for (int x : xs) {
            drawBar(row, static_cast<float>(x), 90.0F, 210.0F, kBlueBar);
        }

        const std::vector<ArmorDetection> found =
            armor::detectArmors(row, TeamColor::Blue);
        assert(!found.empty());

        for (const ArmorDetection& d : found) {
            std::vector<cv::Point2f> poly(d.corners.begin(), d.corners.end());
            for (int x : xs) {
                const cv::Point2f center(static_cast<float>(x), 150.0F);
                // 用**带距离**的判定，容差 2 像素。
                // 两端那两根灯条的中心本来就压在框的左右边上（距离 ≈ 0.5，
                // 因为 drawBar 里的 int(x - 6) 让 12 像素宽的条中心落在 x-0.5）；
                // 夹在中间那根的中心距边有上百像素，两种情形差两个数量级，
                // 2 像素的门限把它们干净地分开。> 0 才是严格在框内。
                assert(cv::pointPolygonTest(poly, center, true) <= 2.0);
            }
        }

        // 四根灯条排成一排：每根只跟相邻那根配上 → 3 块板。
        // 修好之前这里会因为跨板大框胜出而变成别的数（实测 1）。
        assert(found.size() == 3);
    }

    // =======================================================================
    // 目标丢失处理 —— 分类器不在时，板子仍然要返回
    // =======================================================================
    // 放在最后：loadFrom 失败会把全局分类器置成"不可用"，
    // 之后再调用 classify 一律返回 0。前面那些断言都需要模型是好的，
    // 所以这一段必须是最后一段。
    {
        assert(!armor::DigitClassifier::loadFrom("/nonexistent/digit_model.xml"));

        const std::vector<ArmorDetection> found =
            armor::detectArmors(makeBluePlate(), TeamColor::Blue);

        // 关键：板子**没有**被丢掉。角点、大小都还是有效的，PnP 照样能算，
        // 只是没有可报的目标编号。评分表里的「目标丢失处理」要的就是这个。
        // 反过来做（没数字就整块丢）会让"检测坏了"和"分类坏了"
        // 在 Web Debug 上完全分不出来 —— armor_count 都是 0。
        assert(found.size() == 1);
        assert(found[0].size == ArmorSize::Small);
        assert(found[0].target_id == 0);
        assert(found[0].confidence == 0.0F);
        assert(armor::quadSignedArea(found[0].corners) > 0.0F);
    }

    return 0;
}
