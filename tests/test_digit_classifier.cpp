// 数字分类器（M4）的检查。
//
// 分四段，从"最不会变的东西"往"最依赖数据的东西"排：
//   1. 特征维度     —— 钉死 HOG 参数，改了就红
//   2. 类别映射     —— 钉死 label → target_id，9neg 必须是 0
//   3. 留出集准确率 —— 端到端，唯一可信的质量判据
//   4. 模型缺失不崩 —— 掉文件时必须是"降级"不是"崩"
#include "DigitClassifier.hpp"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#ifndef ARMOR_HOLDOUT_DIR
#error "需要 ARMOR_HOLDOUT_DIR 指向 tests/fixtures/digit_holdout"
#endif

namespace {

// 留出清单：每行 "<相对路径> <label>"，`#` 开头的行是注释。
// 这份清单同时被 tools/train_digit_svm.cpp 用来"跳过留出图"，
// 两边共用一份，不会走偏。
//
// 必须逐行读并跳过注释：只写 `while (in >> rel >> label)` 的话，
// 第一行的 "#" 会被读进路径，接着拿说明文字去读 int，流立刻失败，
// 整个清单读成空的 —— 训练程序那边就靠这个静默漏掉了全部 180 张留出图。
std::vector<std::pair<std::string, int>> readManifest() {
    std::vector<std::pair<std::string, int>> items;
    std::ifstream in(std::string(ARMOR_HOLDOUT_DIR) + "/manifest.txt");
    assert(in && "读不到留出清单：fixture 还没生成？");
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }
        std::istringstream stream(line);
        std::string rel;
        int label = 0;
        if (stream >> rel >> label) {
            items.emplace_back(rel, label);
        }
    }
    return items;
}

cv::Mat uniformPatch(int value) {
    return cv::Mat(armor::DigitClassifier::kPatchHeight,
                   armor::DigitClassifier::kPatchWidth, CV_8UC1,
                   cv::Scalar(value));
}

}  // namespace

int main() {
    using armor::DigitClassifier;
    using armor::DigitResult;

    // =======================================================================
    // 1. 特征维度 = 324
    //    winSize(20,28) / cell(5,7) → 4x4 cell；block(10,14) → 2x2 cell；
    //    stride(5,7) → 1 cell；→ 3x3 = 9 个 block。
    //    9 block x 4 cell x 9 bin = 324。
    //    这个数字一旦变了，训练好的模型就废了（特征长度对不上直接抛异常），
    //    所以必须在这里焊死。
    // =======================================================================
    assert(DigitClassifier::kFeatureSize == 324);
    assert(DigitClassifier::describe(uniformPatch(0)).size() == 324);
    assert(DigitClassifier::describe(uniformPatch(255)).size() == 324);

    // 输入不合法 → 返回空，不抛异常、不返回半截数据
    assert(DigitClassifier::describe(cv::Mat()).empty());
    assert(DigitClassifier::describe(cv::Mat::zeros(10, 10, CV_8UC1)).empty());
    assert(DigitClassifier::describe(cv::Mat::zeros(28, 20, CV_8UC3)).empty());
    // 20x28 是"宽 x 高"的反过来写法，也必须被拒 ——
    // 如果哪天有人把行列搞混，这一条会立刻抓住
    assert(DigitClassifier::describe(cv::Mat::zeros(20, 28, CV_8UC1)).empty());

    // =======================================================================
    // 2. 类别映射：label → target_id
    //    索引顺序就是数据集目录的字典序，训练程序必须用同一张表。
    //    9neg（label 8）必须是 0 —— 契约里"0 保留给未知目标/无目标"。
    // =======================================================================
    assert(DigitClassifier::kClassCount == 9);
    assert(std::string(DigitClassifier::className(0)) == "1");
    assert(std::string(DigitClassifier::className(4)) == "5");
    assert(std::string(DigitClassifier::className(5)) == "6outpost");
    assert(std::string(DigitClassifier::className(6)) == "7guard");
    assert(std::string(DigitClassifier::className(7)) == "8base");
    assert(std::string(DigitClassifier::className(8)) == "9neg");

    assert(DigitClassifier::targetIdForLabel(0) == 1);
    assert(DigitClassifier::targetIdForLabel(4) == 5);
    assert(DigitClassifier::targetIdForLabel(5) == 6);
    assert(DigitClassifier::targetIdForLabel(6) == 7);
    assert(DigitClassifier::targetIdForLabel(7) == 8);
    // 这一条就是"目标丢失处理"：板子在、但没有可读数字 → target_id = 0
    assert(DigitClassifier::targetIdForLabel(8) == 0);
    // 越界输入不能越界读数组
    assert(DigitClassifier::targetIdForLabel(-1) == 0);
    assert(DigitClassifier::targetIdForLabel(9) == 0);

    // =======================================================================
    // 3. 留出集准确率 —— 唯一可信的端到端判据
    // =======================================================================
    // 先确认模型真的加载上了。不确认就往下走的话，
    // "没加载"会让每张图都返回 target_id = 0，
    // 而 9neg 的期望值恰好也是 0 —— 会假绿。
    if (!DigitClassifier::ready()) {
        std::cerr << "模型没加载上（models/armor_digit_svm.xml 不在？）\n"
                     "先跑一次 train_digit_svm 再跑本测试。\n";
        return 1;
    }

    const auto items = readManifest();
    assert(!items.empty() && "留出清单是空的");

    constexpr int kClassTotal = DigitClassifier::kClassCount;
    std::vector<int> hit(kClassTotal, 0);
    std::vector<int> total(kClassTotal, 0);
    int correct = 0;
    // 置信度的最小/最大值。用来钉住"它确实随输入变化"这一条 ——
    // 恒返回 1.0 的实现在 [0,1] 契约下完全合法，只有这一条能抓住。
    float confidence_min = 1.0F;
    float confidence_max = 0.0F;

    for (const auto& item : items) {
        const std::string& rel = item.first;
        const int label = item.second;

        const cv::Mat patch =
            cv::imread(std::string(ARMOR_HOLDOUT_DIR) + "/" + rel,
                       cv::IMREAD_GRAYSCALE);
        assert(!patch.empty() && "留出图读不出来");

        const DigitResult result = DigitClassifier::classify(patch);
        // 契约（AutoAimTypes.hpp）：confidence 范围是 [0, 1]。
        // 少了这一条，把置信度写成"原样返回决策函数值"也测不出来 ——
        // 那个值可以是任意实数，下游拿去当概率用就出事。
        assert(result.confidence >= 0.0F && result.confidence <= 1.0F);
        // 判对了却报 0 置信度，等于自信地说不知道。
        // 这一条挡的是"置信度字段忘了算、恒为 0"的实现。
        assert(!(result.target_id != 0 && result.confidence <= 0.0F));
        // 判对的那一张，冠军必须**正面**赢过它的每一个对手，
        // 也就是最小间隔 > 0、置信度 > 0.5。
        // 这一条钉的是间隔的符号约定：置信度是 sigmoid(w · x − rho)，
        // 取成 +rho 或者把符号写反，值会整体掉到 0.5 以下 ——
        // 数值照样在 [0,1] 里、照样随输入变化，只有这一条抓得住。
        // （实测：留出集 180 张的置信度落在 0.731~0.851。）
        if (result.target_id != 0 && result.target_id ==
                                          DigitClassifier::targetIdForLabel(label)) {
            assert(result.confidence > 0.5F &&
                   "冠军没能正面赢过每个对手：间隔符号约定错了？");
        }
        confidence_min = std::min(confidence_min, result.confidence);
        confidence_max = std::max(confidence_max, result.confidence);

        const int expected = DigitClassifier::targetIdForLabel(label);
        if (result.target_id == expected) {
            ++correct;
            ++hit[label];
        }
        ++total[label];
    }

    const int n = static_cast<int>(items.size());
    const double overall = static_cast<double>(correct) / n;
    std::cout << "留出集 " << correct << "/" << n << " = "
              << overall * 100.0 << "%\n";
    std::cout << "置信度区间 [" << confidence_min << ", " << confidence_max
              << "]\n";
    // 恒为常数（0 或 1）的置信度没有任何信息量，却完全满足 [0,1] 契约。
    // 这一条只钉"随输入变化"，不假装它是标定概率 —— 那是另一回事。
    assert(confidence_max - confidence_min > 1e-6F && "置信度是常数，没有信息量");
    for (int c = 0; c < kClassTotal; ++c) {
        std::cout << "  " << DigitClassifier::className(c) << ": "
                  << hit[c] << "/" << total[c] << "\n";
    }

    // 阈值取 0.95 / 0.85：比"同源留出 99.9%"低一档。
    // 留出集只有每类 20 张，一张就是 5%，
    // 定到 99% 会让阈值本身比噪声还细 —— 那是自欺。
    assert(overall >= 0.95 && "留出集整体准确率不达标");
    for (int c = 0; c < kClassTotal; ++c) {
        assert(total[c] > 0);
        const double recall = static_cast<double>(hit[c]) / total[c];
        assert(recall >= 0.85 && "某一类 recall 不达标");
    }

    // =======================================================================
    // 4. 模型缺失不崩 —— 放最后，因为它会把分类器置于"不可用"状态
    // =======================================================================
    assert(!DigitClassifier::loadFrom("/nonexistent/definitely_not_here.xml"));
    assert(!DigitClassifier::ready());
    const DigitResult degraded = DigitClassifier::classify(uniformPatch(255));
    assert(degraded.target_id == 0);
    assert(degraded.confidence == 0.0F);

    std::cout << "全部通过\n";
    return 0;
}
