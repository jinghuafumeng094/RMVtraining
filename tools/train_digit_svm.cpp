// 训练装甲板数字分类器（M4）：HOG 特征 + 线性 SVM。
//
// 用法：train_digit_svm <数据集根> <输出模型> <留出清单>
//
// 三个参数全必填：
//   - 数据集不在仓库里（体积 + 来源都不合适），所以路径不能硬编码；
//   - 留出清单必填是防"忘了排除留出图"。那种错误会让准确率虚高，
//     而且不会报任何错 —— 只会在答辩现场露馅。
//
// 程序输出两类东西：
//   一、诊断。同一个特征在三种输入上分别能到多少准确率：
//         完整补丁 / 只留中心 / 只留最外 2 像素的框
//       这一组数字回答的是"模型到底在靠字形分类，还是在靠边缘杂讯"。
//       只留边缘若还能到 90%，说明特征方案有问题，得回头改设计。
//   二、模型。用全部非留出图训练，存成 XML。
#include "DigitClassifier.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <glob.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ml.hpp>

namespace {

// 固定种子 → 每次划分完全一样，报出来的数字可复现。
constexpr std::uint32_t kSplitSeed = 20261008U;
constexpr double kTestRatio = 0.2;
// C 取 1.0：实测线性核在 C=1 时已经到 99.9% 以上，
// 调大只会让边界更硬、更容易过拟合，没有收益。
constexpr double kSvmC = 1.0;
constexpr int kClassCount = armor::DigitClassifier::kClassCount;

struct Sample {
    std::string relative;  // 相对数据集根，同时也是留出清单里的键
    cv::Mat patch;
    int label{0};
};

// 三种输入变体。用同一套 HOG 参数（20x28 窗口）处理，
// 只改图的内容，这样比较的是"信息在哪里"，而不是"参数变了"。
enum class Mode { Full, CenterOnly, BorderOnly };

std::string join(const std::string& dir, const std::string& name) {
    if (!dir.empty() && dir.back() == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

std::string baseName(const std::string& path) {
    const std::size_t pos = path.find_last_of('/');
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

// glob 默认按文件名排序，顺序确定 —— 可复现性的一部分。
std::vector<std::string> globFiles(const std::string& pattern) {
    std::vector<std::string> files;
    glob_t result{};
    if (glob(pattern.c_str(), 0, nullptr, &result) == 0) {
        for (std::size_t i = 0; i < result.gl_pathc; ++i) {
            files.emplace_back(result.gl_pathv[i]);
        }
    }
    globfree(&result);
    return files;
}

// 逐行读，跳过 # 开头的注释行。
//
// 这里踩过坑，记下来：最初的写法是 `while (in >> relative >> label)`。
// 清单第一行是注释，`>>` 会把 "#" 读进 relative，然后拿"<相对路径>"去读 int，
// 流状态一失败循环立刻结束 —— 整个清单读成 0 条，**不报任何错**，
// 180 张留出图就这么混进了训练集。准确率只会偏高，看不出异常。
// 所以下面除了跳过注释，还加了一道"读完必须是 0 条以外"的检查。
bool readManifest(const std::string& path, std::set<std::string>& keys) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }
        std::istringstream stream(line);
        std::string relative;
        int label = 0;
        if (stream >> relative >> label) {
            keys.insert(relative);
        }
    }
    return true;
}

cv::Mat applyMode(const cv::Mat& patch, Mode mode) {
    if (mode == Mode::Full) {
        return patch;
    }
    cv::Mat masked = patch.clone();
    const int w = masked.cols;
    const int h = masked.rows;
    if (mode == Mode::CenterOnly) {
        // 把最外 2 像素的框涂黑，只留中心
        masked.rowRange(0, 2).setTo(0);
        masked.rowRange(h - 2, h).setTo(0);
        masked.colRange(0, 2).setTo(0);
        masked.colRange(w - 2, w).setTo(0);
    } else {
        // 反过来：把中心涂黑，只留最外 2 像素的框
        masked(cv::Rect(2, 2, w - 4, h - 4)).setTo(0);
    }
    return masked;
}

cv::Mat featureMatrix(const std::vector<Sample>& samples, Mode mode) {
    cv::Mat features(static_cast<int>(samples.size()),
                     armor::DigitClassifier::kFeatureSize, CV_32F);
    for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
        const std::vector<float> descriptor =
            armor::DigitClassifier::describe(applyMode(samples[i].patch, mode));
        if (static_cast<int>(descriptor.size()) !=
            armor::DigitClassifier::kFeatureSize) {
            std::cerr << "特征维度不对：" << samples[i].relative << "\n";
            return cv::Mat();
        }
        std::copy(descriptor.begin(), descriptor.end(), features.ptr<float>(i));
    }
    return features;
}

bool loadDataset(const std::string& root, const std::set<std::string>& holdout,
                 std::vector<Sample>& samples, std::size_t& matched_holdout) {
    samples.clear();
    matched_holdout = 0;
    for (int label = 0; label < kClassCount; ++label) {
        const std::string name = armor::DigitClassifier::className(label);
        const std::vector<std::string> files = globFiles(join(root, name) + "/*");
        if (files.empty()) {
            std::cerr << "类别 " << name << " 一张图都没找到，检查数据集根目录\n";
            return false;
        }
        int kept = 0;
        int skipped_holdout = 0;
        for (const std::string& path : files) {
            const std::string key = name + "/" + baseName(path);
            if (holdout.count(key) > 0) {
                ++skipped_holdout;
                ++matched_holdout;
                continue;  // 留出图绝不能进训练
            }
            const cv::Mat patch = cv::imread(path, cv::IMREAD_GRAYSCALE);
            if (patch.empty()) {
                continue;
            }
            if (patch.cols != armor::DigitClassifier::kPatchWidth ||
                patch.rows != armor::DigitClassifier::kPatchHeight) {
                continue;  // 尺寸不对的直接跳过，不去猜该怎么缩放
            }
            samples.push_back(Sample{key, patch, label});
            ++kept;
        }
        std::cout << "  " << std::setw(9) << std::left << name << " 训练 "
                  << std::setw(5) << std::right << kept << " 留出排除 "
                  << skipped_holdout << "\n";
    }
    return true;
}

// 分层划分：每一类内部按固定种子洗牌，各取 20% 当测试。
// 不能用整体洗牌 —— 9neg 占三分之一，整体洗牌偶尔会全落到一边去。
void stratifiedSplit(const std::vector<Sample>& samples, std::vector<int>& train,
                     std::vector<int>& test) {
    std::vector<std::vector<int>> groups(kClassCount);
    for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
        groups[samples[i].label].push_back(i);
    }
    std::mt19937 rng(kSplitSeed);
    train.clear();
    test.clear();
    for (auto& group : groups) {
        std::shuffle(group.begin(), group.end(), rng);
        const std::size_t count = std::max<std::size_t>(
            1, static_cast<std::size_t>(group.size() * kTestRatio));
        for (std::size_t i = 0; i < group.size(); ++i) {
            (i < count ? test : train).push_back(group[i]);
        }
    }
}

cv::Ptr<cv::ml::SVM> makeSvm() {
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::LINEAR);
    svm->setC(kSvmC);
    return svm;
}

cv::Ptr<cv::ml::SVM> trainSvm(const cv::Mat& features,
                              const std::vector<Sample>& samples,
                              const std::vector<int>& indices) {
    const int n = static_cast<int>(indices.size());
    cv::Mat x(n, features.cols, CV_32F);
    cv::Mat y(n, 1, CV_32S);
    for (int i = 0; i < n; ++i) {
        features.row(indices[i]).copyTo(x.row(i));
        y.at<int>(i, 0) = samples[indices[i]].label;
    }
    cv::Ptr<cv::ml::SVM> svm = makeSvm();
    svm->train(x, cv::ml::ROW_SAMPLE, y);
    return svm;
}

struct Report {
    double accuracy{0.0};
    int test_count{0};
    std::vector<double> recall;
    std::vector<std::vector<int>> confusion;
};

Report evaluate(cv::ml::SVM& svm, const cv::Mat& features,
                const std::vector<Sample>& samples,
                const std::vector<int>& test_indices) {
    Report report;
    report.recall.assign(kClassCount, 0.0);
    report.confusion.assign(kClassCount, std::vector<int>(kClassCount, 0));
    std::vector<int> total(kClassCount, 0);
    int correct = 0;
    for (int index : test_indices) {
        const int truth = samples[index].label;
        cv::Mat out;
        // clone：row() 是不连续的行视图，clone 成一块连续内存再喂给 predict，
        // 免得踩到 OpenCV 对"一行样本"的内存假设。
        svm.predict(features.row(index).clone(), out);
        const int predicted = static_cast<int>(out.at<float>(0) + 0.5F);
        ++total[truth];
        if (predicted >= 0 && predicted < kClassCount) {
            ++report.confusion[truth][predicted];
        }
        if (predicted == truth) {
            ++correct;
        }
    }
    report.test_count = static_cast<int>(test_indices.size());
    report.accuracy = report.test_count > 0
                          ? static_cast<double>(correct) / report.test_count
                          : 0.0;
    for (int c = 0; c < kClassCount; ++c) {
        report.recall[c] =
            total[c] > 0 ? static_cast<double>(report.confusion[c][c]) / total[c]
                         : 0.0;
    }
    return report;
}

void printReport(const std::string& title, const Report& report) {
    std::cout << "  " << std::left << std::setw(26) << title << std::right
              << std::fixed << std::setprecision(2)
              << report.accuracy * 100.0 << "%   (n=" << report.test_count << ")";
    std::cout << "  最低类 recall ";
    double worst = 1.0;
    const char* worst_name = "?";
    for (int c = 0; c < kClassCount; ++c) {
        if (report.recall[c] < worst) {
            worst = report.recall[c];
            worst_name = armor::DigitClassifier::className(c);
        }
    }
    std::cout << worst_name << " " << worst * 100.0 << "%\n";
}

void printConfusion(const Report& report) {
    std::cout << "      预测→ ";
    for (int c = 0; c < kClassCount; ++c) {
        std::cout << std::setw(9) << armor::DigitClassifier::className(c);
    }
    std::cout << "\n";
    for (int r = 0; r < kClassCount; ++r) {
        std::cout << "  " << std::setw(6) << std::left
                  << armor::DigitClassifier::className(r) << std::right;
        for (int c = 0; c < kClassCount; ++c) {
            const int v = report.confusion[r][c];
            std::cout << std::setw(9) << (v == 0 ? std::string(".") : std::to_string(v));
        }
        std::cout << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "用法：train_digit_svm <数据集根> <输出模型> <留出清单>\n"
                     "例：  train_digit_svm ~/armor_digit_dataset/datasets \\\n"
                     "          models/armor_digit_svm.xml \\\n"
                     "          tests/fixtures/digit_holdout/manifest.txt\n";
        return 2;
    }
    const std::string root = argv[1];
    const std::string model_out = argv[2];
    const std::string manifest_path = argv[3];

    std::set<std::string> holdout;
    if (!readManifest(manifest_path, holdout)) {
        std::cerr << "读不到留出清单：" << manifest_path
                  << "\n清单是防数据泄漏用的，不能省。先跑 make_digit_holdout 生成。\n";
        return 2;
    }
    if (holdout.empty()) {
        std::cerr << "留出清单一条都没读出来：" << manifest_path
                  << "\n空清单等于没有留出集，训练集会把留出图吃进去。拒绝继续。\n";
        return 2;
    }
    std::cout << "留出清单 " << holdout.size() << " 条\n读取数据集 " << root << "：\n";

    std::vector<Sample> samples;
    std::size_t matched_holdout = 0;
    if (!loadDataset(root, holdout, samples, matched_holdout)) {
        return 2;
    }
    // 清单里的每一个键都必须真的在数据集里认领到一张图。
    // 认不到就说明路径拼法对不上（清单里的目录名写错了、数据集换了版本……），
    // 后果是那张图照旧进训练集 —— 又是静默泄漏。这里直接拦死。
    if (matched_holdout != holdout.size()) {
        std::cerr << "清单里有 " << (holdout.size() - matched_holdout)
                  << " 条没在数据集里找到对应文件，清单和数据集对不上，拒绝继续。\n";
        return 2;
    }
    std::cout << "共 " << samples.size() << " 张参与训练/评估（留出集 "
              << matched_holdout << " 张已排除）\n";

    std::vector<int> train_indices;
    std::vector<int> test_indices;
    stratifiedSplit(samples, train_indices, test_indices);

    std::cout << "\n[诊断] 同一套特征，三种输入。要回答的是「模型学的是不是字形」\n";
    std::cout << "  划分：分层 80/20，种子固定 " << kSplitSeed
              << "，训练 " << train_indices.size() << " 测试 "
              << test_indices.size() << "\n";
    const cv::Mat features_full = featureMatrix(samples, Mode::Full);
    const cv::Mat features_center = featureMatrix(samples, Mode::CenterOnly);
    const cv::Mat features_border = featureMatrix(samples, Mode::BorderOnly);
    if (features_full.empty() || features_center.empty() ||
        features_border.empty()) {
        return 2;
    }
    const cv::Ptr<cv::ml::SVM> svm_full =
        trainSvm(features_full, samples, train_indices);
    const cv::Ptr<cv::ml::SVM> svm_center =
        trainSvm(features_center, samples, train_indices);
    const cv::Ptr<cv::ml::SVM> svm_border =
        trainSvm(features_border, samples, train_indices);

    // 自己训、自己测 —— 三种输入各自的水平。
    const Report full_report = evaluate(*svm_full, features_full, samples, test_indices);
    printReport("完整补丁", full_report);
    printReport("只留中心（外圈涂黑）",
                evaluate(*svm_center, features_center, samples, test_indices));
    printReport("只留最外2px框",
                evaluate(*svm_border, features_border, samples, test_indices));
    std::cout << "  判读：'只留最外2px框'若还能到 90% 以上，说明边缘本身能唯一确定类别，\n"
                 "        特征是虚的，要回头改方案。\n";

    // 交叉诊断 —— 这几行才是决定性的。
    //
    // "只留最外2px框"那 80% 看着吓人，但它测的是"边缘信息够不够用"，
    // 不是"完整模型有没有在用边缘"。这两件事完全不同。
    // 真正关系到现场风险的是下面第一行：完整模型喂进"边缘被涂掉"的输入
    // （模拟现场补丁的外圈和数据集对不上），准确率掉不掉。
    // 不掉 → 模型靠的是字形，边缘那 80% 是数据集的伪线索，与部署风险无关。
    // 掉 → 模型确实在依赖外圈，现场就有得受了。
    std::cout << "\n[交叉诊断] 用 A 训出的模型，喂 B 的输入（测试图不变）：\n";
    printReport("完整模型 → 只留中心",
                evaluate(*svm_full, features_center, samples, test_indices));
    printReport("完整模型 → 只留最外2px框",
                evaluate(*svm_full, features_border, samples, test_indices));
    printReport("只留中心模型 → 完整补丁",
                evaluate(*svm_center, features_full, samples, test_indices));
    std::cout << "  判读：第一行若接近「完整补丁」那一行，说明外圈丢了也不影响判字，\n"
                 "        模型的依据是字形；若明显下掉，说明它在依赖外圈，现场会崩。\n";

    std::cout << "\n完整补丁的混淆矩阵（. 表示 0）：\n";
    printConfusion(full_report);

    // 最终模型用**全部**非留出图训练 —— 比只在 80% 上训练更强。
    // 上面报的准确率是 80/20 划分上的估计值，比最终模型略悲观，这是正常的。
    const cv::Mat features = featureMatrix(samples, Mode::Full);
    std::vector<int> all_indices(samples.size());
    for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
        all_indices[i] = i;
    }
    const cv::Ptr<cv::ml::SVM> final_svm = trainSvm(features, samples, all_indices);
    try {
        final_svm->save(model_out);
    } catch (const cv::Exception& error) {
        std::cerr << "模型保存失败 " << model_out << "：" << error.what() << "\n";
        return 2;
    }
    std::cout << "\n最终模型：用全部 " << all_indices.size()
              << " 张非留出图训练，已存到 " << model_out << "\n"
              << "留出集（tests/fixtures/digit_holdout）没参与训练，"
                 "由 test_digit_classifier 拿去评估。\n";
    return 0;
}
