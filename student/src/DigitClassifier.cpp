#include "DigitClassifier.hpp"

#include <cmath>
#include <iostream>
#include <mutex>
#include <utility>
#include <vector>

#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>

namespace armor {
namespace {

// 类别表。索引 = 喂给 SVM 的标签。
// 顺序就是数据集目录的字典序：1 2 3 4 5 6outpost 7guard 8base 9neg。
// tools/train_digit_svm.cpp 用同一张表打标签，两边必须一致，
// 不一致的话准确率会正常显示，但每个数字都指向错的装甲板，而且不报错。
constexpr const char* kClassNames[] = {
    "1", "2", "3", "4", "5", "6outpost", "7guard", "8base", "9neg"};

// label 到 target_id 的映射。9neg 是"这块 ROI 里没有数字"，所以给 0。
constexpr int kTargetIds[] = {1, 2, 3, 4, 5, 6, 7, 8, 0};

// HOG 参数，和训练程序用的是同一套。
//   winSize     (20,28)  整幅补丁
//   cellSize    (5,7)    20/5 = 4 列，28/7 = 4 行，共 4x4 个 cell
//   blockSize   (10,14)  2x2 个 cell
//   blockStride (5,7)    步进 1 个 cell，所以 block 是 3x3 = 9 个
//   9 block x 4 cell x 9 bin = 324 维
// 放进函数内静态变量：HOGDescriptor 构造要算一堆内部表，
// 每帧重造是浪费；C++11 起函数内静态变量的初始化是线程安全的。
cv::HOGDescriptor& hog() {
    static cv::HOGDescriptor instance(
        cv::Size(DigitClassifier::kPatchWidth, DigitClassifier::kPatchHeight),
        cv::Size(10, 14), cv::Size(5, 7), cv::Size(5, 7), 9);
    return instance;
}

// 一条 one-vs-one 决策函数的原始参数。
//
// 这里解释为什么置信度要自己算，不能用 predict 的输出：
// 9 类 SVM 在 OpenCV 里是 one-vs-one，共 9*8/2 = 36 个二分类器。
// 多类情况下 `predict(..., RAW_OUTPUT)` 返回的是类别标签本身，不是决策函数值
// （实测：类 1 的样本 raw 恒为 0.0，类 2 恒为 1.0，…… 类 9neg 恒为 8.0）。
// 拿它当 margin，置信度就成了"预测出哪个数字"的纯函数，跟输入图像毫无关系，
// 而且看起来还挺像回事（数字越大越"可信"），这是最难发现的那种错。
//
// 好在二分类器本身是拿得到的：线性核下 OpenCV 把每个二分类器压缩成
// "一个原始权重向量 + 一个偏置 rho"，getSupportVectors() 的每一行就是一个权重。
// 实测验证（1440 个「样本 × 它自己参与的每一对」）：
//     dfval = w · x - rho，> 0 表示 pair.first 胜
//     符号符合率 100%（写成 + rho 只有 81%，是错的）。
struct Pair {
    int first{0};       // 这对里的类别 A
    int second{0};      // 类别 B，first < second
    int weight_row{0};  // 权重在 getSupportVectors() 里的行号
    float rho{0.0F};    // 偏置
};

struct Model {
    cv::Ptr<cv::ml::SVM> svm;
    cv::Mat weights;        // 36 x 324 的原始权重
    std::vector<Pair> pairs;  // 36 个 one-vs-one 对局
    bool loaded{false};     // 当前真的有可用模型
    bool attempted{false};  // 已经试过加载（含失败），不要再自动重试
};

// 全局状态：一个模型句柄加一把锁。
// 主循环是单线程的，但 Web Debug 那边有别的线程在跑，
// 加锁的成本远低于一个偶发数据竞争带来的调试成本。
std::mutex& modelMutex() {
    static std::mutex instance;
    return instance;
}

Model& model() {
    static Model instance;
    return instance;
}

const char* defaultModelPath() {
#ifdef ARMOR_SVM_PATH
    return ARMOR_SVM_PATH;
#else
    return "models/armor_digit_svm.xml";
#endif
}

// 试着加载一个文件。调用方必须已持有 modelMutex()。
// SVM::load 在文件不存在时会抛 cv::Exception，所以这里必须接住：
// 模型丢了是要降级运行，不是要让整个程序挂掉。
//
// 只接受"线性核 + one-vs-one 压缩形式"的模型。别的核（比如 RBF）
// 取不到每对的原始权重，置信度就只能退化成常数，与其那样，
// 不如在这里明确拒绝，让问题暴露在加载阶段而不是答辩现场。
bool tryLoadLocked(const std::string& path) {
    try {
        cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(path);
        if (svm.empty() || !svm->isTrained()) {
            return false;
        }

        // one-vs-one：k 类共 k(k-1)/2 个二分类器
        const int class_count = DigitClassifier::kClassCount;
        const int pair_count = class_count * (class_count - 1) / 2;
        const cv::Mat weights = svm->getSupportVectors();
        if (svm->getKernelType() != cv::ml::SVM::LINEAR ||
            weights.rows != pair_count ||
            weights.cols != DigitClassifier::kFeatureSize) {
            std::cerr << "[digit] 模型 " << path
                      << " 不可用：需要线性核、且能取到 one-vs-one 原始权重。\n"
                         "        （拿到 " << svm->getKernelType() << " 核 / "
                      << weights.rows << "x" << weights.cols
                      << " 权重，期望 " << pair_count << "x"
                      << DigitClassifier::kFeatureSize
                      << "）用 tools/train_digit_svm 产出的模型即可。"
                      << std::endl;
            return false;
        }

        // 枚举顺序和 OpenCV 内部一致：(0,1) (0,2) … (0,8) (1,2) …
        std::vector<Pair> pairs;
        pairs.reserve(static_cast<std::size_t>(pair_count));
        int p = 0;
        for (int i = 0; i < class_count; ++i) {
            for (int j = i + 1; j < class_count; ++j, ++p) {
                cv::Mat alpha;
                cv::Mat svidx;
                const float rho = svm->getDecisionFunction(p, alpha, svidx);
                // 压缩形式：每对只有一个"支持向量"，就是那个原始权重。
                // 不是这个形状就说明压缩没生效，那 weight_row 取法也不成立。
                if (alpha.total() != 1 || svidx.total() != 1) {
                    std::cerr << "[digit] 决策函数 " << p
                              << " 不是压缩形式（alpha " << alpha.total()
                              << " / svidx " << svidx.total()
                              << "），无法当线性权重用。" << std::endl;
                    return false;
                }
                Pair pair;
                pair.first = i;
                pair.second = j;
                pair.weight_row = svidx.at<int>(0);
                pair.rho = rho;
                pairs.push_back(pair);
            }
        }

        Model& state = model();
        state.weights = weights;
        state.pairs = std::move(pairs);
        state.svm = svm;
        return true;
    } catch (const cv::Exception& error) {
        std::cerr << "[digit] 加载模型失败 " << path << "：" << error.what()
                  << std::endl;
        return false;
    }
}

}  // namespace

const char* DigitClassifier::className(int label) {
    if (label < 0 || label >= kClassCount) {
        return "?";
    }
    return kClassNames[label];
}

int DigitClassifier::targetIdForLabel(int label) {
    if (label < 0 || label >= kClassCount) {
        return 0;
    }
    return kTargetIds[label];
}

std::vector<float> DigitClassifier::describe(const cv::Mat& patch) {
    std::vector<float> descriptor;
    // 三个条件缺一不可。尺寸判断要写 cols/rows 而不是 rows/cols，
    // 20 和 28 反了的话，HOG 会算出完全不同的特征，而且一声不吭。
    if (patch.empty() || patch.type() != CV_8UC1 ||
        patch.cols != kPatchWidth || patch.rows != kPatchHeight) {
        return descriptor;
    }
    hog().compute(patch, descriptor);
    return descriptor;
}

bool DigitClassifier::ready() {
    std::lock_guard<std::mutex> lock(modelMutex());
    Model& state = model();
    if (state.loaded) {
        return true;
    }
    if (state.attempted) {
        return false;
    }
    state.attempted = true;

    // 先试编进二进制的绝对路径（不依赖工作目录）。
    if (tryLoadLocked(defaultModelPath())) {
        state.loaded = true;
        std::cerr << "[digit] 模型已加载：" << defaultModelPath() << std::endl;
        return true;
    }
    // 再试相对路径兜底（从仓库根目录运行时能命中）。
    if (tryLoadLocked("models/armor_digit_svm.xml")) {
        state.loaded = true;
        std::cerr << "[digit] 模型已加载：models/armor_digit_svm.xml（相对路径）"
                  << std::endl;
        return true;
    }
    std::cerr << "[digit] 没有可用模型，数字分类全部返回 target_id = 0。\n"
                 "[digit] 请先把 models/armor_digit_svm.xml 放进仓库。"
              << std::endl;
    return false;
}

bool DigitClassifier::loadFrom(const std::string& path) {
    std::lock_guard<std::mutex> lock(modelMutex());
    Model& state = model();
    state.attempted = true;
    // 显式加载失败 = 当前位置没有可用模型，先把旧的扔掉，
    // 免得"加载失败但还在用上一个模型"这种半吊子状态。
    state.svm.release();
    state.loaded = false;
    state.loaded = tryLoadLocked(path);
    return state.loaded;
}

DigitResult DigitClassifier::classify(const cv::Mat& patch) {
    DigitResult result;  // 默认就是 {target_id = 0, confidence = 0}
    if (!ready()) {
        return result;
    }

    const std::vector<float> descriptor = describe(patch);
    if (descriptor.empty()) {
        return result;
    }

    std::lock_guard<std::mutex> lock(modelMutex());
    Model& state = model();
    if (!state.loaded || state.svm.empty()) {
        return result;
    }

    // 单样本：1 行 x 324 列的 CV_32F。这里直接包住 vector 的内存不拷贝，
    // descriptor 在本次调用结束前一直活着，所以是安全的。
    const cv::Mat sample(1, static_cast<int>(descriptor.size()), CV_32F,
                         const_cast<float*>(descriptor.data()));

    cv::Mat label_out;
    state.svm->predict(sample, label_out);
    if (label_out.total() < 1) {
        return result;
    }
    const int label = static_cast<int>(label_out.at<float>(0) + 0.5F);
    result.target_id = targetIdForLabel(label);

    // 置信度：冠军类在所有对局里的最小间隔
    //
    // 取最小值而不是平均值：只要有一个对手没被明显压住，就不该报高置信度。
    // 这正是"这个数字和某个易混数字之间只差一点"的样子。
    //
    // 间隔带符号：> 0 表示冠军确实赢了这个对手。所以 sigmoid 出来的值
    // 天然落在 (0,1) 且以 0.5 为界，低于 0.5 就意味着
    // "预测出的这个类，至少输给过某个对手"，是实打实的没把握。
    bool have_margin = false;
    float worst = 0.0F;
    for (const Pair& pair : state.pairs) {
        int sign = 0;
        if (pair.first == label) {
            sign = 1;
        } else if (pair.second == label) {
            sign = -1;
        } else {
            continue;  // 跟冠军无关的对局，不影响这个结论
        }
        const float dfval =
            static_cast<float>(state.weights.row(pair.weight_row).dot(sample)) -
            pair.rho;
        const float signed_margin = dfval * static_cast<float>(sign);
        if (!have_margin || signed_margin < worst) {
            worst = signed_margin;
            have_margin = true;
        }
    }
    if (!have_margin) {
        return result;  // 理论上到不了；到了就当没分类成功
    }

    // 关于置信度的诚实说明：
    // C++ 版的 cv::ml::SVM 没有 setProbability/getProbability
    // （Python 版才有），所以拿不到标定概率。这里给的是
    // "最小间隔经过 sigmoid" 的单调映射，
    // 它只保证"离分界面越远越大"，不是"有 92% 的概率是这个数字"。
    // 用它排序、显示、卡阈值都可以，但当概率解释会错。
    result.confidence = 1.0F / (1.0F + std::exp(-worst));
    return result;
}

}  // namespace armor
