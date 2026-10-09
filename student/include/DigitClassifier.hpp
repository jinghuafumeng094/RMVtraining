#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

// 装甲板数字分类器（M4）。HOG 特征 + 线性 SVM，模型离线训练好后读文件。
//
// 这个头文件只依赖 OpenCV 的 core 和标准库，故意不依赖 ArmorDetector.hpp /
// AutoAimTypes.hpp。原因是训练程序 tools/train_digit_svm.cpp 也要用它 ——
// 训练程序只需要 OpenCV，不该被迫拖进相机、串口那一堆依赖。
namespace armor {

struct DigitResult {
    int target_id{0};  // 0 = 未识别；板子上没有数字时也是 0
    // [0,1]。冠军类在所有 one-vs-one 对局里的最小间隔，过一道 sigmoid。
    // > 0.5 表示冠军正面赢过了每一个对手；< 0.5 表示它是靠投票数赢的，
    // 至少输给过某个对手 —— 那种情况就是"看着像，其实没把握"。
    // 这是单调映射，**不是标定概率**：0.9 不等于"有 90% 的概率"。
    // （C++ 的 cv::ml::SVM 没有 setProbability，拿不到标定概率。）
    float confidence{0.0F};
};

class DigitClassifier {
public:
    // 输入补丁的尺寸，必须和数据集一致：28 高、20 宽。
    static constexpr int kPatchWidth = 20;
    static constexpr int kPatchHeight = 28;
    // HOG 特征长度。参数见 DigitClassifier.cpp，手算和实测都是 324。
    static constexpr int kFeatureSize = 324;
    static constexpr int kClassCount = 9;

    // 类别表。索引就是喂给 SVM 的标签，顺序是数据集目录的字典序。
    // 训练程序和推理程序必须用同一张表，所以只在这里定义一份。
    static const char* className(int label);
    // label → 装甲板目标编号。9neg（没有数字）映射到 0，
    // 正好对上 AutoAimTypes.hpp 里"0 保留给未知目标/无目标"的约定。
    static int targetIdForLabel(int label);

    // 把 28x20 的 CV_8UC1 补丁转成 324 维特征。
    // 输入不合法（空、类型不对、尺寸不对）时返回空 vector，不抛异常。
    static std::vector<float> describe(const cv::Mat& patch);

    // 端到端：补丁 → 目标编号 + 置信度。
    // 模型不可用时返回 {0, 0.0f}，绝不抛异常 —— 现场掉文件要降级，不能崩。
    static DigitResult classify(const cv::Mat& patch);

    // 显式加载模型。失败返回 false，并把当前分类器置为"不可用"。
    // 只有测试和需要换模型的场合才用，正常流程走 ready() 的懒加载。
    static bool loadFrom(const std::string& path);

    // 是否已有可用模型。首次调用时会尝试加载默认路径（懒加载）。
    // 加载过且失败之后不会反复重试 —— 免得每帧都去开一次文件。
    static bool ready();
};

}  // namespace armor
