// 抓一帧存盘。用途只有一个：验证 student/src/StudentTasks.cpp 里的 get_pic()。
//
// 它**不需要 config/camera.yaml** —— 取流和标定是两回事，内参外参没齐
// 不影响相机能不能出图。所以相机一到手，这是第一件能验的事。
//
// 用法：
//     grab_one [输出文件]        默认 out.png
//
// 现场留证：把落盘的 PNG 存进证据目录，再把这行输出一起贴上去。

#include "StudentTasks.hpp"

#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    const std::string outPath = (argc > 1) ? argv[1] : "out.png";

    // 相机刚上电、或 MVS 客户端刚松手时，头几帧可能拿不到，
    // 给它几次机会。真断了也会在这里把 20 次耗完，然后明确失败。
    constexpr int kAttempts = 20;
    cv::Mat pic;
    for (int i = 0; i < kAttempts; ++i) {
        if (get_pic(pic) && !pic.empty()) {
            break;
        }
        std::cerr << "第 " << (i + 1) << " 次没取到\n";
    }

    if (pic.empty()) {
        std::cerr << "连续 " << kAttempts << " 次都没取到图，放弃\n";
        return 1;
    }

    // 只看 B/G/R 三个均值：三个数几乎相等说明图是灰的（或者 Bayer 相位
    // 弄错被当成了灰度）。但"红蓝对调"这种错**均值看不出来** ——
    // 那个只能靠眼睛看存下来的 PNG。所以下面这句只是粗筛，图必须看。
    const cv::Scalar mean = cv::mean(pic);
    std::cout << "尺寸:     " << pic.cols << "x" << pic.rows << "\n";
    std::cout << "通道:     " << pic.channels()
              << "  位深: " << pic.depth() << "（0 = CV_8U）\n";
    std::cout << "通道均值: B=" << mean[0] << "  G=" << mean[1]
              << "  R=" << mean[2] << "\n";

    // 契约：get_pic 成功时必须给一张非空 CV_8UC3 BGR。
    // 后面每一环（颜色掩膜、灯条、数字）都指望这个前提。
    const bool contractOk = (pic.type() == CV_8UC3);
    std::cout << "get_pic 契约（非空 CV_8UC3）: " << (contractOk ? "✅" : "❌")
              << "\n";

    if (!cv::imwrite(outPath, pic)) {
        std::cerr << "写文件失败: " << outPath
                  << "（目标目录存在吗？）\n";
        return 1;
    }
    std::cout << "已存: " << outPath << "\n";
    return contractOk ? 0 : 1;
}
