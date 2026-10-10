#include "StudentTasks.hpp"

#include "ArmorDetector.hpp"
#include "HikCamera.hpp"

#include <opencv2/imgproc.hpp>

// 任务 1：相机取流。
//
// 真正碰 MVS SDK 的地方全在 HikCamera 里，这里只管一件事：**进程内只开一次**。
// 函数局部 static 的初始化从 C++11 起是线程安全的，所以第一次调用时打开相机，
// 之后每帧复用同一个句柄；失败一律返回 false，不往上抛也不终止进程。
bool get_pic(cv::Mat& pic) {
    static HikCamera camera;
    return camera.grab(pic);
}

// ↓↓↓ 下面两个函数先原样留着，第四、五课再填 ↓↓↓
std::vector<ArmorDetection> armor_detect(const cv::Mat& image, TeamColor team) {
    // 整条检测管线都在 ArmorDetector 里：颜色掩膜 → 灯条提取 → 配对
    // → 角点 → 大小分类 → 去重 → 数字分类。
    // 这里只做转发，保持本文件是"把各部分接起来"的地方，
    // 算法本身留在 ArmorDetector，能脱离相机单独测。
    return armor::detectArmors(image, team);
}

std::vector<ArmorPose> armor_solve(const std::vector<ArmorDetection>&,
                                  const CameraParameters&, const GimbalState&) {
    return {};
}

PredictionResult ekf_predict(const std::vector<ArmorPose>&, const GimbalState&,
                             double) {
    return {};
}
