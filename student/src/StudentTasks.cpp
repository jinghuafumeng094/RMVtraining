#include "StudentTasks.hpp"

#include "ArmorDetector.hpp"

#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace {

// ===========================================================================
// 相机状态。get_pic 只有它自己一个入口，没有"初始化/关闭"的地方，
// 所以状态只能放在这里：第一次调用时打开，之后每帧复用同一个句柄。
// ===========================================================================
cv::VideoCapture g_capture;      // 相机句柄
bool g_camera_ready = false;     // 是否已经成功打开过
int  g_fail_streak  = 0;         // 连续取流失败次数

// 开发期先用笔记本自带摄像头。将来换海康，改这一段 + openCameraOnce 内部即可。
constexpr int kDeviceIndex = 0;  // 0 = /dev/video0
constexpr int kFrameWidth  = 640;
constexpr int kFrameHeight = 480;

// 打开相机 —— 只在第一次调用时真正执行。
bool openCameraOnce() {
    if (g_camera_ready) {
        return true;
    }

    std::cout << "[get_pic] 正在打开相机 ..." << std::endl;

    // CAP_V4L2：Linux 下指定 V4L2 后端，行为比默认后端可预期。
    if (!g_capture.open(kDeviceIndex, cv::CAP_V4L2)) {
        std::cerr << "[get_pic] 打不开相机 " << kDeviceIndex << std::endl;
        return false;
    }

    g_capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    g_capture.set(cv::CAP_PROP_FRAME_WIDTH,  kFrameWidth);
    g_capture.set(cv::CAP_PROP_FRAME_HEIGHT, kFrameHeight);
    // 只留 1 帧缓冲：我们永远只要最新帧，多缓存只会让你拿到旧图。
    g_capture.set(cv::CAP_PROP_BUFFERSIZE, 1);

    std::cout << "[get_pic] 相机已就绪，实际分辨率 "
              << static_cast<int>(g_capture.get(cv::CAP_PROP_FRAME_WIDTH)) << "x"
              << static_cast<int>(g_capture.get(cv::CAP_PROP_FRAME_HEIGHT)) << std::endl;

    g_camera_ready = true;
    return true;
}

}  // namespace

bool get_pic(cv::Mat& pic) {
    // 先清空：保证无论成功还是失败，调用者拿到的都是干净状态。
    pic.release();

    if (!openCameraOnce()) {
        return false;
    }

    cv::Mat raw;
    // read() 内部会等一帧；相机超时或掉线时返回 false。
    if (!g_capture.read(raw) || raw.empty()) {
        ++g_fail_streak;
        std::cerr << "[get_pic] 取流失败，连续第 " << g_fail_streak << " 次" << std::endl;
        // 连着失败多次说明句柄已经废了：释放掉，下次调用重新打开。
        if (g_fail_streak >= 5) {
            g_capture.release();
            g_camera_ready = false;
            g_fail_streak  = 0;
            std::cerr << "[get_pic] 连续失败，已释放句柄，下次重连" << std::endl;
        }
        return false;
    }
    g_fail_streak = 0;

    // 统一转成三通道 BGR —— 后面每一级都按 BGR 处理。
    // 将来接海康，像素格式转换（Bayer -> BGR）就落在这个位置。
    if (raw.channels() == 3) {
        pic = raw;
    } else if (raw.channels() == 4) {
        cv::cvtColor(raw, pic, cv::COLOR_BGRA2BGR);
    } else if (raw.channels() == 1) {
        pic = raw;
        cv::cvtColor(raw, pic, cv::COLOR_GRAY2BGR);
    } else {
        std::cerr << "[get_pic] 不支持的通道数: " << raw.channels() << std::endl;
        return false;
    }

    // 保证内存连续：后面按行整块访问时才不会踩空。
    if (!pic.isContinuous()) {
        pic = pic.clone();
    }
    return true;
}

// ↓↓↓ 下面三个函数先原样留着，第三、四、五课再填 ↓↓↓
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
