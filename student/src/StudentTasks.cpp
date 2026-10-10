#include "StudentTasks.hpp"

#include "ArmorDetector.hpp"
#include "HikCamera.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>
#include <vector>

namespace {

// 装甲板物理尺寸(m)。来源：仓库根目录 装甲板尺寸.txt（135 x 57 / 230 x 57 mm）。
// 板上的数字编码了尺寸（序号 1~5 小板、6~8 大板），所以 size 判错时距离会按
// 错误的比例解出来，而且**不报错** —— 只是数字凭空差一截。
constexpr double kSmallWidthM = 0.135;
constexpr double kLargeWidthM = 0.230;
constexpr double kPlateHeightM = 0.057;

// 观测拒绝阈值。任务书 §3.5 点名要拒绝三类：PnP 失败、深度非法、误差异常。
// 这三个数是**用合成数据探针实测定的**（/tmp 上的一次性程序，不是拍脑袋）：
//   · 正常角点（自己投自己解）重投影误差 = 0.000 px
//   · 单个角点被拉开 10 px → 2.2 px；拉开 30 px → 7.0 px；拉开 60 px → 14.4 px
//   所以 5 px 大致相当于「某个角点错了 20 px 量级」，是"明显不对"而不是"抖了一下"。
// 深度上下限是物理约束：近于 5 cm 板已贴在镜头上，远于 30 m 超过 RM 场地长度
// （场地 28 m），两种都不可能是真目标，只会是把角点缩爆/拉爆后解出来的假位姿。
//
// ⚠️ 真实帧的角点抖动**还没量过** —— 相机一到手就拿真板子复测一遍，再回来调这三个数。
// 现在宁可松一点：误拒一个好目标比放进一个坏观测更难查。
constexpr double kMaxReprojectionErrorPx = 5.0;
constexpr double kMinDepthM = 0.05;
constexpr double kMaxDepthM = 30.0;

// 物体系：x 向右、y 向下、z 指向板背（背离相机）。
// 这样"板正对相机"时物体系与相机系重合，角点顺序也正好是
// 左上 右上 右下 左下，与 ArmorDetection::corners 的约定一致。
// y 取向下（不是向上）是有意的：跟图像/相机系同向，省掉一次 180° 翻转，
// 也让 armor_yaw 在正对时为 0 而不是 ±π。
std::vector<cv::Point3f> plateObjectPoints(ArmorSize size) {
    const double width = (size == ArmorSize::Large) ? kLargeWidthM : kSmallWidthM;
    const float x = static_cast<float>(width * 0.5);
    const float y = static_cast<float>(kPlateHeightM * 0.5);
    return {cv::Point3f(-x, -y, 0.0F), cv::Point3f(x, -y, 0.0F),
            cv::Point3f(x, y, 0.0F), cv::Point3f(-x, y, 0.0F)};
}

cv::Vec3d toVec3d(const cv::Mat& column) {
    return cv::Vec3d(column.at<double>(0), column.at<double>(1),
                     column.at<double>(2));
}

}  // namespace

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

// 任务 3：装甲板位姿解算。
//
// 用已标定的内参 + 装甲板已知物理尺寸做 PnP，得到板心在**相机系**中的位置，
// 再用外参把它刚体变换到**云台系**。
//
// 云台角度(gimbal)在这里用不上：外参 R/t 已经把相机系映射到云台系，
// 而云台系本身就是跟着云台转的那个系，当前角度不是这条变换的输入。
std::vector<ArmorPose> armor_solve(const std::vector<ArmorDetection>& detections,
                                   const CameraParameters& camera,
                                   const GimbalState&) {
    std::vector<ArmorPose> poses;

    // 没标定就不许解算。返回空，而不是拿占位参数硬算 ——
    // 后者给出的距离看着正常、实际全错，比"没有结果"难查得多。
    if (!camera.valid()) {
        return poses;
    }

    const cv::Mat& camera_matrix = camera.camera_matrix;
    const cv::Mat& distortion = camera.distortion_coefficients;
    const cv::Mat& R_camera_to_gimbal = camera.rotation_camera_to_gimbal;
    const cv::Mat& t_camera_to_gimbal = camera.translation_camera_to_gimbal;

    poses.reserve(detections.size());
    for (const ArmorDetection& detection : detections) {
        // 角点合法性闸，两种坏输入一起挡：
        //   · 面积接近 0（四点重合/共线）：solvePnP **不会失败**，会给出一个把板
        //     压成一点的解，重投影误差还是 0 —— 静默错误。
        //   · 面积为**负**：说明角点顺序反了。ArmorDetector.hpp:70 写着「顺序正确
        //     时为正」，所以负号本身就是输入非法的信号。这种输入 PnP 会给出一个
        //     **位置正确、法向翻转**的镜像解，重投影误差**仍然是 0** —— 靠误差和
        //     深度都拦不住，只有符号能拦。
        // 注意这里**不取绝对值**：取绝对值就等于把"顺序反了"当成正常输入放行。
        if (armor::quadSignedArea(detection.corners) < 1.0F) {
            continue;
        }

        const std::vector<cv::Point3f> object = plateObjectPoints(detection.size);
        const std::vector<cv::Point2f> image(detection.corners.begin(),
                                             detection.corners.end());

        cv::Vec3d rvec;
        cv::Vec3d tvec;
        if (!cv::solvePnP(object, image, camera_matrix, distortion, rvec, tvec,
                          false, cv::SOLVEPNP_ITERATIVE)) {
            continue;  // 解不出来的观测直接丢，不塞一个假的进结果
        }

        // 深度非法就丢（任务书 §3.5）。
        // solvePnP 返回 true 不代表解是物理上可能的：把角点缩到 2% 时它给的距离是
        // 100 m，角点摆成蝶形时给过 z = 5e11 —— 两种情况 solvePnP **都返回成功**，
        // 重投影误差还可能是 0。所以只能自己按物理范围兜一道。
        const double depth = tvec[2];
        if (!std::isfinite(depth) || depth < kMinDepthM || depth > kMaxDepthM) {
            continue;
        }

        // 单次观测的重投影误差：把解出来的位姿再投回图像，和实测角点比。
        // 这是这一条观测的自洽度，不是标定精度。注意它**看不见约定错误**：
        // 物体系轴指反、大小分类用错，重投影误差照样接近 0。
        std::vector<cv::Point2f> reprojected;
        cv::projectPoints(object, rvec, tvec, camera_matrix, distortion,
                          reprojected);
        double error_px = 0.0;
        for (std::size_t i = 0; i < image.size(); ++i) {
            error_px += cv::norm(reprojected[i] - image[i]);
        }
        error_px /= static_cast<double>(image.size());

        // 误差异常就丢（任务书 §3.5）。这是这条观测的**自洽度**：解出来的位姿再投
        // 回图像，和实测角点差多远。角点被拉开、灯条配错时它会涨上去，而面积和深度
        // 都还正常。注意它**看不见约定错误** —— 物体系轴指反、大小分类用错，
        // 重投影误差照样接近 0，那类问题只能靠 test_armor_solve 的断言钉住。
        if (!std::isfinite(error_px) || error_px > kMaxReprojectionErrorPx) {
            continue;
        }

        ArmorPose pose;
        pose.detection = detection;
        // solvePnP 的 tvec 就是板心在相机系中的位置(m)
        pose.position_camera_m = tvec;
        pose.reprojection_error = error_px;

        // 相机系 -> 云台系：一条刚体变换，全函数就这一处用到外参。
        const cv::Mat p_camera =
            (cv::Mat_<double>(3, 1) << tvec[0], tvec[1], tvec[2]);
        pose.position_gimbal_m = toVec3d(R_camera_to_gimbal * p_camera +
                                         t_camera_to_gimbal);

        // armor_yaw：装甲板法向在云台系中的偏航角。
        // 物体系 +z（指向板背）在相机系里就是旋转矩阵的第三列，
        // 换到云台系后取 atan2(x, z)。板正对相机时 = 0。
        cv::Mat R_object_to_camera;
        cv::Rodrigues(rvec, R_object_to_camera);
        const cv::Mat normal_gimbal =
            R_camera_to_gimbal * R_object_to_camera.col(2);
        pose.armor_yaw =
            std::atan2(normal_gimbal.at<double>(0), normal_gimbal.at<double>(2));

        poses.push_back(pose);
    }
    return poses;
}

PredictionResult ekf_predict(const std::vector<ArmorPose>&, const GimbalState&,
                             double) {
    return {};
}
