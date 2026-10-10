// armor_solve 位姿解算的测试。
//
// 不需要相机、不需要数据集、不需要标定真值 —— 自己造一组"已知位姿"的三维点，
// 用已知内参投到像素上，再让 armor_solve 从这些像素把位姿解回来。
// 对得上，说明"物体系约定 -> solvePnP -> 外参变换"整条链路自洽。
//
// 这个测试真正防的是**约定类**的静默错误：
//   · 物体系 x/y/z 指反了        -> 位置符号错，但重投影误差照样是 0，不报错
//   · 大小分类用错（135 vs 230） -> 距离按错误比例缩放，同样不报错
//   · 外参那一行漏乘/乘反        -> 云台系坐标全错
// 这些都不会崩、不会警告，只会让打出的距离差一截。所以只能靠断言钉住。
#include "StudentTasks.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

namespace {

// 与实现**独立**地重申物体系约定（故意手写一遍，不调用实现里的 helper，
// 否则实现改了约定、测试跟着一起改，就白测了）：
//   x 向右、y 向下、z 指向板背（背离相机）；角点顺序 左上 右上 右下 左下。
//   单位 m。板高两种尺寸都是 57 mm。
std::vector<cv::Point3f> platePoints(double widthM) {
    const double x = widthM * 0.5, y = 0.057 * 0.5;
    return {cv::Point3f(static_cast<float>(-x), static_cast<float>(-y), 0.0F),
            cv::Point3f(static_cast<float>(x), static_cast<float>(-y), 0.0F),
            cv::Point3f(static_cast<float>(x), static_cast<float>(y), 0.0F),
            cv::Point3f(static_cast<float>(-x), static_cast<float>(y), 0.0F)};
}

// 内参拿的是 2026-10-10 实标的那一组（fx/fy 1806.7/1804.4，cx/cy 743.4/526.1）。
// 用真值而不是整数编一个：既能顺带验证 64 位内参读进来不会被截断，
// 也让"像素 -> 米"的换算量级是真实的。
constexpr double kFx = 1806.7;
constexpr double kFy = 1804.4;
constexpr double kCx = 743.4;
constexpr double kCy = 526.1;

CameraParameters makeCamera() {
    CameraParameters c;
    c.calibrated = true;
    c.image_width = 1440;
    c.image_height = 1080;
    c.camera_matrix = (cv::Mat_<double>(3, 3) << kFx, 0.0, kCx, 0.0, kFy, kCy, 0.0,
                       0.0, 1.0);
    c.distortion_coefficients = cv::Mat::zeros(1, 5, CV_64F);
    c.rotation_camera_to_gimbal = cv::Mat::eye(3, 3, CV_64F);
    c.translation_camera_to_gimbal = cv::Mat::zeros(3, 1, CV_64F);
    c.mean_reprojection_error_px = 0.1;
    return c;
}

// 把给定位姿投成像素角点。物体系与相机系之间的旋转是 R（3x3），
// 平移是 t（3x1，单位 m），即 p_cam = R * p_obj + t。
std::array<cv::Point2f, 4> look(const std::vector<cv::Point3f>& object,
                                const cv::Mat& rvec, const cv::Mat& tvec) {
    const cv::Mat K = (cv::Mat_<double>(3, 3) << kFx, 0.0, kCx, 0.0, kFy, kCy, 0.0,
                       0.0, 1.0);
    const cv::Mat D = cv::Mat::zeros(1, 5, CV_64F);
    std::vector<cv::Point2f> image;
    cv::projectPoints(object, rvec, tvec, K, D, image);
    std::array<cv::Point2f, 4> corners{};
    for (int i = 0; i < 4; ++i) {
        corners[static_cast<std::size_t>(i)] = image[static_cast<std::size_t>(i)];
    }
    return corners;
}

cv::Mat tvecOf(double x, double y, double z) {
    return (cv::Mat_<double>(3, 1) << x, y, z);
}

// 绕 y（竖直）轴转 phi 弧度。用来造"板没正对相机"的情形。
cv::Mat yawRotation(double phi) {
    return (cv::Mat_<double>(3, 3) << std::cos(phi), 0.0, std::sin(phi), 0.0, 1.0,
            0.0, -std::sin(phi), 0.0, std::cos(phi));
}

double dist(const cv::Vec3d& a, const cv::Vec3d& b) {
    const double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

ArmorDetection detectionWith(const std::array<cv::Point2f, 4>& corners,
                             ArmorSize size = ArmorSize::Small, int id = 5,
                             float confidence = 0.9F) {
    ArmorDetection d;
    d.corners = corners;
    d.size = size;
    d.target_id = id;
    d.confidence = confidence;
    return d;
}

}  // namespace

int main() {
    // =======================================================================
    // 1. 正对相机的板：位置要还原到毫米级，重投影误差要接近 0
    // =======================================================================
    {
        const CameraParameters cam = makeCamera();
        const cv::Mat zero_rvec = cv::Mat::zeros(3, 1, CV_64F);
        const cv::Mat pose = tvecOf(0.1, -0.05, 2.0);

        const ArmorDetection d = detectionWith(
            look(platePoints(0.135), zero_rvec, pose));
        const std::vector<ArmorPose> poses = armor_solve({d}, cam, GimbalState{});

        assert(poses.size() == 1);
        // 板心在相机系里的位置。x 右为正、y 下为正、z 前为正。
        assert(std::fabs(poses[0].position_camera_m[0] - 0.1) < 1e-3);
        assert(std::fabs(poses[0].position_camera_m[1] + 0.05) < 1e-3);
        assert(std::fabs(poses[0].position_camera_m[2] - 2.0) < 1e-3);
        // 自己投的图再解回来，重投影误差只该剩浮点噪声
        assert(poses[0].reprojection_error < 1e-3);

        // 检测结果必须原样带出来，下游要靠 target_id 决定开不开火
        assert(poses[0].detection.target_id == 5);
        assert(poses[0].detection.confidence == 0.9F);
        assert(poses[0].detection.size == ArmorSize::Small);

        // 板正对相机 => 法向的偏航角是 0。
        //
        // ↓ 这一条是**唯一**能抓住"平面位姿二义性选错了"的断言：
        //    矩形板有镜像解，解错的话位置一模一样、重投影误差一样是 0，
        //    只有法向会翻成 ±π。位置那几条断言全绿也发现不了。
        assert(std::fabs(poses[0].armor_yaw) < 1e-3);

        // 外参是单位阵 + 零平移 => 云台系坐标必须等于相机系坐标
        assert(dist(poses[0].position_gimbal_m, poses[0].position_camera_m) < 1e-9);
    }

    // =======================================================================
    // 2. 外参那一行真的被用上了
    // =======================================================================
    {
        CameraParameters cam = makeCamera();
        // 一个"相机绕竖直轴转了 90° 再装到云台上"的合法旋转（正交、det = +1）
        cam.rotation_camera_to_gimbal = yawRotation(CV_PI / 2.0);
        cam.translation_camera_to_gimbal = tvecOf(0.06711, 0.0, 0.0515);
        assert(cam.valid());

        const cv::Mat zero_rvec = cv::Mat::zeros(3, 1, CV_64F);
        const cv::Mat pose = tvecOf(0.0, 0.0, 2.0);
        const ArmorDetection d = detectionWith(
            look(platePoints(0.135), zero_rvec, pose));
        const std::vector<ArmorPose> poses = armor_solve({d}, cam, GimbalState{});
        assert(poses.size() == 1);

        // 手算一遍 p_gimbal = R * p_cam + t，必须逐位对上。
        // 漏乘 R、漏加 t、把 R 转置了 —— 三种写错都会在这里露出来。
        const cv::Vec3d p_cam = poses[0].position_camera_m;
        const cv::Mat expected = cam.rotation_camera_to_gimbal *
                                     (cv::Mat_<double>(3, 1) << p_cam[0], p_cam[1],
                                      p_cam[2]) +
                                 cam.translation_camera_to_gimbal;
        assert(std::fabs(poses[0].position_gimbal_m[0] - expected.at<double>(0)) < 1e-9);
        assert(std::fabs(poses[0].position_gimbal_m[1] - expected.at<double>(1)) < 1e-9);
        assert(std::fabs(poses[0].position_gimbal_m[2] - expected.at<double>(2)) < 1e-9);

        // 而且它确实和相机系坐标不同 —— 否则上面那三条在"忘了变换"时也会过
        assert(dist(poses[0].position_gimbal_m, poses[0].position_camera_m) > 0.5);
    }

    // =======================================================================
    // 3. 三个方向的符号：板往哪挪，解出来的坐标就得往哪挪
    // =======================================================================
    // 这一组不看绝对值，只看**单调方向**。物体系/相机系的轴一旦指反
    // （比如把 y 当成向上），绝对值的断言可能因为对称性侥幸通过，
    // 但方向一定会翻。
    {
        const CameraParameters cam = makeCamera();
        const cv::Mat zero_rvec = cv::Mat::zeros(3, 1, CV_64F);

        const double base[3] = {0.0, 0.0, 2.0};
        cv::Vec3d solved[3];
        for (int axis = 0; axis < 3; ++axis) {
            const double moved[3] = {base[0], base[1], base[2]};
            double shifted = moved[axis] + 0.2;
            const cv::Mat pose = tvecOf(axis == 0 ? shifted : moved[0],
                                        axis == 1 ? shifted : moved[1],
                                        axis == 2 ? shifted : moved[2]);
            const ArmorDetection d = detectionWith(
                look(platePoints(0.135), zero_rvec, pose));
            const std::vector<ArmorPose> poses = armor_solve({d}, cam, GimbalState{});
            assert(poses.size() == 1);
            // 挪 +0.2 m 之后，对应分量就该 +0.2
            assert(std::fabs(poses[0].position_camera_m[static_cast<std::size_t>(axis)] -
                             (base[axis] + 0.2)) < 1e-3);
            solved[axis] = poses[0].position_camera_m;
        }
        // 三个分量互不串扰：动 x 不该把 y/z 带跑
        assert(std::fabs(solved[0][1]) < 1e-3 && std::fabs(solved[0][2] - 2.0) < 1e-3);
        assert(std::fabs(solved[1][0]) < 1e-3 && std::fabs(solved[1][2] - 2.0) < 1e-3);
    }

    // =======================================================================
    // 4. 大小分类必须真的参与解算
    // =======================================================================
    // 同一张"230 mm 的板"的像素，按 Large 解出来距离是对的；
    // 按 Small 解就会把距离按错误比例缩掉一截 —— 而且**不报错**。
    // 这就是"大板当小板看，距离凭空少一半"那类事故的来源。
    {
        const CameraParameters cam = makeCamera();
        const cv::Mat zero_rvec = cv::Mat::zeros(3, 1, CV_64F);
        const cv::Mat pose = tvecOf(0.0, 0.0, 3.0);
        const std::array<cv::Point2f, 4> pixels =
            look(platePoints(0.230), zero_rvec, pose);

        const std::vector<ArmorPose> as_large =
            armor_solve({detectionWith(pixels, ArmorSize::Large)}, cam, GimbalState{});
        const std::vector<ArmorPose> as_small =
            armor_solve({detectionWith(pixels, ArmorSize::Small)}, cam, GimbalState{});
        assert(as_large.size() == 1 && as_small.size() == 1);

        assert(std::fabs(as_large[0].position_camera_m[2] - 3.0) < 1e-3);
        // 用错尺寸时距离明显不同 —— 差多少不重要，重要的是"确实不同"
        assert(std::fabs(as_small[0].position_camera_m[2] - 3.0) > 0.1);
    }

    // =======================================================================
    // 5. 板没正对相机时，偏航角要跟着转
    // =======================================================================
    {
        const CameraParameters cam = makeCamera();
        const double phi = 30.0 * CV_PI / 180.0;
        cv::Mat rvec;
        cv::Rodrigues(yawRotation(phi), rvec);
        const cv::Mat pose = tvecOf(0.0, 0.0, 1.5);

        const ArmorDetection d = detectionWith(
            look(platePoints(0.135), rvec, pose));
        const std::vector<ArmorPose> poses = armor_solve({d}, cam, GimbalState{});
        assert(poses.size() == 1);
        assert(poses[0].reprojection_error < 1e-3);
        assert(std::fabs(poses[0].armor_yaw - phi) < 1e-3);
    }

    // =======================================================================
    // 6. 失败的输入必须返回空，不能塞一个看着正常的假解
    // =======================================================================
    {
        const CameraParameters cam = makeCamera();
        const cv::Mat zero_rvec = cv::Mat::zeros(3, 1, CV_64F);
        const ArmorDetection good = detectionWith(
            look(platePoints(0.135), zero_rvec, tvecOf(0, 0, 2)));

        // 没有输入 -> 没有输出
        assert(armor_solve({}, cam, GimbalState{}).empty());

        // 没标定 -> 返回空。拿占位参数硬算出来的"距离"看起来正常，
        // 只是全错 —— 那比返回空难查得多。
        CameraParameters uncalibrated;
        assert(!uncalibrated.valid());
        assert(armor_solve({good}, uncalibrated, GimbalState{}).empty());

        // 角点退化（四个点挤成一点）-> 必须跳过。
        // 注意：solvePnP 在这里**不会失败**，它会返回一个把板压成一点的解，
        // 重投影误差 0、位置看着正常。所以这道闸只能靠几何面积，
        // 靠 solvePnP 的返回值或重投影误差都拦不住。
        ArmorDetection degenerate = good;
        degenerate.corners = {{{100.0F, 100.0F}, {100.0F, 100.0F}, {100.0F, 100.0F},
                               {100.0F, 100.0F}}};
        assert(armor_solve({degenerate}, cam, GimbalState{}).empty());
    }

    // =======================================================================
    // 7. 异常观测必须被拒绝（任务书 §3.5 明写：「对 PnP 失败、深度非法或
    //    误差异常的观测进行拒绝」）
    // =======================================================================
    // 这一组每一条都是**别的闸拦不住**的：面积正常、深度为正、solvePnP 返回
    // 成功、重投影误差还接近 0 —— 只有对应的那道闸能挡。阈值数字见
    // StudentTasks.cpp 顶部的注释（由 /tmp 的探针实测得出）。
    {
        const CameraParameters cam = makeCamera();
        const cv::Mat zero_rvec = cv::Mat::zeros(3, 1, CV_64F);
        const ArmorDetection good = detectionWith(
            look(platePoints(0.135), zero_rvec, tvecOf(0, 0, 2)));

        // 对照：同样的像素、没动过的角点，必须留下。
        // 没有这一条，下面几个 empty() 在"函数永远返回空"时也会过。
        assert(armor_solve({good}, cam, GimbalState{}).size() == 1);

        // (a) 误差异常：把一个角点横向拉开 60 px。
        //     实测面积 7814（正常）、深度 1.37 m（正常）、solvePnP 成功，
        //     但重投影误差从 0 涨到 14.4 px —— 只有误差这道闸看得见。
        ArmorDetection skewed = good;
        skewed.corners[3].x -= 60.0F;
        assert(armor_solve({skewed}, cam, GimbalState{}).empty());

        // (b) 深度非法（太远）：把四个角点朝主点缩到 2%。
        //     实测面积 2.5（> 1，过退化闸）、重投影误差 0.000 —— 一道闸都不响，
        //     解出来的距离却是 100 m。RM 场地长 28 m，这不可能是真目标。
        ArmorDetection tiny = good;
        for (cv::Point2f& p : tiny.corners) {
            p.x = static_cast<float>(kCx) + (p.x - static_cast<float>(kCx)) * 0.02F;
            p.y = static_cast<float>(kCy) + (p.y - static_cast<float>(kCy)) * 0.02F;
        }
        assert(armor_solve({tiny}, cam, GimbalState{}).empty());

        // (c) 深度非法（太近）：放大 100 倍，解出来 0.02 m —— 板已经贴在镜头里，
        //     同样面积正常、误差 0。
        ArmorDetection huge = good;
        for (cv::Point2f& p : huge.corners) {
            p.x = static_cast<float>(kCx) + (p.x - static_cast<float>(kCx)) * 100.0F;
            p.y = static_cast<float>(kCy) + (p.y - static_cast<float>(kCy)) * 100.0F;
        }
        assert(armor_solve({huge}, cam, GimbalState{}).empty());

        // (d) 角点顺序反了 → 有向面积变负。
        //     ArmorDetector.hpp:70 写着「顺序正确时为正」，所以负面积本身就是
        //     输入非法的信号。此时 PnP 会给出一个**位置正确、法向翻转**的镜像解，
        //     重投影误差仍然是 0.000 —— 靠误差和深度都拦不住，只有符号能拦。
        ArmorDetection reversed = good;
        std::reverse(reversed.corners.begin(), reversed.corners.end());
        assert(armor_solve({reversed}, cam, GimbalState{}).empty());
    }

    return 0;
}
