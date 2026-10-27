#include "StudentTasks.hpp"

#include "ArmorDetector.hpp"
#include "ArmorEkf.hpp"
#include "HikCamera.hpp"
#include "TargetSelector.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <vector>

namespace {

// 装甲板物理尺寸(m)。来源：仓库根目录 装甲板尺寸.txt，写成 135 x 57 / 230 x 57 mm。
// size 由两灯条间距比纯几何判出，和读到的数字无关，所以 7 配小板是正常输出。
// 但 size 判错时距离会按错误的比例解出来，而且不报错，只是数字凭空差一截。
constexpr double kSmallWidthM = 0.135;
constexpr double kLargeWidthM = 0.230;
constexpr double kPlateHeightM = 0.057;

// 观测拒绝阈值。任务书 §3.5 点名要拒绝三类：PnP 失败、深度非法、误差异常。
// 这三个数是用合成数据探针实测定的，探针在仓库外，不是拍脑袋：
//   正常角点（自己投自己解）重投影误差 = 0.000 px
//   单个角点被拉开 10 px -> 2.2 px；30 px -> 7.0 px；60 px -> 14.4 px
// 所以 5 px 大致相当于某个角点错了 20 px 量级，是"明显不对"而不是"抖了一下"。
// 深度上下限是物理约束：近于 5 cm 板已贴在镜头上，远于 30 m 超过场地长度
// （场地 28 m），两种都不可能是真目标，只会是把角点缩爆或拉爆后解出来的假位姿。
//
// 2026-10-11 拿真实帧补量了角点抖动，探针 corner_jitter 在仓库外：
// 三帧静止的板，距离 1.08 m，重投影误差 0.70~0.79 px、深度 1.083~1.088 m，
// 同一个角点在任意两帧之间最多动 1.41 px。5 px 的闸门有 3.5 倍以上余量。
// 这只是最容易的那种目标：近距离、正对、曝光正常，而且这三帧的角点落在
// 整数像素上，亚像素抖动量不出来。远处的小目标抖得更厉害，别据此收紧闸门。
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

// 绕相机 x（右）轴的旋转，正角把光轴往上抬，即 +z 转向 -y。
// 只服务安装俯仰偏角这一处，方向约定见 armor_solve 里对 mount_pitch_deg 的说明。
cv::Mat pitchRotationRadians(double radians) {
    const double c = std::cos(radians);
    const double s = std::sin(radians);
    return (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c);
}

}  // namespace

// 任务 1：相机取流。
//
// 真正碰 MVS SDK 的地方全在 HikCamera 里，这里只管一件事：进程内只开一次。
// 函数局部 static 的初始化从 C++11 起是线程安全的，所以第一次调用时打开相机，
// 之后每帧复用同一个句柄；失败一律返回 false，不往上抛也不终止进程。
bool get_pic(cv::Mat& pic) {
    static HikCamera camera;
    return camera.grab(pic);
}

std::vector<ArmorDetection> armor_detect(const cv::Mat& image, TeamColor enemy_color) {
    // 整条检测管线都在 ArmorDetector 里：颜色掩膜 -> 灯条提取 -> 配对 -> 角点
    // -> 大小分类 -> 去重 -> 数字分类。这里只做转发，
    // 让本文件负责"把各部分接起来"，算法本身留在 ArmorDetector，能脱离相机单独测。
    return armor::detectArmors(image, enemy_color);
}

// 任务 3：装甲板位姿解算。
//
// 用已标定的内参 + 装甲板已知物理尺寸做 PnP，得到板心在相机系中的位置，
// 再用外参把它刚体变换到云台系。
//
// 云台角度(gimbal)在这里用不上：外参 R/t 已经把相机系映射到云台系，
// 而云台系本身就是跟着云台转的那个系，当前角度不是这条变换的输入。
std::vector<ArmorPose> armor_solve(const std::vector<ArmorDetection>& detections,
                                   const CameraParameters& camera,
                                   const GimbalState&) {
    std::vector<ArmorPose> poses;

    // 没标定就不许解算。返回空，而不是拿占位参数硬算，
    // 后者给出的距离看着正常、实际全错，比"没有结果"难查得多。
    if (!camera.valid()) {
        return poses;
    }

    const cv::Mat& camera_matrix = camera.camera_matrix;
    const cv::Mat& distortion = camera.distortion_coefficients;
    const cv::Mat& R_camera_to_gimbal = camera.rotation_camera_to_gimbal;
    const cv::Mat& t_camera_to_gimbal = camera.translation_camera_to_gimbal;

    // 安装俯仰偏角，夹在 PnP 和外参之间用。
    // 光轴向下偏 delta 时，板在相机系里的 y 会比光轴水平时小 距离 × tan(delta)，
    // 于是整块板被读得偏高；按下面的顺序解出来，2 m 处能偏 26 cm。
    // 补偿就是把测到的方向先转回"光轴水平"的名义相机系，再交给命名用的外参。
    // 角度取负号：pitchRotationRadians 正角是抬，而这里要抵消的是向下偏。
    // 只转方向不挪平移：相机是绕光心转的，光心在云台系里的位置不变。
    // 若实际是绕支架螺钉转、光心跟着平移，这段还差一个二三毫米量级的修正。
    const cv::Mat R_measured_to_nominal =
        pitchRotationRadians(-camera.mount_pitch_deg * CV_PI / 180.0);
    const cv::Mat R_effective = R_camera_to_gimbal * R_measured_to_nominal;

    poses.reserve(detections.size());
    for (const ArmorDetection& detection : detections) {
        // 角点合法性闸，两种坏输入一起挡：
        //   面积接近 0（四点重合或共线）：solvePnP 不会失败，会给出一个把板
        //   压成一点的解，重投影误差还是 0，是静默错误。
        //   面积为负：说明角点顺序反了。ArmorDetector.hpp:70 写着「顺序正确时
        //   为正」，所以负号本身就是输入非法的信号。这种输入 PnP 会给出一个
        //   位置正确、法向翻转的镜像解，重投影误差仍然是 0，靠误差和深度都
        //   拦不住，只有符号能拦。
        // 注意这里不取绝对值：取绝对值就等于把"顺序反了"当成正常输入放行。
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
        // 100 m，角点摆成蝶形时给过 z = 5e11。两种情况 solvePnP 都返回成功，
        // 重投影误差还可能是 0。所以只能自己按物理范围兜一道。
        const double depth = tvec[2];
        if (!std::isfinite(depth) || depth < kMinDepthM || depth > kMaxDepthM) {
            continue;
        }

        // 单次观测的重投影误差：把解出来的位姿再投回图像，和实测角点比。
        // 这是这条观测的自洽度，不是标定精度。角点被拉开、灯条配错时它会涨上去，
        // 而面积和深度都还正常。注意它看不见约定错误，物体系轴指反、大小分类
        // 用错，重投影误差照样接近 0，那类问题只能靠 test_armor_solve 的断言钉住。
        std::vector<cv::Point2f> reprojected;
        cv::projectPoints(object, rvec, tvec, camera_matrix, distortion,
                          reprojected);
        double error_px = 0.0;
        for (std::size_t i = 0; i < image.size(); ++i) {
            error_px += cv::norm(reprojected[i] - image[i]);
        }
        error_px /= static_cast<double>(image.size());

        // 误差异常就丢（任务书 §3.5）。
        if (!std::isfinite(error_px) || error_px > kMaxReprojectionErrorPx) {
            continue;
        }

        ArmorPose pose;
        pose.detection = detection;
        // solvePnP 的 tvec 就是板心在相机系中的位置(m)
        pose.position_camera_m = tvec;
        pose.reprojection_error = error_px;

        // 板心到画面中心的像素距离，多目标筛选按它排序。
        // 取实测四角点的均值，不再投一次物体系原点：物体系四角点关于原点对称，
        // 两者只差透视下的二阶小量，而实测角点不依赖 PnP，含义也更直接。
        cv::Point2f center_px(0.0F, 0.0F);
        for (const cv::Point2f& corner : detection.corners) {
            center_px += corner;
        }
        center_px *= 0.25F;
        pose.center_offset_px =
            std::hypot(center_px.x - 0.5 * camera.image_width,
                       center_px.y - 0.5 * camera.image_height);

        // 相机系 -> 云台系：一条刚体变换，全函数就这一处用到外参。
        // 这里的 R 不是单位阵：相机系是 右 / 下 / 前，云台系是 前 / 左 / 上，
        // 两者差一个 90 度置换，见 config/camera.yaml 顶部。若把 R 当单位阵，
        // 位置会照常解出来、也不报错，只是每个分量都换了含义。
        const cv::Mat p_camera =
            (cv::Mat_<double>(3, 1) << tvec[0], tvec[1], tvec[2]);
        pose.position_gimbal_m = toVec3d(R_effective * p_camera +
                                         t_camera_to_gimbal);

        // armor_yaw：装甲板法向在云台系中的偏航角，右为正。
        // 物体系 +z（指向板背）在相机系里就是旋转矩阵的第三列，
        // 换到云台系（前 / 左 / 上）后取 atan2(-y, x)。板正对相机时 = 0。
        // 与 main.cpp 的 pnp_yaw_rad、ArmorEkf 的 rel_yaw 同一套口径。
        cv::Mat R_object_to_camera;
        cv::Rodrigues(rvec, R_object_to_camera);
        const cv::Mat normal_gimbal = R_effective * R_object_to_camera.col(2);
        pose.armor_yaw = std::atan2(-normal_gimbal.at<double>(1),
                                    normal_gimbal.at<double>(0));

        poses.push_back(pose);
    }
    return poses;
}

// 任务 4：EKF 跟踪与预测。
//
// 跨帧状态放在函数内的 static 里。main.cpp 的操作线程是唯一调用者，
// 不存在并发问题；ArmorEkf 本身可以独立构造，单测直接测类，不走这个适配器。
//
// 一帧可能出现两辆车。ArmorEkf 只有一套状态和一个四槽关联器，两辆车的板
// 同时灌进去会互相抢槽位，所以在滤波之前先用 TargetSelector 选出一辆车。
PredictionResult ekf_predict(const std::vector<ArmorPose>& observations,
                             const GimbalState& gimbal, double timestamp_seconds) {
    static ArmorEkf filter;
    static armor::TargetSelector selector;
    static double last_timestamp = 0.0;

    const armor::TargetSelection selection =
        selector.select(observations, static_cast<double>(gimbal.yaw), timestamp_seconds,
                        static_cast<double>(gimbal.pitch));

    // 换簇不等于换车：同一辆靶车的两块板拆成两簇也会置 switched，而这时滤波器
    // 可能正处在 Stable。无条件重置会把收敛好的半径与速度一起扔掉，重建时还会
    // 按单板视线假设把车心放偏，第二块板从此落在 max_radius 之外，再也进不来。
    // 解释不了的观测由跳变闸与 NIS 闸挡下，连续 miss 到 reinit_after_misses
    // 就会用这几帧观测就地重建；真换了车也走同一条路。
    if (selection.switched && filter.status() == TrackingState::NoTarget) {
        filter.reset();
    }

    // timestamp_seconds 是 steady_clock 自纪元起的秒数，两帧相减就是帧间隔；
    // 用"现在"减它，得到的是这一帧从采集到此刻消耗掉的检测与解算时间，也就是延迟。
    const double dt = (last_timestamp > 0.0 && std::isfinite(timestamp_seconds))
                          ? std::clamp(timestamp_seconds - last_timestamp, 0.0, 0.5)
                          : 0.0;
    // 时间戳非有限时不要把 NaN 写进 static，否则后续每一帧的 dt 都会被拖成 NaN。
    if (std::isfinite(timestamp_seconds)) {
        last_timestamp = timestamp_seconds;
    }

    const double now_seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count();
    const double latency_s = std::clamp(now_seconds - timestamp_seconds, 0.0, 0.2);

    filter.predict(dt);
    for (int index : selection.observation_indices) {
        filter.update(observations[static_cast<std::size_t>(index)],
                      static_cast<double>(gimbal.yaw),
                      static_cast<double>(gimbal.pitch));
    }
    filter.advanceStateMachine(timestamp_seconds);
    return filter.buildPrediction(gimbal, latency_s);
}
