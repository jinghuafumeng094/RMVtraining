// 网页画面上的检测框与 PnP 标注。判据有六条：
// 1. 没有检测时不改任何一个像素，否则静止画面也会被当成有目标；
// 2. 有效目标的框是绿色，框内中心仍是原像素，说明只画了边没涂满；
// 3. 编号 0 的板换黄色，和有效目标分得开；
// 4. 解算通过的板在板心画出物体系三轴：x 红朝右、y 绿朝下、原点白点；
// 5. 解算通过才有 rvec/tvec 两行字，颜色 (255,255,0)，位置在框上方两行；
// 6. 被 armor_solve 的闸挡下的板标红字 PnP x；没标定时一个 PnP 标记都不画。

#include "AppSupport.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cassert>
#include <cstdio>
#include <vector>

namespace {

const cv::Scalar kBlack(0, 0, 0);

bool isGreen(const cv::Vec3b& pixel) {
    return pixel[0] < 90 && pixel[1] > 180 && pixel[2] < 90;
}

bool isYellow(const cv::Vec3b& pixel) {
    return pixel[0] < 90 && pixel[1] > 180 && pixel[2] > 180;
}

bool isBlack(const cv::Vec3b& pixel) {
    return pixel[0] < 20 && pixel[1] < 20 && pixel[2] < 20;
}

bool isRed(const cv::Vec3b& pixel) {
    return pixel[0] < 90 && pixel[1] < 90 && pixel[2] > 180;
}

// PnP 文字用的 (255,255,0)：蓝+绿=青。白色也满足「亮」，但红通道满，
// 被 p[2] < 90 挡掉，不会把白点和黑色背景算进来。
bool isPnpText(const cv::Vec3b& pixel) {
    return pixel[0] > 180 && pixel[1] > 180 && pixel[2] < 90;
}

bool isWhite(const cv::Vec3b& pixel) {
    return pixel[0] > 200 && pixel[1] > 200 && pixel[2] > 200;
}

int countColored(const cv::Mat& image, const cv::Rect& region,
                 bool (*predicate)(const cv::Vec3b&)) {
    const cv::Rect clipped = region & cv::Rect(0, 0, image.cols, image.rows);
    int count = 0;
    for (int y = clipped.y; y < clipped.y + clipped.height; ++y) {
        for (int x = clipped.x; x < clipped.x + clipped.width; ++x) {
            if (predicate(image.at<cv::Vec3b>(y, x))) {
                ++count;
            }
        }
    }
    return count;
}

ArmorDetection makeDetection(int target_id, ArmorSize size, float confidence) {
    ArmorDetection detection;
    detection.corners = {cv::Point2f(60.0F, 60.0F),
                         cv::Point2f(160.0F, 60.0F),
                         cv::Point2f(160.0F, 120.0F),
                         cv::Point2f(60.0F, 120.0F)};
    detection.target_id = target_id;
    detection.size = size;
    detection.confidence = confidence;
    return detection;
}

// 400x400 的合成相机，光心在正中，fx=800 px，无畸变。
// 板心放在相机系 (0,0,1) 时正好投到 (200,200)，轴长 7 cm 在图上就是 56 px。
CameraParameters makeCamera() {
    CameraParameters camera;
    camera.calibrated = true;
    camera.image_width = 400;
    camera.image_height = 400;
    camera.camera_matrix =
        (cv::Mat_<double>(3, 3) << 800.0, 0.0, 200.0,
                                   0.0, 800.0, 200.0,
                                   0.0, 0.0, 1.0);
    camera.distortion_coefficients = cv::Mat::zeros(1, 5, CV_64F);
    camera.rotation_camera_to_gimbal = cv::Mat::eye(3, 3, CV_64F);
    camera.translation_camera_to_gimbal = cv::Mat::zeros(3, 1, CV_64F);
    camera.mean_reprojection_error_px = 0.3;
    return camera;
}

ArmorPose makePose(const ArmorDetection& detection, const cv::Vec3d& tvec) {
    ArmorPose pose;
    pose.detection = detection;
    pose.position_camera_m = tvec;
    pose.rvec = cv::Vec3d(0.0, 0.0, 0.0);
    return pose;
}

}  // namespace

int main() {
    // 1. 空列表：一个像素都不许动。
    cv::Mat empty_case(200, 200, CV_8UC3, kBlack);
    annotateDetections(empty_case, {});
    assert(cv::countNonZero(empty_case.reshape(1)) == 0);

    // 2. 有效目标：顶边中点是绿色，框内中心保持原样。
    cv::Mat identified(200, 200, CV_8UC3, kBlack);
    annotateDetections(identified, {makeDetection(3, ArmorSize::Small, 0.82F)});
    assert(isGreen(identified.at<cv::Vec3b>(60, 110)));
    assert(isBlack(identified.at<cv::Vec3b>(95, 110)));
    assert(isBlack(identified.at<cv::Vec3b>(150, 110)));

    // 3. 编号 0：框换黄色，一眼看出"检出但没读出数字"。
    cv::Mat unidentified(200, 200, CV_8UC3, kBlack);
    annotateDetections(unidentified, {makeDetection(0, ArmorSize::Large, 0.55F)});
    assert(isYellow(unidentified.at<cv::Vec3b>(60, 110)));
    assert(isBlack(unidentified.at<cv::Vec3b>(95, 110)));
    // 没传标定参数就等于没算 PnP，整幅不许出现任何 PnP 标记，
    // 否则每块板都会被误标成"被闸挡下"。
    assert(countColored(unidentified, cv::Rect(0, 0, 200, 200), isRed) == 0);

    // 4. 两块板各画各的，颜色互不串。
    cv::Mat two_plates(300, 300, CV_8UC3, kBlack);
    ArmorDetection second = makeDetection(5, ArmorSize::Small, 0.71F);
    second.corners = {cv::Point2f(60.0F, 200.0F),
                      cv::Point2f(160.0F, 200.0F),
                      cv::Point2f(160.0F, 260.0F),
                      cv::Point2f(60.0F, 260.0F)};
    annotateDetections(two_plates, {makeDetection(3, ArmorSize::Small, 0.82F), second});
    assert(isGreen(two_plates.at<cv::Vec3b>(60, 110)));
    assert(isGreen(two_plates.at<cv::Vec3b>(200, 110)));

    // 5. 解算通过的板：板心处画出物体系三轴，并在框上方写 rvec / tvec 两行。
    //    板心在相机系 (0,0,1)，投到 (200,200)；轴长 7 cm、fx=800 ⇒ 图上 56 px。
    cv::Mat solved(400, 400, CV_8UC3, kBlack);
    ArmorDetection solved_plate = makeDetection(3, ArmorSize::Small, 0.82F);
    solved_plate.corners = {cv::Point2f(60.0F, 120.0F),
                            cv::Point2f(160.0F, 120.0F),
                            cv::Point2f(160.0F, 180.0F),
                            cv::Point2f(60.0F, 180.0F)};
    annotateDetections(solved, {solved_plate},
                       {makePose(solved_plate, cv::Vec3d(0.0, 0.0, 1.0))},
                       makeCamera());
    // 原点白点画在板心上。三条轴从原点出发，正中心的像素会被轴线盖住，
    // 所以找的是白点的余下部分，不是 (200,200) 那一个像素。
    assert(countColored(solved, cv::Rect(196, 196, 10, 10), isWhite) > 0);
    // x 轴朝右是红，取轴上避开箭头与字母的一段。
    assert(countColored(solved, cv::Rect(210, 195, 30, 10), isRed) > 10);
    // y 轴朝下是绿，同理。
    assert(countColored(solved, cv::Rect(195, 210, 10, 30), isGreen) > 10);
    // rvec 一行在角点上方 62 px，tvec 在 34 px；角点 (60,120) ⇒ 基线 58 与 86。
    assert(countColored(solved, cv::Rect(60, 36, 300, 24), isPnpText) > 10);
    assert(countColored(solved, cv::Rect(60, 64, 300, 24), isPnpText) > 10);

    // 6. 没有对应位姿的板：标红字 PnP x，且不画轴。
    cv::Mat rejected(400, 400, CV_8UC3, kBlack);
    ArmorDetection rejected_plate = makeDetection(6, ArmorSize::Small, 0.40F);
    rejected_plate.corners = {cv::Point2f(60.0F, 300.0F),
                              cv::Point2f(160.0F, 300.0F),
                              cv::Point2f(160.0F, 360.0F),
                              cv::Point2f(60.0F, 360.0F)};
    annotateDetections(rejected, {rejected_plate},
                       {makePose(solved_plate, cv::Vec3d(0.0, 0.0, 1.0))},
                       makeCamera());
    // PnP x 的基线在角点上方 34 px ⇒ 300-34 = 266，字高约 16 px。
    assert(countColored(rejected, cv::Rect(60, 248, 100, 20), isRed) > 5);
    // 这条没有位姿，板心位置不该出现白点或轴。
    assert(isBlack(rejected.at<cv::Vec3b>(200, 200)));

    std::puts("frame_overlay ok");
    return 0;
}
