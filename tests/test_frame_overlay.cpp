// 网页画面上的检测框。判据有三条：
// 1. 没有检测时不改任何一个像素，否则静止画面也会被当成有目标；
// 2. 有效目标的框是绿色，框内中心仍是原像素，说明只画了边没涂满；
// 3. 编号 0 的板换黄色，和有效目标分得开。

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

    std::puts("frame_overlay ok");
    return 0;
}
