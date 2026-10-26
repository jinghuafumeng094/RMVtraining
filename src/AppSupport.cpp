#include "AppSupport.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <string>

namespace {

float finiteOrZero(float value) noexcept {
    return std::isfinite(value) ? value : 0.0F;
}

bool finitePrediction(const PredictionResult& prediction) {
    // 同时检查国际单位制数值，以及米转换为毫米后的 float 通信表示。
    const float wire_center_x_mm = prediction.center_x_m * 1000.0F;
    const float wire_center_y_mm = prediction.center_y_m * 1000.0F;
    return prediction.target_id != 0U &&
           (prediction.state == TrackingState::Stable ||
            prediction.state == TrackingState::Unstable) &&
           std::isfinite(prediction.center_x_m) &&
           std::isfinite(prediction.center_y_m) &&
           std::isfinite(wire_center_x_mm) && std::isfinite(wire_center_y_mm) &&
           std::isfinite(prediction.center_velocity_x_m_s) &&
           std::isfinite(prediction.center_velocity_y_m_s) &&
           std::isfinite(prediction.body_yaw) &&
           std::isfinite(prediction.body_yaw_velocity) &&
           std::isfinite(prediction.radius_1_m) &&
           std::isfinite(prediction.radius_2_m) && prediction.radius_1_m >= 0.0F &&
           prediction.radius_2_m >= 0.0F &&
           std::isfinite(prediction.distance_m) && std::isfinite(prediction.target_yaw) &&
           std::isfinite(prediction.target_pitch) &&
           std::isfinite(prediction.target_yaw_velocity) &&
           std::isfinite(prediction.target_pitch_velocity) && prediction.distance_m >= 0.0F;
}

}  // 匿名命名空间

bool isAutoAimStatus(std::uint8_t status) noexcept {
    return status == 0U || status == 5U;
}

TeamColor enemyColorFromStatus(std::uint8_t status) noexcept {
    const bool own_team_is_blue = status == 5U;
    return own_team_is_blue ? TeamColor::Red : TeamColor::Blue;
}

GimbalState gimbalStateFromMessage(const MessData_AutoAim& message) noexcept {
    return {message.yaw, message.pitch, message.roll, message.bias};
}

void setSafeOutput(Translator& translator) noexcept {
    // 安全不变量：无目标、无运动前馈，并且绝不允许开火。
    // 控制器提供的姿态/偏置字段在有限时保留，便于诊断。
    auto& message = translator.message;
    message.head = kFrameHead;
    message.yaw = finiteOrZero(message.yaw);
    message.pitch = finiteOrZero(message.pitch);
    message.roll = finiteOrZero(message.roll);
    message.bias = finiteOrZero(message.bias);
    message.armor_flag = 0U;
    message.latency = 0.0F;
    message.distance = 0.0F;
    message.pitch_offset = 0.0F;
    message.coo_x = 0.0F;
    message.coo_y = 0.0F;
    message.wheel_w = 0.0F;
    message.fire_allowance = 0U;
    message.target_yaw = message.yaw;
    message.target_pitch = message.pitch;
    message.yaw_vel = 0.0F;
    message.pitch_vel = 0.0F;
    message.crc = static_cast<std::uint16_t>(TrackingState::NoTarget);
    message.tail = kFrameTail;
}

bool applyPrediction(Translator& translator, const PredictionResult& prediction) noexcept {
    // 学生结果只要不完整或包含非有限值，就降级为同样的安全帧。
    if (prediction.state == TrackingState::NoTarget || !finitePrediction(prediction)) {
        setSafeOutput(translator);
        return false;
    }

    auto& message = translator.message;
    message.head = kFrameHead;
    message.armor_flag = prediction.target_id;
    message.distance = prediction.distance_m;
    message.pitch_offset = 0.0F;
    // 旧协议的 coo_x/coo_y 字段单位为毫米；内部算法统一使用米。
    message.coo_x = prediction.center_x_m * 1000.0F;
    message.coo_y = prediction.center_y_m * 1000.0F;
    message.wheel_w = 0.0F;
    message.target_yaw = prediction.target_yaw;
    message.target_pitch = prediction.target_pitch;
    message.yaw_vel = prediction.target_yaw_velocity;
    message.pitch_vel = prediction.target_pitch_velocity;
    message.crc = static_cast<std::uint16_t>(prediction.state);
    message.fire_allowance = prediction.state == TrackingState::Stable &&
                                     prediction.fire_allowed
                                 ? 1U
                                 : 0U;
    message.tail = kFrameTail;
    return true;
}

void annotateDetections(cv::Mat& image,
                        const std::vector<ArmorDetection>& detections) {
    // 尺寸按 1440x1080 的画面定：框 3 px、字 1.0 倍。网页那条流是缩过的 JPEG，
    // 线再细就看不清了。
    constexpr int kBoxThickness = 3;
    constexpr int kFontThickness = 2;
    constexpr double kFontScale = 1.0;
    constexpr int kLabelLiftPx = 8;

    if (image.empty() || image.type() != CV_8UC3) {
        return;
    }

    for (const ArmorDetection& detection : detections) {
        // 编号 0 表示板子检出了、但没能读出 1~8 的数字。框照样画，颜色换成黄色，
        // 这样「检出几块」和「其中几块能当目标」在画面上一眼分得开。
        const cv::Scalar color =
            detection.target_id != 0 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);
        for (int i = 0; i < 4; ++i) {
            cv::line(image, detection.corners[i], detection.corners[(i + 1) % 4],
                     color, kBoxThickness, cv::LINE_AA);
        }

        const std::string label =
            std::to_string(detection.target_id) +
            (detection.size == ArmorSize::Large ? " L " : " S ") +
            cv::format("%.2f", static_cast<double>(detection.confidence));

        // 标签挂在左上角点上方。贴边的板会把它顶出画面，所以两个方向都要夹回图内：
        // x 夹到「右边还剩得下这行字」，y 夹到「字顶不越出上边界」。
        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(
            label, cv::FONT_HERSHEY_SIMPLEX, kFontScale, kFontThickness, &baseline);
        const int x = static_cast<int>(detection.corners[0].x);
        const int y = static_cast<int>(detection.corners[0].y) - kLabelLiftPx;
        const cv::Point anchor(std::max(0, std::min(x, image.cols - text_size.width)),
                               std::max(text_size.height, y));
        cv::putText(image, label, anchor, cv::FONT_HERSHEY_SIMPLEX, kFontScale,
                    color, kFontThickness, cv::LINE_AA);
    }
}
