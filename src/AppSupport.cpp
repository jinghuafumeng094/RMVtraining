#include "AppSupport.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

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

namespace {

// 物体系三条轴的画法，数值与仓库外的 armor_live 逐项一致，两块屏对着看能重合：
// 轴长 7 cm，比板高 5.7 cm、小板半宽 6.75 cm 都长一点，箭头落在板外不被板面挡住。
constexpr double kAxisLengthM = 0.07;
constexpr double kAxisFontScale = 0.7;
constexpr double kPnpTextFontScale = 0.7;

// 在板心画物体系的 x / y / z。颜色跟报告里《三个坐标系的方向与原点关系》一致，
// x 红、y 绿、z 蓝。z 是板背法向，板正对相机时几乎沿光轴，投影出来很短，属正常。
// 不用 cv::drawFrameAxes：它配色是 x 蓝 / y 绿 / z 红，跟本项目反着，也不标轴名。
void drawObjectAxes(cv::Mat& image, const CameraParameters& camera,
                    const cv::Vec3d& rvec, const cv::Vec3d& tvec) {
    const float length = static_cast<float>(kAxisLengthM);
    const std::vector<cv::Point3f> axes = {{0.0F, 0.0F, 0.0F},
                                           {length, 0.0F, 0.0F},
                                           {0.0F, length, 0.0F},
                                           {0.0F, 0.0F, length}};
    std::vector<cv::Point2f> projected;
    cv::projectPoints(axes, rvec, tvec, camera.camera_matrix,
                      camera.distortion_coefficients, projected);

    const cv::Point2f origin = projected[0];
    const cv::Scalar colors[3] = {cv::Scalar(0, 0, 255),   // x 红
                                  cv::Scalar(0, 255, 0),   // y 绿
                                  cv::Scalar(255, 0, 0)};  // z 蓝
    const char* names[3] = {"x", "y", "z"};

    // 原点就是板心，点一个白点标出来。
    cv::circle(image, origin, 3, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);
    for (int i = 0; i < 3; ++i) {
        const cv::Point2f tip = projected[i + 1];
        cv::arrowedLine(image, origin, tip, colors[i], 2, cv::LINE_AA, 0, 0.25);
        cv::putText(image, names[i], tip + cv::Point2f(6.0F, -6.0F),
                    cv::FONT_HERSHEY_SIMPLEX, kAxisFontScale, colors[i], 2,
                    cv::LINE_AA);
    }
}

// 按左上角点在同一条检测上认位姿。armor_solve 复制进 ArmorPose 的就是同一组
// float 角点，除此之外没有共同的编号可用。
const ArmorPose* findPose(const std::vector<ArmorPose>& observations,
                          const cv::Point2f& anchor) {
    for (const ArmorPose& pose : observations) {
        if (pose.detection.corners[0] == anchor) {
            return &pose;
        }
    }
    return nullptr;
}

}  // namespace

void annotateDetections(cv::Mat& image,
                        const std::vector<ArmorDetection>& detections,
                        const std::vector<ArmorPose>& observations,
                        const CameraParameters& camera) {
    // 尺寸按 1440x1080 的画面定：框 3 px、字 1.0 倍。网页那条流是缩过的 JPEG，
    // 线再细就看不清了。
    constexpr int kBoxThickness = 3;
    constexpr int kFontThickness = 2;
    constexpr double kFontScale = 1.0;
    constexpr int kLabelLiftPx = 8;

    if (image.empty() || image.type() != CV_8UC3) {
        return;
    }

    // 没标定就没有位姿，整幅不画 PnP 标记：否则每块板都会被标成被闸挡下，
    // 把"没标定"伪装成"观测被拒"。
    const bool pnp_enabled = camera.valid();

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

        if (!pnp_enabled) {
            continue;
        }

        // 在这批观测里找这条检测对应的位姿。armor_solve 的闸会丢掉一部分观测，
        // 被丢掉的按角点找不到位姿，标成 PnP x，"观测被拒"就成了画面上看得见的
        // 事，不是只写在文档里。
        const cv::Point2f& plate_anchor = detection.corners[0];
        const cv::Point text_anchor(static_cast<int>(plate_anchor.x),
                                    static_cast<int>(plate_anchor.y));
        const ArmorPose* pose = findPose(observations, plate_anchor);
        if (pose == nullptr) {
            cv::putText(image, "PnP x", cv::Point(text_anchor.x, text_anchor.y - 34),
                        cv::FONT_HERSHEY_SIMPLEX, kPnpTextFontScale,
                        cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            continue;
        }

        // 过了三道闸的位姿才有意义，坐标轴也只画在这一条上。
        drawObjectAxes(image, camera, pose->rvec, pose->position_camera_m);

        // PnP 的原始两个输出，一行一个。y 轴朝下，想让它显示在上面，y 偏移要更小，
        // 所以 -62 在上、-34 在下。
        cv::putText(image,
                    cv::format("rvec=(%+.3f, %+.3f, %+.3f)", pose->rvec[0],
                               pose->rvec[1], pose->rvec[2]),
                    cv::Point(text_anchor.x, text_anchor.y - 62),
                    cv::FONT_HERSHEY_SIMPLEX, kPnpTextFontScale,
                    cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
        cv::putText(image,
                    cv::format("tvec=(%+.3f, %+.3f, %+.3f)",
                               pose->position_camera_m[0],
                               pose->position_camera_m[1],
                               pose->position_camera_m[2]),
                    cv::Point(text_anchor.x, text_anchor.y - 34),
                    cv::FONT_HERSHEY_SIMPLEX, kPnpTextFontScale,
                    cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
    }
}
