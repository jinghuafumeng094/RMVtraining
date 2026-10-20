#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

enum class TeamColor : std::uint8_t {
    Red = 0,
    Blue = 1,
};

enum class ArmorSize : std::uint8_t {
    Small = 0,
    Large = 1,
};

enum class TrackingState : std::uint16_t {
    // 数值直接对应历史通信字段 crc。
    NoTarget = 0,
    Stable = 1,
    Unstable = 2,
};

struct GimbalState {
    // 控制器快照：角度单位为弧度，prediction_bias_s 单位为秒。
    float yaw{0.0F};
    float pitch{0.0F};
    float roll{0.0F};
    float prediction_bias_s{0.0F};
};

struct CameraParameters {
    // 所有矩阵均为单通道 CV_64F。刚体变换为
    // p_gimbal = R_camera_to_gimbal * p_camera + t_camera_to_gimbal。
    bool calibrated{false};
    int image_width{0};
    int image_height{0};
    cv::Mat camera_matrix;
    cv::Mat distortion_coefficients;
    cv::Mat rotation_camera_to_gimbal;
    cv::Mat translation_camera_to_gimbal;  // 3x1，单位为米，表示在云台坐标系中的平移。
    double mean_reprojection_error_px{-1.0};

    bool valid() const;
};

struct ArmorDetection {
    // 图像角点按顺时针排列：左上、右上、右下、左下。
    std::array<cv::Point2f, 4> corners{};
    ArmorSize size{ArmorSize::Small};
    int target_id{0};       // 0 保留给未知目标/无目标。
    float confidence{0.0F}; // 分类器置信度，范围为 [0, 1]。
};

struct ArmorPose {
    // 平移向量单位为米。armor_yaw 是云台坐标系中的偏航角（弧度），
    // reprojection_error 是单个观测的重投影误差（像素）。
    // center_offset_px 是板心投影点到画面中心的像素距离，供多目标筛选用。
    ArmorDetection detection;
    cv::Vec3d position_camera_m{0.0, 0.0, 0.0};
    cv::Vec3d position_gimbal_m{0.0, 0.0, 0.0};
    double armor_yaw{0.0};
    double reprojection_error{0.0};
    double center_offset_px{0.0};
};

struct PredictionResult {
    // EKF 在约定跟踪坐标系中的车体状态：位置/半径单位为米，
    // 速度单位为米每秒，角度单位为弧度，角速度单位为弧度每秒。
    TrackingState state{TrackingState::NoTarget};
    std::uint8_t target_id{0};
    float center_x_m{0.0F};
    float center_y_m{0.0F};
    float center_z_m{0.0F};
    float center_velocity_x_m_s{0.0F};
    float center_velocity_y_m_s{0.0F};
    float center_velocity_z_m_s{0.0F};
    float body_yaw{0.0F};
    float body_yaw_velocity{0.0F};
    float radius_1_m{0.0F};
    float radius_2_m{0.0F};
    // 最终云台指令：距离单位为米，角度单位为弧度，角速度单位为弧度每秒。
    float distance_m{0.0F};
    float target_yaw{0.0F};
    float target_pitch{0.0F};
    float target_yaw_velocity{0.0F};
    float target_pitch_velocity{0.0F};
    bool fire_allowed{false};

    // 跟踪过程的可观测出口，供 Web Debug 画出装甲板切换与遮挡恢复。
    // aim_index 是当前瞄准的装甲板槽位，-1 表示当前没有可瞄的板；
    // hit_streak 与 miss_count 分别是连续命中、连续无有效观测的帧数，二者互斥；
    // rejected_count 是累计被离群闸挡下的观测条数。
    int aim_index{-1};
    int hit_streak{0};
    int miss_count{0};
    int rejected_count{0};

    // 最近若干次被接受的更新的 NIS 均值。NIS 是残差相对「H P H^T + R」的马氏距离
    // 平方，三维观测下理论均值等于观测维数，也就是 3；明显偏离就说明 Q 或 R 填得
    // 不准，是唯一能把这两种病因分开的量。窗口里还没有样本时取 -1，打点跳过。
    float nis_mean{-1.0F};

    // 本帧各条观测在槽位比较里取到的最小 NIS 的最大值，取值在 NIS 闸之前，
    // 所以被拒的观测也留得下数：大于 nis_gate 说明是统计闸挡的，-1 说明本帧
    // 没有一条观测走到槽位比较，被跳变闸、重投影闸或重复板判据挡在前面。
    // 与 nis_mean 的口径不同：那个只统计被接受的更新，是窗口均值。
    float best_nis{-1.0F};
};

// 读取 OpenCV YAML，将矩阵统一为 CV_64F，并拒绝占位数据或几何上无效的标定结果。
// 失败时会在 error 中写入错误信息。
bool loadCameraParameters(const std::string& path, CameraParameters& parameters,
                          std::string& error);
