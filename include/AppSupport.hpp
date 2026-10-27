#pragma once

#include "AutoAimTypes.hpp"
#include "Communication.hpp"

#include <cstdint>

// 本作业只处理状态 0（我方红）和 5（我方蓝）。
bool isAutoAimStatus(std::uint8_t status) noexcept;
// status 编码我方颜色；装甲板识别需要使用相反的敌方颜色。
TeamColor enemyColorFromStatus(std::uint8_t status) noexcept;
// 提取由控制器提供的姿态和预测时序字段。
GimbalState gimbalStateFromMessage(const MessData_AutoAim& message) noexcept;

// 在图上画出这一帧的检测结果：四角连成框，左上角标编号、尺寸、置信度。
// 编号 0 的板照样画框，颜色换成黄色，用来区分「检出但没读出数字」。
// 与 armor_live 的标注对齐：observations 里留下的位姿在板心画物体系三轴
// （x 红、y 绿、z 蓝，长 7 cm），框上方写 rvec / tvec 两行；同一条检测在
// observations 里找不到位姿，说明被 armor_solve 的闸挡下了，标红字 PnP x。
// 只有 camera.valid() 时才画 PnP 标注：没标定就没有位姿，全标成被拒会误导。
// 只写 image 的像素，不改 detections；空列表时一个像素都不动。
void annotateDetections(cv::Mat& image,
                        const std::vector<ArmorDetection>& detections,
                        const std::vector<ArmorPose>& observations = {},
                        const CameraParameters& camera = CameraParameters{});

// 生成确定性的无目标帧，并清空所有与开火相关的字段。
void setSafeOutput(Translator& translator) noexcept;
// 校验 EKF 结果，并将国际单位制单位映射到保持不变的通信协议。
// 特别地，center_x/y 会从米转换为毫米。
bool applyPrediction(Translator& translator, const PredictionResult& prediction) noexcept;
