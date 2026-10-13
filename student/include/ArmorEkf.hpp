#pragma once

#include "AutoAimTypes.hpp"

#include <Eigen/Dense>

#include <array>
#include <cstdint>

// 旋转装甲板目标的扩展卡尔曼滤波。
//
// 状态 10 维，全部国际单位制：
//   x = [ px, py, pz, vx, vy, vz, theta_c, omega, r1, r2 ]^T
//     px/py/pz  旋转中心在 W 世界系中的位置 (m)
//     vx/vy/vz  旋转中心在 W 世界系中的速度 (m/s)
//     theta_c   车体偏航角，W 世界系中的方位角，左为正 (rad)
//     omega     车体偏航角速度 (rad/s)
//     r1/r2     偶数/奇数编号装甲板到旋转中心的半径 (m)
//
// 两套坐标系，互相变换只允许走 gimbalToWorld / worldToGimbal 两个函数：
//   G 云台系  x 右、y 下、z 前
//             armor_solve 输出的 position_gimbal_m 就在这个系里，
//             与相机系同向（外参 R 是单位阵，见 StudentTasks.cpp 的物体系约定）。
//   W 世界系  x 前、y 左、z 上
//             与 armor/include/BulletModel.hpp 的弹道模型约定一致。
//
// 运动模型必须建在 W 系。云台会转，同一个静止目标在 G 系里每帧坐标都在变，
// 在 G 系里做运动模型会把云台的转动误当成目标的运动。
//
// 两个偏航角的转正方向不同，别混：theta_c 是 W 系里的方位角，从 +x 转向 +y
// 为正，而 W 系 y 朝左，所以 theta_c 是左为正；gimbal.yaw 经 G 系定义成右为正。
// 两者只在观测模型里各用一次，各按各的定义，滤波器仍然自洽，不需要统一。
// 与电控实际符号对不对只能上车确认，见 ArmorEkf.cpp 顶部的说明；若云台方向
// 反了，把 gimbalToWorld 换成它的转置即可，只有那一处。
// tests/test_ekf_predict.cpp 的无噪收敛用例逐分量比对真值，两个符号都被钉住。
class ArmorEkf {
public:
    using StateVector = Eigen::Matrix<double, 10, 1>;
    using Covariance = Eigen::Matrix<double, 10, 10>;
    using ObservationJacobian = Eigen::Matrix<double, 3, 10>;

    // 状态分量的下标，避免代码里散落裸数字。
    enum Index : int {
        Px = 0,
        Py = 1,
        Pz = 2,
        Vx = 3,
        Vy = 4,
        Vz = 5,
        Yaw = 6,
        YawRate = 7,
        Radius1 = 8,
        Radius2 = 9,
        Dimension = 10,
    };

    struct Config {
        // 每块装甲板相对旋转中心的高度差 (m)，下标是板编号 0..3。
        // 默认全零。前哨站不是共面的，例如 {0, -0.102, 0.102, 0}
        // （来源 armor/include/Planner.hpp 的前哨站参数）。
        std::array<double, 4> plate_height_m{0.0, 0.0, 0.0, 0.0};

        // 装甲板在车体周向上的块数，以及相邻两块之间的夹角。
        // 默认 4 块 90 度，即标准步兵；前哨站是 3 块 120 度。
        // 板数直接决定半径状态能不能被观测到：若只有 2 块而间距仍按 90 度排，
        // 两块会落在同奇偶的槽位里，r2 永远没有观测，报出去的就是先验假数。
        int plate_count{4};
        double plate_spacing_rad{1.5707963267948966};  // 90 度

        // 过程噪声，按随机加速度模型离散化。
        double accel_noise_m_s2{6.0};               // 旋转中心三轴加速度标准差
        double angular_accel_noise_rad_s2{12.0};    // 车体 yaw 角加速度标准差
        double radius_drift_m{0.2};                 // 半径随机游走强度
        double radius_variance_floor_m2{1.0e-6};    // 半径过程噪声下限，(1 mm)^2

        // 观测噪声 sigma = scale * 距离 + base，横向与深度分开。
        double lateral_noise_scale{0.002};
        double lateral_noise_base_m{0.005};
        double depth_noise_scale{0.010};
        double depth_noise_base_m{0.010};

        // 离群观测量闸门。
        double jump_gate_m{0.6};                // 物理跳变闸：新息超过它就是坏观测
        double nis_gate{11.34};                 // 卡方分布 (3 自由度, 99%) 的分位点
        double max_reprojection_error_px{5.0};  // 单次观测重投影误差上限

        // 状态机。
        int init_frames{3};                 // 建滤波器前的连续一致帧数
        double init_consistency_m{0.3};     // 初始化窗口内板心的最大允许离散
        int stable_frames{15};              // 首次进入 Stable 需要的连续命中帧数
        int recover_frames{10};             // 丢失后回到 Stable 需要的连续命中帧数
        int stable_hold_misses{3};          // 丢这么多帧以内仍维持 Stable
        double reset_timeout_s{1.5};        // 超过这么久没有有效观测就重置
        int nis_window{5};                  // 最近 NIS 均值窗口
        double nis_mean_gate{8.0};          // 进入 Stable 要求最近 NIS 均值低于它
        double stable_center_std_m{0.10};   // 进入 Stable 的位置标准差上限
        double stable_radius_std_m{0.05};   // 进入 Stable 的半径标准差上限
        double stable_yaw_std_rad{0.10};    // 进入 Stable 的 yaw 标准差上限

        // 初始化时假定的装甲板半径 (m)。标准步兵两块板到中心同量级。
        double initial_radius_m{0.25};

        // 预测。
        double max_step_s{0.5};             // predict 的 dt 上限
        double max_latency_s{0.2};          // 单帧处理延迟上限
        int latency_window{20};             // 延迟滑动均值窗口
        double max_prediction_s{0.5};       // 总预测时长上限

        // 选板与开火。
        double side_plate_angle_rad{1.0471975511965976};  // 60 度
        double aim_switch_margin_rad{0.08726646259971647};// 5 度
        double fire_yaw_tolerance_rad{0.02};
        double fire_pitch_tolerance_rad{0.02};
        double fire_spin_tolerance_rad_s{1.0};
    };

    ArmorEkf();
    explicit ArmorEkf(const Config& config);

    // 回到初始状态。距上次有效观测超过 reset_timeout_s 时由状态机调用。
    void reset();

    // 主循环三步，按顺序每帧调用一次。
    void predict(double dt);
    // 关联、门控并更新。返回是否吃下了这条观测。
    bool update(const ArmorPose& observation, double gimbal_yaw);
    // 状态迁移与超时重置。必须每帧调用一次，包括没有观测的帧。
    void advanceStateMachine(double timestamp_seconds);

    // 生成云台指令。status 为 NoTarget 时返回全零结果。
    PredictionResult buildPrediction(const GimbalState& gimbal, double latency_s);

    // 只读访问，供测试与 Web Debug 用。
    TrackingState status() const { return status_; }
    std::uint8_t targetId() const { return target_id_; }
    int hitStreak() const { return hit_streak_; }
    int missCount() const { return miss_count_; }
    int rejectedCount() const { return rejected_count_; }
    bool hasTarget() const { return status_ != TrackingState::NoTarget; }
    const StateVector& state() const { return x_; }
    const Covariance& covariance() const { return P_; }
    const Config& config() const { return config_; }

    // 坐标系变换：p_world = gimbalToWorld(yaw) * p_gimbal。
    // M(yaw) = [ -sin  0  cos ]
    //          [ -cos  0 -sin ]
    //          [   0  -1   0  ]
    // 绕世界 z 轴转动，行列式为 +1。
    static Eigen::Matrix3d gimbalToWorld(double gimbal_yaw);
    static Eigen::Matrix3d worldToGimbal(double gimbal_yaw);

    // 观测模型与雅可比，纯函数。测试用有限差分逐元素核对解析雅可比。
    // armor_index 必须已在 [0, plate_count) 内，调用方负责折算。
    static Eigen::Vector3d observationModel(const StateVector& x, int armor_index,
                                            double gimbal_yaw,
                                            const std::array<double, 4>& plate_height_m,
                                            double plate_spacing_rad);
    static ObservationJacobian observationJacobian(const StateVector& x, int armor_index,
                                                   double gimbal_yaw,
                                                   const std::array<double, 4>& plate_height_m,
                                                   double plate_spacing_rad);

    // 把角度折算到 (-pi, pi]。
    static double wrapToPi(double angle);

private:
    // 装甲板编号折算到 [0, plate_count)，以及它对应的半径状态下标。
    static int normalizeIndex(int armor_index, int plate_count);
    static int radiusIndex(int armor_index);

    bool tryInitialize(const ArmorPose& observation, double gimbal_yaw);
    bool associateAndUpdate(const ArmorPose& observation, double gimbal_yaw);
    // commit=false 只查询不写 aim_index_，供弹道迭代这类"试算"使用。
    int selectArmorIndex(const StateVector& x, double gimbal_yaw, bool commit);
    Eigen::Vector3d platePositionWorld(const StateVector& x, int armor_index) const;

    void propagate(StateVector& x, double dt) const;
    void fillTransition(double dt, Eigen::Matrix<double, 10, 10>& F) const;
    void fillProcessNoise(double dt, Covariance& Q) const;
    Eigen::Matrix3d observationNoise(const Eigen::Vector3d& z) const;

    bool covarianceIsStable() const;
    double recentNisMean() const;
    void recordNis(double nis);
    double smoothedLatency(double raw_latency_s);

    static Covariance initialCovariance();

    Config config_;
    StateVector x_{StateVector::Zero()};
    Covariance P_{Covariance::Zero()};

    TrackingState status_{TrackingState::NoTarget};
    std::uint8_t target_id_{0};
    int hit_streak_{0};
    int miss_count_{0};
    int rejected_count_{0};
    bool updated_this_frame_{false};
    bool ever_stable_{false};
    double last_obs_ts_{0.0};

    // 一帧内的多块板不能重复配对到同一个编号。
    std::array<bool, 4> used_indices_{{false, false, false, false}};

    // 每个槽位最近一次配上的观测自带的 target_id（板上读到的数字）。
    // 通信里的 armor_flag 应该描述"正瞄着的那块板"，不是最后看到的那块，
    // 两者在多板同帧时并不是同一块。
    std::array<std::uint8_t, 4> index_target_id_{{0, 0, 0, 0}};

    // 初始化窗口：连续若干帧板心彼此接近才建滤波器，防止把两块不同的板
    // 拼成一个中心。
    Eigen::Vector3d init_mean_{Eigen::Vector3d::Zero()};
    int init_count_{0};
    // 初始化窗口按帧计数。同一帧里若有几块板，只用第一条观测，
    // 否则两块板之间的间距会顶掉一致门，计数永远凑不满。
    bool init_used_this_frame_{false};

    std::array<double, 8> nis_history_{};
    int nis_count_{0};
    int nis_head_{0};

    std::array<double, 32> latency_history_{};
    int latency_count_{0};
    int latency_head_{0};

    int aim_index_{-1};
};
