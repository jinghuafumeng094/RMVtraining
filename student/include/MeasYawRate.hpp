#pragma once

// 用相机直接量出来的车体角速度，不经过滤波器。
//
// 为什么要单独有这么一个量。页面上现有的曲线全部来自 EKF 自己的估计：
// 它自己算、自己报。滤波器错了但错得前后一致的时候，页面上看不出来，
// 因为它不会跟自己打架。本类只用两个相机侧的量算角速度
//     armor_solve 给的 ArmorPose::armor_yaw
//     串口给的 GimbalState::yaw
// 得到一条不经过滤波器的参照曲线，用来和 ekf_yaw_rate_rad_s 对照。
// 两条线贴在一起，说明滤波器跟得上相机；分叉说明它信自己的模型多过信相机。
//
// 推导。armor_yaw 是装甲板背向在云台系里的方位角，右为正，板正对镜头时为 0，
// 见 student/src/StudentTasks.cpp 的 armor_solve。云台系跟着云台转，
// 所以它不是世界系的量，要先加回世界系：
//     theta_world = armor_yaw + gimbal_yaw
// 板焊在车体上，theta_world 的下降速率就是车体自转速率；
// 而车体 yaw 在 EKF 状态里是左为正，与 theta_world 的右为正相反，故取负：
//     omega_car = -d(theta_world)/dt
// 两个偏航角的符号约定不一致不是笔误，见 student/src/ArmorEkf.cpp 顶部。
// 加回世界系之后符号自然消掉，调用处不需要各补一次。
//
// 两个必须挡住的跳变。
//   换板      同一辆车换一块板，板相位正好差 90 度。
//   双解翻转  平板 PnP 的两个解之间翻转，世界方位角差约 1.2 度。
//             法向本身差 42.1 度，但方位角才是本量在意的分量。
// 两者都表现为单帧突变。闸门取 45 度：165 fps 下真运动每帧只走零点几度，
// 离 45 度差两个数量级，误伤不到；换板的 90 度必被挡住。
// 跳变帧不更新输出，但参考点要挪过去，否则下一帧会再跳一次。
//
// 传的是采集时刻，不是帧间隔。原因是缺帧：某一帧没有观测时，方位角这一路
// 拿不到值，只能把参考点留在原地；但时间照样在走。若调用方传的是"一帧的长度"，
// 下一帧就会拿跨了两帧的角度差去除一帧的时间，速率翻倍。所以参考角与参考时刻
// 必须成对更新，只有调用方知道每一帧的真实时刻，这里就要求传时刻。
// 缺帧那一帧传非有限值即可，内部既不挪参考角也不挪参考时刻。
//
// 已知残留：双解翻转的 1.2 度小于闸门，会被原样放进输出，单帧差分出来
// 约 3.5 rad/s 的假尖刺。所以留了 smoothing_alpha 这个旋钮，
// 默认 1.0 即不平滑、给原始差分。车上按实测噪声决定要不要调小。
class MeasYawRate {
public:
    struct Config {
        // 单次 theta_world 变化超过这个值就判为跳变，不参与差分。45 度。
        double jump_gate_rad{0.7853981633974483};
        // 参考帧到本帧的间隔超过这个值就不使用：丢失一秒之后前后两点没有可比性。
        double max_dt_s{0.1};
        // 输出的指数平滑系数，1.0 表示不平滑、直接给原始差分。
        // 小于 1 时每次更新做 rate += alpha * (raw - rate)，
        // 时间常数约 1 / (alpha * 帧率) 秒。落在 (0, 1] 之外按 1.0 处理。
        double smoothing_alpha{1.0};
    };

    MeasYawRate() = default;
    explicit MeasYawRate(const Config& config);

    // 每帧调用一次。两个角单位都是弧度，符号约定见文件头，
    // timestamp_s 是本帧的采集时刻，单位秒，用单调钟即可，只取差值。
    // 首帧、跳变帧、间隔过大、输入非有限，这四种情况都只返回上一帧的值；
    // 非有限输入连参考点都不更新。
    double update(double armor_orientation_yaw, double gimbal_yaw,
                  double timestamp_s);

    // 丢弃参考点，下一次 update 重新起算，输出清零。
    // 目标丢失、滤波器重新初始化之后调用，避免拿断档前后的两点做差。
    void reset();

    // 最近一次输出的值，单位 rad/s，左正，与 ekf_yaw_rate_rad_s 同向。
    double rate() const { return rate_; }

private:
    Config config_{};
    bool has_previous_{false};
    double previous_world_yaw_{0.0};
    double previous_timestamp_s_{0.0};
    double rate_{0.0};
};
