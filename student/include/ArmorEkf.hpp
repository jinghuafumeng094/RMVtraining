#pragma once

#include "AutoAimTypes.hpp"

#include <Eigen/Dense>

#include <array>
#include <cstdint>
#include <limits>

// 旋转装甲板目标的扩展卡尔曼滤波。
//
// 状态 13 维，全部国际单位制：
//   x = [ px, py, pz, vx, vy, vz, theta_c, omega, r1, r2, h1, h2, h3 ]^T
//     px/py/pz  旋转中心在 W 世界系中的位置 (m)
//     vx/vy/vz  旋转中心在 W 世界系中的速度 (m/s)
//     theta_c   车体偏航角，W 世界系中的方位角，左为正 (rad)
//     omega     车体偏航角速度 (rad/s)
//     r1/r2     偶数/奇数编号装甲板到旋转中心的半径 (m)
//     h1/h2/h3  1/2/3 号槽位的板心相对 0 号槽位板心的高度差 (m)
//
// 四块板的高度是状态量，不是填死的表。观测模型里槽位 i 的板心高度写成
// pz + h_i，只有当前看得见的那块板给出一条高度方程，换板以后另一块才被
// 观测到，落差就从残差里学出来。
//
// 与 armor/src/Planner.cpp 的 z0..z3 是同一组数。planner 直接记四块板的
// 绝对高度，这里把 0 号槽位的高度当基准记进 pz，其余三块只记相对量，
// 两边张成同一个四维空间。差别只在规范怎么定：planner 靠删掉车心高度，
// 这里靠把 h0 钉成 0。二者都只有一个自由度被固定，不能两个都不定。
//
// 两套坐标系，互相变换只允许走 gimbalToWorld / worldToGimbal 两个函数：
//   G 云台系  x 前、y 左、z 上，跟着云台的 yaw 与 pitch 一起转
//             armor_solve 输出的 position_gimbal_m 就在这个系里。
//   W 世界系  x 前、y 左、z 上，水平，不随云台动
//             与 armor/include/BulletModel.hpp 的弹道模型约定一致。
//
// 两系轴向相同，没有基础变换，只差云台的两个角：先绕 y 轴去 pitch，
// 再绕 z 轴去 yaw，即 M = Rz(-yaw) * Ry(-pitch)。
//
// 2026-10-27 之前这里只去 yaw。云台是有俯仰的，那时的「世界系」其实是炮管系：
// 同一块板在 G 里的 z 是 h*cos(p) - d*sin(p)，距离项被原样算进了车心高度，
// 靶车前后走几米，ekf_z_m 就跟着走几十厘米。5 m、10 度时是 0.87 m。
//
// roll 仍不进变换。GimbalState.roll 目前没有任何一路输入给它，恒为 0；
// 真车上要加，按 Rz(-yaw) * Ry(-pitch) * Rx(-roll) 接在右边。
//
// 学长 2026-10-15 的「云台系和世界系一样」指的是轴向与没有基础变换，见
// docs/上车前问题清单.md 第 1.1 节；两个云台角仍然要转，pitch 这一层 10-27 补上。
// 代价是相机系（右 / 下 / 前）与云台系不再同向，config/camera.yaml 的旋转外参
// 因此不是单位阵而是 90 度置换矩阵；那是外参的事，本类只按上面的轴向写。
//
// 运动模型必须建在 W 系。云台会转，同一个静止目标在 G 系里每帧坐标都在变，
// 在 G 系里做运动模型会把云台的转动误当成目标的运动。
//
// 两个偏航角的转正方向不同，别混：theta_c 是 W 系里的方位角，从 +x 转向 +y
// 为正，而 +y 朝左，所以 theta_c 是左为正；gimbal.yaw 定义成右为正。
// 两者只在观测模型里各用一次，各按各的定义，滤波器仍然自洽，不需要统一。
// 与电控实际符号对不对只能上车确认，见 ArmorEkf.cpp 顶部的说明；若云台方向
// 反了，把 gimbalToWorld 换成它的转置即可，只有那一处。
// tests/test_ekf_predict.cpp 的无噪收敛用例逐分量比对真值，两个符号都被钉住。
class ArmorEkf {
public:
    using StateVector = Eigen::Matrix<double, 13, 1>;
    using Covariance = Eigen::Matrix<double, 13, 13>;
    using ObservationJacobian = Eigen::Matrix<double, 3, 13>;

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
        H1 = 10,
        H2 = 11,
        H3 = 12,
        Dimension = 13,
    };

    struct Config {
        // 四块板的初始高度差 (m)，下标是板槽位 0..3，相对量，基准是槽位 0。
        //
        // 只当先验，不是结论。板高在状态里，跑起来以后由观测自己收敛，
        // 不要求上车拿尺子量。四块板可以各不相等，四个数互相独立。
        //
        // 默认全零表示四块共面起步。填一组数也不会更准：哪个槽位对应哪块
        // 物理板取决于建滤波器那一刻第一块通过的板，相位本身随机，硬填一个
        // 相位在全片上的期望收益是零。要改就先验，不改也无所谓。
        std::array<double, 4> initial_plate_height_m{0.0, 0.0, 0.0, 0.0};
        // 初始高度差的标准差 (m)。取 3 cm，与四块板真实极差 2.7 cm 同量级：
        // 先验不把收敛拖住，也不放开到让高度与 pz 互相吞。
        double initial_plate_height_std_m{0.03};
        // 高度随机游走强度 (m/sqrt(s))。板高是刚体几何，只有换车或换镜头
        // 重新标定才变，给得很小，只留一点让它跟得上真实变化。
        double plate_height_drift_m{0.005};
        // 每帧高度过程噪声的方差下限，(0.1 mm)^2。与半径那组同理：收敛后
        // 不让滤波器把高度彻底冻死。
        double plate_height_variance_floor_m2{1.0e-8};

        // 装甲板在车体周向上的块数，以及相邻两块之间的夹角。
        // 默认 4 块 90 度，即标准步兵；前哨站是 3 块 120 度。
        // 板数直接决定半径状态能不能被观测到：若只有 2 块而间距仍按 90 度排，
        // 两块会落在同奇偶的槽位里，r2 永远没有观测，报出去的就是先验假数。
        int plate_count{4};
        double plate_spacing_rad{1.5707963267948966};  // 90 度

        // 过程噪声，按随机加速度模型离散化。
        //
        // 6.0 是 2026-10-22 初调时定的，当时 R 的横向 sigma 只有 3 mm，滤波器
        // 在噪声里找速度，这个数只能调大到压不住。R 按实测重标之后（见下面
        // observationNoise 那一组），v1.avi 上一档一档量下来。
        // 指标取"被接受的帧"上的车心帧间跳变，被拒帧保留预测、本来就不跳；
        // 统计口径见 trace_stats2。
        //   sigma_a  拒收  Stable  跳变p50  p90    p99   速度跳变p50  |v|p50
        //   6.0      37%    209    22.4 mm  68.3  225.5    71.1 mm/s   2.27
        //   3.0      38%    179    18.6 mm  68.3  285.4    51.6 mm/s   1.33
        //   1.5      39%    207    18.4 mm  63.1  218.8    37.7 mm/s   1.35
        //   0.8      41%    210    17.1 mm  61.7  218.8    33.2 mm/s   1.02
        // 靶车静止，2 m/s 以上的速度全是噪声。1.5 比 6.0 在位置、速度两处都
        // 平滑，Stable 帧数持平；3.0 那一档实测偏噪（|omega| 中位 2.07 而相邻
        // 两档是 1.05 与 0.82），单次抽到的不是趋势。0.8 再往下只剩 6% 的
        // 收益、拒收涨到 41%，取 1.5。
        double accel_noise_m_s2{1.5};               // 旋转中心三轴加速度标准差
        double angular_accel_noise_rad_s2{12.0};    // 车体 yaw 角加速度标准差
        // 半径已在车上量过，只剩几毫米的装配与量法误差，随机游走给到
        // 0.12 mm/帧即可；166 fps 下真正生效的是下面的下限。
        double radius_drift_m{0.02};                // 半径随机游走强度 (m/s)
        double radius_variance_floor_m2{9.0e-8};    // 半径过程噪声下限，(0.3 mm)^2

        // 观测噪声 sigma = scale * 距离 + base，横向与深度分开。
        //
        // 标定口径是 PnP 板心到板心的实测离散，不是角点像素抖动的折算值。
        // 2026-10-24 在 v1.avi 上逐轴量相邻帧同板板的板心差、再除以 sqrt(2)：
        // 3.96 m 处深度 62 mm、横向 22 mm、竖直 12 mm，靶车静止（整段板心斜率
        // 0.01 m/s）。旧值 0.0005 / 0.001 是在车上按角点 1.41 px 折算来的，
        // 只覆盖了亚像素定位噪声，漏掉了双解翻转、角点归属错、板模型误差这些
        // 端到端成分，横向因此小了 7 倍。后果是滤波器逐帧照抄测量，同一个真实
        // 新息算出的 NIS 被放大，v1 上 58% 的帧被闸拒掉。
        //
        // 竖直与横向同取一个值：观测噪声里两轴的物理来源不同，但 R 在云台系里
        // 本来就是对角阵，真要分开就是再加一组键，眼下没有分开的依据。
        double lateral_noise_scale{0.004};
        double lateral_noise_base_m{0.002};
        double depth_noise_scale{0.013};
        double depth_noise_base_m{0.010};

        // 离群观测量闸门。
        double jump_gate_m{0.6};                // 物理跳变闸：新息超过它就是坏观测
        double nis_gate{11.34};                 // 卡方分布 (3 自由度, 99%) 的分位点
        double max_reprojection_error_px{5.0};  // 单次观测重投影误差上限

        // armor_yaw 是第四条观测：板面法向在云台系里的偏航角，直接读出车体偏航。
        //
        // 一帧一块板只给两个位置约束，偏航与车心沿切向的分量在单板下混在一起，
        // 位置残差按方差分配，偏航与半径都失去可辨识性。接上这条观测后偏航
        // 单独被钉住，位置观测只剩车心与半径的径向分量要分。
        // 噪声按 PnP 法向的实测抖动取 0.05 rad。闸门取 0.5 rad：初始化已经按
        // 同一条公式定了偏航，正常新息是 asin(r/d) 量级，0.5 rad 挡得住换板与
        // 错关联，又不至于卡住收敛。这条观测不进 NIS 窗口，nis_mean_gate 的口径
        // 仍是三维卡方。
        double armor_yaw_noise_rad{0.05};
        double armor_yaw_gate_rad{0.5};

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

        // 连续拒收后的协方差放大。拒收既不改进状态也不改进协方差，收敛后的 P
        // 很小，一条偏离真值的坏观测能把门永久锁死，只能等 reset_timeout_s 做
        // 整段重置。连续 reject_recovery_frames 帧一条都没吃下时，把 P 的对角线
        // 乘上 reject_inflation_factor，上限是 initialCovariance() 的对角线。
        //
        // 3 帧与 stable_hold_misses 对齐：不再信任 Stable 的那一刻起就让自己
        // 愿意重新接受观测。每 3 帧翻一倍，位置标准差从毫米级抬到厘米级只要
        // 十几帧，远快于 1.5 秒的重置；上限保证不会比初始化时更不确定。
        int reject_recovery_frames{3};
        double reject_inflation_factor{2.0};

        // 模型离每一块板都超过 jump_gate_m 时的重建门槛。
        //
        // 跳变闸是物理闸，与 P 无关，放大协方差撑不开它。一旦状态被带偏，
        // 所有槽位都在闸外，观测永远进不来，只能等 reset_timeout_s 的整段
        // 重置。丢了这么多帧之后，把新观测喂进初始化窗口，攒够 init_frames
        // 条彼此一致的观测就地重建，比整段重置快一个数量级。
        // 3 帧与 init_frames 对齐：正是初始化本身要求的帧数。
        int reinit_after_misses{3};

        // 装甲板半径 (m)，旋转中心到板心。靶车记录 Video_1 里相邻两板心的刚性
        // 弦长恒为 0.428 m，四板近似等距，按方阵算出 0.303 m。
        // 2026-10-20 用卷尺量到 27.3 cm，比这个值小 10.9%。内参已用 92 张棋盘图
        // 独立重标（fx 1814.75，比交付的 1782.6 还大 1.8%，方向相反），灯条中心线
        // 135 mm 已两次确认，2026-10-21 判定是卷尺读数偏小，物理半径取 0.303。
        // 同位 A/B：0.303 的 Stable 占 97.3%、累计拒收 346，0.273 只有 88.4%
        // 与 1318。三处取 0.303，r1 与 r2 同值。
        // 2026-10-24 换了我方相机的镜头，目标车没换，物理半径不变，这个数不动。
        // 这个数是 PnP 口径的：只有当某份 K 的比例偏了才需要缩放，那时正确做法是
        // 按 PnP 距离与卷尺之比改本项，见 docs/上车前问题清单.md 2.2。
        // 新镜头 fx 2433.8 若标得准，同一距离的 PnP 读数应与卷尺相符，本项照旧。
        double initial_radius_m{0.303};
        // 初始半径的标准差 (m)。半径已由上下界钉死，这个数不再决定估计，
        // 取 3 mm 只为初始化时不给出一份与钳位矛盾的协方差。
        double initial_radius_std_m{0.003};

        // 装甲板半径的范围 (m)，上下界都取 initial_radius_m，半径是常量而不是待估量。
        // 单板帧里半径与车心径向严格不可辨识，放开它只会让 r 滑到边界，再靠
        // clampRadii 平移车心维持自洽，车心因此每帧多出几厘米的位移。
        // 钉死之后 clampRadii 仍把车心对齐到刚吃下的那条观测。状态里保留
        // Radius1/Radius2 只为与上游 Planner 的状态向量同构。
        double min_radius_m{0.303};
        double max_radius_m{0.303};

        // 枪口相对云台 P 轴原点的前移量 (m)。相机、云台 P 轴、摩擦轮三者刚体固定，
        // 所以在云台系里这是个常向量 (muzzle_offset_m, 0, 0)，沿炮管前方，不随 yaw
        // 或 pitch 变。来源 docs/云台外参.md 的「云台 P 轴 -> 摩擦轮」74.1 mm，与
        // armor/src/Planner.cpp 的 translation_vector{74.1, 0.0, 0.0} 是同一个量。
        //
        // 弹丸从枪口飞出，所以弹道模型的两个输入必须从枪口量起。这一项只进
        // buildPrediction 的弹道解，不进状态、不进观测。状态原点留在云台 P 轴：
        // 观测天然相对它，而且它是三条转动轴的交点，在云台任何转动下都不动。
        double muzzle_offset_m{0.0741};

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
    // gimbal_pitch 抬头为正，与 GimbalState.pitch 同口径。默认 0 只给不带
    // 俯仰的回放工具与测试用，主链路必须传真实值，否则世界系又变回炮管系。
    bool update(const ArmorPose& observation, double gimbal_yaw,
                double gimbal_pitch = 0.0);
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
    // 当前瞄准的装甲板槽位，-1 表示没有可瞄的板。
    int aimIndex() const { return aim_index_; }
    bool hasTarget() const { return status_ != TrackingState::NoTarget; }
    const StateVector& state() const { return x_; }
    const Covariance& covariance() const { return P_; }
    const Config& config() const { return config_; }
    // 最近 nis_window 次被接受的更新的 NIS 均值。窗口里没有样本时返回正无穷，
    // 好让进入 Stable 的那条闸「< nis_mean_gate」判成假：没有样本就不许进稳态。
    // 这个哨兵值是为闸门挑的，往 Web Debug 打点前要先判 std::isfinite，
    // 见 PredictionResult::nis_mean 与 main.cpp 的 logPrediction。
    double recentNisMean() const;

    // 坐标系变换：p_world = gimbalToWorld(yaw, pitch) * p_gimbal。
    // M = Rz(-yaw) * Ry(-pitch)
    //   Rz(-yaw) = [  cos  sin  0 ]      Ry(-pitch) = [ cos  0 -sin ]
    //              [ -sin  cos  0 ]                   [  0   1   0  ]
    //              [   0    0   1 ]                   [ sin  0  cos ]
    // 两系轴向相同，只有这两个云台角，行列式为 +1。
    // gimbal_yaw 取正值时云台向右转，世界系里的视线朝 -y 走；
    // gimbal_pitch 取正值时炮管抬头，云台 +x 在世界系里朝 +z 抬。
    static Eigen::Matrix3d gimbalToWorld(double gimbal_yaw, double gimbal_pitch = 0.0);
    static Eigen::Matrix3d worldToGimbal(double gimbal_yaw, double gimbal_pitch = 0.0);

    // 弹道模型的两个输入：目标相对枪口的水平距离与高度差，都在水平的云台系里量。
    // 枪口 = 云台原点 + (muzzle_offset_m, 0, 0)。hypot(x, y) 与 z 都不过 yaw，
    // 所以在只去 yaw 的系里算，与在 W 里减掉旋转后的那个向量逐位相同。
    // 反过来，若在 W 里直接减常向量 (muzzle_offset_m, 0, 0)，误差随 yaw 走，
    // yaw = 90 度时正好等于一整个 muzzle_offset_m。
    //
    // p_gimbal 必须已经去掉 pitch，即 worldToGimbal(yaw) 的结果。弹道要的是
    // 重力方向下的分解，叠进 pitch 后 z 不再竖直。枪口高 δsinφ 按 0 处理，是已知近似。
    struct BallisticInputs {
        double horizontal_m{0.0};
        double vertical_m{0.0};
    };
    static BallisticInputs ballisticInputsFromMuzzle(const Eigen::Vector3d& p_gimbal,
                                                     double muzzle_offset_m);

    // 槽位 armor_index 的板心相对 pz 的高度差 (m)。0 号槽位恒为 0，其余读状态。
    // 观测模型与 platePositionWorld 都走这一个函数，两处不会各写一份。
    static double plateHeightOffset(const StateVector& x, int armor_index);

    // 观测模型与雅可比，纯函数。测试用有限差分逐元素核对解析雅可比。
    // armor_index 必须已在 [0, plate_count) 内，调用方负责折算。
    static Eigen::Vector3d observationModel(const StateVector& x, int armor_index,
                                            double gimbal_yaw,
                                            double plate_spacing_rad,
                                            double gimbal_pitch = 0.0);
    static ObservationJacobian observationJacobian(const StateVector& x, int armor_index,
                                                   double gimbal_yaw,
                                                   double plate_spacing_rad,
                                                   double gimbal_pitch = 0.0);

    // 把角度折算到 (-pi, pi]。
    static double wrapToPi(double angle);

private:
    // 装甲板编号折算到 [0, plate_count)，以及它对应的半径状态下标。
    static int normalizeIndex(int armor_index, int plate_count);
    static int radiusIndex(int armor_index);
    // 槽位对应的板高状态下标，0 号槽位没有独立状态，返回 -1。
    static int plateHeightIndex(int armor_index);

    bool tryInitialize(const ArmorPose& observation, double gimbal_yaw,
                       double gimbal_pitch);
    bool associateAndUpdate(const ArmorPose& observation, double gimbal_yaw,
                            double gimbal_pitch);
    // 板面法向的偏航是一条与板心位置独立的观测，按标量卡尔曼只更新 Yaw。
    // 位置更新被 NIS 闸拒掉时也照样调用，见 associateAndUpdate 里的说明。
    // 只收 gimbal_yaw：板面法向是水平的，绕 y 轴转 pitch 只会让它在这个系里
    // 的方位角变化一个二阶量，30 度俯仰下最大约 4 度，远在 0.5 rad 的闸之内，
    // 也当不成观测。位置那条链路才是 pitch 必须进去的地方。
    void updateYaw(const ArmorPose& observation, int armor_index, double gimbal_yaw);
    // 把一帧板心喂进初始化窗口，返回窗口是否已经攒够。
    bool accumulateInitWindow(const Eigen::Vector3d& plate_world);
    // 按一块世界系板心反推旋转中心，建起滤波器。初始化与跳闸外重建共用。
    void startTracking(const Eigen::Vector3d& plate_world, std::uint8_t target_id);
    // commit=false 只查询不写 aim_index_，供弹道迭代这类"试算"使用。
    // 比较的是方位角，方位角绕的是竖直轴，与俯仰无关，所以只收 gimbal_yaw。
    int selectArmorIndex(const StateVector& x, double gimbal_yaw, bool commit);
    Eigen::Vector3d platePositionWorld(const StateVector& x, int armor_index) const;
    // 把两个半径夹进 [min_radius_m, max_radius_m]，并平移车心，保持
    // observed_index 这块板的位置不动，刚吃下的观测仍然成立。
    void clampRadii(int observed_index);

    void propagate(StateVector& x, double dt) const;
    void fillTransition(double dt, Eigen::Matrix<double, 13, 13>& F) const;
    void fillProcessNoise(double dt, Covariance& Q) const;
    Eigen::Matrix3d observationNoise(const Eigen::Vector3d& z) const;

    bool covarianceIsStable() const;
    void recordNis(double nis);
    double smoothedLatency(double raw_latency_s);

    Covariance initialCovariance() const;

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

    // 本帧观测在槽位比较里取到的最小 NIS 的最大值，每帧在 predict 里清成 -1。
    // 与 nis_history_ 的区别：这个在闸门之前取值，被拒的观测也记。NIS 非负，
    // 所以 -1 可以当"本帧没有一条观测走到槽位比较"的哨兵，直接进出口结构体。
    // 不用正无穷：清零后取 max 会一直停在无穷上，那是个只会静默失效的写法。
    double frame_best_nis_{-1.0};

    std::array<double, 32> latency_history_{};
    int latency_count_{0};
    int latency_head_{0};

    int aim_index_{-1};
};
