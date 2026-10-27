#include "AppSupport.hpp"
#include "MeasYawRate.hpp"
#include "SerialPort.hpp"
#include "StudentTasks.hpp"
#include "WebLogger.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <string>
#include <thread>
#include <vector>

// 这个文件有意保留 aimbot_26/main.cpp 的组织方式：
// main 负责初始化，ReadFunction 负责收串口，OperationFunction 负责整条视觉流水线。
// 学生只需要替换 student/src/StudentTasks.cpp 中的四个任务函数。

namespace {

constexpr int kDefaultWebPort = 8080;
constexpr int kInitialReadAttempts = 5;
constexpr auto kIdleDelay = std::chrono::milliseconds(10);
// 控制器数据过期后，结果必须降级为无目标，不能继续允许开火。
constexpr auto kMessageMaxAge = std::chrono::milliseconds(500);

// 程序退出标志。原子变量同时供信号处理函数、读线程和操作线程使用。
static_assert(std::atomic<bool>::is_always_lock_free,
              "退出标志必须支持无锁原子操作");
std::atomic<bool> shutdown_requested{false};

// 下面几个全局变量对应参考工程中的 camera、temp、translator、pic。
// 相机 SDK 的对象和生命周期由学生在 get_pic() 中自行管理，这里保存已标定参数。
CameraParameters camera;
Translator temp;       // 读线程写入的最新控制器帧
Translator translator; // 操作线程使用并回写的当前帧
cv::Mat pic;           // 操作线程当前取得的图像

// temp 及其时间戳由同一把锁保护，避免读线程写入时操作线程同时复制。
std::mutex message_mutex;
bool message_available = false;
std::chrono::steady_clock::time_point message_received_at{};

int web_port = kDefaultWebPort;

struct Options {
    std::string serial_device;
    int baud_rate{115200};
    std::string camera_config;
};

void request_shutdown(int) {
    shutdown_requested.store(true, std::memory_order_relaxed);
}

void printUsage(const char* program) {
    std::cout << "用法: " << program
              << " <串口设备> <波特率> <相机标定文件> [Web端口]\n"
              << "示例: " << program
              << " /dev/ttyACM0 115200 config/camera.yaml 8080\n";
}

bool parsePositiveInt(const char* text, int& value) {
    try {
        std::size_t parsed = 0;
        const int candidate = std::stoi(text, &parsed);
        if (text[parsed] != '\0' || candidate <= 0) {
            return false;
        }
        value = candidate;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool parseOptions(int argc, char** argv, Options& options) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        printUsage(argv[0]);
        return false;
    }
    if (argc < 4 || argc > 5) {
        printUsage(argv[0]);
        return false;
    }

    options.serial_device = argv[1];
    options.camera_config = argv[3];
    if (!parsePositiveInt(argv[2], options.baud_rate)) {
        std::cerr << "波特率无效: " << argv[2] << '\n';
        return false;
    }
    if (argc == 5 && !parsePositiveInt(argv[4], web_port)) {
        std::cerr << "Web端口无效: " << argv[4] << '\n';
        return false;
    }
    return true;
}

bool hasFrameMarkers(const char* frame) noexcept {
    // 帧头和帧尾只用于确认边界；浮点数是否有效在操作线程中检查。
    return static_cast<std::uint8_t>(frame[0]) == kFrameHead &&
           static_cast<std::uint8_t>(frame[kCommunicationFrameSize - 1]) == kFrameTail;
}

bool hasFiniteControlInput(const MessData_AutoAim& message) noexcept {
    return std::isfinite(message.yaw) && std::isfinite(message.pitch) &&
           std::isfinite(message.roll) && std::isfinite(message.bias);
}

void publishFrame(const Translator& incoming) {
    std::lock_guard<std::mutex> lock(message_mutex);
    temp = incoming;
    message_received_at = std::chrono::steady_clock::now();
    message_available = true;
}

bool copyLatestFrame(Translator& destination,
                     std::chrono::steady_clock::time_point& received_at) {
    std::lock_guard<std::mutex> lock(message_mutex);
    if (!message_available) {
        return false;
    }
    destination = temp;
    received_at = message_received_at;
    return true;
}

bool frameIsFresh(const std::chrono::steady_clock::time_point& received_at) {
    return std::chrono::steady_clock::now() - received_at <= kMessageMaxAge;
}

bool writeFrame(SerialPort& serialPort, Translator& frame) {
    // 统一设置协议边界，并保证一次写出完整的 64 字节帧。
    frame.message.head = kFrameHead;
    frame.message.tail = kFrameTail;
    const int written =
        serialPort.Write(frame.data, static_cast<int>(kCommunicationFrameSize));
    if (written != static_cast<int>(kCommunicationFrameSize)) {
        std::cerr << "串口写入失败\n";
        shutdown_requested.store(true, std::memory_order_relaxed);
        return false;
    }
    return true;
}

void logPrediction(const PredictionResult& result) {
    // ekf_state 是 0/1/2 三态，0 就表示没有目标，不再单独记一个布尔量。
    WEB_LOG("ekf_state", static_cast<double>(result.state));
    if (result.state == TrackingState::NoTarget) {
        return;
    }
    WEB_LOG("target_id", result.target_id);
    WEB_LOG("target_yaw", result.target_yaw);
    WEB_LOG("target_pitch", result.target_pitch);
    WEB_LOG("target_distance", result.distance_m);
    WEB_LOG("ekf_x_m", result.center_x_m);
    WEB_LOG("ekf_y_m", result.center_y_m);
    WEB_LOG("ekf_z_m", result.center_z_m);
    WEB_LOG("ekf_vx_m_s", result.center_velocity_x_m_s);
    WEB_LOG("ekf_vy_m_s", result.center_velocity_y_m_s);
    WEB_LOG("ekf_vz_m_s", result.center_velocity_z_m_s);
    WEB_LOG("ekf_yaw_rad", result.body_yaw);
    WEB_LOG("ekf_yaw_rate_rad_s", result.body_yaw_velocity);
    // 切换与遮挡的直接证据：当前瞄准的槽位、连续命中与连续丢失、累计被离群闸挡下的观测条数。
    // hit_streak 与 miss_count 互斥，任何时刻只有一个非零，所以合成一条带符号的曲线：
    // 正值为连续命中帧数，负值为连续丢失帧数，0 表示刚重置。
    WEB_LOG("ekf_aim_index", result.aim_index);
    WEB_LOG("ekf_streak", result.hit_streak - result.miss_count);
    WEB_LOG("ekf_rejected_count", result.rejected_count);
    // 调参唯一能自查的一个数：残差相对「H P H^T + R」的马氏距离平方，取最近几次
    // 被接受的更新的均值。三维观测下理论均值等于观测维数，也就是 3。持续大于 3
    // 说明 R 给小了、滤波器比实际更自信，持续小于 3 说明给大了。它和别处的曲线
    // 不同，不描述目标，只描述滤波器对自己有多信。
    // 窗口里还没有样本时是 -1，这一帧不打点，免得 -1 和无穷大把纵轴拉飞。
    if (result.nis_mean >= 0.0F) {
        WEB_LOG("ekf_nis_mean", result.nis_mean);
    }
    // 本帧被评估的那条观测的 NIS，取值在闸门之前。上面那条只统计被接受的更新，
    // 观测被拒的帧在那条线上什么都不留；这一条让被拒的帧也留下数：大于 nis_gate
    // 说明是统计闸挡的，-1 说明这一帧没有观测走到槽位比较。
    if (result.best_nis >= 0.0F) {
        WEB_LOG("ekf_best_nis", result.best_nis);
    }
    // 以下三个是派生角，不参与任何控制，也不回流进滤波器。它们的作用是把「估出来的状态
    // 到底符不符合物理」变成肉眼可判的形状：车体匀速自转时，这三个角各自有一条已知的
    // 波形，形状、周期、幅值都能事先算出来，对不上就说明相应的状态没收敛或者模型写错了。
    constexpr double kPi = 3.14159265358979323846;
    constexpr double kHalfPi = 0.5 * kPi;

    // 一、顶点在敌车旋转中心。一条边指向固定 0 号槽位的装甲板，另一条指向我方云台原点。
    // 取无符号夹角，所以它是一路匀速涨到 pi 再匀速折回的三角波，速率恒为
    // ekf_yaw_rate_rad_s，周期 2*pi/omega。
    // 板方位用 body_yaw 拼，不用 aim_index：编号一旦跟着瞄点切换，这个角会塌成 0 附近的平线。
    // 只记 0 号槽位，四块板的这个角只差 90 度的固定相位，形状完全一样。
    const double to_me_azimuth = std::atan2(-result.center_y_m, -result.center_x_m);
    WEB_LOG("ekf_plate_me_angle_rad",
            std::abs(std::remainder(result.body_yaw - to_me_azimuth, 2.0 * kPi)));

    // 二、车体相位折到八分之一圈。取值 0 到 pi/4，峰在 body_yaw 为 45 度加 90 度整数倍处，
    // 谷在 0 度加 90 度整数倍处，周期是上面那条的四分之一。
    // 它只由 body_yaw 决定，与位置和距离无关，所以能把两种病因分开：这条干净而上面那条毛，
    // 问题在中心；两条一起毛，问题在 yaw。慢转时上面那条一个周期要几十秒，这条只要几分之一，
    // 是现场唯一看得全的三角波。
    WEB_LOG("ekf_slot_phase_rad", std::abs(std::remainder(result.body_yaw, kHalfPi)));

    // 三、0 号槽位板对我方云台的方位角，减去车心对我方云台的方位角，即这块板相对车心的
    // 视差角。角顶点在我方云台，所以它不随车体平移漂走，是绕 0 振荡的正弦，周期同样是
    // 2*pi/omega，幅值上限 asin(radius_1_m / 距离)。
    // 幅值偏大说明半径估大了，周期不对说明角速度不对，整条慢慢偏离 0 说明中心估偏了。
    // 方位角口径与 pnp_yaw_rad 一致，右正。
    const double slot0_x = result.center_x_m + result.radius_1_m * std::cos(result.body_yaw);
    const double slot0_y = result.center_y_m + result.radius_1_m * std::sin(result.body_yaw);
    const double to_center_azimuth = std::atan2(-result.center_y_m, result.center_x_m);
    const double to_slot0_azimuth = std::atan2(-slot0_y, slot0_x);
    WEB_LOG("ekf_plate0_parallax_rad",
            std::remainder(to_slot0_azimuth - to_center_azimuth, 2.0 * kPi));

    // 四、第一条在竖直面里的同一个角。顶点仍在敌车旋转中心，一条边指向当前瞄准的
    // 那块板，另一条指向我方云台原点，两条各取仰角再相减。
    // 板心相对车心的仰角 atan2(板高, 半径) 只取决于这是第几块板，瞄点一换整段跳，
    // 所以这条是四级矩形波，台阶高度约为相邻两块板的高度差除以半径。
    // 这一条必须用 aim_index，跟第一条正好相反：不跟着瞄点走，它就退化成一条常数线。
    // 段内是斜坡而不是平台，说明车心高度在漂，斜率对应 ekf_vz_m_s 的误差。
    // 取带符号的差，板心低于云台原点时值为负，四级台阶的先后顺序因此也看得出来。
    // 板高从 result 里取，出处仍是 ArmorEkfConfig::plate_height_m，不在本文件另抄一份。
    if (result.aim_index >= 0) {
        const double plate_radius =
            result.aim_index % 2 == 0 ? result.radius_1_m : result.radius_2_m;
        const double plate_elevation =
            std::atan2(result.aim_plate_height_m, plate_radius);
        const double me_elevation = std::atan2(
            -result.center_z_m, std::hypot(result.center_x_m, result.center_y_m));
        WEB_LOG("ekf_plate_me_pitch_rad", plate_elevation - me_elevation);
    }
}

// 相机侧的两个量，与上面那一组的关键区别是不经过滤波器。
//
// 页面上其余的曲线全部来自 EKF，它自己算、自己报，错得前后一致时看不出来。
// 这一组只用 armor_solve 给的板背方位角和串口给的云台角做差，得到一条独立于
// 滤波器的车体角速度参照曲线：和 ekf_yaw_rate_rad_s 贴在一起说明滤波器跟得上相机，
// 分叉说明它更信自己的模型。这正好把"跟不上"和"模型有偏"这两种病因分开。
//
// armor_orientation_yaw 是装甲板背向在云台系里的方位角，右正，板正对镜头时为 0。
// 它单独画没有意义，云台一转它就动；要和 gimbal_yaw 一起看，两者相加才是板在
// 世界系里的方位角，而世界系方位角的下降速率就是车体自转速率。
//
// 多块板同时可见时按"跟着上一帧那块板走"选取，取与上一帧方位角最接近的一块。
// 这样物理换板才是一次 90 度跳变，而不是每帧在几块板之间乱跳；单块板可见时
// 这条规则退化成"就取那一块"。跟随用的上一帧值由本函数自己维护，与滤波器无关。
void logMeasYawRate(const std::vector<ArmorPose>& observations,
                    const GimbalState& gimbal, double timestamp_s) {
    static MeasYawRate meas;
    static bool has_previous = false;
    static double previous_yaw = 0.0;
    constexpr double kTwoPi = 6.28318530717958647692;

    WEB_LOG("gimbal_yaw", gimbal.yaw);

    if (observations.empty()) {
        // 没有观测就不喂角，MeasYawRate 会保持上一帧的输出；
        // 参考时刻也不动，缺的这一帧的时间摊进下一帧的分母，速率不被放大。
        WEB_LOG("ekf_meas_yaw_rate_rad_s", meas.rate());
        return;
    }

    const ArmorPose* chosen = &observations.front();
    if (has_previous) {
        double best =
            std::fabs(std::remainder(chosen->armor_yaw - previous_yaw, kTwoPi));
        for (const ArmorPose& observation : observations) {
            const double distance =
                std::fabs(std::remainder(observation.armor_yaw - previous_yaw, kTwoPi));
            if (distance < best) {
                best = distance;
                chosen = &observation;
            }
        }
    }

    const double rate = meas.update(chosen->armor_yaw, gimbal.yaw, timestamp_s);
    has_previous = true;
    previous_yaw = chosen->armor_yaw;

    WEB_LOG("armor_orientation_yaw", chosen->armor_yaw);
    WEB_LOG("ekf_meas_yaw_rate_rad_s", rate);
}

// 网页俯视图的蓝色框：本帧每一块解出位姿的板，位置和板背方位角。
//
// 位置保持云台系，不在这里换世界坐标。俯视图的 --camera 模式已经有一条
// 从云台系到世界系的换算路径，两处各算一遍迟早会不一致；这里只负责把
// 原始观测送出去，由 target_topdown_ui 统一换算。
//
// 每帧都写 obs_count，没有观测时写 0，把上一帧的蓝框清掉，否则屏幕上会
// 留着已经丢掉的板。上限取 4，是哨兵车能同时看到的板数上限，同时也是
// 序号只用一位数字的前提，超出部分不画。
constexpr int kMaxLoggedObservations = 4;

void logObservations(const std::vector<ArmorPose>& observations) {
    const int count = std::min(static_cast<int>(observations.size()),
                               kMaxLoggedObservations);
    WEB_LOG("obs_count", static_cast<double>(count));
    for (int i = 0; i < count; ++i) {
        const ArmorPose& observation = observations[static_cast<std::size_t>(i)];
        WEB_LOG("obs" + std::to_string(i) + "_x", observation.position_gimbal_m[0]);
        WEB_LOG("obs" + std::to_string(i) + "_y", observation.position_gimbal_m[1]);
        WEB_LOG("obs" + std::to_string(i) + "_yaw", observation.armor_yaw);
    }
}

// main 中启动线程前先同步读一帧，保持参考工程的初始化顺序。
// 返回 1 表示读到合法帧，0 表示暂时超时，-1 表示串口出错。
int readInitialFrame(SerialPort& serialPort, Translator& frame) {
    for (int attempt = 0; attempt < kInitialReadAttempts &&
                            !shutdown_requested.load(std::memory_order_relaxed);
         ++attempt) {
        Translator incoming{};
        const int length =
            serialPort.Read(incoming.data, static_cast<int>(kCommunicationFrameSize));
        if (length < 0) {
            return -1;
        }
        if (length == static_cast<int>(kCommunicationFrameSize) &&
            hasFrameMarkers(incoming.data)) {
            frame = incoming;
            return 1;
        }
    }
    return 0;
}

}  // 匿名命名空间

// 串口读线程：只负责读取完整帧并更新共享的 temp。
void *ReadFunction(void *arg);
// 操作线程：按照“复制 -> 取流 -> 检测 -> PnP -> EKF -> 写回”的顺序处理。
void *OperationFunction(void *arg);

int main(int argc, char** argv) {
    std::signal(SIGINT, request_shutdown);
    std::signal(SIGTERM, request_shutdown);

    Options options;
    if (!parseOptions(argc, argv, options)) {
        return argc == 2 && std::string(argv[1]) == "--help" ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    std::string camera_error;
    if (!loadCameraParameters(options.camera_config, camera, camera_error)) {
        std::cerr << camera_error << '\n';
        return EXIT_FAILURE;
    }

    // 与参考工程一样，串口对象在 main 中创建，再把指针传给两个线程。
    SerialPort* serialPort = new SerialPort(options.serial_device);
    if (!serialPort->isOpen()) {
        std::cerr << "无法打开串口设备: " << options.serial_device << '\n';
        delete serialPort;
        return EXIT_FAILURE;
    }
    if (!serialPort->InitSerialPort(options.baud_rate, 8, 1, 'N')) {
        std::cerr << "串口配置失败\n";
        delete serialPort;
        return EXIT_FAILURE;
    }

    // 先同步读取首帧并缓存控制器状态；后续每帧仍会重新读取 status 来确定敌方颜色。
    Translator first_frame{};
    const int initial_result = readInitialFrame(*serialPort, first_frame);
    if (initial_result < 0) {
        std::cerr << "读取串口首帧失败\n";
        delete serialPort;
        return EXIT_FAILURE;
    }
    if (initial_result > 0) {
        publishFrame(first_frame);
        std::cout << "收到首帧，status = "
                  << static_cast<unsigned int>(first_frame.message.status) << '\n';
    } else {
        // 首帧超时不直接退出，读线程启动后仍会继续等待控制器数据。
        std::cerr << "未在启动阶段收到合法首帧，继续等待串口数据\n";
    }

    pthread_t readThread{};
    pthread_t operationThread{};
    bool read_started = false;
    bool operation_started = false;

    // 保留参考工程的 pthread 双线程结构：先创建读线程，再创建操作线程。
    if (pthread_create(&readThread, nullptr, ReadFunction, serialPort) == 0) {
        read_started = true;
    } else {
        std::cerr << "创建串口读线程失败\n";
    }
    if (read_started &&
        pthread_create(&operationThread, nullptr, OperationFunction, serialPort) == 0) {
        operation_started = true;
    } else {
        std::cerr << "创建视觉操作线程失败\n";
        shutdown_requested.store(true, std::memory_order_relaxed);
    }

    // 先等待操作线程结束，再通知读线程退出，和原 main 的生命周期一致。
    if (operation_started) {
        pthread_join(operationThread, nullptr);
    }
    shutdown_requested.store(true, std::memory_order_relaxed);
    if (read_started) {
        pthread_join(readThread, nullptr);
    }

    STOP_WEB_SERVER();
    delete serialPort;
    return operation_started ? EXIT_SUCCESS : EXIT_FAILURE;
}

void *ReadFunction(void *arg) {
    SerialPort* serialPort = static_cast<SerialPort*>(arg);
    if (serialPort == nullptr) {
        shutdown_requested.store(true, std::memory_order_relaxed);
        return nullptr;
    }

    // SerialPort::Read 已经负责短读缓存；这里额外保留滑动缓冲，
    // 使程序从半帧启动或遇到干扰字节时仍能重新找到 0x71 ... 0x4C。
    std::vector<char> stream_buffer;
    stream_buffer.reserve(kCommunicationFrameSize * 2);

    while (!shutdown_requested.load(std::memory_order_relaxed)) {
        std::array<char, kCommunicationFrameSize> chunk{};
        const int length =
            serialPort->Read(chunk.data(), static_cast<int>(chunk.size()));
        if (length < 0) {
            std::cerr << "串口读取失败\n";
            shutdown_requested.store(true, std::memory_order_relaxed);
            break;
        }
        if (length == 0) {
            // 超时不是错误；短暂让出 CPU 后继续等待下一帧。
            std::this_thread::sleep_for(kIdleDelay);
            continue;
        }
        if (length > static_cast<int>(chunk.size())) {
            continue;
        }

        stream_buffer.insert(stream_buffer.end(), chunk.begin(), chunk.begin() + length);
        std::size_t consumed = 0;
        while (stream_buffer.size() - consumed >= kCommunicationFrameSize) {
            const char* candidate = stream_buffer.data() + consumed;
            if (!hasFrameMarkers(candidate)) {
                ++consumed;
                continue;
            }

            Translator incoming{};
            std::memcpy(incoming.data, candidate, kCommunicationFrameSize);
            consumed += kCommunicationFrameSize;
            publishFrame(incoming);
        }
        if (consumed != 0U) {
            stream_buffer.erase(stream_buffer.begin(),
                                stream_buffer.begin() + static_cast<std::ptrdiff_t>(consumed));
        }
    }
    return nullptr;
}

void *OperationFunction(void *arg) {
    SerialPort* serialPort = static_cast<SerialPort*>(arg);
    if (serialPort == nullptr) {
        shutdown_requested.store(true, std::memory_order_relaxed);
        return nullptr;
    }

    // Web 服务和相机 SDK 都属于操作线程的资源，生命周期与原工程一致。
    START_WEB_SERVER(web_port);

    auto last_frame_time = std::chrono::steady_clock::time_point{};
    int empty_frame_count = 0;

    while (!shutdown_requested.load(std::memory_order_relaxed)) {
        const auto process_start = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point received_at{};

        // 对应 aimbot_26 中的 MManager.copy(temp, translator)。
        if (!copyLatestFrame(translator, received_at)) {
            std::this_thread::sleep_for(kIdleDelay);
            continue;
        }

        // 不支持的状态或异常姿态不进入学生算法，直接回传无目标帧。
        if (!frameIsFresh(received_at) || !isAutoAimStatus(translator.message.status) ||
            !hasFiniteControlInput(translator.message)) {
            setSafeOutput(translator);
            writeFrame(*serialPort, translator);
            // 同一份过期快照可能会被重复看到，避免无休止地高速回包。
            std::this_thread::sleep_for(kIdleDelay);
            continue;
        }

        const GimbalState gimbal = gimbalStateFromMessage(translator.message);

        // 对应参考工程中的 camera.get_pic(&pic, ...)。具体 SDK 初始化和取流由学生完成。
        pic.release();
        const bool got_pic = get_pic(pic);
        bool using_fallback_frame = false;
        if (!got_pic || pic.empty()) {
            ++empty_frame_count;
            WEB_LOG("empty_frame", static_cast<double>(empty_frame_count));
            if (empty_frame_count > 3) {
                // 与参考工程一致，连续多次取流失败后结束操作线程。
                setSafeOutput(translator);
                translator.message.latency = static_cast<float>(
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - process_start)
                        .count());
                writeFrame(*serialPort, translator);
                std::cerr << "连续取流失败，操作线程退出\n";
                shutdown_requested.store(true, std::memory_order_relaxed);
                break;
            }
            // 临时补一张黑帧，使后面的检测、PnP 和 EKF 阶段仍按原顺序执行。
            // 该帧不是相机观测，最终输出会被强制设为无目标。
            pic = cv::Mat(camera.image_height, camera.image_width, CV_8UC3,
                          cv::Scalar(0, 0, 0));
            using_fallback_frame = true;
        } else {
            empty_frame_count = 0;
        }
        if (!using_fallback_frame) {
            WEB_LOG("empty_frame", 0.0);
        }
        const auto frame_time = std::chrono::steady_clock::now();
        if (last_frame_time.time_since_epoch().count() != 0) {
            const double frame_seconds =
                std::chrono::duration<double>(frame_time - last_frame_time).count();
            if (frame_seconds > 0.0) {
                WEB_LOG("camera_fps", 1.0 / frame_seconds);
            }
        }
        last_frame_time = frame_time;

        // 1. 装甲板检测。检测结果为空时不提前结束，继续向下传递空观测。
        const std::vector<ArmorDetection> detections =
            armor_detect(pic, enemyColorFromStatus(translator.message.status));
        WEB_LOG("armor_count", static_cast<double>(detections.size()));
        if (!detections.empty()) {
            // 记录第一块候选装甲板，便于在网页中快速判断检测器是否工作。
            const ArmorDetection& first_detection = detections.front();
            WEB_LOG("detection_id", first_detection.target_id);
            WEB_LOG("detection_confidence", first_detection.confidence);
            const cv::Point2f center =
                (first_detection.corners[0] + first_detection.corners[1] +
                 first_detection.corners[2] + first_detection.corners[3]) * 0.25F;
            WEB_LOG("target_x_px", center.x);
            WEB_LOG("target_y_px", center.y);
        }

        // 2. PnP 位姿解算。PnP 失败时同样保留空 observations，交给 EKF 做纯预测。
        const std::vector<ArmorPose> observations =
            armor_solve(detections, camera, gimbal);
        if (!observations.empty()) {
            const ArmorPose& observation = observations.front();
            WEB_LOG("pnp_reprojection_error", observation.reprojection_error);
            WEB_LOG("pnp_distance_m", cv::norm(observation.position_gimbal_m));

            // 目标 yaw/pitch：板心在云台系里的方位角，也就是"云台该往哪转"。
            // 云台系 x 向前、y 向左、z 向上，与 EKF 的世界系同向，
            // 见 student/include/ArmorEkf.hpp 顶部。所以：
            //   yaw   = atan2(-y, x)           右正（y 朝左，取负才得到右正）
            //   pitch =  atan2(z, hypot(x, y)) 上正
            // 注意这两个是观测值，不要和 EKF 输出的 target_yaw/target_pitch 混看：
            // 那两条是滤波后的云台指令，这里是单帧 PnP 的原始方位。
            const cv::Vec3d& p_gimbal = observation.position_gimbal_m;
            WEB_LOG("pnp_yaw_rad", std::atan2(-p_gimbal[1], p_gimbal[0]));
            WEB_LOG("pnp_pitch_rad",
                    std::atan2(p_gimbal[2], std::hypot(p_gimbal[0], p_gimbal[1])));
        } else {
            // 没有观测时把这一帧显式标成无效，否则网页会一直显示上一帧的旧值，
            // 看着像"还在跟踪"，其实早就丢了目标。
            WEB_LOG("pnp_reprojection_error", -1.0);
            WEB_LOG("pnp_yaw_rad", -1.0);
            WEB_LOG("pnp_pitch_rad", -1.0);
        }

        // 网页上的相机画面就是这一帧：检测、PnP、EKF 后面用的都是同一份像素，
        // 不另开取流。画框放在 armor_detect 之后，检测器看到的始终是原始像素，
        // 不会把自己的框线当成灯条。坐标轴与 rvec/tvec 要等 armor_solve 的三道闸
        // 走完才知道哪条观测留下了，所以整个标注放在解算之后。递交之后这个循环
        // 不再写 pic，编码线程拿的就是这里这个 Mat 头，不是副本。
        annotateDetections(pic, detections, observations, camera);
        WEB_FRAME(pic);

        // 3. 记下时间戳和通信延迟，再调用 EKF。即使 observations 为空也必须调用。
        // 不单独记帧间隔：相机帧率就是同一对 frame_time 差值的倒数，两条曲线互为倒影。
        const double time_stamp =
            std::chrono::duration<double>(frame_time.time_since_epoch()).count();
        translator.message.latency = static_cast<float>(
            std::chrono::duration<double, std::milli>(frame_time - process_start).count());

        const PredictionResult prediction = ekf_predict(observations, gimbal, time_stamp);
        bool prediction_valid = applyPrediction(translator, prediction);

        // 学生算法耗时过长时，丢弃本次结果，避免旧控制器数据触发动作。
        if (using_fallback_frame || !frameIsFresh(received_at)) {
            setSafeOutput(translator);
            prediction_valid = false;
        }

        const auto process_end = std::chrono::steady_clock::now();
        translator.message.latency = static_cast<float>(
            std::chrono::duration<double, std::milli>(process_end - process_start).count());
        WEB_LOG("process_time_ms", translator.message.latency);
        logPrediction(prediction_valid ? prediction : PredictionResult{});
        // 传采集时刻，不是帧间隔：缺观测的帧要把时间摊到下一帧的分母里。
        logMeasYawRate(observations, gimbal, time_stamp);
        // 放在后面：obs 的序号来自本帧的 observations，而换算世界坐标要用的
        // gimbal_yaw 由 logMeasYawRate 打点，同一条 SSE 消息里先到云台角。
        logObservations(observations);

        // 4. 将结果写回原 64 字节结构体。
        writeFrame(*serialPort, translator);
    }

    STOP_WEB_SERVER();
    return nullptr;
}
