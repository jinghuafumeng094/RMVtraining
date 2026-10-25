#pragma once

#include <string>

// web_debug 的 CMake 目标会导出 WEB_DEBUG。未启用时，文件末尾的宏会变为空操作，
// 同时不会求值传入的参数。
#ifdef WEB_DEBUG

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "json.hpp"

namespace httplib {
class Server;
}

class WebLogger {
public:
    static WebLogger& getInstance();

    // 在后台线程启动 HTTP 服务。服务运行期间重复调用不会产生任何操作；
    // 默认监听本机地址。
    void start(int port = 8080, const std::string& host = "127.0.0.1");

    // 停止监听，并等待服务器线程和正在进行的 SSE 流退出。
    // 可以重复调用，之后也可以再次调用 start() 启动服务。
    void stop();

    // 向有容量上限的广播缓冲区加入一个数值样本。记录日志不会等待网页读取样本，
    // 内存占用始终受到限制。
    void log(const std::string& key, double value);

    // 交出一帧 BGR 图像给网页的 /frame 通道。只换 cv::Mat 头不拷像素，
    // 编码在内部线程做，调用方不会被 JPEG 编码卡住。
    // 调用方必须保证这帧的像素之后不被原地改写：本函数的实现依赖 cv::Mat
    // 的引用计数让内存活到编码线程用完为止，但共享像素不等于共享写入权。
    void publishImage(const cv::Mat& frame);

    bool isRunning() const noexcept;

    // 返回不会改变内部状态的快照，供诊断和测试使用。
    nlohmann::json getBufferJSON() const;

private:
    struct LogData {
        std::string key;
        double value;
        std::int64_t timestamp;
        std::uint64_t sequence;
    };

    // 所有客户端共享这段历史记录，但通过各自独立的序号读取。
    static constexpr std::size_t kBufferCapacity = 10000;

    // 画面通道的上限。1440x1080 编一张 JPEG 实测约 10 ms，跟主循环同步做会
    // 吃掉 165 fps 里的大半帧预算，所以才单开线程并限速到这里。
    static constexpr int kImageFps = 20;
    static constexpr int kImageQuality = 70;

    WebLogger();
    ~WebLogger();
    WebLogger(const WebLogger&) = delete;
    WebLogger& operator=(const WebLogger&) = delete;

    void configureServer(httplib::Server& server);
    void serverLoop();
    void encoderLoop();

    std::atomic<bool> running_{false};
    mutable std::mutex lifecycle_mutex_;
    std::thread server_thread_;
    std::thread encoder_thread_;
    std::unique_ptr<httplib::Server> server_;

    mutable std::mutex buffer_mutex_;
    std::condition_variable data_cv_;
    std::deque<LogData> buffer_;
    std::uint64_t next_sequence_{1};

    // 最新一帧只保留头，序号用来判断有没有新帧。两把锁从不嵌套，
    // 编码线程先读 image_mutex_ 再单独读 encoded_sequence_。
    mutable std::mutex image_mutex_;
    std::condition_variable image_cv_;
    cv::Mat latest_image_;
    std::uint64_t image_sequence_{0};

    mutable std::mutex encoded_mutex_;
    std::condition_variable encoded_cv_;
    std::vector<unsigned char> encoded_jpeg_;
    std::atomic<std::uint64_t> encoded_sequence_{0};

    // 没有客户端时不编码：调参页面关掉以后 CPU 应该是零。
    std::atomic<int> image_clients_{0};
};

#define START_WEB_SERVER(port) WebLogger::getInstance().start(port)
#define STOP_WEB_SERVER() WebLogger::getInstance().stop()
#define WEB_LOG(key, value) WebLogger::getInstance().log((key), (value))
#define WEB_FRAME(image) WebLogger::getInstance().publishImage((image))

#else

// 这些宏会有意忽略传入参数。例如关闭调试时，WEB_LOG("fps", expensive())
// 不会调用 expensive()。
#define START_WEB_SERVER(port) ((void)0)
#define STOP_WEB_SERVER() ((void)0)
#define WEB_LOG(key, value) ((void)0)
#define WEB_FRAME(image) ((void)0)

#endif // WEB_DEBUG
