// Web Debug 的画面通道。
//
// 学长要的是相机画面和曲线同时在网页上看。实现上最容易出错的两处：
//   1. 网页拿到的不是主循环正在处理的那一帧（另开一路取流，或者停在旧帧）；
//   2. HTTP 媒体类型写错，<img> 标签根本不刷新。
// 所以这里绕过 httplib 客户端，直接用裸 socket 读浏览器会收到的那串字节，
// 按 multipart 切出 JPEG 再解码，跟 publishImage() 交进去的图逐像素比。
//
// 测试不碰相机，图是合成的。

#include "WebLogger.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kFirstPort = 18081;
constexpr int kPortTries = 8;
constexpr int kReadTimeoutS = 5;

// 一条长连接。反复 readUntil 就能拿到后续的帧，和浏览器的 <img> 行为一致。
class StreamReader {
public:
    StreamReader(int port, const std::string& path) {
        timeval timeout{};
        timeout.tv_sec = kReadTimeoutS;

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<std::uint16_t>(port));
        ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

        for (int attempt = 0; attempt < 100 && fd_ < 0; ++attempt) {
            const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (fd < 0) {
                return;
            }
            ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
                fd_ = fd;
            } else {
                ::close(fd);
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }
        if (fd_ < 0) {
            return;
        }

        const std::string request =
            "GET " + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n";
        if (::send(fd_, request.data(), request.size(), 0) < 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    ~StreamReader() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    StreamReader(const StreamReader&) = delete;
    StreamReader& operator=(const StreamReader&) = delete;

    bool valid() const { return fd_ >= 0; }

    bool readUntil(bool (*done)(const std::string&)) {
        std::vector<char> buffer(64 * 1024);
        while (!done(data_)) {
            const ssize_t got = ::recv(fd_, buffer.data(), buffer.size(), 0);
            if (got <= 0) {
                return false;
            }
            data_.append(buffer.data(), static_cast<std::size_t>(got));
        }
        return true;
    }

    const std::string& data() const { return data_; }

private:
    int fd_ = -1;
    std::string data_;
};

// 数一数收了多少段完整 JPEG。multipart 分隔符和 chunk 长度行都在 SOI 之前，
// SOI 到 EOI 之间没有别的字节，所以按标记切就是准的。
std::size_t jpegCount(const std::string& raw) {
    std::size_t count = 0;
    std::size_t from = 0;
    while (true) {
        const std::size_t start = raw.find("\xFF\xD8", from);
        if (start == std::string::npos) {
            return count;
        }
        const std::size_t end = raw.find("\xFF\xD9", start + 2);
        if (end == std::string::npos) {
            return count;
        }
        ++count;
        from = end + 2;
    }
}

std::vector<unsigned char> nthJpeg(const std::string& raw, std::size_t index) {
    std::size_t from = 0;
    for (std::size_t seen = 0; seen <= index; ++seen) {
        const std::size_t start = raw.find("\xFF\xD8", from);
        if (start == std::string::npos) {
            return {};
        }
        const std::size_t end = raw.find("\xFF\xD9", start + 2);
        if (end == std::string::npos) {
            return {};
        }
        if (seen == index) {
            return std::vector<unsigned char>(
                raw.begin() + static_cast<std::ptrdiff_t>(start),
                raw.begin() + static_cast<std::ptrdiff_t>(end + 2));
        }
        from = end + 2;
    }
    return {};
}

bool firstJpegComplete(const std::string& raw) { return jpegCount(raw) >= 1; }
bool twoJpegsComplete(const std::string& raw) { return jpegCount(raw) >= 2; }
bool hasMetricPoint(const std::string& raw) {
    return raw.find("selftest") != std::string::npos;
}

bool pixelNear(const cv::Vec3b& actual, const cv::Scalar& expected, int tolerance) {
    for (int channel = 0; channel < 3; ++channel) {
        const int diff = actual[channel] - static_cast<int>(expected[channel]);
        if (diff > tolerance || diff < -tolerance) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    auto& logger = WebLogger::getInstance();

    // 两帧只差底色，用来确认网页拿到的是"最新一帧"而不是重播第一帧。
    cv::Mat frame_a(120, 160, CV_8UC3, cv::Scalar(30, 120, 210));
    cv::rectangle(frame_a, cv::Rect(20, 20, 80, 60), cv::Scalar(10, 10, 230), cv::FILLED);
    cv::Mat frame_b(120, 160, CV_8UC3, cv::Scalar(200, 60, 40));

    int port = 0;
    for (int attempt = 0; attempt < kPortTries && port == 0; ++attempt) {
        logger.start(kFirstPort + attempt, "127.0.0.1");
        if (logger.isRunning()) {
            port = kFirstPort + attempt;
        }
    }
    assert(port != 0);

    // 先交一帧再连：网页刚打开就该立刻有图，不用等下一帧。
    logger.publishImage(frame_a);

    StreamReader stream(port, "/frame");
    assert(stream.valid());
    assert(stream.readUntil(firstJpegComplete));

    const std::string& raw = stream.data();
    if (raw.find("HTTP/1.1 200") == std::string::npos ||
        raw.find("multipart/x-mixed-replace") == std::string::npos ||
        raw.find("Content-Type: image/jpeg") == std::string::npos) {
        std::fprintf(stderr, "画面响应头不对，前 200 字节：\n%.200s\n", raw.c_str());
    }
    assert(raw.find("HTTP/1.1 200") != std::string::npos);
    assert(raw.find("multipart/x-mixed-replace") != std::string::npos);
    assert(raw.find("Content-Type: image/jpeg") != std::string::npos);

    const cv::Mat decoded_a = cv::imdecode(nthJpeg(raw, 0), cv::IMREAD_COLOR);
    assert(decoded_a.rows == frame_a.rows && decoded_a.cols == frame_a.cols);
    assert(pixelNear(decoded_a.at<cv::Vec3b>(5, 5), cv::Scalar(30, 120, 210), 12));
    assert(pixelNear(decoded_a.at<cv::Vec3b>(40, 60), cv::Scalar(10, 10, 230), 40));

    // 同一条连接继续收：交第二帧，页面上必须换图。
    logger.publishImage(frame_b);
    assert(stream.readUntil(twoJpegsComplete));
    const cv::Mat decoded_b = cv::imdecode(nthJpeg(stream.data(), 1), cv::IMREAD_COLOR);
    assert(decoded_b.rows == frame_b.rows && decoded_b.cols == frame_b.cols);
    assert(pixelNear(decoded_b.at<cv::Vec3b>(5, 5), cv::Scalar(200, 60, 40), 12));

    // 数值通道不受影响：/data 仍然是 SSE。
    logger.log("selftest", 1.0);
    StreamReader sse(port, "/data");
    assert(sse.valid());
    assert(sse.readUntil(hasMetricPoint));
    assert(sse.data().find("text/event-stream") != std::string::npos);

    logger.stop();
    std::printf("web_frame_stream: ok\n");
    return 0;
}
