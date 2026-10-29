#pragma once

#include <chrono>

#include <opencv2/core.hpp>

// 海康 MVS 相机的取流封装。

// 约定：任何失败都只返回 false，绝不终止进程。上层的双线程框架
// 靠这个 false 走"空帧继续安全回包"的分支，相机掉线不该把程序带走。

// 非线程安全：一个实例只服务一个调用者（现在就是操作线程）。
class HikCamera {
public:
    // 取一帧。成功时 pic 是非空的 CV_8UC3 BGR；失败时 pic 被清空并返回 false。
    // 掉线后不会一直拿失败：每 kReopenIntervalMs 试一次重开，插回来就恢复。
    bool grab(cv::Mat& pic);

    // 释放相机（停流 + 关设备 + 销毁句柄）。可重复调用，没开也安全。
    // 只放设备，不动 SDK，掉了线还能重开。
    void close();

    // 放设备 + 反初始化 SDK。只能在退出钩子里调，
    // 调完这个实例就不能再用了。给 std::atexit 用的，别的地方别调。
    void releaseAll();

    ~HikCamera();

    HikCamera() = default;
    HikCamera(const HikCamera&) = delete;
    HikCamera& operator=(const HikCamera&) = delete;

private:
    // 第一次调用时枚举 -> 打开 -> 锁参数 -> 开始取流；之后直接返回 true。
    bool openOnce();

    void* handle_ = nullptr;   // MV_CC_CreateHandle 出来的设备句柄
    bool  sdkReady_ = false;   // MV_CC_Initialize 是否已成功（进程级，只做一次）
    int   failStreak_ = 0;     // 连续取流失败次数，到阈值就重开相机

    // 重开节流。设备不在时上层的循环还在按帧调 grab，每次枚举设备既慢又刷屏。
    std::chrono::steady_clock::time_point nextOpenAttempt_{};
    // 已经报过"离线"，只在与在线状态切换时打印，避免每秒一行。
    bool offlineReported_ = false;
};
