#include "HikCamera.hpp"

#include "MvCameraControl.h"   // 海康 MVS SDK

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

// ===========================================================================
// 相机参数 —— 要调就调这里
// ===========================================================================
// 曝光和增益都**不用自动**。自动曝光的"目标亮度"是相机自己定的，灯一变
// 整幅图的灰度跟着变 —— 下游那些固定的颜色阈值会被它整死。
// （现场第一批图就是这么废掉的：AE 顶格 + AWB 还在 Continuous。）
constexpr float    kExposureUs     = 5000.0F;   // 曝光时间，微秒
constexpr float    kGainDb         = 0.0F;      // 增益，dB
constexpr unsigned kGrabTimeoutMs  = 1000;       // 单帧等待上限
constexpr int      kFailStreakLimit = 5;         // 连败几次就重开相机

// MVS 的错误码是十六进制位掩码，印成十进制看不出东西来。
void reportError(const char* what, int nRet) {
    std::cerr << "[camera] " << what << " 失败，nRet = 0x"
              << std::hex << nRet << std::dec << std::endl;
}

// 当前活着的相机（最多一台，进程内只有一个 get_pic）。退出钩子靠它找设备。
HikCamera* g_liveCamera = nullptr;

// 进程退出时释放相机。**注册点在 StartGrabbing 之后**，这一点是全部关键，
// 见 openOnce() 里的说明。
void releaseCameraAtExit() {
    if (g_liveCamera != nullptr) {
        g_liveCamera->releaseAll();
    }
}

}  // namespace

// ⚠️ 这里踩过一个坑，别再改回去。事情的经过：
//
// 第一版把 SDK 的初始化/反初始化拆给两个函数局部 static 管（"退出时自动
// 收尾"）。函数局部 static 按**构造完成顺序的逆序**析构，于是退出时钩子先跑
// （MV_CC_Finalize），camera 后跑（StopGrabbing 一个已死的句柄）——
// 段错误，栈顶在 MVS 自己的线程里（libMvUsb3vTL.so），完全不像是这一层的错。
//
// 改成"一个对象管到底"之后**仍然崩**，这才挖到底：MVS 自己在
// Initialize/OpenDevice 那一串调用里注册了退出清理。atexit 处理器按
// **注册的逆序**执行 —— 它注册得更早，就跑得更早；等轮到这一层时，
// 设备早被它释放了。实测那种时序下：
//     StopGrabbing  -> 0x80000000（MV_E_HANDLE，无效句柄）
//     CloseDevice   -> 0x0
//     DestroyHandle -> **段错误**（在已释放的内存上再释放一次，double free）
// 且时崩时不崩，是最难查的一类。
//
// 解法：把本层的注册点推到 StartGrabbing **之后**，比 MVS 的清理注册得晚，
// 于是跑得比它早——那时设备还活着。实测四个调用全部 0x0，稳定干净退出。
// 所以：**别把 std::atexit 那一行挪到 StartGrabbing 之前。**
void HikCamera::releaseAll() {
    close();
    if (sdkReady_) {
        MV_CC_Finalize();
        sdkReady_ = false;
    }
    if (g_liveCamera == this) {
        g_liveCamera = nullptr;
    }
}

// 正常退出时 releaseAll() 已经在退出钩子里跑过了，这里就是空转；
// 万一对象被提前销毁（不是退出路径），这里兜底。
//
// ⚠️ 全程只在**退出阶段之外**碰 SDK：退出阶段由 releaseCameraAtExit 负责。
// ⚠️ SIGINT（Ctrl-C）下静态对象析构**不会**执行；但 main.cpp 的信号处理只是
// 置标志、让线程正常收尾后 return，所以仍然会走完整的退出流程。
HikCamera::~HikCamera() {
    releaseAll();
}

void HikCamera::close() {
    if (handle_ == nullptr) {
        return;
    }
    // 顺序不能反：先停流，再关设备，最后销毁句柄。
    MV_CC_StopGrabbing(handle_);
    MV_CC_CloseDevice(handle_);
    MV_CC_DestroyHandle(handle_);
    handle_ = nullptr;
    failStreak_ = 0;
}

bool HikCamera::openOnce() {
    if (handle_ != nullptr) {
        return true;
    }

    // MV_CC_Initialize 是**进程级**的（不像句柄那样一台设备一份）。
    // 用成员变量记，而不是它自己的 static —— 这样"初始化"和"反初始化"
    // 都归同一个对象管，退出顺序不可能搞错（见析构函数的说明）。
    if (!sdkReady_) {
        const int nRet = MV_CC_Initialize();
        if (nRet != MV_OK) {
            reportError("MV_CC_Initialize", nRet);
            return false;
        }
        sdkReady_ = true;
    }

    // ---- 枚举 ----
    MV_CC_DEVICE_INFO_LIST devList;
    std::memset(&devList, 0, sizeof(devList));
    int nRet = MV_CC_EnumDevices(MV_USB_DEVICE | MV_GIGE_DEVICE, &devList);
    if (nRet != MV_OK) {
        reportError("MV_CC_EnumDevices", nRet);
        return false;
    }
    if (devList.nDeviceNum == 0) {
        std::cerr << "[camera] 未枚举到相机。USB 未连接，"
                     "或设备被 MVS 客户端占用\n";
        return false;
    }
    if (devList.nDeviceNum > 1) {
        std::cerr << "[camera] 枚举到 " << devList.nDeviceNum
                  << " 台相机，用第 0 台\n";
    }

    // ---- 建句柄 → 打开 ----
    nRet = MV_CC_CreateHandle(&handle_, devList.pDeviceInfo[0]);
    if (nRet != MV_OK) {
        reportError("MV_CC_CreateHandle", nRet);
        handle_ = nullptr;
        return false;
    }

    nRet = MV_CC_OpenDevice(handle_);
    if (nRet != MV_OK) {
        reportError("MV_CC_OpenDevice", nRet);
        MV_CC_DestroyHandle(handle_);
        handle_ = nullptr;
        return false;
    }

    // ---- 锁参数：先关自动，再写值 ----
    // 这个顺序不能反：ExposureAuto 还开着的时候写 ExposureTime，会被自动曝光
    // 立刻覆盖回去，而且**不报错**。只看到"设了没用"。
    MV_CC_SetEnumValue(handle_, "ExposureAuto", 0);       // 0 = Off
    MV_CC_SetFloatValue(handle_, "ExposureTime", kExposureUs);
    MV_CC_SetEnumValue(handle_, "GainAuto", 0);           // 0 = Off
    MV_CC_SetFloatValue(handle_, "Gain", kGainDb);
    MV_CC_SetEnumValue(handle_, "BalanceWhiteAuto", 0);   // 0 = Off，别让 AWB 把红推成橙

    // 自由运行，不要外部触发。
    MV_CC_SetEnumValue(handle_, "TriggerMode", 0);

    // ---- 开始取流 ----
    nRet = MV_CC_StartGrabbing(handle_);
    if (nRet != MV_OK) {
        reportError("MV_CC_StartGrabbing", nRet);
        MV_CC_CloseDevice(handle_);
        MV_CC_DestroyHandle(handle_);
        handle_ = nullptr;
        return false;
    }

    // ---- 注册退出钩子（位置很重要，别往上挪）----
    // 必须在 StartGrabbing **之后**注册：atexit 按注册的逆序执行，注册得越晚
    // 跑得越早。MVS 自己的退出清理是在 Initialize/OpenDevice 那一串里注册的，
    // 比这里早 —— 只有注册点排在它后面，本层的清理才会赶到它前面，
    // 那时设备还活着。顺序错了就是 double free 段错误，详见 releaseAll()。
    // 每次成功开流都注册一次（不是只注册一次）：注册得越晚跑得越早，
    // 所以重开相机之后重新注册，才能继续压过 SDK 新注册的那份。
    // 重复注册无害 —— releaseAll() 是幂等的，第一次跑完 g_liveCamera 就空了。
    std::atexit(&releaseCameraAtExit);
    g_liveCamera = this;

    std::cout << "[camera] 已打开。曝光 " << kExposureUs << " us，增益 "
              << kGainDb << " dB，自动曝光/增益/白平衡已关闭\n";
    failStreak_ = 0;
    return true;
}

bool HikCamera::grab(cv::Mat& pic) {
    // 无论成败都先清空：调用方拿到的永远不是上一帧。
    pic.release();

    if (!openOnce()) {
        return false;
    }

    // ---- 拿一帧的缓冲（零拷贝，用完必须还） ----
    MV_FRAME_OUT frame;
    std::memset(&frame, 0, sizeof(frame));
    int nRet = MV_CC_GetImageBuffer(handle_, &frame, kGrabTimeoutMs);
    if (nRet != MV_OK) {
        ++failStreak_;
        std::cerr << "[camera] 取帧失败 nRet = 0x" << std::hex << nRet << std::dec
                  << "，连续 " << failStreak_ << " 次\n";
        if (failStreak_ >= kFailStreakLimit) {
            std::cerr << "[camera] 连续失败 " << kFailStreakLimit
                      << " 次，重开相机\n";
            close();
        }
        return false;
    }

    const MV_FRAME_OUT_INFO_EX& info = frame.stFrameInfo;

    // 用 nExtend* 而不是 nWidth/nHeight：前者是"真正的行宽/列高"，
    // Bayer 图可能有对齐填充，用错了整幅错位。1440 恰好是 4 的倍数，
    // 现在两者数值相同 —— 但别把这个巧合写进代码。
    const unsigned int w = info.nExtendWidth;
    const unsigned int h = info.nExtendHeight;

    // ---- 像素格式转换：交给 SDK，不用 cv::cvtColor ----
    // MVS 的 "BayerRG8" 与 OpenCV 的 COLOR_BayerRG 是不是同一个相位，官方没说。
    // 猜错了 R/B 会对调 —— **不报错**，只会让红蓝队判反，是最难查的那种错。
    std::vector<unsigned char> bgr(static_cast<size_t>(w) * h * 3);

    MV_CC_PIXEL_CONVERT_PARAM_EX cvt;
    std::memset(&cvt, 0, sizeof(cvt));
    cvt.nWidth         = w;
    cvt.nHeight        = h;
    cvt.enSrcPixelType = info.enPixelType;
    cvt.pSrcData       = frame.pBufAddr;
    cvt.nSrcDataLen    = info.nFrameLen;
    cvt.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
    cvt.pDstBuffer     = bgr.data();
    cvt.nDstBufferSize = static_cast<unsigned int>(bgr.size());

    nRet = MV_CC_ConvertPixelTypeEx(handle_, &cvt);

    // 不管转换成功还是失败，这帧的 SDK 缓冲都要还回去。忘了还，
    // 缓冲池几十帧就耗干净，之后再也取不到图。
    MV_CC_FreeImageBuffer(handle_, &frame);

    if (nRet != MV_OK) {
        reportError("MV_CC_ConvertPixelTypeEx", nRet);
        std::cerr << "[camera] 源像素格式 = 0x" << std::hex
                  << static_cast<unsigned int>(info.enPixelType) << std::dec
                  << "，预期 BayerRG8 = 0x1080009\n";
        return false;
    }
    if (w == 0 || h == 0) {
        std::cerr << "[camera] 转换后尺寸为 0\n";
        return false;
    }

    // clone 是必须的：bgr 是栈上临时缓冲，出了这个函数就没了。
    pic = cv::Mat(static_cast<int>(h), static_cast<int>(w), CV_8UC3,
                  bgr.data())
              .clone();
    failStreak_ = 0;
    return true;
}
