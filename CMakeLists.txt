cmake_minimum_required(VERSION 3.16)
project(RMVTraining LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_WEB_DEBUG "Build the scalar Web Debug dashboard" ON)
option(BUILD_TESTING "Build the starter-project tests" ON)

find_package(Eigen3 REQUIRED)
# core     : cv::Mat 等基础类型
# imgproc  : cv::cvtColor / resize 等图像处理
# calib3d  : cv::solvePnP 位姿解算（第四课用）
# objdetect: cv::HOGDescriptor（数字分类的特征）
# ml       : cv::ml::SVM（数字分类的分类器）
# imgcodecs: cv::imread / imwrite（离线工具读写图片）
#
# 没有 videoio：取流走海康 MVS SDK，不再用 cv::VideoCapture，
# 全仓已无 videoio 符号（见 student/src/HikCamera.cpp）。
find_package(OpenCV 4 REQUIRED COMPONENTS
	core imgproc calib3d
	objdetect
	ml
	imgcodecs
)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# 海康 MVS 相机 SDK —— get_pic() 靠它取流（student/src/HikCamera.cpp）。
#
# 路径写死 /opt/MVS 而不是 find_package：SDK 不自带 CMake 配置文件，
# 没有 config mode 可找。装在别处就 -DMVS_ROOT=<安装根目录>。
# ---------------------------------------------------------------------------
set(MVS_ROOT "/opt/MVS" CACHE PATH "海康 MVS SDK 安装根目录")

find_path(MVS_INCLUDE_DIR MvCameraControl.h
    PATHS "${MVS_ROOT}/include" NO_DEFAULT_PATH)
find_library(MVS_LIBRARY MvCameraControl
    PATHS "${MVS_ROOT}/lib/64" NO_DEFAULT_PATH)

if(NOT MVS_INCLUDE_DIR OR NOT MVS_LIBRARY)
    message(FATAL_ERROR
        "找不到海康 MVS SDK。\n"
        "  找过: ${MVS_ROOT}/include 和 ${MVS_ROOT}/lib/64\n"
        "  当前: MVS_INCLUDE_DIR=${MVS_INCLUDE_DIR} MVS_LIBRARY=${MVS_LIBRARY}\n"
        "装着 SDK 的话，用 -DMVS_ROOT=<安装根目录> 重新 configure。")
endif()

add_library(hik_mvs INTERFACE)
target_include_directories(hik_mvs INTERFACE "${MVS_INCLUDE_DIR}")
target_link_libraries(hik_mvs INTERFACE "${MVS_LIBRARY}")

# 把库目录烤进可执行文件的 rpath —— 官方 Sample 的 Makefile 也这么干
# （-Wl,-rpath=$(MVCAM_COMMON_RUNENV)/64）。这样跑自己的程序**完全不依赖
# MVCAM_* 环境变量**：那四个只写在 ~/.bashrc 里，而登录 shell 是 zsh，
# 干净终端里它们是空的 —— 靠环境变量就等于靠运气。
get_filename_component(MVS_LIB_DIR "${MVS_LIBRARY}" DIRECTORY)
target_link_options(hik_mvs INTERFACE "-Wl,-rpath,${MVS_LIB_DIR}")

add_subdirectory(serialPort)
if(ENABLE_WEB_DEBUG)
    add_subdirectory(web_debug)
endif()

add_library(algorithm_support STATIC
    armor/src/BulletModel.cpp
    armor/src/KalmanFilter.cpp
)
target_include_directories(algorithm_support PUBLIC armor/include)
target_link_libraries(algorithm_support PUBLIC Eigen3::Eigen)

add_library(training_core STATIC
    src/AppSupport.cpp
    src/AutoAimTypes.cpp
    student/src/StudentTasks.cpp
    student/src/HikCamera.cpp
    student/src/ArmorDetector.cpp
    student/src/DigitClassifier.cpp
    student/src/ArmorEkf.cpp
)
target_include_directories(training_core
    PUBLIC
        include
        student/include
        serialPort/include
        ${OpenCV_INCLUDE_DIRS}
)
# hik_mvs 用 PUBLIC：链了 training_core 的目标都会被 get_pic 拖上
# libMvCameraControl，头文件和 -rpath 得一路传下去。
# algorithm_support 也放 PUBLIC：ArmorEkf 用 Eigen 算矩阵，弹道飞行时间
# 复用 armor/ 里的 Bullet_Model，链这一个目标就把 Eigen 的头、
# armor/include 目录和 Bullet_Model 的实现一起带过来。
target_link_libraries(training_core PUBLIC serialPort hik_mvs algorithm_support ${OpenCV_LIBS})
# 数字分类模型的绝对路径。编进二进制而不是运行时拼相对路径 ——
# 现场从来不及猜"当前工作目录是哪儿"。
target_compile_definitions(training_core PRIVATE
    ARMOR_SVM_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/armor_digit_svm.xml")

add_executable(rmv_training main.cpp)
target_include_directories(rmv_training PRIVATE web_debug/include)
target_link_libraries(rmv_training
    PRIVATE
        training_core
        algorithm_support
        Threads::Threads
)
if(ENABLE_WEB_DEBUG)
    target_link_libraries(rmv_training PRIVATE web_debug)
endif()

# ---------------------------------------------------------------------------
# 离线工具：训练 / 抽留出集。
#
# 这两个**故意不链 training_core**：只借 DigitClassifier.cpp 一份源码
# （训练和推理共用同一个 describe()，从物理上不可能分叉），
# 链 OpenCV 就够了。链了 training_core 会把 serialPort、相机那套全拖进来，
# 训练机上不一定有串口设备。
# ---------------------------------------------------------------------------
add_executable(train_digit_svm
    tools/train_digit_svm.cpp
    student/src/DigitClassifier.cpp
)
target_include_directories(train_digit_svm PRIVATE
    student/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(train_digit_svm PRIVATE ${OpenCV_LIBS})

add_executable(make_digit_holdout
    tools/make_digit_holdout.cpp
    student/src/DigitClassifier.cpp
)
target_include_directories(make_digit_holdout PRIVATE
    student/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(make_digit_holdout PRIVATE ${OpenCV_LIBS})

# ---------------------------------------------------------------------------
# 抓一帧存盘：验证 get_pic() 本身（相机 SDK 调通了没有）。
#
# 它**不需要 config/camera.yaml** —— 取流和标定是两回事，
# 内参外参还没齐不影响相机能不能出图。所以这是相机到手后第一件能验的事。
#
# 这个要链 training_core：验的就是 get_pic 这个真入口
# （StudentTasks.cpp 里那个 static HikCamera），不是绕过它直接调 HikCamera。
# ---------------------------------------------------------------------------
add_executable(grab_one tools/grab_one.cpp)
target_link_libraries(grab_one PRIVATE training_core)

if(BUILD_TESTING)
    enable_testing()
    add_executable(test_scaffold tests/test_scaffold.cpp)
    target_link_libraries(test_scaffold PRIVATE training_core)
    add_test(NAME scaffold_safety COMMAND test_scaffold)
    add_executable(test_armor_detect tests/test_armor_detect.cpp)
    target_link_libraries(test_armor_detect PRIVATE training_core)
    add_test(NAME armor_detect_geometry COMMAND test_armor_detect)
    add_executable(test_armor_pipeline tests/test_armor_pipeline.cpp)
    target_link_libraries(test_armor_pipeline PRIVATE training_core)
    add_test(NAME armor_detect_pipeline COMMAND test_armor_pipeline)
    add_executable(test_armor_solve tests/test_armor_solve.cpp)
    target_link_libraries(test_armor_solve PRIVATE training_core)
    add_test(NAME armor_solve_pnp COMMAND test_armor_solve)
    add_executable(test_digit_classifier tests/test_digit_classifier.cpp)
    target_link_libraries(test_digit_classifier PRIVATE training_core)
    # 留出集在源码树里，不在构建目录里。ctest 的工作目录是构建目录，
    # 用相对路径会找不到，所以把绝对路径编进二进制。
    target_compile_definitions(test_digit_classifier PRIVATE
        ARMOR_HOLDOUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/digit_holdout")
    add_test(NAME digit_classifier COMMAND test_digit_classifier)
    add_executable(test_ekf_predict tests/test_ekf_predict.cpp)
    target_link_libraries(test_ekf_predict PRIVATE training_core)
    # 测试全是裸 assert，NDEBUG 会把它们整个编译掉，ctest 就变成假绿。
    target_compile_options(test_ekf_predict PRIVATE -UNDEBUG)
    add_test(NAME ekf_predict COMMAND test_ekf_predict)
    add_test(NAME cli_help COMMAND rmv_training --help)
endif()
