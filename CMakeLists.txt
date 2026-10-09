cmake_minimum_required(VERSION 3.16)
project(RMVTraining LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_WEB_DEBUG "Build the scalar Web Debug dashboard" ON)
option(BUILD_TESTING "Build the starter-project tests" ON)

find_package(Eigen3 REQUIRED)
# core    : cv::Mat 等基础类型
# imgproc : cv::cvtColor 等图像处理
# videoio : cv::VideoCapture / VideoWriter 取流
# calib3d : cv::solvePnP 位姿解算（第四课用）
find_package(OpenCV 4 REQUIRED COMPONENTS 
	core imgproc videoio calib3d
	objdetect
	ml
	imgcodecs
)
find_package(Threads REQUIRED)

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
    student/src/ArmorDetector.cpp
    student/src/DigitClassifier.cpp
)
target_include_directories(training_core
    PUBLIC
        include
        student/include
        serialPort/include
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(training_core PUBLIC serialPort ${OpenCV_LIBS})
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
    add_executable(test_digit_classifier tests/test_digit_classifier.cpp)
    target_link_libraries(test_digit_classifier PRIVATE training_core)
    # 留出集在源码树里，不在构建目录里。ctest 的工作目录是构建目录，
    # 用相对路径会找不到，所以把绝对路径编进二进制。
    target_compile_definitions(test_digit_classifier PRIVATE
        ARMOR_HOLDOUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/digit_holdout")
    add_test(NAME digit_classifier COMMAND test_digit_classifier)
    add_test(NAME cli_help COMMAND rmv_training --help)
endif()
