// ekf_predict 适配器的离线烟测。
//
// 适配器把 TargetSelector 与 ArmorEkf 串在一起，并持有跨帧 static 状态，
// 直接测类的那批用例走不到这一层。这里只钉一件事：选择器换簇不等于跟踪对象
// 换了，不能因此把滤波器整个重置。
// 运行：ctest -R ekf_adapter

#include "StudentTasks.hpp"

#include "TargetSelector.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

using namespace armor;

namespace {

ArmorPose plate(const Eigen::Vector3d& p_gimbal, int target_id, double offset_px) {
    ArmorPose pose;
    pose.position_gimbal_m = cv::Vec3d(p_gimbal.x(), p_gimbal.y(), p_gimbal.z());
    pose.detection.target_id = target_id;
    pose.center_offset_px = offset_px;
    // 第四条纹观测是板面法向偏航角，这里不需要，关掉它只留三维位置观测。
    pose.armor_yaw = std::numeric_limits<double>::quiet_NaN();
    return pose;
}

void testSwitchedClusterKeepsFilterState() {
    // 同一辆车的两块板，间距 0.60 m 越过 cluster_link_m 0.55，各自成簇。
    // 板 A 在画面里偏 300 px，板 B 居中，所以 B 的分数一直更高。
    const ArmorPose a = plate(Eigen::Vector3d(3.0, 0.30, 0.0), 3, 300.0);
    const ArmorPose b = plate(Eigen::Vector3d(3.0, -0.30, 0.0), 3, 0.0);
    const std::vector<ArmorPose> only_a{a};
    const std::vector<ArmorPose> both{a, b};

    // 第一帧只看到板 A，选它当跟踪对象并开始初始化；后两帧两块板都在画面里，
    // 选择器仍然只把 A 喂给滤波器，三帧攒够 init_frames。
    GimbalState gimbal;
    PredictionResult result = ekf_predict(only_a, gimbal, 0.000);
    result = ekf_predict(both, gimbal, 0.016);
    result = ekf_predict(both, gimbal, 0.032);
    assert(result.state != TrackingState::NoTarget);
    assert(result.distance_m > 0.0F);

    // 前提：选择器确实会在下一帧换簇。这里用一个镜像选择器走同样的输入，
    // 免得用例因为"根本没换簇"而白白通过。
    TargetSelector mirror;
    mirror.select(only_a, 0.0, 0.000);
    mirror.select(both, 0.0, 0.016);
    mirror.select(both, 0.0, 0.032);
    const TargetSelection switched = mirror.select(both, 0.0, 0.048);
    assert(switched.switched);
    assert(switched.observation_indices.size() == 1);
    assert(switched.observation_indices[0] == 1);

    // 换簇那一帧：还是同一辆车，滤波器不许被重置成 NoTarget。
    result = ekf_predict(both, gimbal, 0.048);
    assert(result.state != TrackingState::NoTarget);
    assert(result.distance_m > 0.0F);
}

}  // namespace

int main() {
    testSwitchedClusterKeepsFilterState();
    std::printf("switched_cluster_keeps_state ok\n");
    return 0;
}
