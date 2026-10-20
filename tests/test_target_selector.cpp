// 多目标筛选的离线烟测。
//
// 一帧画面里出现两辆车时，全帧观测不能直接灌进同一个 ArmorEkf：滤波器只有一套
// 状态、一个关联器，两辆车的板会互相抢槽位。这个用例把「先选车、再滤波」这一层
// 单独测掉：聚簇、两种打分模式、迟滞、遮挡兜底、云台转动下的跨帧认同。
// 运行：ctest -R target_selector

#include "TargetSelector.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace armor;

namespace {

constexpr double kPi = 3.14159265358979323846;

// 一辆车的 4 块板，绕着 world_center 按 90 度排布，半径 radius。
// 编号统一给 target_id，像素偏置统一给 offset_px，便于按分数推期望值。
std::vector<ArmorPose> vehicle(const Eigen::Vector3d& world_center, double gimbal_yaw,
                               int target_id, double offset_px, double radius = 0.25,
                               int plate_count = 4) {
    std::vector<ArmorPose> poses;
    const Eigen::Matrix3d to_gimbal = ArmorEkf::worldToGimbal(gimbal_yaw);
    for (int i = 0; i < plate_count; ++i) {
        const double psi = static_cast<double>(i) * 0.5 * kPi;
        const Eigen::Vector3d p_world =
            world_center + Eigen::Vector3d(radius * std::cos(psi), radius * std::sin(psi), 0.0);
        const Eigen::Vector3d p_gimbal = to_gimbal * p_world;
        ArmorPose pose;
        pose.position_gimbal_m = cv::Vec3d(p_gimbal.x(), p_gimbal.y(), p_gimbal.z());
        pose.detection.target_id = target_id;
        pose.center_offset_px = offset_px;
        poses.push_back(pose);
    }
    return poses;
}

// 一辆车只露一块板，位置直接给在云台系里。两车场景里更贴近现场的常态。
ArmorPose singlePlate(const Eigen::Vector3d& p_gimbal, int target_id, double offset_px) {
    ArmorPose pose;
    pose.position_gimbal_m = cv::Vec3d(p_gimbal.x(), p_gimbal.y(), p_gimbal.z());
    pose.detection.target_id = target_id;
    pose.center_offset_px = offset_px;
    return pose;
}

std::vector<ArmorPose> concat(const std::vector<ArmorPose>& a,
                              const std::vector<ArmorPose>& b) {
    std::vector<ArmorPose> out = a;
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

// 单帧打分，不经过跨帧状态，用来单独看两种模式谁赢。
TargetCluster scoreOf(const std::vector<ArmorPose>& poses, double gimbal_yaw,
                      const TargetSelectorConfig& config, int cluster_index) {
    const std::vector<TargetCluster> clusters =
        TargetSelector::buildClusters(poses, gimbal_yaw, config);
    assert(cluster_index >= 0 && cluster_index < static_cast<int>(clusters.size()));
    TargetCluster cluster = clusters[static_cast<std::size_t>(cluster_index)];
    cluster.score = TargetSelector::scoreOf(cluster, config);
    return cluster;
}

void testClustersTwoVehicles() {
    const std::vector<ArmorPose> poses =
        concat(vehicle(Eigen::Vector3d(3.0, 0.0, 0.0), 0.0, 3, 40.0),
               vehicle(Eigen::Vector3d(3.0, -1.6, 0.0), 0.0, 1, 120.0));
    const TargetSelectorConfig config;
    const std::vector<TargetCluster> clusters =
        TargetSelector::buildClusters(poses, 0.0, config);

    assert(clusters.size() == 2);
    assert(clusters[0].indices.size() == 4);
    assert(clusters[1].indices.size() == 4);
    assert(clusters[0].target_id == 3);
    assert(clusters[1].target_id == 1);
    // 索引必须指回原始观测，且从左到右升序，回放时才对得上。
    assert(clusters[0].indices[0] == 0 && clusters[0].indices[3] == 3);
    assert(clusters[1].indices[0] == 4 && clusters[1].indices[3] == 7);
    // 距离取簇内最近那块板，不是板心均值。
    assert(std::abs(clusters[0].distance_m - 2.75) < 1e-9);
}

void testClusterUsesModeOfTargetId() {
    // 同一辆车的四块板被分类器读出三个不同编号，聚簇不能因此裂开。
    std::vector<ArmorPose> poses = vehicle(Eigen::Vector3d(3.0, 0.0, 0.0), 0.0, 3, 40.0);
    poses[1].detection.target_id = 2;
    poses[2].detection.target_id = 0;
    poses[3].detection.target_id = 3;
    const TargetSelectorConfig config;
    const std::vector<TargetCluster> clusters =
        TargetSelector::buildClusters(poses, 0.0, config);
    assert(clusters.size() == 1);
    assert(clusters[0].target_id == 3);
}

void testNearestToCenterPicksCenteredVehicle() {
    TargetSelectorConfig config;
    config.mode = TargetSelectionMode::NearestToCenter;
    // 近的那辆在左边 400 px，远的那辆在正前方 20 px。
    const std::vector<ArmorPose> poses =
        concat(vehicle(Eigen::Vector3d(2.0, 1.0, 0.0), 0.0, 4, 400.0),
               vehicle(Eigen::Vector3d(5.0, 0.0, 0.0), 0.0, 2, 20.0));
    const TargetCluster left = scoreOf(poses, 0.0, config, 0);
    const TargetCluster center = scoreOf(poses, 0.0, config, 1);
    assert(center.score > left.score);

    TargetSelector selector(config);
    const TargetSelection selection = selector.select(poses, 0.0, 0.0);
    assert(selection.has_target);
    assert(selection.target_id == 2);
}

void testWeightedPrefersDigitOverGuard() {
    // 哨兵在 2 m、正对画面中心；带数字的车在 4 m、偏 300 px。
    // 只比像素距离时哨兵赢，三因素加权时数字车赢。
    const std::vector<ArmorPose> poses{
        singlePlate(Eigen::Vector3d(2.0, 0.0, 0.0), 7, 0.0),
        singlePlate(Eigen::Vector3d(4.0, 0.0, 0.0), 2, 300.0),
    };
    const std::vector<ArmorPose> guard_first{poses[0], poses[1]};

    TargetSelectorConfig centered_config;
    centered_config.mode = TargetSelectionMode::NearestToCenter;
    const TargetCluster guard_centered = scoreOf(guard_first, 0.0, centered_config, 0);
    const TargetCluster digit_centered = scoreOf(guard_first, 0.0, centered_config, 1);
    assert(guard_centered.score > digit_centered.score);

    TargetSelectorConfig weighted_config;
    const TargetCluster guard_weighted = scoreOf(guard_first, 0.0, weighted_config, 0);
    const TargetCluster digit_weighted = scoreOf(guard_first, 0.0, weighted_config, 1);
    assert(digit_weighted.score > guard_weighted.score);

    TargetSelector selector(weighted_config);
    const TargetSelection selection = selector.select(guard_first, 0.0, 0.0);
    assert(selection.has_target);
    assert(selection.target_id == 2);
}

void testWeightedPrefersCloserWhenTypesEqual() {
    const std::vector<ArmorPose> poses{
        singlePlate(Eigen::Vector3d(2.0, 0.0, 0.0), 2, 100.0),
        singlePlate(Eigen::Vector3d(4.0, 0.0, 0.0), 3, 100.0),
    };
    const TargetSelectorConfig config;
    TargetSelector selector(config);
    const TargetSelection selection = selector.select(poses, 0.0, 0.0);
    assert(selection.target_id == 2);
}

void testHysteresisKeepsIncumbent() {
    // 挑战者只比当前目标高一点点，低于 switch_margin，连续多帧都不许切。
    // 首帧只有当前目标，否则「无跟踪对象即选最优」会直接选中分数更高的挑战者。
    const std::vector<ArmorPose> incumbent{
        singlePlate(Eigen::Vector3d(2.5, 0.0, 0.0), 2, 100.0),
    };
    const std::vector<ArmorPose> both{
        incumbent[0],
        singlePlate(Eigen::Vector3d(2.5, 1.4, 0.0), 3, 60.0),
    };
    TargetSelectorConfig config;
    config.switch_confirm_frames = 3;
    TargetSelector selector(config);

    const TargetSelection first = selector.select(incumbent, 0.0, 0.0);
    assert(first.switched);
    assert(first.target_id == 2);

    for (int frame = 1; frame <= 10; ++frame) {
        const TargetSelection selection = selector.select(both, 0.0, 0.01 * frame);
        assert(!selection.switched);
        assert(selection.target_id == 2);
        assert(selection.track_id == first.track_id);
    }
}

void testSwitchAfterConfirmFrames() {
    // 挑战者明显更好：要连续 switch_confirm_frames 帧才切，切的那帧 switched 为真。
    // 首帧只有当前目标，先把它确立为前提；挑战者从第二帧才出现。
    const std::vector<ArmorPose> incumbent{
        singlePlate(Eigen::Vector3d(5.0, 0.0, 0.0), 4, 500.0),
    };
    const std::vector<ArmorPose> both{
        incumbent[0],
        singlePlate(Eigen::Vector3d(2.0, 1.0, 0.0), 2, 10.0),
    };
    TargetSelectorConfig config;
    config.switch_confirm_frames = 3;
    TargetSelector selector(config);

    const TargetSelection first = selector.select(incumbent, 0.0, 0.0);
    assert(first.switched);
    assert(first.target_id == 4);

    // 前两帧只记账，不切。
    for (int frame = 1; frame <= 2; ++frame) {
        const TargetSelection selection = selector.select(both, 0.0, 0.01 * frame);
        assert(!selection.switched);
        assert(selection.target_id == 4);
    }
    // 第三帧完成确认。
    const TargetSelection switched = selector.select(both, 0.0, 0.03);
    assert(switched.switched);
    assert(switched.target_id == 2);
    assert(switched.track_id != first.track_id);
    // 切换后喂给 EKF 的观测必须换成新车那一条。
    assert(switched.observation_indices.size() == 1);
    assert(switched.observation_indices[0] == 1);

    // 切完就稳定，不再反复。
    const TargetSelection settled = selector.select(both, 0.0, 0.04);
    assert(!settled.switched);
    assert(settled.target_id == 2);
}

void testCoastKeepsTarget() {
    const std::vector<ArmorPose> poses = vehicle(Eigen::Vector3d(3.0, 0.0, 0.0), 0.0, 3, 40.0);
    TargetSelectorConfig config;
    config.coast_timeout_s = 0.5;
    TargetSelector selector(config);

    const TargetSelection first = selector.select(poses, 0.0, 0.0);
    assert(first.switched);
    assert(first.observation_indices.size() == 4);

    // 目标被挡住，画面里还有别的车；超时之前不许换车，也不许喂观测。
    const std::vector<ArmorPose> other{
        singlePlate(Eigen::Vector3d(2.0, 1.5, 0.0), 2, 10.0),
    };
    for (int frame = 1; frame <= 40; ++frame) {
        const TargetSelection selection = selector.select(other, 0.0, 0.01 * frame);
        assert(selection.has_target);
        assert(selection.coasting);
        assert(!selection.switched);
        assert(selection.target_id == 3);
        assert(selection.track_id == first.track_id);
        assert(selection.observation_indices.empty());
    }

    // 超时之后立刻改选最优的，并让 EKF 重置。
    const TargetSelection expired = selector.select(other, 0.0, 0.60);
    assert(expired.has_target);
    assert(!expired.coasting);
    assert(expired.switched);
    assert(expired.target_id == 2);
    assert(expired.observation_indices.size() == 1);
}

void testCoastExpiresToNoTargetWhenSceneEmpty() {
    const std::vector<ArmorPose> poses = vehicle(Eigen::Vector3d(3.0, 0.0, 0.0), 0.0, 3, 40.0);
    TargetSelectorConfig config;
    config.coast_timeout_s = 0.5;
    TargetSelector selector(config);
    selector.select(poses, 0.0, 0.0);

    const std::vector<ArmorPose> empty;
    const TargetSelection coasting = selector.select(empty, 0.0, 0.2);
    assert(coasting.has_target);
    assert(coasting.coasting);

    const TargetSelection gone = selector.select(empty, 0.0, 0.6);
    assert(!gone.has_target);
    assert(!gone.switched);
    assert(gone.target_id == 0);
    assert(gone.observation_indices.empty());
}

void testGimbalRotationDoesNotBreakIdentity() {
    // 同一辆车，云台转了 90 度。云台系里的坐标全变了，世界系没变，不该换车。
    const Eigen::Vector3d car_a(3.0, 0.0, 0.0);
    const Eigen::Vector3d car_b(3.0, -1.6, 0.0);
    const std::vector<ArmorPose> before =
        concat(vehicle(car_a, 0.0, 3, 40.0), vehicle(car_b, 0.0, 1, 120.0));

    TargetSelectorConfig config;
    TargetSelector selector(config);
    const TargetSelection first = selector.select(before, 0.0, 0.0);
    assert(first.switched);
    assert(first.target_id == 3);

    const double yaw = 0.5 * kPi;
    const std::vector<ArmorPose> after =
        concat(vehicle(car_a, yaw, 3, 40.0), vehicle(car_b, yaw, 1, 120.0));
    const TargetSelection rotated = selector.select(after, yaw, 0.01);
    assert(!rotated.switched);
    assert(rotated.target_id == 3);
    assert(rotated.track_id == first.track_id);
}

void testStickerChangeDoesNotHijackTrack() {
    // 跟丢之后，0.9 m 外的另一辆车编号不同，不能被当成本车。
    // 距离在 identity_match_m 以内，只有编号能挡住它。
    TargetSelectorConfig config;
    config.coast_timeout_s = 0.5;
    TargetSelector selector(config);

    const std::vector<ArmorPose> tracked{
        singlePlate(Eigen::Vector3d(3.0, 0.0, 0.0), 3, 40.0),
    };
    const TargetSelection first = selector.select(tracked, 0.0, 0.0);
    assert(first.switched);
    assert(first.target_id == 3);

    const std::vector<ArmorPose> rival{
        singlePlate(Eigen::Vector3d(3.0, 0.9, 0.0), 5, 40.0),
    };
    const TargetSelection selection = selector.select(rival, 0.0, 0.01);
    assert(selection.has_target);
    assert(selection.coasting);
    assert(!selection.switched);
    assert(selection.target_id == 3);
    assert(selection.track_id == first.track_id);
    assert(selection.observation_indices.empty());
}

void testUnknownStickerFallsBackToGeometry() {
    // 读不出编号时没有贴纸信用，只剩几何：0.55 m 以内仍认，0.9 m 不认。
    const std::vector<ArmorPose> tracked{
        singlePlate(Eigen::Vector3d(3.0, 0.0, 0.0), 3, 40.0),
    };

    TargetSelector near_selector;
    const TargetSelection first = near_selector.select(tracked, 0.0, 0.0);
    assert(first.target_id == 3);
    const std::vector<ArmorPose> near_unknown{
        singlePlate(Eigen::Vector3d(3.0, 0.4, 0.0), 0, 40.0),
    };
    const TargetSelection near = near_selector.select(near_unknown, 0.0, 0.01);
    assert(!near.coasting);
    assert(near.track_id == first.track_id);
    assert(near.observation_indices.size() == 1);

    TargetSelector far_selector;
    far_selector.select(tracked, 0.0, 0.0);
    const std::vector<ArmorPose> far_unknown{
        singlePlate(Eigen::Vector3d(3.0, 0.9, 0.0), 0, 40.0),
    };
    const TargetSelection far = far_selector.select(far_unknown, 0.0, 0.01);
    assert(far.coasting);
    assert(far.target_id == 3);
    assert(far.observation_indices.empty());
}

void testStickerKeepsIdentityAcrossLargeJump() {
    // 编号相同，0.9 m 的簇心跳变仍算同一辆车，不许断跟。
    TargetSelector selector;
    const std::vector<ArmorPose> tracked{
        singlePlate(Eigen::Vector3d(3.0, 0.0, 0.0), 3, 40.0),
    };
    const TargetSelection first = selector.select(tracked, 0.0, 0.0);
    assert(first.switched);

    const std::vector<ArmorPose> jumped{
        singlePlate(Eigen::Vector3d(3.0, 0.9, 0.0), 3, 40.0),
    };
    const TargetSelection selection = selector.select(jumped, 0.0, 0.01);
    assert(!selection.coasting);
    assert(!selection.switched);
    assert(selection.target_id == 3);
    assert(selection.track_id == first.track_id);
    assert(selection.observation_indices.size() == 1);
}

void testBridgePlateDoesNotMergeVehicles() {
    // v4.avi 第 137 帧的实测位置：近车 id3、远车 id1，中间一块被误识别成 id0 的板
    // 离近车 0.77 m、离远车 0.42 m。单链接聚簇会把三块串成一簇，两辆车一起进 EKF。
    const std::vector<ArmorPose> poses{
        singlePlate(Eigen::Vector3d(1.23, 0.21, 0.14), 3, 30.0),
        singlePlate(Eigen::Vector3d(1.65, -0.40, 0.35), 0, 260.0),
        singlePlate(Eigen::Vector3d(2.06, -0.47, 0.28), 1, 470.0),
    };
    const TargetSelectorConfig config;
    const std::vector<TargetCluster> clusters =
        TargetSelector::buildClusters(poses, 0.0, config);

    assert(clusters.size() == 2);
    // 桥接板跟远车走，近车自己一簇。
    bool found_near = false;
    bool found_far = false;
    for (const TargetCluster& cluster : clusters) {
        if (cluster.indices.size() == 1) {
            assert(cluster.indices[0] == 0);
            assert(cluster.target_id == 3);
            found_near = true;
        } else {
            assert(cluster.indices.size() == 2);
            assert(cluster.target_id == 1);
            found_far = true;
        }
    }
    assert(found_near && found_far);
}

void testClusterDiameterCap() {
    // v4.avi 第 801 帧实测：本车两块板相距 0.44 m，远车一块板离本车近的那块 0.54 m。
    // 两段都在 cluster_link_m 之内，单链接会把三块串成一簇，簇直径 0.94 m；一辆车
    // 四块板最远也就 0.69 m，所以要么本车两块走、要么远车那块被拉走，取决于谁先
    // 当种子。卡直径后最近的一对先合并，远车那块单独成簇。
    const std::vector<ArmorPose> poses{
        singlePlate(Eigen::Vector3d(2.04, -0.46, 0.28), 1, 470.0),
        singlePlate(Eigen::Vector3d(1.82, 0.03, 0.27), 3, 40.0),
        singlePlate(Eigen::Vector3d(1.88, 0.47, 0.16), 3, 150.0),
    };
    const TargetSelectorConfig config;
    const std::vector<TargetCluster> clusters =
        TargetSelector::buildClusters(poses, 0.0, config);

    assert(clusters.size() == 2);
    assert(clusters[0].indices.size() == 1);
    assert(clusters[0].indices[0] == 0);
    assert(clusters[0].target_id == 1);
    assert(clusters[1].indices.size() == 2);
    assert(clusters[1].indices[0] == 1);
    assert(clusters[1].indices[1] == 2);
    assert(clusters[1].target_id == 3);

    // 反证：把直径上限放到 1.0 m，同一组观测就串回一簇。差别只能来自上限本身。
    TargetSelectorConfig loose = config;
    loose.cluster_max_extent_m = 1.0;
    const std::vector<TargetCluster> merged =
        TargetSelector::buildClusters(poses, 0.0, loose);
    assert(merged.size() == 1);
    assert(merged[0].indices.size() == 3);
}

void testStickerDecidesAmbiguousPairing() {
    // v4.avi 第 733 帧实测：远车 id1 的板离本车近的那块只有 0.39 m，比本车两块
    // 之间 0.43 m 还近。两两都能连，先连哪一对就决定了结果。纯按距离会把本车
    // 拆开、把远车那块并进来；编号相同的一对必须优先。
    const std::vector<ArmorPose> poses{
        singlePlate(Eigen::Vector3d(2.05, -0.46, 0.28), 1, 470.0),
        singlePlate(Eigen::Vector3d(1.93, -0.09, 0.18), 3, 150.0),
        singlePlate(Eigen::Vector3d(1.87, 0.34, 0.26), 3, 60.0),
    };
    const TargetSelectorConfig config;
    const std::vector<TargetCluster> clusters =
        TargetSelector::buildClusters(poses, 0.0, config);

    assert(clusters.size() == 2);
    assert(clusters[0].indices.size() == 1);
    assert(clusters[0].indices[0] == 0);
    assert(clusters[1].indices.size() == 2);
    assert(clusters[1].indices[0] == 1);
    assert(clusters[1].indices[1] == 2);
    assert(clusters[1].target_id == 3);
}

void testAdjacentPlatesOfOneVehicleStayTogether() {
    // 同一辆车上相邻两块板实测间距 0.43 到 0.49 m，聚簇阈值不能低到把车自己拆开。
    const std::vector<ArmorPose> poses{
        singlePlate(Eigen::Vector3d(2.0, 0.0, 0.0), 3, 40.0),
        singlePlate(Eigen::Vector3d(2.0 + 0.48, 0.0, 0.0), 3, 140.0),
    };
    const TargetSelectorConfig config;
    const std::vector<TargetCluster> clusters =
        TargetSelector::buildClusters(poses, 0.0, config);

    assert(clusters.size() == 1);
    assert(clusters[0].indices.size() == 2);
}

void testSplitClusterTieKeepsHigherScoringCluster() {
    // 2026-10-21 实车尾段 f=4672 到 f=4673 的复现。两板同簇时簇心是两板均值；
    // 下一帧两板之间拉开越过 cluster_link_m 就拆成两簇，两个子簇心对上一帧的
    // 簇心精确等距。并列时如果留任的是分数更低的那簇，挑战者连续三帧领先就会
    // 换簇，调用方据此重置滤波器。并列必须按分数挑，让 best 与 matched 重合。
    const std::vector<ArmorPose> merged{
        singlePlate(Eigen::Vector3d(2.0, -0.25, 0.0), 3, 300.0),
        singlePlate(Eigen::Vector3d(2.0, 0.25, 0.0), 3, 40.0),
    };
    // 关于 (2.0, 0.0, 0.0) 对称，间距 0.60 m 越过 cluster_link_m 0.55。
    const std::vector<ArmorPose> split{
        singlePlate(Eigen::Vector3d(2.0, -0.30, 0.0), 3, 300.0),
        singlePlate(Eigen::Vector3d(2.0, 0.30, 0.0), 3, 40.0),
    };

    const TargetSelectorConfig config;
    const std::vector<TargetCluster> merged_clusters =
        TargetSelector::buildClusters(merged, 0.0, config);
    assert(merged_clusters.size() == 1);
    const std::vector<TargetCluster> split_clusters =
        TargetSelector::buildClusters(split, 0.0, config);
    assert(split_clusters.size() == 2);

    // 前提一：两个子簇心到上一帧簇心的距离真的相等，不是近似相等。
    const Eigen::Vector3d previous = merged_clusters[0].center_world_m;
    const double left_gap = (split_clusters[0].center_world_m - previous).norm();
    const double right_gap = (split_clusters[1].center_world_m - previous).norm();
    assert(left_gap == right_gap);

    // 前提二：下标 0 那簇分数更低，差得够 switch_margin。
    assert(TargetSelector::scoreOf(split_clusters[1], config) >
           TargetSelector::scoreOf(split_clusters[0], config) + config.switch_margin);

    TargetSelector selector(config);
    const TargetSelection first = selector.select(merged, 0.0, 0.0);
    assert(first.switched);
    assert(first.target_id == 3);

    // 拆分后的两帧里不许换簇，票数永远不该走起来。
    for (int frame = 1; frame <= 2 * config.switch_confirm_frames; ++frame) {
        const TargetSelection selection = selector.select(split, 0.0, 0.01 * frame);
        assert(!selection.switched);
        assert(selection.track_id == first.track_id);
    }
}

void testEmptyInputFromStart() {
    TargetSelector selector;
    const std::vector<ArmorPose> empty;
    const TargetSelection selection = selector.select(empty, 0.0, 0.0);
    assert(!selection.has_target);
    assert(!selection.switched);
    assert(selection.track_id == 0);
    assert(selection.observation_indices.empty());
}

}  // namespace

int main() {
    testClustersTwoVehicles();
    std::printf("clusters_two_vehicles ok\n");
    testClusterUsesModeOfTargetId();
    std::printf("cluster_mode_of_target_id ok\n");
    testNearestToCenterPicksCenteredVehicle();
    std::printf("nearest_to_center ok\n");
    testWeightedPrefersDigitOverGuard();
    std::printf("weighted_digit_over_guard ok\n");
    testWeightedPrefersCloserWhenTypesEqual();
    std::printf("weighted_closer ok\n");
    testHysteresisKeepsIncumbent();
    std::printf("hysteresis ok\n");
    testSwitchAfterConfirmFrames();
    std::printf("switch_confirm ok\n");
    testCoastKeepsTarget();
    std::printf("coast ok\n");
    testCoastExpiresToNoTargetWhenSceneEmpty();
    std::printf("coast_expire_empty ok\n");
    testGimbalRotationDoesNotBreakIdentity();
    std::printf("gimbal_rotation ok\n");
    testStickerChangeDoesNotHijackTrack();
    std::printf("sticker_change ok\n");
    testUnknownStickerFallsBackToGeometry();
    std::printf("unknown_sticker ok\n");
    testStickerKeepsIdentityAcrossLargeJump();
    std::printf("sticker_large_jump ok\n");
    testBridgePlateDoesNotMergeVehicles();
    std::printf("bridge_plate ok\n");
    testStickerDecidesAmbiguousPairing();
    std::printf("sticker_pairing ok\n");
    testClusterDiameterCap();
    std::printf("cluster_diameter_cap ok\n");
    testAdjacentPlatesOfOneVehicleStayTogether();
    std::printf("adjacent_plates ok\n");
    testSplitClusterTieKeepsHigherScoringCluster();
    std::printf("split_cluster_tie ok\n");
    testEmptyInputFromStart();
    std::printf("empty_input ok\n");
    return 0;
}
