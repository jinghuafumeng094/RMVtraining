#pragma once

#include "ArmorEkf.hpp"
#include "AutoAimTypes.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

// 多目标筛选。
//
// 一帧里可能出现两辆车，而 ArmorEkf 只有一套状态和一个四槽关联器，全帧观测直接
// 灌进去会被两辆车互相抢槽位。这一层在 EKF 之前把观测按车聚簇，选出一辆车，
// 只把它的观测交给滤波器。
//
// 两条硬约束：
// 1. 打分只改变喂给 EKF 的观测，不改变 EKF 内部的任何状态。
// 2. 换车等价于换了物理对象，滤波器必须重置，否则旧状态会把新车挡在跳变闸外。

namespace armor {

// 两种筛选策略。NearestToCenter 只看板心到画面中心的像素距离；
// Weighted 同时看距离、目标类型和像素距离。
enum class TargetSelectionMode {
    NearestToCenter,
    Weighted,
};

struct TargetSelectorConfig {
    TargetSelectionMode mode{TargetSelectionMode::Weighted};

    // 同车相邻两块板间距 2*r*sin(45) 约 0.35 m，板环直径不到 0.7 m；
    // 两辆车相距常在 1 m 以上。0.80 m 把两者分开。
    double cluster_link_m{0.80};

    // 跨帧认同：新簇心与上次选中簇心在世界系中的距离不超过此值，视为同一辆车。
    // 必须小于两车典型间距，否则会吃掉另一辆车。
    double identity_match_m{1.20};

    // 挑战者分数要高出当前目标这么多才记一次切换票，抑制两车分数胶着时来回跳。
    double switch_margin{0.08};
    int switch_confirm_frames{3};

    // 目标被遮挡时保持跟踪的时间。超时后重新选，并让 EKF 重置。
    double coast_timeout_s{0.5};

    // 三因素权重，内部按总和归一化，改比例即可。
    double weight_distance{0.35};
    double weight_type{0.35};
    double weight_center{0.30};

    // 距离与像素偏置的软化尺度：分数分别是 1/(1+d/scale) 与 1/(1+(px/scale)^2)。
    double distance_scale_m{5.0};
    double center_scale_px{300.0};

    // 目标类型得分。带数字的装甲板给满分，读不出编号给半分，哨兵给零分。
    double type_score_digit{1.0};
    double type_score_unknown{0.5};
    double type_score_guard{0.0};
    int guard_target_id{7};
};

// 一帧里被聚成一簇的若干块板，对应一辆车。
struct TargetCluster {
    std::vector<int> indices;             // 指回输入观测的下标，升序
    Eigen::Vector3d center_gimbal_m = Eigen::Vector3d::Zero();  // 板心均值
    Eigen::Vector3d center_world_m = Eigen::Vector3d::Zero();   // 同上，换到世界系
    double distance_m{0.0};               // 簇内最近一块板到云台原点的距离
    int target_id{0};                     // 簇内非零编号的众数，全零则为 0
    double center_offset_px{0.0};         // 簇内最小像素偏置
    double score{0.0};                    // scoreOf 的输出
};

struct TargetSelection {
    bool has_target{false};
    bool switched{false};                 // 这一帧换了车，调用方必须重置滤波器
    bool coasting{false};                 // 目标这帧没观测，保持跟踪中
    int track_id{0};                      // 跟踪序号，换车时递增
    int target_id{0};
    std::vector<int> observation_indices; // 这一帧该喂给 EKF 的观测下标
    double score{0.0};
};

class TargetSelector {
public:
    TargetSelector() = default;
    explicit TargetSelector(const TargetSelectorConfig& config) : config_(config) {}

    const TargetSelectorConfig& config() const { return config_; }

    void setMode(TargetSelectionMode mode) { config_.mode = mode; }

    // 清空跟踪状态。换车由 select 内部处理，这里供外部整体复位。
    void reset();

    // timestamp_seconds 用于遮挡计时，必须单调不减；非有限值时按 0 处理。
    TargetSelection select(const std::vector<ArmorPose>& observations, double gimbal_yaw,
                           double timestamp_seconds);

    // 纯函数，不依赖跨帧状态，供测试与调试。
    static std::vector<TargetCluster> buildClusters(const std::vector<ArmorPose>& observations,
                                                    double gimbal_yaw,
                                                    const TargetSelectorConfig& config);
    static double scoreOf(const TargetCluster& cluster, const TargetSelectorConfig& config);

private:
    void acquire(const TargetCluster& cluster, double timestamp_seconds);
    TargetSelection makeSelection(const TargetCluster& cluster) const;

    TargetSelectorConfig config_;
    int track_id_{0};
    int next_track_id_{1};
    int target_id_{0};
    Eigen::Vector3d tracked_center_world_m_{Eigen::Vector3d::Zero()};
    double tracked_timestamp_s_{0.0};
    bool has_tracked_timestamp_{false};
    int switch_votes_{0};
};

namespace detail {

inline Eigen::Vector3d toEigen(const cv::Vec3d& value) {
    return Eigen::Vector3d(value[0], value[1], value[2]);
}

// 簇内非零编号的众数；票数相同时取编号小的。全零返回 0。
inline int dominantTargetId(const std::vector<ArmorPose>& observations,
                            const std::vector<int>& indices) {
    int best_id = 0;
    int best_count = 0;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        const int candidate = observations[static_cast<std::size_t>(indices[i])].detection.target_id;
        if (candidate == 0) {
            continue;
        }
        int count = 0;
        for (std::size_t j = 0; j < indices.size(); ++j) {
            if (observations[static_cast<std::size_t>(indices[j])].detection.target_id == candidate) {
                ++count;
            }
        }
        if (count > best_count || (count == best_count && (best_id == 0 || candidate < best_id))) {
            best_count = count;
            best_id = candidate;
        }
    }
    return best_id;
}

}  // namespace detail

inline std::vector<TargetCluster> TargetSelector::buildClusters(
    const std::vector<ArmorPose>& observations, double gimbal_yaw,
    const TargetSelectorConfig& config) {
    std::vector<TargetCluster> clusters;
    const std::size_t count = observations.size();
    if (count == 0) {
        return clusters;
    }

    // 单链接聚簇：只要两块板间距不超过阈值就连成一簇。不用 target_id 当屏障，
    // 同一辆车四块板的编号可能被分类器读错，硬拆会把一辆车劈成两辆。
    std::vector<int> label(count, -1);
    for (std::size_t seed = 0; seed < count; ++seed) {
        if (label[seed] >= 0) {
            continue;
        }
        const int cluster_index = static_cast<int>(clusters.size());
        clusters.push_back(TargetCluster{});
        label[seed] = cluster_index;
        std::vector<std::size_t> pending{seed};
        while (!pending.empty()) {
            const std::size_t current = pending.back();
            pending.pop_back();
            clusters[static_cast<std::size_t>(cluster_index)].indices.push_back(
                static_cast<int>(current));
            for (std::size_t other = 0; other < count; ++other) {
                if (label[other] >= 0) {
                    continue;
                }
                if (cv::norm(observations[other].position_gimbal_m -
                             observations[current].position_gimbal_m) <= config.cluster_link_m) {
                    label[other] = cluster_index;
                    pending.push_back(other);
                }
            }
        }
    }

    const Eigen::Matrix3d to_world = ArmorEkf::gimbalToWorld(gimbal_yaw);
    for (TargetCluster& cluster : clusters) {
        std::sort(cluster.indices.begin(), cluster.indices.end());
        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
        double nearest = std::numeric_limits<double>::infinity();
        double offset = std::numeric_limits<double>::infinity();
        for (int index : cluster.indices) {
            const ArmorPose& pose = observations[static_cast<std::size_t>(index)];
            const Eigen::Vector3d position = detail::toEigen(pose.position_gimbal_m);
            sum += position;
            nearest = std::min(nearest, position.norm());
            offset = std::min(offset, pose.center_offset_px);
        }
        cluster.center_gimbal_m = sum / static_cast<double>(cluster.indices.size());
        cluster.center_world_m = to_world * cluster.center_gimbal_m;
        cluster.distance_m = nearest;
        cluster.center_offset_px = offset;
        cluster.target_id = detail::dominantTargetId(observations, cluster.indices);
    }
    return clusters;
}

inline double TargetSelector::scoreOf(const TargetCluster& cluster,
                                      const TargetSelectorConfig& config) {
    const double center_scale = std::max(config.center_scale_px, 1e-6);
    const double normalized_center = cluster.center_offset_px / center_scale;
    const double center_score = 1.0 / (1.0 + normalized_center * normalized_center);
    if (config.mode == TargetSelectionMode::NearestToCenter) {
        return center_score;
    }

    const double distance_scale = std::max(config.distance_scale_m, 1e-6);
    const double distance_score = 1.0 / (1.0 + cluster.distance_m / distance_scale);
    double type_score = config.type_score_digit;
    if (cluster.target_id == 0) {
        type_score = config.type_score_unknown;
    } else if (cluster.target_id == config.guard_target_id) {
        type_score = config.type_score_guard;
    }

    const double weight_sum = config.weight_distance + config.weight_type + config.weight_center;
    if (!(weight_sum > 0.0)) {
        return center_score;
    }
    return (config.weight_distance * distance_score + config.weight_type * type_score +
            config.weight_center * center_score) /
           weight_sum;
}

inline void TargetSelector::reset() {
    track_id_ = 0;
    next_track_id_ = 1;
    target_id_ = 0;
    tracked_center_world_m_.setZero();
    tracked_timestamp_s_ = 0.0;
    has_tracked_timestamp_ = false;
    switch_votes_ = 0;
}

inline void TargetSelector::acquire(const TargetCluster& cluster, double timestamp_seconds) {
    track_id_ = next_track_id_;
    ++next_track_id_;
    target_id_ = cluster.target_id;
    tracked_center_world_m_ = cluster.center_world_m;
    tracked_timestamp_s_ = timestamp_seconds;
    has_tracked_timestamp_ = std::isfinite(timestamp_seconds);
    switch_votes_ = 0;
}

inline TargetSelection TargetSelector::makeSelection(const TargetCluster& cluster) const {
    TargetSelection selection;
    selection.has_target = true;
    selection.track_id = track_id_;
    selection.target_id = cluster.target_id;
    selection.observation_indices = cluster.indices;
    selection.score = cluster.score;
    return selection;
}

inline TargetSelection TargetSelector::select(const std::vector<ArmorPose>& observations,
                                              double gimbal_yaw, double timestamp_seconds) {
    TargetSelection selection;
    std::vector<TargetCluster> clusters = buildClusters(observations, gimbal_yaw, config_);
    for (TargetCluster& cluster : clusters) {
        cluster.score = scoreOf(cluster, config_);
    }

    // 本帧最优簇；同分取更近的，再同分取下标小的，保证确定性。
    int best = -1;
    for (int i = 0; i < static_cast<int>(clusters.size()); ++i) {
        if (best < 0) {
            best = i;
            continue;
        }
        const TargetCluster& candidate = clusters[static_cast<std::size_t>(i)];
        const TargetCluster& current = clusters[static_cast<std::size_t>(best)];
        if (candidate.score > current.score ||
            (candidate.score == current.score && candidate.distance_m < current.distance_m)) {
            best = i;
        }
    }

    // 还没有跟踪对象：直接选最优的一簇。
    if (track_id_ == 0) {
        if (best < 0) {
            return selection;
        }
        acquire(clusters[static_cast<std::size_t>(best)], timestamp_seconds);
        selection = makeSelection(clusters[static_cast<std::size_t>(best)]);
        selection.switched = true;
        return selection;
    }

    // 找和上次选中簇在世界系里对得上的那一簇。
    int matched = -1;
    double matched_distance = std::numeric_limits<double>::infinity();
    for (int i = 0; i < static_cast<int>(clusters.size()); ++i) {
        const double distance =
            (clusters[static_cast<std::size_t>(i)].center_world_m - tracked_center_world_m_)
                .norm();
        if (distance <= config_.identity_match_m && distance < matched_distance) {
            matched = i;
            matched_distance = distance;
        }
    }

    if (matched < 0) {
        // 目标这帧没有对得上的簇：先保持跟踪，超时才另选。
        const double elapsed =
            (has_tracked_timestamp_ && std::isfinite(timestamp_seconds))
                ? std::max(0.0, timestamp_seconds - tracked_timestamp_s_)
                : 0.0;
        if (elapsed <= config_.coast_timeout_s) {
            selection.has_target = true;
            selection.coasting = true;
            selection.track_id = track_id_;
            selection.target_id = target_id_;
            return selection;
        }
        if (best < 0) {
            reset();
            return selection;
        }
        acquire(clusters[static_cast<std::size_t>(best)], timestamp_seconds);
        selection = makeSelection(clusters[static_cast<std::size_t>(best)]);
        selection.switched = true;
        return selection;
    }

    tracked_timestamp_s_ = timestamp_seconds;
    has_tracked_timestamp_ = std::isfinite(timestamp_seconds);
    const TargetCluster& incumbent = clusters[static_cast<std::size_t>(matched)];

    // 挑战者要连续多帧领先才换，避免两车分数胶着时来回跳。
    const int confirm_frames = std::max(1, config_.switch_confirm_frames);
    if (best != matched &&
        clusters[static_cast<std::size_t>(best)].score > incumbent.score + config_.switch_margin) {
        ++switch_votes_;
    } else {
        switch_votes_ = 0;
    }

    if (switch_votes_ >= confirm_frames) {
        switch_votes_ = 0;
        acquire(clusters[static_cast<std::size_t>(best)], timestamp_seconds);
        selection = makeSelection(clusters[static_cast<std::size_t>(best)]);
        selection.switched = true;
        return selection;
    }

    tracked_center_world_m_ = incumbent.center_world_m;
    target_id_ = incumbent.target_id;
    return makeSelection(incumbent);
}

}  // namespace armor
