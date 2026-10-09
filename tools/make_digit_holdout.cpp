// 从数据集里抽一份"留出集"，复制进仓库并写清单。
//
// 用法：make_digit_holdout <数据集根> <留出集目录> [每类张数]
//
// 为什么要有这个工具，而不是手工挑 180 张：
//   1. 可复现。换台机器、换次答辩，抽出来的是同一批图。
//   2. 清单是训练程序和测试程序共用的同一份文件，
//      训练时按它排除、测试时按它评估 —— 不可能出现"训练用了又拿去测"。
//
// 抽法是"每类固定种子洗牌后取前 N 张"，不是"取前 N 张"：
// 数据集很可能是按采集顺序排的，取前 N 张会让留出集全来自同一段时间、
// 同一种光照，测出来的数字会偏乐观。
#include "DigitClassifier.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <sys/stat.h>

#include <glob.h>
#include <opencv2/imgcodecs.hpp>

namespace {

std::string join(const std::string& dir, const std::string& name) {
    if (!dir.empty() && dir.back() == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

std::string baseName(const std::string& path) {
    const std::size_t pos = path.find_last_of('/');
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

// glob 默认按文件名排序，所以返回顺序是确定的 —— 可复现靠这个。
std::vector<std::string> globFiles(const std::string& pattern) {
    std::vector<std::string> files;
    glob_t result{};
    if (glob(pattern.c_str(), 0, nullptr, &result) == 0) {
        for (std::size_t i = 0; i < result.gl_pathc; ++i) {
            files.emplace_back(result.gl_pathv[i]);
        }
    }
    globfree(&result);
    return files;
}

void makeDirectory(const std::string& path) {
    // 已经存在时返回 -1，这里不区分"已存在"和"失败"：
    // 真失败了后面的 imwrite 会报错，报错信息更有用。
    ::mkdir(path.c_str(), 0755);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "用法：make_digit_holdout <数据集根> <留出集目录> [每类张数]\n"
                     "例：  make_digit_holdout ~/armor_digit_dataset/datasets \\\n"
                     "          tests/fixtures/digit_holdout 20\n";
        return 2;
    }
    const std::string root = argv[1];
    const std::string out_dir = argv[2];
    const int per_class = (argc == 4) ? std::stoi(argv[3]) : 20;
    if (per_class <= 0) {
        std::cerr << "每类张数必须为正\n";
        return 2;
    }

    makeDirectory(out_dir);
    std::ofstream manifest(join(out_dir, "manifest.txt"));
    if (!manifest) {
        std::cerr << "写不了清单文件：" << join(out_dir, "manifest.txt") << "\n";
        return 2;
    }
    manifest << "# <相对路径> <label>   —— 训练时跳过、测试时评估，两边共用这一份\n";

    int total = 0;
    for (int label = 0; label < armor::DigitClassifier::kClassCount; ++label) {
        const std::string name = armor::DigitClassifier::className(label);
        std::vector<std::string> files = globFiles(join(root, name) + "/*");
        if (files.empty()) {
            std::cerr << "类别 " << name << " 一张图都没有，检查数据集根目录\n";
            return 2;
        }

        // 种子带上 label，各类抽的是不同的位置，但每次都一样。
        std::mt19937 rng(1000U + static_cast<unsigned>(label));
        std::shuffle(files.begin(), files.end(), rng);

        const int take = std::min<int>(per_class, static_cast<int>(files.size()));
        makeDirectory(join(out_dir, name));

        int copied = 0;
        for (int i = 0; i < take; ++i) {
            const cv::Mat patch = cv::imread(files[i], cv::IMREAD_GRAYSCALE);
            if (patch.empty()) {
                continue;
            }
            const std::string file_name = baseName(files[i]);
            const std::string saved = join(join(out_dir, name), file_name);
            if (!cv::imwrite(saved, patch)) {
                std::cerr << "写不了 " << saved << "\n";
                return 2;
            }
            manifest << name << "/" << file_name << " " << label << "\n";
            ++copied;
        }
        std::cout << "  " << name << " : " << copied << " 张\n";
        total += copied;
    }

    std::cout << "共 " << total << " 张，清单写到 " << join(out_dir, "manifest.txt")
              << "\n";
    return 0;
}
