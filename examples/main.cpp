// examples/main.cpp
#include "YOLODetector.h"
#include <iostream>

int main() {
    YOLODetector detector("../11.onnx", 0.5f, 0.45f);
    // 检测并保存，同时拿到结果
    auto dets = detector.detectAndSave("../test.jpeg", "../result.jpeg");

    // 输出每个检测框的信息
    for (auto& d : dets) {
        std::cout
          << d.class_name << " (" << d.class_id << ") "
          << "score=" << d.score
          << " box=[" << d.x << "," << d.y
          << "," << d.width << "," << d.height << "]\n";
    }
    std::cout << "检测结果已保存到 result.jpeg\n";
    return 0;
}



