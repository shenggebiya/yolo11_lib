// include/YOLODetector.h
#ifndef YOLODETECTOR_H
#define YOLODETECTOR_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// 一条检测结果
struct Detection {
    int         class_id;
    std::string class_name;
    float       score;
    int         x, y, width, height;
};

class YOLODetector {
public:
    YOLODetector(const std::string& model_path,
                 float conf_thresh = 0.5f,
                 float iou_thresh  = 0.45f);
    ~YOLODetector();

    std::vector<Detection> detect(const std::string& image_path);
    std::vector<Detection> detectAndSave(const std::string& image_path,
                                         const std::string& output_path);

private:
    void letterbox(const cv::Mat& src,
                   cv::Mat&       dst,
                   float&         scale,
                   int&           top,
                   int&           left);

    Ort::Env             env_;
    Ort::Session*        session_;
    Ort::SessionOptions  session_options_;

    float                conf_thresh_;
    float                iou_thresh_;
    int                  input_width_;
    int                  input_height_;
    std::vector<cv::Scalar> color_palette_;
};

#endif // YOLODETECTOR_H



