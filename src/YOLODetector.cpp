// src/YOLODetector.cpp
#include "YOLODetector.h"
#include <opencv2/dnn.hpp>
#include <random>

// 按模型实际类别数和名称修改
static const std::vector<std::string> CLASS_NAMES = {
    "class_name1","class_name2","class_name3","class_name4"
};

YOLODetector::YOLODetector(const std::string& model_path,
                           float conf_thresh,
                           float iou_thresh)
  : env_(ORT_LOGGING_LEVEL_ERROR,"yolo"),
    session_(nullptr),
    conf_thresh_(conf_thresh),
    iou_thresh_(iou_thresh),
    input_width_(640),
    input_height_(640)
{
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_ = new Ort::Session(env_,
                                model_path.c_str(),
                                session_options_);
    // 随机颜色
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> dist(0,255);
    for(size_t i=0;i<CLASS_NAMES.size();++i)
      color_palette_.emplace_back(dist(rng),dist(rng),dist(rng));
}

YOLODetector::~YOLODetector(){
  delete session_;
}

void YOLODetector::letterbox(const cv::Mat& src, cv::Mat& dst,
                             float& scale, int& top, int& left){
    int h=src.rows, w=src.cols;
    scale = std::min(float(input_width_)/w,
                     float(input_height_)/h);
    int new_w=int(w*scale), new_h=int(h*scale);
    cv::Mat resized;
    cv::resize(src,resized,cv::Size(new_w,new_h));
    left=(input_width_-new_w)/2;
    top=(input_height_-new_h)/2;
    cv::copyMakeBorder(resized,dst,
       top, input_height_-new_h-top,
       left, input_width_-new_w-left,
       cv::BORDER_CONSTANT,cv::Scalar(114,114,114));
}

std::vector<Detection> YOLODetector::detect(const std::string& image_path){
    cv::Mat img=cv::imread(image_path);
    if(img.empty())return{};
    cv::Mat box;
    float scale; int top,left;
    letterbox(img,box,scale,top,left);

    cv::Mat rgb; cv::cvtColor(box,rgb,cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb,CV_32F,1.f/255.f);
    std::vector<cv::Mat> ch(3);
    cv::split(rgb,ch);
    std::vector<float> in_vals(input_width_*input_height_*3);
    int ch_sz=input_width_*input_height_;
    for(int c=0;c<3;++c)
      memcpy(in_vals.data()+c*ch_sz,ch[c].data,
             ch_sz*sizeof(float));

    std::array<int64_t,4> dims={1,3,
      int64_t(input_height_),int64_t(input_width_)};
    auto mem=Ort::MemoryInfo::CreateCpu(
      OrtDeviceAllocator,OrtMemTypeCPU);
    Ort::Value tensor=Ort::Value::CreateTensor<float>(
      mem,in_vals.data(),in_vals.size(),
      dims.data(),dims.size());

    const char* in_name="images";
    const char* out_name="output0";
    auto outputs=session_->Run(
      Ort::RunOptions{nullptr},
      &in_name,&tensor,1,
      &out_name,1);

    // 解析输出 [1,8,8400]
    float* data=outputs[0].GetTensorMutableData<float>();
    auto info=outputs[0].GetTensorTypeAndShapeInfo();
    auto shape=info.GetShape();
    int64_t channels=shape[1],num_preds=shape[2];

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> cls_ids;
    for(int64_t p=0;p<num_preds;++p){
      // 找最优类别
      float best=0;int bid=0;
      for(int c=4;c<channels;++c){
        float sc=data[c*num_preds+p];
        if(sc>best){best=sc;bid=c-4;}
      }
      if(best<conf_thresh_)continue;
      // 反向映射到原图
      float cx=data[0*num_preds+p],
            cy=data[1*num_preds+p],
            w  =data[2*num_preds+p],
            h  =data[3*num_preds+p];
      float x0=(cx-left)/scale - w/2/scale;
      float y0=(cy-top)/scale  - h/2/scale;
      float w0=w/scale, h0=h/scale;
      int x=int(x0), y=int(y0),
          iw=int(w0), ih=int(h0);
      boxes.emplace_back(x,y,iw,ih);
      scores.push_back(best);
      cls_ids.push_back(bid);
    }
    std::vector<int> idxs;
    cv::dnn::NMSBoxes(boxes,scores,conf_thresh_,iou_thresh_,idxs);
    std::vector<Detection> res;
    for(int i:idxs){
      Detection d;
      d.class_id=cls_ids[i];
      d.class_name=CLASS_NAMES[d.class_id];
      d.score=scores[i];
      d.x=boxes[i].x; d.y=boxes[i].y;
      d.width=boxes[i].width;
      d.height=boxes[i].height;
      res.push_back(d);
    }
    return res;
}

std::vector<Detection> YOLODetector::detectAndSave(
    const std::string& image_path,
    const std::string& output_path){
    auto dets=detect(image_path);
    cv::Mat img=cv::imread(image_path);
    for(auto& d:dets){
      cv::Scalar col=color_palette_[d.class_id];
      cv::rectangle(img,
        cv::Point(d.x,d.y),
        cv::Point(d.x+d.width,d.y+d.height),
        col,2);
      std::string lbl=d.class_name+":"+
        std::to_string(d.score).substr(0,4);
      int bl;
      auto ts=cv::getTextSize(lbl,
        cv::FONT_HERSHEY_SIMPLEX,
        0.5,1,&bl);
      cv::rectangle(img,
        cv::Point(d.x,d.y-ts.height-bl),
        cv::Point(d.x+ts.width,d.y),
        col,cv::FILLED);
      cv::putText(img,lbl,
        cv::Point(d.x,d.y-bl),
        cv::FONT_HERSHEY_SIMPLEX,
        0.5,cv::Scalar(0,0,0),1);
    }
    cv::imwrite(output_path,img);
    return dets;
}


