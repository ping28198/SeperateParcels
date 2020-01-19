#pragma once
//#define PLATFORM_WINDOWS   // 指定使用tensorflow/core/platform/windows/cpu_info.h
#ifndef COMPILER_MSVC
#define COMPILER_MSVC
#endif // !1
#ifndef NOMINMAX
#define NOMINMAX
#endif // !NOMINMAX
#ifndef PLATFORM_WINDOWS
#define PLATFORM_WINDOWS
#endif

   // 指定使用tensorflow/core/platform/windows/cpu_info.h
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/cc/ops/array_ops.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <time.h>
#include "math.h"


class YOLO_V3
{
public:
	YOLO_V3();
	~YOLO_V3();
	int initial(std::string model_path, float score_threshold = 0.3, int max_instance_per_class = 1);
	int detect_image(const std::string &image_path,std::vector<cv::Rect> &detected_box,
		std::vector<int> &detected_class, std::vector<float> &detected_scores, float score_threshold=0);
	int detect_mat(cv::Mat &src_mat, std::vector<cv::Rect> &detected_box,
		std::vector<int> &detected_class, std::vector<float> &detected_scores, float score_threshold=0);
private:
	
	tensorflow::Session* session;
	tensorflow::GraphDef graph_def;
	tensorflow::Tensor input_tensor; // (DT_FLOAT, TensorShape({ 1, 544, 544, 3 }));
	std::vector<std::string> output_nodes;
	std::vector<cv::Size> anchors;
	float m_score_threshold =0.3;
	int m_max_instance_per_class=1;
	int m_initial_state = 0;//==0是没有初始化,==1完成初始化
private:
	
	int resize_image_pad(cv::Mat srcimg, cv::Size msize, cv::Mat &dstimg);
	int yolo_eval(std::vector<tensorflow::Tensor> inputs, std::vector<cv::Size> anchors, int num_class,
		cv::Size image_shape, size_t max_box, float score_threshold, float iou_threshold,
		std::vector<cv::Rect> &box_detected, std::vector<float> &box_scores, 
		std::vector<int> &class_detected);
	int yolo_eval_layer(tensorflow::Tensor input_t, std::vector<cv::Size> anchors, int num_class,
		cv::Size input_shape, std::vector<Eigen::Tensor<float, 2, 1>> &boxes, 
		std::vector<Eigen::Tensor<float, 2, 1>> &scores);
	int yolo_eval_layer_tf(tensorflow::Tensor input_t, std::vector<cv::Size> anchors, int num_class,
		cv::Size input_shape, std::vector<Eigen::Tensor<float, 2, 1>> &boxes,
		std::vector<Eigen::Tensor<float, 2, 1>> &scores);
	int crop_rect(cv::Rect main_rect, cv::Rect &to_crop_rect);

	int prepare_image_data(cv::Mat src_img, cv::Size input_shape, tensorflow::Tensor * dst_tensor);
};

