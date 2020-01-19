#pragma once
#ifndef COMPILER_MSVC
#define COMPILER_MSVC
#endif // !1
#ifndef NOMINMAX
#define NOMINMAX
#endif // !NOMINMAX
#ifndef PLATFORM_WINDOWS
#define PLATFORM_WINDOWS
#endif
#include <vector>
#include <string>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/cc/ops/array_ops.h"
#include <opencv2/opencv.hpp>
//#include <opencv2/dnn.hpp>


class Digits_HWR_CNN
{
public:
	Digits_HWR_CNN();
	~Digits_HWR_CNN();
	int initial(std::string model_path);
	int initial(std::string model_path,std::string output_tensor_name);
	int detect_mat(const std::vector<cv::Mat> &src_mat_vec, std::vector<int> &class_index_vec, std::vector<float> &confidence_vec);//imagesize:28x28
	int detect_mat_avg(const std::vector<cv::Mat> &src_mat_vec, std::vector<int> &class_index_vec, std::vector<float> &confidence_vec);
protected:
	tensorflow::Session* session;
	tensorflow::GraphDef graph_def;
	//tensorflow::Tensor input_tensor; // (DT_FLOAT, TensorShape({ 1, 544, 544, 3 }));
	//std::vector<std::string> output_nodes;
	std::string output_node;
	int m_initial_state = 0;//==0是没有初始化,==1完成初始化
	std::string madel_path;
private:
	int prepare_image_data(std::vector<cv::Mat> src_img_vec, tensorflow::Tensor * dst_tensor);

	int global_average_pool(tensorflow::Tensor src_tensor, Eigen::Tensor<float, 2, Eigen::RowMajor> *dst_tensor);

private:
};