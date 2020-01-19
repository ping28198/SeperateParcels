#include "HandwriteDigitsRecognition.h"


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
#include <tensorflow/cc/ops/nn_ops.h>



Digits_HWR_CNN::Digits_HWR_CNN()
{
	session = NULL;
	//input_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 544, 544, 3 }));
	m_initial_state = 0;
	output_node = "activation_1/Softmax";
}

Digits_HWR_CNN::~Digits_HWR_CNN()
{
	session->Close();
}

int Digits_HWR_CNN::initial(std::string model_path)
{
	if (session != NULL) return 1;
	//logger.TraceInfo("enter_initial");

	tensorflow::Status status = NewSession(tensorflow::SessionOptions(), &session);
	if (!status.ok()) {
		std::cerr << status.ToString() << std::endl;
		return 0;
	}
	else {
		std::cout << "Session created successfully" << std::endl;
	}
	//logger.TraceInfo("new_session_over");

	// 读取二进制的模型文件到graph中
	status = ReadBinaryProto(tensorflow::Env::Default(), model_path, &graph_def);
	if (!status.ok()) {
		std::cerr << status.ToString() << std::endl;
		return 0;
	}
	else {
		std::cout << "Load graph protobuf successfully" << std::endl;
	}
	//logger.TraceInfo("read_model_over");

	// 将graph加载到session
	status = session->Create(graph_def);
	if (!status.ok()) {
		std::cerr << status.ToString() << std::endl;
		return 0;
	}
	else {
		std::cout << "Add graph to session successfully" << std::endl;
	}
	//logger.TraceInfo("create_graph_over");
	
	m_initial_state = 1;
	//logger.TraceInfo("return yolo initial");
	return 1;
}

int Digits_HWR_CNN::initial(std::string model_path, std::string output_tensor_name/*= "activation_1/Softmax"*/)
{
	output_node = output_tensor_name;
	return initial(model_path);
}

int Digits_HWR_CNN::detect_mat(const std::vector<cv::Mat> &src_mat_vec, std::vector<int> &class_index_vec, std::vector<float> &confidence_vec)
{

	if (src_mat_vec.empty()) return -1;
	if (m_initial_state == 0) return -1;
	cv::Size input_image_size(28,28);
	cv::Size input_shape(28, 28);
	cv::Mat resized_mat;
	const int img_num = src_mat_vec.size();

	tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ img_num, 28, 28, 1 }));
	prepare_image_data(src_mat_vec, &input_tensor);
	std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
	{ "input_1", input_tensor }
	};
	// 输出outputs
	std::vector<tensorflow::Tensor> outputs;

	std::vector<std::string> output_nodes;
	output_nodes.push_back(output_node);
	// 运行会话，最终结果保存在outputs中
	tensorflow::Status status = session->Run(inputs, { output_nodes }, {}, &outputs);
	if (!status.ok()) {
		std::cerr << status.ToString() << std::endl;
		return -1;
	}
	else {
		//std::cout << "Run session successfully" << endl;
	}

	auto input_tensor_mapped = outputs[0].tensor<float, 2>(); //tensormap
	auto src_tensor = Eigen::Tensor<float, 2, Eigen::RowMajor>(input_tensor_mapped); //转为tensor

	int dims = src_tensor.NumDimensions;//=2



	long batch_size = src_tensor.dimension(0);
	long class_num = src_tensor.dimension(1);
	std::vector<int> class_index;
	std::vector<float> confidence_;
	for (int m=0;m<batch_size;m++)
	{
		float tt = 0;
		int index_ = 0;
		for (int i=0;i<class_num;i++)
		{
			float a = src_tensor(m, i);
			if (tt<a)
			{
				tt = a;
				index_ = i;
			}
		}
		class_index.push_back(index_);
		confidence_.push_back(tt);
	}
	class_index_vec = class_index;
	confidence_vec = confidence_;
	return 1;

}

int Digits_HWR_CNN::detect_mat_avg(const std::vector<cv::Mat> &src_mat_vec, std::vector<int> &class_index_vec, std::vector<float> &confidence_vec)
{
	if (src_mat_vec.empty()) return -1;
	if (m_initial_state == 0) return -1;
	cv::Size input_image_size(28, 28);
	cv::Size input_shape(28, 28);
	cv::Mat resized_mat;
	int batch_size_ = src_mat_vec.size();
	const int img_num = src_mat_vec.size();

	tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ img_num, 28, 28, 1 }));
	prepare_image_data(src_mat_vec, &input_tensor);
	std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
	{ "input_1", input_tensor }
	};
	// 输出outputs
	std::vector<tensorflow::Tensor> outputs;

	std::vector<std::string> output_nodes;
	output_nodes.push_back(output_node);
	// 运行会话，最终结果保存在outputs中
	tensorflow::Status status = session->Run(inputs, { output_nodes }, {}, &outputs);
	if (!status.ok()) {
		std::cerr << status.ToString() << std::endl;
		return -1;
	}
	else {
		//std::cout << "Run session successfully" << endl;
	}
	Eigen::Tensor<float, 2, Eigen::RowMajor> output_eig_tensor = { batch_size_ ,10};
	global_average_pool(outputs[0], &output_eig_tensor);

	output_eig_tensor = output_eig_tensor.exp();

	for (int i=0;i<batch_size_;i++)
	{
		float sum_v = 0;
		for (int j=0;j<10;j++)
		{
			sum_v += output_eig_tensor(i, j);
		}
		for (int j = 0; j < 10; j++)
		{
			output_eig_tensor(i, j) = output_eig_tensor(i, j)/sum_v;
		}

	}

	//auto input_tensor_mapped = outputs[0].tensor<float, 2>(); //tensormap
	auto src_tensor = output_eig_tensor; //转为tensor

	int dims = src_tensor.NumDimensions;//=2



	long batch_size = src_tensor.dimension(0);
	long class_num = src_tensor.dimension(1);
	std::vector<int> class_index;
	std::vector<float> confidence_;
	for (int m = 0; m < batch_size; m++)
	{
		float tt = 0;
		int index_ = 0;
		for (int i = 0; i < class_num; i++)
		{
			float a = src_tensor(m, i);
			if (tt < a)
			{
				tt = a;
				index_ = i;
			}
		}
		class_index.push_back(index_);
		confidence_.push_back(tt);
	}
	class_index_vec = class_index;
	confidence_vec = confidence_;
	return 1;
}

int Digits_HWR_CNN::prepare_image_data(std::vector<cv::Mat> src_img_vec, tensorflow::Tensor * dst_tensor)
{
	if (src_img_vec.empty()) return 0;
	const int img_num = src_img_vec.size();
	const int iw = 28;
	const int ih = 28;
	const int channels = 1;

	auto input_tensor_mapped = dst_tensor->tensor<float, 4>();
	for (int n = 0; n < img_num; n++)
	{
		cv::Mat m_ = src_img_vec[n];
		if (m_.rows != ih || m_.cols != iw) 
			cv::resize(m_, m_, cv::Size(iw, ih), 0, 0, cv::INTER_AREA);
		if (m_.channels()==3)
		{
			cv::cvtColor(m_, m_, cv::COLOR_BGR2GRAY);
		}
		const tensorflow::uint8* source_data = m_.data;

		for (int i = 0; i < iw; i++) {
			const tensorflow::uint8* source_row = source_data + (i * iw * channels);
			for (int j = 0; j < ih; j++) {
				const tensorflow::uint8* source_pixel = source_row + (j * channels);
				for (int c = 0; c < channels; c++) {
					const tensorflow::uint8* source_value = source_pixel + c;
					input_tensor_mapped(n, i, j, c) = (float)(*source_value)/255.0f;
				}
			}
		}



	}
	
	


	return img_num;






}

int Digits_HWR_CNN::global_average_pool(tensorflow::Tensor src_tensor, Eigen::Tensor<float,2, Eigen::RowMajor> *dst_tensor)
{
	auto input_tensor_mapped = src_tensor.tensor<float, 4>(); //tensormap
	auto src_eig_tensor = Eigen::Tensor<float, 4, Eigen::RowMajor>(input_tensor_mapped); //转为tensor
	assert(src_tensor.dims == 4);
	int batch_size = src_eig_tensor.dimension(0);
	int width = src_eig_tensor.dimension(1);
	int height = src_eig_tensor.dimension(2);
	int channels = src_eig_tensor.dimension(3);
	for (int b = 0; b < batch_size; b++)
	{
		for (int c =0; c < channels; c++)
		{
			float ave_value = 0;
			for (int w = 0; w < width; w++)
			{
				for (int h = 0; h < height; h++)
				{
					ave_value += src_eig_tensor(b, w, h, c);
				}
			}
			ave_value /= (width*height);
			(*dst_tensor)(b, c) = ave_value;
		}
	}
	return 1;
}

