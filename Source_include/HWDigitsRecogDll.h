#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#ifdef DLLEXPORTTEMPLATE_EXPORTS
#define DLL_FUNC_API __declspec(dllexport)
#else
#define DLL_FUNC_API __declspec(dllimport)
#endif



class DLL_FUNC_API HWDigitsRecog;
class HWDigitsRecog
{
public:
	HWDigitsRecog();
	~HWDigitsRecog();
	//初始化加载pb模型
	int initial(std::string model_path);

	//采用全局池化层的模型结构(暂未使用)
	int initial_avg(std::string model_path,std::string output_node= "");//output_node = "Conv10/Addbias"
	
	//输入图片的背景为黑色即0值，前景为白色即255，软阈值，应该存在中间值，输入尺寸28x28 单通道
	//src_mat_vec 内部图片必须经过深拷贝，否则结果可能不正确
	int detect_mat(const std::vector<cv::Mat> &src_mat_vec, std::vector<int> &class_index_vec, std::vector<float> &confidence_vec);//imagesize:28x28
	

	int detect_mat_avg(const std::vector<cv::Mat> &src_mat_vec, std::vector<int> &class_index_vec, std::vector<float> &confidence_vec);//imagesize:28x28
private:
	void* pDetector = NULL;
	void* pDetector_avg = NULL;
};

