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
	//��ʼ������pbģ��
	int initial(std::string model_path);

	//����ȫ�ֳػ����ģ�ͽṹ(��δʹ��)
	int initial_avg(std::string model_path,std::string output_node= "");//output_node = "Conv10/Addbias"
	
	//����ͼƬ�ı���Ϊ��ɫ��0ֵ��ǰ��Ϊ��ɫ��255������ֵ��Ӧ�ô����м�ֵ������ߴ�28x28 ��ͨ��
	//src_mat_vec �ڲ�ͼƬ���뾭����������������ܲ���ȷ
	int detect_mat(const std::vector<cv::Mat> &src_mat_vec, std::vector<int> &class_index_vec, std::vector<float> &confidence_vec);//imagesize:28x28
	

	int detect_mat_avg(const std::vector<cv::Mat> &src_mat_vec, std::vector<int> &class_index_vec, std::vector<float> &confidence_vec);//imagesize:28x28
private:
	void* pDetector = NULL;
	void* pDetector_avg = NULL;
};

