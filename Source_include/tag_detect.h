#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#ifdef DLLEXPORTTEMPLATE_EXPORTS
#define DLL_FUNC_API __declspec(dllexport)
#else
#define DLL_FUNC_API __declspec(dllimport)
#endif



class DLL_FUNC_API tag_detector;
class tag_detector
{
public:
	tag_detector();
	~tag_detector();
	int initial(std::string model_path, float score_threshold, int max_instance_per_class);
	int detect_image(const std::string &image_path, std::vector<cv::Rect> &detected_box,
		std::vector<int> &detected_class, std::vector<float> &detected_scores, float score_threshold = 0);
	int detect_mat(cv::Mat &src_mat, std::vector<cv::Rect> &detected_box,
		std::vector<int> &detected_class, std::vector<float> &detected_scores, float score_threshold = 0);
private:
	void* pDetector=NULL;
};
