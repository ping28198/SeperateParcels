#include "tag_detect.h"
#include "yolo_v3.h"
tag_detector::tag_detector()
{
	pDetector = new YOLO_V3;
}

tag_detector::~tag_detector()
{
	if (pDetector != NULL) delete pDetector;
}

int tag_detector::initial(std::string model_path, float score_threshold, int max_instance_per_class)
{
	return ((YOLO_V3*)pDetector)->initial(model_path, score_threshold, max_instance_per_class);
}

int tag_detector::detect_image(const std::string &image_path, std::vector<cv::Rect> &detected_box,
	std::vector<int> &detected_class, std::vector<float> &detected_scores, float score_threshold /*= 0*/)
{
	return ((YOLO_V3*)pDetector)->detect_image(image_path, detected_box, detected_class, detected_scores, score_threshold);
}

int tag_detector::detect_mat(cv::Mat &src_mat, std::vector<cv::Rect> &detected_box, 
	std::vector<int> &detected_class, std::vector<float> &detected_scores, float score_threshold /*= 0*/)
{
	return ((YOLO_V3*)pDetector)->detect_mat(src_mat, detected_box, detected_class, detected_scores, score_threshold);
}
