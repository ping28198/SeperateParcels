#include "HWDigitsRecogDll.h"
#include "HandwriteDigitsRecognition.h"

HWDigitsRecog::HWDigitsRecog()
{
	pDetector = new Digits_HWR_CNN;
	pDetector_avg = new Digits_HWR_CNN;
}

HWDigitsRecog::~HWDigitsRecog()
{
	if (pDetector!=NULL)
	{
		delete pDetector;
	}
	if (pDetector != NULL)
	{
		delete pDetector_avg;
	}
}

int HWDigitsRecog::initial(std::string model_path)
{
	return ((Digits_HWR_CNN*)pDetector)->initial(model_path);
}


int HWDigitsRecog::initial_avg(std::string model_path, std::string output_node/*= "Conv10/BiasAdd"*/)
{
	if (output_node.empty())
	{
		output_node = "conv10/BiasAdd";
	}
	return ((Digits_HWR_CNN*)pDetector_avg)->initial(model_path,output_node);
}

int HWDigitsRecog::detect_mat(const std::vector<cv::Mat> &src_mat_vec, std::vector<int> &class_index_vec, std::vector<float> &confidence_vec)
{
	return ((Digits_HWR_CNN*)pDetector)->detect_mat(src_mat_vec, class_index_vec, confidence_vec);
}

int HWDigitsRecog::detect_mat_avg(const std::vector<cv::Mat> &src_mat_vec, std::vector<int> &class_index_vec, std::vector<float> &confidence_vec)
{
	return ((Digits_HWR_CNN*)pDetector_avg)->detect_mat_avg(src_mat_vec, class_index_vec, confidence_vec);
}
