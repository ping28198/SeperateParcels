#pragma once
#include <opencv.hpp>
#include <string>

#ifdef DLLEXPORTTEMPLATE_EXPORTS
#define DLL_FUNC_API __declspec(dllexport)
#else
#define DLL_FUNC_API __declspec(dllimport)
#endif
//DLL_FUNC_API int wavelet_transform(cv::Mat srcMat, cv::Mat &dstMat, double threshold=2.0, int coef_direc=4, const std::string wavelet_name="haar");

class DLL_FUNC_API WaveletTransformer
{
public:
	int initial();
	int wavelet_transform(cv::Mat srcMat, cv::Mat &dstMat, double threshold = 2.0, int coef_direc = 4, const std::string wavelet_name = "haar");
	static int globalInitial();
	~WaveletTransformer();
};

