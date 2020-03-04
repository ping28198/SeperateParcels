
#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#ifdef DLLEXPORTTEMPLATE_EXPORTS
#define DLL_FUNC_API __declspec(dllexport)
#else
#define DLL_FUNC_API __declspec(dllimport)
#endif



class DLL_FUNC_API ParcelsRecog;
class ParcelsRecog
{
public:
	ParcelsRecog();
	~ParcelsRecog();

	int initial_model();

	//************************************
	// 函数:    detect_mat		
	// 作用：
	// 全名:  ParcelsRecog::detect_mat		
	// 返回值:   int		#
	// 参数: cv::Mat & pMat			# 输入图像
	// 参数: std::vector<std::vector<double>> & outParcelsRanges			#输出包裹顶点坐标信息，里面的vector放置了一个包裹的4个顶点的坐标
	//************************************
	int detect_mat(cv::Mat &pMat, std::vector<std::vector<cv::Point2f>> &outParcelsRanges);


	//************************************
	// 函数:    detect_mat_		
	// 作用：
	// 全名:  ParcelsRecog::detect_mat_		
	// 返回值:   int		#
	// 参数: unsigned char * pImgData			#图像数据指针
	// 参数: int channels			#图像通道数，1或3
	// 参数: int width			#图像宽度
	// 参数: int height			#图像高度
	// 参数: double * outData			#输出数据缓冲区
	// 参数: int bufferlength			#缓冲区长度
	// 参数: int & parcel_num			#检测到的包裹数量，每8个数据构成一个包裹，数据结构：P1x,P1y,P2x,P2y,P3x,P3y,P4x,P4y,...
	//************************************
	int detect_mat_(unsigned char *pImgData,int channels,int width,int height,double *outData,int bufferlength,int &parcel_num);
private:
	void* pDetector = NULL;
	void* pDetector_avg = NULL;
};

