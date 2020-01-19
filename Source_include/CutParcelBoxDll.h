#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#ifdef DLLEXPORTTEMPLATE_EXPORTS
#define DLL_CUT_PARCEL_API __declspec(dllexport)
#else
#define DLL_CUT_PARCEL_API __declspec(dllimport)
#endif



//class DLL_CUT_PARCEL_API CutParcelBoxDll;

class DLL_CUT_PARCEL_API CutParcelBoxDll
{
public:
	CutParcelBoxDll();

	//default boxSizeThreshold=1,,binaryThreshold=50
	//applid_rotate 是否旋转图片获取包裹最小box
	//************************************
	// Method:    getMailBox_top
	// FullName:  CutParcelBoxDll::getMailBox_top
	// Access:    public static 
	// Returns:   int
	// Parameter: const void * pImgData
	// Parameter: int iw
	// Parameter: int ih
	// Parameter: int ic
	// Parameter: int * RectPoints
	// Parameter: int applid_rotate
	// Parameter: double boxSizeThreshold 目标面积占图片面积千分比阈值
	// Parameter: double binaryThreshold 灰度阈值
	//************************************
	static int getMailBox_top_c(const void *pImgData, int iw, int ih, int ic, int *RectPoints,
		int applid_rotate = 1, double boxSizeThreshold = 1.0, double binaryThreshold = 50);//RectPoints：int数组，长度8

	static int getMailBox_top_path(std::string &srcImgPath, std::string &dstImgPath, int applid_rotate = 1,
		double boxSizeThreshold = 1.0, double binaryThreshold = 50);

	static int getMailBox_top_path_c(const char* pSrcImgPath, const char* pDstImgPath, int applid_rotate = 1,
		double boxSizeThreshold = 1.0, double binaryThreshold = 50);

	static int getMailBox_top_rect(cv::Mat &srcMat, cv::RotatedRect &dstRtRect, double boxSizeThreshold = 1.0,
		double binaryThreshold = 50);

	//************************************
	// Method:    getMailBox_top_mat
	// FullName:  CutParcelBoxDll::getMailBox_top
	// Access:    public static 
	// Returns:   int
	// Parameter: cv::Mat & srcMat
	// Parameter: cv::Mat & dstMat
	// Parameter: int applid_rotate
	// Parameter: double boxSizeThreshold 目标面积占图片面积千分比阈值
	// Parameter: double binaryThreshold 灰度阈值
	//************************************
	static int getMailBox_top_mat(cv::Mat &srcMat, cv::Mat &dstMat, int applid_rotate = 1,
		double boxSizeThreshold = 1.0, double binaryThreshold = 50);

	//用于侧面和前后面的分割
	//************************************
	// Method:    getMailBox_side_mat
	// FullName:  CutParcelBoxDll::getMailBox_side
	// Access:    public static 
	// Returns:   int
	// Parameter: cv::Mat & srcMat
	// Parameter: cv::Mat & dstMat
	// Parameter: double boxSizeThreshold 目标面积占图片面积千分比阈值
	// Parameter: double binaryThreshold 梯度阈值
	//************************************
	static int getMailBox_side_mat(cv::Mat &srcMat, cv::Mat &dstMat,
		double boxSizeThreshold = 1.0, double binaryThreshold = 5.0);

	static int getMailBox_side_rect(cv::Mat &srcMat, cv::Rect &dstRect,
		double boxSizeThreshold = 1.0, double binaryThreshold = 5.0);

};