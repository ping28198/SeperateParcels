#pragma once
#include <string>
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"


#define default_pixel_threshold 40
//#define CUT_PARCEL_BOX_DEBUG

class CutParcelBox
{
public:
	CutParcelBox();

	//default boxSizeThreshold=1,,binaryThreshold=50
	//************************************
	// Method:    getMailBox_Rect_Raw
	// FullName:  CutParcelBox::getMailBox_Rect_Raw
	// Access:    public 
	// Returns:   int =1 ������=0 ʧ��
	// Parameter: const void * pImgData
	// Parameter: int iw 
	// Parameter: int ih
	// Parameter: int ic
	// Parameter: int * RectPoints
	// Parameter: int applid_rotate //ʯͷ��תͼƬ����ʹ��Ŀ���뷽�����Ǻ�
	// Parameter: double boxSizeThreshold //Ŀ��ռͼƬ���ǧ�ֱ���ֵ
	// Parameter: double binaryThreshold //�Ҷȷָ���ֵ
	//************************************
	int getMailBox_Rect_Raw(const void *pImgData, int iw, int ih, int ic, int *RectPoints,
		int applid_rotate=1, double boxSizeThreshold = 1, double binaryThreshold = default_pixel_threshold);

	//************************************
	// Method:    getMailBox
	// FullName:  CutParcelBox::getMailBox
	// Access:    public 
	// Returns:   int
	// Parameter: std::string & srcImgPath
	// Parameter: std::string & dstImgPath
	// Parameter: int applid_rotate //ʯͷ��תͼƬ����ʹ��Ŀ���뷽�����Ǻ�
	// Parameter: double boxSizeThreshold //Ŀ��ռͼƬ���ǧ�ֱ���ֵ
	// Parameter: double binaryThreshold //�Ҷȷָ���ֵ
	//************************************
	int getMailBox(std::string &srcImgPath, std::string &dstImgPath, int applid_rotate = 1,
		double boxSizeThreshold=1, double binaryThreshold= default_pixel_threshold);

	//************************************
	// Method:    getMailBox_c
	// FullName:  CutParcelBox::getMailBox_c
	// Access:    public 
	// Returns:   int
	// Parameter: const char * pSrcImgPath
	// Parameter: const char * pDstImgPath
	// Parameter: int applid_rotate //ʯͷ��תͼƬ����ʹ��Ŀ���뷽�����Ǻ�
	// Parameter: double boxSizeThreshold //Ŀ��ռͼƬ���ǧ�ֱ���ֵ
	// Parameter: double binaryThreshold //�Ҷȷָ���ֵ
	//************************************
	int getMailBox_c(const char* pSrcImgPath, const char* pDstImgPath, int applid_rotate = 1,
		double boxSizeThreshold = 1, double binaryThreshold = default_pixel_threshold);

	//��Ӧbox����״��С��
	//************************************
	// Method:    getMailBox_RtRect
	// FullName:  CutParcelBox::getMailBox_RtRect
	// Access:    public 
	// Returns:   int
	// Parameter: cv::Mat & srcMat
	// Parameter: int applid_rotate //ʯͷ��תͼƬ����ʹ��Ŀ���뷽�����Ǻ�
	// Parameter: double boxSizeThreshold //Ŀ��ռͼƬ���ǧ�ֱ���ֵ
	// Parameter: double binaryThreshold //�Ҷȷָ���ֵ
	//************************************
	int getMailBox_RtRect(cv::Mat &srcMat, cv::RotatedRect &dstRtRect, double boxSizeThreshold = 1,
		double binaryThreshold = default_pixel_threshold);

	//************************************
	// Method:    getMailBox_Mat
	// FullName:  CutParcelBox::getMailBox_Mat
	// Access:    public 
	// Returns:   int
	// Parameter: cv::Mat & srcMat
	// Parameter: cv::Mat & dstMat
	// Parameter: int applid_rotate //ʯͷ��תͼƬ����ʹ��Ŀ���뷽�����Ǻ�
	// Parameter: double boxSizeThreshold //Ŀ��ռͼƬ���ǧ�ֱ���ֵ
	// Parameter: double binaryThreshold //�Ҷȷָ���ֵ
	//************************************
	int getMailBox_Mat(cv::Mat &srcMat, cv::Mat &dstMat, int applid_rotate = 1, 
		double boxSizeThreshold = 1, double binaryThreshold = default_pixel_threshold);
	
	//************************************
	// Method:    getMailBox_side
	// FullName:  CutParcelBox::getMailBox_side
	// Access:    public 
	// Returns:   int
	// Parameter: cv::Mat & srcMat
	// Parameter: cv::Mat & dstMat
	// Parameter: double boxSizeThreshold  //Ŀ��ռͼƬ���ǧ�ֱ���ֵ
	// Parameter: double binaryThreshold //�ݶ���ֵ�ָĬ��Ϊ5
	//************************************
	int getMailBox_side(cv::Mat &srcMat, cv::Mat &dstMat,
		double boxSizeThreshold = 1.0, double binaryThreshold = 5.0);
	int getMailBox_side(cv::Mat &srcMat, cv::Rect &dstRect,
		double boxSizeThreshold = 1.0, double binaryThreshold = 5.0);

public:
	static int findRect(std::vector<cv::Point> contour, cv::Rect &mrect);
	static int getMatFromRotateRect(const cv::Mat &src_mat, cv::Mat &dst_mat, cv::RotatedRect rRc);
	static int CropRect(cv::Rect main_rect, cv::Rect &to_crop_rect);
};
