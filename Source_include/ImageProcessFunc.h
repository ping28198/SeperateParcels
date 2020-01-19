#pragma once
#include "opencv2/opencv.hpp"
#include <vector>
#include <string>

//////////////////////////////////////////////////////////////////////////
//���ļ����ṩ��������opencv�ĳ��ú���

class ImageProcessFunc
{
public:
	//�������ȶԱȶ�,alpha���Աȶ�ϵ����beta����ϵ����anchor�ԱȶȵĻ�׼��
	//************************************
	// ����:    adJustBrightness		
	// ȫ��:  ImageProcessFunc::adJustBrightness		
	// ����ֵ:   int		#������
	// ����: cv::Mat & src			#����ͼ��
	// ����: double alpha			#�Աȶȵ���ϵ��
	// ����: double beta				#���ȵ�������
	// ����: double anchor			#�Աȶȵ���ê��
	//************************************
	static int adJustBrightness(cv::Mat& src, double alpha, double beta, double anchor);


	//************************************
	// ����:    makeBoarderConstant		
	// ���ã�	ΪͼƬ��ߣ�����ı�ͼ��ߴ磬
	// ȫ��:  ImageProcessFunc::makeBoarderConstant		
	// ����ֵ:   int		#
	// ����: cv::Mat & srcMat			#
	// ����: unsigned char boarder_value			#
	// ����: int boarder_width			#
	//************************************
	static int makeBoarderConstant(cv::Mat &srcMat, unsigned char boarder_value, int boarder_width);
	
	//��������Ƕ���ת������
	static void rotate_arbitrarily_angle(cv::Mat &src, cv::Mat &dst, float angle);

	//axis = 0 ,����x��ͶӰ��y �ᣬaxis=1,�෴
	static int sumPixels(cv::Mat &srcimg, int axis, std::vector<unsigned int> &resultsVec);

	//************************************
	// ����:    getAverageBrightness		
	// ���ã�	��ȡͼƬ��ƽ������ֵ
	// ȫ��:  ImageProcessFunc::getAverageBrightness		
	// ����ֵ:   double		#
	// ����: cv::Mat src			#
	//************************************
	static double getAverageBrightness(cv::Mat src);
	static double getAveragePixelInRect(cv::Mat& src, cv::Rect &mRect);
	


	//************************************
	// ����:    getContourRect		
	// ���ã�	����������������
	// ȫ��:  ImageProcessFunc::getContourRect		
	// ����ֵ:   int		#
	// ����: std::vector<cv::Point2f> & points_vec			#
	// ����: cv::Rect & mRect			#
	//************************************
	static int getContourRect(std::vector<cv::Point2f> & points_vec, cv::Rect &mRect);
	static int getContourRect(std::vector<cv::Point> & points_vec, cv::Rect &mRect);

	//************************************
	// ����:    CropRect		
	// ���ã�	����rect
	// ȫ��:  ImageProcessFunc::CropRect		
	// ����ֵ:   int		#
	// ����: cv::Rect main_rect			#
	// ����: cv::Rect & to_crop_rect			#
	//************************************
	static int CropRect(cv::Rect main_rect, cv::Rect &to_crop_rect);

	//************************************
	// ����:    IsPointInRect		
	// ���ã��жϵ��Ƿ�λ��Rect�ڲ�
	// ȫ��:  ImageProcessFunc::IsPointInRect
	// ����ֵ:   bool		#
	// ����: cv::Point pt			#
	// ����: cv::Rect rc			#
	//************************************
	static bool IsPointInRect(cv::Point pt, cv::Rect rc);

	//************************************
	// ����:    getMatFromRotatedRect		
	// ���ã�����rotatedRect ����ͼƬ
	// ȫ��:  ImageProcessFunc::getMatFromRotatedRect		
	// ����ֵ:   int		#
	// ����: const cv::Mat & src_mat			#
	// ����: cv::Mat & dst_mat			#
	// ����: cv::RotatedRect rRc			#
	// ����: unsigned char border_value			#�������ֵ
	//************************************
	static int getMatFromRotatedRect(const cv::Mat &src_mat, cv::Mat &dst_mat, cv::RotatedRect rRc, unsigned char border_value=255);

	//************************************
	// ����:    rotatePoints		
	// ���ã� ��תpoints�����磺����
	// ȫ��:  ImageProcessFunc::rotatePoints		
	// ����ֵ:   int		#
	// ����: std::vector<cv::Point2f> & points_vec			#
	// ����: double angle			#
	//************************************
	int rotatePoints(std::vector<cv::Point2f> & points_vec, double angle,cv::Point2f center_point= cv::Point2f(0,0)); //�ƶ���ԭ��
	int rotatePoints(std::vector<cv::Point> & points_vec, double angle, cv::Point center_point= cv::Point(0,0));
};