#pragma once
#include "opencv2/opencv.hpp"
#include <vector>
#include <string>

//////////////////////////////////////////////////////////////////////////
//该文件中提供仅仅依赖opencv的常用函数

class ImageProcessFunc
{
public:
	//调整亮度对比度,alpha，对比度系数，beta亮度系数，anchor对比度的基准点
	//************************************
	// 函数:    adJustBrightness		
	// 全名:  ImageProcessFunc::adJustBrightness		
	// 返回值:   int		#无意义
	// 参数: cv::Mat & src			#输入图像
	// 参数: double alpha			#对比度调整系数
	// 参数: double beta				#亮度调整像素
	// 参数: double anchor			#对比度调整锚点
	//************************************
	static int adJustBrightness(cv::Mat& src, double alpha, double beta, double anchor);


	//************************************
	// 函数:    makeBoarderConstant		
	// 作用：	为图片描边，不会改变图像尺寸，
	// 全名:  ImageProcessFunc::makeBoarderConstant		
	// 返回值:   int		#
	// 参数: cv::Mat & srcMat			#
	// 参数: unsigned char boarder_value			#
	// 参数: int boarder_width			#
	//************************************
	static int makeBoarderConstant(cv::Mat &srcMat, unsigned char boarder_value, int boarder_width);
	
	//沿着任意角度旋转，幅度
	static void rotate_arbitrarily_angle(cv::Mat &src, cv::Mat &dst, float angle);

	//axis = 0 ,沿着x轴投影到y 轴，axis=1,相反
	static int sumPixels(cv::Mat &srcimg, int axis, std::vector<unsigned int> &resultsVec);

	//************************************
	// 函数:    getAverageBrightness		
	// 作用：	获取图片的平均像素值
	// 全名:  ImageProcessFunc::getAverageBrightness		
	// 返回值:   double		#
	// 参数: cv::Mat src			#
	//************************************
	static double getAverageBrightness(cv::Mat src);
	static double getAveragePixelInRect(cv::Mat& src, cv::Rect &mRect);
	


	//************************************
	// 函数:    getContourRect		
	// 作用：	获得轮廓点的最大外框
	// 全名:  ImageProcessFunc::getContourRect		
	// 返回值:   int		#
	// 参数: std::vector<cv::Point2f> & points_vec			#
	// 参数: cv::Rect & mRect			#
	//************************************
	static int getContourRect(std::vector<cv::Point2f> & points_vec, cv::Rect &mRect);
	static int getContourRect(std::vector<cv::Point> & points_vec, cv::Rect &mRect);

	//************************************
	// 函数:    CropRect		
	// 作用：	裁切rect
	// 全名:  ImageProcessFunc::CropRect		
	// 返回值:   int		#
	// 参数: cv::Rect main_rect			#
	// 参数: cv::Rect & to_crop_rect			#
	//************************************
	static int CropRect(cv::Rect main_rect, cv::Rect &to_crop_rect);

	//************************************
	// 函数:    IsPointInRect		
	// 作用：判断点是否位于Rect内部
	// 全名:  ImageProcessFunc::IsPointInRect
	// 返回值:   bool		#
	// 参数: cv::Point pt			#
	// 参数: cv::Rect rc			#
	//************************************
	static bool IsPointInRect(cv::Point pt, cv::Rect rc);

	//************************************
	// 函数:    getMatFromRotatedRect		
	// 作用：按照rotatedRect 裁切图片
	// 全名:  ImageProcessFunc::getMatFromRotatedRect		
	// 返回值:   int		#
	// 参数: const cv::Mat & src_mat			#
	// 参数: cv::Mat & dst_mat			#
	// 参数: cv::RotatedRect rRc			#
	// 参数: unsigned char border_value			#填充像素值
	//************************************
	static int getMatFromRotatedRect(const cv::Mat &src_mat, cv::Mat &dst_mat, cv::RotatedRect rRc, unsigned char border_value=255);

	//************************************
	// 函数:    rotatePoints		
	// 作用： 旋转points，例如：轮廓
	// 全名:  ImageProcessFunc::rotatePoints		
	// 返回值:   int		#
	// 参数: std::vector<cv::Point2f> & points_vec			#
	// 参数: double angle			#
	//************************************
	int rotatePoints(std::vector<cv::Point2f> & points_vec, double angle,cv::Point2f center_point= cv::Point2f(0,0)); //移动至原点
	int rotatePoints(std::vector<cv::Point> & points_vec, double angle, cv::Point center_point= cv::Point(0,0));
};