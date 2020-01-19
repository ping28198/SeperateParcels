#pragma once
#include "tesseract/baseapi.h"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "HWDigitsRecogDll.h"

//#define ROTATE_IMG_DEBUG  //用于打印标签旋转调试
//#define POSTCODE_ROI_DEBUG //用于打印标签邮编定位调试
//#define OCR_DEBUG //用于ocr识别邮编调试
//#define POSTCODE_BOX_DEBUG //用于手写邮编识别调试

//用于标签匹配
class MatchDataStruct
{
public:
	cv::Mat descriptors2;
	cv::Mat descriptors3;
	cv::Mat descriptors_handwrite_addr;
	std::vector<cv::KeyPoint> keypoints2;
	std::vector<cv::KeyPoint> keypoints3;
	std::vector<cv::KeyPoint> keypoints_handwrite_addr;
	int loadMatchData(const std::string &xmlfile);
	int saveMatchData(const std::string &xmlfile);
	int getMatchDataFromImg_tagRotate_SIFT(const std::string &refImg1, const std::string &refImg2);
	int getMatchDataFromImg_tagRotate_SURF(const std::string &refImg1, const std::string &refImg2);
	int getMatchDataFromImg_handwrite_addr(const std::string &refImg);
};

//用于ocr的配置类
class OcrAlgorithm_config
{
public:
	std::string tess_data_path;
	std::string template_img_path;
	std::string ORB_template_img1_path;//标签图样1
	std::string ORB_template_img2_path; //标签图样2
	std::string detect_model_file_path;
	std::string handwrite_ocr_model_path;
	std::string handwrite_ref_img1_path;//手写框图样1
	MatchDataStruct match_data;
	void *pTess;
	void *pHWDigitsRecog;
	void *pTagDetector;
	void *pLogger;
	double TagDetectConfidence;
	double HandwriteDigitsConfidence;
	int max_instance_per_class;
	int Run_OCR_on_standard_tag; //是否检测标准标签邮编
	int Run_OCR_on_handwrite_box; //是否检测手写邮编
	int Run_OCR_on_unknown_tag; //是否检测任意标签
	int is_test_model; //是否为测试模式，仅为测试验收使用
	OcrAlgorithm_config()
	{
		pTess = NULL;
		pHWDigitsRecog = NULL;
		pTagDetector = NULL;
		pLogger = NULL;
		HandwriteDigitsConfidence = 0.85;
		TagDetectConfidence = 0.3;
		max_instance_per_class = 1;
		Run_OCR_on_handwrite_box = 1;
		Run_OCR_on_standard_tag = 1;
		is_test_model = 0;
	}
};





class OcrAlgorithm
{
public:
	OcrAlgorithm();
	int getOcrRoi(cv::Mat& src_img, std::vector<cv::Mat> &roi_imgs);
	int runOcr(cv::Mat& RoiMat, char* pResults, size_t bufferlenth, tesseract::TessBaseAPI* pTess=NULL);
	int getOcrResultString(cv::Mat src_img, tesseract::TessBaseAPI *psTess, std::string &resultstring, OcrAlgorithm_config *pConfig);
	int rotateImg_SIFT(cv::Mat &src_img, cv::Mat &dst_img1, cv::Mat &dst_img2, OcrAlgorithm_config *pConfig);
	int rotateImg_SURF(cv::Mat &src_img, cv::Mat &dst_img1, cv::Mat &dst_img2, OcrAlgorithm_config *pConfig);
	//从本地加载存储的特征点数据
	int loadMatchData(const std::string &xmlfile,cv::Mat &descriptors2, cv::Mat &descriptors3, 
		std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::KeyPoint> &keypoints3);

	int saveMatchData(const std::string &xmlfile, cv::Mat &descriptors2, cv::Mat &descriptors3,
		std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::KeyPoint> &keypoints3);
public:
	static int getParcelBoxFromSick(std::string xml_path, cv::RotatedRect &parcel_box);
	static void rotate_arbitrarily_angle(cv::Mat &src, cv::Mat &dst, float angle);
	double postcodeStringScore(std::string srcStr, std::string &resultStr);//ceshi
	static int sumPixels(cv::Mat &srcimg, int axis, std::vector<unsigned int> &resultsVec); //axis =0 x轴，=1 y轴
	static int adJustBrightness(cv::Mat& src, double alpha, double beta, double anchor = 0);//alpha对比度调整 <1 降低对比度，beta亮度调整，<0变暗, anchor 锚点
protected:

private:
	int _getOcrRoi(cv::Mat& src_img, std::vector<cv::Mat> &roi_imgs);
	int _runOcr(cv::Mat& RoiMat, char* pResults, size_t bufferlenth);
	int _runOcrPreload(cv::Mat& RoiMat,char* pResults, size_t bufferlenth, tesseract::TessBaseAPI* pTess);
	int getBarcodePos(cv::Mat &src_img, cv::Rect &bRect);
	int rotateImg(cv::Mat &src_img, cv::Mat &dst_img);
	int rotateImg_v1(cv::Mat &src_img, cv::Mat &dst_img);
	int rotateImg_v2(cv::Mat &src_img, cv::Mat &dst_img, double mthreshold);//mthreshold 默认为2.5
	double rotateImg_ORB(cv::Mat src_img, cv::Mat referenceMat, cv::Mat &dst_img); //返回平均距离
	
	int getPostcodeRoi(cv::Mat &srcImg, std::vector<cv::Rect> &dstRects, cv::Rect mrect, std::string tmplatePath);
	int getPostcodeRoi_TPmatch(cv::Mat &srcImg,  cv::Rect barcoderect, 
		std::string templateImgPath, std::vector<cv::Rect> &roiRects);
	int getPostcodeRoiInRectImg(cv::Mat srcImg,cv::Rect sRect, cv::Rect &roiRect);
	int getPostcodeRoiInRectImg_accordPos(cv::Mat srcImg, cv::Rect sRect,int anchor_row, cv::Rect &roiRect);//anchor_row：给定的邮编所在行
	int getPostcodeRoiInRectImg_SiftMatch(cv::Mat srcImg, cv::Rect sRect, cv::Rect &roiRect);
	
	
	int rotatePoints(std::vector<cv::Point2f> & points_vec, double angle); //移动至原点
	int rotatePoints(std::vector<cv::Point> & points_vec, double angle);
	double getOptimumRotateAngle(std::vector<cv::Point2f> & points_vec); //步长旋转，直到达到最优，points_vec也被旋转
	double ratio_w_h(std::vector<cv::Point2f> & points_vec);
	double getAveragePixelInRect(cv::Mat& src, cv::Rect &mRect);
	int getContourRect(std::vector<cv::Point2f> & points_vec, cv::Rect &mRect);
	
	double getAverageBrightness(cv::Mat src);
	int getLargestContour(cv::Mat srcimg, std::vector<cv::Point> &largest_contour);//srcimg 需经过canny边缘检测
	int getLargestContourAccordKernelSize(cv::Mat srcimg,
		std::vector<cv::Point> &largest_contour, cv::Size ksize);
	double iou_y(cv::Rect r1, cv::Rect r2); //y方向的iou
	bool isStanderPostcode(std::string srcstr, std::string &postcodestr);
	//double postcodeStringScore(std::string srcStr, std::string &resultStr);//ocr得到的字符串是否为邮编打分。
	int matchImage(cv::Mat srcImg,cv::Mat targetImg, std::vector<cv::Rect> rRects);//模板匹配
	double continuous5DigitScore(std::string srcStr);//为得到的字符串是否是邮编做评分，0-1
	double continuousDigitsScore(std::string srcStr, int continus_num);//为字符串是否为连续数字得分
	int maxNumContinuousDigits(std::string srcStr);//获得最大连续数字个数
	double scoreForPostcodeString(std::string srcStr);
	size_t getFirstContinuousDigits(std::string srcStr,int conti_num ,std::string &dstStr);//返回最后的位置
};


//基于opencv343源码修改
class FindHomography_2d
{
public:
	int modelPoints;
	double threshold;
	double confidence;
	int maxIters;
public:
	FindHomography_2d();
	bool haveCollinearPoints(const cv::Mat& m, int count) const;
	bool checkSubset(cv::Mat &ms1, cv::Mat &ms2, int count) const;
	bool getSubset(const cv::Mat& m1, const cv::Mat& m2,
		cv::Mat& ms1, cv::Mat& ms2, cv::RNG& rng,
		int maxAttempts = 1000) const;
	void computeError(cv::InputArray _m1, cv::InputArray _m2, cv::InputArray _model, cv::OutputArray _err) const;
	int runKernel_(cv::InputArray _m1, cv::InputArray _m2, cv::OutputArray _model) const;
	int runKernel_NoZoom(cv::InputArray _m1, cv::InputArray _m2, cv::OutputArray _model, cv::Point2f &whole_scale) const;
	int RANSACUpdateNumIters(double p, double ep, int modelPoints, int maxIters);
	int findInliers(const cv::Mat& m1, const cv::Mat& m2, const cv::Mat& model, cv::Mat& err, cv::Mat& mask, double thresh) const;
	bool run(cv::InputArray _m1, cv::InputArray _m2, cv::OutputArray _model, cv::OutputArray _mask);


};

//识别泰国手写数字类，包括定位和ocr
class HWDigitsOCR
{
public:
	HWDigitsOCR();
	int getPostCode2String(std::string srcImgPath, std::string &postcode, OcrAlgorithm_config* pConfig);
	
	int getPostCode2String_test(cv::Mat srcImgPath, std::string &postcode, OcrAlgorithm_config* pConfig);
	//返回1代表找到了目的地邮编，但是没有寄出地邮编，返回2代表找到了目的地和寄出地邮编，返回0什么也没有
	int getPostCode2String(cv::Mat srcMat, std::string &postcode, OcrAlgorithm_config* pConfig);
private:
	
	int rotateImage(const cv::Mat &srcMat, cv::Mat &dstMat);
	//refMat为收发标志行，dstmat为邮编行
	int getPostcodeLine(const cv::Mat &srcMat, cv::Mat &refMat, cv::Mat &dstMat);

	int segPostCode(const cv::Mat &srcMat, std::vector<cv::Mat> &dstMat_vec);

	//返回值：1-from，2-To
	int getPostcodeType(const cv::Mat &srcMat, tesseract::TessBaseAPI *psTess);

	//输入mat数据
	int getPostCodeFromBoxMats(std::vector<cv::Mat> &srcMat_vec, std::string &result_str, float &confidence, OcrAlgorithm_config* pConfig);

	//返回0 为失败
	int getHandWriteAddressRangeMat(const cv::Mat &parcelMat, OcrAlgorithm_config* pConfig,std::vector<cv::Mat> &dstMat_vec);

	int thresholdImgs(std::vector<cv::Mat> &srcMat_vec, std::vector<cv::Mat> &dstMat_vec);

	//查询match列表中 new_match点的个数
	int query_match_count(std::vector<cv::DMatch> &matches, cv::DMatch & new_match);
	int query_near_count(std::vector<float> &datas, float down_value, float up_value);

	//test 函数使用
	int score_for_rect(std::vector<cv::Rect> rcs, int im_width, int im_height, std::vector<float> &rc_scores);
	//int getHandWriteRange_cluster_nobox()
	int getPostCodeLine_nobox(cv::Mat srcMat, std::vector<cv::Mat> &toMats, std::vector<cv::Mat> &fromMats);
	int getHandWriteRange(cv::Mat &srcMat, cv::Rect &srcRect, cv::Rect &dstRect, bool &need_rotate);
	int split_digits_nobox(cv::Mat &srcMat, std::vector<cv::Mat> &dstDigits);
	int getPostCode_nobox(std::vector<cv::Mat> &srcMat_vec, std::vector<std::string> &result_str, std::vector<float> &confidence, OcrAlgorithm_config* pConfig);

};


