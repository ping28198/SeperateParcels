#pragma once
#include "tesseract/baseapi.h"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "HWDigitsRecogDll.h"

//#define ROTATE_IMG_DEBUG  //���ڴ�ӡ��ǩ��ת����
//#define POSTCODE_ROI_DEBUG //���ڴ�ӡ��ǩ�ʱඨλ����
//#define OCR_DEBUG //����ocrʶ���ʱ����
//#define POSTCODE_BOX_DEBUG //������д�ʱ�ʶ�����

//���ڱ�ǩƥ��
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

//����ocr��������
class OcrAlgorithm_config
{
public:
	std::string tess_data_path;
	std::string template_img_path;
	std::string ORB_template_img1_path;//��ǩͼ��1
	std::string ORB_template_img2_path; //��ǩͼ��2
	std::string detect_model_file_path;
	std::string handwrite_ocr_model_path;
	std::string handwrite_ref_img1_path;//��д��ͼ��1
	MatchDataStruct match_data;
	void *pTess;
	void *pHWDigitsRecog;
	void *pTagDetector;
	void *pLogger;
	double TagDetectConfidence;
	double HandwriteDigitsConfidence;
	int max_instance_per_class;
	int Run_OCR_on_standard_tag; //�Ƿ����׼��ǩ�ʱ�
	int Run_OCR_on_handwrite_box; //�Ƿ�����д�ʱ�
	int Run_OCR_on_unknown_tag; //�Ƿ��������ǩ
	int is_test_model; //�Ƿ�Ϊ����ģʽ����Ϊ��������ʹ��
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
	//�ӱ��ؼ��ش洢������������
	int loadMatchData(const std::string &xmlfile,cv::Mat &descriptors2, cv::Mat &descriptors3, 
		std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::KeyPoint> &keypoints3);

	int saveMatchData(const std::string &xmlfile, cv::Mat &descriptors2, cv::Mat &descriptors3,
		std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::KeyPoint> &keypoints3);
public:
	static int getParcelBoxFromSick(std::string xml_path, cv::RotatedRect &parcel_box);
	static void rotate_arbitrarily_angle(cv::Mat &src, cv::Mat &dst, float angle);
	double postcodeStringScore(std::string srcStr, std::string &resultStr);//ceshi
	static int sumPixels(cv::Mat &srcimg, int axis, std::vector<unsigned int> &resultsVec); //axis =0 x�ᣬ=1 y��
	static int adJustBrightness(cv::Mat& src, double alpha, double beta, double anchor = 0);//alpha�Աȶȵ��� <1 ���ͶԱȶȣ�beta���ȵ�����<0�䰵, anchor ê��
protected:

private:
	int _getOcrRoi(cv::Mat& src_img, std::vector<cv::Mat> &roi_imgs);
	int _runOcr(cv::Mat& RoiMat, char* pResults, size_t bufferlenth);
	int _runOcrPreload(cv::Mat& RoiMat,char* pResults, size_t bufferlenth, tesseract::TessBaseAPI* pTess);
	int getBarcodePos(cv::Mat &src_img, cv::Rect &bRect);
	int rotateImg(cv::Mat &src_img, cv::Mat &dst_img);
	int rotateImg_v1(cv::Mat &src_img, cv::Mat &dst_img);
	int rotateImg_v2(cv::Mat &src_img, cv::Mat &dst_img, double mthreshold);//mthreshold Ĭ��Ϊ2.5
	double rotateImg_ORB(cv::Mat src_img, cv::Mat referenceMat, cv::Mat &dst_img); //����ƽ������
	
	int getPostcodeRoi(cv::Mat &srcImg, std::vector<cv::Rect> &dstRects, cv::Rect mrect, std::string tmplatePath);
	int getPostcodeRoi_TPmatch(cv::Mat &srcImg,  cv::Rect barcoderect, 
		std::string templateImgPath, std::vector<cv::Rect> &roiRects);
	int getPostcodeRoiInRectImg(cv::Mat srcImg,cv::Rect sRect, cv::Rect &roiRect);
	int getPostcodeRoiInRectImg_accordPos(cv::Mat srcImg, cv::Rect sRect,int anchor_row, cv::Rect &roiRect);//anchor_row���������ʱ�������
	int getPostcodeRoiInRectImg_SiftMatch(cv::Mat srcImg, cv::Rect sRect, cv::Rect &roiRect);
	
	
	int rotatePoints(std::vector<cv::Point2f> & points_vec, double angle); //�ƶ���ԭ��
	int rotatePoints(std::vector<cv::Point> & points_vec, double angle);
	double getOptimumRotateAngle(std::vector<cv::Point2f> & points_vec); //������ת��ֱ���ﵽ���ţ�points_vecҲ����ת
	double ratio_w_h(std::vector<cv::Point2f> & points_vec);
	double getAveragePixelInRect(cv::Mat& src, cv::Rect &mRect);
	int getContourRect(std::vector<cv::Point2f> & points_vec, cv::Rect &mRect);
	
	double getAverageBrightness(cv::Mat src);
	int getLargestContour(cv::Mat srcimg, std::vector<cv::Point> &largest_contour);//srcimg �辭��canny��Ե���
	int getLargestContourAccordKernelSize(cv::Mat srcimg,
		std::vector<cv::Point> &largest_contour, cv::Size ksize);
	double iou_y(cv::Rect r1, cv::Rect r2); //y�����iou
	bool isStanderPostcode(std::string srcstr, std::string &postcodestr);
	//double postcodeStringScore(std::string srcStr, std::string &resultStr);//ocr�õ����ַ����Ƿ�Ϊ�ʱ��֡�
	int matchImage(cv::Mat srcImg,cv::Mat targetImg, std::vector<cv::Rect> rRects);//ģ��ƥ��
	double continuous5DigitScore(std::string srcStr);//Ϊ�õ����ַ����Ƿ����ʱ������֣�0-1
	double continuousDigitsScore(std::string srcStr, int continus_num);//Ϊ�ַ����Ƿ�Ϊ�������ֵ÷�
	int maxNumContinuousDigits(std::string srcStr);//�������������ָ���
	double scoreForPostcodeString(std::string srcStr);
	size_t getFirstContinuousDigits(std::string srcStr,int conti_num ,std::string &dstStr);//��������λ��
};


//����opencv343Դ���޸�
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

//ʶ��̩����д�����࣬������λ��ocr
class HWDigitsOCR
{
public:
	HWDigitsOCR();
	int getPostCode2String(std::string srcImgPath, std::string &postcode, OcrAlgorithm_config* pConfig);
	
	int getPostCode2String_test(cv::Mat srcImgPath, std::string &postcode, OcrAlgorithm_config* pConfig);
	//����1�����ҵ���Ŀ�ĵ��ʱ࣬����û�мĳ����ʱ࣬����2�����ҵ���Ŀ�ĵغͼĳ����ʱ࣬����0ʲôҲû��
	int getPostCode2String(cv::Mat srcMat, std::string &postcode, OcrAlgorithm_config* pConfig);
private:
	
	int rotateImage(const cv::Mat &srcMat, cv::Mat &dstMat);
	//refMatΪ�շ���־�У�dstmatΪ�ʱ���
	int getPostcodeLine(const cv::Mat &srcMat, cv::Mat &refMat, cv::Mat &dstMat);

	int segPostCode(const cv::Mat &srcMat, std::vector<cv::Mat> &dstMat_vec);

	//����ֵ��1-from��2-To
	int getPostcodeType(const cv::Mat &srcMat, tesseract::TessBaseAPI *psTess);

	//����mat����
	int getPostCodeFromBoxMats(std::vector<cv::Mat> &srcMat_vec, std::string &result_str, float &confidence, OcrAlgorithm_config* pConfig);

	//����0 Ϊʧ��
	int getHandWriteAddressRangeMat(const cv::Mat &parcelMat, OcrAlgorithm_config* pConfig,std::vector<cv::Mat> &dstMat_vec);

	int thresholdImgs(std::vector<cv::Mat> &srcMat_vec, std::vector<cv::Mat> &dstMat_vec);

	//��ѯmatch�б��� new_match��ĸ���
	int query_match_count(std::vector<cv::DMatch> &matches, cv::DMatch & new_match);
	int query_near_count(std::vector<float> &datas, float down_value, float up_value);

	//test ����ʹ��
	int score_for_rect(std::vector<cv::Rect> rcs, int im_width, int im_height, std::vector<float> &rc_scores);
	//int getHandWriteRange_cluster_nobox()
	int getPostCodeLine_nobox(cv::Mat srcMat, std::vector<cv::Mat> &toMats, std::vector<cv::Mat> &fromMats);
	int getHandWriteRange(cv::Mat &srcMat, cv::Rect &srcRect, cv::Rect &dstRect, bool &need_rotate);
	int split_digits_nobox(cv::Mat &srcMat, std::vector<cv::Mat> &dstDigits);
	int getPostCode_nobox(std::vector<cv::Mat> &srcMat_vec, std::vector<std::string> &result_str, std::vector<float> &confidence, OcrAlgorithm_config* pConfig);

};


