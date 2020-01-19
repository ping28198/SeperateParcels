#include "OcrAlgorithm.h"
#include <algorithm>

#include "opencv/cv.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/highgui.h"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "tinyxml2.h"
#include "tesseract/baseapi.h"
#include "CommonFunc.h" 
#include "CutParcelBox.h"
#include "time.h"
#include "leptonica/allheaders.h"
#include <iostream>
#include <memory>
#include "ImageProcessFunc.h"
#include "logger.h"
using namespace cv;
using namespace std;

int query_match_count_(std::vector<DMatch> &matches, DMatch & new_match)
{
	int match_count = 0;
	std::vector<DMatch>::iterator it;
	for (it = matches.begin(); it != matches.end(); it++)
	{
		if (new_match.trainIdx == it->trainIdx) match_count++;
	}
	return match_count;
}



OcrAlgorithm::OcrAlgorithm()
{

}

int OcrAlgorithm::getOcrRoi(Mat& src_img, std::vector<cv::Mat> &roi_imgs)
{

	int res = _getOcrRoi(src_img, roi_imgs);

	return res;
}

int OcrAlgorithm::runOcr(Mat& RoiMat, char* pResults, size_t bufferlenth, tesseract::TessBaseAPI* pTess)
{
	int h = RoiMat.rows;
	int sh = 30;
	float scal_ = sh / float(h);
	cv::Mat resizedMat = RoiMat;
	if (scal_ > 1.1)
	{
		cv::resize(RoiMat, resizedMat, cv::Size(), scal_, scal_,cv::INTER_AREA);
	}

	cv::Rect mR;
	mR.x = 0;
	mR.y = 0;
	mR.height = resizedMat.rows;
	mR.width = resizedMat.cols / 4;

#ifdef OCR_DEBUG
	imshow("_runOcr调整亮度前", resizedMat);
#endif // OCR_DEBUG
	double vPix = getAveragePixelInRect(resizedMat, mR);
	double anchor = 170;
	double alpha = 3.0;
	double beta = 200 - vPix;
	adJustBrightness(resizedMat, alpha, beta, anchor);
#ifdef OCR_DEBUG
	imshow("_runOcr调整亮度后", resizedMat);
#endif // OCR_DEBUG


	int res = 0;
	try
	{
		if (pTess==NULL) 
		{
			res = this->_runOcr(resizedMat, pResults, bufferlenth);
		}
		else
		{
			res = _runOcrPreload(resizedMat, pResults, bufferlenth, pTess);
		}
		
	}
	catch (...)
	{
#ifdef _DEBUG
		printf("运行OCR出错\n");
#endif // _DEBUG
		return 0;
	}

	return res;
}

int OcrAlgorithm::getOcrResultString(Mat src_img,tesseract::TessBaseAPI *pTess, string &resultstring, OcrAlgorithm_config *pConfig)
{
	if (pConfig==NULL)
	{
		return 0;
	}
	//tesseract::TessBaseAPI* pTess = psTess;
	std::vector<cv::Mat> roiMats;
	string mstring;
	char pOcrResutls[128] = { 0 };
	int res = _getOcrRoi(src_img, roiMats);
	if (res ==0 || roiMats.size()==0)
	{
		printf("未获得邮编区域\n");
	}
	else
	{
		for (int i = 0; i < roiMats.size(); i++)
		{
			res = runOcr(roiMats[i], pOcrResutls, 128, pTess);
			if (res == 0) continue;
			mstring = pOcrResutls;
			std::cout << "OCR:" << mstring << std::endl;
			if (isStanderPostcode(mstring, resultstring) == true)
			{

				return 1;
			}
		}
	}

	if (pConfig->pLogger != NULL)
	{
		((Logger*)(pConfig->pLogger))->TraceInfo("Apply sift match!");
	}

	printf("采用SIFT匹配矫正\n");
	cv::Mat rotatedImg1, rotatedImg2;
	//std::string refImgPath1 = pConfig->ORB_template_img1_path;
	//std::string refImgPath2 = pConfig->ORB_template_img2_path;
	//cv::Mat refImg1 = imread(refImgPath1);
	//cv::Mat refImg2 = imread(refImgPath2);
	//rotateImg_ORB(src_img, refImg1, rotatedImg1);
	//rotateImg_ORB(src_img, refImg2, rotatedImg2);
	//imshow("ORB rotatedimg1", rotatedImg1);
	//imshow("ORB rotatedimg2", rotatedImg2);
	//
	rotateImg_SIFT(src_img, rotatedImg1, rotatedImg2, pConfig);


	std::string roistring1, roistring2;
	std::string mstring1, mstring2;
	cv::Rect mr;
	cv::Rect mroi;
	bool isStandard1 = false;
	if (!rotatedImg1.empty())
	{
		mr.x = rotatedImg1.cols / 2 - 1;
		mr.y = 0;
		mr.width = rotatedImg1.cols / 2;
		mr.height = rotatedImg1.rows / 2;
		res = getPostcodeRoiInRectImg(rotatedImg1, mr, mroi);
		if (res!=0)
		{
			if (float(mroi.height) / mroi.width < 3.0)
			{
				cv::Mat m = rotatedImg1(mroi);
				res = runOcr(m, pOcrResutls, 128, pTess);
				if (res != 0)
				{
					mstring1 = pOcrResutls;
					std::cout << "OCR:" << mstring1;
					isStandard1 = isStanderPostcode(mstring1, roistring1);
				}
			}
		}
		if (!isStandard1)
		{
			res = getPostcodeRoiInRectImg_accordPos(rotatedImg1, mr,mr.height*0.661,mroi);
			if (res!=0)
			{
				if (float(mroi.height) / mroi.width < 3.0)
				{
					cv::Mat m = rotatedImg1(mroi);
					res = runOcr(m, pOcrResutls, 128, pTess);
					if (res != 0)
					{
						mstring1 = pOcrResutls;
						std::cout << "OCR:" << mstring1;
						isStandard1 = isStanderPostcode(mstring1, roistring1);
					}
				}
			}
		}
	}
	bool isStandard2 = false;
	if (!rotatedImg2.empty())
	{
		mr.x = rotatedImg2.cols / 2 - 1;
		mr.y = 0;
		mr.width = rotatedImg2.cols / 2;
		mr.height = rotatedImg2.rows / 2;
		res = getPostcodeRoiInRectImg(rotatedImg2, mr, mroi);
		
		if (res != 0)
		{
			if (float(mroi.height) / mroi.width < 3.0)
			{
				cv::Mat m = rotatedImg2(mroi);
				res = runOcr(m, pOcrResutls, 128, pTess);
				if (res != 0)
				{
					mstring2 = pOcrResutls;
					std::cout << "OCR:" << mstring2;
					isStandard2 = isStanderPostcode(mstring2, roistring2);
				}
			}
		}
		if (!isStandard2)
		{
			res = getPostcodeRoiInRectImg_accordPos(rotatedImg2, mr, mr.height*0.496, mroi);
			if (res != 0)
			{
				if (float(mroi.height) / mroi.width < 3.0)
				{
					cv::Mat m = rotatedImg2(mroi);
					res = runOcr(m, pOcrResutls, 128, pTess);
					if (res != 0)
					{
						mstring2 = pOcrResutls;
						std::cout << "OCR:" << mstring2;
						isStandard2 = isStanderPostcode(mstring2, roistring2);
					}
				}
			}
		}
	}

	if (isStandard1 && isStandard2)
	{
		double sc1 = postcodeStringScore(mstring1, roistring1);
		double sc2 = postcodeStringScore(mstring2, roistring2);
		resultstring = (sc1 < sc2) ? roistring2 : roistring1;
		return 1;
	}
	else if(isStandard1)
	{
		resultstring = roistring1;
		return 1;
	}
	else if(isStandard2)
	{
		resultstring = roistring2;
		return 1;
	}


	return 0;
}

int OcrAlgorithm::getParcelBoxFromSick(std::string xml_path, cv::RotatedRect &parcel_box)
{
	tinyxml2::XMLDocument doc;
	doc.LoadFile(xml_path.c_str());
	//std::cout << doc.ErrorID() << std::endl;
	//std::cout << doc.ErrorIDToName(doc.ErrorID()) << std::endl;
	//char boxpos_str[32] = { 0 };
	std::string box_pos;
	tinyxml2::XMLElement *camera_root = doc.FirstChildElement("camera");
	if (camera_root == NULL) return 0;
	tinyxml2::XMLElement *image_node = camera_root->FirstChildElement("image");
	if (image_node == NULL) return 0;
	tinyxml2::XMLElement *parcel_node = image_node->FirstChildElement("parcel");
	if (parcel_node!=NULL)
	{
		const tinyxml2::XMLAttribute *attr = parcel_node->FindAttribute("pos");
		if (attr!=NULL)
		{
			box_pos = attr->Value();
		}
	}
	if(box_pos.empty())
	{
		tinyxml2::XMLElement * box_ele = image_node->FirstChildElement("box");
		while (box_ele!=NULL)
		{
			const tinyxml2::XMLAttribute *attr_name = box_ele->FindAttribute("name");
			if (attr_name != NULL)
			{
				string name_str = attr_name->Value();
				if (name_str == "Parcel" || name_str == "parcel")
				{
					const tinyxml2::XMLAttribute *attr_pos = box_ele->FindAttribute("pos");
					if (attr_pos != NULL)
					{
						box_pos = attr_pos->Value();
					}
					break;
				}
			}

			box_ele = box_ele->NextSiblingElement();//实例中有这种操作，不会造成内存泄漏
		}
	}
	if (box_pos.empty()) return 0;
	tinyxml2::XMLElement *image_info = image_node->FirstChildElement("imageinfo");
	tinyxml2::XMLElement *size_node = NULL;
	if (image_info!=NULL)
	{
		size_node = image_info->FirstChildElement("size");
	}
	int img_width = 0;
	int img_height = 0;
	if (size_node != NULL)
	{
		const tinyxml2::XMLAttribute *attr = size_node->FindAttribute("width");
		if (attr != NULL)
		{
			img_height = attr->IntValue();
		}
		attr = size_node->FindAttribute("length");
		if (attr != NULL)
		{
			img_width = attr->IntValue();
		}
	}

	if (img_height <= 0 || img_width <= 0) return 0;
	
	int pos_int[8] = { 0 };
	int ind = 0;
	int pre_i = 0;
	for (int i = 0; i < box_pos.length(); i++)
	{
		if (ind >=8) break;
		char c = box_pos[i];
		if (!isdigit(c) && c != ' ') return 0;//判断非数字字符
		if (c==' ')
		{
			std::string substr = box_pos.substr(pre_i, i - pre_i);
			if (!substr.empty())
			{
				pos_int[ind++] = atoi(substr.c_str());
			}
			pre_i = i+1;
		}
		if (i==box_pos.length()-1 && c!=' ')
		{
			std::string substr = box_pos.substr(pre_i, i - pre_i+1);
			if (!substr.empty())
			{
				pos_int[ind++] = atoi(substr.c_str());
			}
			pre_i = i + 1;
		}
	}
	//if (pos_int[1] >= pos_int[5] || pos_int[2] >= pos_int[6]) return 0; //剔除
	//选取第一个点（上（右）顶点）

	//for (int i=1;i<4;i++)
	//{
	//	if (pos_int[1]>pos_int[i*2+1])
	//	{
	//		int tmp = pos_int[1];
	//		pos_int[1] = pos_int[i * 2 + 1]; pos_int[i * 2 + 1] = tmp;
	//		tmp = pos_int[0];
	//		pos_int[0] = pos_int[i * 2]; 
	//	}
	//	else if(pos_int[1] == pos_int[i * 2 + 1] && pos_int[0] < pos_int[i * 2])
	//	{
	//		int tmp = pos_int[1];
	//		pos_int[1] = pos_int[i * 2 + 1]; pos_int[i * 2 + 1] = tmp;
	//		tmp = pos_int[0];
	//		pos_int[0] = pos_int[i * 2]; pos_int[i * 2] = tmp;
	//	}
	//}
	////选取第二个点（左（上）顶点）
	//for (int i = 2; i < 4; i++)
	//{
	//	if (pos_int[2] > pos_int[i * 2])
	//	{
	//		int tmp = pos_int[2];
	//		pos_int[2] = pos_int[i * 2]; pos_int[i * 2] = tmp;
	//		tmp = pos_int[3];
	//		pos_int[3] = pos_int[i * 2 + 1]; pos_int[i * 2 + 1] = tmp;
	//	}
	//	else if (pos_int[2] == pos_int[i * 2] && pos_int[3] > pos_int[i * 2+1])
	//	{
	//		int tmp = pos_int[2];
	//		pos_int[2] = pos_int[i * 2]; pos_int[i * 2] = tmp;
	//		tmp = pos_int[3];
	//		pos_int[3] = pos_int[i * 2 + 1]; pos_int[i * 2 + 1] = tmp;
	//	}
	//}
	////选取低三个点（下（左）顶点）
	//for (int i = 3; i < 4; i++)
	//{
	//	if (pos_int[5] < pos_int[i * 2+1])
	//	{
	//		int tmp = pos_int[4];
	//		pos_int[4] = pos_int[i * 2]; pos_int[i * 2] = tmp;
	//		tmp = pos_int[5];
	//		pos_int[5] = pos_int[i * 2 + 1]; pos_int[i * 2 + 1] = tmp;
	//	}
	//	else if (pos_int[5] == pos_int[i * 2+1] && pos_int[4] > pos_int[i * 2])
	//	{
	//		int tmp = pos_int[4];
	//		pos_int[4] = pos_int[i * 2]; pos_int[i * 2] = tmp;
	//		tmp = pos_int[5];
	//		pos_int[5] = pos_int[i * 2 + 1]; pos_int[i * 2 + 1] = tmp;
	//	}
	//}


	cv::RotatedRect RtRect;
	//角度
	if (pos_int[0] == pos_int[2])
	{
		RtRect.angle = 90;
	}
	else
	{
		float ang1 = atan(float(pos_int[1] - pos_int[3]) / (pos_int[0] - pos_int[2])) / CV_PI * 180;
		float ang2 = atan(float(pos_int[7] - pos_int[5]) / (pos_int[6] - pos_int[4])) / CV_PI * 180;
		RtRect.angle = (ang2+ang1)/2;
	}

	//宽高
	float wd1, wd2, ht1, ht2;
	wd1 = sqrt(pow(pos_int[6] - pos_int[0], 2)+ pow(pos_int[7] - pos_int[1], 2));
	wd2 = sqrt(pow(pos_int[4] - pos_int[2], 2) + pow(pos_int[5] - pos_int[3], 2));
	ht1 = sqrt(pow(pos_int[3] - pos_int[1], 2) + pow(pos_int[0] - pos_int[2], 2));
	ht2 = sqrt(pow(pos_int[6] - pos_int[4], 2) + pow(pos_int[5] - pos_int[7], 2));
	RtRect.size = cv::Size2f(max(ht1,ht2), max(wd1, wd2));

	//中心
	RtRect.center = cv::Point2f((pos_int[0] + pos_int[2] + pos_int[4] + pos_int[6]) / 4.0f,
		(pos_int[1] + pos_int[3] + pos_int[5] + pos_int[7]) / 4.0f);

	parcel_box = RtRect;

	//测试
	//cv::Mat image = cv::Mat::zeros(img_width, img_height, CV_8UC3);
	//Point2f vertices[4];
	//RtRect.points(vertices);
	//for (int i = 0; i < 4; i++)
	//{
	//	cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 10);
	//	cv::Point2f pt1(pos_int[2 * i + 0], pos_int[2 * i + 1]);
	//	cv::Point2f pt2(pos_int[(2 * i + 2) % 8 + 0], pos_int[(2 * i + 2) % 8 + 1]);
	//	cv::line(image, pt1, pt2, cv::Scalar(0, 0, 255), 10);
	//}
	//float scal = 1000.0 / max(img_width, img_height);
	//cv::resize(image, image, cv::Size(), scal, scal);
	//cv::imshow("框框", image);

	//cv::waitKey(0);


	



	return 1;
}

int OcrAlgorithm::_getOcrRoi(Mat& src_img, std::vector<cv::Mat> &roi_imgs)
{
	Mat rotated_img;
	cvtColor(src_img, src_img, COLOR_BGR2GRAY);
	int res = rotateImg(src_img, rotated_img);
	if (res == 0)
	{
#ifdef ROTATE_IMG_DEBUG
		printf("旋转图像失败\n");
		imshow("旋转前图像", src_img);
#endif // _DEBUG
		return 0;
	}

#ifdef ROTATE_IMG_DEBUG
	imshow("旋转的图像", rotated_img);
#endif // _DEBUG
	Rect mRect;
	res = getBarcodePos(rotated_img, mRect);
	if (res == 0)
	{
#ifdef ROTATE_IMG_DEBUG
		printf("定位条形码失败\n");
		imshow("旋转的图像", rotated_img);
#endif // _DEBUG
		return 0;
	}
	Mat roimat;
	std::vector<cv::Rect> rois;
	std::string tmpl = "./templateimg.jpg";
	res = getPostcodeRoi(rotated_img,rois, mRect, tmpl);
	if (res == 0)
	{
#ifdef _DEBUG
		printf("定位邮编位置失败\n");
		imshow("旋转的图像", rotated_img);
#endif // _DEBUG
		return 0;
	}
	for (int i=0;i<rois.size();i++)
	{
		roi_imgs.push_back(rotated_img(rois[i]));
	}
	return 1;
}

int OcrAlgorithm::_runOcr(Mat& RoiMat, char* pResults, size_t bufferlenth)
{

	tesseract::TessBaseAPI tess;
	if (tess.Init("./tessdata", "eng"))
	{
#ifdef _DEBUG
		std::cout << "OCRTesseract: Could not initialize tesseract." << std::endl;
#endif // _DEBUG
		return 0;
	}
	_runOcrPreload(RoiMat, pResults, bufferlenth, &tess);


	tess.Clear();
	return 1;
}

int OcrAlgorithm::_runOcrPreload(Mat& RoiMat, char* pResults, size_t bufferlenth, tesseract::TessBaseAPI* pTess)
{
	if (RoiMat.empty()) return 0;



	int w = RoiMat.cols;
	int h = RoiMat.rows;
	unsigned char *pImgData = RoiMat.data;

	pTess->SetPageSegMode(tesseract::PageSegMode::PSM_SINGLE_LINE);
	//pTess->SetVariable("save_best_choices", "T");
	pTess->SetVariable("classify_bln_numeric_mode", "1");
	pTess->SetImage(RoiMat.data, w, h, RoiMat.channels(), RoiMat.step1());
	pTess->Recognize(0);

	// get result and delete[] returned char* string
#ifdef OCR_DEBUG
	std::cout << std::unique_ptr<char[]>(pTess->GetUTF8Text()).get() << std::endl;
#endif // OCR_DEBUG
	//
	strcpy_s(pResults, bufferlenth, pTess->GetUTF8Text());

	return 1;

}

int OcrAlgorithm::getBarcodePos(Mat &src_img, Rect &bRect)
{
	Mat g_img;
	GaussianBlur(src_img, g_img, Size(3, 3), 0);

	//边缘检测
	Mat sobel_x, sobel_y;
	Sobel(g_img, sobel_x, CV_8U, 1, 0);
	Sobel(g_img, sobel_y, CV_8U, 0, 1);
	g_img = sobel_x - sobel_y;
	//Canny(src_img, g_img, 30, 255);
#ifdef POSTCODE_ROI_DEBUG
	imshow("barcode原始图像", src_img);
	imshow("sobel边缘检测", g_img);
#endif // POSTCODE_ROI_DEBUG

	
	//
	threshold(g_img, g_img, 20, 255, CV_THRESH_BINARY);
	//imshow("边缘检测", g_img);
#ifdef POSTCODE_ROI_DEBUG
	imshow("sobel二值化", g_img);
#endif // POSTCODE_ROI_DEBUG
	//确定条形码竖直方向
	int h = g_img.rows;
	int w = g_img.cols;
	vector<unsigned int> PixelsAdd;
	for (int i = 0; i < h; i++)
	{
		unsigned int tmp = 0;
		for (int j = 0; j < w; j++)
		{
			tmp += g_img.at<uchar>(i, j);
		}
		PixelsAdd.push_back(tmp);
	}
	unsigned int max_pixes = 0;
	for (int i = 0; i < PixelsAdd.size(); i++)
	{
		if (max_pixes < PixelsAdd[i])
		{
			max_pixes = PixelsAdd[i];
		}
	}
	if (max_pixes < w * 10)
	{
		return 0;
	}
	vector<Point> continus_flag;
	unsigned int threshold_pix = 0.6*max_pixes;
	Point pt; //标记连续点
	bool cont_flat = false;
	for (int i = 0; i < PixelsAdd.size(); i++)
	{
		if (cont_flat == false && PixelsAdd[i] > threshold_pix)
		{
			pt.x = i;
			cont_flat = true;
		}
		if (cont_flat == true && PixelsAdd[i] < threshold_pix)
		{
			pt.y = i;
			continus_flag.push_back(pt);
			cont_flat = false;
		}
		if (cont_flat == true && i == PixelsAdd.size() - 1)
		{
			pt.y = i;
			continus_flag.push_back(pt);
			cont_flat = false;
		}
	}
	if (continus_flag.size() == 0)
	{
		return 0;
	}
	int max_l = 0;
	int max_w = 0;
	for (int i = 0; i < continus_flag.size(); i++)
	{
		if (max_w < continus_flag[i].y - continus_flag[i].x)
		{
			max_w = continus_flag[i].y - continus_flag[i].x;
			max_l = i;
		}
	}
	Rect rect_y = Rect(0, continus_flag[max_l].x, g_img.cols, continus_flag[max_l].y - continus_flag[max_l].x);
	Mat cut_img = g_img(rect_y);
	if (cut_img.empty())
	{
		return 0;
	}

	//确定条形码水平方向
	h = cut_img.rows;
	w = cut_img.cols;
	vector<unsigned int>().swap(PixelsAdd);
	for (int i = 0; i < w; i++)
	{
		unsigned int tmp = 0;
		for (int j = 0; j < h; j++)
		{
			tmp += cut_img.at<uchar>(j, i);
		}
		PixelsAdd.push_back(tmp);
	}
	max_pixes = 0;
	for (int i = 0; i < PixelsAdd.size(); i++)
	{
		if (max_pixes < PixelsAdd[i])
		{
			max_pixes = PixelsAdd[i];
		}
	}
	if (max_pixes < h * 10)
	{
		return 0;
	}
	threshold_pix = 0.5*max_pixes;
	int contin_len = h * 0.4;
	int dis_contin_count = 0;
	int last_i = 0;
	cont_flat = false;
	vector<Point>().swap(continus_flag);
	for (int i = 0; i < PixelsAdd.size(); i++)
	{
		if (threshold_pix < PixelsAdd[i])
		{
			dis_contin_count = 0;
			if (cont_flat == false)
			{
				pt.x = i;
				cont_flat = true;
			}
			last_i = i;
		}
		else
		{
			if (cont_flat == true)
			{
				dis_contin_count++;
				if (dis_contin_count > contin_len)
				{
					cont_flat = false;
					pt.y = last_i;
					continus_flag.push_back(pt);
				}
			}

		}
		if (i == PixelsAdd.size() - 1 && cont_flat == true)
		{
			cont_flat = false;
			pt.y = last_i;
			continus_flag.push_back(pt);
		}
	}
	if (continus_flag.size() == 0)
	{
		return 0;
	}
	max_l = 0;
	max_w = 0;
	for (int i = 0; i < continus_flag.size(); i++)
	{
		if (max_w < continus_flag[i].y - continus_flag[i].x)
		{
			max_w = continus_flag[i].y - continus_flag[i].x;
			max_l = i;
		}
	}
	Rect rect_x = Rect(continus_flag[max_l].x, 0, continus_flag[max_l].y - continus_flag[max_l].x, cut_img.rows);
	bRect.x = rect_x.x;
	bRect.y = rect_y.y;
	bRect.width = rect_x.width;
	bRect.height = rect_y.height;

	//if (cut_img.empty())
	//{
	//	return 0;
	//}

	//imshow("sobel_x", g_img);
	//imshow("cutbox", cut_img);
	//waitKey(0);
	return 1;
}

int OcrAlgorithm::rotateImg_v1(Mat &src_img, Mat &dst_img)
{
	Mat binaryImg, edgeImg;//彩色图像转化成灰度图  
	if (src_img.empty())
	{
		printf("没有文件\n");
		return 0;
	}
	

	//imshow("滤波处理", src_gray);
	//medianBlur(src_gray, src_gray,3);
	Canny(src_img, edgeImg, 20, 250);
	//imshow("边缘检测", edgeImg);

	// 检测直线
	vector<Vec4i>lines;//定义一个存放直线信息的向量						//Hough直线检测API	
	vector<Vec4i>old_lines;
	int threshold_points = 100;
	while (true)
	{
		vector<Vec4i>().swap(lines);
		HoughLinesP(edgeImg, lines, 1, CV_PI / 180, threshold_points, 100, 20);
		//线段最短长度，相邻点最大距离
		if (lines.size() >= 4)
		{
			old_lines = lines;
			threshold_points = threshold_points + 5;
			continue;
		}
		if (lines.size() > 0 && lines.size() < 4)
		{
			old_lines = lines;
			break;
		}
		else
		{
			break;
		}
	}
	//InputArray src, // 输入图像，必须8-bit的灰度图像；OutputArray lines, // 输出的极坐标来表示直线；double rho, // 生成极坐标时候的像素扫描步长；double theta, //生成极坐标时候的角度步长，一般取值CV_PI/180；int threshold, // 阈值，只有获得足够交点的极坐标点才被看成是直线；
	//printf("直线个数：%d\n", old_lines.size());
	if (old_lines.size() == 0)
	{
		return 0;
	}
#ifdef ROTATE_IMG_DEBUG
	for (size_t i = 0; i < old_lines.size(); i++)
	{
		Vec4i l = old_lines[i];
		line(src_img, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(i * 50, i * 50, i * 50), 2, CV_AA);
		imshow("霍夫变换结果", src_img);
		imshow("边缘检测结果", edgeImg);
	}
#endif // ROTATE_IMG_DEBUG



	// 查找最长的线段
	double line_length = 0;
	int line_ind = 0;
	for (size_t i = 0; i < old_lines.size(); i++)
	{
		Vec4i l = old_lines[i];
		double tmp_length = (l[0] - l[2])*(l[0] - l[2]) + (l[1] - l[3])*(l[1] - l[3]);
		if (line_length < tmp_length)
		{
			line_length = tmp_length;
			line_ind = i;
		}
	}

	//计算角度并旋转图像
	float k = float(old_lines[line_ind][3] - old_lines[line_ind][1]) / float(old_lines[line_ind][2] - old_lines[line_ind][0]);
	float angle_radian = atan(k);
	//Mat rotatedImg;
	ImageProcessFunc::rotate_arbitrarily_angle(src_img, dst_img, angle_radian);
	//imshow("旋转之后", rotatedImg);
	return 1;
}

int OcrAlgorithm::rotateImg(Mat &src_img, Mat &dst_img)
{
	//
	//亮度调整
	cv::Rect mR;
	mR.width = src_img.cols / 4;
	mR.height = src_img.rows / 4;
	mR.x = mR.width;
	mR.y = mR.height;
	double brt = 0;
	brt += getAverageBrightness(src_img(mR));//图像中间采样
	mR.x = mR.width * 2;
	brt += getAverageBrightness(src_img(mR));//图像中间采样
	mR.y = mR.height * 2;
	brt += getAverageBrightness(src_img(mR));//图像中间采样
	mR.x = mR.width;
	brt += getAverageBrightness(src_img(mR));//图像中间采样
	brt = brt / 4;
	double standardBrt = 120;
	double alpha = 1.5;
	double beta = standardBrt - brt;
	adJustBrightness(src_img, alpha, beta);

#ifdef ROTATE_IMG_DEBUG
	if (beta>0)
	{
		printf("提升亮度%f\n", beta);
	}
	else
	{
		printf("降低亮度%f\n", beta);
	}
	imshow("亮度调整", src_img);
#endif // ROTATE_IMG_DEBUG


	//采用形态学运算
	int res = 0;
	res = rotateImg_v2(src_img, dst_img, 2.5);
	if (res > 0) return 1;
#ifdef ROTATE_IMG_DEBUG
	printf("形态学矫正失败，采用霍夫变化!\n");
#endif // ROTATE_IMG_DEBUG
	//采用霍夫变化
	res = rotateImg_v1(src_img, dst_img);
	if (res > 0) return 1;
#ifdef ROTATE_IMG_DEBUG
	printf("霍夫变换矫正失败，采用低阈值形态学矫正!\n");
#endif // ROTATE_IMG_DEBUG

	res = rotateImg_v2(src_img, dst_img, 0);
	if (res > 0) return 1;

	return 0;
}

int OcrAlgorithm::rotateImg_v2(Mat &src_img, Mat &dst_img, double mthreshold)
{
	Mat cannymat;
	Mat resized_mat;
	int s_h = src_img.rows;
	int s_w = src_img.cols;
	int s_size = (s_w > s_h) ? s_h : s_w;
	//int s_a = s_w * s_h;
	int dst_size = 360;
	double _rate = dst_size / double(s_size);
	int d_w = s_w * _rate;
	int d_h = s_h * _rate;
	resize(src_img, resized_mat, Size(d_w, d_h));
	d_w = resized_mat.cols;
	d_h = resized_mat.rows;
	//Mat kernel = (Mat_<int>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);

	//imshow("resized", resized_mat);

	//filter2D(resized_mat, resized_mat, -1, kernel);
	//
	//imshow("shaped", resized_mat);
	//Mat sobelmat;
	//Sobel(resized_mat, sobelmat, CV_8U, 0, 1);
	//imshow("sobel_y", sobelmat);

	//Sobel(resized_mat, sobelmat, CV_8U, 1, 0);
	//imshow("sobel_x", sobelmat);

	//Mat grdmat = Mat(cannymat.rows,cannymat.cols,CV_32FC1);
	Canny(resized_mat, cannymat, 40, 255);
#ifdef ROTATE_IMG_DEBUG
	imshow("canny", cannymat);
#endif // ROTATE_IMG_DEBUG

	//获取最大轮廓
	std::vector<cv::Point> contour;
	int res = getLargestContour(cannymat, contour);
	if (res==0)
	{
		return 0;
	}

	vector<Vec4i> hierarchy_x;
	//绘制轮廓
	vector<vector<Point>> mcontours;
	mcontours.push_back(contour);

#ifdef ROTATE_IMG_DEBUG
	Mat newmat = Mat(cannymat.rows, cannymat.cols, CV_8UC1, Scalar(0, 0, 0));
	drawContours(newmat, mcontours, 0, Scalar(255, 255, 255));
	imshow("最大轮廓", newmat);
#endif // ROTATE_IMG_DEBUG
	

	//寻找中心
	Point cent_p = Point(0, 0);
	for (int i = 0; i < contour.size(); i++)
	{
		cent_p += contour[i];
	}
	cent_p.x = cent_p.x / int(contour.size());
	cent_p.y = cent_p.y / int(contour.size());

	//找到距离中心最远点
	int max_dist = 0;
	int max_dist_ind = 0;
	for (int i = 0; i < contour.size(); i++)
	{
		int tmp_dist = abs(cent_p.x - contour[i].x) + abs(cent_p.y - contour[i].y);
		if (tmp_dist > max_dist)
		{
			max_dist = tmp_dist;
			max_dist_ind = i;
		}
	}

	//找到相距最远的两个点
	max_dist = 0;
	int max_dist_ind2 = 0;
	for (int i = 0; i < contour.size(); i++)
	{
		int tmp_dist = abs(contour[max_dist_ind].x - contour[i].x) + abs(contour[max_dist_ind].y - contour[i].y);
		if (tmp_dist > max_dist)
		{
			max_dist = tmp_dist;
			max_dist_ind2 = i;
		}
	}

#ifdef ROTATE_IMG_DEBUG
	circle(newmat, contour[max_dist_ind], 5, Scalar(170, 170, 170));
	circle(newmat, contour[max_dist_ind2], 5, Scalar(170, 170, 170));
#endif // ROTATE_IMG_DEBUG
	//计算角度
	//double delt_y = contour[max_dist_ind2].y - contour[max_dist_ind].y;
	//double delt_x = contour[max_dist_ind2].x - contour[max_dist_ind].x;
	Point first_p, second_p;
	if (contour[max_dist_ind2].y > contour[max_dist_ind].y)
	{
		first_p = contour[max_dist_ind2];
		second_p = contour[max_dist_ind];
	}
	else
	{
		first_p = contour[max_dist_ind];
		second_p = contour[max_dist_ind2];
	}
	double ang_tan = double(first_p.y - second_p.y) / (first_p.x - second_p.x);
	//double dist_pp = sqrt(pow(contour[max_dist_ind2].y - contour[max_dist_ind].y,2) + pow(contour[max_dist_ind2].x - contour[max_dist_ind].x,2));
	double angle = atan(ang_tan);
	//设置旋转矩阵


	for (int i = 0; i < contour.size(); i++)
	{
		contour[i] = contour[i] - cent_p;
	}
	vector<Point2f> contourf;
	for (int i = 0; i < contour.size(); i++)
	{
		Point2f ptf;
		ptf.x = float(contour[i].x);
		ptf.y = float(contour[i].y);
		contourf.push_back(ptf);
	}

	rotatePoints(contourf, -angle);

	double optimum_angle = getOptimumRotateAngle(contourf);

	cv::Rect contRect;
	getContourRect(contourf, contRect);

	cv::Size imSize;
	imSize.width = 1.5f*(contRect.width);
	imSize.height = 2 * (contRect.height);
	cv::Mat cmat = cv::Mat(imSize, CV_8UC1, cv::Scalar(0, 0, 0));

	cv::Point2f img_cent_p = cv::Point2f(cmat.cols / 2.0f, cmat.rows / 2.0f);
	for (int i = 0; i < contourf.size(); i++)
	{
		contourf[i] = contourf[i] + img_cent_p;
		contour[i] = contourf[i];
	}
	


	//重新绘制新的contour//为了提高旋转精度
	std::vector<std::vector<cv::Point>>().swap(mcontours);
	mcontours.push_back(contour);
	cv::drawContours(cmat, mcontours, 0, cv::Scalar(255, 255, 255),CV_FILLED);
	int kernel_size = contRect.width/5;
	kernel_size = (kernel_size > 30) ? 30 : kernel_size;
	kernel_size = (kernel_size < 5) ? 5 : kernel_size;
	Mat element  = getStructuringElement(MORPH_RECT, Size(kernel_size, 1));
	morphologyEx(cmat, cmat, MORPH_OPEN, element);
	//cv::fillConvexPoly(cmat, contour, Scalar(255, 255, 255));

#ifdef ROTATE_IMG_DEBUG
	imshow("1填充轮廓", cmat);
#endif // ROTATE_IMG_DEBUG

	
	//重新寻找contour
	std::vector<std::vector<cv::Point>>().swap(mcontours);
	std::vector<cv::Vec4i>().swap(hierarchy_x);
	cv::findContours(cmat, mcontours, hierarchy_x, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	if (mcontours.size()!=0)
	{
		if (mcontours[0].size()!=0)
		{
			std::vector<cv::Point2f>().swap(contourf);
			for (int i = 0; i < mcontours[0].size(); i++)
			{
				cv::Point2f ptf_t;
				ptf_t.x = mcontours[0][i].x;
				ptf_t.y = mcontours[0][i].y;
				contourf.push_back(ptf_t);
			}
			optimum_angle += getOptimumRotateAngle(contourf);
		}
	}


	//optimum_angle = -getOptimumRotateAngle(contourf);
	double new_ratio = ratio_w_h(contourf);

	optimum_angle = optimum_angle + angle;
	if (new_ratio < mthreshold)
	{
		return 0;
	}
	ImageProcessFunc::rotate_arbitrarily_angle(src_img, dst_img, optimum_angle);
#ifdef ROTATE_IMG_DEBUG
	printf("宽高比：%f\n", new_ratio);
	imshow("rotateed_img", dst_img);
	//vector<vector<Point>>().swap(contours_x);
	//contours_x.push_back(contour);
	//drawContours(newmat, contours_x, 0, Scalar(255, 255, 255));
	//imshow("rotate_contour", newmat);
#endif // ROTATE_IMG_DEBUG
	//

	return 1;

}

double OcrAlgorithm::rotateImg_ORB(Mat src_img, Mat referenceMat, Mat &dst_img)
{

	const int MAX_FEATURES = 500;
	const float GOOD_MATCH_PERCENT = 0.15f;
	// Convert images to grayscale


	// Variables to store keypoints and descriptors
	std::vector<KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;

	// Detect ORB features and compute descriptors.
	cv::Ptr<cv::Feature2D> orb = cv::ORB::create(MAX_FEATURES);
	orb->detectAndCompute(src_img, cv::Mat(), keypoints1, descriptors1);
	orb->detectAndCompute(referenceMat, cv::Mat(), keypoints2, descriptors2);
	// Match features.
	std::vector<cv::DMatch> matches;

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptors1, descriptors2, matches, Mat());

	// Sort matches by score
	std::sort(matches.begin(), matches.end());

	// Remove not so good matches
	const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
	matches.erase(matches.begin() + numGoodMatches, matches.end());


	// Draw top matches

#ifdef ROTATE_IMG_DEBUG
	Mat imMatches;
	drawMatches(src_img, keypoints1, referenceMat, keypoints2, matches, imMatches);
	imshow("imMatches", imMatches);
#endif // ROTATE_IMG_DEBUG

	//imwrite("matches.jpg", imMatches);

	//计算距离
	double distance1 = 0;

	for (size_t i = 0; i < matches.size(); i++)
	{
		distance1 += matches[i].distance;
	}
	distance1 = distance1 / matches.size();


	// Extract location of good matches
	std::vector<Point2f> points1, points2;

	for (size_t i = 0; i < matches.size(); i++)
	{
		points1.push_back(keypoints1[matches[i].queryIdx].pt);
		points2.push_back(keypoints2[matches[i].trainIdx].pt);
	}

	// Find homography
	cv::Mat h = cv::findHomography(points1, points2, RANSAC);
	if (h.empty())
	{
		return 0;
	}
	// Use homography to warp image
	cv::warpPerspective(src_img, dst_img, h, referenceMat.size());
	

	return distance1;
}

int OcrAlgorithm::rotateImg_SIFT(cv::Mat &src_img, cv::Mat &dst_img1, cv::Mat &dst_img2,OcrAlgorithm_config *pConfig)
{
	const int MAX_KEYPOINTS = 500;
	using namespace cv;
	using namespace std;
	using namespace cv::xfeatures2d;

	const cv::Size refSize(504, 293); //参考的图像宽高。

	cv::Mat im1Gray = src_img;

	if (src_img.channels()==3)
	{
		cv::cvtColor(src_img, im1Gray, CV_BGR2GRAY);
	}

	std::vector<KeyPoint> keypoints1;
	cv::Mat descriptors1;
	cv::Ptr<Feature2D> sift1 = cv::xfeatures2d::SIFT::create(MAX_KEYPOINTS);
	sift1->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);


	//waitKey(0);
	// Match features.
	//vector<vector<DMatch>> m_knnMatches;
	//vector<DMatch>m_Matches;
	//vector<DMatch>n_Matches;
	//std::vector<DMatch> best_matches;

	std::vector<DMatch> matches1;
	std::vector<DMatch> matches2;
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(6);
	matcher->match(pConfig->match_data.descriptors2, descriptors1, matches1, Mat());
	matcher->match(pConfig->match_data.descriptors3, descriptors1, matches2, Mat());

	//matcher->knnMatch(descriptors2, descriptors1, m_knnMatches, 4);
	//// Sort matches by score
	//for (int i = 0; i < m_knnMatches.size(); i++)
	//{
	//	double dist_rate1 = m_knnMatches[i][0].distance / m_knnMatches[i][1].distance;
	//	double dist_rate2 = m_knnMatches[i][1].distance / m_knnMatches[i][2].distance;
	//	double dist_rate3 = m_knnMatches[i][2].distance / m_knnMatches[i][3].distance;
	//	//if (dist_rate1< 0.5 && dist_rate2<0.5)
	//	//{
	//	//	continue;
	//	//}
	//	//if (dist_rate2 < 0.7)
	//	//{
	//	//	best_matches.push_back(m_knnMatches[i][0]);
	//	//}
	//	if (dist_rate1 < 0.4)
	//	{
	//		best_matches.push_back(m_knnMatches[i][0]);
	//	}
	//}
	//排序

	//std::sort(matches.begin(), matches.end());

	std::sort(matches1.begin(), matches1.end());
	std::sort(matches2.begin(), matches2.end());




	std::vector<DMatch> best_matches_1;
	std::vector<DMatch> best_matches_2;
	std::vector<DMatch>::iterator it;
	it = (matches1.size() < 50) ? matches1.end() : matches1.begin() + 50;
	best_matches_1.insert(best_matches_1.end(), matches1.begin(), it);
	it = (matches2.size() < 50) ? matches2.end() : matches2.begin() + 50;
	best_matches_2.insert(best_matches_2.end(), matches2.begin(), it);

	//将出现大于1个对应点的match 删除多余对应点
	std::vector<DMatch> best_matches_1_;
	std::vector<DMatch> best_matches_2_;
	for (int i = 0; i < best_matches_1.size(); i++)
	{
		int res = query_match_count_(best_matches_1_, best_matches_1[i]);
		if (res == 0)
		{
			best_matches_1_.push_back(best_matches_1[i]);
		}
	}
	for (int i = 0; i < best_matches_2.size(); i++)
	{
		int res = query_match_count_(best_matches_2_, best_matches_2[i]);
		if (res == 0)
		{
			best_matches_2_.push_back(best_matches_2[i]);
		}
	}




	if (best_matches_1_.size()>=4)
	{
		std::vector<Point2f> points_1, points_ref;
		for (size_t i = 0; i < best_matches_1_.size(); i++)
		{
			points_1.push_back(keypoints1[best_matches_1_[i].trainIdx].pt);
			points_ref.push_back(pConfig->match_data.keypoints2[best_matches_1_[i].queryIdx].pt);
		}
		Mat h1 = cv::findHomography(points_1, points_ref, cv::RANSAC);
		//// Use homography to warp image
		if (!h1.empty())
		{
			cv::warpPerspective(src_img, dst_img1, h1, refSize);
#ifdef ROTATE_IMG_DEBUG
			imshow("sift旋转1", dst_img1);
#endif // DEBUG

		}

	}
	if (best_matches_2_.size() >= 4)
	{
		std::vector<Point2f> points_1, points_ref;
		for (size_t i = 0; i < best_matches_2_.size(); i++)
		{
			points_1.push_back(keypoints1[best_matches_2_[i].trainIdx].pt);
			points_ref.push_back(pConfig->match_data.keypoints3[best_matches_2_[i].queryIdx].pt);
		}
		Mat h1 = cv::findHomography(points_1, points_ref, cv::RANSAC);
		//// Use homography to warp image
		if (!h1.empty())
		{
			cv::warpPerspective(src_img, dst_img2, h1, refSize);
#ifdef ROTATE_IMG_DEBUG
			imshow("sift旋转2", dst_img2);
#endif // DEBU

		}
	}
	//imshow("srd", src_img);
	return 1;
}


int OcrAlgorithm::rotateImg_SURF(cv::Mat &src_img, cv::Mat &dst_img1, cv::Mat &dst_img2, OcrAlgorithm_config *pConfig)
{
	const int MAX_KEYPOINTS = 500;
	using namespace cv;
	using namespace std;
	using namespace cv::xfeatures2d;

	const cv::Size refSize(504, 293); //参考的图像宽高。

	cv::Mat im1Gray = src_img;

	if (src_img.channels() == 3)
	{
		cv::cvtColor(src_img, im1Gray, CV_BGR2GRAY);
	}

	std::vector<KeyPoint> keypoints1;
	cv::Mat descriptors1;
	cv::Ptr<Feature2D> sift1 = cv::xfeatures2d::SURF::create(MAX_KEYPOINTS);
	sift1->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);


	//waitKey(0);
	// Match features.
	//vector<vector<DMatch>> m_knnMatches;
	//vector<DMatch>m_Matches;
	//vector<DMatch>n_Matches;
	//std::vector<DMatch> best_matches;

	std::vector<DMatch> matches1;
	std::vector<DMatch> matches2;
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(6);
	matcher->match(pConfig->match_data.descriptors2, descriptors1, matches1, Mat());
	matcher->match(pConfig->match_data.descriptors3, descriptors1, matches2, Mat());

	//matcher->knnMatch(descriptors2, descriptors1, m_knnMatches, 4);
	//// Sort matches by score
	//for (int i = 0; i < m_knnMatches.size(); i++)
	//{
	//	double dist_rate1 = m_knnMatches[i][0].distance / m_knnMatches[i][1].distance;
	//	double dist_rate2 = m_knnMatches[i][1].distance / m_knnMatches[i][2].distance;
	//	double dist_rate3 = m_knnMatches[i][2].distance / m_knnMatches[i][3].distance;
	//	//if (dist_rate1< 0.5 && dist_rate2<0.5)
	//	//{
	//	//	continue;
	//	//}
	//	//if (dist_rate2 < 0.7)
	//	//{
	//	//	best_matches.push_back(m_knnMatches[i][0]);
	//	//}
	//	if (dist_rate1 < 0.4)
	//	{
	//		best_matches.push_back(m_knnMatches[i][0]);
	//	}
	//}
	//排序

	//std::sort(matches.begin(), matches.end());

	std::sort(matches1.begin(), matches1.end());
	std::sort(matches2.begin(), matches2.end());




	std::vector<DMatch> best_matches_1;
	std::vector<DMatch> best_matches_2;
	std::vector<DMatch>::iterator it;
	it = (matches1.size() < 50) ? matches1.end() : matches1.begin() + 50;
	best_matches_1.insert(best_matches_1.end(), matches1.begin(), it);
	it = (matches2.size() < 50) ? matches2.end() : matches2.begin() + 50;
	best_matches_2.insert(best_matches_2.end(), matches2.begin(), it);

	//将出现大于1个对应点的match 删除多余对应点
	std::vector<DMatch> best_matches_1_;
	std::vector<DMatch> best_matches_2_;
	for (int i = 0; i < best_matches_1.size(); i++)
	{
		int res = query_match_count_(best_matches_1_, best_matches_1[i]);
		if (res == 0)
		{
			best_matches_1_.push_back(best_matches_1[i]);
		}
	}
	for (int i = 0; i < best_matches_2.size(); i++)
	{
		int res = query_match_count_(best_matches_2_, best_matches_2[i]);
		if (res == 0)
		{
			best_matches_2_.push_back(best_matches_2[i]);
		}
	}




	if (best_matches_1_.size() >= 4)
	{
		std::vector<Point2f> points_1, points_ref;
		for (size_t i = 0; i < best_matches_1_.size(); i++)
		{
			points_1.push_back(keypoints1[best_matches_1_[i].trainIdx].pt);
			points_ref.push_back(pConfig->match_data.keypoints2[best_matches_1_[i].queryIdx].pt);
		}
		Mat h1 = cv::findHomography(points_1, points_ref, cv::RANSAC);
		//// Use homography to warp image
		if (!h1.empty())
		{
			cv::warpPerspective(src_img, dst_img1, h1, refSize);
#ifdef ROTATE_IMG_DEBUG
			imshow("sift旋转1", dst_img1);
#endif // DEBUG

		}

	}
	if (best_matches_2_.size() >= 4)
	{
		std::vector<Point2f> points_1, points_ref;
		for (size_t i = 0; i < best_matches_2_.size(); i++)
		{
			points_1.push_back(keypoints1[best_matches_2_[i].trainIdx].pt);
			points_ref.push_back(pConfig->match_data.keypoints3[best_matches_2_[i].queryIdx].pt);
		}
		Mat h1 = cv::findHomography(points_1, points_ref, cv::RANSAC);
		//// Use homography to warp image
		if (!h1.empty())
		{
			cv::warpPerspective(src_img, dst_img2, h1, refSize);
#ifdef ROTATE_IMG_DEBUG
			imshow("sift旋转2", dst_img2);
#endif // DEBU

		}
	}
	//imshow("srd", src_img);
	return 1;

}

int OcrAlgorithm::loadMatchData(const std::string &xmlfile, cv::Mat &descriptors2, cv::Mat &descriptors3, std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::KeyPoint> &keypoints3)
{
	cv::FileStorage fs(xmlfile,cv::FileStorage::READ);
	if (!fs.isOpened()) return 0;
	
	fs["descriptors2"] >> descriptors2;
	fs["descriptors3"] >> descriptors3;
	std::vector<KeyPoint>().swap(keypoints2);
	int ind = 0;
	while (true)
	{
		string s = std::to_string(ind);
		s = "keypoints2" + s;
		KeyPoint kpt;
		fs[s] >> kpt;
		keypoints2.push_back(kpt);
		if (kpt.octave == 0) break;
		ind++;
	}
	ind = 0;
	while (true)
	{
		string s = std::to_string(ind);
		s = "keypoints3" + s;
		KeyPoint kpt;
		fs[s] >> kpt;
		keypoints3.push_back(kpt);
		if (kpt.octave == 0) break;
		ind++;
	}
	fs.release();

	return 1;
}

int OcrAlgorithm::saveMatchData(const std::string &xmlfile, cv::Mat &descriptors2, cv::Mat &descriptors3, std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::KeyPoint> &keypoints3)
{
	cv::FileStorage fs(xmlfile, cv::FileStorage::WRITE);
	if (!fs.isOpened()) return 0;

	fs << "descriptors2" << descriptors2;
	fs << "descriptors3" << descriptors3;
	for (int i = 0; i < keypoints2.size(); i++)
	{
		string s = std::to_string(i);
		s = "keypoints2" + s;
		fs << s << keypoints2[i];
	}
	for (int i = 0; i < keypoints3.size(); i++)
	{
		string s = std::to_string(i);
		s = "keypoints3" + s;
		fs << s << keypoints3[i];
	}
	fs.release();

	return 1;
}

int OcrAlgorithm::getPostcodeRoi(Mat &srcImg, std::vector<cv::Rect> &dstRects, Rect mrect, std::string tmplatePath)
{
	// 进行图像翻转
	//cv::rectangle(srcImg, mrect, Scalar(255, 255, 255), 2);
	//imshow("原始图像", srcImg);
	//限制行高

	int w = srcImg.cols;
	int h = srcImg.rows;
	//Mat flipedMat = srcImg;
	Point rect_centor = Point(mrect.x + mrect.width / 2, mrect.y + mrect.height / 2);
	if (rect_centor.x < w / 2.0)
	{
		flip(srcImg, srcImg, 1);
		mrect.x = w - (mrect.x + mrect.width);
	}
	//rect_centor = Point(mrect.x + mrect.width / 2.0, mrect.y + mrect.y / 2.0);
	if (rect_centor.y < h / 2.0)
	{
		flip(srcImg, srcImg, 0);
		mrect.y = h - (mrect.y + mrect.height);
	}
	//cv::rectangle(flipedMat, mrect,Scalar(255, 255, 255),2);

	//Mat tmplimg = imread("E:\\cpp_projects\\code\\ImageAdjust\\ImageAdjust\\templateimg.jpg");
	//matchImage(flipedMat, tmplimg);

	// 定位邮编,r表示rect
	int r_w = mrect.width;
	int r_h = mrect.height;
	int r_center_x = mrect.x + mrect.width / 2;
	int start_x = r_center_x - r_h / 2;
	if (start_x < 0) start_x = 0;
	int start_y = mrect.y - 4.0*r_h;
	if (start_y < 0) start_y = 0;
	int end_x = mrect.x + mrect.width + mrect.height;
	if (end_x > w) end_x = w;
	Rect searchRech;
	searchRech.x = start_x;
	searchRech.y = start_y;
	searchRech.width = end_x - start_x;
	searchRech.height = mrect.y - start_y;
	//cv::rectangle(flipedMat, searchRech, Scalar(255, 255, 255), 2);
	//Mat roiMat = srcImg(searchRech);
	//Mat sRoiMat = srcImg(searchRech);
	std::vector<cv::Rect> mRects;
	cv::Rect roiRect;
	int res = getPostcodeRoiInRectImg(srcImg, searchRech, roiRect);
	if (res != 0)
	{
		mRects.push_back(roiRect);
	}

	//通过模板匹配的方式进行y方向定位
	//getPostcodeRoi_TPmatch(srcImg, mrect, tmplatePath, mRects);

	//剔除重复的结果
	vector<cv::Rect>::iterator it;
	vector<cv::Rect>::iterator pre_it = mRects.begin();
	if (mRects.size() != 0)
	{
		for (pre_it = mRects.begin(); pre_it != mRects.end(); pre_it++)
		{
			for (it = pre_it + 1; it != mRects.end();)
			{
				double miou = iou_y(*pre_it, *it);
				if (miou>0.5)
				{
					it = mRects.erase(it);
				}
				else
				{
					it++;
				}
			}
		}
	}

	dstRects.insert(dstRects.end(), mRects.begin(), mRects.end());



#ifdef POSTCODE_ROI_DEBUG
	cv::Mat showMat;
	srcImg.copyTo(showMat);
	for (int i=0;i<mRects.size();i++)
	{
		rectangle(showMat, mRects[i], cv::Scalar(0, 0, 0), 3);
	}
	imshow("邮编定位候选框", showMat);


#endif // POSTCODE_ROI_DEBUG

	//
	//imshow("翻转图像", flipedMat);
	//waitKey(0);
	
	return 1;
}

int OcrAlgorithm::getPostcodeRoi_TPmatch(cv::Mat &srcImg, Rect barcoderect, std::string templateImgPath, std::vector<cv::Rect> &roiRects)
{
	if (templateImgPath.empty())
	{
		printf("模板图片路径为空\n");
		return 0;
	}
	Mat templateImg = imread(templateImgPath);
	if (templateImg.empty())
	{
		printf("模板图片为空\n");
		return 0;
	}

	//旋转图片
	int w = srcImg.cols;
	int h = srcImg.rows;
	//Mat flipedMat = srcImg;
	Point rect_centor = Point(barcoderect.x + barcoderect.width / 2, barcoderect.y + barcoderect.height / 2);
	if (rect_centor.x < w / 2.0)
	{
		flip(srcImg, srcImg, 1);
		barcoderect.x = w - (barcoderect.x + barcoderect.width);
	}
	//rect_centor = Point(mrect.x + mrect.width / 2.0, mrect.y + mrect.y / 2.0);
	if (rect_centor.y < h / 2.0)
	{
		flip(srcImg, srcImg, 0);
		barcoderect.y = h - (barcoderect.y + barcoderect.height);
	}

	//模板匹配
	std::vector<cv::Rect> TpRects;
	if (matchImage(srcImg, templateImg, TpRects) == 0)
	{
		printf("模板匹配未找到目标\n");
		return 0;
	}

	//剔除不正确的匹配结果
	std::vector<cv::Rect>::iterator it;
	int anchorPos = barcoderect.x - 2 * barcoderect.height;
	for (it=TpRects.begin();it!=TpRects.end();)
	{
		if (it->x > anchorPos)
		{
			it = TpRects.erase(it);
		}
		else
		{
			it++;
		}
	}
	if (TpRects.size()==0)
	{
		printf("匹配的目标不正确\n");
		return 0;
	}
	
	//获取邮编roi
	for (int i=0;i<TpRects.size();i++)
	{
		int r_w = barcoderect.width;
		int r_h = barcoderect.height;
		int r_center_x = barcoderect.x + barcoderect.width / 2;
		int start_x = r_center_x - r_h / 2;
		if (start_x < 0) start_x = 0;
		int start_y = TpRects[i].y;
		if (start_y < 0) start_y = 0;
		int end_x = barcoderect.x + barcoderect.width + barcoderect.height;
		if (end_x > w) end_x = w;
		if (start_x >= end_x)
		{
			continue;
		}
		if (start_y >= barcoderect.y)
		{
			continue;
		}
		Rect searchRech;
		searchRech.x = start_x;
		searchRech.y = start_y;
		searchRech.width = end_x - start_x;
		searchRech.height = barcoderect.y - start_y;
		Rect roiRect;
		int res = getPostcodeRoiInRectImg(srcImg, searchRech, roiRect);
		if (res!=0)
		{
			roiRects.push_back(roiRect);
		}
	}

	return 1;
}

int OcrAlgorithm::getPostcodeRoiInRectImg(cv::Mat srcImg, cv::Rect sRect, cv::Rect &roiRect)
{
	cv::Mat roiMat = srcImg(sRect);

	//对图像进行缩放
	int s_h = roiMat.rows;
	int s_w = roiMat.cols;
	int s_size = s_h;
	//int s_a = s_w * s_h;
	int dst_size = 200;
	double _rate = dst_size / double(s_size);
	int d_w = s_w * _rate;
	int d_h = s_h * _rate;
	resize(roiMat, roiMat, Size(d_w, d_h));
#ifdef POSTCODE_ROI_DEBUG
	imshow("resized 图像", roiMat);
#endif // 
	d_w = roiMat.cols;
	d_h = roiMat.rows;


	int roi_h = roiMat.rows;
	int roi_w = roiMat.cols;
	Mat g_img;
	GaussianBlur(roiMat, g_img, Size(3, 3), 0);
	Canny(g_img, g_img, 20, 250);

#ifdef POSTCODE_ROI_DEBUG
	imshow("canny处理结果", g_img);
#endif // POSTCODE_ROI_DEBUG

	
	threshold(g_img, g_img, 30, 255, CV_THRESH_BINARY);
	//bitwise_not(g_img, g_img);

	// 去除字符区域右侧干扰
	Mat element = getStructuringElement(MORPH_RECT, Size(10, 1));
	morphologyEx(g_img, g_img, MORPH_CLOSE, element);
	vector<unsigned int> PixelsAdd;
	sumPixels(g_img, 1, PixelsAdd);
	int n = PixelsAdd.size();
	int cutPos = 0;
	for (int i = n * 2 / 3; i < n; i++)
	{
		if (PixelsAdd[i] < 5000)
		{
			cutPos = i;
			break;
		}
	}
	if (cutPos != 0)
	{
		//填充多边形
		g_img = g_img(Rect(0, 0, cutPos, roi_h-1));
	}


	//形态学运算
	element = getStructuringElement(MORPH_RECT, Size(25, 1));
	morphologyEx(g_img, g_img, MORPH_CLOSE, element);
	element = getStructuringElement(MORPH_RECT, Size(25, 3));
	morphologyEx(g_img, g_img, MORPH_ERODE, element);
	element = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(g_img, g_img, MORPH_ERODE, element);
#ifdef POSTCODE_ROI_DEBUG
	imshow("形态学处理后图像", g_img);
#endif // POSTCODE_ROI_DEBUG
	//
	//waitKey(0);

	//缩小统计区域
	g_img = g_img(Rect(g_img.cols/3, 0, g_img.cols*2/3, g_img.rows));

	// 计算可能包含文字的行
	vector<unsigned int>().swap(PixelsAdd);
	sumPixels(g_img, 0, PixelsAdd);

	//将
	unsigned int max_pixes = 0;
	int isolate_count = 0;
	bool is_continue = false;
	vector<Point> bars_vec;
	Point bar_pt;
	for (int i = 0; i < PixelsAdd.size(); i++)
	{
		if (PixelsAdd[i] > 0 && is_continue == false)
		{
			is_continue = true;
			bar_pt.x = i;
		}
		if (is_continue == true && PixelsAdd[i] == 0)
		{
			bar_pt.y = i - 1;
			is_continue = false;
			bars_vec.push_back(bar_pt);
		}
		if (i == PixelsAdd.size() - 1 && is_continue == true)
		{
			bar_pt.y = i - 1;
			is_continue = false;
			bars_vec.push_back(bar_pt);
		}

	}

	// 联通短暂的不连续区域
	if (bars_vec.size() == 0)
	{
#ifdef POSTCODE_ROI_DEBUG
		printf("未找到字符行\n");
#endif // POSTCODE_ROI_DEBUG
		return 0;
	}
	//Point pre_pt = bars_vec[0];
	vector<Point>::iterator it;
	vector<Point>::iterator pre_it = bars_vec.begin();
	for (it = bars_vec.begin() + 1; it != bars_vec.end();)
	{
		if ((it->x - pre_it->y) <= 2)
		{
			pre_it->y = it->y;
			it = bars_vec.erase(it);
		}
		else
		{
			pre_it = it;
			it++;
		}

	}
	if (bars_vec.size() == 0)
	{
#ifdef POSTCODE_ROI_DEBUG
		printf("联通字符行后，未找到字符行\n");
#endif // POSTCODE_ROI_DEBUG
		return 0;
	}


	// 删除不连续行区域，如果区域高小于5，者删除之
	for (it = bars_vec.begin(); it != bars_vec.end(); )
	{
		if (it->y - it->x < 5)
		{
			it = bars_vec.erase(it);
		}
		else
		{
			it++;
		}
	}

	if (bars_vec.size() == 0)
	{
#ifdef POSTCODE_ROI_DEBUG
		printf("未找到字符行\n");
#endif // POSTCODE_ROI_DEBUG
		return 0;
	}

	// 删除干扰区域，如果下一个区域位置超过本区域高的两倍，认为不是正确区域
	pre_it = bars_vec.begin();
	for (it = bars_vec.begin() + 1; it != bars_vec.end();)
	{
		if ((it->x - pre_it->y) > 2 * (pre_it->y - pre_it->x))
		{
			int n = it - bars_vec.begin();
			if (n >= 3)
			{
				break;
			}
			it = bars_vec.erase(bars_vec.begin(), it);
			pre_it = it;
			it++;
		}
		else
		{
			pre_it = it;
			it++;
		}

	}
	if (bars_vec.size() < 3)
	{
#ifdef POSTCODE_ROI_DEBUG
		printf("字符行不正确\n");
#endif // POSTCODE_ROI_DEBUG
		return 0;
	}


	// 确定邮编行区域
	int bar_pos = (bars_vec[2].x + bars_vec[2].y) / 2;
	int bar_width = bars_vec[2].y - bars_vec[2].x;

	int postcode_h = bar_width * 1.7+3;
	int postcode_y = (bar_pos - postcode_h / 2 < 0) ? 0 : (bar_pos - postcode_h / 2);
	Rect Rec_postcode = Rect(0, postcode_y, roi_w, postcode_h);
	//cv::rectangle(roiMat, Rec_postcode, Scalar(255, 255, 255), 2);
	if (postcode_y + postcode_h >= roi_h)
	{
#ifdef POSTCODE_ROI_DEBUG
		printf("邮编定位超出界限\n");
#endif // POSTCODE_ROI_DEBUG
		return 0;
	}
	Rec_postcode.x = Rec_postcode.x / _rate;
	Rec_postcode.y = Rec_postcode.y / _rate;
	Rec_postcode.width = Rec_postcode.width / _rate;
	Rec_postcode.height = Rec_postcode.height / _rate;

	Rec_postcode.x += sRect.x;
	Rec_postcode.y += sRect.y;


	roiRect = Rec_postcode;
	return 1;
}

int OcrAlgorithm::getPostcodeRoiInRectImg_accordPos(cv::Mat srcImg, cv::Rect sRect, int anchor_row, cv::Rect &roiRect)
{
	cv::Mat roiMat ;
	srcImg(sRect).copyTo(roiMat);
	adJustBrightness(roiMat, 1.5, 0, 120); //图像增强
	//对图像进行缩放
	int s_h = roiMat.rows;
	int s_w = roiMat.cols;
	int s_size = s_h;
	//int s_a = s_w * s_h;
	int dst_size = 200;
	double _rate = dst_size / double(s_size);
	int d_w = s_w * _rate;
	int d_h = s_h * _rate;
	resize(roiMat, roiMat, Size(d_w, d_h));
	anchor_row = anchor_row * _rate;
//#ifdef POSTCODE_ROI_DEBUG
//	imshow("resized 图像", roiMat);
//#endif // 
	
	//imshow("roimat", roiMat);
	//waitKey(0);

	int roi_h = roiMat.rows;
	int roi_w = roiMat.cols;
	Mat g_img;
	GaussianBlur(roiMat, g_img, Size(3, 3), 0);
	Canny(g_img, g_img, 20, 250);

	threshold(g_img, g_img, 30, 255, CV_THRESH_BINARY);

	cv::Mat element = getStructuringElement(MORPH_RECT, Size(25, 1));
	morphologyEx(g_img, g_img, MORPH_CLOSE, element);
	element = getStructuringElement(MORPH_RECT, Size(25, 3));
	morphologyEx(g_img, g_img, MORPH_ERODE, element);
	element = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(g_img, g_img, MORPH_ERODE, element);

	vector<unsigned int> PixelsAdd;
	sumPixels(g_img, 0, PixelsAdd);

	//imshow("roimat", g_img);
	//waitKey(0);
	//将提取区域
	unsigned int max_pixes = 0;
	int isolate_count = 0;
	bool is_continue = false;
	vector<Point> bars_vec;
	Point bar_pt;
	for (int i = 0; i < PixelsAdd.size(); i++)
	{
		if (PixelsAdd[i] > 0 && is_continue == false)
		{
			is_continue = true;
			bar_pt.x = i;
		}
		if (is_continue == true && PixelsAdd[i] == 0)
		{
			bar_pt.y = i - 1;
			is_continue = false;
			bars_vec.push_back(bar_pt);
		}
		if (i == PixelsAdd.size() - 1 && is_continue == true)
		{
			bar_pt.y = i - 1;
			is_continue = false;
			bars_vec.push_back(bar_pt);
		}

	}
	
	//剔除宽度较小的bar
	std::vector<cv::Point>::iterator it;
	for (it = bars_vec.begin(); it != bars_vec.end(); )
	{
		if ((it->y - it->x) < 5)
		{
			it = bars_vec.erase(it);
		}
		else
		{
			it++;
		}
	}


	int ind = -1;
	cv::Point dstbar;
	for (int i=0;i<bars_vec.size();i++)
	{
		if (bars_vec[i].x<= anchor_row && bars_vec[i].y >= anchor_row)
		{
			dstbar = bars_vec[i];
			ind = i;
			break;
		}
	}
	if (ind==-1)
	{
		//printf("通过指定位置未能确定邮编位置\n");
		//printf("采用动态邮编位置\n");
		//int res = getPostcodeRoiInRectImg(srcImg, sRect, roiRect);
		//if (res == 0)
		//{
		//	return 0;
		//}
		return 0;
	}
	else
	{
		dstbar.x = dstbar.x / _rate;
		dstbar.y = dstbar.y / _rate;
		int bar_cent = (dstbar.y + dstbar.x) / 2;
		int bar_width = 1.5*(dstbar.y - dstbar.x + 1);
		roiRect.x = sRect.x;
		roiRect.y = sRect.y + bar_cent - bar_width/2;
		roiRect.y = (roiRect.y > 0) ? roiRect.y : 0;
		roiRect.width = sRect.width;
		roiRect.height = bar_width;
	}

	return 1;

}

int OcrAlgorithm::getPostcodeRoiInRectImg_SiftMatch(cv::Mat srcImg, cv::Rect sRect, cv::Rect &roiRect)
{




	return 1;

}

//幅度
void OcrAlgorithm::rotate_arbitrarily_angle(Mat &src, Mat &dst, float angle)
{
	float radian = angle;//(float)(angle / 180.0 * CV_PI);     //填充图像
	float angle_dec = angle / CV_PI * 180;
	int maxBorder = (int)(max(src.cols, src.rows)* 1.414); //即为sqrt(2)*max   
	int dx = (maxBorder - src.cols) / 2;
	int dy = (maxBorder - src.rows) / 2;
	copyMakeBorder(src, dst, dy, dy, dx, dx, BORDER_CONSTANT, Scalar(0, 0 , 0));     //旋转    
	Point2f center((float)(dst.cols / 2), (float)(dst.rows / 2));
	Mat affine_matrix = getRotationMatrix2D(center, angle_dec, 1.0);//求得旋转矩阵    
	warpAffine(dst, dst, affine_matrix, dst.size());     //计算图像旋转之后包含图像的最大的矩形    
	float sinVal = abs(sin(radian));
	float cosVal = abs(cos(radian));
	Size targetSize((int)(src.cols * cosVal + src.rows * sinVal),
		(int)(src.cols * sinVal + src.rows * cosVal));     //剪掉多余边框    
	int x = (dst.cols - targetSize.width) / 2.0;
	int y = (dst.rows - targetSize.height) / 2.0;
	Rect rect(x, y, targetSize.width, targetSize.height);
	if (rect.x + rect.width > dst.cols)
	{
		rect.width = dst.cols - rect.x;
	}
	if (rect.y + rect.height > dst.rows)
	{
		rect.height = dst.rows - rect.y;
	}

	dst = Mat(dst, rect);
}

int OcrAlgorithm::sumPixels(Mat &srcimg, int axis, vector<unsigned int> &resultsVec)
{
	int h = srcimg.rows;
	int w = srcimg.cols;
	unsigned int sump = 0;
	if (axis == 0)
	{
		for (int i = 0; i < h; i++)
		{
			sump = 0;
			for (int j = 0; j < w; j++)
			{
				sump += srcimg.at<uchar>(i, j);
			}
			resultsVec.push_back(sump);
		}
	}
	if (axis == 1)
	{
		for (int i = 0; i < w; i++)
		{
			sump = 0;
			for (int j = 0; j < h; j++)
			{
				sump += srcimg.at<uchar>(j, i);
			}
			resultsVec.push_back(sump);
		}
	}
	return 1;
}

int OcrAlgorithm::rotatePoints(vector<Point2f> & points_vec, double angle)
{
	double sin_x = sin(angle);
	double cos_x = cos(angle);
	Mat r_mat = (Mat_<float>(2, 2) << cos_x, -sin_x, sin_x, cos_x);

	Mat points_mat = Mat(2, points_vec.size(), CV_32FC1);
	for (int i = 0; i < points_vec.size(); i++)
	{
		points_mat.at<float>(0, i) = points_vec[i].x;
		points_mat.at<float>(1, i) = points_vec[i].y;
	}
	Mat times_resut_m;
	times_resut_m = r_mat * points_mat;
	for (int i = 0; i < points_vec.size(); i++)
	{
		points_vec[i].x = times_resut_m.at<float>(0, i);
		points_vec[i].y = times_resut_m.at<float>(1, i);
	}
	return 0;
}

int OcrAlgorithm::rotatePoints(vector<Point> & points_vec, double angle)
{
	vector<Point2f> points_vecf;
	cv::Point2f pt;
	for (int i=0;i<points_vec.size();i++)
	{
		pt.x = points_vec[i].x;
		pt.y = points_vec[i].y;
		points_vecf.push_back(pt);
	}
	rotatePoints(points_vecf, angle);
	for (int i = 0; i < points_vec.size(); i++)
	{
		points_vec[i].x = round(points_vecf[i].x);
		points_vec[i].y = round(points_vecf[i].y);
	}
	return 1;
}

double OcrAlgorithm::getOptimumRotateAngle(vector<Point2f> & points_vec)
{
	int old1_dirct_f = 1;
	int old2_dirct_f = 1;
	int direc_flag = 1;//旋转方向
	double rate_step = 1.0 / 180 * CV_PI;
	double old_ratio = ratio_w_h(points_vec);
	double new_ratio = 0;
	int count_m = 0;
	double whole_rate_angle =0;
	while (true)
	{
		rotatePoints(points_vec, -direc_flag * rate_step);
		whole_rate_angle += direc_flag * rate_step;
		new_ratio = ratio_w_h(points_vec);
		old2_dirct_f = old1_dirct_f;
		old1_dirct_f = direc_flag;
		if (new_ratio < old_ratio)
		{
			direc_flag = -direc_flag;
		}
		old_ratio = new_ratio;
		if (old1_dirct_f*old2_dirct_f == -1 && old1_dirct_f*direc_flag == -1)
		{
			break;
		}
		count_m++;
		if (count_m >= 90) break;
	}
	return whole_rate_angle;
}

double OcrAlgorithm::ratio_w_h(vector<Point2f> & points_vec)
{
	Point2f lf = points_vec[0];
	Point2f rt = points_vec[0];
	Point2f tp = points_vec[0];
	Point2f bt = points_vec[0];
	for (auto pt : points_vec)
	{
		if (pt.x < lf.x) lf = pt;
		if (pt.x > rt.x) rt = pt;
		if (pt.y < tp.y) tp = pt;
		if (pt.y > bt.y) bt = pt;
	}
	float w = rt.x - lf.x;
	float h = bt.y - tp.y;
	return w / h;
}

double OcrAlgorithm::getAveragePixelInRect(Mat& src, Rect &mRect)
{
	Mat tmat = src(mRect);
	int w = tmat.cols;
	int h = tmat.rows;
	int tatalpix = 0;
	for (int i = 0; i < w; i++)
	{
		for (int j = 0; j < h; j++)
		{
			tatalpix = tatalpix + tmat.at<uchar>(j, i);
		}
	}
	return double(tatalpix) / (w*h);
}

int OcrAlgorithm::getContourRect(std::vector<cv::Point2f> & points_vec, cv::Rect &mRect)
{
	Point2f lf = points_vec[0];
	Point2f rt = points_vec[0];
	Point2f tp = points_vec[0];
	Point2f bt = points_vec[0];
	for (auto pt : points_vec)
	{
		if (pt.x < lf.x) lf = pt;
		if (pt.x > rt.x) rt = pt;
		if (pt.y < tp.y) tp = pt;
		if (pt.y > bt.y) bt = pt;
	}
	float w = rt.x - lf.x;
	float h = bt.y - tp.y;
	mRect.x = lf.x;
	mRect.width = w;
	mRect.y = tp.y;
	mRect.height = h;

	return 1;
}

int OcrAlgorithm::adJustBrightness(Mat& src, double alpha, double beta, double anchor)
{
	int height = src.rows;
	int width = src.cols;
	if (src.channels() !=1)
	{
		return 0;
	}
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			float v = src.at<uchar>(row, col);
			src.at<uchar>(row, col) = saturate_cast<uchar>(v*alpha+ (1-alpha)*anchor + beta);
		}
	}
	return 1;
}

double OcrAlgorithm::getAverageBrightness(Mat src)
{
	int height = src.rows;
	int width = src.cols;
	double b = 0;
	if (src.channels() != 1)
	{
		return 0;
	}
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			float v = src.at<uchar>(row, col);
			b += v;
		}
	}
	b = b / (height*width);
	return b;
}

int OcrAlgorithm::getLargestContour(Mat srcimg, std::vector<cv::Point> &largest_contour)
{

	//根据图像的密度调整核尺寸
	int d_w = srcimg.cols;
	int d_h = srcimg.rows;
	int step_x = d_w / 4;
	int step_y = d_h / 4;
	Rect subRect = Rect(0, 0, step_x, step_y);
	double avepix[16] = { 0 };
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			int ind = i * 4 + j;
			subRect.x = i * step_x;
			subRect.y = j * step_y;
			avepix[ind] = getAveragePixelInRect(srcimg, subRect);
		}
	}
	double a_avepix = 0;
	for (int i = 0; i < 16; i++)
	{
		a_avepix += avepix[i];
	}
	a_avepix = a_avepix / 16;//求均值像素

	//double variance_m = 0;
	//for (int i = 0; i < 16; i++)
	//{
	//	variance_m += sqrt(pow(avepix[i]-a_avepix,2));
	//}

	//printf("均值：%f,方差%f\n", a_avepix,variance_m);

	//核大小，根据图像的标签的大小动态调整
	int base_val = 10;
	int kernel_size = (a_avepix - 15) / 3 + base_val;

	kernel_size = (kernel_size < base_val) ? base_val : kernel_size;

#ifdef ROTATE_IMG_DEBUG
	printf("核大小%d\n", kernel_size);
#endif // ROTATE_IMG_DEBUG
	std::vector<std::vector<cv::Point>> large_contours;
	std::vector<cv::Point> mcontour;
	getLargestContourAccordKernelSize(srcimg, mcontour, cv::Size(kernel_size, 1));
	if (!mcontour.empty())
	{
		large_contours.push_back(mcontour);
		std::vector<cv::Point>().swap(mcontour);
	}
	getLargestContourAccordKernelSize(srcimg, mcontour, cv::Size(1,kernel_size));
	if (!mcontour.empty())
	{
		large_contours.push_back(mcontour);
		std::vector<cv::Point>().swap(mcontour);
	}

	//旋转图像45度
	//Mat img_rotate_45degree;
	//double rdegree = 45.0 / 180 * CV_PI;
	//rotate_arbitrarily_angle(srcimg, img_rotate_45degree, rdegree);
	//getLargestContourAccordKernelSize(img_rotate_45degree, mcontour, cv::Size(kernel_size, 1));
	//if (!mcontour.empty())
	//{
	//	rotatePoints(mcontour, rdegree);
	//	large_contours.push_back(mcontour);
	//	std::vector<cv::Point>().swap(mcontour);
	//}
	//getLargestContourAccordKernelSize(img_rotate_45degree, mcontour, cv::Size(1, kernel_size));
	//if (!mcontour.empty())
	//{
	//	rotatePoints(mcontour, rdegree);
	//	large_contours.push_back(mcontour);
	//	std::vector<cv::Point>().swap(mcontour);
	//}
	if (large_contours.empty())
	{
#ifdef ROTATE_IMG_DEBUG
		printf("未找到任何轮廓！\n");
#endif // 
		return 0;
	}
	double area_contour = 0;
	int larg_congour_i = 0;
	for (int i=0;i<large_contours.size();i++)
	{
		double area = contourArea(large_contours[i]);
		if (area>area_contour)
		{
			area_contour = area;
			larg_congour_i = i;
		}
	}

	largest_contour = large_contours[larg_congour_i];

	return 1;

}

int OcrAlgorithm::getLargestContourAccordKernelSize(Mat srcimg, 
	std::vector<cv::Point> &largest_contour, cv::Size ksize)
{
	//形态学开闭操作
	Mat morpholo_x;
	Mat element_x = cv::getStructuringElement(MORPH_RECT, ksize);
	cv::morphologyEx(srcimg, morpholo_x, MORPH_CLOSE, element_x);
	Mat element = cv::getStructuringElement(MORPH_RECT, Size(5, 5));
	cv::morphologyEx(morpholo_x, morpholo_x, MORPH_ERODE, element);
	int kernel_size = max(ksize.height, ksize.width);
	element = cv::getStructuringElement(MORPH_RECT, Size(kernel_size, kernel_size));
	cv::morphologyEx(morpholo_x, morpholo_x, MORPH_OPEN, element);
	//imshow("形态学运算", morpholo_x);
	//waitKey(0);
	//寻找最大轮廓
	vector<vector<Point>>contours_x;
	vector<Vec4i>hierarchy_x;
	double aera = 0;
	int max_contour_index_x = 0;
	//src_gray = src_gray > 100;
	cv::findContours(morpholo_x, contours_x,RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	if (contours_x.size() == 0) return 0;
	for (int j = 0; j < contours_x.size(); j++)
	{
		double maera = cv::contourArea(contours_x[j]);
		if (maera > aera)
		{
			max_contour_index_x = j;
			aera = maera;
		}
	}
	largest_contour = contours_x[max_contour_index_x];
	return 1;
}

double OcrAlgorithm::iou_y(cv::Rect r1, cv::Rect r2)
{
	double inters = 0;
	double unites = 0;
	cv::Point pt1, pt2;
	pt1.x = r1.y;
	pt1.y = r1.y + r1.height;
	pt2.x = r2.y;
	pt2.y = r2.y + r2.height;
	if (pt1.x >= pt2.x && pt1.x <= pt2.y)
	{
		inters = min(pt1.y, pt2.y) - pt1.x;
		unites = max(pt1.y, pt2.y) - pt2.x;
		return inters / unites;
	}
	else if(pt2.x >= pt1.x && pt2.x <= pt1.y)
	{
		inters = min(pt1.y, pt2.y) - pt2.x;
		unites = max(pt1.y, pt2.y) - pt1.x;
		return inters / unites;
	}
	return 0;
}

bool OcrAlgorithm::isStanderPostcode(std::string srcstr, std::string &postcodestr)
{
	double score_ = postcodeStringScore(srcstr, postcodestr);
	if (score_ >= 0.5)
	{
		return true;
	}

	return false;
}

double OcrAlgorithm::postcodeStringScore(std::string srcStr,std::string &resultStr)
{
	//////////////////////////////////////////////////////////////////////////
//根据协议设定
	const int length_left = 5;
	const int length_right_4 = 4;
	const int length_right_5 = 5;
	if (srcStr.size() < length_left) return 0;

	double whole_score = 0;
	///去除空格
	for (int i = 0; i < srcStr.length();)//剔除空格
	{
		unsigned char c = srcStr.at(i);
		if (c == ' ' )
		{
			srcStr.erase(i, 1);
		}
		else
		{
			i++;
		}
	}


	//////////////////////////////////////////////////////////////////////////
//有“-”的情况 5+5,5+4的情况,
	size_t _pos = srcStr.find('-');
	if (_pos != srcStr.npos)
	{
		//判断-号位置，
		float div_pos_score = (fabs(_pos - srcStr.size() / 2.0)) / srcStr.size() * 2.0;//约往中间得分越低(0-1)
		//if (div_pos_score < 0.75)//-号在中间
		//{
		std::string substr1 = srcStr.substr(0, _pos);
		std::string substr2 = srcStr.substr(_pos + 1);

		double score_left = continuousDigitsScore(substr1, length_left);

		std::string resStr1, resStr2;
		getFirstContinuousDigits(substr1, length_left, resStr1);

		double score_right_5 = continuousDigitsScore(substr2, length_right_5);
		double score_right_4 = continuousDigitsScore(substr2, length_right_4);
		double score_right = max(score_right_5, score_right_4);
		int max_length_ = (score_right_5 < score_right_4) ? length_right_4 : length_right_5;
		getFirstContinuousDigits(substr2, max_length_, resStr2);
		resultStr = resStr1 + resStr2;
		whole_score = score_left * score_right;

		return whole_score;
		//}
	}

	//////////////////////////////////////////////////////////////////////////
	//没有找到正确的-号的情况
	//for (int i = 0; i < srcStr.length();)//剔除空格
	//{
	//	unsigned char c = srcStr.at(i);
	//	if (c == '-')
	//	{
	//		srcStr.erase(i, 1);
	//	}
	//	else
	//	{
	//		i++;
	//	}
	//}

	//是否考虑-被识别为空格的情况。。。待定
	double score_l_r5 = continuousDigitsScore(srcStr, length_left+ length_right_5);
	double score_l_r4 = continuousDigitsScore(srcStr, length_left + length_right_4);
	double score_r5 = continuousDigitsScore(srcStr, length_right_5);

	int max_i = 0;
	if (score_l_r4 > score_l_r5)
	{
		if (score_l_r4 > score_r5)
		{
			getFirstContinuousDigits(srcStr, length_left + length_right_4, resultStr);
			whole_score = score_l_r4;

		}
		else
		{
			getFirstContinuousDigits(srcStr, length_right_5, resultStr);
			whole_score = score_r5;
		}
	}
	else
	{
		if (score_l_r5 > score_r5)
		{
			getFirstContinuousDigits(srcStr, length_left + length_right_5, resultStr);
			whole_score = score_l_r5;

		}
		else
		{
			getFirstContinuousDigits(srcStr, length_right_5, resultStr);
			whole_score = score_r5;
		}
	}


	return whole_score;

}

int OcrAlgorithm::matchImage(cv::Mat srcImg, cv::Mat targetImg, std::vector<cv::Rect> rRects)
{
	//cvtColor(targetImg, targetImg, COLOR_BGR2GRAY);
	cv::Point minPoint;
	cv::Point maxPoint;
	double *minVal = 0;
	double *maxVal = 0;
	cv::Mat dstImg;
	//dstImg.create(srcImg.dims, srcImg.size, srcImg.type());
	int t_w = targetImg.cols;
	int t_h = targetImg.rows;
	//cv::imshow("createImg", dstImg);
	cv::Mat resizedTargetMat[3];//1, resizedTargetMat2;
	resizedTargetMat[0] = targetImg;
	float scal_1 = 0.85;
	float scal_2 = 1.15;
	cv::resize(targetImg, resizedTargetMat[1], cv::Size(t_w*scal_1, t_h*scal_1));
	cv::resize(targetImg, resizedTargetMat[2], cv::Size(t_w*scal_2, t_h*scal_2));

	std::vector<cv::Rect> tRects;

	//在3种尺度下进行匹配
	for (int i=0;i<3;i++)
	{
		//imshow("源文件", srcImg);
		//imshow("模板图片", resizedTargetMat[i]);
		//waitKey(0);
		cv::Mat tg = resizedTargetMat[i];
		int result_cols = srcImg.cols - tg.cols + 1;
		int result_rows = srcImg.rows - tg.rows + 1;
		dstImg.create(result_cols, result_rows, CV_32FC1);


		cv::matchTemplate(srcImg, tg, dstImg, 0);//重新编译后 运行一直出现异常，暂时不用

		cv::normalize(dstImg, dstImg, 0, 1, 32);
		cv::minMaxLoc(dstImg, minVal, maxVal, &minPoint, &maxPoint);
		cv::Rect mR = cv::Rect(minPoint, cv::Point(minPoint.x + resizedTargetMat[i].cols, minPoint.y + resizedTargetMat[i].rows));
		if ((mR.x + mR.width / 2) < srcImg.cols / 2 && (mR.y + mR.height / 2) < srcImg.rows / 2)
		{
			tRects.push_back(mR);
		}
	}

	//剔除重复的结果
	vector<cv::Rect>::iterator it;
	vector<cv::Rect>::iterator pre_it = tRects.begin();
	if (tRects.size()!=0)
	{
		for (pre_it = tRects.begin();pre_it!= tRects.end();pre_it++)
		{
			for (it = pre_it + 1; it != tRects.end();)
			{
				cv::Point cPoint;
				cPoint.x = it->x + it->width / 2;
				//cPoint.y = it->y + it->height / 2;

				if (cPoint.x > pre_it->x && cPoint.x < (pre_it->x + pre_it->width))
				{
					it = tRects.erase(it);
				}
				else
				{
					it++;
				}

			}
		}
	}
	

	//绘图
	for (int i=0;i<tRects.size();i++)
	{
		cv::rectangle(srcImg, tRects[i], cv::Scalar(0, 0, 0),3);
	}


	//cv::rectangle(srcImg, minPoint, cv::Point(minPoint.x + targetImg.cols, minPoint.y + targetImg.rows), cv::Scalar(0, 255, 0), 2, 8);
	cv::imshow("【匹配后的图像】", srcImg);
	//cv::rectangle(dstImg, minPoint, cv::Point(minPoint.x + targetImg.cols, minPoint.y + targetImg.rows), cv::Scalar(0, 0, 0), 3, 8);
	//cv::imshow("【匹配后的计算过程图像】", dstImg);

	if (tRects.size()!=0)
	{
		rRects.insert(rRects.end(), tRects.begin(), tRects.end());
	}
	
	return tRects.size();
}

double OcrAlgorithm::continuous5DigitScore(string srcStr)
{
	if (srcStr.empty())
	{
		return 0.0;
	}
	for (int i = 0; i < srcStr.length();)//剔除空格
	{
		unsigned char c = srcStr.at(i);
		if (c == ' '|| c == '-')
		{
			srcStr.erase(i, 1);
		}
		else
		{
			i++;
		}
	}

	int src_length = srcStr.length();

	//判断是否有连续五个数字
	double score_0 = 0;
	double score_1 = 0;
	for (size_t i = 0; i < srcStr.size(); i++)
	{
		unsigned char c = srcStr[i];
		if (isdigit(c))
		{
			for (size_t j=i+1;j<srcStr.size();j++)
			{
				unsigned char c1 = srcStr[j];
				if (isdigit(c1))
				{
					if (j-i>=4 && score_0 < 0.5 )//根据邮编位数设定
					{
						score_0 = 1.0;
						i = j + 1;
						break;
					}
					else if(j - i >= 4 && score_0 > 0.5)
					{
						score_1 = 1.0;
						i = j + 1;
						break;
					}
				}
				else
				{
					break;
				}
			}
		}
	}
	if (score_0 < 0.5)
	{
		return 0.0;
	}

	for (int i = 0; i < srcStr.length();)//剔除非数字符
	{
		unsigned char c = srcStr.at(i);
		if (!isdigit(c))
		{
			srcStr.erase(i, 1);
		}
		else
		{
			i++;
		}
	}
	double score_2 = 0;
	int n = srcStr.length();
	float score_length = float(n) / src_length;


	if (n<10)//根据邮编位数设定
	{
		score_2 = 1.0 - abs(n - 5) / 5.0;
		return score_2* score_length;
	}
	else if(n>=10)
	{
		if (score_1>=0.5)
		{
			score_2 = 1;
		}
		else if(score_1 < 0.5)
		{
			score_2 = 0.5;
		}
		score_2 = score_2 * (1 - (n - 10.0) / 20.0);
		return score_2* score_length;
	}
	return 0;

}

double OcrAlgorithm::continuousDigitsScore(std::string srcStr, int continus_num)
{
	if (continus_num <= 0) return 0.0;
	if (srcStr.empty()) return 0.0;

	for (int i = 0; i < srcStr.length();)//剔除空格
	{
		unsigned char c = srcStr.at(i);
		if (c == ' ' || c == '-')
		{
			srcStr.erase(i, 1);
		}
		else
		{
			i++;
		}
	}

	int src_length = srcStr.length();

	int m_continues_num = maxNumContinuousDigits(srcStr);

	if (m_continues_num < continus_num) return 0.0;
	
	///为连续数字的质量打分
	double conti_socre = 1.0 - (m_continues_num - continus_num) / double(continus_num);
	conti_socre = max(conti_socre, 0.0);


	///为字符串长度打分

	double length_socre = 1.0 - (src_length - continus_num) / (1.2*double(continus_num));//对分数的影响打折
	length_socre = max(length_socre, 0.0);

	//返回复合分数
	return conti_socre * length_socre;

}

int OcrAlgorithm::maxNumContinuousDigits(std::string srcStr)
{
	if (srcStr.empty()) return 0;
	int max_digits = 0;
	for (int i=0;i<srcStr.size();)
	{
		unsigned char c = srcStr.at(i);
		if (isdigit(c))
		{
			int digits_num =1;
			int j=i+1;
			for (;j<srcStr.size();j++)
			{
				unsigned char c = srcStr.at(j);
				if (isdigit(c))
				{
					digits_num++;
				}
				else
				{
					break;
				}
			}
			i = j+1;
			max_digits = (max_digits < digits_num) ? digits_num : max_digits;
		}
		else
		{
			i++;
		}
	}
	return max_digits;
}

double OcrAlgorithm::scoreForPostcodeString(std::string srcStr)
{











	return 0;

}

size_t OcrAlgorithm::getFirstContinuousDigits(std::string srcStr, int conti_num, std::string &dstStr)
{
	unsigned char c;
	for (int i = 0; i < srcStr.length();)
	{
		c = srcStr[i];
		if (isdigit(c))
		{
			int j = i + 1;
			for (;j<srcStr.size();j++)
			{
				c = srcStr[j];
				if (isdigit(c))
				{
					if (j-i+1 == conti_num)
					{
						dstStr = srcStr.substr(i, conti_num);
						return 1;
					}
				}
				else
				{
					break;
				}
			}
			i = j + 1;
		}
		else
		{
			i++;
		}

	}
	return 0;
}

int MatchDataStruct::loadMatchData(const std::string &xmlfile)
{
	cv::FileStorage fs(xmlfile, cv::FileStorage::READ);
	if (!fs.isOpened()) return 0;

	fs["descriptors2"] >> descriptors2;
	fs["descriptors3"] >> descriptors3;
	std::vector<KeyPoint>().swap(keypoints2);
	int ind = 0;
	while (true)
	{
		string s = std::to_string(ind);
		s = "keypoints2" + s;
		KeyPoint kpt;
		fs[s] >> kpt;
		keypoints2.push_back(kpt);
		if (kpt.octave == 0) break;
		ind++;
	}
	ind = 0;
	while (true)
	{
		string s = std::to_string(ind);
		s = "keypoints3" + s;
		KeyPoint kpt;
		fs[s] >> kpt;
		keypoints3.push_back(kpt);
		if (kpt.octave == 0) break;
		ind++;
	}
	fs.release();

	return 1;

}

int MatchDataStruct::saveMatchData(const std::string &xmlfile)
{
	cv::FileStorage fs(xmlfile, cv::FileStorage::WRITE);
	if (!fs.isOpened()) return 0;

	fs << "descriptors2" << descriptors2;
	fs << "descriptors3" << descriptors3;
	for (int i = 0; i < keypoints2.size(); i++)
	{
		string s = std::to_string(i);
		s = "keypoints2" + s;
		fs << s << keypoints2[i];
	}
	for (int i = 0; i < keypoints3.size(); i++)
	{
		string s = std::to_string(i);
		s = "keypoints3" + s;
		fs << s << keypoints3[i];
	}
	fs.release();

	return 1;
}

int MatchDataStruct::getMatchDataFromImg_tagRotate_SIFT(const std::string &refImg1, const std::string &refImg2)
{
	const int MAX_KEYPOINTS = 500;
	cv::Mat referenceMat1 = imread(refImg1);
	cv::Mat referenceMat2 = imread(refImg2);
	if (referenceMat1.empty()|| referenceMat2.empty())
	{
		return 0;
	}

	const cv::Size refSize(504, 293); //设置的图像宽高。

	cv::resize(referenceMat1, referenceMat1, refSize);
	cv::resize(referenceMat2, referenceMat2, refSize);

	cv::Mat im2Gray= referenceMat1;
	cv::Mat im3Gray = referenceMat2;

	if (referenceMat1.channels() == 3)
	{
		cv::cvtColor(referenceMat1, im2Gray, CV_BGR2GRAY);
	}
	if (referenceMat2.channels() == 3)
	{
		cv::cvtColor(referenceMat2, im3Gray, CV_BGR2GRAY);
	}

	// Detect ORB features and compute descriptors.
	cv::Ptr<Feature2D> sift1 = cv::xfeatures2d::SIFT::create(MAX_KEYPOINTS);

	sift1->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);
	sift1->detectAndCompute(im3Gray, Mat(), keypoints3, descriptors3);

	return 1;
}


int MatchDataStruct::getMatchDataFromImg_tagRotate_SURF(const std::string &refImg1, const std::string &refImg2)
{
	const int MAX_KEYPOINTS = 500;
	cv::Mat referenceMat1 = imread(refImg1);
	cv::Mat referenceMat2 = imread(refImg2);
	if (referenceMat1.empty() || referenceMat2.empty())
	{
		return 0;
	}

	const cv::Size refSize(504, 293); //设置的图像宽高。

	cv::resize(referenceMat1, referenceMat1, refSize);
	cv::resize(referenceMat2, referenceMat2, refSize);

	cv::Mat im2Gray = referenceMat1;
	cv::Mat im3Gray = referenceMat2;

	if (referenceMat1.channels() == 3)
	{
		cv::cvtColor(referenceMat1, im2Gray, CV_BGR2GRAY);
	}
	if (referenceMat2.channels() == 3)
	{
		cv::cvtColor(referenceMat2, im3Gray, CV_BGR2GRAY);
	}

	// Detect ORB features and compute descriptors.
	cv::Ptr<Feature2D> sift1 = cv::xfeatures2d::SURF::create(MAX_KEYPOINTS);

	sift1->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);
	sift1->detectAndCompute(im3Gray, Mat(), keypoints3, descriptors3);

	return 1;
}

int MatchDataStruct::getMatchDataFromImg_handwrite_addr(const std::string &refImg)
{
	const int MAX_KEYPOINTS = 512;
	cv::Mat referenceMat1 = imread(refImg);

	if (referenceMat1.empty() )
	{
		return 0;
	}

	//float scal = 300.0 / max(referenceMat1.cols, referenceMat1.rows);

	cv::resize(referenceMat1, referenceMat1, cv::Size(300,188), 0, 0,INTER_AREA);
	
	cv::Mat im2Gray = referenceMat1;

	if (referenceMat1.channels() == 3)
	{
		cv::cvtColor(referenceMat1, im2Gray, CV_BGR2GRAY);
	}


	// Detect ORB features and compute descriptors.
	cv::Ptr<Feature2D> sift1 = cv::xfeatures2d::SIFT::create(MAX_KEYPOINTS);

	sift1->detectAndCompute(im2Gray, Mat(), keypoints_handwrite_addr, descriptors_handwrite_addr);


	return 1;
}

FindHomography_2d::FindHomography_2d()
{
	this->modelPoints = 4;
	this->threshold = 12;
	this->confidence = 0.99;
	this->maxIters = 1000;
}
bool FindHomography_2d::haveCollinearPoints(const Mat& m, int count) const
{
	int j, k, i = count - 1;
	const Point2f* ptr = m.ptr<Point2f>();

	// check that the i-th selected point does not belong
	// to a line connecting some previously selected points
	// also checks that points are not too close to each other
	for (j = 0; j < i; j++)
	{
		double dx1 = ptr[j].x - ptr[i].x;
		double dy1 = ptr[j].y - ptr[i].y;
		for (k = 0; k < j; k++)
		{
			double dx2 = ptr[k].x - ptr[i].x;
			double dy2 = ptr[k].y - ptr[i].y;
			if (fabs(dx2*dy1 - dy2 * dx1) <= FLT_EPSILON * (fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2)))
				return true;
		}
	}
	return false;
}
bool FindHomography_2d::checkSubset(cv::Mat &ms1, cv::Mat &ms2, int count) const
{
	using namespace cv;
	//Mat ms1 = _ms1.getMat(), ms2 = _ms2.getMat();
	if (haveCollinearPoints(ms1, count) || haveCollinearPoints(ms2, count))
		return false;

	// We check whether the minimal set of points for the homography estimation
	// are geometrically consistent. We check if every 3 correspondences sets
	// fulfills the constraint.
	//
	// The usefullness of this constraint is explained in the paper:
	//
	// "Speeding-up homography estimation in mobile devices"
	// Journal of Real-Time Image Processing. 2013. DOI: 10.1007/s11554-012-0314-1
	// Pablo Marquez-Neila, Javier Lopez-Alberca, Jose M. Buenaposada, Luis Baumela
	if (count == modelPoints)
	{
		static const int tt[][3] = { {0, 1, 2}, {1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {2, 3, 4} };
		const Point2f* src = ms1.ptr<Point2f>();
		const Point2f* dst = ms2.ptr<Point2f>();
		int negative = 0;

		for (int i = 0; i < count; i++)
		{
			const int* t = tt[i];
			Matx33d A(src[t[0]].x, src[t[0]].y, 1., src[t[1]].x, src[t[1]].y, 1., src[t[2]].x, src[t[2]].y, 1.);
			Matx33d B(dst[t[0]].x, dst[t[0]].y, 1., dst[t[1]].x, dst[t[1]].y, 1., dst[t[2]].x, dst[t[2]].y, 1.);

			negative += determinant(A)*determinant(B) < 0;
		}
		if (negative != 0 && negative != count)
			return false;
	}

	return true;
}
bool FindHomography_2d::getSubset(const Mat& m1, const Mat& m2,
	Mat& ms1, Mat& ms2, RNG& rng,
	int maxAttempts) const
{
	cv::AutoBuffer<int> _idx(modelPoints);
	int* idx = _idx.data();
	int i = 0, j, k, iters = 0;
	int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
	int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
	int esz1 = (int)m1.elemSize1()*d1, esz2 = (int)m2.elemSize1()*d2;
	int count = m1.checkVector(d1), count2 = m2.checkVector(d2);
	const int *m1ptr = m1.ptr<int>(), *m2ptr = m2.ptr<int>();

	ms1.create(modelPoints, 1, CV_MAKETYPE(m1.depth(), d1));
	ms2.create(modelPoints, 1, CV_MAKETYPE(m2.depth(), d2));

	int *ms1ptr = ms1.ptr<int>(), *ms2ptr = ms2.ptr<int>();

	CV_Assert(count >= modelPoints && count == count2);
	CV_Assert((esz1 % sizeof(int)) == 0 && (esz2 % sizeof(int)) == 0);
	esz1 /= sizeof(int);
	esz2 /= sizeof(int);

	for (; iters < maxAttempts; iters++)
	{
		for (i = 0; i < modelPoints && iters < maxAttempts; )
		{
			int idx_i = 0;
			for (;;)
			{
				idx_i = idx[i] = rng.uniform(0, count);
				for (j = 0; j < i; j++)
					if (idx_i == idx[j])
						break;
				if (j == i)
					break;
			}
			for (k = 0; k < esz1; k++)
				ms1ptr[i*esz1 + k] = m1ptr[idx_i*esz1 + k];
			for (k = 0; k < esz2; k++)
				ms2ptr[i*esz2 + k] = m2ptr[idx_i*esz2 + k];
			i++;
		}
		if (i == modelPoints && !checkSubset(ms1, ms2, i))
			continue;
		break;
	}

	return i == modelPoints && iters < maxAttempts;
}
void FindHomography_2d::computeError(InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err) const
{

	Mat m1 = _m1.getMat(), m2 = _m2.getMat(), model = _model.getMat();
	//cout << model << endl;
	int i, count = m1.checkVector(2);
	const Point2f* M = m1.ptr<Point2f>();
	const Point2f* m = m2.ptr<Point2f>();

	float Hf[] = { model.at<float>(0,0), model.at<float>(0,1), model.at<float>(0,2),
		model.at<float>(1,0), model.at<float>(1,1), model.at<float>(1,2),
		model.at<float>(2,0), model.at<float>(2,1), model.at<float>(2,2),
	};

	_err.create(count, 1, CV_32F);
	float* err = _err.getMat().ptr<float>();

	for (i = 0; i < count; i++)
	{
		float dx = (Hf[0] * M[i].x + Hf[3] * M[i].y + Hf[6]) - m[i].x;
		float dy = (Hf[1] * M[i].x + Hf[4] * M[i].y + Hf[7]) - m[i].y;
		err[i] = dx * dx + dy * dy;
	}
	//cout << _err.getMat() << endl;
}
int FindHomography_2d::runKernel_(InputArray _m1, InputArray _m2, OutputArray _model) const
{
	Mat m1 = _m1.getMat(), m2 = _m2.getMat();
	int i, count = m1.checkVector(2);
	const Point2f* M = m1.ptr<Point2f>();
	const Point2f* m = m2.ptr<Point2f>();


	Point2f cM(0, 0), cm(0, 0), sM(0, 0), sm(0, 0);//平移，缩放
	Point2f dM(0, 0), dm(0, 0);
	for (i = 0; i < count; i++)
	{
		cm.x += m[i].x; cm.y += m[i].y;
		cM.x += M[i].x; cM.y += M[i].y;
	}

	cm.x /= count;//平移
	cm.y /= count;
	cM.x /= count;
	cM.y /= count;

	//主方向向量，用于旋转
	//复杂版
	for (i = 0; i < count; i++)
	{
		dm.x += (m[i].x - cm.x) / sqrt(pow(m[i].x - cm.x, 2) + pow(m[i].y - cm.y, 2));
		dm.y += (m[i].y - cm.y) / sqrt(pow(m[i].x - cm.x, 2) + pow(m[i].y - cm.y, 2));
		dM.x += (M[i].x - cM.x) / sqrt(pow(M[i].x - cM.x, 2) + pow(M[i].y - cM.y, 2));
		dM.y += (M[i].y - cM.y) / sqrt(pow(M[i].x - cM.x, 2) + pow(M[i].y - cM.y, 2));
	}
	//精简版
	//dm.x = m[0].x - cm.x;
	//dm.y = m[0].y - cm.y;
	//dM.x = M[0].x - cM.x;
	//dM.y = M[0].y - cM.y;
	if (fabs(dm.x) < FLT_EPSILON || fabs(dm.y) < FLT_EPSILON ||
		fabs(dM.x) < FLT_EPSILON || fabs(dM.y) < FLT_EPSILON)
		return 0;

	float absm_2 = pow(dm.x, 2) + pow(dm.y, 2);
	float absM_2 = pow(dM.x, 2) + pow(dM.y, 2);
	float sqrt_abs2 = sqrt(absM_2*absm_2);
	float mutMm = dm.x*dM.x + dm.y*dM.y;//点乘
	float cos_angle = (mutMm) / sqrt_abs2;
	float mut_cross_u = dm.x*dM.y - dm.y*dM.x;
	float sin_angle = mut_cross_u / sqrt_abs2;//×乘

	float rotate_metrix_data[9] = { cos_angle,-sin_angle,0,   sin_angle,cos_angle,0,   0,0,1 };
	float translate_metrix_data_m[9] = { 1,0,0,   0,1,0,   cm.x,cm.y,1 };
	float translate_metrix_data_M[9] = { 1,0,0,   0,1,0,   -cM.x,-cM.y,1 };
	cv::Mat rotate_metrix(3, 3, CV_32F, rotate_metrix_data);
	cv::Mat translate_metrix_m(3, 3, CV_32F, translate_metrix_data_m);
	cv::Mat translate_metrix_M(3, 3, CV_32F, translate_metrix_data_M);

	//平移到原点，旋转
	cv::Mat model_mat(3, 3, CV_32F);
	model_mat = translate_metrix_M * rotate_metrix;//先平移和旋转
	Mat m1_r(count,3,CV_32F);
	for (i = 0; i < count; i++)
	{
		m1_r.at<float>(i, 0) = M[i].x;
		m1_r.at<float>(i, 1) = M[i].y;
		m1_r.at<float>(i, 2) = 1.0f;
	}
	m1_r = m1_r * model_mat;

	//确定缩放系数
	for (i = 0; i < count; i++)
	{
		sm.x += fabs(m[i].x - cm.x);
		sm.y += fabs(m[i].y - cm.y);

		sM.x += fabs(m1_r.at<float>(i, 0));
		sM.y += fabs(m1_r.at<float>(i, 1));
	}

	//缩放

	if (fabs(sm.x) < FLT_EPSILON || fabs(sm.y) < FLT_EPSILON ||
		fabs(sM.x) < FLT_EPSILON || fabs(sM.y) < FLT_EPSILON)
		return 0;
	cv::Point2f scale_m;
	scale_m.x = (sm.x / sM.x+ sm.y / sM.y)/2.0;//xy坐标系缩放系数一致
	scale_m.y = scale_m.x;
	float scale_metrix_data[9] = { scale_m.x,0,0,   0,scale_m.y,0,     0,0,1 };
	cv::Mat scale_metrix(3, 3, CV_32F, scale_metrix_data);
	



	model_mat = translate_metrix_M * rotate_metrix * scale_metrix * translate_metrix_m;
	//std::cout << model_mat << std::endl;

	model_mat.convertTo(_model, model_mat.type());

	return 1;
}

int FindHomography_2d::runKernel_NoZoom(cv::InputArray _m1, cv::InputArray _m2, cv::OutputArray _model,cv::Point2f &whole_scale) const
{
	Mat m1 = _m1.getMat(), m2 = _m2.getMat();
	int i, count = m1.checkVector(2);
	const Point2f* M = m1.ptr<Point2f>();
	const Point2f* m = m2.ptr<Point2f>();


	Point2f cM(0, 0), cm(0, 0), sM(0, 0), sm(0, 0);//平移，缩放
	Point2f dM(0, 0), dm(0, 0);
	for (i = 0; i < count; i++)
	{
		cm.x += m[i].x; cm.y += m[i].y;
		cM.x += M[i].x; cM.y += M[i].y;
	}

	cm.x /= count;//平移
	cm.y /= count;
	cM.x /= count;
	cM.y /= count;

	//主方向向量，用于旋转
	//复杂版
	for (i = 0; i < count; i++)
	{
		dm.x += (m[i].x - cm.x) / sqrt(pow(m[i].x - cm.x, 2) + pow(m[i].y - cm.y, 2));
		dm.y += (m[i].y - cm.y) / sqrt(pow(m[i].x - cm.x, 2) + pow(m[i].y - cm.y, 2));
		dM.x += (M[i].x - cM.x) / sqrt(pow(M[i].x - cM.x, 2) + pow(M[i].y - cM.y, 2));
		dM.y += (M[i].y - cM.y) / sqrt(pow(M[i].x - cM.x, 2) + pow(M[i].y - cM.y, 2));
	}
	//精简版
	//dm.x = m[0].x - cm.x;
	//dm.y = m[0].y - cm.y;
	//dM.x = M[0].x - cM.x;
	//dM.y = M[0].y - cM.y;
	if (fabs(dm.x) < FLT_EPSILON || fabs(dm.y) < FLT_EPSILON ||
		fabs(dM.x) < FLT_EPSILON || fabs(dM.y) < FLT_EPSILON)
		return 0;

	float absm_2 = pow(dm.x, 2) + pow(dm.y, 2);
	float absM_2 = pow(dM.x, 2) + pow(dM.y, 2);
	float sqrt_abs2 = sqrt(absM_2*absm_2);
	float mutMm = dm.x*dM.x + dm.y*dM.y;//点乘
	float cos_angle = (mutMm) / sqrt_abs2;
	float mut_cross_u = dm.x*dM.y - dm.y*dM.x;
	float sin_angle = mut_cross_u / sqrt_abs2;//×乘

	float rotate_metrix_data[9] = { cos_angle,-sin_angle,0,   sin_angle,cos_angle,0,   0,0,1 };
	float translate_metrix_data_m[9] = { 1,0,0,   0,1,0,   cm.x,cm.y,1 };
	float translate_metrix_data_M[9] = { 1,0,0,   0,1,0,   -cM.x,-cM.y,1 };
	cv::Mat rotate_metrix(3, 3, CV_32F, rotate_metrix_data);
	cv::Mat translate_metrix_m(3, 3, CV_32F, translate_metrix_data_m);
	cv::Mat translate_metrix_M(3, 3, CV_32F, translate_metrix_data_M);

	//平移到原点，旋转
	cv::Mat model_mat(3, 3, CV_32F);
	model_mat = translate_metrix_M * rotate_metrix;//先平移和旋转
	Mat m1_r(count, 3, CV_32F);
	for (i = 0; i < count; i++)
	{
		m1_r.at<float>(i, 0) = M[i].x;
		m1_r.at<float>(i, 1) = M[i].y;
		m1_r.at<float>(i, 2) = 1.0f;
	}
	m1_r = m1_r * model_mat;

	//确定缩放系数
	for (i = 0; i < count; i++)
	{
		sm.x += fabs(m[i].x - cm.x);
		sm.y += fabs(m[i].y - cm.y);

		sM.x += fabs(m1_r.at<float>(i, 0));
		sM.y += fabs(m1_r.at<float>(i, 1));
	}

	//缩放
	if (fabs(sm.x) < FLT_EPSILON || fabs(sm.y) < FLT_EPSILON ||
		fabs(sM.x) < FLT_EPSILON || fabs(sM.y) < FLT_EPSILON)
		return 0;
	cv::Point2f scale_m;
	scale_m.x = (sm.x / sM.x + sm.y / sM.y) / 2.0f;//xy缩放限制为一致
	scale_m.y = scale_m.x;
	float scale_metrix_data[9] = { scale_m.x,0,0,   0,scale_m.y,0,     0,0,1 };
	cv::Mat scale_metrix(3, 3, CV_32F, scale_metrix_data);


	translate_metrix_data_m[6] = translate_metrix_data_m[6] / whole_scale.x;
	translate_metrix_data_m[7] = translate_metrix_data_m[7] / whole_scale.y;
	translate_metrix_data_M[6] = translate_metrix_data_M[6] / whole_scale.x;
	translate_metrix_data_M[7] = translate_metrix_data_M[7] / whole_scale.y;


	whole_scale.x = 1.0f / whole_scale.x;
	whole_scale.y = 1.0f / whole_scale.y;

	model_mat = translate_metrix_M * rotate_metrix *scale_metrix* translate_metrix_m;
	//std::cout << model_mat << std::endl;

	model_mat.convertTo(_model, model_mat.type());

	return 1;





}

int FindHomography_2d::RANSACUpdateNumIters(double p, double ep, int modelPoints, int maxIters)
{
	if (modelPoints <= 0)
		CV_Error(Error::StsOutOfRange, "the number of model points should be positive");

	p = MAX(p, 0.);
	p = MIN(p, 1.);
	ep = MAX(ep, 0.);
	ep = MIN(ep, 1.);

	// avoid inf's & nan's
	double num = MAX(1. - p, DBL_MIN);
	double denom = 1. - std::pow(1. - ep, modelPoints);
	if (denom < DBL_MIN)
		return 0;

	num = std::log(num);
	denom = std::log(denom);

	return denom >= 0 || -num >= maxIters * (-denom) ? maxIters : cvRound(num / denom);
}
int FindHomography_2d::findInliers(const Mat& m1, const Mat& m2, const Mat& model, Mat& err, Mat& mask, double thresh) const
{
	this->computeError(m1, m2, model, err);
	mask.create(err.size(), CV_8U);

	CV_Assert(err.isContinuous() && err.type() == CV_32F && mask.isContinuous() && mask.type() == CV_8U);
	const float* errptr = err.ptr<float>();
	uchar* maskptr = mask.ptr<uchar>();
	float t = (float)(thresh*thresh);
	int i, n = (int)err.total(), nz = 0;
	for (i = 0; i < n; i++)
	{
		int f = errptr[i] <= t;
		//cout << errptr[i] << "; ";
		maskptr[i] = (uchar)f;
		nz += f;
	}
	//cout << endl;
	return nz;
}




bool FindHomography_2d::run(InputArray _m1, InputArray _m2, OutputArray _model, OutputArray _mask)
{
	bool result = false;
	Mat m1 = _m1.getMat(), m2 = _m2.getMat();
	Mat err, mask, model, bestModel, ms1, ms2;

	int iter, niters = MAX(maxIters, 1);
	int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
	int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
	int count = m1.checkVector(d1), count2 = m2.checkVector(d2), maxGoodCount = 0;

	RNG rng((uint64)-1);

	assert(confidence > 0 && confidence < 1);

	assert(count >= 0 && count2 == count);
	if (count < modelPoints)
		return false;

	Mat bestMask0, bestMask;

	if (_mask.needed())
	{
		_mask.create(count, 1, CV_8U, -1, true);
		bestMask0 = bestMask = _mask.getMat();
		CV_Assert((bestMask.cols == 1 || bestMask.rows == 1) && (int)bestMask.total() == count);
	}
	else
	{
		bestMask.create(count, 1, CV_8U);
		bestMask0 = bestMask;
	}

	if (count == modelPoints)
	{
		if (runKernel_(m1, m2, bestModel) <= 0)
			return false;
		bestModel.copyTo(_model);
		bestMask.setTo(Scalar::all(1));
		return true;
	}

	for (iter = 0; iter < niters; iter++)
	{
		int i, nmodels;
		if (count > modelPoints)
		{
			bool found = getSubset(m1, m2, ms1, ms2, rng, 10000);
			if (!found)
			{
				if (iter == 0)
					return false;
				break;
			}
		}
		//cout << ms1 << endl;
		//cout << ms2 << endl;
		nmodels = runKernel_(ms1, ms2, model);
		//cout << model << endl;

		if (nmodels <= 0)
			continue;

		//Mat _err;
		//computeError(ms1, ms2, model, _err);
		//cout << _err << endl;
		CV_Assert(model.rows % nmodels == 0);
		Size modelSize(model.cols, model.rows / nmodels);


		//Mat model_i = model.rowRange(i*modelSize.height, (i + 1)*modelSize.height);
		int goodCount = findInliers(m1, m2, model, err, mask, threshold);

		if (goodCount > MAX(maxGoodCount, modelPoints - 1))
		{
			std::swap(mask, bestMask);
			model.copyTo(bestModel);
			maxGoodCount = goodCount;
			niters = RANSACUpdateNumIters(confidence, (double)(count - goodCount) / count, modelPoints, niters);
		}

	}
	cout << "获得的最大点数:" << maxGoodCount << endl;
	if (maxGoodCount > 0)
	{
		if (bestMask.data != bestMask0.data)
		{
			if (bestMask.size() == bestMask0.size())
				bestMask.copyTo(bestMask0);
			else
				transpose(bestMask, bestMask0);
		}
		bestModel.copyTo(_model);
		result = true;
	}
	else
		_model.release();

	return result;
}

HWDigitsOCR::HWDigitsOCR()
{

}

int HWDigitsOCR::getPostCode2String(std::string srcImgPath, std::string &postcode, OcrAlgorithm_config* pConfig)
{
	cv::Mat srcImg = cv::imread(srcImgPath);
	return getPostCode2String(srcImg, postcode, pConfig);
}

int HWDigitsOCR::getPostCode2String(cv::Mat srcImg, std::string &postcode, OcrAlgorithm_config* pConfig)
{
	if (srcImg.empty())
		return 0;
	float confidence = pConfig->HandwriteDigitsConfidence;
#ifdef POSCODE_BOX_DEBUG
	float scal = 1000.0 / max(srcImg.rows, srcImg.cols);
	cv::Mat reszMat;
	cv::resize(srcImg, reszMat, cv::Size(), scal, scal);
	imshow("PostcodeBox_srcImg", reszMat);
#endif // POSCODE_BOX_DEBUG

	std::vector<cv::Mat> AddrRangeMat_vec;
	//clock_t t0 = clock();
	int res = getHandWriteAddressRangeMat(srcImg, pConfig, AddrRangeMat_vec);
	//clock_t t1 = clock();
	//double timeconsume = double(t1 - t0) / CLOCKS_PER_SEC;
	//cout << "获取手写地址框消耗时间：" << timeconsume << endl;
	if (res == 0)
		return 0;
	cv::Mat refMat, postcodeLineMat;
	int postcode_type_ = 0;//1:from,2:To
	std::string post_from, post_to;
	int post_code_num = 0;
	tesseract::TessBaseAPI*pTess = (tesseract::TessBaseAPI*)(pConfig->pTess);
	for (int i = 0; i < AddrRangeMat_vec.size(); i++)
	{
		cv::Mat mt;
#ifdef POSTCODE_BOX_DEBUG
		imshow("addr_range"+ std::to_string(i), AddrRangeMat_vec[i]);
#endif // POSTCODE_BOX_DEBUG

		res = rotateImage(AddrRangeMat_vec[i], mt);
		if (res == 0)
		{
			std::cout << "旋转图像失败" << std::endl;
			continue;
		}
#ifdef POSTCODE_BOX_DEBUG
		imshow("addr_range_rotate" + std::to_string(i), mt);
#endif // POSTCODE_BOX_DEBUG

		res = getPostcodeLine(mt, refMat, postcodeLineMat);
		if (res == 0)
		{
			std::cout << "获取邮编行失败" << std::endl;
			continue;
		}
#ifdef POSTCODE_BOX_DEBUG
		cv::imshow("postcode_line" + std::to_string(i), postcodeLineMat);
		cv::imshow("postcode_type" + std::to_string(i), refMat);
#endif // POSTCODE_BOX_DEBUG

		res = getPostcodeType(refMat, pTess);
		if (res == 0)
		{
			std::cout << "判断邮编类型失败" << std::endl;
			//waitKey(0);
			continue;
		}
		postcode_type_ = res;
		std::vector<cv::Mat> boxMats;
		res = segPostCode(postcodeLineMat, boxMats);

		if (res == 0)
		{
			std::cout << "拆分邮编格失败" << std::endl;
			//waitKey(0);
			continue;
		}
		std::string pcd_str;
		float confidence_v = 0;
		res = getPostCodeFromBoxMats(boxMats, pcd_str, confidence_v, pConfig);
		if (res == 0)
		{
			std::cout << "运行OCR失败" << std::endl;
			continue;
		}
		if (confidence_v < confidence)
		{
			std::cout << "置信度过低-放弃" << std::endl;
			continue;
		}
		if (postcode_type_ == 1)
		{
			post_from = pcd_str;
		}
		if (postcode_type_ == 2)
		{
			post_to = pcd_str;
		}
#ifdef POSTCODE_BOX_DEBUG
		cv::waitKey(0);
#endif // POSTCODE_BOX_DEBUG
	}
	if (post_to.empty())
	{
		return 0;
	}
	if (post_from.empty())
	{
		postcode = post_to;
		return 1;
	}
	else
	{
		postcode = post_from + "-" + post_to;
		return 2;
	}

	return 1;
}



int HWDigitsOCR::getPostCode2String_test(cv::Mat srcMat, std::string &postcode, OcrAlgorithm_config* pConfig)
{
	if (srcMat.empty())
	{
		return 0;
	}
	//Mat fromMat, toMat;
	if (srcMat.rows > srcMat.cols) cv::rotate(srcMat, srcMat, ROTATE_90_CLOCKWISE);//旋转图片
	vector<Mat> toMats, fromMats;
	int res = getPostCodeLine_nobox(srcMat, toMats, fromMats);
	if (res == 0)
	{
		return 0;
	}
#ifdef POSTCODE_BOX_DEBUG
	for (int i=0;i<fromMats.size();i++)
	{
		imshow("FromMat" + to_string(i), fromMats[i]);
	}
	for (int i = 0; i < toMats.size(); i++)
	{
		imshow("ToMat" + to_string(i), toMats[i]);
	}

#endif // POSTCODE_BOX_DEBUG

	vector<Mat> to_digits_vec;
	for (int i=0;i<toMats.size();i++)
	{
		res = split_digits_nobox(toMats[i], to_digits_vec);
		if (res ==0 )
		{
			continue;
		}
	}
	vector<Mat> from_digits_vec;
	for (int i = 0; i < fromMats.size(); i++)
	{
		res = split_digits_nobox(fromMats[i], from_digits_vec);
		if (res == 0)
		{
			continue;
		}
	}
	if (to_digits_vec.empty())
	{
		cout << "未找到目的地邮编！" << endl;
		return 0;
	}
	vector<string> to_code_vec;
	vector<float> to_confidence_vec;
	vector<string> from_code_vec;
	vector<float> from_confidence_vec;
	getPostCode_nobox(to_digits_vec, to_code_vec, to_confidence_vec, pConfig);

	float confidence = pConfig->HandwriteDigitsConfidence;
	int code_index = -1;
	float tem_confidence = 0;
	for (int i=0;i<to_confidence_vec.size();i++)
	{
		if (tem_confidence < to_confidence_vec[i] && to_confidence_vec[i] >= confidence)
		{
			code_index = i;
			tem_confidence = to_confidence_vec[i];
		}
	}
	string to_postcode, from_postcode;
	if (code_index!=-1)
	{
		to_postcode = to_code_vec[code_index];
	}
	else
	{
		cout << "目的地邮编置信度较低" << endl;
		return 0;
	}


	if (!from_digits_vec.empty())
	{
		getPostCode_nobox(from_digits_vec, from_code_vec, from_confidence_vec, pConfig);
	}
	tem_confidence = 0;
	code_index = -1;
	for (int i = 0; i < from_confidence_vec.size(); i++)
	{
		if (tem_confidence < from_confidence_vec[i] && from_confidence_vec[i] >= confidence)
		{
			code_index = i;
			tem_confidence = from_confidence_vec[i];
		}
	}
	if (code_index != -1)
	{
		from_postcode = from_code_vec[code_index];
	}


	if (from_postcode.empty())
	{
		postcode = to_postcode;
		return 1;
	}
	else
	{
		postcode = from_postcode + "-" + to_postcode;
		return 2;
	}
	
	
}

int HWDigitsOCR::rotateImage(const cv::Mat &srcMat, cv::Mat &dstMat)
{
	using namespace cv;
	using namespace xfeatures2d;
	const int MAX_FEATURES = 500;

	float scal_max2 = 720.0 / max(srcMat.cols, srcMat.rows);
	float scal_min2 = 450.0 / min(srcMat.cols, srcMat.rows);

	float scal1 = max(scal_max2, scal_min2);

	Mat im1;
	cv::resize(srcMat, im1, cv::Size(), scal1, scal1, cv::INTER_AREA);
	//imshow("resized", im1);

	//二值化

	cv::Mat srcMat_g = im1;
	if (im1.channels() == 3)
	{
		cv::cvtColor(im1, srcMat_g, cv::COLOR_BGR2GRAY);

	}


	//if (refMat.channels() == 3)
	//{
	//	cv::cvtColor(refMat, refMat_g, cv::COLOR_BGR2GRAY);
	//}

	//cv::threshold(srcMat_g, srcMat_g, 40, 255, cv::THRESH_TRIANGLE);
	//cv::threshold(refMat_g, refMat_g, 40, 255, cv::THRESH_TRIANGLE);
	//Mat edge_mat;
	//cv::Canny(srcMat_g, edge_mat, 50, 150);
	//cv::Canny(refMat_g, refMat_g, 10, 150);
	//imshow("canny_src", edge_mat);
	//imshow("canny_ref", refMat_g);
	//cv::LineSegmentDetector

	//Hough直线检测API	
	//vector<Vec4i>lines;//定义一个存放直线信息的向量						
	//int threshold_points = 50;
	//HoughLinesP(edge_mat, lines, 1, CV_PI / 180, threshold_points, 200, 15);


	//LSD线段检测
	Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_ADV);//LSD_REFINE_ADV//LSD_REFINE_NONE
	vector<Vec4f> lines_std;
	// Detect the lines
	ls->detect(srcMat_g, lines_std);//这里把检测到的直线线段都存入了lines_std中，4个float的值，分别为起止点的坐标
	
	//去除短线
	vector<Vec4f>::iterator it = lines_std.begin();
	for (;it!= lines_std.end();)
	{
		//Vec4f pts = *it;
		int len_x = abs((*it)[2] - (*it)[0]);
		int len_y = abs((*it)[3] - (*it)[1]);
		int distance_ = len_y + len_x - min(len_x, len_y) / 2;//快速计算距离，精度不高
		if (distance_ < 100)
		{
			it = lines_std.erase(it);
		}
		else
		{
			it++;
		}
	}


#ifdef POSTCODE_BOX_DEBUG
	for (size_t i = 0; i < lines_std.size(); i++)
	{
		cv::line(srcMat_g, Point(lines_std[i][0], lines_std[i][1]),
			Point(lines_std[i][2], lines_std[i][3]), Scalar(255, 255, 255), 2, 8);
	}
	imshow("lines", srcMat_g);
#endif // POSTCODE_BOX_DEBUG



	//计算角度
	it = lines_std.begin();
	vector<float>lines_angle;
	for (; it != lines_std.end();it++)
	{
		//Vec4f pts = *it;
		float len_x = (*it)[2] - (*it)[0];
		float len_y = (*it)[3] - (*it)[1];
		//len_x = (len_x < 0.001) ? 0.001 : len_x;
		float a = atan(len_y / len_x);
		lines_angle.push_back(a);
	}

	//std::sort(lines_angle.begin(), lines_angle.end());
	//计算角度方向线密度
	float delta_angle = 1.0 / 180.0*CV_PI;
	int max_num = 0;
	float best_angle = 0;
	for (float i=-CV_PI/2.0;i< CV_PI / 2.0;i += delta_angle)
	{
		int q_num = query_near_count(lines_angle, i - delta_angle / 2, i + delta_angle / 2);
		if (max_num<q_num)
		{
			max_num = q_num;
			best_angle = i;
		}
	}


	cv::Mat src_m = srcMat;
	ImageProcessFunc::rotate_arbitrarily_angle(src_m, dstMat, best_angle);

	
	return 1;
}

int HWDigitsOCR::getPostcodeLine(const cv::Mat &srcMat, cv::Mat &refMat, cv::Mat &dstMat)
{
	//二值化

	float scal_max2 = 720.0 / max(srcMat.cols, srcMat.rows);
	float scal_min2 = 500.0 / min(srcMat.cols, srcMat.rows);

	float scal1 = max(scal_max2, scal_min2);

	Mat im1_g,im1;
	cv::resize(srcMat, im1, cv::Size(), scal1, scal1, cv::INTER_AREA);
	im1_g = im1;
	if (im1.channels()==3)
	{
		cvtColor(im1, im1_g, COLOR_BGR2GRAY);
	}


	Mat bnrMat;
	cv::threshold(im1_g, bnrMat, 40, 255, CV_THRESH_BINARY);

	int wd = bnrMat.cols;
	int ht = bnrMat.rows;

	Mat morp_mat;

	cv::Mat element = getStructuringElement(MORPH_RECT, Size(5, 1));
	morphologyEx(bnrMat, morp_mat, MORPH_OPEN, element);
	element = getStructuringElement(MORPH_RECT, Size(1, 5));
	morphologyEx(morp_mat, morp_mat, MORPH_OPEN, element);
	



	//Mat edgeMat;
	//cv::Canny(bnrMat, edgeMat, 50, 150);
	//imshow("edgemat", edgeMat);


	//cv::line(bnrMat, cv::Point(0, 0), cv::Point(wd - 1, 0), cv::Scalar(0,0,0),4);
	//cv::line(bnrMat, cv::Point(0, 0), cv::Point(0, ht - 1), cv::Scalar(0, 0, 0),4);
	//cv::line(bnrMat, cv::Point(wd - 1, ht - 1), cv::Point(wd - 1, 0), cv::Scalar(0, 0, 0),4);
	//cv::line(bnrMat, cv::Point(wd - 1, ht - 1), cv::Point(0, ht - 1), cv::Scalar(0, 0, 0),4);


//#ifdef POSTCODE_BOX_DEBUG
//	imshow("threshold2", morp_mat);
//#endif // _DEBUG

	std::vector<std::vector<cv::Point>>contours;
	std::vector<cv::Vec4i>hierarchy;
	std::vector<cv::Point> contour;
	double aera = 0;
	//src_gray = src_gray > 100;
	cv::findContours(morp_mat, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

	//绘制轮廓图
	//cv::Mat dstImage = cv::Mat::zeros(rsd_img.size(), CV_8UC3);
	std::vector<double> vec_eare;
	cv::Rect m_Rect;
	cv::Rect best_Rect;
	std::vector<cv::Point> best_contour;
	bool isgoodRect = false;
	int index_hierc = -1;
	int index_contour = -1;

	std::vector<std::vector<cv::Point>> contours_candidate;//候选contour
	std::vector<double> aera_countours;
	//Scalar color = Scalar(rand() % 255, rand() % 255, rand() % 255);
	//drawContours(dstImage, contours, i, color, CV_FILLED, 8, hierarchy);
	int image_aera = wd * ht;
	for (int j = 0; j < contours.size(); j++)
	{
		double maera = cv::contourArea(contours[j]);
		if (maera > image_aera/3)
		{
			contours_candidate.push_back(contours[j]);
			aera_countours.push_back(maera);
			//best_Rect = m_Rect;
			//index_contour = j;
			//isgoodRect = true;
			//aera = maera;
		}
	}
	if (contours_candidate.size() == 0)
	{
		std::cout << "没有找到候选手写框" << endl;
		return 0;
	}

#ifdef POSTCODE_BOX_DEBUG
	//for (int i=0;i<contours_candidate.size();i++)
	//{
	//	Scalar color(rand() & 255, rand() & 255, rand() & 255);
	//	drawContours(im1, contours_candidate, i, color, 4, 8);
	//}
	//imshow("CONTOURS", im1);
#endif // POSTCODE_BOX_DEBUG
	
	if (contours_candidate.size()>1)
	{
		double aera = wd*ht;
		int min_aera_i = -1;
		for (int i=0;i<contours_candidate.size();i++)
		{
			if (aera > aera_countours[i])
			{
				aera = aera_countours[i];
				min_aera_i = i;
			}
		}
		best_contour = contours_candidate[min_aera_i];
	}
	else
	{
		best_contour = contours_candidate[0];
	}

#ifdef POSTCODE_BOX_DEBUG
	std::vector<std::vector<cv::Point>> draw_contour;
	draw_contour.push_back(best_contour);
	drawContours(im1, draw_contour, 0, Scalar(255,255,255), 4, 8);
	imshow("best_countour", im1);
#endif // POSTCODE_BOX_DEBUG

	cv::Rect outer_rect;
	CutParcelBox::findRect(best_contour, outer_rect);
	if (outer_rect.x > wd/2 || (outer_rect.x+outer_rect.width)<wd/2||
		outer_rect.y > ht / 2 || (outer_rect.y + outer_rect.height) < ht / 2)
	{
		return 0;
	}
	if (outer_rect.width<100 ||outer_rect.height<100)
	{
		return 0;
	}

	//检查是否变形
	int left_up = outer_rect.x + outer_rect.width;
	int left_bottom = outer_rect.x + outer_rect.width;
	int right_up = outer_rect.x;
	int right_bottom = outer_rect.x;
	std::vector<cv::Point>::iterator it;

	for (it=best_contour.begin();it!= best_contour.end();it++)
	{
		if (it->y < outer_rect.y + 20 && left_up > it->x) left_up = it->x;
		if (it->y < outer_rect.y + 20 && right_up < it->x) right_up = it->x;
		if (it->y > outer_rect.y + outer_rect.height - 20 && left_bottom > it->x) left_bottom = it->x;
		if (it->y > outer_rect.y + outer_rect.height - 20 && right_bottom < it->x) right_bottom = it->x;
	}
	float shift_pixes = (right_up + left_up) / 2.0 - (right_bottom + left_bottom) / 2.0;
	//waitKey(0);
	cv::Mat script_range_mat;
	im1_g(outer_rect).copyTo(script_range_mat);//复制原图

	//设置变形恢复mat
	cv::Point2f srcPts[3];
	cv::Point2f dstPts[3];
	srcPts[0] = cv::Point2f(0, 0);
	dstPts[0] = cv::Point2f(0, 0);

	srcPts[1] = cv::Point2f(script_range_mat.cols-1, 0);
	dstPts[1] = cv::Point2f(script_range_mat.cols - 1, 0);

	srcPts[2] = cv::Point2f(script_range_mat.cols - 1, script_range_mat.rows-1);
	dstPts[2] = cv::Point2f(script_range_mat.cols + shift_pixes, script_range_mat.rows - 1);
	cv::Mat Correct_range_mat = script_range_mat;

	//判断变形是否合理
	if ((shift_pixes > -100) && (shift_pixes < 100))
	{
		Mat M1 = cv::getAffineTransform(srcPts, dstPts);
		warpAffine(script_range_mat, Correct_range_mat, M1, script_range_mat.size());
	}


//#ifdef POSTCODE_BOX_DEBUG
//	imshow("script_range_mat", script_range_mat);
//	imshow("Correct_range_mat", Correct_range_mat);
//#endif // POSTCODE_BOX_DEBUG


	//裁剪边框
	int border_width = 10;
	int border_height = 5;
	cv::Rect borderRect = cv::Rect(border_width, border_height,
		script_range_mat.cols - border_width * 2, script_range_mat.rows - border_height * 2);
	Correct_range_mat = Correct_range_mat(borderRect);



	//确定手写框坐标
	cv::Rect post_code_rect;
	post_code_rect.x = Correct_range_mat.cols*0.3;
	post_code_rect.y = Correct_range_mat.rows*0.85;
	post_code_rect.width = Correct_range_mat.cols*0.9 - post_code_rect.x;
	post_code_rect.height = Correct_range_mat.rows - post_code_rect.y;

	//确定手写框水平位置
	//cv::Mat post_code_mat = script_range_mat(post_code_rect);
	//post_code_mat = ~post_code_mat;
	Correct_range_mat(post_code_rect).copyTo(dstMat);




	//确定参考区域位置
	cv::Rect refRect;
	refRect.width = Correct_range_mat.cols * 0.35;
	refRect.height = Correct_range_mat.rows *0.15;

	if (refRect.width == 0 || refRect.height == 0) 
	{
		return 0;
	}


	 Correct_range_mat(refRect).copyTo(refMat);



//#ifdef POSTCODE_BOX_DEBUG
//	imshow("ref_mat", refMat);
//	imshow("post_code_line_mat", dstMat);
//#endif // _DEBUG
//
	//cv::Mat element = getStructuringElement(MORPH_RECT, Size(10, 1));
	//morphologyEx(bnrMat, bnrMat, MORPH_ERODE, element);
	//element = getStructuringElement(MORPH_RECT, Size(1, 10));
	//morphologyEx(bnrMat, bnrMat, MORPH_ERODE, element);
	//imshow("erode", bnrMat);



	return 1;


}

int HWDigitsOCR::segPostCode(const cv::Mat &srcMat, std::vector<cv::Mat> &dstMat_vec)
{

	Mat bnrMat;
	float ave_pixels = ImageProcessFunc::getAverageBrightness(srcMat);
	cv::threshold(srcMat, bnrMat, ave_pixels/2, 255, CV_THRESH_BINARY);
	bnrMat = ~bnrMat;
	//imshow("bnrMat", bnrMat);
	std::vector<unsigned int> pixels_sum_y;
	OcrAlgorithm::sumPixels(bnrMat, 1, pixels_sum_y);

	//确定手写box水平方向位置
	bool start_cut = false;
	bool continu_flag = false;
	int rect_start = 0;
	int rect_end = 0;
	cv::Rect word_rect;
	word_rect.x = 0;
	word_rect.height = bnrMat.rows;
	word_rect.y = 0;
	std::vector<cv::Rect> word_rect_vec;
	for (int i = 0; i < pixels_sum_y.size(); i++)
	{
		if (pixels_sum_y[i] == 0)
		{
			start_cut = true;
		}
		if (start_cut == true && continu_flag == false && pixels_sum_y[i] > 0)
		{
			rect_start = i;
			continu_flag = true;
		}
		if (start_cut == true && continu_flag == true && pixels_sum_y[i] == 0)
		{
			rect_end = i;
			continu_flag = false;
			if (rect_end - rect_start > bnrMat.rows*1.5 || rect_end - rect_start < bnrMat.rows*0.3)
			{
				continue;
			}
			word_rect.x = rect_start;
			word_rect.width = rect_end - rect_start;
			word_rect_vec.push_back(word_rect);
			if (word_rect_vec.size() == 5) break;
			rect_start = 0;
			rect_end = 0;
		}
	}
	if (word_rect_vec.size() != 5)
	{
		return 0;
	}

	//确定手写box Y轴位置
	std::vector<unsigned int> pixels_sum_x;
	cv::Rect sumRange;
	sumRange.x = word_rect_vec[0].x;
	sumRange.y = 0;
	sumRange.height = word_rect_vec[0].height;
	sumRange.width = word_rect_vec[4].x + word_rect_vec[4].width - word_rect_vec[0].x;
	OcrAlgorithm::sumPixels(bnrMat(sumRange), 0, pixels_sum_x);
	int max_pix = 0;
	for (int i = 0; i < pixels_sum_x.size(); i++)
	{
		if (max_pix < pixels_sum_x[i]) max_pix = pixels_sum_x[i];
	}
	int up_range = max_pix * 0.8;
	int down_range = max_pix * 0.5;

	start_cut = false;
	int up_pos = -1;
	int down_pos = -1;
	for (int i = 0; i < pixels_sum_x.size(); i++)
	{
		if (pixels_sum_x[i] >= up_range && start_cut == false)
		{
			start_cut = true;
		}
		if (pixels_sum_x[i] <= down_range && start_cut == true)
		{
			up_pos = i;
			break;
		}
	}
	start_cut = false;
	for (int i = pixels_sum_x.size() - 1; i > 0; i--)
	{
		if (pixels_sum_x[i] >= up_range && start_cut == false)
		{
			start_cut = true;
		}
		if (pixels_sum_x[i] <= down_range && start_cut == true)
		{
			down_pos = i;
			break;
		}
	}
	if (down_pos < 0 || up_pos < 0 || down_pos <= up_pos)
	{
		return 0;
	}

	for (int i = 0; i < word_rect_vec.size(); i++)
	{
		word_rect_vec[i].y = up_pos;
		word_rect_vec[i].height = down_pos - up_pos + 1;
	}

	//排除手写框外框，
	for (int j = 0; j < word_rect_vec.size(); j++)
	{
		std::vector<unsigned int>().swap(pixels_sum_x);
		ImageProcessFunc::sumPixels(bnrMat(word_rect_vec[j]), 1, pixels_sum_x);
		max_pix = 0;
		for (int i = 0; i < pixels_sum_x.size(); i++)
		{
			if (max_pix < pixels_sum_x[i]) max_pix = pixels_sum_x[i];
		}
		up_range = max_pix * 0.8;
		down_range = max_pix * 0.3;

		start_cut = false;
		up_pos = -1;
		down_pos = -1;
		for (int i = 0; i < pixels_sum_x.size(); i++)
		{
			if (pixels_sum_x[i] >= up_range && start_cut == false)
			{
				start_cut = true;
			}
			if (pixels_sum_x[i] <= down_range && start_cut == true)
			{
				up_pos = i;
				break;
			}
		}
		start_cut = false;
		for (int i = pixels_sum_x.size() - 1; i > 0; i--)
		{
			if (pixels_sum_x[i] >= up_range && start_cut == false)
			{
				start_cut = true;
			}
			if (pixels_sum_x[i] <= down_range && start_cut == true)
			{
				down_pos = i;
				break;
			}
		}
		if (down_pos < 0 || up_pos < 0 || down_pos <= up_pos)
		{
			return 0;
		}
		word_rect_vec[j].x += up_pos;
		word_rect_vec[j].width = down_pos - up_pos + 1;
	}

	//根据外框设置比例取内框
	//for (int j = 0; j < word_rect_vec.size(); j++)
	//{
	//	int box_thickenss_h = round(word_rect_vec[0].height*0.09);
	//	//int box_thickenss_w = round(word_rect_vec[0].height*0.16);
	//	//word_rect_vec[j].x += box_thickenss_w;
	//	word_rect_vec[j].y += box_thickenss_h;
	//	//word_rect_vec[j].width -= 2*box_thickenss_w;
	//	word_rect_vec[j].height -= 2 * box_thickenss_h;
	//	if (word_rect_vec[j].width <=0 || word_rect_vec[j].height<=0)
	//	{
	//		return 0;
	//	}
	//}




#ifdef POSTCODE_BOX_DEBUG
	for (int i = 0; i < word_rect_vec.size(); i++)
	{
		rectangle(bnrMat, word_rect_vec[i], cv::Scalar(125, 125, 125));
	}
	imshow("seg_post_code_box", bnrMat);
#endif // POSTCODE_BOX_DEBUG



	for (int i = 0; i < word_rect_vec.size(); i++)
	{
		cv::Mat m_t;
		srcMat(word_rect_vec[i]).copyTo(m_t);
		dstMat_vec.push_back(m_t);
	}
	return 1;
}

int HWDigitsOCR::getPostcodeType(const cv::Mat &srcMat, tesseract::TessBaseAPI *pTess)
{
	cv::Mat adjust_mat;
	adjust_mat = srcMat;


	float ave_pixls = ImageProcessFunc::getAverageBrightness(adjust_mat);
	OcrAlgorithm::adJustBrightness(adjust_mat, 10, 30, ave_pixls/2.5);

	float scal_ = 40.0 / adjust_mat.rows;
	resize(adjust_mat, adjust_mat, cv::Size(), scal_, scal_);

	

	//imshow("adjust", adjust_mat);
	int w = adjust_mat.cols;
	int h = adjust_mat.rows;
	//unsigned char *pImgData = adjust_mat.data;

	pTess->SetPageSegMode(tesseract::PageSegMode::PSM_SINGLE_LINE);
	pTess->SetVariable("save_best_choices", "T");
	pTess->SetImage(adjust_mat.data, w, h, adjust_mat.channels(), adjust_mat.step1());
	pTess->Recognize(0);

	// get result and delete[] returned char* string
#ifdef POSTCODE_BOX_DEBUG
	cv::imshow("邮编类型调整后", adjust_mat);
	//std::cout << std::unique_ptr<char[]>(pTess->GetUTF8Text()).get() << std::endl;
#endif // OCR_DEBUG
	//
	std::string result_str(pTess->GetUTF8Text());
	cout << result_str<<endl;
	

	size_t _pos = result_str.find_first_of('/');

	if (_pos==result_str.npos)
	{
		return 0;
	}
	string substr = result_str.substr(_pos);
	transform(substr.begin(), substr.end(), substr.begin(), ::tolower);
	
	for (int i=0;i<substr.length();i++)
	{
		unsigned char c = substr[i];
		if (c == '1') substr[i] = 't';
		if (c == '0') substr[i] = 'o';
	}
	//cout << substr << endl;

	_pos = substr.find("/to");
	if (_pos!=substr.npos)
	{
		return 2;
	}
	_pos = substr.find("/from");
	if (_pos != substr.npos)
	{
		return 1;
	}

	return 0;
}

int HWDigitsOCR::getPostCodeFromBoxMats(std::vector<cv::Mat> &srcMat_vec, std::string &result_str, float &confidence, OcrAlgorithm_config* pConfig)
{
	if (srcMat_vec.size() == 0) return 0;
	std::vector<cv::Mat> m_vec;
	thresholdImgs(srcMat_vec, m_vec);
	std::vector<int> class_vec;
	std::vector<float> configdenc_vec;


#ifdef POSTCODE_BOX_DEBUG
	cv::Mat showMat(cv::Size(m_vec.size()*28,28),CV_8UC1);
	for (int i = 0; i < m_vec.size(); i++)
	{
		cv::Rect r(i * 28, 0, 28, 28);
		m_vec[i].copyTo(showMat(r));
	}
	imshow("post_code_boxes",showMat);
#endif // POSTCODE_BOX_DEBUG


	//imshow("s", m_vec[4]);
	//	waitKey(0);
	//}
	HWDigitsRecog *pRecogor = (HWDigitsRecog *)(pConfig->pHWDigitsRecog);
	pRecogor->detect_mat(m_vec, class_vec, configdenc_vec);
	
//#ifdef POSTCODE_BOX_DEBUG
	std::cout << "OCR结果:";
	for (int i = 0; i < class_vec.size(); i++)
	{
		cout << class_vec[i] << "@" << configdenc_vec[i] << "   ";
	}
	std::cout << endl;
//#endif // POSTCODE_BOX_DEBUG


	std::string res_str;
	float m_configence = 1;
	for (int i=0;i<class_vec.size();i++)
	{
		res_str.append(std::to_string(class_vec[i]));
		m_configence *= configdenc_vec[i];
	}
	result_str = res_str;
	confidence = m_configence;
	return class_vec.size();

}

int HWDigitsOCR::getHandWriteAddressRangeMat(const cv::Mat &parcelMat, OcrAlgorithm_config* pConfig, std::vector<cv::Mat> &dstMat_vec)
{
	using namespace cv;
	const int MAX_FEATURES = 512;
	const float GOOD_MATCH_PERCENT = 0.15f;
	//Mat src_im1 = imread(refImg);// "E:/cpp_projects/Thailand_projects/资源文件/handwriterange.jpg");
	Mat src_im2 = parcelMat;
	//assert(!src_im1.empty());


	//float scal_max1 = 300.0 / max(src_im1.cols, src_im1.rows);
	//float scal_min1 = 150.0 / min(src_im1.cols, src_im1.rows);

	//float scal1 = max(scal_max1, scal_min1);

	float scal_max2 = 1000.0 / max(src_im2.cols, src_im2.rows);
	float scal_min2 = 500.0 / min(src_im2.cols, src_im2.rows);

	float scal2 = max(scal_max2, scal_min2);


	const cv::Size im1_size(300,188);//外部参考图片尺寸
	Mat im2;
	cv::resize(src_im2, im2, cv::Size(), scal2, scal2, cv::INTER_AREA);
	//cv::resize(src_im1, im1, cv::Size(), scal1, scal1, cv::INTER_AREA);

	//Mat im1Gray = im1;
	Mat im2Gray = im2;
	//if (im1.channels()==3)
	//{
	//	cvtColor(im1, im1Gray, CV_BGR2GRAY);
	//}
	if (im2.channels() == 3)
	{
		cvtColor(im2, im2Gray, CV_BGR2GRAY);
	}
	

	//cv::Canny(im1Gray, im1Gray, 40, 255);
	//cv::Canny(im2Gray, im2Gray, 40, 255);


	// Variables to store keypoints and descriptors
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	keypoints1 = pConfig->match_data.keypoints_handwrite_addr;
	descriptors1 = pConfig->match_data.descriptors_handwrite_addr;


	// Detect ORB features and compute descriptors.
	Ptr<Feature2D> sift1 = cv::xfeatures2d::SIFT::create(MAX_FEATURES);
	//Ptr<Feature2D> sift2 = SIFT::create(MAX_FEATURES*2);
	clock_t t_start, t_end;
	t_start = clock();
	//sift1->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
	sift1->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);
	t_end = clock();
	double timeconsume = (double)(t_end - t_start) / CLOCKS_PER_SEC;
	cout << "time comsume:" << timeconsume << endl;

	//Mat img_keypoints_1, img_keypoints_2;
	//drawKeypoints(im1Gray, keypoints1, img_keypoints_1, Scalar::all(-1), 0);
	//drawKeypoints(im2Gray, keypoints2, img_keypoints_2, Scalar::all(-1), 0);
	//imshow("img_keypoints_1", img_keypoints_1);
	//imshow("img_keypoints_2", img_keypoints_2);



	//waitKey(0);
	// Match features.
	vector<vector<DMatch>> m_knnMatches;
	vector<DMatch>m_Matches;
	vector<DMatch>n_Matches;
	std::vector<DMatch> best_matches;

	std::vector<DMatch> matches;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(6);//经过测试，6最好
	//matcher->match(descriptors2, descriptors1, matches, Mat());
	matcher->knnMatch(descriptors2, descriptors1, m_knnMatches, 4);
	//// Sort matches by score
	for (int i = 0; i < m_knnMatches.size(); i++)
	{
		double dist_rate1 = m_knnMatches[i][0].distance / m_knnMatches[i][1].distance;
		double dist_rate2 = m_knnMatches[i][1].distance / m_knnMatches[i][2].distance;
		double dist_rate3 = m_knnMatches[i][2].distance / m_knnMatches[i][3].distance;
		//if (dist_rate1< 0.5 && dist_rate2<0.5)
		//{
		//	continue;
		//}
		//if (dist_rate2 < 0.7)
		//{
		//	best_matches.push_back(m_knnMatches[i][0]);
		//}
		if (dist_rate1 < 0.6)
		{
			best_matches.push_back(m_knnMatches[i][0]);
		}
	}
	//排序

	//std::sort(matches.begin(), matches.end());
	//std::vector<DMatch>::iterator it;
	//it = (matches.size() < 50) ? matches.end() : matches.begin() + 50;
	//best_matches.insert(best_matches.end(), matches.begin(), it);




	std::sort(best_matches.begin(), best_matches.end());

	//将出现大于2个对应点的match 删除多余对应点
	std::vector<DMatch> best_matches_2;
	for (int i = 0; i < best_matches.size(); i++)
	{
		int res = query_match_count(best_matches_2, best_matches[i]);
		if (res <= 1)
		{
			best_matches_2.push_back(best_matches[i]);
		}
	}
	if (best_matches_2.size() < 10) return 0;

	//进行2分类聚类
	cv::Mat pt_kmeans(best_matches_2.size(), 1, CV_32FC2);
	for (int i = 0; i < best_matches_2.size(); i++)
	{
		pt_kmeans.at<Vec2f>(i, 0)[0] = keypoints2[best_matches_2[i].queryIdx].pt.x;
		pt_kmeans.at<Vec2f>(i, 0)[1] = keypoints2[best_matches_2[i].queryIdx].pt.y;
	}
	std::vector<int> labels;
	std::vector<Point2f> centers;
	int clusters_num = 2;
	cv::kmeans(pt_kmeans, clusters_num, labels, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

	std::vector<DMatch> best_matches_class1;
	std::vector<DMatch> best_matches_class2;
	for (int i = 0; i < labels.size(); i++)
	{
		if (labels[i] == 0)
		{
			best_matches_class1.push_back(best_matches_2[i]);
		}
		else
		{
			best_matches_class2.push_back(best_matches_2[i]);
		}
	}
	if (best_matches_class1.size() < 5 || best_matches_class2.size() < 5)
	{
		return 0;
	}



	//std::sort(matches.begin(), matches.end());



	// Remove not so good matches


	// Draw top matches
	//Mat imMatches1, imMatches2;
	//drawMatches(im2, keypoints2, im1, keypoints1, best_matches_class1, imMatches1);
	//drawMatches(im2, keypoints2, im1, keypoints1, best_matches_class2, imMatches2);

	//imshow("m1", imMatches1);
	//imshow("m2", imMatches2);




	// Extract location of good matches
	std::vector<Point2f> points_class1, points_class2, points_ref1, points_ref2;

	for (size_t i = 0; i < best_matches_class1.size(); i++)
	{
		points_class1.push_back(keypoints2[best_matches_class1[i].queryIdx].pt);
		points_ref1.push_back(keypoints1[best_matches_class1[i].trainIdx].pt);
	}
	for (size_t i = 0; i < best_matches_class2.size(); i++)
	{
		points_class2.push_back(keypoints2[best_matches_class2[i].queryIdx].pt);
		points_ref2.push_back(keypoints1[best_matches_class2[i].trainIdx].pt);
	}

	FindHomography_2d findhg;
	Mat ms1, ms2;
	cv::RNG rng;
	cv::Mat m1(points_class1);
	cv::Mat m2(points_ref1);
	cv::Mat m3(points_class2);
	cv::Mat m4(points_ref2);
	//findhg.getSubset(m1, m2, ms1, ms2, rng);
	cout << "参与1的点数：" << points_class1.size() << endl;
	cout << "参与2的点数：" << points_class2.size() << endl;

	Mat h1, h2, mask1, mask2;
	bool r = findhg.run(m1, m2, h1, mask1);
	r = findhg.run(m3, m4, h2, mask2);
	if (h1.empty() || h2.empty())
	{
		return 0;
	}
	//cout << mask2 << endl;
	cv::transpose(h1, h1);
	cv::transpose(h2, h2);
	//cout << h1 << endl;


	std::vector<Point2f> points_class1_0, points_class2_0, points_ref1_0, points_ref2_0;
	uchar* maskptr = mask1.ptr<uchar>();
	int n = mask1.total();
	for (int i = 0; i < n; i++)
	{
		if (maskptr[i] == 1)
		{
			points_class1_0.push_back(points_class1[i]);
			points_ref1_0.push_back(points_ref1[i]);
		}

	}
	maskptr = mask2.ptr<uchar>();
	n = mask2.total();
	for (int i = 0; i < n; i++)
	{
		if (maskptr[i] == 1)
		{
			points_class2_0.push_back(points_class2[i]);
			points_ref2_0.push_back(points_ref2[i]);
		}

	}
	cv::Mat m1_0(points_class1_0);
	cv::Mat m2_0(points_ref1_0);
	cv::Mat m3_0(points_class2_0);
	cv::Mat m4_0(points_ref2_0);
	cv::Point2f scal_xy_1, scal_xy_2;
	scal_xy_1.x = scal2; scal_xy_1.y = scal2;
	scal_xy_2.x = scal2; scal_xy_2.y = scal2;
	Mat h1_0, h2_0;
	findhg.runKernel_NoZoom(m1_0, m2_0, h1_0, scal_xy_1);
	findhg.runKernel_NoZoom(m3_0, m4_0, h2_0, scal_xy_2);
	cv::transpose(h1_0, h1_0);
	cv::transpose(h2_0, h2_0);

	cv::Size result1_size = im1_size;
	cv::Size result2_size = im1_size;
	float zoom_s = 1.1;
	result1_size.width *= (scal_xy_1.x*zoom_s);
	result1_size.height *= (scal_xy_1.y*zoom_s);
	result2_size.width *= (scal_xy_2.x*zoom_s);
	result2_size.height *= (scal_xy_2.y*zoom_s);




	//cout << h1 << endl;
	//cout << h2 << endl;

	int tt = 0;
	//// Find homography
	//Mat h = findHomography(points2,points1, RANSAC);
	Mat im1Reg, im2Reg;
	//// Use homography to warp image
	if (!h1.empty() && !h2.empty())
	{
		//warpPerspective(im2, im1Reg, h1, im1.size());
		//warpPerspective(im2, im2Reg, h2, im1.size());

		warpPerspective(src_im2, im1Reg, h1_0, result1_size);
		warpPerspective(src_im2, im2Reg, h2_0, result2_size);
		dstMat_vec.push_back(im1Reg);
		dstMat_vec.push_back(im2Reg);
		//imshow("检测到1", im1Reg);
		//imshow("检测到2", im2Reg);
		return 1;
	}


	//waitKey(0);

	return 0;




}

int HWDigitsOCR::thresholdImgs(std::vector<cv::Mat> &srcMat_vec, std::vector<cv::Mat> &dstMat_vec)
{
	for (int i=0;i<srcMat_vec.size();i++)
	{
		cv::Mat mm;
		srcMat_vec[i].copyTo(mm);
		if (mm.channels() == 3)
		{
			cv::cvtColor(mm, mm, cv::COLOR_BGR2GRAY);
		}
		double av_pixels = ImageProcessFunc::getAverageBrightness(mm);
		//std::cout << av_pixels << endl;
		ImageProcessFunc::adJustBrightness(mm, 10, 0, av_pixels / 2.0);
		//cv::threshold(srcm, srcm, 40, 255,cv::THRESH_BINARY);
		mm = ~mm;
		int boarder_w = 1;
		//ImageProcessFunc::makeBoarderConstant(mm, 0, boarder_w);
		int mm_h = mm.rows, mm_w = mm.cols;
		cv::Mat boarder_mat = mm(cv::Rect(boarder_w, boarder_w, mm_w - 2 * boarder_w, mm_h - 2 * boarder_w));//去除边框
		cv::Mat resizeMat;
		cv::resize(boarder_mat, resizeMat, cv::Size(28, 28));
		dstMat_vec.push_back(resizeMat);
		//if (i == 2)
		//{
		//	cv::imwrite("F:/手写框/hardwork/0_00000.jpg", mm);
		//}
	}
	return 1;
}

int HWDigitsOCR::query_match_count(std::vector<DMatch> &matches, DMatch & new_match)
{
	int match_count = 0;
	std::vector<DMatch>::iterator it;
	for (it = matches.begin(); it != matches.end(); it++)
	{
		if (new_match.trainIdx == it->trainIdx) match_count++;
	}
	return match_count;
}

int HWDigitsOCR::query_near_count(std::vector<float> &datas, float down_value, float up_value)
{
	std::vector<float> m_data = datas;
	std::sort(m_data.begin(), m_data.end());
	std::vector<float>::iterator it;
	it = m_data.begin();
	int count_ = 0;

	for (;it!= m_data.end();it++)
	{
		if (*it>up_value)
		{
			break;
		}
		if (*it >=down_value && *it <=up_value)
		{
			count_++;
		}
	}
	return count_;
}

int HWDigitsOCR::score_for_rect(std::vector<cv::Rect> rcs, int im_width, int im_height, std::vector<float> &rc_scores)
{
	const int ref_width = 200;
	const float ref_ratio = 7;
	const int ref_height_up = 35;
	const int ref_height_down = 10;

	for (int i=0;i<rcs.size();i++)
	{
		float score_ = 1;
		int c_x = rcs[i].x + rcs[i].width / 2;
		int c_y = rcs[i].y + rcs[i].height / 2;
		if ((c_x>im_width/2 && c_y<im_height/2) ||(c_x<im_width/2 && c_y>im_height/2))
		{
			rc_scores.push_back(0);
			continue;
		}
		if (rcs[i].height>ref_height_up || rcs[i].height < ref_height_down )
		{
			rc_scores.push_back(0);
			continue;
		}
		float _ratio = float(rcs[i].width) / rcs[i].height;
		float score_tmp = 1 - fabs(_ratio - ref_ratio) / ref_ratio; //长宽比打分
		score_tmp = (score_tmp < 0) ? 0 : score_tmp;
		score_ *= score_tmp;

		rc_scores.push_back(score_);


	}

	return 1;
}

int HWDigitsOCR::getPostCodeLine_nobox(cv::Mat srcMat, std::vector<cv::Mat> &toMats, std::vector<cv::Mat> &fromMats)
{
	if (srcMat.empty()) return 0;

	Mat grymat;

	float scal_max2 = 1000.0 / max(srcMat.cols, srcMat.rows);

	cv::resize(srcMat, grymat, cv::Size(), scal_max2, scal_max2, cv::INTER_AREA);

	if (grymat.channels() == 3) cvtColor(grymat, grymat, COLOR_BGR2GRAY);

	//if (grymat.rows > grymat.cols) cv::rotate(grymat, grymat,ROTATE_90_CLOCKWISE);//旋转图片

	//imshow("srcm", grymat);

	if (grymat.rows < 200 || grymat.cols < 200) return 0;

	
	//Rect cr(bord_wd, 0, grymat.cols - 2 * bord_wd, grymat.rows);
	Rect cr_sample(grymat.cols*0.2, grymat.rows*0.1, grymat.cols*0.6, grymat.rows*0.4);

	//imshow("srcm2", centm);
	double avg_pix = ImageProcessFunc::getAveragePixelInRect(grymat, cr_sample);

	Mat bnmat;
	cv::threshold(grymat, bnmat, avg_pix / 1.5, 255, CV_THRESH_BINARY);
	bnmat = ~bnmat;

#ifdef POSTCODE_BOX_DEBUG
	imshow("bnmat", bnmat);
#endif // POSTCODE_BOX_DEBUG


	cv::Mat element;
	element = getStructuringElement(MORPH_RECT, Size(5, 20));
	morphologyEx(bnmat, bnmat, MORPH_CLOSE, element);
	element = getStructuringElement(MORPH_RECT, Size(50, 5));
	morphologyEx(bnmat, bnmat, MORPH_CLOSE, element);

	element = getStructuringElement(MORPH_RECT, Size(1, 5));
	morphologyEx(bnmat, bnmat, MORPH_OPEN, element);


	element = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(bnmat, bnmat, MORPH_ERODE, element);




#ifdef POSTCODE_BOX_DEBUG
	imshow("morpmat", bnmat);
#endif // POSTCODE_BOX_DEBUG

	//裁切掉黑边
	vector<unsigned int> sum_pixels;
	ImageProcessFunc::sumPixels(bnmat, 0, sum_pixels);
	int iw = bnmat.cols;
	int cut_line_up = 0;
	for (int i = 0; i < sum_pixels.size(); i++)
	{
		int avpix = sum_pixels[i] / iw;
		if (avpix >= 127)
		{
			cut_line_up++;
		}
		else
		{
			break;
		}
	}
	int cut_line_down = 0;
	for (int i = sum_pixels.size() - 1; i >= 0; i--)
	{
		int avpix = sum_pixels[i] / iw;
		if (avpix >= 120)
		{
			cut_line_down++;
		}
		else
		{
			break;
		}
	}
	if ((cut_line_down + cut_line_up) > (bnmat.rows / 2))
	{
		cout << "裁切上下异常" << endl;
		return 0;
	}
	Rect rc_tb_cut(0, 0, bnmat.cols, bnmat.rows);
	if (cut_line_down != 0 || cut_line_up != 0)
	{
		rc_tb_cut.y = cut_line_up;
		rc_tb_cut.height = rc_tb_cut.height - (cut_line_up + cut_line_down);
	}
	bnmat = bnmat(rc_tb_cut);

	vector<unsigned int>().swap(sum_pixels);
	ImageProcessFunc::sumPixels(bnmat, 1, sum_pixels);
	int ih = bnmat.rows;
	int cut_line_left = 0;
	for (int i = 0; i < sum_pixels.size(); i++)
	{
		int avpix = sum_pixels[i] / ih;
		if (avpix >= 127)
		{
			cut_line_left++;
		}
		else
		{
			break;
		}
	}
	int cut_line_right = 0;
	for (int i = sum_pixels.size() - 1; i >= 0; i--)
	{
		int avpix = sum_pixels[i] / ih;
		if (avpix >= 127)
		{
			cut_line_right++;
		}
		else
		{
			break;
		}
	}
	if ((cut_line_left + cut_line_right) > (bnmat.cols / 2))
	{
		cout << "裁切左右异常" << endl;
		return 0;
	}
	Rect rc_lr_cut(0, 0, bnmat.cols, bnmat.rows);
	if (cut_line_down != 0 || cut_line_up != 0)
	{
		rc_lr_cut.x =cut_line_left;
		rc_lr_cut.width = rc_lr_cut.width - (cut_line_left + cut_line_right);
	}
	bnmat = bnmat(rc_lr_cut);



	////imshow("cut_tb", bnmat);

	//获得轮廓的rect
	std::vector<std::vector<cv::Point>>contours;
	std::vector<cv::Vec4i> hierarchy;
	std::vector<cv::Point> contour;
	double aera = 0;
	//src_gray = src_gray > 100;
	cv::findContours(bnmat, contours, hierarchy, RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	vector<float> contours_score;
	vector<Rect> contours_rect;
	if (contours.size()==0)
	{
		return 0;
	}
	for (int i = 0; i < contours.size(); i++)
	{
		Rect _rc;
		ImageProcessFunc::getContourRect(contours[i], _rc);
		double maera = cv::contourArea(contours[i]);
		if (_rc.width*_rc.height*0.5 > maera)//轮廓面积只有rect面积的1/2，弃掉
		{
			continue;
		}
		//rectangle(bnmat, _rc, Scalar(125, 125, 125));
		contours_rect.push_back(_rc);
	}
	score_for_rect(contours_rect, bnmat.cols, bnmat.rows, contours_score);

	//查找评分大于阈值的rect
	float score_threshold = 0.3;//评分阈值
	vector<cv::Rect> candidate_rects;
	for (int i = 0; i < contours_score.size(); i++)
	{
		if (contours_score[i] < score_threshold)
		{
			continue;
		}
		candidate_rects.push_back(contours_rect[i]);
	}

	if (candidate_rects.empty())
	{
		cout << "没有找到合适的候选框" << endl;
		return 0;
	}

	//拆分rect
	vector<cv::Rect> candidate_lt_rects; //左上角
	vector<cv::Rect> candidate_rb_rects; //右下角
	for (int i=0;i<candidate_rects.size();i++)
	{
		if ((candidate_rects[i].x+candidate_rects[i].width/2)<bnmat.cols/2
			|| (candidate_rects[i].y + candidate_rects[i].height / 2) < bnmat.rows / 2)
		{
			candidate_lt_rects.push_back(candidate_rects[i]);
		}
		else
		{
			candidate_rb_rects.push_back(candidate_rects[i]);
		}
	}

	rc_lr_cut.x /= scal_max2;
	rc_lr_cut.y /= scal_max2;
	rc_lr_cut.height /= scal_max2;
	rc_lr_cut.width /= scal_max2;

	rc_tb_cut.x /= scal_max2;
	rc_tb_cut.y /= scal_max2;
	rc_tb_cut.width /= scal_max2;
	rc_tb_cut.height /= scal_max2;


	//获取候选的框
	vector<cv::Mat> fromMatVec;
	vector<cv::Mat> toMatVec;
	for (int i=0;i<candidate_lt_rects.size();i++)
	{
		cv::Rect refinedRect = candidate_lt_rects[i];
		refinedRect.x /= scal_max2;
		refinedRect.y /= scal_max2;
		refinedRect.width /= scal_max2;
		refinedRect.height /= scal_max2;
		bool needRotate = false;
		getHandWriteRange(srcMat(rc_tb_cut)(rc_lr_cut), refinedRect, refinedRect, needRotate);
		cv::Mat candi_mat = srcMat(rc_tb_cut)(rc_lr_cut)(refinedRect);
		if (needRotate)
		{
			cv::rotate(candi_mat, candi_mat, ROTATE_180);
			toMatVec.push_back(candi_mat);
		}
		else
		{
			candi_mat = candi_mat.clone();
			fromMatVec.push_back(candi_mat);
		}

	}

	for (int i = 0; i < candidate_rb_rects.size(); i++)
	{
		cv::Rect refinedRect = candidate_rb_rects[i];
		refinedRect.x /= scal_max2;
		refinedRect.y /= scal_max2;
		refinedRect.width /= scal_max2;
		refinedRect.height /= scal_max2;
		bool needRotate = false;
		getHandWriteRange(srcMat(rc_tb_cut)(rc_lr_cut), refinedRect, refinedRect, needRotate);
		cv::Mat candi_mat = srcMat(rc_tb_cut)(rc_lr_cut)(refinedRect);
		if (needRotate)
		{
			cv::rotate(candi_mat, candi_mat, ROTATE_180);
			fromMatVec.push_back(candi_mat);
		}
		else
		{
			candi_mat = candi_mat.clone();
			toMatVec.push_back(candi_mat);
		}

	}
	toMatVec.swap(toMats);
	fromMatVec.swap(fromMats);



	/*
	   	  
	cout << "best socre:" << best_score_tmp << "/" << best_score_tmp2 << endl;
	if (best_ind1 == -1 && best_ind2 == -1)//未找到合适的框
	{
		cout << "没有找到合适的rect" << endl;
		return 0;
	}
	if (best_ind1 != -1 && best_ind2 != -1)//找到两个rect
	{
		//如果两个rect在同一区域
		if (((contours_rect[best_ind1].x+ contours_rect[best_ind1].width / 2) < bnmat.cols / 2) 
			== ((contours_rect[best_ind2].x + contours_rect[best_ind2].width / 2) < bnmat.cols / 2))
		{
			best_ind2 = -1;
		}
	}
	if ((contours_rect[best_ind1].x + contours_rect[best_ind1].width / 2) > bnmat.cols / 2)//
	{
		int tmp = best_ind1;
		best_ind1 = best_ind2;
		best_ind2 = tmp;
	}


#ifdef POSTCODE_BOX_DEBUG
	Mat mshow = bnmat.clone();
	if(best_ind1!=-1) rectangle(mshow, contours_rect[best_ind1], Scalar(125, 125, 125));
	if (best_ind2 != -1) rectangle(mshow, contours_rect[best_ind2], Scalar(125, 125, 125));
	imshow("rect", mshow);
#endif // POSTCODE_BOX_DEBUG

	rc_lr_cut.x /= scal_max2;
	rc_lr_cut.y /= scal_max2;
	rc_lr_cut.height /= scal_max2;
	rc_lr_cut.width /= scal_max2;

	rc_tb_cut.x /= scal_max2;
	rc_tb_cut.y /= scal_max2;
	rc_tb_cut.width /= scal_max2;
	rc_tb_cut.height /= scal_max2;

	bool is_rotate1 = false;
	bool is_rotate2 = false;
	cv::Mat m_from_mat, m_to_mat;
	Rect best_rect1_refine;
	Rect best_rect2_refine;
	if (best_ind1!=-1)
	{
		best_rect1_refine = contours_rect[best_ind1];
		best_rect1_refine.x /= scal_max2;
		best_rect1_refine.y /= scal_max2;
		best_rect1_refine.width /= scal_max2;
		best_rect1_refine.height /= scal_max2;
		getHandWriteRange(srcMat(rc_tb_cut)(rc_lr_cut), best_rect1_refine, best_rect1_refine,is_rotate1);
		m_from_mat = srcMat(rc_tb_cut)(rc_lr_cut)(best_rect1_refine);
	}
	if (best_ind2 != -1)
	{
		best_rect2_refine = contours_rect[best_ind2];
		best_rect2_refine.x /= scal_max2;
		best_rect2_refine.y /= scal_max2;
		best_rect2_refine.width /= scal_max2;
		best_rect2_refine.height /= scal_max2;
		getHandWriteRange(srcMat(rc_tb_cut)(rc_lr_cut), best_rect2_refine, best_rect2_refine, is_rotate2);
		m_to_mat = srcMat(rc_tb_cut)(rc_lr_cut)(best_rect2_refine);
	}
	if (best_ind1 != -1 && best_ind2 != -1)
	{
		int top2rect1 = contours_rect[best_ind1].y + contours_rect[best_ind1].height / 2;
		int bottom2rect2 = bnmat.rows - (contours_rect[best_ind2].y + contours_rect[best_ind2].height / 2);
		int left2rect1 = contours_rect[best_ind1].x + contours_rect[best_ind1].width / 2;
		int right2rect2 = bnmat.cols - (contours_rect[best_ind2].x + contours_rect[best_ind2].width / 2);
		float pos_rate_y = float(top2rect1) / bottom2rect2;
		float pos_rate_x = float(left2rect1) / right2rect2;
		if (pos_rate_y < 0.75 || pos_rate_x < 0.75)
		{
			cv::rotate(m_from_mat, m_from_mat, ROTATE_180);
			cv::rotate(m_to_mat, m_to_mat, ROTATE_180);
			m_from_mat.copyTo(toMat);
			m_to_mat.copyTo(fromMat);
		}
		else
		{
			m_from_mat.copyTo(fromMat);
			m_to_mat.copyTo(toMat);
		}

	}
	else
	{
		if (is_rotate2 || is_rotate1)
		{
			if (!m_from_mat.empty())
			{
				cv::rotate(m_from_mat, m_from_mat, ROTATE_180);
				m_from_mat.copyTo(toMat);
			}
			if (!m_to_mat.empty())
			{
				cv::rotate(m_to_mat, m_to_mat, ROTATE_180);
				m_to_mat.copyTo(fromMat);
			}
		}
		else
		{
			m_from_mat.copyTo(fromMat);
			m_to_mat.copyTo(toMat);
		}
	}
	*/
	return 1;
}



int HWDigitsOCR::getHandWriteRange(cv::Mat &srcMat, cv::Rect &srcRect, cv::Rect &dstRect, bool &need_rotate)
{
	if (srcMat.empty()) return 0;
	if (srcMat.channels() == 3)
	{
		cv::cvtColor(srcMat, srcMat, COLOR_BGR2GRAY);
	}


	Rect best_rect1_refine = srcRect;
	//Rect best_rect2_refine = contours_rect[best_ind2];

	float zoom_scal_x = 1.5;
	float zoom_scal_y = 2.0;

	best_rect1_refine.width = best_rect1_refine.width*zoom_scal_x;
	best_rect1_refine.height = best_rect1_refine.height*zoom_scal_y;
	best_rect1_refine.x = best_rect1_refine.x - srcRect.width*(zoom_scal_x - 1) / 2;
	best_rect1_refine.y = best_rect1_refine.y - srcRect.height*(zoom_scal_y - 1) / 2;


	int res = ImageProcessFunc::CropRect(Rect(0, 0, srcMat.cols, srcMat.rows), best_rect1_refine);
	if (res == 0) return 0;

	dstRect = best_rect1_refine;

	//检查是否旋转
	int top2rect1 = best_rect1_refine.y + best_rect1_refine.height / 2;
	int left2rect1 = best_rect1_refine.x + best_rect1_refine.width / 2;
	int bottom2rect1 = srcMat.rows - top2rect1;
	int right2rect1 = srcMat.cols - left2rect1;

	if (top2rect1<bottom2rect1)
	{
		if (top2rect1 < srcMat.rows / 4 && left2rect1 < srcMat.cols / 4)
		{
			need_rotate = true;
		}
		else
		{
			need_rotate = false;
		}
	}
	else
	{
		if (bottom2rect1 > srcMat.rows / 4 && right2rect1 > srcMat.cols / 4)
		{
			need_rotate = true;
		}
		else
		{
			need_rotate = false;
		}
	}

	return 1;

}

int HWDigitsOCR::split_digits_nobox(cv::Mat &srcMat, std::vector<cv::Mat> &dstDigits)
{
	if (srcMat.empty())
	{
		return 0;
	}
	if (srcMat.channels()==3)
	{
		cvtColor(srcMat, srcMat, COLOR_BGR2GRAY);
	}

	float avg_pix = ImageProcessFunc::getAverageBrightness(srcMat);
	Mat adj_mat = srcMat.clone();
	ImageProcessFunc::adJustBrightness(adj_mat, 10, 0, avg_pix/2.0);
	adj_mat = ~adj_mat;
	
#ifdef POSTCODE_BOX_DEBUG
	imshow("digits", adj_mat);
#endif // POSTCODE_BOX_DEBUG

	vector<unsigned int> sum_pixs;
	ImageProcessFunc::sumPixels(adj_mat, 1, sum_pixs);

	int ih = adj_mat.rows;
	vector<Point> seg_points_x;
	bool start_flag = false;
	Point pt;
	for (size_t i = 0; i < sum_pixs.size(); i++)
	{
		int _p = sum_pixs[i] / ih;
		if (_p > 3 && start_flag==false)
		{
			pt.x = i;
			start_flag = true;
		}
		if (_p <=3 && start_flag==true)
		{
			pt.y = i;
			seg_points_x.push_back(pt);
			start_flag = false;
		}
	}
	//if (seg_points_x.size() != 5)
	//{
	//	cout << "分割数字失败" << endl;
	//	return 0;
	//}
	vector<Rect> seg_rects;
	for (size_t i = 0; i < seg_points_x.size(); i++)
	{
		Rect _rc(seg_points_x[i].x, 0, seg_points_x[i].y - seg_points_x[i].x, adj_mat.rows);
		int iw = _rc.width;
		vector<unsigned int> sum_pixs;
		ImageProcessFunc::sumPixels(adj_mat(_rc), 0, sum_pixs);
		Point pt;
		for (size_t j = 0; j < sum_pixs.size(); j++)
		{
			int _p = sum_pixs[j] / iw;
			if (_p > 10 )
			{
				pt.x = j;
				break;
			}
		}
		for (size_t j = sum_pixs.size()-1; j >= 0 ; j--)
		{
			int _p = sum_pixs[j] / iw;
			if (_p > 10)
			{
				pt.y = j;
				break;
			}
		}
		if (pt.x == pt.y)
		{
			cout << "分割数字失败" << endl;
			return 0;
		}
		_rc.y = pt.x;
		_rc.height = pt.y - pt.x + 1;
		seg_rects.push_back(_rc);
	}

	//过滤rect
	if (seg_rects.size()<5)
	{
		cout << "分割数字失败" << endl;
		return 0;
	}
	if (seg_rects.size() > 5)
	{
		vector<Rect>::iterator it = seg_rects.begin();
		int area_threshold = 50;
		for (;it!=seg_rects.end();)
		{
			if (it->area() < area_threshold)
			{
				it = seg_rects.erase(it);
			}
			else
			{
				it++;
			}
		}
	}
#ifdef POSTCODE_BOX_DEBUG
	Mat mshow = adj_mat.clone();
	for (size_t i = 0; i < seg_rects.size(); i++)
	{
		rectangle(mshow, seg_rects[i], Scalar(150, 150, 150));
	}
	imshow("adj_mat", mshow);
#endif // POSTCODE_BOX_DEBUG


	if (seg_rects.size() != 5)
	{
		cout << "分割数字失败" << endl;
		return 0;
	}

	//调整rect尺寸
	for (size_t i = 0; i < seg_rects.size(); i++)
	{
		Rect _rc = seg_rects[i];
		int bd = 3;
		_rc.x -= bd;
		_rc.y -= bd;
		_rc.width += 2 * bd;
		_rc.height += 2 * bd;
		if (_rc.width<_rc.height)
		{
			int cent_x = _rc.x + _rc.width / 2;
			_rc.width = _rc.height;
			_rc.x = cent_x - _rc.width / 2;
		}
		if (_rc.width > _rc.height)
		{
			int cent_y = _rc.y + _rc.height / 2;
			_rc.height = _rc.width;
			_rc.y = cent_y - _rc.height / 2;
		}
		ImageProcessFunc::CropRect(Rect(0, 0, adj_mat.cols, adj_mat.rows), _rc);
		seg_rects[i] = _rc;
	}
	Mat cropMat = srcMat.clone();
	ImageProcessFunc::adJustBrightness(cropMat, 10, 0, avg_pix / 1.4);
	cropMat = ~cropMat;


	for (size_t i = 0; i < seg_rects.size(); i++)
	{
		Mat _m = cropMat(seg_rects[i]);
		cv::resize(_m, _m, cv::Size(28, 28));
		dstDigits.push_back(_m.clone());
	}
	return 1;
}

int HWDigitsOCR::getPostCode_nobox(std::vector<cv::Mat> &srcMat_vec, std::vector<std::string> &result_str, std::vector<float> &confidence, OcrAlgorithm_config* pConfig)
{
	if (srcMat_vec.size() == 0) return 0;
	std::vector<cv::Mat> m_vec = srcMat_vec;

	std::vector<int> class_vec;
	std::vector<float> configdenc_vec;

#ifdef POSTCODE_BOX_DEBUG
	cv::Mat showMat(cv::Size(m_vec.size() * 28, 28), CV_8UC1);
	for (int i = 0; i < m_vec.size(); i++)
	{
		cv::Rect r(i * 28, 0, 28, 28);
		m_vec[i].copyTo(showMat(r));
	}
	imshow("post_code_boxes", showMat);
#endif // POSTCODE_BOX_DEBUG

	//imshow("s", m_vec[4]);
	//	waitKey(0);
	//}
	HWDigitsRecog *pRecogor = (HWDigitsRecog *)(pConfig->pHWDigitsRecog);
	pRecogor->detect_mat(m_vec, class_vec, configdenc_vec);

#ifdef POSTCODE_BOX_DEBUG
	std::cout << "OCR结果:";
	for (int i = 0; i < class_vec.size(); i++)
	{
		cout << class_vec[i] << "@" << configdenc_vec[i] << "   ";
	}
	std::cout << endl;
#endif // POSTCODE_BOX_DEBUG

	int postcodeline_num = class_vec.size() / 5;


	for (int i=0;i<postcodeline_num;i++)
	{

		std::string res_str;
		float m_configence = 1;
		for (int j = 0; j < 5; j++)
		{
			res_str.append(std::to_string(class_vec[i+j]));
			m_configence *= configdenc_vec[i+j];
		}
		result_str.push_back(res_str);
		confidence.push_back(m_configence);

	}

	return postcodeline_num;

}
