#include "CutParcelBoxDll.h"
#include "CutParcelBox.h"
CutParcelBoxDll::CutParcelBoxDll()
{

}

int CutParcelBoxDll::getMailBox_top_c(const void *pImgData, int iw, int ih, int ic, int *RectPoints, int applid_rotate /*= 1*/, double boxSizeThreshold /*= 50*/, double binaryThreshold /*= 50*/)
{
	CutParcelBox cpb;
	return cpb.getMailBox_Rect_Raw(pImgData, iw, ih, ic, RectPoints, applid_rotate, boxSizeThreshold, binaryThreshold);
}

int CutParcelBoxDll::getMailBox_top_path(std::string &srcImgPath, std::string &dstImgPath, int applid_rotate /*= 1*/, double boxSizeThreshold /*= 50*/, double binaryThreshold /*= 50*/)
{
	CutParcelBox cpb;
	return cpb.getMailBox(srcImgPath, dstImgPath, applid_rotate, boxSizeThreshold, binaryThreshold);
}

int CutParcelBoxDll::getMailBox_top_path_c(const char* pSrcImgPath, const char* pDstImgPath, int applid_rotate /*= 1*/, double boxSizeThreshold /*= 50*/, double binaryThreshold /*= 50*/)
{
	CutParcelBox cpb;
	return cpb.getMailBox_c(pSrcImgPath, pDstImgPath, applid_rotate, binaryThreshold);
}

int CutParcelBoxDll::getMailBox_top_rect(cv::Mat &srcMat, cv::RotatedRect &dstRtRect, double boxSizeThreshold /*= 50*/, double binaryThreshold /*= 50*/)
{
	CutParcelBox cpb;
	return cpb.getMailBox_RtRect(srcMat, dstRtRect, boxSizeThreshold, binaryThreshold);
}

int CutParcelBoxDll::getMailBox_top_mat(cv::Mat &srcMat, cv::Mat &dstMat, int applid_rotate /*= 1*/, double boxSizeThreshold /*= 50*/, double binaryThreshold /*= 50*/)
{
	CutParcelBox cpb;
	return cpb.getMailBox_Mat(srcMat, dstMat, applid_rotate, boxSizeThreshold, binaryThreshold);
}

int CutParcelBoxDll::getMailBox_side_mat(cv::Mat &srcMat, cv::Mat &dstMat, double boxSizeThreshold /*= 50*/, double binaryThreshold /*= 50*/)
{
	CutParcelBox cpb;
	return cpb.getMailBox_side(srcMat, dstMat, boxSizeThreshold, binaryThreshold);
}

int CutParcelBoxDll::getMailBox_side_rect(cv::Mat &srcMat, cv::Rect &dstRect, double boxSizeThreshold /*= 50*/, double binaryThreshold /*= 50*/)
{
	CutParcelBox cpb;
	return cpb.getMailBox_side(srcMat, dstRect, boxSizeThreshold, binaryThreshold);
}


