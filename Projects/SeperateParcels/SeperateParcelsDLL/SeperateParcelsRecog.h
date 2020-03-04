
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
	// ����:    detect_mat		
	// ���ã�
	// ȫ��:  ParcelsRecog::detect_mat		
	// ����ֵ:   int		#
	// ����: cv::Mat & pMat			# ����ͼ��
	// ����: std::vector<std::vector<double>> & outParcelsRanges			#�����������������Ϣ�������vector������һ��������4�����������
	//************************************
	int detect_mat(cv::Mat &pMat, std::vector<std::vector<cv::Point2f>> &outParcelsRanges);


	//************************************
	// ����:    detect_mat_		
	// ���ã�
	// ȫ��:  ParcelsRecog::detect_mat_		
	// ����ֵ:   int		#
	// ����: unsigned char * pImgData			#ͼ������ָ��
	// ����: int channels			#ͼ��ͨ������1��3
	// ����: int width			#ͼ����
	// ����: int height			#ͼ��߶�
	// ����: double * outData			#������ݻ�����
	// ����: int bufferlength			#����������
	// ����: int & parcel_num			#��⵽�İ���������ÿ8�����ݹ���һ�����������ݽṹ��P1x,P1y,P2x,P2y,P3x,P3y,P4x,P4y,...
	//************************************
	int detect_mat_(unsigned char *pImgData,int channels,int width,int height,double *outData,int bufferlength,int &parcel_num);
private:
	void* pDetector = NULL;
	void* pDetector_avg = NULL;
};

