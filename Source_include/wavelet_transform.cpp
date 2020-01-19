
#include <opencv.hpp>
#include <opencv2/opencv.hpp>
#include "wavelet_coef.h"
#include "wavelet_transform.h"
int WaveletTransformer::wavelet_transform(cv::Mat srcMat, cv::Mat &dstMat, double threshold, int coef_direc, const std::string wavelet_name)
{
	if (srcMat.empty())
	{
		return 0;
	}
	if (srcMat.channels() == 3)
	{
		cv::cvtColor(srcMat, srcMat, CV_BGR2GRAY);
	}
	int w = srcMat.cols;
	int h = srcMat.rows;
	//mwSize dims[2] = { h, w }; 
	//mxArray *pMat = NULL;
	//UINT8 *input = NULL;
	mwArray mMat(h, w, mxUINT8_CLASS, mxREAL);
	//pMat = mxCreateNumericArray(2, dims, mxUINT8_CLASS, mxREAL);
	cv::transpose(srcMat, srcMat);
	mMat.SetData(srcMat.data, w*h);

	mwArray mThreshold(1, 1, mxDOUBLE_CLASS, mxREAL);
	//double m_threshold[1];
	//m_threshold[0] = threshold;
	//mThreshold.SetData(m_threshold,1);
	mwArray outMat;
	if (wavelet_name.empty()) return 0;
	mwArray mWaveletName(wavelet_name.c_str());
	uint coef_index[1];
	coef_index[0] = coef_direc;
	mwArray mcoef_index(1, 1, mxUINT8_CLASS, mxREAL);
	mcoef_index.SetData(coef_index, 1);



	//cout << "into wavelet_coef" << endl;
	wavelet_coef(1, outMat, mMat, mcoef_index, mWaveletName);

	//cout << "out wavelet_coef" << endl;

	if (outMat.IsEmpty()) /// 是否为空
	{
		//cout << "返回为空" << endl;
		return 0;
	}


	mwArray dimn = outMat.GetDimensions();
	//cout << dimn << endl;
	//cout << "get dim out" << endl;
	int oh = dimn.Get(1, 1);
	int ow = dimn.Get(1, 2);
	//cout << "output h:" << oh << endl;
	//cout << "output w:" << ow << endl;
	//std::ofstream testfile("d:/test.txt", std::ios::out);
	cv::Mat dst = cv::Mat::zeros(oh, ow, CV_8UC1);
	uchar* pdata = dst.data;
	for (int j(0); j < oh; ++j)
	{
		//uchar* pdata = dst.ptr<uchar>(j);
		for (int i(0); i < ow; ++i)
		{
			//double a = ;
			dst.at<uchar>(j, i) = ((double)outMat(j + 1, i + 1) > threshold) ? 255 : 0; /// 元素访问（行号，列号）
			//testfile << outMat(j + 1, i + 1) << " ";
		}
		//testfile << endl;
	}
	//testfile.flush();
	//testfile.close();
	dstMat = dst;
	return 1;
}

int WaveletTransformer::globalInitial()
{
	if (!mclInitializeApplication(NULL, 0))
	{
		return 0;
	}
	if (!wavelet_coefInitialize())
	{
		return 0;
	}
	return 1;
}

WaveletTransformer::~WaveletTransformer()
{
	wavelet_coefTerminate();
}

int WaveletTransformer::initial()
{
	if (!mclInitializeApplication(NULL, 0))
	{
		return 0;
	}
	if (!wavelet_coefInitialize())
	{
		return 0;
	}
	return 1;
}
