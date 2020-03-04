#include "SeperateParcelsRecog.h"
#include "opencv2/opencv.hpp"
#include <string>
#include <vector>
using namespace std;
using namespace cv;


void main()
{
	//检测类，用于初始化
	ParcelsRecog parRec;
	parRec.initial_model();


	Mat srcm = imread("image_file.jpg");

	int margin_x = 50;
	int margin_y = 50;

	//裁剪掉边缘影响判断的图像
	Rect croprect(margin_x, margin_y, srcm.cols - margin_x * 2, srcm.rows - margin_y * 2);
	Mat cropm;
	cropm = srcm(croprect).clone();



	vector<vector<Point2f>> cordPts;
	parRec.detect_mat(cropm, cordPts);
	cout << cordPts.size() << endl;

	//结果补偿
	vector<vector<Point>> cordPtsInt;
	for (int i = 0; i < cordPts.size(); i++)
	{
		vector<Point> pts;
		for (int j = 0; j < cordPts[i].size(); j++)
		{
			pts.push_back(cordPts[i][j] + Point2f(margin_x, margin_y));
		}
		cordPtsInt.push_back(pts);
	}

	for (int i = 0; i < cordPts.size(); i++)
	{
		drawContours(srcm, cordPtsInt, -1, Scalar(0, 0, 255), 1.5);
	}

	imshow("srcm", srcm);
	waitKey(0);
}

