// SeperateParcelsDLL.cpp : 定义 DLL 应用程序的导出函数。
//
#include "SeperateParcelsRecog.h"
#include "opencv2/opencv.hpp"
#include <algorithm>
using namespace cv;
using namespace std;


//叉乘 pt0-pt1,  pt2-pt1
static double cross_product(Point &pt0, Point &pt1, Point &pt2)
{
	Point vec1 = pt0 - pt1;
	Point vec2 = pt2 - pt1;
	return vec1.x * vec2.y - vec1.y * vec2.x;
}

static double cross_product_v(Point &vec1, Point &vec2)
{
	return vec1.x * vec2.y - vec1.y * vec2.x;
}
static double dot_product_v(Point &vec1, Point &vec2)
{
	return vec1.x * vec2.x + vec1.y * vec2.y;
}

static double angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1 * dy2) / sqrt((dx1*dx1 + dy1 * dy1)*(dx2*dx2 + dy2 * dy2) + 1e-10);
}

static double getCountourRatio(vector<Point> contour)
{
	cv::RotatedRect rrc;
	rrc = cv::minAreaRect(contour);
	double rc_area = rrc.size.area();
	double ct_area = cv::contourArea(contour);
	return ct_area / rc_area;
}

// 线段的距离的，快速算法，精度不高
int dist_pt2pt(Point segv)
{
	Point vec_ = segv;
	vec_.x = abs(vec_.x);
	vec_.y = abs(vec_.y);
	return min(vec_.x, vec_.y) + (max(vec_.x, vec_.y) - min(vec_.x, vec_.y)) / 2; //快速距离计算,精度再10%左右
}

//拆分contour
int splitContour(vector<Point>scontour, vector<vector<Point>> &rcontours, double thresh = 1, double area_thresh = 100)
{
	int minEdgeLength = 30;//可以分割的最小边长
	int minContourArea = 10000;//可以分割的最小面积


	//少于6个点不具备可分性,面积小于一定的数字也不进行拆分，超参数。。
	if (scontour.size() < 6 || contourArea(scontour) < minContourArea)
	{
		rcontours.push_back(scontour);
		return 1;
	}

	//获得所有内折点 //计算内折点的分数
	vector<int> inner_turn_pts_ind; //内折点的索引和分数
	vector<int> inner_turn_pts_score;
	vector<double> point_angle_vec;
	vector<Point>::iterator it;
	Point pre_pt, pos_pt;
	int ind = -1;
	for (it = scontour.begin(); it != scontour.end(); it++)
	{
		ind++;
		if (it == scontour.begin())
		{
			pre_pt = *(it + scontour.size() - 1);
			pos_pt = *(it + 1);
		}
		else if (it == scontour.begin() + scontour.size() - 1)
		{
			pre_pt = *(it - 1);
			pos_pt = *scontour.begin();
		}
		else
		{
			pre_pt = *(it - 1);
			pos_pt = *(it + 1);
		}
		double cpv = (pre_pt - *it).cross((pos_pt - *it));
		if (cpv < 0)//内折点？
		{

			int score_ = max(max(dist_pt2pt(pre_pt - pos_pt), dist_pt2pt(pre_pt - *it)), dist_pt2pt(pos_pt - *it));
			//inner_turn_pts_ind.push_back(ind);
			inner_turn_pts_score.push_back(score_);
		}
		else
		{
			inner_turn_pts_score.push_back(0);
		}
		point_angle_vec.push_back(cpv);
		//cout << "point angle:" << i << "\t" << angle_v << endl;
	}


	//获得所有内折对
	int antiAngleNum = 0;
	vector<pair<int, int>> antiAngle_pairs;
	vector<int> antiAngle_pairs_score;
	int angle_max_dist = 0;
	for (int i = 0; i < point_angle_vec.size(); i++)
	{
		if (point_angle_vec[i] >= 0) continue;//不是内折点
		if (inner_turn_pts_score[i] < minEdgeLength) continue;//分数不够
		antiAngleNum += 1;
		int start_pt = i;

		int pre_pt1_ind = (i == 0) ? point_angle_vec.size() - 1 : i - 1;
		for (int j = i + 1; j < point_angle_vec.size(); j++)
		{
			if (point_angle_vec[j] < 0 && (j - i) >= 3 && (point_angle_vec.size() - j + i) >= 3)
			{
				int pre_pt2_ind = j - 1;
				int post_pt1_ind = i + 1;
				int post_pt2_ind = (j == point_angle_vec.size() - 1) ? 0 : j + 1;
				Point Cent1Pt = (scontour[pre_pt1_ind] + scontour[post_pt1_ind]) / 2;
				Point Cent2Pt = (scontour[pre_pt2_ind] + scontour[post_pt2_ind]) / 2;
				if (inner_turn_pts_score[j] < minEdgeLength) continue;//分数不够
				if (((Cent1Pt - scontour[i]).ddot(Cent2Pt - scontour[j])) > 0)//如果不是对角
				{
					continue;
				}
				double cc_dist = dist_pt2pt(Cent1Pt - Cent2Pt);
				double pp_dist = dist_pt2pt(scontour[i] - scontour[j]);
				if (pp_dist / cc_dist > thresh)//分割不达标不予分割
				{
					continue;
				}
				antiAngle_pairs.push_back(std::pair<int, int>(i, j));
				antiAngle_pairs_score.push_back(inner_turn_pts_score[i] + inner_turn_pts_score[j] - pp_dist);
			}
		}
	}
	//如果没有找到返回
	if (antiAngle_pairs.size() == 0)
	{
		rcontours.push_back(scontour);
		return 1;
	}
	//获得最高分索引
	vector<int>::iterator iit = std::max_element(antiAngle_pairs_score.begin(), antiAngle_pairs_score.end());
	int ipos = std::distance(antiAngle_pairs_score.begin(), iit);


	//获得距离最近的两个内折点
	//int min_dist_pair_index = 0;
	//if (antiAngle_pairs.size()>=2)
	//{
	//	int min_dist = -1;
	//	for (int i = 0; i < antiAngle_pairs.size(); i++)
	//	{
	//		//cout << "pairs:" << antiAngle_pairs[i].first << "-" << antiAngle_pairs[i].second << endl;
	//		Point vec_ = scontour[antiAngle_pairs[i].first] - scontour[antiAngle_pairs[i].second];
	//		int _dist = dist_pt2pt(vec_); //快速距离计算
	//		if (min_dist< 0 || min_dist>_dist)
	//		{
	//			min_dist = _dist;
	//			min_dist_pair_index = i;
	//		}
	//	}
	//}




	//检查轮廓是否符合切割要求
	//int pt1_ind = antiAngle_pairs[min_dist_pair_index].first;
	//int pt2_ind = antiAngle_pairs[min_dist_pair_index].second;
	//int pre_pt1_ind = (pt1_ind == 0) ? scontour.size() - 1 : pt1_ind - 1;
	//int pre_pt2_ind = pt2_ind-1;
	//int post_pt1_ind = pt1_ind+1;
	//int post_pt2_ind = (pt2_ind == scontour.size() - 1) ? 0 : pt2_ind + 1;
	//Point cent_dist = (scontour[pre_pt1_ind] + scontour[post_pt1_ind]) / 2 - (scontour[pre_pt2_ind] + scontour[post_pt2_ind]) / 2;
	//Point pt_dist = scontour[pt1_ind] - scontour[pt2_ind];
	//double dist_ratio = sqrt(pow(pt_dist.x, 2) + pow(pt_dist.y, 2)) / sqrt(pow(cent_dist.x, 2) + pow(cent_dist.y, 2));
	////cout << "dist_ratio " << pt1_ind << "-" << pt2_ind << ":" << dist_ratio << endl;

	////如果最小距离不符合要求不切分
	//if (dist_ratio > thresh)
	//{
	//	rcontours.push_back(scontour);
	//	return 1;
	//}

	//切分轮廓
	vector<vector<Point>> candidate_contours;
	vector<Point>::iterator it1 = scontour.begin();
	vector<Point>::iterator it2 = scontour.begin();
	it1 = it1 + antiAngle_pairs[ipos].first;
	it2 = it2 + antiAngle_pairs[ipos].second;
	vector<Point>contour_tmp1(it1, it2);
	contour_tmp1.push_back(*it2);
	candidate_contours.push_back(contour_tmp1);
	vector<Point> contour_tmp(scontour.begin(), it1);
	contour_tmp.push_back(*it1);
	contour_tmp.insert(contour_tmp.end(), it2, scontour.end());
	candidate_contours.push_back(contour_tmp);
	//如果切分后存在较小轮廓，则不切分
	//if (contourArea(candidate_contours[0])<area_thresh || contourArea(candidate_contours[1]) < area_thresh)
	//{
	//	rcontours.push_back(scontour);
	//	return 1;
	//}
	while (true)
	{
		int res = splitContour(candidate_contours[0], rcontours, thresh);
		if (res == 1)
		{
			break;
		}
	}
	while (true)
	{
		int res = splitContour(candidate_contours[1], rcontours, thresh);
		if (res == 1)
		{
			break;
		}
	}
	return 1;
}

//求两个rect 相交的面积比上最小rect的面积
double IoMin(RotatedRect& r1, RotatedRect&r2)
{
	vector<Point2f> interContour;
	cv::rotatedRectangleIntersection(r1, r2, interContour);
	if (interContour.size() < 3)
	{
		return 0;
	}
	double interarea = contourArea(interContour);
	return interarea / min(r1.size.area(), r2.size.area());
}

//合并两个rect为一个rect
RotatedRect fuseTwoRects(RotatedRect& r1, RotatedRect&r2)
{
	Point2f r1_pts[8];
	r1.points(r1_pts);
	r2.points(r1_pts + 4);
	vector<Point2f> contour_1(r1_pts, r1_pts + 8);
	return minAreaRect(contour_1);
}

//排序函数
bool sortRects(RotatedRect& r1, RotatedRect &r2)
{
	return r1.size.area() > r2.size.area();
}

bool sortContours(vector<Point> &c1, vector<Point> &c2)
{
	return c1.size() > c2.size();
}

//计算两个线段的交点
bool crossPoint_2Segs(Point2f Apt1, Point2f Apt2, Point2f Bpt1, Point2f Bpt2, Point2f &crossPoint)
{
	if (MIN(Apt1.x, Apt2.x) > MAX(Bpt1.x, Bpt2.x))
		return false;
	if (MIN(Apt1.y, Apt2.y) > MAX(Bpt1.y, Bpt2.y))
		return false;
	if (MIN(Bpt1.x, Bpt2.x) > MAX(Apt1.x, Apt2.x))
		return false;
	if (MIN(Bpt1.y, Bpt2.y) > MAX(Apt1.y, Apt2.y))
		return false;
	double xa = (Apt1 - Bpt1).cross(Bpt2 - Bpt1)*(Apt2 - Bpt1).cross(Bpt2 - Bpt1);
	double xb = (Bpt2 - Apt1).cross(Apt2 - Apt1)*(Bpt1 - Apt1).cross(Apt2 - Apt1);
	if (xa > 0 || xb > 0)
		return false;

	Point2f sB = Bpt2 - Bpt1;
	double d1 = abs(sB.cross(Apt1 - Bpt1));
	double  d2 = abs(sB.cross(Apt2 - Bpt1));
	double t = d1 / (d1 + d2);
	crossPoint = (Apt2 - Apt1)*t + Apt1;
	return true;

}



double dist_Rect2Rect(RotatedRect &r_big, RotatedRect &r_small)
{
	Point2f pts[8];
	r_big.points(pts);
	r_small.points(pts + 4);
	Point2f c_big = r_big.center;
	Point2f c_small = r_small.center;

	Point2f Across_pt;
	for (int i = 0; i < 4; i++)
	{
		if (true == crossPoint_2Segs(c_big, c_small, pts[i], pts[(i + 1) % 4], Across_pt))
			break;
	}



	Point2f Bcross_pt;
	for (int i = 0; i < 4; i++)
	{
		if (true == crossPoint_2Segs(c_big, c_small, pts[i + 4], pts[(i + 1) % 4 + 4], Bcross_pt))
			break;
	}
	//cout << "dist rects:" << Across_pt.x << " " << Across_pt.y << ";" << Bcross_pt.x << " " << Bcross_pt.y << endl;

	double _flag = 0;
	if (MIN(c_big.x, Bcross_pt.x) < Across_pt.x && Across_pt.x < MAX(c_big.x, Bcross_pt.x) &&
		MIN(c_big.y, Bcross_pt.y) < Across_pt.y && Across_pt.y < MAX(c_big.y, Bcross_pt.y))
	{
		_flag = 1;
	}
	else
	{
		_flag = -1;
	}
	return  dist_pt2pt(Across_pt - Bcross_pt)*_flag;

}

//辅助绘制
void drawRotateRect(Mat &drawm, RotatedRect &rtc, cv::Scalar sc, int linewidth)
{
	cv::Point2f vertices[4];
	rtc.points(vertices);
	vector<vector<Point>> contours;
	contours.push_back(vector<Point>(vertices, vertices + 4));
	drawContours(drawm, contours, 0, sc, linewidth);
}

//根据rect的尺寸和相交的面积来合并rect
void mergeRotateRects(vector<RotatedRect> &rotrecs, double threshArea, double ignoreArea, double min_dist = 50)
{
	//合并可以合并的
	sort(rotrecs.begin(), rotrecs.end(), sortRects);
	vector<RotatedRect>::iterator itpre, itpos;
	for (itpre = rotrecs.begin(); itpre != rotrecs.end(); itpre++)
	{
		for (itpos = itpre + 1; itpos != rotrecs.end();)
		{
			double ioarea = IoMin(*itpre, *itpos);
			double ioarea_fix = ioarea * (threshArea / min((float)threshArea, itpos->size.area()));
			if (ioarea_fix > 0.6)
			{
				*itpre = fuseTwoRects(*itpre, *itpos);
				itpos = rotrecs.erase(itpos);
				itpos = itpre + 1;
			}
			else
			{
				itpos++;
			}

		}
	}

	//拆分
	vector<RotatedRect> big_rects;
	vector<RotatedRect> small_rects;
	for (itpre = rotrecs.begin(); itpre != rotrecs.end(); itpre++)
	{
		if (itpre->size.area() < ignoreArea)
		{
			small_rects.push_back(*itpre);
		}
		else
		{
			big_rects.push_back(*itpre);
		}
	}

	//小的合并到大的
	/*
	for (itpos = small_rects.begin(); itpos != small_rects.end();)
	{
		double _min_dist = 100000;
		int min_dist_ind = -1;
		int n = -1;
		for (itpre = big_rects.begin(); itpre != big_rects.end(); itpre++)
		{
			n++;
			double dist_rr = dist_Rect2Rect(*itpre, *itpos);
			if (_min_dist > dist_rr)
			{
				_min_dist = dist_rr;
				min_dist_ind = n;
			}
		}
		if (min_dist_ind>=0 && _min_dist < min_dist)
		{
			big_rects[min_dist_ind] = fuseTwoRects(big_rects[min_dist_ind], *itpos);
			itpos = small_rects.erase(itpos);
			continue;
		}
		else
		{
			itpos++;
		}
	}
	*/

	for (itpre = small_rects.begin(); itpre != small_rects.end(); itpre++)
	{
		double _min_dist = 100000;
		int min_dist_ind = -1;
		int n = -1;
		for (itpos = itpre + 1; itpos != small_rects.end(); itpos++)
		{
			n++;
			double dist_rr = dist_Rect2Rect(*itpre, *itpos);
			if (_min_dist > dist_rr)
			{
				_min_dist = dist_rr;
				min_dist_ind = n;
			}
		}
		if (min_dist_ind >= 0 && _min_dist < min_dist)
		{
			*itpre = fuseTwoRects(*(itpre + 1 + min_dist_ind), *itpre);
			small_rects.erase(itpre + 1 + min_dist_ind);
		}

	}
	std::vector<RotatedRect> a;
	rotrecs.swap(a);
	rotrecs.insert(rotrecs.begin(), big_rects.begin(), big_rects.end());
	rotrecs.insert(rotrecs.begin(), small_rects.begin(), small_rects.end());

}

int getMaskFromGrayValue(Mat &srcm, vector<RotatedRect> &contour_rects, double threshArea, double ignoreArea, int threshGray)
{
	Mat BnMat;

	//阈值分割
	if (srcm.channels() == 3)
	{
		//cvtColor(srcm, graym, cv::COLOR_BGR2GRAY);
		Mat aChannels[3];
		Mat aMask[3];
		cv::split(srcm, aChannels);
		for (int i = 0; i < 3; i++)
		{
			threshold(aChannels[i], aMask[i], threshGray, 255, THRESH_BINARY);
		}

		BnMat = aMask[0] + aMask[1] + aMask[2];

	}
	else
	{
		threshold(srcm, BnMat, threshGray, 255, THRESH_BINARY);
	}

	//imshow("bnmat",BnMat);


	//查找轮廓
	Mat element = getStructuringElement(cv::MORPH_RECT, Size(3, 3));
	morphologyEx(BnMat, BnMat, MORPH_ERODE, element);
	element = getStructuringElement(cv::MORPH_RECT, Size(3, 3));
	morphologyEx(BnMat, BnMat, MORPH_DILATE, element);
	element = getStructuringElement(cv::MORPH_RECT, Size(7, 7));
	morphologyEx(BnMat, BnMat, MORPH_CLOSE, element);
	//imshow("morph", BnMat);

	vector<vector<Point>> contours;
	cv::findContours(BnMat, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	Mat drawm = cv::Mat::zeros(BnMat.size(), CV_8UC3);



	//轮廓化简
	vector<Point> approx;
	vector<vector<Point>> squares;
	int n = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() < 100)
		{
			continue;
		}

		approxPolyDP(contours[i], approx, (srcm.rows + srcm.cols)*0.015, true);
		//approxPolyDP(contours[i], approx, arcLength(contours[i], true)*0.012, true);

		squares.push_back(approx);

		//vector<vector<Point>> dwcountour;
		//dwcountour.push_back(approx);
		//cv::drawContours(drawm, contours, i, cv::Scalar(0, 100 + n * 50, 0), 1);
		//cv::drawContours(drawm, dwcountour, 0, cv::Scalar(100+ n*50, 0, 0), 1);
		//for (int j=0;j< approx.size();j++)
		//{
		//	cv::putText(drawm, to_string(j), approx[j], FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 100 + n * 50));
		//}
		//n++;

	}

	//拆分轮廓
	vector<vector<Point>> split_contours;
	for (int i = 0; i < squares.size(); i++)
	{
		//cout << "contour:" << i << endl;
		double area_r = getCountourRatio(squares[i]);
		//cout << "area ratio:" << area_r << endl;
		splitContour(squares[i], split_contours, 0.55, 400);
		//drawContours(drawm, squares, i, Scalar(0, 0, 255));
		//
	}

	//获得轮廓外界rect
	vector<RotatedRect> rotrecs;
	for (int i = 0; i < split_contours.size(); i++)
	{
		//cv::drawContours(drawm, split_contours, i, cv::Scalar(255, 255, 255));
		RotatedRect rtc;
		rtc = minAreaRect(split_contours[i]);
		rotrecs.push_back(rtc);
		//绘制
		//drawRotateRect(drawm, rtc, cv::Scalar(0, 0, 255));
	}


	//计算interArea比上最小rect的面积的比值,test

	if (rotrecs.size() > 1)
	{
		mergeRotateRects(rotrecs, threshArea, ignoreArea);
	}

	contour_rects = rotrecs;
	//合并rect



	//绘制
	//vector<RotatedRect>::iterator itpre;
	//for (itpre = rotrecs.begin(); itpre != rotrecs.end(); itpre++)
	//{
	//	drawRotateRect(drawm, *itpre, Scalar(255, 0, 0));
	//}

	/*
	//拆分大rect和小rect
	vector<RotatedRect> bigRects;
	vector<RotatedRect> smallRects;
	for (int i=0;i<rotrecs.size();i++)
	{
		if (rotrecs[i].size.area()<threshArea)
		{
			smallRects.push_back(rotrecs[i]);
		}
		else
		{
			bigRects.push_back(rotrecs[i]);
		}
	}
	cout << "big rect num:" << bigRects.size() << endl;
	cout << "small rect num" << smallRects.size() << endl;
	if (smallRects.size()==0)
	{
		contour_rects = bigRects;
	}
	vector<RotatedRect>::iterator it;
	for (int i=0;i<bigRects.size();i++)
	{
		for (it=smallRects.begin(); it!=smallRects.end();)
		{
			double ioarea = IoMin(bigRects[i], *it);
			if (ioarea>0.5)
			{
				bigRects[i] = fuseTwoRects(bigRects[i], *it);
				it = smallRects.erase(it);
			}
			else
			{
				it++;
			}

		}
	}
	*/





	//imshow("draw", drawm);

	//element = getStructuringElement(cv::MORPH_RECT, Size(50, 50));
	//morphologyEx(drawm, drawm, MORPH_BLACKHAT, element);
	//imshow("draw_open", drawm);



	return 1;

}
void closeBoundaryContour(vector<Point> &contour, Size img_size)
{
	vector<int> margin_point_inds;
	int max_pos[4] = { -1,-1,-1,-1 };
	int min_pos[4] = { -1,-1,-1,-1 };
	int max_ind[4] = { -1,-1,-1,-1 };
	int min_ind[4] = { -1,-1,-1,-1 };
	int num_pt[4] = { 0,0,0,0 };
	int last_bound = -1;

	//查找边缘点
	for (int j = 0; j < contour.size(); j++)
	{
		if (contour[j].x == 0)//分离
		{
			const int k = 0;
			num_pt[k]++;
			if (max_pos[k] < 0)
			{
				max_pos[k] = contour[j].y;
				max_ind[k] = j;
				continue;
			}
			if (min_pos[k] < 0)
			{
				if (contour[j].y > max_pos[k])
				{
					min_pos[k] = max_pos[k];
					max_pos[k] = contour[j].y;
					min_ind[k] = max_ind[k];
					max_ind[k] = j;
				}
				else
				{
					min_pos[k] = contour[j].y;
					max_ind[k] = j;
				}
				continue;
			}
			if (contour[j].y < min_pos[k])
			{
				min_pos[k] = contour[j].y;
				min_ind[k] = j;
			}
			else if (contour[j].y > max_pos[k])
			{
				max_pos[k] = contour[j].y;
				max_ind[k] = j;
			}
		}
		if (contour[j].x == img_size.width - 1)
		{
			const int k = 1;
			num_pt[k]++;
			if (max_pos[k] < 0)
			{
				max_pos[k] = contour[j].y;
				max_ind[k] = j;
				continue;
			}
			if (min_pos[k] < 0)
			{
				if (contour[j].y > max_pos[k])
				{
					min_pos[k] = max_pos[k];
					max_pos[k] = contour[j].y;
					min_ind[k] = max_ind[k];
					max_ind[k] = j;
				}
				else
				{
					min_pos[k] = contour[j].y;
					min_ind[k] = j;
				}
				continue;
			}
			if (contour[j].y < min_pos[k])
			{
				min_pos[k] = contour[j].y;
				min_ind[k] = j;
			}
			else if (contour[j].y > max_pos[k])
			{
				max_pos[k] = contour[j].y;
				max_ind[k] = j;
			}
		}
		if (contour[j].y == 0)
		{
			const int k = 2;
			num_pt[k]++;
			if (max_pos[k] < 0)
			{
				max_pos[k] = contour[j].x;
				max_ind[k] = j;
				continue;
			}
			if (min_pos[k] < 0)
			{
				if (contour[j].x > max_pos[k])
				{
					min_pos[k] = max_pos[k];
					max_pos[k] = contour[j].x;
					min_ind[k] = max_ind[k];
					max_ind[k] = j;
				}
				else
				{
					min_pos[k] = contour[j].x;
					min_ind[k] = j;
				}
				continue;
			}
			if (contour[j].x < min_pos[k])
			{
				min_pos[k] = contour[j].x;
				min_ind[k] = j;
			}
			else if (contour[j].x > max_pos[k])
			{
				max_pos[k] = contour[j].x;
				max_ind[k] = j;
			}
		}
		if (contour[j].y == img_size.height - 1)
		{
			const int k = 3;
			num_pt[k]++;
			if (max_pos[k] < 0)
			{
				max_pos[k] = contour[j].x;
				max_ind[k] = j;
				continue;
			}
			if (min_pos[k] < 0)
			{
				if (contour[j].x > max_pos[k])
				{
					min_pos[k] = max_pos[k];
					max_pos[k] = contour[j].x;
					min_ind[k] = max_ind[k];
					max_ind[k] = j;
				}
				else
				{
					min_pos[k] = contour[j].x;
					min_ind[k] = j;
				}
				continue;
			}
			if (contour[j].x < min_pos[k])
			{
				min_pos[k] = contour[j].x;
				min_ind[k] = j;
			}
			else if (contour[j].x > max_pos[k])
			{
				max_pos[k] = contour[j].x;
				max_ind[k] = j;
			}
		}
	}
	//判断边缘点
	vector<int> bondour_fix;
	for (int i = 0; i < 4; i++)
	{
		if (max_pos[i] >= 0 && min_pos[i] >= 0 && num_pt[i] > 2)
		{
			bondour_fix.push_back(i);
		}
	}
	if (bondour_fix.size() == 0)
	{
		return;
	}

	for (int j = 0; j < bondour_fix.size(); j++)
	{
		int i = bondour_fix[j];
		vector<Point> m_contour;
		if (i == 0 || i == 3)
		{
			vector<Point>::iterator min_it, max_it;
			int min_i = min_ind[i];
			int max_i = max_ind[i];
			min_it = contour.begin() + min_i;
			max_it = contour.begin() + max_i;
			if (min_i < max_i)
			{
				m_contour.insert(m_contour.end(), contour.begin(), min_it);
				m_contour.push_back(*min_it);
				m_contour.insert(m_contour.end(), max_it, contour.end());
			}
			else
			{
				m_contour.insert(m_contour.end(), max_it, min_it);
				m_contour.push_back(*min_it);
			}
			m_contour.swap(contour);

			closeBoundaryContour(contour, img_size);

			return;
		}
		if (i == 1 || i == 2)
		{
			vector<Point>::iterator min_it, max_it;
			int min_i = min_ind[i];
			int max_i = max_ind[i];
			min_it = contour.begin() + min_i;
			max_it = contour.begin() + max_i;
			if (min_i < max_i)
			{
				m_contour.insert(m_contour.end(), min_it, max_it);
				m_contour.push_back(*max_it);
			}
			else
			{
				m_contour.insert(m_contour.end(), contour.begin(), max_it);
				m_contour.push_back(*max_it);
				m_contour.insert(m_contour.end(), min_it, contour.end());
			}
			m_contour.swap(contour);

			closeBoundaryContour(contour, img_size);

			return;
		}

	}



}
int eraseSharpConer(vector<Point> &contour, Size image_size)
{
	if (contour.size() < 6) return 0;
	vector<Point>::iterator it;
	bool isShap = false;
	for (it = contour.begin(); it != contour.end(); (isShap ? it : (it++)))
	{
		isShap = false;
		if (contour.size() < 6) return 0;
		Point vec1, vec2;
		if (it == contour.begin())
		{
			vec1 = *(it + contour.size() - 1) - *it;
			vec2 = *(it + 1) - *it;
		}
		else if (it == contour.begin() + contour.size() - 1)
		{
			vec1 = *(it - 1) - *it;
			vec2 = *contour.begin() - *it;
		}
		else
		{
			vec1 = *(it - 1) - *it;
			vec2 = *(it + 1) - *it;
		}
		double cpv = vec1.cross(vec2);
		if (cpv > 0)
		{
			double cpv_dt = vec1.ddot(vec2);
			if (cpv_dt > 0)
			{
				double cosA = cpv_dt / (sqrt(pow(vec1.x, 2) + pow(vec1.y, 2))*sqrt(pow(vec2.x, 2) + pow(vec2.y, 2)));
				if (cosA > 0.75)
				{
					if (it->x == 0 || it->x == image_size.width - 1 || it->y == 0 || it->y == image_size.height - 1)
					{
						continue;
					}
					it = contour.erase(it);
					isShap = true;
					//cout << "erased a point" << endl;
					eraseSharpConer(contour, image_size);
					return 1;
				}

			}
		}

	}
	return 0;
}











void getRectsFromContourAndEdge(const Mat &edgem, vector<RotatedRect> &rtc, double threshArea, double ignoreArea)
{
	double area_thresh = 10000;//最小的邮件面积
	double length_thresh = sqrt(area_thresh);


	Mat cm = Mat::zeros(edgem.size(), CV_8U);
	Mat maskm = Mat::zeros(edgem.size(), CV_8U);
	Mat drawm = Mat::zeros(edgem.size(), CV_8UC3);
	//for (int i = 0; i < rtc.size(); i++)
	//{
	//	drawRotateRect(maskm, rtc[i], Scalar(255, 255, 255), -1);
	//}
	//imshow("maskm", maskm);
	edgem.copyTo(cm);


	Mat element = getStructuringElement(cv::MORPH_RECT, Size(3, 3));
	morphologyEx(cm, cm, MORPH_DILATE, element);
	element = getStructuringElement(cv::MORPH_RECT, Size(7, 7));
	morphologyEx(cm, cm, MORPH_CLOSE, element);

	vector<vector<Point>> edgcontours;
	findContours(cm, edgcontours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	vector<vector<Point>>::iterator it;
	for (it = edgcontours.begin(); it != edgcontours.end();)
	{
		if (it->size() < 50)
		{
			it = edgcontours.erase(it);
		}
		else
		{
			it++;
		}
	}


	for (int i = 0; i < edgcontours.size(); i++)
	{
		//drawContours(drawm, edgcontours, i, Scalar(0, 255, 0));

		closeBoundaryContour(edgcontours[i], edgem.size());


		//drawContours(drawm, edgcontours, i, Scalar(255, 255, 255));
	}




	vector<RotatedRect> contour_rects;

	for (it = edgcontours.begin(); it != edgcontours.end(); )
	{
		vector<Point> roughVec = *it;

		//cv::putText(drawm, to_string(i), roughVec[roughVec.size()/2], FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

		//approxPolyDP(edgcontours[i], roughVec, (edgem.cols+edgem.rows)*0.01, true);
		double _perimeter = arcLength(roughVec, true);
		vector<Point> contHull;
		vector<vector<Point>> contHull_defects;
		cv::convexHull(roughVec, contHull);
		//cv::convexityDefects(roughVec, contHull, contHull_defects);
		double _area = contourArea(roughVec);
		double _area_ratio = contourArea(roughVec) / contourArea(contHull);
		if (_perimeter / _area > 0.5 || _area < 10000 || _area_ratio < 0.6)
		{
			contour_rects.push_back(minAreaRect(*it));
			it = edgcontours.erase(it);
		}
		else
		{
			it++;
		}
	}



	//对边缘进行直线处理
	vector<Point> approx;
	vector<vector<Point>> fixed_contours;
	for (int i = 0; i < edgcontours.size(); i++)
	{
		approxPolyDP(edgcontours[i], approx, 15, true);
		fixed_contours.push_back(approx);
	}

	//去除轮廓中锋利的凸角
	for (int i = 0; i < fixed_contours.size(); i++)
	{
		//drawContours(drawm, fixed_contours, i, Scalar(255, 0, 0), 1);


		int res = eraseSharpConer(fixed_contours[i], edgem.size());

		//drawContours(drawm, fixed_contours, i, Scalar(0, 0, 255), 1);

	}

	//拆分轮廓
	vector<vector<Point>> splited_contours;
	for (int i = 0; i < fixed_contours.size(); i++)
	{
		splitContour(fixed_contours[i], splited_contours, 0.65);
	}

	//获得最小矩形
	for (int i = 0; i < splited_contours.size(); i++)
	{
		contour_rects.push_back(minAreaRect(splited_contours[i]));
		//drawContours(drawm, splited_contours, i, Scalar(0, 255, 0), 1);
	}
	//for (int i = 0; i < contour_rects.size(); i++)
	//{
	//	drawRotateRect(drawm, contour_rects[i], Scalar(255, 255, 0));
	//}


	//合并矩形
	mergeRotateRects(contour_rects, threshArea, ignoreArea);

	//for (int i = 0; i < contour_rects.size(); i++)
	//{

	//	drawRotateRect(drawm, contour_rects[i], Scalar(0, 255, 0));
	rtc = contour_rects;

	//}

	//rtc = contour_rects;



	//imshow("edgdrawm", drawm);

	//imshow("cm", cm);


}



int getEdge(Mat &srcm, Mat &edgem)
{
	
	if (srcm.channels() == 3)
	{
		cv::Mat channelm[3];
		split(srcm, channelm);
		cv::max(channelm[0], channelm[1], channelm[0]);
		cv::max(channelm[0], channelm[2], channelm[0]);
		srcm = channelm[2];
	}
	Mat gimg;
	medianBlur(srcm, gimg, 3);

	Mat cannymat;
	Canny(gimg, cannymat, 20, 80);
	edgem = cannymat;
	return 1;
}






ParcelsRecog::ParcelsRecog()
{

}

ParcelsRecog::~ParcelsRecog()
{

}

int ParcelsRecog::initial_model()
{
	return 1;
}

int ParcelsRecog::detect_mat(cv::Mat &pMat, std::vector<std::vector<cv::Point2f>> &outParcelsRanges)
{
	Mat srcm = pMat.clone();
	int slength = max(srcm.cols, srcm.rows);
	double scal = 720.0 / slength;
	resize(srcm, srcm, cv::Size(), scal, scal, cv::INTER_NEAREST);
	Mat edgeM;
	getEdge(srcm, edgeM);


	//imshow("edgeMat", edgeM);
	std::vector<cv::RotatedRect> rrects;
	getRectsFromContourAndEdge(edgeM, rrects, 8000, 4000);
	//cout << rrects.size() << endl;
	std::vector<cv::RotatedRect>::iterator it;
	cv::Point2f pts[4];
	for (it=rrects.begin();it!=rrects.end();it++)
	{
		it->center = it->center / scal;
		it->size.height = it->size.height / scal;
		it->size.width = it->size.width / scal;
		it->points(pts);
		vector<Point2f> spt;
		for (int j=0;j<4;j++)
		{
			spt.push_back(pts[j]);
		}
		outParcelsRanges.push_back(spt);
	}
	return outParcelsRanges.size();
}

int ParcelsRecog::detect_mat_(unsigned char *pImgData, int channels, int width, int height, double *outData, int bufferlength, int &parcel_num)
{
	cv::Mat srcm;
	if (channels==1)
	{
		srcm = cv::Mat(height, width, CV_8UC1, pImgData);
	}
	else if(channels==3)
	{
		srcm = cv::Mat(height, width, CV_8UC3, pImgData);
	}
	else
	{
		return 0;
	}
	std::vector<std::vector<cv::Point2f>> outParcelsRanges;
	detect_mat(srcm, outParcelsRanges);
	int pos = 0;
	for (int i=0;i<outParcelsRanges.size();i++)
	{
		if (bufferlength-pos<4)
		{
			parcel_num = i;
			return 0;
		}
		for (int j=0;j< outParcelsRanges[i].size();j++)
		{
			*(outData + pos) = outParcelsRanges[i][j].x;
			pos++;
			*(outData + pos) = outParcelsRanges[i][j].y;
			pos++;
		}
		parcel_num = i;
	}
	return 1;
}


