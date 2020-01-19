#pragma once
#include <fftw3.h>
#include <vector>
using namespace std;
int filtcoefs(string name, vector<double> &lp1, vector<double> &hp1, vector<double> &lp2,
	vector<double> &hp2);
class Wavelet2d
{
public:
	Wavelet2d();
	fftw_plan plan_forward_inp;
	fftw_plan plan_forward_filt;
	fftw_plan plan_backward;
	unsigned int transient_size_of_fft;

	void* dwt_2d_OMP(vector<vector<double> > &, int, string, vector<double> &, vector<double> &,
		vector<int> &);
	void* dwt1_m_OMP(string wname, vector<double> &signal, vector<double> &cA, vector<double> &cD);
	double convfftm_OMP(vector<double> &, vector<double> &, vector<double> &);
	void* dwt2_OMP(string, vector<vector<double> > &, vector<vector<double> >  &,
		vector<vector<double> >  &, vector<vector<double> > &, vector<vector<double> > &);
	void downsamp_OMP(vector<double> &, int, vector<double> &);
	void per_ext(vector<double> &sig, int a);
};
