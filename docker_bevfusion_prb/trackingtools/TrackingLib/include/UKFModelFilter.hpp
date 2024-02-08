

/*** ///////////////////////////////////////////////////////////////////////////////////////

Unscented Kalman Filter implementation based on the original OpenCV implementation
see License below
///////////////////////////////////////////////////////////////////////////////////////
***/

/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2015, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/



#ifndef _UKFMODELFILTER_HPP
#define _UKFMODELFILTER_HPP

#include <opencv2/opencv.hpp>
#include <vector>

#define _USE_MATH_DEFINES
#include <math.h>
using namespace cv;


class UKFModelFilter
{
public:


	UKFModelFilter(int numStates, int numObs, int numStatesAug, int numObsAug, double alpha_ = 0.001, double beta_ = 2.0, double k_ = 1);
	UKFModelFilter(const UKFModelFilter& ukf);
	virtual ~UKFModelFilter();

	virtual void predict(double dt,bool ego_static=false);
	virtual bool correct(const Mat& measurement);
	virtual void correctPose(double dx, double dy, double dyaw)= 0;
	virtual double getMalahanobis(const Mat& measurement);
	
	Mat drawFromCurrentState();
	virtual UKFModelFilter* clone() const = 0;
protected:
	virtual Mat updateMeasureMatrix(const Mat& measurement) { return measurement; };
	virtual void correctCovarianceMatrix() {};
	virtual void updateErrorMatrix(double dt,bool ego_static=false) {};
	virtual void stateConversionFunction(const Mat& x_k, const Mat& u_k, const Mat& v_k, Mat& x_kplus1, double dt) = 0;
	virtual void measurementFunction(const Mat& x_k, const Mat& n_k, Mat& z_k) = 0;
	virtual bool checkMalahanobisDistance(double distMalhahanobis) { return true; };
	virtual void correctPredictedObservations(Mat& measurement) {};
public:

	//State Estimation
	Mat m_state;
	Mat m_errorCov;
	Mat m_errorCrossCov;
	//Noise matrices
	Mat m_processNoiseCov;
	Mat m_measurementNoiseCov;

	//State Estimation
	Mat m_stateAug;
	Mat m_errorCovAug;

	//likelihood of the osbervations
	double m_likelihood;

	// Sigma points
	Mat m_sigmaPoints;

	// Predicted sigma points
	Mat m_predVals;



protected:
	int m_numStatesAug;
	int m_numObsAug;
	int m_numStatesTotal;

	int m_numStates;
	int m_numObs;
	Mat m_measurementEstimate;

	Mat m_gain;
	Mat m_xyCov, m_yyCov;

	// Parameters of algorithm
	double m_alpha, m_k, m_beta;
	double m_lambda, m_tmpLambda;


	Mat m_measureVals;
	Mat m_predValsCenter, m_measureValsCenter;

	Mat m_Wm, m_Wc;
	Mat m_q, m_r;

	// Used for the smoother
	Mat m_valsCenter;
};


template <typename T>
Mat getSigmaPoints(const Mat& mean, const Mat& covMatrix, double coef);
double Gaussianpdf(const Mat& x, const Mat& mean, const Mat& sigma);



#endif //_UKFMODELFILTER_HPP