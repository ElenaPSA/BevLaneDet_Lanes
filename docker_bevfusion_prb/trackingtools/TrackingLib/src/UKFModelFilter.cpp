
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


#include "UKFModelFilter.hpp"




double Gaussianpdf(const Mat& x, const Mat& mean, const Mat& sigma)
{
	double n = x.rows;
	double sqrt2pi = std::sqrt(2 * M_PI);
	double quadform = ((cv::Mat)((x - mean).t() * sigma.inv(cv::DECOMP_CHOLESKY) * (x - mean))).at<double>(0);

	double norm = std::pow(sqrt2pi, -n) *
		std::pow(cv::determinant(sigma), -0.5);

	return max(0.00000000000000001, norm * exp(-0.5 * quadform));
}



UKFModelFilter::UKFModelFilter(int numStates, int numObs, int numStatesAug, int numObsAug, double alpha_, double beta_, double k_) :
	m_numStatesAug(numStatesAug),
	m_numObsAug(numObsAug),
	m_likelihood(0.0),
	m_numStates(numStates),
	m_numObs(numObs)

{

	m_numStatesTotal = numStates + numStatesAug + numObsAug;

	m_stateAug = Mat::zeros(m_numStatesTotal, 1, CV_64F);
	m_state = m_stateAug(Rect(0, 0, 1, numStates));

	m_errorCovAug = Mat::zeros(m_numStatesTotal, m_numStatesTotal, CV_64F);
	m_errorCov = m_errorCovAug(Rect(0, 0, numStates, numStates));

	m_processNoiseCov = Mat::zeros(numStates, numStates, CV_64F);
	m_measurementNoiseCov = Mat::zeros(numObs, numObs, CV_64F);

	m_alpha = alpha_;
	m_beta = beta_;
	m_k = k_;

	m_measurementEstimate = Mat::zeros(numObs, 1, CV_64F);

	m_q = Mat::zeros(numStates, 1, CV_64F);
	m_r = Mat::zeros(numObs, 1, CV_64F);

	m_gain = Mat::zeros(numStates, numStates, CV_64F);

	m_predVals = Mat::zeros(numStates, 2 * m_numStatesTotal + 1, CV_64F);
	m_measureVals = Mat::zeros(numObs, 2 * m_numStatesTotal + 1, CV_64F);

	m_predValsCenter = Mat::zeros(numStates, 2 * m_numStatesTotal + 1, CV_64F);
	m_measureValsCenter = Mat::zeros(numObs, 2 * m_numStatesTotal + 1, CV_64F);

	m_lambda = m_alpha * m_alpha*(m_numStatesTotal + m_k) - m_numStatesTotal;
	m_tmpLambda = m_lambda + m_numStatesTotal;

	double tmp2Lambda = 0.5 / m_tmpLambda;

	m_Wm = tmp2Lambda * Mat::ones(2 * m_numStatesTotal + 1, 1, CV_64F);
	m_Wc = tmp2Lambda * Mat::eye(2 * m_numStatesTotal + 1, 2 * m_numStatesTotal + 1, CV_64F);

	m_Wm.at<double>(0, 0) = m_lambda / m_tmpLambda;
	m_Wc.at<double>(0, 0) = m_lambda / m_tmpLambda + 1.0 - m_alpha * m_alpha + m_beta;

	// Variables used for the smoother
	m_valsCenter = Mat::zeros(numStates, 2 * m_numStatesTotal + 1, CV_64F);

};
UKFModelFilter::UKFModelFilter(const UKFModelFilter& ukf)
	:m_numStatesAug(ukf.m_numStatesAug),
	 m_numObsAug(ukf.m_numObsAug),
	 m_numStatesTotal(ukf.m_numStatesTotal),
	 m_numStates(ukf.m_numStates),
	 m_numObs(ukf.m_numObs)
{
	m_stateAug = ukf.m_stateAug.clone();
	m_state = m_stateAug(Rect(0, 0, 1, m_numStates));

	m_errorCovAug = ukf.m_errorCovAug.clone();
	m_errorCov = m_errorCovAug(Rect(0, 0, m_numStates, m_numStates));

	m_processNoiseCov = ukf.m_processNoiseCov.clone();
	m_measurementNoiseCov = ukf.m_measurementNoiseCov.clone();

	m_alpha = ukf.m_alpha;
	m_beta = ukf.m_beta;
	m_k = ukf.m_k;

	m_measurementEstimate = ukf.m_measurementEstimate.clone();

	m_q = ukf.m_q.clone();
	m_r = ukf.m_r.clone();

	m_gain = ukf.m_gain.clone();

	m_predVals = ukf.m_predVals.clone();
	m_measureVals = ukf.m_measureVals.clone();

	m_predValsCenter = ukf.m_predValsCenter.clone();
	m_measureValsCenter = ukf.m_measureValsCenter.clone();

	m_lambda = ukf.m_lambda;
	m_tmpLambda = ukf.m_tmpLambda;
	
	m_Wm = ukf.m_Wm.clone();
	m_Wc = ukf.m_Wc.clone();

	// Variables used for the smoother
	m_valsCenter = ukf.m_valsCenter.clone();
	
}
UKFModelFilter::~UKFModelFilter()
{

}


/* Cholesky decomposition
The function performs Cholesky decomposition <https://en.wikipedia.org/wiki/Cholesky_decomposition>.
A - the Hermitian, positive-definite matrix,
astep - size of row in A,
asize - number of cols and rows in A,
L - the lower triangular matrix, A = L*Lt.
*/
template<typename _Tp> bool
inline choleskyDecomposition(const _Tp* A, size_t astep, const int asize, _Tp* L)
{
	int i, j, k;
	double s;
	astep /= sizeof(A[0]);
	for (i = 0; i < asize; i++)

	{
		for (j = 0; j < i; j++)
		{
			s = A[i*astep + j];
			for (k = 0; k < j; k++)
				s -= L[i*astep + k] * L[j*astep + k];
			L[i*astep + j] = (_Tp)(s / L[j*astep + j]);
		}
		s = A[i*astep + i];
		for (k = 0; k < i; k++)
		{
			double t = L[i*astep + k];
			s -= t * t;
		}
		if (s < std::numeric_limits<_Tp>::epsilon())
			return false;
		L[i*astep + i] = (_Tp)(std::sqrt(s));
	}

	for (i = 0; i < asize; i++)
		for (j = i + 1; j < asize; j++)
		{
			L[i*astep + j] = 0.0;
		}

	return true;
}


template<typename _Tp> Mat getSigmaPoints(const Mat &mean, const Mat &covMatrix, double coef)
{

	int n = mean.rows;
	Mat points = repeat(mean, 1, 2 * n + 1);

	Mat covMatrixL = covMatrix.clone();
	covMatrixL.setTo(0);

	choleskyDecomposition<_Tp>(covMatrix.ptr<_Tp>(), covMatrix.step, covMatrix.rows, covMatrixL.ptr<_Tp>());
	covMatrixL = coef * covMatrixL;

	Mat p_plus = points(Rect(1, 0, n, n));
	Mat p_minus = points(Rect(n + 1, 0, n, n));

	add(p_plus, covMatrixL, p_plus);
	subtract(p_minus, covMatrixL, p_minus);

	return points;
}


Mat  UKFModelFilter::drawFromCurrentState()
{
	Mat covMatrixL = m_errorCov.clone();
	covMatrixL.setTo(0);
	cv::Mat state = m_state.clone();
	choleskyDecomposition<double>(m_errorCov.ptr<double>(), m_errorCov.step, m_errorCov.rows, covMatrixL.ptr<double>());

	cv::randn(state, cv::Scalar::all(0), cv::Scalar::all(1.0));

	state = m_state + covMatrixL * state;
	return state;
}


void UKFModelFilter::predict(double dt, bool ego_static)
{
	if (dt <= 0.0)
		return;

	updateErrorMatrix(dt,ego_static);

	m_sigmaPoints = getSigmaPoints<double>(m_stateAug, m_errorCovAug, sqrt(m_tmpLambda));

	Mat x, fx;

	for (int i = 0; i < 2 * m_numStatesTotal + 1; i++)
	{
		x = m_sigmaPoints(Rect(i, 0, 1, m_numStates));
		fx = m_predVals(Rect(i, 0, 1, m_numStates));

		if (m_numStatesAug != 0)
			m_q = m_sigmaPoints(Rect(i, m_numStates, 1, m_numStatesAug));
		cv::Mat control=cv::Mat();
		stateConversionFunction(x, control, m_q, fx, dt);
	}

	// Cross covariance data computation 
	x = m_sigmaPoints(Rect(0, 0, 2 * m_numStatesTotal + 1, m_numStates));
	subtract(x, repeat(m_state, 1, 2 * m_numStatesTotal + 1), m_valsCenter);
	////////

	m_state = m_predVals * m_Wm;

	subtract(m_predVals, repeat(m_state, 1, 2 * m_numStatesTotal + 1), m_predValsCenter);

	m_errorCov = m_predValsCenter * m_Wc * m_predValsCenter.t();
	m_errorCov += m_processNoiseCov;

	// Cross covariance computation between xk and xk+1 for the smoother
	m_errorCrossCov = m_valsCenter * m_Wc * m_predValsCenter.t();
	////////

	
}


bool UKFModelFilter::correct(const Mat& measurement)
{
	correctCovarianceMatrix();
	Mat measurementCorrected = updateMeasureMatrix(measurement);

	
	m_sigmaPoints = getSigmaPoints<double>(m_stateAug, m_errorCovAug, sqrt(m_tmpLambda));
	
	Mat x, hx;

	for (int i = 0; i < 2 * m_numStatesTotal + 1; i++)
	{
		x = m_sigmaPoints(Rect(i, 0, 1, m_numStates));
		hx = m_measureVals(Rect(i, 0, 1, m_numObs));
		if (m_numObsAug != 0)
			m_r = m_sigmaPoints(Rect(i, m_numStates + m_numStatesAug, 1, m_numObsAug));

		measurementFunction(x, m_r, hx);
	}
	


	correctPredictedObservations(measurementCorrected);

	m_measurementEstimate = m_measureVals * m_Wm;

	subtract(m_measureVals, repeat(m_measurementEstimate, 1, 2 * m_numStatesTotal + 1), m_measureValsCenter);

	m_yyCov = m_measureValsCenter * m_Wc * m_measureValsCenter.t();

	m_yyCov += m_measurementNoiseCov;

	m_xyCov = m_predValsCenter * m_Wc * m_measureValsCenter.t();

	cv::Mat invCov = m_yyCov.inv(DECOMP_CHOLESKY);

	// compute the Kalman gain matrix
	// K = Sxy * Syy^(-1)
	m_gain = m_xyCov * invCov;

	// compute the corrected estimate of state
	// x* = x* + K*(y - y*), y - current measurement
	m_state = m_state + m_gain * (measurementCorrected - m_measurementEstimate);

	// compute the corrected estimate of the state cross-covariance matrix
	// P = P - K*Sxy.t
	m_errorCov = m_errorCov - m_gain * m_xyCov.t();

	
	m_likelihood = Gaussianpdf(measurementCorrected, m_measurementEstimate, m_yyCov);
	
	return true;
}


double UKFModelFilter::getMalahanobis(const Mat& measurement)
{
	correctCovarianceMatrix();
	Mat measurementCorrected = updateMeasureMatrix(measurement);
	
	m_sigmaPoints = getSigmaPoints<double>(m_stateAug, m_errorCovAug, sqrt(m_tmpLambda));

	Mat x, hx;

	for (int i = 0; i < 2 * m_numStatesTotal + 1; i++)
	{
		x = m_sigmaPoints(Rect(i, 0, 1, m_numStates));

		hx = m_measureVals(Rect(i, 0, 1, m_numObs));

		if (m_numObsAug != 0)
			m_r = m_sigmaPoints(Rect(i, m_numStates + m_numStatesAug, 1, m_numObsAug));

		measurementFunction(x, m_r, hx);
	}
	
	correctPredictedObservations(measurementCorrected);
	m_measurementEstimate = m_measureVals * m_Wm;
	subtract(m_measureVals, repeat(m_measurementEstimate, 1, 2 * m_numStatesTotal + 1), m_measureValsCenter);

	m_yyCov = m_measureValsCenter * m_Wc * m_measureValsCenter.t();

	if (m_numObsAug == 0)
	{
		m_yyCov += m_measurementNoiseCov;
	}
	
	cv::Mat invCov = m_yyCov.inv(DECOMP_CHOLESKY);

	cv::Mat X = (measurement - m_measurementEstimate);
	
//	std::cout << "ukf " << (t2 - t1) << "  " << (t3 - t2) << " " << (t4 - t3)  << " "<< (t5 - t4)<<std::endl;
	double distMalhahanobis = (cv::Mat(X.t()*invCov*X)).at<double>(0);

	return distMalhahanobis;
}

