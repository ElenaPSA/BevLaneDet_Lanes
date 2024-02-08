
#ifndef TRACKING_FILTER_BEV_HPP
#define TRACKING_FILTER_BEV_HPP

#include "UKFModelFilter.hpp"
#define NOMINMAX
#include <opencv2/opencv.hpp>

#include "globalDefinitions.hpp"
#include <vector>

#define _USE_MATH_DEFINES
#include <math.h>
using namespace cv;




class FilterModelBev : public UKFModelFilter
{

public:

	FilterModelBev(ObservationBev initObs, int numStatesAug, int numObsAug,
					double varX = g_variance_BevX,
					double varY = g_variance_BevY,
					double varZ = g_variance_BevZ,
					double varTheta=g_variance_BevTheta
					);
	FilterModelBev(const FilterModelBev& other);

	virtual FilterModelBev* clone() const = 0;

	bool isvalid() const { return fabs(m_state.at<double>(SV)) < g_maxSpeed; }
		
	cv::Point2d getEstimation(){ return cv::Point2d(m_state.at<double>(SX), m_state.at<double>(SY)); }

	virtual void getEstimation(TrackOutput& output);
	
	void correctPose(double dx, double dy, double dyaw);
	
	

	//virtual bool correct(const Mat& measurement);

protected:
	virtual bool checkMalahanobisDistance(double distMalhahanobis) { return true; }
	virtual Mat updateMeasureMatrix(const Mat& measurement);
	virtual void correctCovarianceMatrix() {};
	virtual void stateConversionFunction(const Mat& x_k, const Mat& u_k, const Mat& v_k, Mat& x_kplus1, double dt) = 0;
	virtual void updateErrorMatrix(double dt,bool ego_static) {};
	virtual void measurementFunction(const Mat& x_k, const Mat& n_k, Mat& z_k) ;
	void correctPredictedObservations(Mat& measurement);
protected:
	
	double m_varX, m_varY, m_varZ, m_varTheta;	
	

};

class ConstantPositionModelBev : public FilterModelBev
{

public:
	ConstantPositionModelBev(const  ConstantPositionModelBev& other);

	ConstantPositionModelBev* clone() const
	{
		return new ConstantPositionModelBev(*this);
	};
	ConstantPositionModelBev(
		ObservationBev initObs,
		double varX = g_variance_BevX,
		double varY = g_variance_BevY,
		double varZ = g_variance_BevZ,
		double varTheta = g_variance_BevTheta);

protected:
	void stateConversionFunction(const Mat& x_k, const Mat& u_k, const Mat& v_k, Mat& x_kplus1, double dt);
	void updateErrorMatrix(double dt,bool ego_static);

};

class ConstantSpeedModelBev : public FilterModelBev
{

public:
	ConstantSpeedModelBev(const  ConstantSpeedModelBev& other);

	ConstantSpeedModelBev* clone() const
	{
		return new ConstantSpeedModelBev(*this);
	};
	ConstantSpeedModelBev(
		ObservationBev initObs,
		double varX = g_variance_BevX,
		double varY = g_variance_BevY,
		double varZ = g_variance_BevZ,
		double varTheta = g_variance_BevTheta,
		double varAccel = g_varAccelConstantVelocity,
		double varYawRate = g_varAngularRateConstantVelocity);

protected:
	void stateConversionFunction(const Mat& x_k, const Mat& u_k, const Mat& v_k, Mat& x_kplus1, double dt);
	void updateErrorMatrix(double dt,bool ego_static);
private:
	double m_varAccel;
	double m_varYawRate;
};


class ConstTurnRateConstantSpeedModelBev : public FilterModelBev
{

public:
	ConstTurnRateConstantSpeedModelBev(const  ConstTurnRateConstantSpeedModelBev& other);

	ConstTurnRateConstantSpeedModelBev* clone() const
	{
		return new ConstTurnRateConstantSpeedModelBev(*this);
	};
	ConstTurnRateConstantSpeedModelBev(
		ObservationBev initObs,
		double varX = g_variance_BevX,
		double varY = g_variance_BevY,
		double varZ = g_variance_BevZ,
		double varTheta = g_variance_BevTheta,
		double varAccel = g_varAccelConstantVelocity,
		double varYawAccel = g_varAngularRateDotConstantTurnRate);

protected:
	void stateConversionFunction(const Mat& x_k, const Mat& u_k, const Mat& v_k, Mat& x_kplus1, double dt);
	void updateErrorMatrix(double dt,bool ego_static);

private:
	double m_varAccel;
	double m_varYawAccel;
};


#endif // TRACKING_FILTER_BEV_HPP