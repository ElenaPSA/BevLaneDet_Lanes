#ifndef IMMFILTERS_HPP
#define IMMFILTERS_HPP



#include "TrackingFilterBev.hpp"

class IMMFilter
{
public:
	
	typedef std::shared_ptr<IMMFilter> Ptr;
	typedef std::shared_ptr<const IMMFilter> ConstPtr;

	IMMFilter();
	IMMFilter(std::vector<UKFModelFilter*>& filters, cv::Mat& initialProbabilities, cv::Mat& transitionMatrix);
	IMMFilter(const IMMFilter& other);

	virtual ~IMMFilter()
	{
	}

	void initFilters(std::vector<UKFModelFilter*>& filters, cv::Mat& initialProbabilities, cv::Mat& transitionMatrix);

	

	virtual void predict(double dt,bool ego_static=false);
	virtual void predictInEgoFrame(double dt,double dx, double dy, double dyaw);

	virtual void correct(const std::vector<double>& measurements) =0;
	
	virtual void getEstimation(TrackOutput& output);

	
	virtual double mahalanobisDistance(const Mat& measurement, const Mat& errorMatrix)=0;

	
	virtual double fastCheck(const Mat& measurement, const Mat& errorMatrix)= 0;

	void virtual disableHeadingObservation() {};
	void virtual enableHeadingObservation() {};

	bool isValid() const
	{
		return fabs(m_state.at<double>(SV)) < g_maxSpeed;
	}
	virtual IMMFilter* clone() const = 0;
	const cv::Mat& state() { return m_state; }
	const cv::Mat& covar() { return m_errorCov; }
	const cv::Mat& transitionMatrix() { return m_transitionMatrix; }
	const cv::Mat& sizestate() { return m_sizeFilter.statePost; }
	const cv::Mat& sizecovar() { return m_sizeFilter.errorCovPost; }
	Mat getModelProbabilities() const { return m_modelProbabilities.clone(); }
	void getEllipseConfidenceOnPosition(Point2d& vec, double& alpha) const;
	int getNumFilters(){return m_filters.size();}
	std::vector<UKFModelFilter*> getFilters(){return m_filters;}

	std::vector<cv::Mat>& getMixedStates(){return m_mixedstates;}
	std::vector<cv::Mat>& getMixedCov(){return m_mixederrorCov;}
protected:

	void compute_mixing_probabilities();
	void compute_state_estimate();

protected:
	std::vector<UKFModelFilter*> m_filters;
	Mat m_modelProbabilities;
	Mat m_transitionMatrix;
	Mat m_state;
	Mat m_errorCov;
	Mat m_likelihood;
	Mat m_omega;
	Mat m_cbar;
	std::vector<cv::Mat> m_mixedstates;
	std::vector<cv::Mat> m_mixederrorCov;
	cv::KalmanFilter m_sizeFilter;
};

class  CVIMMFilterBev : public IMMFilter
{

public:
	CVIMMFilterBev(
		double x, double y, double z,
		double theta, double width, double height, double length,
		double varX = g_variance_BevX,
		double varY = g_variance_BevY,
		double varZ = g_variance_BevZ,
		double varTheta = g_variance_BevTheta,
		double varAccelCV = g_varAccelConstantVelocity,
		double varAccelCT = g_varAccelConstantVelocityConstantTurnRate,
		double varYawRateCV = g_varAngularRateConstantVelocity,
		double varYawAccelCT = g_varAngularRateDotConstantTurnRate);
	CVIMMFilterBev(const CVIMMFilterBev&);
	virtual CVIMMFilterBev* clone() const { return new CVIMMFilterBev(*this); };
	CVIMMFilterBev(cv::Mat& measurement);
	void correct(const std::vector<double>& measurements);
	double mahalanobisDistance(const Mat& measurement, const Mat& errorMatrix);
	double fastCheck(const Mat& measurement, const Mat& errorMatrix);

private:
	double m_varX, m_varY, m_varZ, m_varTheta;
};


#endif