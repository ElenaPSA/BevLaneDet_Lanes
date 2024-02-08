#include "IMMFilters.hpp"

IMMFilter::IMMFilter(const IMMFilter& other):m_sizeFilter(3,3,0,CV_64F)
{

	m_sizeFilter.statePre = other.m_sizeFilter.statePre.clone();
	m_sizeFilter.statePost = other.m_sizeFilter.statePost.clone();
	m_sizeFilter.transitionMatrix = other.m_sizeFilter.transitionMatrix.clone();
	m_sizeFilter.controlMatrix = other.m_sizeFilter.controlMatrix.clone();
	m_sizeFilter.measurementMatrix = other.m_sizeFilter.measurementMatrix.clone();
	m_sizeFilter.processNoiseCov = other.m_sizeFilter.processNoiseCov.clone();
	m_sizeFilter.measurementNoiseCov = other.m_sizeFilter.measurementNoiseCov.clone();
	m_sizeFilter.errorCovPre = other.m_sizeFilter.errorCovPre.clone();
	m_sizeFilter.gain = other.m_sizeFilter.gain.clone();
	m_sizeFilter.errorCovPost = other.m_sizeFilter.errorCovPost.clone();


	m_filters.resize(other.m_filters.size());

	for (size_t i = 0; i < m_filters.size(); i++)
	{
		m_filters[i] = other.m_filters[i]->clone();
	}
	m_modelProbabilities = other.m_modelProbabilities.clone();
	m_transitionMatrix = other.m_transitionMatrix.clone();
	m_state = other.m_state.clone();
	m_errorCov = other.m_errorCov.clone();
	m_likelihood = other.m_likelihood.clone();
	m_omega = other.m_omega.clone();
	m_cbar = other.m_cbar.clone();
	m_mixedstates.resize(other.m_mixedstates.size());
	m_mixederrorCov.resize(other.m_mixederrorCov.size());
	for (size_t i = 0; i < m_mixedstates.size(); i++)
	{
		m_mixedstates[i] = other.m_mixedstates[i].clone();
	}

	for (size_t i = 0; i < m_mixederrorCov.size(); i++)
	{
		m_mixederrorCov[i] = other.m_mixederrorCov[i].clone();
	}
	
}
IMMFilter::IMMFilter():m_sizeFilter(3, 3, 0, CV_64F)
{

	m_sizeFilter.transitionMatrix = (Mat_<double>(3, 3) << 1, 0, 0, 0,1,0,0,0,1);
	setIdentity(m_sizeFilter.measurementMatrix);
	setIdentity(m_sizeFilter.processNoiseCov, Scalar::all(g_varianceSize));
	setIdentity(m_sizeFilter.measurementNoiseCov, Scalar::all(g_varianceSizeObs));
	setIdentity(m_sizeFilter.errorCovPost, Scalar::all(g_varianceSizeObs));
}

IMMFilter::IMMFilter(std::vector<UKFModelFilter*>& filters, cv::Mat& initialProbabilities, cv::Mat& transitionMatrix):m_sizeFilter(3, 3, 0, CV_64F)
{
	m_sizeFilter.transitionMatrix = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	setIdentity(m_sizeFilter.measurementMatrix);
	setIdentity(m_sizeFilter.processNoiseCov, Scalar::all(g_varianceSize));
	setIdentity(m_sizeFilter.measurementNoiseCov, Scalar::all(g_varianceSizeObs));
	setIdentity(m_sizeFilter.errorCovPost, Scalar::all(g_varianceSizeObs));
	initFilters(filters, initialProbabilities, transitionMatrix);
}


void IMMFilter::initFilters(std::vector<UKFModelFilter*>& filters, cv::Mat& initialProbabilities, cv::Mat& transitionMatrix)
{
	m_filters = filters;
	m_modelProbabilities = initialProbabilities;
	m_transitionMatrix = transitionMatrix;
	m_state = Mat(filters[0]->m_state.rows, 1, CV_64F, cv::Scalar(0.0));
	m_errorCov = Mat(filters[0]->m_errorCov.rows, filters[0]->m_errorCov.cols, CV_64F, cv::Scalar(0.0));
	m_likelihood = Mat((int)filters.size(), 1, CV_64F, cv::Scalar(0.0));
	m_omega = Mat((int)filters.size(), (int)filters.size(), CV_64F, cv::Scalar(0.0));
	m_cbar = Mat((int)filters.size(), 1, CV_64F, cv::Scalar(0.0));
	m_mixedstates.resize((int)filters.size());
	m_mixederrorCov.resize((int)filters.size());

	for (unsigned int i = 0; i < filters.size(); i++)
	{
		m_mixedstates[i] = cv::Mat(filters[0]->m_state.rows, 1, CV_64F, cv::Scalar(0.0));
		m_mixederrorCov[i] = cv::Mat(filters[0]->m_errorCov.rows, filters[0]->m_errorCov.cols, CV_64F, cv::Scalar(0.0));
	}

	compute_mixing_probabilities();
	compute_state_estimate();
}


void IMMFilter::compute_mixing_probabilities()
{
	for (unsigned int j = 0; j < m_filters.size(); j++)
	{
		m_cbar.at<double>(j) = 0.0;
		for (unsigned int i = 0; i < m_filters.size(); i++)
		{
			m_cbar.at<double>(j) += m_modelProbabilities.at<double>(i)*m_transitionMatrix.at<double>(i, j);
		}

	}

	for (unsigned int j = 0; j < m_filters.size(); j++)
	{
		for (unsigned int i = 0; i < m_filters.size(); i++)
		{
			m_omega.at<double>(i, j) = m_transitionMatrix.at<double>(i, j)* m_modelProbabilities.at<double>(i) / m_cbar.at<double>(j);
		}
	}
}

void IMMFilter::compute_state_estimate()
{
	m_state.setTo(0.0);

	for (unsigned int i = 0; i < m_filters.size(); i++)
	{
		m_state += m_filters[i]->m_state*m_modelProbabilities.at<double>(i);
	}

	m_errorCov.setTo(0.0);

	for (unsigned int i = 0; i < m_filters.size(); i++)
	{
		cv::Mat error = m_filters[i]->m_state - m_state;
		
		m_errorCov += (m_filters[i]->m_errorCov + error * error.t())*m_modelProbabilities.at<double>(i);
	}
}


void IMMFilter::predict(double dt,bool ego_static)
{
	compute_mixing_probabilities();

	for (unsigned int j = 0; j < m_filters.size(); j++)
	{
		m_mixedstates[j].setTo(cv::Scalar(0.0));
		for (unsigned int i = 0; i < m_filters.size(); i++)
		{
			m_mixedstates[j] += m_omega.at<double>(i, j)*m_filters[i]->m_state;
		}
	}

	for (unsigned int j = 0; j < m_filters.size(); j++)
	{
		m_mixederrorCov[j].setTo(cv::Scalar(0.0));
		for (unsigned int i = 0; i < m_filters.size(); i++)
		{
			cv::Mat error = m_filters[i]->m_state - m_mixedstates[j];
			m_mixederrorCov[j] += (m_filters[i]->m_errorCov + error * error.t())*m_omega.at<double>(i, j);
		}
	}

	for (unsigned int i = 0; i < m_filters.size(); i++)
	{
		m_mixedstates[i].copyTo(m_filters[i]->m_state);
		m_mixederrorCov[i].copyTo(m_filters[i]->m_errorCov);

		m_filters[i]->predict(dt,ego_static);
	}
	compute_state_estimate();

	Mat prediction = m_sizeFilter.predict();
}

void IMMFilter::predictInEgoFrame(double dt, double dx, double dy, double dyaw)
{
	bool ego_static=(dx==0.0);
	
	predict(dt,ego_static);

	for (auto& filter : m_filters)
		filter->correctPose(dx,dy,dyaw);
	compute_state_estimate();
}
void IMMFilter::getEllipseConfidenceOnPosition(Point2d& vec, double& alpha) const
{
	cv::Mat X = (cv::Mat_<double>(2, 2) << m_errorCov.at<double>(SX, SX), m_errorCov.at<double>(SX, SY), m_errorCov.at<double>(SY, SX), m_errorCov.at<double>(SY, SY));
	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(X, eigenvalues, eigenvectors);

	vec.x = sqrtl(eigenvalues.at<double>(0)*9.21);
	vec.y = sqrtl(eigenvalues.at<double>(1)*9.21);

	alpha = atan2(eigenvectors.at<double>(0, 1), eigenvectors.at<double>(0, 0));
	//std::cout << "eigne vector" << eigenvectors.at<double>(0, 0) << " " << eigenvectors.at<double>(0, 1) << std::endl;
}


void IMMFilter::getEstimation(TrackOutput& output)
{
	output.theta = m_state.at<double>(STHETA);
	output.V = m_state.at<double>(SV);
	output.x = m_state.at<double>(SX);
	output.y = m_state.at<double>(SY);
	output.z = m_state.at<double>(SZ);
	output.yawRate= m_state.at<double>(STHETA_DOT);
	output.width =  m_sizeFilter.statePost.at<double>(0);
	output.height = m_sizeFilter.statePost.at<double>(1);
	output.length = m_sizeFilter.statePost.at<double>(2);


/*	if (output.V < 0)
	{
		output.V = -output.V;
		output.theta = output.theta + M_PI;
	}
	*/
	memcpy(&output.errorCov[0],m_errorCov.data, NUMSTATES*NUMSTATES*sizeof(double));
	output.errorCovSize[0] = m_sizeFilter.errorCovPost.at<double>(0,0);
	output.errorCovSize[1] = m_sizeFilter.errorCovPost.at<double>(1,1);
	output.errorCovSize[2] = m_sizeFilter.errorCovPost.at<double>(2,2);
}



double  modulo(double a, double b)
{
	return a - std::floor(a / b)*b;
}

double mod_angle(double theta, double thetar)
{
	double theta_d1 = modulo(theta - thetar + M_PI, 2 * M_PI) - M_PI;
	double theta_d2 = modulo(theta+M_PI - thetar + M_PI, 2 * M_PI) - M_PI;
	if(fabs(theta_d1)>fabs(theta_d2))
		return thetar + theta_d2;
	else
		return thetar + theta_d1;
}



CVIMMFilterBev::CVIMMFilterBev(
	double x, double y, double z, double theta,
	double width, double height, double length,
	double varX,double varY,  double varZ,
	double varTheta,
	double varAccelCV, double varAccelCT, double varYawRateCV, double varYawAccelCT)
	:IMMFilter(), m_varX(varX), m_varY(varY), m_varZ(varZ), m_varTheta(varTheta)
{
	cv::Mat modelProbabilities = (cv::Mat_<double>(3, 1) << 0.8, 0.1, 0.1);
	cv::Mat transitionMatrix = (cv::Mat_<double>(3, 3) << 0.996, 0.002, 0.002,
		0.01, 0.98, 0.01,
	 	0.05, 0.01, 0.94);

	// cv::Mat modelProbabilities = (cv::Mat_<double>(2, 1) << 0.7, 0.3);
	// cv::Mat transitionMatrix = (cv::Mat_<double>(2, 2) << 0.996, 0.004,
	// 													  0.02, 0.98);
	ObservationBev obsInit;
	obsInit.x = x;
	obsInit.y = y;
	obsInit.z = z;
	obsInit.theta = theta;

	obsInit.theta = atan2(sin(theta), cos(theta));

	m_sizeFilter.statePre.at<double>(OWR) = m_sizeFilter.statePost.at<double>(OWR) = width;
	m_sizeFilter.statePre.at<double>(OHR) = m_sizeFilter.statePost.at<double>(OHR) = height;
	m_sizeFilter.statePre.at<double>(OLR) = m_sizeFilter.statePost.at<double>(OLR) = length;
	//TODO: make transition matrix function of dt : use frequency of change and dt to update 
	m_filters.resize(3);

	
	m_filters[0] = new ConstantSpeedModelBev(obsInit, m_varX, m_varY, m_varZ, m_varTheta, varAccelCV, varYawRateCV);
	m_filters[1] = new ConstTurnRateConstantSpeedModelBev(obsInit, m_varX, m_varY, m_varZ, m_varTheta, varAccelCT, varYawAccelCT);
	m_filters[2] = new ConstantPositionModelBev(obsInit, m_varX, m_varY, m_varZ, m_varTheta);
	

	initFilters(m_filters, modelProbabilities, transitionMatrix);
};



CVIMMFilterBev::CVIMMFilterBev(cv::Mat& measurement)
	:CVIMMFilterBev( measurement.at<double>(OBS_BEV::OX_BEV), measurement.at<double>(OBS_BEV::OY_BEV), measurement.at<double>(OBS_BEV::OZ_BEV), measurement.at<double>(OBS_BEV::OTHETA_BEV), 
		measurement.at<double>(NUMOBSBEV + OWR), measurement.at<double>(NUMOBSBEV + OHR),measurement.at<double>(NUMOBSBEV + OLR))
{

};
CVIMMFilterBev::CVIMMFilterBev(const CVIMMFilterBev& other)
	:IMMFilter(other),
	m_varX(other.m_varX),
	m_varY(other.m_varY),
	m_varZ(other.m_varZ),
	m_varTheta(other.m_varTheta)
{

}

void CVIMMFilterBev::correct(const std::vector<double>& measurements)
{
	
	cv::Mat_<double> measurement_corrected(NUMOBSBEV, 1);

	measurement_corrected.at<double>(OBS_BEV::OX_BEV) = measurements[OBS_BEV::OX_BEV];
	measurement_corrected.at<double>(OBS_BEV::OY_BEV) = measurements[OBS_BEV::OY_BEV];
	measurement_corrected.at<double>(OBS_BEV::OZ_BEV) = measurements[OBS_BEV::OZ_BEV];
	measurement_corrected.at<double>(OBS_BEV::OTHETA_BEV) = measurements[OBS_BEV::OTHETA_BEV];

	double theta = measurements[OBS_BEV::OTHETA_BEV];
	double theta_ref = m_state.at<double>(STHETA);

	
	theta = mod_angle(theta, theta_ref);
	measurement_corrected.at<double>(OBS_BEV::OTHETA_BEV) = theta;

	//std::cout << measurement_corrected << std::endl;
	for (unsigned int i = 0; i < m_filters.size(); i++)
	{
		m_filters[i]->correct(measurement_corrected);
		m_likelihood.at<double>(i) = m_filters[i]->m_likelihood;
		
		m_modelProbabilities.at<double>(i) = m_cbar.at<double>(i)*m_likelihood.at<double>(i);
	}

	m_modelProbabilities /= cv::sum(m_modelProbabilities).val[0];
	//std::cout << "Probabilities : " << m_modelProbabilities << std::endl;
	//	std::cout << "Distance : " << md << std::endl;

	compute_state_estimate();

	cv::Mat_<double> sizeMeasurement(3, 1);
	sizeMeasurement.at<double>(0) = measurements[NUMOBSBEV];
	sizeMeasurement.at<double>(1) = measurements[NUMOBSBEV+1];
	sizeMeasurement.at<double>(2) = measurements[NUMOBSBEV+2];
	m_sizeFilter.correct(sizeMeasurement);

}


double CVIMMFilterBev::fastCheck(const Mat& measurement, const Mat& errorMatrix)
{
	double x = m_state.at<double>(SX);
	double y = m_state.at<double>(SY);
	double z = m_state.at<double>(SZ);

	

	double xobs = measurement.at<double>(OBS_BEV::OX_BEV);
	double yobs = measurement.at<double>(OBS_BEV::OY_BEV);
	
	return (x - xobs)*(x - xobs) < m_varX * 16 && (yobs - y)*(yobs - y) < m_varY * 16;
}
double CVIMMFilterBev::mahalanobisDistance(const Mat& measurement, const Mat& errorMatrix)
{

	cv::Mat_<double> measurement_corrected(NUMOBSBEV, 1);

	measurement_corrected.at<double>(OBS_BEV::OX_BEV) = measurement.at<double>(OBS_BEV::OX_BEV);
	measurement_corrected.at<double>(OBS_BEV::OY_BEV) = measurement.at<double>(OBS_BEV::OY_BEV);
	measurement_corrected.at<double>(OBS_BEV::OZ_BEV) = measurement.at<double>(OBS_BEV::OZ_BEV);
	measurement_corrected.at<double>(OBS_BEV::OTHETA_BEV) = measurement.at<double>(OBS_BEV::OTHETA_BEV);

	double theta = measurement.at<double>(OBS_BEV::OTHETA_BEV);
	double theta_ref = m_state.at<double>(STHETA);


	theta = mod_angle(theta, theta_ref);
	measurement_corrected.at<double>(OBS_BEV::OTHETA_BEV) = theta;

	std::vector<double> distanceMalhahanobis(m_filters.size());

	for (int i = 0; i < m_filters.size(); i++)
	{
		distanceMalhahanobis[i] = m_filters[i]->getMalahanobis(measurement_corrected);
	}

	return *std::min_element(distanceMalhahanobis.begin(), distanceMalhahanobis.end());

}
