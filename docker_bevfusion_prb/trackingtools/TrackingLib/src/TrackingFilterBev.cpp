#include "TrackingFilterBev.hpp"

#define DTMIN_ODOM 0.001



FilterModelBev::FilterModelBev(const FilterModelBev& other)
	:UKFModelFilter(other),
	m_varX(other.m_varX),
	m_varY(other.m_varY),
	m_varZ(other.m_varZ),
	m_varTheta(other.m_varTheta)
	
{
	
}


FilterModelBev::FilterModelBev(
	
	ObservationBev initObs,
	int numStatesAug, 
	int numObsAug, 
	double varX, 
	double varY,
	double varZ,
	double varTheta):
	UKFModelFilter(NUMSTATES, NUMOBSBEV, numStatesAug, numObsAug),
	m_varX(varX),
	m_varY(varY),
	m_varZ(varZ),
	m_varTheta(varTheta)
{
	
	m_state.at<double>(SX) = initObs.x;
	m_state.at<double>(SY) = initObs.y;
	m_state.at<double>(SZ) = initObs.z;
	
	//std::cout <<"init theta "<<initObs.theta<< " "<< m_camera->camFrameToReferenceFrame(initObs.theta) << std::endl;
	m_state.at<double>(STHETA) =initObs.theta;
	
	m_state.at<double>(SV) = 0.0;
	m_state.at<double>(STHETA_DOT) = 0.0;

	m_errorCov.setTo(cv::Scalar(0.0));

	m_errorCov.at<double>(SX, SX) = g_initVariancePosition;
	m_errorCov.at<double>(SY, SY) = g_initVariancePosition;
	m_errorCov.at<double>(SZ, SZ) = g_initVariancePosition;

	m_errorCov.at<double>(SV, SV) = g_initVarianceSpeed;
	m_errorCov.at<double>(STHETA, STHETA) = m_varTheta;
	m_errorCov.at<double>(STHETA_DOT, STHETA_DOT) = g_initVarianceTheta;


	m_measurementNoiseCov.at<double>(OBS_BEV::OX_BEV, OBS_BEV::OX_BEV) = m_varX;
	m_measurementNoiseCov.at<double>(OBS_BEV::OY_BEV, OBS_BEV::OY_BEV) = m_varY;
	m_measurementNoiseCov.at<double>(OBS_BEV::OZ_BEV, OBS_BEV::OZ_BEV) = m_varZ;
	m_measurementNoiseCov.at<double>(OBS_BEV::OTHETA_BEV, OBS_BEV::OTHETA_BEV) = m_varTheta;	
}




void FilterModelBev::getEstimation(TrackOutput& output)
{
	output.theta = m_state.at<double>(STHETA);
	output.V = m_state.at<double>(SV);
	output.x = m_state.at<double>(SX);
	output.y = m_state.at<double>(SY);
	output.z = m_state.at<double>(SZ);
	

	if (output.V < 0)
	{
		output.V = -output.V;
		output.theta = output.theta + M_PI;
	}
}




void FilterModelBev::measurementFunction(const Mat& x_k, const Mat& n_k, Mat& z_k)
{
	double x = x_k.at<double>(SX);
	double y = x_k.at<double>(SY);
	double z = x_k.at<double>(SZ);
	double theta = x_k.at<double>(STHETA);

	
	z_k.at<double>(OBS_BEV::OX_BEV) = x;
	z_k.at<double>(OBS_BEV::OY_BEV) = y;
	z_k.at<double>(OBS_BEV::OZ_BEV) = z;
	z_k.at<double>(OBS_BEV::OTHETA_BEV) = theta;
}


Mat FilterModelBev::updateMeasureMatrix(const Mat& measurement)
{
	
	return measurement;
}

void FilterModelBev::correctPredictedObservations(Mat& measurement)
{
	for (int i = 0; i < m_measureVals.cols; i++)
	{
		double theta_obs = m_measureVals.at<double>(OBS_BEV::OTHETA_BEV, i);
		m_measureVals.at<double>(OBS_BEV::OTHETA_BEV, i) = mod_angle(theta_obs, measurement.at<double>(OBS_BEV::OTHETA_BEV));
	}
	
	
}

void FilterModelBev::correctPose(double dx, double dy, double dyaw)
{
	double c = cos(dyaw);
	double s = sin(dyaw);
	double x = m_state.at<double>(SX) - dx;
	double y = m_state.at<double>(SY) - dy;
	//std::cout<<"correct pose "<<dx<<" "<<dy<<" "<<dyaw<<std::endl;
	m_state.at<double>(STHETA) -= dyaw;

	//TODO : verify
	m_state.at<double>(SX) = x * c + y * s;
	m_state.at<double>(SY) = -x * s + y * c;

	cv::Mat G = Mat::eye(m_state.rows, m_state.rows, CV_64F);

	G.at<double>(SX, SX) = c;
	G.at<double>(SX, SY) = s;
	G.at<double>(SY, SX) = -s;
	G.at<double>(SY, SY) = c;
	cv::Mat erroCovCorrected = G * m_errorCov*G.t();
	erroCovCorrected.copyTo(m_errorCov);

};



ConstantSpeedModelBev::ConstantSpeedModelBev(
	ObservationBev initObs,
	double varX,
	double varY,
	double varZ,
	double varTheta,
	double varAccel,
	double varYawRate) :
	FilterModelBev(initObs, 0, 0, varX, varY, varZ, varTheta),
	m_varAccel(varAccel),
	m_varYawRate(varYawRate)
{

}
ConstantSpeedModelBev::ConstantSpeedModelBev(const  ConstantSpeedModelBev& other) :
	FilterModelBev(other),
	m_varAccel(other.m_varAccel),
	m_varYawRate(other.m_varYawRate)
{

}



void ConstantSpeedModelBev::stateConversionFunction(const Mat& x_k, const Mat& u_k, const Mat& v_k, Mat& x_kplus1, double dt)
{
	double x = x_k.at<double>(SX);
	double y = x_k.at<double>(SY);
	double z = x_k.at<double>(SZ);
	double theta = x_k.at<double>(STHETA);
	double theta_dot = x_k.at<double>(STHETA_DOT);
	double V = x_k.at<double>(SV);



	
	x_kplus1.at<double>(SX) = x + dt * V * cos(theta);
	x_kplus1.at<double>(SY) = y + dt * V * sin(theta);
	x_kplus1.at<double>(SZ) = z;
	x_kplus1.at<double>(STHETA) = theta;
	x_kplus1.at<double>(SV) = V;
	x_kplus1.at<double>(STHETA_DOT) = 0.0;

}

void ConstantSpeedModelBev::updateErrorMatrix(double dt,bool ego_static)
{
	double ctheta = std::cos(m_state.at<double>(STHETA));
	double stheta = std::sin(m_state.at<double>(STHETA));

	double x= m_state.at<double>(SX);
	double y = m_state.at<double>(SY);

	double d =  std::sqrt(x*x+y*y);
	double ego_theta = std::atan2(y,x);
	double scalar=1.0;
	if (ego_static)
	{
		scalar=0.0;
	}
	m_processNoiseCov.at<double>(SX, SX) = m_varAccel * (dt*dt*dt*dt / 4.0)*ctheta*ctheta + scalar*(dt*dt)*d*d*sin(ego_theta)*sin(ego_theta)*g_varianceEgoAngularRate + scalar*(dt*dt)*g_varianceEgoVelocity;
	m_processNoiseCov.at<double>(SY, SY) = m_varAccel * (dt*dt*dt*dt / 4.0)*stheta*stheta + scalar*(dt*dt)*d*d*cos(ego_theta)*cos(ego_theta)*g_varianceEgoAngularRate ;
	m_processNoiseCov.at<double>(SZ, SZ) = g_varianceConstantPosition;
	m_processNoiseCov.at<double>(SV, SV) = m_varAccel * (dt*dt);
	m_processNoiseCov.at<double>(SX, SV) = m_processNoiseCov.at<double>(SV, SX) = m_varAccel * ctheta*(dt*dt*dt / 2.0);
	m_processNoiseCov.at<double>(SY, SV) = m_processNoiseCov.at<double>(SV, SY) = m_varAccel * stheta*(dt*dt*dt / 2.0);

	m_processNoiseCov.at<double>(STHETA, STHETA) = m_varYawRate * dt*dt + scalar*(dt*dt)*g_varianceEgoAngularRate;
	m_processNoiseCov.at<double>(STHETA, STHETA_DOT) = 0.0;
	m_processNoiseCov.at<double>(STHETA_DOT, STHETA) = 0.0;
	m_processNoiseCov.at<double>(STHETA_DOT, STHETA_DOT) = g_minVariance;


}



ConstTurnRateConstantSpeedModelBev::ConstTurnRateConstantSpeedModelBev(
	ObservationBev initObs,
	double varX,
	double varY,
	double varZ,
	double varTheta,
	double varAccel,
	double varYawAccel):
	FilterModelBev(initObs, 0, 0, varX, varY, varZ, varTheta),
	m_varAccel(varAccel),
	m_varYawAccel(varYawAccel)
{


}

ConstTurnRateConstantSpeedModelBev::ConstTurnRateConstantSpeedModelBev(const  ConstTurnRateConstantSpeedModelBev& other) :
	FilterModelBev(other),
	m_varAccel(other.m_varAccel),
	m_varYawAccel(other.m_varYawAccel)
{

}

void ConstTurnRateConstantSpeedModelBev::stateConversionFunction(const Mat& x_k, const Mat& u_k, const Mat& v_k, Mat& x_kplus1, double dt)
{
	double x = x_k.at<double>(SX);
	double y = x_k.at<double>(SY);
	double z = x_k.at<double>(SZ);
	double theta = x_k.at<double>(STHETA);
	double theta_dot = x_k.at<double>(STHETA_DOT);
	double V = x_k.at<double>(SV);

	

	if (fabs(theta_dot) < 0.00001)
	{

		x_kplus1.at<double>(SX) = x + dt * V * cos(theta);
		x_kplus1.at<double>(SY) = y + dt * V * sin(theta);
	}
	else
	{
		x_kplus1.at<double>(SX) = x + (V / theta_dot) * (sin(theta + theta_dot * dt) - sin(theta));
		x_kplus1.at<double>(SY) = y + (V / theta_dot) * (cos(theta) - cos(theta + theta_dot * dt));
	}
	x_kplus1.at<double>(SZ) = z;
	x_kplus1.at<double>(STHETA) = theta + theta_dot * dt;
	x_kplus1.at<double>(SV) = V;
	x_kplus1.at<double>(STHETA_DOT) = theta_dot;
}

void ConstTurnRateConstantSpeedModelBev::updateErrorMatrix(double dt,bool ego_static)
{
	double ctheta = std::cos(m_state.at<double>(STHETA));
	double stheta = std::sin(m_state.at<double>(STHETA));


	double x= m_state.at<double>(SX);
	double y = m_state.at<double>(SY);

	double d =  std::sqrt(x*x+y*y);
	double ego_theta = std::atan2(y,x);

	double scalar=1.0;
	if (ego_static)
	{
		scalar=0.0;
	}

	m_processNoiseCov.at<double>(SX, SX) = m_varAccel * (dt*dt*dt*dt / 4.0)*ctheta*ctheta+(dt*dt/2.0)*m_varAccel+scalar*(dt*dt)*d*d*sin(ego_theta)*sin(ego_theta)*g_varianceEgoAngularRate + scalar*(dt*dt)*g_varianceEgoVelocity;;
	m_processNoiseCov.at<double>(SY, SY) = m_varAccel * (dt*dt*dt*dt / 4.0)*stheta*stheta+(dt*dt/2.0)*m_varAccel+scalar*(dt*dt)*d*d*cos(ego_theta)*cos(ego_theta)*g_varianceEgoAngularRate ;
	m_processNoiseCov.at<double>(SZ, SZ) = g_varianceConstantPosition;
	m_processNoiseCov.at<double>(SV, SV) = m_varAccel * (dt*dt);
	m_processNoiseCov.at<double>(SX, SV) = m_processNoiseCov.at<double>(SV, SX) = m_varAccel * ctheta*(dt*dt*dt / 2.0);
	m_processNoiseCov.at<double>(SY, SV) = m_processNoiseCov.at<double>(SV, SY) = m_varAccel * stheta*(dt*dt*dt / 2.0);

	m_processNoiseCov.at<double>(STHETA, STHETA) = m_varYawAccel * (dt*dt*dt*dt / 4.0) + scalar*(dt*dt)*g_varianceEgoAngularRate;
	m_processNoiseCov.at<double>(STHETA, STHETA_DOT) = m_varYawAccel * (dt*dt*dt / 2.0);
	m_processNoiseCov.at<double>(STHETA_DOT, STHETA) = m_varYawAccel * (dt*dt*dt / 2.0);
	m_processNoiseCov.at<double>(STHETA_DOT, STHETA_DOT) = m_varYawAccel * (dt*dt);


}



ConstantPositionModelBev::ConstantPositionModelBev(const  ConstantPositionModelBev& other):
	FilterModelBev(other)
{

}
ConstantPositionModelBev::ConstantPositionModelBev(
	
	ObservationBev initObs,
	double varX,
	double varY,
	double varZ,
	double varTheta) :
	FilterModelBev(initObs, 0, 0, varX, varY, varZ, varTheta)
{

}


void ConstantPositionModelBev::stateConversionFunction(const Mat& x_k, const Mat& u_k, const Mat& v_k, Mat& x_kplus1, double dt)
{
	double x = x_k.at<double>(SX);
	double y = x_k.at<double>(SY);
	double z = x_k.at<double>(SZ);
	double theta = x_k.at<double>(STHETA);
	double theta_dot = x_k.at<double>(STHETA_DOT);
	double V = x_k.at<double>(SV);


	x_kplus1.at<double>(SX) = x ;
	x_kplus1.at<double>(SY) = y ;
	x_kplus1.at<double>(SZ) = z;
	x_kplus1.at<double>(STHETA) = theta;
	x_kplus1.at<double>(SV) = 0.0;
	x_kplus1.at<double>(STHETA_DOT) = 0.0;
}

void ConstantPositionModelBev::updateErrorMatrix(double dt,bool ego_static)
{
	double ctheta = std::cos(m_state.at<double>(STHETA));
	double stheta = std::sin(m_state.at<double>(STHETA));
	double x= m_state.at<double>(SX);
	double y = m_state.at<double>(SY);

	double d =  std::sqrt(x*x+y*y);
	double ego_theta = std::atan2(y,x);
	double scalar=1.0;
	if (ego_static)
	{
		scalar=0.0;
	}

	m_processNoiseCov.at<double>(SX, SX) = g_varianceConstantPosition+scalar*(dt*dt)*d*d*sin(ego_theta)*sin(ego_theta)*g_varianceEgoAngularRate + scalar*(dt*dt)*g_varianceEgoVelocity;;
	m_processNoiseCov.at<double>(SY, SY) = g_varianceConstantPosition+scalar*(dt*dt)*d*d*cos(ego_theta)*cos(ego_theta)*g_varianceEgoAngularRate ;
	m_processNoiseCov.at<double>(SZ, SZ) = g_varianceConstantPosition;
	m_processNoiseCov.at<double>(SV, SV) = g_varianceConstantPosition;
	m_processNoiseCov.at<double>(STHETA, STHETA) = g_minVariance+scalar*(dt*dt)*g_varianceEgoAngularRate;
	m_processNoiseCov.at<double>(STHETA_DOT, STHETA_DOT) = g_minVariance;

}

