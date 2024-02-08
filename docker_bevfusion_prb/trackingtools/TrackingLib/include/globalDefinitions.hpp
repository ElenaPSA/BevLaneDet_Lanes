#pragma once

#define _USE_MATH_DEFINES 
#include <math.h>  


static const  double g_maxPredictionTime = 0.5;

static const double g_initVarianceSpeed = 20.0*20.0;
static const double g_initVariancePosition = 4.0*4.0;
static const double g_initVarianceTheta = (M_PI)*(M_PI);

static const double g_varianceSize = 0.000001;
static const double g_varianceSizeObs = 0.5*0.5;

static const double g_varianceEgoVelocity = 5.0 / 3.0;
static const double g_varianceEgoAngularRate = (8.0*M_PI/180)*(8.0*M_PI/180);


static const double g_varianceConstantPosition = 0.0001;
static const double g_minVariance = 0.000001;
static const double g_varAccelConstantVelocity = 15.0 / 3.0; //per sec
static const double g_varAccelConstantVelocityConstantTurnRate = 1.0 / 3.0; //per sec
static const double g_varAngularRateConstantVelocity = (0.5 * M_PI / 180)*(0.5 * M_PI / 180); //per sec
static const double g_varAngularRateDotConstantTurnRate = (20.0 * M_PI / 180)*(20.0 * M_PI / 180); //per sec-2

const static double g_maxSpeed = 100.0;
const static double g_minIoUReprojection = 0.35;


//for Bev Filters
static const double g_variance_BevX = 0.75*0.75;
static const double g_variance_BevY = 0.75*0.75;
static const double g_variance_BevZ = 0.5*0.5;
static const double g_variance_BevTheta = (10*M_PI)/ 180 *  (10*M_PI) / 180;;


#define NUMSTATES 6
#define NUMSTATESSIZE 3
#define NUMOBSBEV 4 
#define NUMOBSSIZE 3

static const int g_objectTrack3dNbElements = 2 + NUMSTATES + 2* NUMSTATESSIZE + NUMSTATES* NUMSTATES;

enum STATES {
	SX = 0,    // position x
	SY,	       // position y
	SZ,	       // position z
	SV,			// speed
	STHETA,		// heading
	STHETA_DOT, // yaw rate
	
};

enum OBS_SIZE
{
	OWR = 0,
	OHR,
	OLR
};
enum OBS_BEV
{
	OX_BEV= 0,
	OY_BEV,
	OZ_BEV,
	OTHETA_BEV,
};




typedef struct
{
	double x,y,z;
	double w,h,l;
	double theta;
}
ObservationBev;

typedef struct
{
	double x, y,z;
	double V;
	double theta;
	double yawRate;
	double width,length,height;
	int class_id;
	uint64_t id;
	double errorCov[NUMSTATES*NUMSTATES];
	double errorCovSize[NUMSTATESSIZE];

	int serializeTrackToBuffer(double* data)
	{
		data[0] = x;
		data[1] = y;
		data[2] = z;
		data[3] = V;
		data[4] = theta;
		data[5] = yawRate;
		data[6] = width;
		data[7] = height;
		data[8] = length;
		data[9] = class_id;
		data[10] = id;
		data[11] = errorCovSize[0];
		data[12] = errorCovSize[1];
		data[13] = errorCovSize[2];
		memcpy(&data[14], errorCov, sizeof(double)*NUMSTATES*NUMSTATES);

		return g_objectTrack3dNbElements;
	}
	void readTrackFromToBuffer(const double* data)
	{
		x = data[0];
		y = data[1];
		z = data[2];
		V= data[3];
		theta = data[4];
		yawRate =  data[5];
		width = data[6];
		height = data[7];
		length = data[8];
		class_id = data[9];
		id =  data[10];
		errorCovSize[0] = data[11];
		errorCovSize[1] = data[12];
		errorCovSize[2] = data[13];
		memcpy(errorCov,&data[14], sizeof(double)*NUMSTATES*NUMSTATES);

	}
}
TrackOutput;

static const double CHI2INV_95[] = {
	-1.0,
	3.84145882069412,   // 1
	5.99146454710798,   // 2
	7.81472790325118,   // 3
	9.48772903678115,   // 4
	11.0704976935163,   // 5
	12.591587243744,     // 6
	14.0671				//7
};

static const double CHI2INV_99[] = {
	-1.0,
	6.63489660102121,   // 1
	9.21034037197618,   // 2
	11.3448667301444,   // 3
	13.2767041359876,   // 4
	15.086272469389,    // 5
	16.8118938297709,    // 6
	18.4753				//7
};

static const double CHI2INV_999[] = {
	   -1.0,             
	   10.8276,            // 1
	   13.8155,            // 2
	   16.2662,            // 3
	   18.4668,            // 4
	   20.5150,            // 5
	   22.4577,             // 6
	   24.3219				  //7
};


double  modulo(double a, double b);
double mod_angle(double theta, double thetar);


int serializeTrackToBuffer(const TrackOutput& track, double* data);
