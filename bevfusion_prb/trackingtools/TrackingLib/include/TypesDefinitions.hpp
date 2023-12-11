#pragma once

#include <string>
#include <vector>
#include <list>
#include <fstream>
#include <cassert>
#include <iostream>
#include <opencv2/opencv.hpp>

typedef struct
{
	float x;
	float y;
	float w;
	float h;
	float score;
	int class_num;
}
ROIObject;

static const int g_objectBox3dNbElements = 16;
static const float g_minVehicleRatio = 0.5f;

class ObjectBox3d
{

public:
	
	ObjectBox3d():id(0) {};
	ObjectBox3d(const float* data) :
		rect_im(data[0], data[1], data[2], data[3]),
		class_id((unsigned int)data[4]),
		confidence(data[5]),
		id((uint64_t)data[6]),
		position(data[7], data[8], data[9]),
		size(data[10], data[11], data[12]),
		theta(data[13]),
		proj_center(data[14], data[15])
	{
	}

	int serializetoBuffer(float* data)
	{
		data[0] = rect_im.x;
		data[1] = rect_im.y;
		data[2] = rect_im.width;
		data[3] = rect_im.height;
		data[4] = (float)class_id;
		data[5] = (float)confidence;
		data[6] = (float)id;
		data[7] = position.x;
		data[8] = position.y;
		data[9] = position.z;
		data[10] = size[0];
		data[11] = size[1];
		data[12] = size[2];
		data[13] = theta;
		data[14] = proj_center.x;
		data[15] = proj_center.y;

		return g_objectBox3dNbElements;
	}


public:
	cv::Rect2f rect_im;
	unsigned int class_id;
	float confidence;
	uint64_t id;
	cv::Point2d proj_center;
	cv::Point3d position;
	cv::Vec3f size; //w, h, l
	double theta;
};




typedef struct ImageObjectTrack_
{
	cv::Rect2d rect;
	unsigned int class_id;
	float confidence;
	uint64_t id;
	uint count;
	ImageObjectTrack_() :
		count(0), confidence(0.0)
	{

	}
}ImageObjectTrack;





enum {
	CAR_ID = 1,
	TRUCK_ID,
	MOTO_ID,
	BUS_ID,
	BICYCLE_ID,
	PEOPLE_ID,
	UNKNOW_ID
};
