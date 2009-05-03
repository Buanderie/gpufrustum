#ifndef __TEST_DATA_H__
#define __TEST_DATA_H__

float frustum_planes[24] = {	0.0f,			0.0f,			-1.0f,			1.0f,		//Near
								0.0f,			0.0f,			1.0f,			-10.0f,		//Far
								0.0f,			-0.81649655f,	-0.57755026f,	0.0f,		//Down
								0.0f,			0.81649655f,	-0.57735026f,	0.0f,		//Up
								-0.81649655f,	0.0f,			-0.57735026f,	0.0f,		//Left
								0.81649605f,	0.0f,			-0.57735026f,	0.0f		//Right
							};
float boxA_points[24] = {	-1.0f,	8.0f,	1.0f,
							1.0f,	8.0f,	1.0f,
							-1.0f,	6.0f,	1.0f,
							1.0f,	6.0f,	1.0f,
							-1.0f,	8.0f,	-1.0f,
							1.0f,	8.0f,	-1.0f,
							-1.0f,	6.0f,	-1.0f,
							1.0f,	6.0f,	-1.0f
						};
#endif