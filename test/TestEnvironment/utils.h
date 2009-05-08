#ifndef __UTILS_H__
#define __UTILS_H__

#include <vector>
#include "glPyramidalFrustum.h"
#include "glAABB.h"

float randf();

void generateRandomPyrFrustums( int n,
								float minFOV, float maxFOV,
								float minNear, float maxNear,
								float minFar, float maxFar,
								float minAspectRatio, float maxAspectRatio,
								float minPosX, float maxPosX,
								float minPosY, float maxPosY,
								float minPosZ, float maxPosZ,
								float minRotX, float maxRotX,
								float minRotY, float maxRotY,
								float minRotZ, float maxRotZ,
								std::vector<glPyramidalFrustum>& list );
void generateRandomAABBs(	int n,
							float minWidth, float maxWidth,
							float minHeight, float maxHeight,
							float minDepth, float maxDepth,
							float minPosX, float maxPosX,
							float minPosY, float maxPosY,
							float minPosZ, float maxPosZ,
							std::vector<glAABB>& list
							);

void getFrustumPlanesArray( std::vector<glPyramidalFrustum>& list, float* a );
void getFrustumCornersArray( std::vector<glPyramidalFrustum> list, float* a );
void getAABBCornersArray( std::vector<glAABB> list, float* a );

#endif