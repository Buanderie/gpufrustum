#include <math.h>
#include "utils.h"

float randf()
{
	return (((float)(rand())) / ((float)(RAND_MAX) + 1.0f));
}

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
								std::vector<glPyramidalFrustum>& list )
{
	for(int i = 0; i < n; ++i )
	{
		float FOV = minFOV + randf()*( maxFOV - minFOV );
		float Near = minNear + randf()*( maxNear - minNear );
		float Far = minFar + randf()*( maxFar - minFar );
		float AspectRatio = minAspectRatio + randf()*( maxAspectRatio - minAspectRatio );
		float PosX = minPosX + randf()*( maxPosX - minPosX );
		float PosY = minPosY + randf()*( maxPosY - minPosY );
		float PosZ = minPosZ + randf()*( maxPosZ - minPosZ );
		float RotX = minRotX + randf()*( maxRotX - minRotX );
		float RotY = minRotY + randf()*( maxRotY - minRotY );
		float RotZ = minRotZ + randf()*( maxRotZ - minRotZ );
		glPyramidalFrustum f(FOV, Near, Far, AspectRatio, glVector4f(PosX, PosY, PosZ, 0), RotX, RotY, RotZ );
		list.push_back( f );
	}
}

void generateClusteredPyrFrustums( int n,
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
								std::vector<glPyramidalFrustum>& list )
{
	int clusterSize = 20;
	int clusterPop = 0;
	int tmpN = n;

	float clusterWidth = ((minNear+maxNear)/(minFar+maxFar)/2.0f)*clusterSize/6;
	float clusterHeight = (minAspectRatio*maxAspectRatio)*clusterWidth*clusterSize/6;
	float clusterDepth = ((minNear+maxNear)/2.0f)*clusterSize/6;

	while( 1 )
	{
		if( tmpN < 0 )
			return;

		if( tmpN >= clusterSize )
			clusterPop = clusterSize;
		else
			clusterPop = tmpN;

		float PosX = minPosX + randf()*( maxPosX - minPosX );
		float PosY = minPosY + randf()*( maxPosY - minPosY );
		float PosZ = minPosZ + randf()*( maxPosZ - minPosZ );
		float clusterMinX = PosX - clusterWidth/2;
		float clusterMinY = PosY - clusterDepth/2;
		float clusterMinZ = PosZ - clusterHeight/2;
		float clusterMaxX = PosX + clusterWidth/2;
		float clusterMaxY = PosY + clusterDepth/2;
		float clusterMaxZ = PosZ + clusterHeight/2;

		generateRandomPyrFrustums(	clusterPop,
									minFOV, maxFOV,
									minNear, maxNear,
									minFar, maxFar,
									minAspectRatio, maxAspectRatio,
									clusterMinX, clusterMaxX,
									clusterMinY, clusterMaxY, 
									clusterMinZ, clusterMaxZ,
									minRotX, maxRotX,
									minRotY, maxRotY,
									minRotZ, maxRotZ,
									list);
		tmpN -= clusterSize;
	}
}

void generateMixedPyrFrustums( int n,
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
								std::vector<glPyramidalFrustum>& list )
{
	int mid1 = (int)ceilf(3.0f*(float)n/4.0f);
	int mid2 = (int)ceilf((float)n/4.0f);
	while( mid1 + mid2 != n )
		mid2--;

	generateRandomPyrFrustums(	mid1,
							minFOV, maxFOV,
							minNear, maxNear,
							minFar, maxFar,
							minAspectRatio, maxAspectRatio,
							minPosX, maxPosX,
							minPosY, maxPosY, 
							minPosZ, maxPosZ,
							minRotX, maxRotX,
							minRotY, maxRotY,
							minRotZ, maxRotZ,
							list);

	generateRandomPyrFrustums(	mid2,
								minFOV, maxFOV,
								minNear, maxNear,
								minFar, maxFar,
								minAspectRatio, maxAspectRatio,
								minPosX, maxPosX,
								minPosY, maxPosY, 
								minPosZ, maxPosZ,
								minRotX, maxRotX,
								minRotY, maxRotY,
								minRotZ, maxRotZ,
								list);
}

void generateRandomAABBs(	int n,
							float minWidth, float maxWidth,
							float minHeight, float maxHeight,
							float minDepth, float maxDepth,
							float minPosX, float maxPosX,
							float minPosY, float maxPosY,
							float minPosZ, float maxPosZ,
							std::vector<glAABB>& list
							)
{
	for(int i = 0; i < n; ++i)
	{
		float width = minWidth + randf()*(maxWidth-minWidth);
		float height = minHeight + randf()*(maxHeight-minHeight);
		float depth = minDepth + randf()*(maxDepth-minDepth);
		float PosX = minPosX + randf()*( maxPosX - minPosX );
		float PosY = minPosY + randf()*( maxPosY - minPosY );
		float PosZ = minPosZ + randf()*( maxPosZ - minPosZ );
		glAABB a( glVector4f( PosX - width/2, PosY - depth/2, PosZ - height/2, 0 ), glVector4f( PosX + width/2, PosY + depth/2, PosZ + height/2, 0 ));
		list.push_back( a );
	}
}

void generateClusteredAABBs(	int n,
								float minWidth, float maxWidth,
								float minHeight, float maxHeight,
								float minDepth, float maxDepth,
								float minPosX, float maxPosX,
								float minPosY, float maxPosY,
								float minPosZ, float maxPosZ,
								std::vector<glAABB>& list
							)
{
	int clusterSize = 20;
	int clusterPop = 0;
	int tmpN = n;
	float clusterWidth = ((minWidth+maxWidth)/2.0f)*clusterSize/6;
	float clusterHeight = ((minHeight+maxHeight)/2.0f)*clusterSize/6;
	float clusterDepth = ((minDepth+maxDepth)/2.0f)*clusterSize/6;

	while( 1 )
	{
		if( tmpN < 0 )
			return;

		if( tmpN >= clusterSize )
			clusterPop = clusterSize;
		else
			clusterPop = tmpN;

		float PosX = minPosX + randf()*( maxPosX - minPosX );
		float PosY = minPosY + randf()*( maxPosY - minPosY );
		float PosZ = minPosZ + randf()*( maxPosZ - minPosZ );
		float clusterMinX = PosX - clusterWidth/2;
		float clusterMinY = PosY - clusterDepth/2;
		float clusterMinZ = PosZ - clusterHeight/2;
		float clusterMaxX = PosX + clusterWidth/2;
		float clusterMaxY = PosY + clusterDepth/2;
		float clusterMaxZ = PosZ + clusterHeight/2;

		generateRandomAABBs(	clusterPop,
								minWidth,
								maxWidth,
								minHeight,
								maxHeight,
								minDepth,
								maxDepth,
								clusterMinX,
								clusterMaxX,
								clusterMinY,
								clusterMaxY,
								clusterMinZ,
								clusterMaxZ,
								list);
		tmpN -= clusterSize;
	}
}

void generateMixedAABBs(		int n,
								float minWidth, float maxWidth,
								float minHeight, float maxHeight,
								float minDepth, float maxDepth,
								float minPosX, float maxPosX,
								float minPosY, float maxPosY,
								float minPosZ, float maxPosZ,
								std::vector<glAABB>& list
						)
{
	int mid1 = (int)ceilf(3.0f*(float)n/4.0f);
	int mid2 = (int)ceilf((float)n/4.0f);
	while( mid1 + mid2 != n )
		mid2--;

	//Generate half clustered
	{
		generateClusteredAABBs(	mid1,
								minWidth, maxWidth,
								minHeight, maxHeight,
								minDepth, maxDepth,
								minPosX, maxPosX,
								minPosY, maxPosY,
								minPosZ,maxPosZ,
								list
								);
	}

	//Generate half uniform
	{
		generateRandomAABBs(	mid2,
								minWidth, maxWidth,
								minHeight, maxHeight,
								minDepth, maxDepth,
								minPosX, maxPosX,
								minPosY, maxPosY,
								minPosZ, maxPosZ,
								list
								);
	}
}

void getFrustumPlanesArray( std::vector<glPyramidalFrustum>& list, float* a )
{
	for(int i = 0; i < list.size(); ++i )
	{
		list[i].extractPlanesData(a + i*24);
	}
}

void getFrustumCornersArray( std::vector<glPyramidalFrustum>& list, float* a )
{
	for(int i = 0; i < list.size(); ++i )
	{
		list[i].extractCornersData(a + i*24);
	}
}

void getAABBCornersArray( std::vector<glAABB> list, float* a )
{
	for(int i = 0; i < list.size(); ++i )
	{
		list[i].extractCornersData(a + i*24);
	}
}