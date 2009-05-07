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