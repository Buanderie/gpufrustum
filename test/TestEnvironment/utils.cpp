#include "utils.h"
#include <SFML/System/Randomizer.hpp>

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

void generateRandomSpheres(	int n,
						 float minRadius, float maxRadius,
						 float minPosX, float maxPosX,
						 float minPosY, float maxPosY,
						 float minPosZ, float maxPosZ,
						 std::vector<glSphere>& list
							)
{
	for( int i = 0; i < n; ++i )
	{
		float radius	= sf::Randomizer::Random( minRadius,	maxRadius	);
		float x			= sf::Randomizer::Random( minPosX,		maxPosX		);
		float y			= sf::Randomizer::Random( minPosY,		maxPosY		);
		float z			= sf::Randomizer::Random( minPosZ,		maxPosZ		);

		list.push_back( glSphere( glPoint( x, y, z ), radius ) );
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

void getSpheresArray( const std::vector<glSphere>& spheres, float* data )
{
	for(int i = 0; i < spheres.size(); ++i )
	{
		int index = i * 4;

		data[ index		] = spheres[ i ].GetCenter().x;
		data[ index + 1 ] = spheres[ i ].GetCenter().y;
		data[ index + 2 ] = spheres[ i ].GetCenter().z;
		data[ index + 3 ] = spheres[ i ].GetRadius();
	}
}