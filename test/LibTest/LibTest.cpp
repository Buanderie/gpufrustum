// LibTest.cpp : Defines the entry point for the console application.
//

#include <stdlib.h>
#include <gpuCuller.h>

typedef struct aabb{
	float min_x, min_y, min_z;
	float max_x, max_y, max_z;
} aabb_t;

int main(int argc, char** argv)
{
	unsigned int n = 500;

	aabb_t* pol = new aabb_t[n];
	for( int i = 0; i < n; ++i )
	{
		float x = -50 + rand()%100;
		float y = -50 + rand()%100;
		pol[ i ].min_x = x - 5;
		pol[ i ].min_y = y - 5;
		pol[ i ].min_z = - 5;
		pol[ i ].max_x = x + 5;
		pol[ i ].max_y = y + 5;
		pol[ i ].max_z = 5;
	}

	gculInitialize( argc, argv );

	gculSetBVHDepth( 4 );
	
	gculSetUniverseAABB( -50, -50, -50, 50, 50, 50 );

	gculLoadAABB( n, (void*)pol );
	
	gculBuildLBVH();

	return 0;
}

