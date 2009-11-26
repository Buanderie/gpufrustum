// LibTest.cpp : Defines the entry point for the console application.
//

#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <gpuCuller.h>
#include "CPrecisionTimer.h"

using namespace std;

typedef struct aabb{
	float min_x, min_y, min_z;
	float max_x, max_y, max_z;
} aabb_t;

typedef struct hnode
{
	unsigned int splitLevel;
	unsigned int primStart;
	unsigned int primStop;
	unsigned int ID;
	unsigned int childrenStart;
	unsigned int childrenStop;
	aabb_t bbox;
} hnode_t;

int main(int argc, char** argv)
{
	srand ( time(NULL) );

	CPrecisionTimer timer;

	unsigned int n = atoi(argv[1]);
	unsigned int d = atoi(argv[2]);

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

	gculSetBVHDepth( d );
	
	gculSetUniverseAABB( -50, -50, -50, 50, 50, 50 );

	timer.Start();
	gculLoadAABB( n, (void*)pol );
	double tmemtrans = timer.Stop();
	timer.Start();
	gculBuildLBVH();
	double texec = timer.Stop();

	cout << "Time to load AABB data = " << tmemtrans << endl;
	cout << "Time to build LBVH = " << texec << endl;
	cout << "Total time = " << texec + tmemtrans << endl;
	cout << "Expected FPS = " << 1.0f/(texec+tmemtrans) << endl;

	//Get the hierarchy information
	hnode_t* bak = new hnode_t[ gculGetHierarchySize() ];
	gculGetHierarchyInformation( (void*)bak );
	for( int i = 0; i < gculGetHierarchySize(); ++i )
	{
		cout	<< "lvl=" << bak[i].splitLevel
				<< " min=(" << bak[i].bbox.min_x << "," << bak[i].bbox.min_y << "," << bak[i].bbox.min_z << ")"
				<< " max=(" << bak[i].bbox.max_x << "," << bak[i].bbox.max_y << "," << bak[i].bbox.max_z << ")"
				<< endl;
	}
	//

	system("PAUSE");
	return 0;
}

