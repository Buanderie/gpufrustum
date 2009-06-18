#include <gpuCuller.h>
#include <SFML/System/Clock.hpp>
#include "PyrFrustumGenerator.h"
#include "SphereGenerator.h"
#include "AABoxGenerator.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <istream>
#include <math.h>

#define WORLD_DIM					1000.f
#define TESTS_PER_CLASSIFICATION	5

template<typename T>
bool FromString(const std::string & source, T & dest)
{
	std::istringstream iss(source);

	return iss>>dest != 0;
}

enum CullingType
{
	PyrBox,
	PyrSphere,
	SphereSphere,
	SphereBox
};

float RunBenchPyrBox		( int frustumCount, int boxCount	);
float RunBenchPyrSphere		( int frustumCount, int sphereCount );
float RunBenchSphereSphere	( int frustumCount, int sphereCount );
float RunBenchSphereBox		( int frustumCount, int boxCount	);


int main( int argc, char** argv )
{	
	gculInitialize( 1, argv );

	if( argc != 4 )
	{
		return -1;
	}

	int powBoxes;
	int powFrustums;
	
	CullingType cullingType;
	int			iCullingType;

	FromString( argv[ 1 ], iCullingType	);
	FromString( argv[ 2 ], powBoxes		);
	FromString( argv[ 3 ], powFrustums	);

	cullingType = ( CullingType )iCullingType;

	std::vector<float> result;

	for( int i = 0; i < powFrustums; ++i )
	{
		int fmin = ( int )pow( 10.f, i     );
		int fmax = ( int )pow( 10.f, i + 1 );
		int fpas = fmin;

		for( int j = 0; j < powBoxes; ++j )
		{
			int bmin = ( int )pow( 10.f, j     );
			int bmax = ( int )pow( 10.f, j + 1 );
			int bpas = bmin;

			for( int f = fmin; f <= fmax ; f += fpas )
			{
				for( int b = bmin; b <= bmax ; b += bpas )
				{
					switch( cullingType )
					{
					case PyrBox:
						result.push_back( RunBenchPyrBox( f, b ) );
						break;
					case PyrSphere:
						result.push_back( RunBenchPyrSphere( f, b ) );
						break;
					case SphereSphere:
						result.push_back( RunBenchSphereSphere( f, b ) );
						break;
					case SphereBox:
						result.push_back( RunBenchSphereBox( f, b ) );
						break;
					}	
				}
			}
		}
	}

	std::ofstream fileX( "./x.txt" );
	std::ofstream fileY( "./y.txt" );
	std::ofstream fileZ( "./z.txt" );

	// write x axis
	for( int i = 0; i < powFrustums; ++i )
	{
		int fmin = ( int )pow( 10.f, i     );
		int fmax = ( int )pow( 10.f, i + 1 );
		int fpas = fmin;

		for( int f = fmin; f <= fmax ; f += fpas )
		{
			fileX << f << ";";
		}
	}

	std::cout << std::endl;

	// write y axis
	for( int i = 0; i < powBoxes; ++i )
	{
		int bmin = ( int )pow( 10.f, i     );
		int bmax = ( int )pow( 10.f, i + 1 );
		int bpas = bmin;

		for( int b = bmin; b <= bmax ; b += bpas )
		{
			fileY << b << ";";
		}
	}

	std::cout << std::endl;

	// write result
	int column = 0;
	for( int i = 0; i < result.size(); ++i )
	{
		if( column >= powFrustums * 10 )
		{
			fileZ << std::endl;
			column = 0;
		}
		fileZ << result[ i ] << ";";
		++column;
	}
}

float RunBenchPyrBox( int frustumCount, int boxCount )
{
	Bench::PyrFrustumGenerator	frustumGenerator( WORLD_DIM, WORLD_DIM );
	Bench::AABoxGenerator		boxGenerator	( WORLD_DIM, WORLD_DIM );

	frustumGenerator.SetVolumeDistances( 20.f, 10.f, 15.f );

	boxGenerator.SetBoxDimensions( 5.f, 5.f, 5.f );

	int planeSize	=  frustumCount * 6 * 4;
	int cornerSize	=  frustumCount * 8 * 4; 

	float* frustumData	= new float[ planeSize + cornerSize ];
	float* boxData		= new float[ boxCount * 8 * 4		];

	GCUL_Classification* result = new GCUL_Classification[ frustumCount * boxCount ];

	float totalDuration	= 0.f;

	for( int k = 0; k < TESTS_PER_CLASSIFICATION; ++k )
	{
		frustumGenerator.Generate( frustumCount, frustumData );

		boxGenerator.Generate( boxCount, boxData );

		gculBoxesPointer( boxCount, GCUL_FLOAT, boxData );

		gculPyramidalFrustumPlanesPointer ( frustumCount, GCUL_FLOAT, frustumData							);
		gculPyramidalFrustumCornersPointer( frustumCount, GCUL_FLOAT, &frustumData[ frustumCount * 6 * 4 ]	);

		gculEnableArray( GCUL_PYRAMIDALFRUSTUMCORNERS_ARRAY );
		gculEnableArray( GCUL_PYRAMIDALFRUSTUMPLANES_ARRAY	);

		gculEnableArray( GCUL_BBOXES_ARRAY );

		sf::Clock timer; timer.Reset();

		gculProcessFrustumCulling( result );

		float duration = timer.GetElapsedTime();

		totalDuration += duration;
	}

	delete[] result;
	delete[] boxData;
	delete[] frustumData;

	return totalDuration;
}

float RunBenchPyrSphere( int frustumCount, int sphereCount )
{
	Bench::PyrFrustumGenerator	frustumGenerator( WORLD_DIM, WORLD_DIM );
	Bench::SphereGenerator		sphereGenerator	( WORLD_DIM, WORLD_DIM );

	frustumGenerator.SetVolumeDistances( 20.f, 10.f, 15.f );

	sphereGenerator.SetSphereRadius( 1.f, 5.f );

	int planeSize	=  frustumCount * 6 * 4;
	int cornerSize	=  frustumCount * 8 * 4; 

	float* frustumData	= new float[ planeSize + cornerSize ];
	float* sphereData	= new float[ sphereCount * 4		];

	GCUL_Classification* result = new GCUL_Classification[ frustumCount * sphereCount ];

	float totalDuration	= 0.f;

	for( int k = 0; k < TESTS_PER_CLASSIFICATION; ++k )
	{
		frustumGenerator.Generate( frustumCount, frustumData );

		sphereGenerator.Generate( sphereCount, sphereData );

		gculSpheresPointer( sphereCount, GCUL_FLOAT, sphereData );

		gculPyramidalFrustumPlanesPointer ( frustumCount, GCUL_FLOAT, frustumData							);
		gculPyramidalFrustumCornersPointer( frustumCount, GCUL_FLOAT, &frustumData[ frustumCount * 6 * 4 ]	);

		gculEnableArray( GCUL_PYRAMIDALFRUSTUMCORNERS_ARRAY );
		gculEnableArray( GCUL_PYRAMIDALFRUSTUMPLANES_ARRAY	);

		gculEnableArray( GCUL_BSPHERES_ARRAY );

		sf::Clock timer; timer.Reset();

		int err = gculProcessFrustumCulling( result );

		if( err != 0 )
		{
			std::cout << "Error!" << std::endl;
		}

		float duration = timer.GetElapsedTime();

		totalDuration += duration;
	}

	delete[] result;
	delete[] sphereData;
	delete[] frustumData;

	return totalDuration;
}

float RunBenchSphereSphere( int frustumCount, int sphereCount )
{
	Bench::SphereGenerator	frustumGenerator( WORLD_DIM, WORLD_DIM );
	Bench::SphereGenerator	sphereGenerator	( WORLD_DIM, WORLD_DIM );

	frustumGenerator.SetSphereRadius( 2.f, 10.f );

	sphereGenerator.SetSphereRadius( 1.f, 5.f );

	int planeSize	=  frustumCount * 6 * 4;
	int cornerSize	=  frustumCount * 8 * 4; 

	float* frustumData	= new float[ frustumCount * 4 ];
	float* sphereData	= new float[ sphereCount  * 4 ];

	GCUL_Classification* result = new GCUL_Classification[ frustumCount * sphereCount ];

	float totalDuration	= 0.f;

	for( int k = 0; k < TESTS_PER_CLASSIFICATION; ++k )
	{
		frustumGenerator.Generate( frustumCount, frustumData );

		sphereGenerator.Generate( sphereCount, sphereData );

		gculSpheresPointer( sphereCount, GCUL_FLOAT, sphereData );

		gculSphericalFrustumPointer( frustumCount, GCUL_FLOAT, frustumData );

		gculEnableArray( GCUL_SPHERICALFRUSTUM_ARRAY );

		gculEnableArray( GCUL_BSPHERES_ARRAY );

		sf::Clock timer; timer.Reset();

		gculProcessFrustumCulling( result );

		float duration = timer.GetElapsedTime();

		totalDuration += duration;
	}

	delete[] result;
	delete[] sphereData;
	delete[] frustumData;

	return totalDuration;
}

float RunBenchSphereBox( int frustumCount, int boxCount	)
{
	Bench::SphereGenerator	frustumGenerator( WORLD_DIM, WORLD_DIM );
	Bench::AABoxGenerator	boxGenerator	( WORLD_DIM, WORLD_DIM );

	frustumGenerator.SetSphereRadius( 2.f, 10.f );

	boxGenerator.SetBoxDimensions( 5.f, 5.f, 5.f );

	float* frustumData	= new float[ frustumCount * 4 ];
	float* boxData		= new float[ boxCount * 8 * 4 ];

	GCUL_Classification* result = new GCUL_Classification[ frustumCount * boxCount ];

	float totalDuration	= 0.f;

	for( int k = 0; k < TESTS_PER_CLASSIFICATION; ++k )
	{
		frustumGenerator.Generate( frustumCount, frustumData );

		boxGenerator.Generate( boxCount, boxData );

		gculBoxesPointer( boxCount, GCUL_FLOAT, boxData );

		gculSphericalFrustumPointer( frustumCount, GCUL_FLOAT, frustumData );

		gculEnableArray( GCUL_SPHERICALFRUSTUM_ARRAY );

		gculEnableArray( GCUL_BBOXES_ARRAY );

		sf::Clock timer; timer.Reset();

		gculProcessFrustumCulling( result );

		float duration = timer.GetElapsedTime();

		totalDuration += duration;
	}

	delete[] result;
	delete[] boxData;
	delete[] frustumData;

	return totalDuration;
}



int planeSize	=  frustumCount * 6 * 4;
int cornerSize	=  frustumCount * 8 * 4; 

float* frustumData	= new float[ planeSize + cornerSize ];
float* boxData		= new float[ boxCount * 8 * 4		];

// result contiendra la classification des boites par rapport aux frustums
GCUL_Classification* result = new GCUL_Classification[ frustumCount * boxCount ];

// L'utilisateur rempli ici les tableaux relatifs aux
// boites et aux frustums.
frustumGenerator.Generate( frustumCount, frustumData );	
boxGenerator	.Generate( boxCount,	 boxData	 );

// Définition des données sur lesquelles travailler
gculBoxesPointer( boxCount, GCUL_FLOAT, boxData );
gculPyramidalFrustumPlanesPointer ( frustumCount, GCUL_FLOAT, frustumData							);
gculPyramidalFrustumCornersPointer( frustumCount, GCUL_FLOAT, &frustumData[ frustumCount * 6 * 4 ]	);

// Activation des tableaux nous intéressant
gculEnableArray( GCUL_PYRAMIDALFRUSTUMCORNERS_ARRAY );
gculEnableArray( GCUL_PYRAMIDALFRUSTUMPLANES_ARRAY	);
gculEnableArray( GCUL_BBOXES_ARRAY );

// Calcul du culling
gculProcessFrustumCulling( result );

// La classification de la boite i par rapport au frustum j
// est accessible à l'indice i*frustumCount + j
int i = 0; int j = 0;
GCUL_Classification c = result[i*frustumCount + j];

delete[] result;
delete[] boxData;
delete[] frustumData;
