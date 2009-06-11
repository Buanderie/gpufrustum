#include <gpuCuller.h>
#include <SFML/System/Clock.hpp>
#include "PyrFrustumConstGenerator.h"
#include "AABoxConstGenerator.h"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <istream>
#include <math.h>

#define WORLD_DIM					1000.f
#define TESTS_PER_CLASSIFICATION	5

float RunBench( int frustumCount, int boxCount );

template<typename T>
bool FromString(const std::string & source, T & dest)
{
	std::istringstream iss(source);

	return iss>>dest != 0;
}

int main( int argc, char** argv )
{	
	gculInitialize( 1, argv );

	if( argc != 3 )
	{
		return -1;
	}

	int powBoxes;
	int powFrustums;

	FromString( argv[ 1 ], powBoxes		);
	FromString( argv[ 2 ], powFrustums	);

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
					result.push_back( RunBench( f, b ) );
				}
			}
		}
	}

	// write x axis
	for( int i = 0; i < powFrustums; ++i )
	{
		int fmin = ( int )pow( 10.f, i     );
		int fmax = ( int )pow( 10.f, i + 1 );
		int fpas = fmin;

		for( int f = fmin; f <= fmax ; f += fpas )
		{
			std::cout << f << ";";
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
			std::cout << b << ";";
		}
	}

	std::cout << std::endl;

	// write result
	int column = 0;
	for( int i = 0; i < result.size(); ++i )
	{
		if( column >= powFrustums * 10 )
		{
			std::cout << std::endl;
			column = 0;
		}
		std::cout << result[ i ] << ";";
		++column;
	}
}

float RunBench( int frustumCount, int boxCount )
{
	Bench::PyrFrustumConstGenerator frustumGenerator( WORLD_DIM, WORLD_DIM );
	Bench::AABoxConstGenerator		boxGenerator	( WORLD_DIM, WORLD_DIM );

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