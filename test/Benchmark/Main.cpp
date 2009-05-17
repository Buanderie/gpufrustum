#include <gpuCuller.h>
#include <SFML/System/Clock.hpp>
#include "PyrFrustumConstGenerator.h"
#include "AABoxConstGenerator.h"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <istream>

#define WORLD_DIM					1000.f
#define TESTS_PER_CLASSIFICATION	5

template<typename T>
bool FromString(const std::string & source, T & dest)
{
	std::istringstream iss(source);

	return iss>>dest != 0;
}

int main( int argc, char** argv )
{
	int frustumStart, frustumEnd, frustumInc;
	int boxStart, boxEnd, boxInc;

	if( argc != 7 )
	{
		return -1;
	}

	// Read parameters
	FromString( argv[ 1 ], frustumStart	);
	FromString( argv[ 2 ], frustumEnd	);
	FromString( argv[ 3 ], frustumInc	);
	FromString( argv[ 4 ], boxStart		);
	FromString( argv[ 5 ], boxEnd		);
	FromString( argv[ 6 ], boxInc		);
	
	Bench::PyrFrustumConstGenerator frustumGenerator( WORLD_DIM, WORLD_DIM );
	Bench::AABoxConstGenerator		boxGenerator	( WORLD_DIM, WORLD_DIM );

	frustumGenerator.SetVolumeDistances( 20.f, 10.f, 15.f );

	boxGenerator.SetBoxDimensions( 5.f, 5.f, 5.f );

	int planeSize	=  frustumEnd * 6 * 4;
	int cornerSize	=  frustumEnd * 8 * 4; 

	float* frustumData	= new float[ planeSize + cornerSize ];
	float* boxData		= new float[ boxEnd * 8 * 4			];

	gculInitialize( 1, argv );

	for( int i = frustumStart; i <= frustumEnd ; i += frustumInc )
	{
		for( int j = boxStart; j <= boxEnd ; j += boxInc )
		{
			GCUL_Classification* result = new GCUL_Classification[ i * j ];

			float totalDuration	= 0.f;

			for( int k = 0; k < TESTS_PER_CLASSIFICATION; ++k )
			{
				frustumGenerator.Generate( i, frustumData );

				boxGenerator.Generate( j, boxData );

				gculBoxesPointer( j, GCUL_FLOAT, boxData );

				gculPyramidalFrustumPlanesPointer ( i, GCUL_FLOAT, frustumData					);
				gculPyramidalFrustumCornersPointer( i, GCUL_FLOAT, &frustumData[ i * 6 * 4 ]	);

				gculEnableArray( GCUL_PYRAMIDALFRUSTUMCORNERS_ARRAY );
				gculEnableArray( GCUL_PYRAMIDALFRUSTUMPLANES_ARRAY	);

				gculEnableArray( GCUL_BBOXES_ARRAY );

				sf::Clock timer; timer.Reset();

				gculProcessFrustumCulling( result );

				float duration = timer.GetElapsedTime();

				totalDuration += duration;
			}

			std::cout
			<< 
				"Frustums = " << i << 
				" Boxes = "  << j << 
				" Duration = " << totalDuration / TESTS_PER_CLASSIFICATION << "s" 
			<< 
			std::endl;

			delete[] result;
		}
	}

	delete[] boxData;
	delete[] frustumData;
}