#include <gpuCuller.h>
#include <SFML/System/Clock.hpp>
#include "PyrFrustumConstGenerator.h"
#include "AABoxConstGenerator.h"
#include <iostream>

int main( int argc, char** argv )
{
	float	worldDim		= 1000.f;
	int		frustumCount	= 200;
	int		boxCount		= 600;
	int		testCount		= 100;
	float	totalDuration	= 0.f;

	Bench::PyrFrustumConstGenerator frustumGenerator( worldDim, worldDim );
	Bench::AABoxConstGenerator		boxGenerator	( worldDim, worldDim );

	frustumGenerator.SetVolumeDistances( 20.f, 10.f, 15.f );

	boxGenerator.SetBoxDimensions( 5.f, 5.f, 5.f );

	int planeSize	=  frustumCount * 6 * 4;
	int cornerSize	=  frustumCount * 8 * 4; 

	float* frustumData	= new float[ planeSize + cornerSize ];
	float* boxData		= new float[ boxCount * 8 * 4 ];

	GCUL_Classification* result = new GCUL_Classification[ boxCount * frustumCount ];

	gculInitialize( argc, argv );

	for( int i = 0; i < testCount ; ++i )
	{
		frustumGenerator.Generate( 200, frustumData );

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

		std::cout<< "Culling duration = " << duration << std::endl;
	}

	std::cout<< "Average duration = " << totalDuration / testCount << std::endl;

	delete[] boxData;
	delete[] frustumData;
	delete[] result;

	//system( "pause" );
}