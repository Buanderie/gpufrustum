#include <gpuCuller.h>
#include <SFML/System/Clock.hpp>
#include "PyrFrustumConstGenerator.h"
#include "AABoxConstGenerator.h"
#include <iostream>

int main( int argc, char** argv )
{
	float	worldDim		= 1000.f;
	int		frustumCount	= 1;
	int		maxBoxCount		= 5000;
	int		testCount		= 500;
	int		testCount2		= 10;

	Bench::PyrFrustumConstGenerator frustumGenerator( worldDim, worldDim );
	Bench::AABoxConstGenerator		boxGenerator	( worldDim, worldDim );

	frustumGenerator.SetVolumeDistances( 20.f, 10.f, 15.f );

	boxGenerator.SetBoxDimensions( 5.f, 5.f, 5.f );

	int planeSize	=  frustumCount * 6 * 4;
	int cornerSize	=  frustumCount * 8 * 4; 

	float* frustumData	= new float[ planeSize + cornerSize ];
	float* boxData		= new float[ maxBoxCount * 8 * 4    ];

	gculInitialize( argc, argv );

	for( int j = 1; j < testCount + 1; ++j )
	{
		int boxCount = int( ( float)maxBoxCount / testCount ) * j;

		GCUL_Classification* result = new GCUL_Classification[ boxCount * frustumCount ];

		float totalDuration	= 0.f;

		for( int i = 0; i < testCount2; ++i )
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

		std::cout<< boxCount << "," << totalDuration / testCount2 << std::endl;

		delete[] result;
	}

	delete[] boxData;
	delete[] frustumData;
}