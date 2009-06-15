#include <stdio.h>
#include <gpuCuller.h>
#include "test_data.h"

void main( int argc, char** argv)
{
	//Initialize gpuCuller
	gculInitialize( argc, argv );

	gculDisableArray( GCUL_SPHERICALFRUSTUM_ARRAY );
	gculDisableArray( GCUL_BSPHERES_ARRAY );

	gculEnable( GCUL_OCCLUSION_CULLING );

	//Initialize data	
	//Add 1 frustum
	gculEnableArray( GCUL_PYRAMIDALFRUSTUMPLANES_ARRAY );
	gculEnableArray( GCUL_PYRAMIDALFRUSTUMCORNERS_ARRAY );
	gculPyramidalFrustumCornersPointer( 2, GCUL_FLOAT, frustum_corners );
	gculPyramidalFrustumPlanesPointer( 2, GCUL_FLOAT, frustum_planes );
	
	//Add 1 AABB
	gculEnableArray( GCUL_BBOXES_ARRAY );
	gculBoxesPointer( 4, GCUL_FLOAT, boxA_points );

	//Process

	GCUL_Classification* result = new GCUL_Classification[8];
	gculProcessFrustumCulling( result );
	
	for( int i = 0; i < 2; ++i )
	{
		for( int j = 0; j < 4  ; ++j )
		{
			printf( "Frustum %d Box %d Result %d\n", i, j, result[ i + j * 2 ] );
		}
	}
}
