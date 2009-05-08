#include <stdio.h>
#include <gpuCuller.h>
#include "test_data.h"

void main( int argc, char** argv)
{
	//Initialize gpuCuller
	gculInitialize( argc, argv );

	gculDisableArray( GCUL_SPHERICALFRUSTUM_ARRAY );
	gculDisableArray( GCUL_BSPHERES_ARRAY );

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
	printf("%d %d %d %d", result[1], result[2], result[3], result[4]);
}
