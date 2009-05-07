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
	gculPyramidalFrustumCornersPointer( 1, GCUL_FLOAT, frustum_corners );
	gculPyramidalFrustumPlanesPointer( 1, GCUL_FLOAT, frustum_planes );
	
	//Add 1 AABB
	gculEnableArray( GCUL_BBOXES_ARRAY );
	gculBoxesPointer( 1, GCUL_FLOAT, boxA_points );

	//Process
	const unsigned int gridsize[2] = {100, 100};
	const unsigned int blocksize[3] = {10, 10, 0};
	GCUL_Classification* result = new GCUL_Classification[1];
	gculProcessFrustumCulling( result );
}
