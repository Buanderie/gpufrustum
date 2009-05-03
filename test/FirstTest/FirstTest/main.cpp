#include <gpuCuller.h>
#include "test_data.h"

void main( int argc, char** argv)
{
	//Initialize gpuCuller
	gculInitialize( argc, argv );

	//Initialize data
	gculEnableArray( GCUL_PYRAMIDALFRUSTUMPLANES_ARRAY );
	//Add 1 frustum
	gculPyramidalFrustumPlanesPointer( 1, GCUL_FLOAT, frustum_planes );
	gculEnableArray( GCUL_BBOXES_ARRAY );
	//Add 1 AABB
	gculBoxesPointer( 1, GCUL_FLOAT, boxA_points );

	//Process
	const unsigned int gridsize[2] = {0, 0};
	const unsigned int blocksize[3] = {10, 10, 0};
	GCUL_Classification* result = new GCUL_Classification[1];
	gculProcessFrustumCulling( gridsize, blocksize, result );
}
