#include <gpuCuller_internal.h>
#include <cutil_inline.h>
#include <gpuCuller_kernel.h>

extern ArrayInfo ArrayInfos[ GCUL_END_ARRAY ];

#define ARRAY_ASSERT()	assert( pointer != NULL,	"Null pointer detected." ); \
						assert( type == GCUL_INT	|| \
								type ==	GCUL_FLOAT  || \
								type ==	GCUL_DOUBLE, "Invalid type." ) \

void __stdcall gculPyramidalFrustumPlanesPointer( GCULuint size, GCULenum type, const GCULvoid* pointer )
{
	//--------------------
	// Pre-conditions
	//ARRAY_ASSERT();
	//--------------------

	ArrayInfo* info = &ArrayInfos[ GCUL_PYRAMIDALFRUSTUMPLANES_ARRAY ];

	info->size		= size;
	info->type		= ( GCUL_ArrayDataType )type;
	info->pointer	= pointer;
}

void __stdcall gculPyramidalFrustumCornersPointer( GCULuint size, GCULenum type, const GCULvoid* pointer )
{
	//--------------------
	// Pre-conditions
	//ARRAY_ASSERT();
	//--------------------

	ArrayInfo* info = &ArrayInfos[ GCUL_PYRAMIDALFRUSTUMCORNERS_ARRAY ];

	info->size		= size;
	info->type		= ( GCUL_ArrayDataType )type;
	info->pointer	= pointer;
}

void __stdcall gculPyramidalFrustumPointers( GCULuint size, GCULenum type, const GCULvoid* planes, const GCULvoid* corners )
{
	gculPyramidalFrustumPlanesPointer ( size, type, planes  );
	gculPyramidalFrustumCornersPointer( size, type, corners );
}

void __stdcall gculBoxesPointer( GCULuint size, GCULenum type, const GCULvoid* pointer )
{
	//--------------------
	// Pre-conditions
	//ARRAY_ASSERT();
	//--------------------

	ArrayInfo* info = &ArrayInfos[ GCUL_BBOXES_ARRAY ];

	info->size		= size;
	info->type		= ( GCUL_ArrayDataType )type;
	info->pointer	= pointer;
}

GCULint __stdcall gculProcessFrustumCulling( GCUL_Classification* result )
{
	//--------------------
	// Pre-conditions
	assert( result != NULL, "The pointer result must be not null." );
	//--------------------

	// Determine the frustum matrix to use.
	FrustumType currentFrustumType = CurrentFrustumType();

	if( currentFrustumType == FRUSTUMTYPE_UNDEFINED )
	{
		assert( false, "No frustum array enabled." );
		return -1;
	}

	// Determine the bounding volumes matrix to use.
	BoundingVolumeType currentBoundingVolumeType = CurrentBoundingVolumeType();

	if( currentBoundingVolumeType == BOUNDINGVOLUMETYPE_UNDEFINED )
	{
		assert( false, "No bounding volume array enabled." );
		return -1;
	}

	if( currentFrustumType == FRUSTUMTYPE_PYRAMIDAL && currentBoundingVolumeType == BOUNDINGVOLUMETYPE_BOX )
	{
		return ProcessPyramidalFrustumAABBoxCulling( result );
	}
	else
	{
		assert( false, "Not implemented yet." );
		return -1;
	}
}

int ProcessPyramidalFrustumAABBoxCulling( GCUL_Classification* result )
{
	//--------------------
	// First pass.

	// Allocate frustum planes on device memory.
	const ArrayInfo&	frustumPlanesInfo	= ArrayInfos[ GCUL_PYRAMIDALFRUSTUMPLANES_ARRAY ];
	GCULvoid*			frustumsPlanes		= NULL;

	AllocArrayDeviceMemory( &frustumsPlanes, frustumPlanesInfo );

	if( frustumsPlanes == NULL )
	{
		assert( false, "Can not allocate device memory for frustum planes." );
		return -1;
	}

	// Allocate bounding volumes on device memory.
	const ArrayInfo&	boundingBoxesInfo	= ArrayInfos[ GCUL_BBOXES_ARRAY ];
	GCULvoid*			boundingBoxes		= NULL;

	AllocArrayDeviceMemory( &boundingBoxes, boundingBoxesInfo );

	if( boundingBoxes == NULL )
	{
		assert( false, "Can not allocate device memory for bounding volumes." );
		return -1;
	}

	// Initialize input data on device memory.
	CopyArrayToDeviceMemory( frustumsPlanes, frustumPlanesInfo	);
	CopyArrayToDeviceMemory( boundingBoxes,	 boundingBoxesInfo	);

	// Allocate the matrix for the first pass.
	GCULvoid* pointPlaneIntersection = NULL;
	cudaMalloc( &pointPlaneIntersection, (frustumPlanesInfo.size*6) * (boundingBoxesInfo.size*8) * sizeof( int ));

	//Get some infos from the device and compute Block/Grid sizes
	cudaDeviceProp deviceProp;
	cutilSafeCall(cudaGetDeviceProperties(&deviceProp, cutGetMaxGflopsDeviceId()));
	int maxThreadDimXPerBlock = deviceProp.maxThreadsDim[0];
	int maxThreadDimYPerBlock = deviceProp.maxThreadsDim[1];
	int maxGridDimX = deviceProp.maxGridSize[0];
	int maxGridDimY = deviceProp.maxGridSize[1];
	int blockDimX; int gridDimX;
	int blockDimY; int gridDimY;

	//
	if( frustumPlanesInfo.size*6 > maxThreadDimXPerBlock )
	{
		blockDimX = maxThreadDimXPerBlock;
		gridDimX = (int)ceil( (double)(frustumPlanesInfo.size*6)/(double)(maxThreadDimXPerBlock) );
	}
	else
	{
		blockDimX = frustumPlanesInfo.size*6;
		gridDimX = 1;
	}
	//
	if( boundingBoxesInfo.size*8 > maxThreadDimYPerBlock )
	{
		blockDimY = maxThreadDimYPerBlock;
		gridDimY = (int)ceil( (double)(boundingBoxesInfo.size*8)/(double)(maxThreadDimYPerBlock) );
	}
	else
	{
		blockDimY = boundingBoxesInfo.size*8;
		gridDimY = 1;
	}
	//

	dim3 dimBlock1stPass(blockDimX, blockDimY);
	dim3 dimGrid1stPass(gridDimX, gridDimY);

	// Process first pass : intersect each plane with each box point.
	ClassifyPlanesPoints( 
		dimBlock1stPass,
		dimGrid1stPass,
		frustumsPlanes, 
		boundingBoxes, 
		frustumPlanesInfo.size * 6, 
		boundingBoxesInfo.size * 8, 
		(int*)pointPlaneIntersection 
	);

	// Free device input memory.
	FreeDeviceMemory( frustumsPlanes );

	//Debug stuff
	/*int* h_odata = new int[(frustumPlanesInfo.size*6) * (boundingBoxesInfo.size*8)];
	cutilSafeCall( cudaMemcpy( h_odata, pointPlaneIntersection, (frustumPlanesInfo.size*6) * (boundingBoxesInfo.size*8)*sizeof(int),
                                cudaMemcpyDeviceToHost) );
	//for each point
	for(int i = 0; i < boundingBoxesInfo.size * 8; ++i )
	{
		//for each plane
		for( int j = 0; j < frustumPlanesInfo.size * 6; ++j )
		{
			printf("%d ", h_odata[i*frustumPlanesInfo.size*6 + j]);		
		}
		printf("\r\n");
	}*/
	//

	//--------------------
	
	//--------------------
	// Second pass.

	// Allocate the result matrix on device memory.
	GCULvoid* resultDeviceMemory = NULL;	
	AllocResultDeviceMemory( &resultDeviceMemory, frustumPlanesInfo, boundingBoxesInfo );

	// Allocate frustum corners on device memory.
	const ArrayInfo& frustumCornersInfo	= ArrayInfos[ GCUL_PYRAMIDALFRUSTUMCORNERS_ARRAY ];
	GCULvoid*		 frustumsCorners	= NULL;

	AllocArrayDeviceMemory( &frustumsCorners, frustumCornersInfo );

	if( frustumsCorners == NULL )
	{
		assert( false, "Can not allocate device memory for frustum corners." );
		return -1;
	}

	// Initialize input data on device memory.
	CopyArrayToDeviceMemory( frustumsCorners, frustumCornersInfo );

	//
	if( frustumPlanesInfo.size > maxThreadDimXPerBlock )
	{
		blockDimX = maxThreadDimXPerBlock;
		gridDimX = (int)ceil( (double)(frustumPlanesInfo.size)/(double)(maxThreadDimXPerBlock) );
	}
	else
	{
		blockDimX = frustumPlanesInfo.size;
		gridDimX = 1;
	}
	//
	if( boundingBoxesInfo.size > maxThreadDimYPerBlock )
	{
		blockDimY = maxThreadDimYPerBlock;
		gridDimY = (int)ceil( (double)(boundingBoxesInfo.size)/(double)(maxThreadDimYPerBlock) );
	}
	else
	{
		blockDimY = boundingBoxesInfo.size;
		gridDimY = 1;
	}
	//

	dim3 dimBlock2ndPass(blockDimX, blockDimY);
	dim3 dimGrid2ndPass(gridDimX, gridDimY);

	// Process second pass : determine from first pass output the intersection between each frustum with each box.
	ClassifyPyramidalFrustumBoxes( 
		dimBlock2ndPass,
		dimGrid2ndPass,
		(const float*)frustumsCorners, 
		(const float*)boundingBoxes,
		(const int*)pointPlaneIntersection, 
		frustumPlanesInfo.size * 6, 
		boundingBoxesInfo.size * 8, 
		(int*)resultDeviceMemory 
	);

	// Free device input memory.
	FreeDeviceMemory( frustumsCorners		 );
	FreeDeviceMemory( boundingBoxes			 );
	FreeDeviceMemory( pointPlaneIntersection );

	// Copy the result from device memory.
	cutilSafeCall(cudaMemcpy( result, resultDeviceMemory, frustumPlanesInfo.size * boundingBoxesInfo.size * sizeof(int), cudaMemcpyDeviceToHost));

	FreeDeviceMemory( resultDeviceMemory );

	//--------------------

	return 0;
}
