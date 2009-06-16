#include <gpuCuller_internal.h>
#include <cutil_inline.h>
#include <gpuCuller_BVH.h>
#include <gpuCuller_kernel.h>

extern ArrayInfo ArrayInfos[ GCUL_END_ARRAY ];

extern bool EnableStates[ GCUL_END_ENABLESTATE ];

extern DeviceFunctionEnv ClassifyPlanesPointsEnv;
extern DeviceFunctionEnv ClassifyPyramidalFrustumBoxesEnv;
extern DeviceFunctionEnv ClassifyPlanesSpheresEnv;
extern DeviceFunctionEnv ClassifyPyramidalFrustumSpheresEnv;
extern DeviceFunctionEnv InverseClassifyPyramidalFrustumBoxesEnv;
extern DeviceFunctionEnv ClassifySphericalFrustumSpheresEnv;

BVHTree* tree;

#define ARRAY_ASSERT()	assert( pointer != NULL,	"Null pointer detected." ); \
						assert( type == GCUL_INT	|| \
								type ==	GCUL_FLOAT  || \
								type ==	GCUL_DOUBLE, "Invalid type." ) \

void __stdcall gculPyramidalFrustumPlanesPointer( GCULuint size, GCULenum type, const GCULvoid* pointer )
{
	//--------------------
	// Pre-conditions
	ARRAY_ASSERT();
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
	ARRAY_ASSERT();
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

void __stdcall gculSphericalFrustumPointer( GCULuint size, GCULenum type, const GCULvoid* pointer )
{
	//--------------------
	// Pre-conditions
	ARRAY_ASSERT();
	//--------------------

	ArrayInfo* info = &ArrayInfos[ GCUL_SPHERICALFRUSTUM_ARRAY ];

	info->size		= size;
	info->type		= ( GCUL_ArrayDataType )type;
	info->pointer	= pointer;
}

void __stdcall gculBoxesPointer( GCULuint size, GCULenum type, const GCULvoid* pointer )
{
	//--------------------
	// Pre-conditions
	ARRAY_ASSERT();
	//--------------------

	ArrayInfo* info = &ArrayInfos[ GCUL_BBOXES_ARRAY ];

	info->size		= size;
	info->type		= ( GCUL_ArrayDataType )type;
	info->pointer	= pointer;

	tree = new BVHTree();
	tree->buildFromAABBList((aabb_bvh_t*)pointer, size);
}

void __stdcall gculSpheresPointer( GCULuint size, GCULenum type, const GCULvoid* pointer )
{
	//--------------------
	// Pre-conditions
	ARRAY_ASSERT();
	//--------------------

	ArrayInfo* info = &ArrayInfos[ GCUL_BSPHERES_ARRAY ];

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
		assert( gculIsEnableArray( GCUL_PYRAMIDALFRUSTUMCORNERS_ARRAY ), "Frustum corners not available to process the frustum culling." );

		return ProcessPyramidalFrustumAABBoxCulling( result );
	}
	else if( currentFrustumType == FRUSTUMTYPE_PYRAMIDAL && currentBoundingVolumeType == BOUNDINGVOLUMETYPE_SPHERE )
	{
		return ProcessPyramidalFrustumSphereCulling( result );
	}
	else if( currentFrustumType == FRUSTUMTYPE_SPHERICAL && currentBoundingVolumeType == BOUNDINGVOLUMETYPE_SPHERE )
	{
		return ProcessSphericalFrustumSphereCulling( result );
	}
	else
	{
		assert( false, "Not implemented yet." );
		return -1;
	}
}

int ProcessPyramidalFrustumAABBoxCulling( GCUL_Classification* result )
{
//// FIRST PASS //////////////////////////////////////////////////////////////

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
	cuda_call( cudaMalloc( &pointPlaneIntersection, (frustumPlanesInfo.size*6) * (boundingBoxesInfo.size*8) * sizeof( char )) );

	// Compute Block/Grid sizes.
	int planeCount = frustumPlanesInfo.size * 6;	// six planes per frustum.
	int pointCount = boundingBoxesInfo.size * 8;	// eight corners per box.

	dim3 dimBlock1stPass;
	dim3 dimGrid1stPass;

	ComputeGridSizes( 
		planeCount, 
		pointCount, 
		ClassifyPlanesPointsEnv.desiredThreadPerBlock,
		dimGrid1stPass.x, 
		dimGrid1stPass.y, 
		dimBlock1stPass.x, 
		dimBlock1stPass.y 
	);

	// Process first pass : intersect each plane with each box point.
	ClassifyPlanesPoints( 
		dimGrid1stPass,
		dimBlock1stPass,
		frustumsPlanes, 
		boundingBoxes, 
		frustumPlanesInfo.size * 6, 
		boundingBoxesInfo.size * 8, 
		(char*)pointPlaneIntersection 
	);

	// Free device input memory.
	FreeDeviceMemory( frustumsPlanes );
//// FIRST PASS //////////////////////////////////////////////////////////////
	
//// SECOND PASS /////////////////////////////////////////////////////////////

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

	dim3 dimBlock2ndPass;
	dim3 dimGrid2ndPass;

	int frustumCount	= frustumPlanesInfo.size;
	int boxCount		= boundingBoxesInfo.size;

	ComputeGridSizes( 
		frustumCount, 
		boxCount, 
		ClassifyPlanesPointsEnv.desiredThreadPerBlock,
		dimGrid2ndPass.x, 
		dimGrid2ndPass.y, 
		dimBlock2ndPass.x, 
		dimBlock2ndPass.y 
	);

	// Process second pass : determine from first pass output the intersection between each frustum with each box.
	ClassifyPyramidalFrustumBoxes( 
		dimGrid2ndPass,
		dimBlock2ndPass,
		(const float*)frustumsCorners, 
		(const float*)boundingBoxes,
		(const char*)pointPlaneIntersection, 
		frustumPlanesInfo.size * 6, 
		boundingBoxesInfo.size * 8, 
		(int*)resultDeviceMemory 
	);
//// SECOND PASS ////////////////////////////////////////////////////////////

//// THIRD PASS /////////////////////////////////////////////////////////////
	ComputeGridSizes( 
		frustumCount, 
		boxCount, 
		InverseClassifyPyramidalFrustumBoxesEnv.desiredThreadPerBlock,
		dimGrid2ndPass.x, 
		dimGrid2ndPass.y, 
		dimBlock2ndPass.x, 
		dimBlock2ndPass.y 
	);

	// Process second pass : determine from first pass output the intersection between each frustum with each box.
	InverseClassifyPyramidalFrustumBoxes( 
		dimGrid2ndPass,
		dimBlock2ndPass,
		(const float*)frustumsCorners, 
		(const float*)boundingBoxes,
		frustumPlanesInfo.size * 6, 
		boundingBoxesInfo.size * 8, 
		(int*)resultDeviceMemory 
	);
//// THIRD PASS //////////////////////////////////////////////////////////////

	// Copy the result from device memory.
	cutilSafeCall(cudaMemcpy( result, resultDeviceMemory, frustumPlanesInfo.size * boundingBoxesInfo.size * sizeof(int), cudaMemcpyDeviceToHost));

	//If occlusion cullng is enabled, process it.
	if( EnableStates[GCUL_OCCLUSION_CULLING] )
	{
		int ret = ProcessPyramidalFrustumAABBOcclusionCulling((float*)boundingBoxes, (float*)frustumsCorners, boxCount, frustumCount, 10, 10, (int*)resultDeviceMemory, (int*)result);
	}
	//

	// Free device input memory.
	FreeDeviceMemory( frustumsCorners		 );
	FreeDeviceMemory( boundingBoxes			 );
	FreeDeviceMemory( pointPlaneIntersection );

	FreeDeviceMemory( resultDeviceMemory );

	//--------------------

	return 0;
}

int ProcessPyramidalFrustumSphereCulling( GCUL_Classification* result )
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
	const ArrayInfo&	boundingSpheresInfo	= ArrayInfos[ GCUL_BSPHERES_ARRAY ];
	GCULvoid*			boundingSpheres		= NULL;

	AllocArrayDeviceMemory( &boundingSpheres, boundingSpheresInfo );

	if( boundingSpheres == NULL )
	{
		assert( false, "Can not allocate device memory for bounding volumes." );
		return -1;
	}

	// Initialize input data on device memory.
	CopyArrayToDeviceMemory( frustumsPlanes,  frustumPlanesInfo	  );
	CopyArrayToDeviceMemory( boundingSpheres, boundingSpheresInfo );

	// Compute Block/Grid sizes.
	int planeCount  = frustumPlanesInfo.size * 6;	// six planes per frustum.
	int sphereCount = boundingSpheresInfo.size;		// 

	// Allocate the matrix for the first pass.
	GCULvoid* spherePlaneIntersection = NULL;
	cuda_call( cudaMalloc( &spherePlaneIntersection, frustumPlanesInfo.size * 6 * sphereCount * sizeof( char ) ) );

	dim3 dimBlock1stPass;
	dim3 dimGrid1stPass;

	ComputeGridSizes( 
		planeCount, 
		sphereCount, 
		ClassifyPlanesSpheresEnv.desiredThreadPerBlock,
		dimGrid1stPass.x, 
		dimGrid1stPass.y, 
		dimBlock1stPass.x, 
		dimBlock1stPass.y 
		);

	// Process first pass : intersect each plane with each box point.
	ClassifyPlanesSpheres( 
		dimGrid1stPass,
		dimBlock1stPass,
		frustumsPlanes, 
		boundingSpheres, 
		planeCount, 
		sphereCount, 
		(char*)spherePlaneIntersection 
		);

	// Free device input memory.
	FreeDeviceMemory( frustumsPlanes  );
	FreeDeviceMemory( boundingSpheres );

	char* dest = new char[ frustumPlanesInfo.size * 6 * sphereCount ];
	cudaMemcpy( dest, spherePlaneIntersection, frustumPlanesInfo.size * 6 * sphereCount * sizeof( char ), cudaMemcpyDeviceToHost );

	//--------------------

	//--------------------
	// Second pass.

	// Allocate the result matrix on device memory.
	GCULvoid* resultDeviceMemory = NULL;	
	AllocResultDeviceMemory( &resultDeviceMemory, frustumPlanesInfo, boundingSpheresInfo );

	dim3 dimBlock2ndPass;
	dim3 dimGrid2ndPass;

	int frustumCount = frustumPlanesInfo.size;

	ComputeGridSizes( 
		frustumCount, 
		sphereCount, 
		ClassifyPyramidalFrustumSpheresEnv.desiredThreadPerBlock,
		dimGrid2ndPass.x, 
		dimGrid2ndPass.y, 
		dimBlock2ndPass.x, 
		dimBlock2ndPass.y 
		);

	// Process second pass : determine from first pass output the intersection between each frustum with each sphere.
	ClassifyPyramidalFrustumSpheres( 
		dimGrid2ndPass,
		dimBlock2ndPass,
		(const char*)spherePlaneIntersection, 
		frustumCount, 
		sphereCount, 
		(int*)resultDeviceMemory 
		);

	// Free device input memory.
	FreeDeviceMemory( spherePlaneIntersection );

	// Copy the result from device memory.
	cutilSafeCall(cudaMemcpy( result, resultDeviceMemory, frustumCount * sphereCount * sizeof(int), cudaMemcpyDeviceToHost));

	FreeDeviceMemory( resultDeviceMemory );

	//--------------------

	return 0;
}

int ProcessSphericalFrustumSphereCulling( GCUL_Classification* result )
{
	// Allocate spherical frustums on device memory.
	const ArrayInfo&	frustumInfo	= ArrayInfos[ GCUL_SPHERICALFRUSTUM_ARRAY ];
	GCULvoid*			frustums	= NULL;

	AllocArrayDeviceMemory( &frustums, frustumInfo );

	if( frustums == NULL )
	{
		assert( false, "Can not allocate device memory for spherical frustum." );
		return -1;
	}

	// Allocate bounding volumes on device memory.
	const ArrayInfo&	boundingSpheresInfo	= ArrayInfos[ GCUL_BSPHERES_ARRAY ];
	GCULvoid*			boundingSpheres		= NULL;

	AllocArrayDeviceMemory( &boundingSpheres, boundingSpheresInfo );

	if( boundingSpheres == NULL )
	{
		assert( false, "Can not allocate device memory for bounding volumes." );
		return -1;
	}

	// Initialize input data on device memory.
	CopyArrayToDeviceMemory( frustums,			frustumInfo			);
	CopyArrayToDeviceMemory( boundingSpheres,	boundingSpheresInfo );

	// Allocate the result matrix on device memory.
	GCULvoid* resultDeviceMemory = NULL;	
	AllocResultDeviceMemory( &resultDeviceMemory, frustumInfo, boundingSpheresInfo );

	dim3 dimBlock1stPass;
	dim3 dimGrid1stPass;

	int frustumCount = frustumInfo.size;
	int sphereCount  = boundingSpheresInfo.size;

	ComputeGridSizes( 
		frustumCount, 
		sphereCount, 
		ClassifySphericalFrustumSpheresEnv.desiredThreadPerBlock,
		dimGrid1stPass.x, 
		dimGrid1stPass.y, 
		dimBlock1stPass.x, 
		dimBlock1stPass.y 
		);

	// Process first pass : intersect each plane with each box point.
	ClassifySphericalFrustumSpheres( 
		dimGrid1stPass,
		dimBlock1stPass,
		( float* )frustums, 
		( float* )boundingSpheres, 
		frustumCount, 
		sphereCount, 
		( int* )resultDeviceMemory 
		);

	// Free device input memory.
	FreeDeviceMemory( frustums		  );
	FreeDeviceMemory( boundingSpheres );

	// Copy the result from device memory.
	cutilSafeCall( cudaMemcpy( result, resultDeviceMemory, frustumCount * sphereCount * sizeof(int), cudaMemcpyDeviceToHost ) );

	FreeDeviceMemory( resultDeviceMemory );

	return 0;
}
