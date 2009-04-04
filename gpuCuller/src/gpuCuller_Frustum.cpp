#include <gpuCuller_internal.h>
#include <cutil_inline.h>
#include <gpuCuller_kernel.h>

extern ArrayInfo ArrayInfos[ GCUL_END_ARRAY ];

#define ARRAY_ASSERT()	assert( pointer != NULL,	"Null pointer detected." ); \
						assert( type == GCUL_INT	|| \
								type ==	GCUL_FLOAT  || \
								type ==	GCUL_DOUBLE, "Invalid type." ) \

void __stdcall gculPyramidalFrustumPointer( GCULuint size, GCULenum type, GCULsizei stride, const GCULvoid* pointer )
{
	//--------------------
	// Pre-conditions
	ARRAY_ASSERT();
	//--------------------

	ArrayInfo* info = &ArrayInfos[ GCUL_PYRAMIDALFRUSTUM_ARRAY ];

	info->size		= size;
	info->type		= ( GCUL_ArrayDataType )type;
	info->stride	= stride;
	info->pointer	= pointer;
}

void __stdcall gculBoxesPointer( GCULuint size, GCULenum type, GCULsizei stride, const GCULvoid* pointer )
{
	//--------------------
	// Pre-conditions
	ARRAY_ASSERT();
	//--------------------

	ArrayInfo* info = &ArrayInfos[ GCUL_BBOXES_ARRAY ];

	info->size		= size;
	info->type		= ( GCUL_ArrayDataType )type;
	info->stride	= stride;
	info->pointer	= pointer;
}

GCULint __stdcall gculProcessFrustumCulling( const GCULuint gridSize[ 2 ], const GCULuint blockSize[ 3 ], GCUL_Classification* result )
{
	//--------------------
	// Pre-conditions
	assert( result != NULL, "The pointer result must be not null." );
	//--------------------

	// Determine the frustum matrix to use.
	GCULenum currentFrustumArray = CurrentFrustumArray();

	if( currentFrustumArray == GCUL_END_ARRAY )
	{
		assert( false, "No frustum array enabled." );
		return -1;
	}

	// Determine the bouding volumes matrix to use.
	GCULenum currentBoundingObjectArray = CurrentBoundingObjectArray();

	if( currentBoundingObjectArray == GCUL_END_ARRAY )
	{
		assert( false, "No bounding volume array enabled." );
		return -1;
	}

	// Allocate frustums on device memory.
	const ArrayInfo&	frustumInfo	= ArrayInfos[ currentFrustumArray ];
	GCULvoid*			frustums	= NULL;

	AllocArrayDeviceMemory( &frustums, frustumInfo );

	if( frustums == NULL )
	{
		assert( false, "Can not allocate device memory for frustums." );
		return -1;
	}

	// Allocate bounding volumes on device memory.
	const ArrayInfo&	boundingInfo	= ArrayInfos[ currentFrustumArray ];
	GCULvoid*			boundingVolumes	= NULL;

	AllocArrayDeviceMemory( &boundingVolumes, boundingInfo );

	if( boundingVolumes == NULL )
	{
		assert( false, "Can not allocate device memory for bounding volumes." );
		return -1;
	}

	// Initialize input data on device memory.
	CopyArrayToDeviceMemory( frustums,			frustumInfo		);
	CopyArrayToDeviceMemory( boundingVolumes,	boundingInfo	);

	// Initialize output data on device memory.
	GCULvoid* resultDeviceMemory = NULL;
	AllocResultDeviceMemory( &resultDeviceMemory, frustumInfo, boundingInfo );

	if( currentFrustumArray == GCUL_PYRAMIDALFRUSTUM_ARRAY && currentBoundingObjectArray == GCUL_BBOXES_ARRAY )
	{
		// Process first pass : intersect each plane with each box point.
		
		// Allocate the matrix for the first pass.
		GCULvoid* pointPlaneIntersection = NULL;
		cudaMalloc( &pointPlaneIntersection, frustumInfo.size * boundingInfo.size * sizeof( int ) );

		dim3 dimBlock(12, 12);
		dim3 dimGrid(10 / dimBlock.x, 10 / dimBlock.y);

		ClassifyPlanesPoints( 
			dimBlock,
			dimGrid,
			frustumInfo.pointer, 
			boundingInfo.pointer, 
			frustumInfo.size * 6, 
			boundingInfo.size * 8, 
			(int*)pointPlaneIntersection 
		);

		// Free device input memory.
		FreeDeviceMemory( frustums		  );
		FreeDeviceMemory( boundingVolumes );

		// Process second pass : determine from first pass the intersection between each frustum with each box.
	}
	else
	{
		assert( false, "Not implemented yet." );
	}
	
	FreeDeviceMemory( resultDeviceMemory );

	return 0;
}
