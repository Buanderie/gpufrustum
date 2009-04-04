#include <gpuCuller_internal.h>
#include <cutil_inline.h>

extern bool ArrayStates[ GCUL_END_ARRAY ];

void __stdcall gculInitialize( int argc, char** argv )
{
	// Initializes CUDA device
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );
}

void __stdcall gculEnableArray( GCULenum array )
{
	if( array < GCUL_END_ARRAY )
	{
		ArrayStates[ array ] = true;
	}
}

void __stdcall gculDisableArray( GCULenum array )
{
	if( array < GCUL_END_ARRAY )
	{
		ArrayStates[ array ] = false;
	}
}

GCUL_Array CurrentFrustumArray( void )
{
	if( ArrayStates[ GCUL_PYRAMIDALFRUSTUM_ARRAY ] )
	{
		assert( ArrayStates[ GCUL_SPHERICALFRUSTUM_ARRAY ], "Pyramidal and spherical frustums can not be used at the same time." );

		return GCUL_PYRAMIDALFRUSTUM_ARRAY;
	}

	if( ArrayStates[ GCUL_SPHERICALFRUSTUM_ARRAY ] )
	{
		return GCUL_SPHERICALFRUSTUM_ARRAY;
	}
	
	return GCUL_END_ARRAY;
}

GCUL_Array CurrentBoundingObjectArray( void )
{
		if( ArrayStates[ GCUL_BBOXES_ARRAY ] )
	{
		assert( ArrayStates[ GCUL_BSPHERES_ARRAY ], "Boxes and spheres bounding volumes can not be used at the same time." );

		return GCUL_BBOXES_ARRAY;
	}

	if( ArrayStates[ GCUL_BSPHERES_ARRAY ] )
	{
		return GCUL_BSPHERES_ARRAY;
	}
	
	return GCUL_END_ARRAY;
}

void AllocArrayDeviceMemory( GCULvoid** pointer, const ArrayInfo& info )
{
	int size = info.size * info.elementWidth * SizeInBytes( info.type );
	
	cudaMalloc((void**)&pointer, size);
}

void CopyArrayToDeviceMemory( GCULvoid* array, const ArrayInfo& info )
{
	int size = info.size * info.elementWidth * SizeInBytes( info.type );

	cudaMemcpy( array, info.pointer, size, cudaMemcpyHostToDevice);
}

void AllocResultDeviceMemory( GCULvoid** memory, const ArrayInfo& frustumInfo, const ArrayInfo& boundingInfo )
{
	cudaMalloc( memory, frustumInfo.size * boundingInfo.size * sizeof( GCUL_Classification ) );
}

void FreeDeviceMemory( GCULvoid* memory )
{
	if( memory != NULL )
	{
		cudaFree( memory );
	}
}

int SizeInBytes( GCUL_ArrayDataType type )
{
	switch( type )
	{
	case GCUL_INT		: return sizeof( int	); 
	case GCUL_FLOAT		: return sizeof( float	);
	case GCUL_DOUBLE	: return sizeof( double ); 
	default				: assert( false, "Unknown type." ); return -1;
	}
}
