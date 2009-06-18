#include <gpuCuller_internal.h>
#include <cutil_inline.h>
#include <cfloat>

extern bool ArrayStates[ GCUL_END_ARRAY ];
extern bool EnableStates[ GCUL_END_ENABLESTATE ];

void __stdcall gculInitialize( int argc, char** argv )
{
	// Initializes CUDA device
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );
}

void __stdcall gculEnable( GCUL_EnableState state )
{
	if( state < GCUL_END_ENABLESTATE )
	{
		EnableStates[ state ] = true;
	}
}

void __stdcall gculDisable( GCUL_EnableState state )
{
	if( state < GCUL_END_ENABLESTATE )
	{
		EnableStates[ state ] = false;
	}
}

void __stdcall gculEnableArray( GCUL_Array array )
{
	if( array < GCUL_END_ARRAY )
	{
		ArrayStates[ array ] = true;
	}
}

void __stdcall gculDisableArray( GCUL_Array array )
{
	if( array < GCUL_END_ARRAY )
	{
		ArrayStates[ array ] = false;
	}
}

bool __stdcall gculIsEnableArray( GCUL_Array array )
{
	if( array < GCUL_END_ARRAY )
	{
		return ArrayStates[ array ];
	}
	return false;
}

GCULint __stdcall gculMemoryUsed( GCUL_Array frustums, GCUL_Array boundingVolumes, GCULuint frustumCount, GCULuint volumeCount )
{
	return 0;
}

FrustumType CurrentFrustumType( void )
{
	if( ArrayStates[ GCUL_PYRAMIDALFRUSTUMPLANES_ARRAY ] )
	{
		assert( ArrayStates[ GCUL_SPHERICALFRUSTUM_ARRAY ] == false, "Pyramidal and spherical frustums can not be used at the same time." );

		return FRUSTUMTYPE_PYRAMIDAL;
	}

	if( ArrayStates[ GCUL_SPHERICALFRUSTUM_ARRAY ] )
	{
		assert( ArrayStates[ GCUL_PYRAMIDALFRUSTUMCORNERS_ARRAY ] == false, "Pyramidal and spherical frustums can not be used at the same time." );

		return FRUSTUMTYPE_SPHERICAL;
	}
	
	return FRUSTUMTYPE_UNDEFINED;
}

BoundingVolumeType CurrentBoundingVolumeType( void )
{
	if( ArrayStates[ GCUL_BBOXES_ARRAY ] )
	{
		assert( ArrayStates[ GCUL_BSPHERES_ARRAY ] == false, "Boxes and spheres bounding volumes can not be used at the same time." );

		return BOUNDINGVOLUMETYPE_BOX;
	}

	if( ArrayStates[ GCUL_BSPHERES_ARRAY ] )
	{
		return BOUNDINGVOLUMETYPE_SPHERE;
	}
	
	return BOUNDINGVOLUMETYPE_UNDEFINED;
}

void AllocArrayDeviceMemory( GCULvoid** pointer, const ArrayInfo& info )
{
	int size = info.size * info.elementWidth * SizeInBytes( info.type );
	
	cuda_call( cudaMalloc( ( void** )pointer, size ) );
}

void CopyArrayToDeviceMemory( GCULvoid* array, const ArrayInfo& info )
{
	int size = info.size * info.elementWidth * SizeInBytes( info.type );

	cuda_call( cudaMemcpy( array, info.pointer, size, cudaMemcpyHostToDevice ) );
}

void AllocResultDeviceMemory( GCULvoid** memory, const ArrayInfo& frustumInfo, const ArrayInfo& boundingInfo )
{
	cuda_call( cudaMalloc( memory, frustumInfo.size * boundingInfo.size * sizeof( GCUL_Classification ) ) );
}

void FreeDeviceMemory( GCULvoid* memory )
{
	if( memory != NULL )
	{
		cuda_call( cudaFree( memory ) );
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

void ComputeGridSizes( int threadWidth, int threadHeight, int desiredThreadPerBlock, unsigned int& gridDimX, unsigned int& gridDimY, unsigned int& blockDimX, unsigned int& blockDimY )
{
	static cudaDeviceProp deviceProp;
	cuda_call( cudaGetDeviceProperties( &deviceProp, cutGetMaxGflopsDeviceId() ) );

	// Get some information about the device.
	static int maxGridDimX			= deviceProp.maxGridSize[0];
	static int maxGridDimY			= deviceProp.maxGridSize[1];
	static int maxThreadPerBlock	= deviceProp.maxThreadsPerBlock;
	static int maxRegisterPerBlock	= deviceProp.regsPerBlock;

	int dimBlock = (int)sqrt( (double)desiredThreadPerBlock );

	gridDimX = (int)ceil( (double)threadWidth / (double)dimBlock );
	gridDimY = (int)ceil( (double)threadHeight / (double)dimBlock );
	blockDimX = dimBlock;
	blockDimY = dimBlock;

	assert( gridDimX <= maxGridDimX, "Maximum grid size reached." );
	assert( gridDimY <= maxGridDimY, "Maximum grid size reached." );

	assert( blockDimX * blockDimY <= maxThreadPerBlock, "Maximum thread per block reached." );
}
