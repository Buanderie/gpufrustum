#include <gpuCuller_internal.h>
#include <cutil_inline.h>
#include <cfloat>

extern bool ArrayStates[ GCUL_END_ARRAY ];

void __stdcall gculInitialize( int argc, char** argv )
{
	// Initializes CUDA device
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );
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

FrustumType CurrentFrustumType( void )
{
	if( ArrayStates[ GCUL_PYRAMIDALFRUSTUMPLANES_ARRAY ] && ArrayStates[ GCUL_PYRAMIDALFRUSTUMCORNERS_ARRAY ] )
	{
		assert( ArrayStates[ GCUL_SPHERICALFRUSTUM_ARRAY ] == false, "Pyramidal and spherical frustums can not be used at the same time." );

		return FRUSTUMTYPE_PYRAMIDAL;
	}

	if( ArrayStates[ GCUL_SPHERICALFRUSTUM_ARRAY ] )
	{
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
	
	cudaMalloc((void**)pointer, size);
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

void ComputeGridSizes( int threadWidth, int threadHeight, unsigned int& gridDimX, unsigned int& gridDimY, unsigned int& blockDimX, unsigned int& blockDimY )
{
	static cudaDeviceProp deviceProp;
	cutilSafeCall(cudaGetDeviceProperties(&deviceProp, cutGetMaxGflopsDeviceId()));

	static int maxGridDimX = deviceProp.maxGridSize  [0];
	static int maxGridDimY = deviceProp.maxGridSize  [1];

	static int sizes[] = { 512, 256, 128, 64, 32, 16, 8, 4, 2, 1 };
	
	float minDelta  = FLT_MAX;
	int   best		= 0;
	for( int i = 0; i < 10 ; ++i )
	{
		float blockCountX = ( float )threadWidth  / sizes[ i     ];
		float blockCountY = ( float )threadHeight / sizes[ 9 - i ];

		float deltaX = ( ceil( blockCountX ) - blockCountX ) * sizes[ i	 ];
		float deltaY = ( ceil( blockCountY ) - blockCountY ) * sizes[ 9 - i ];

		float deltaSum = deltaX + deltaY;
		if( deltaSum < minDelta )
		{
			minDelta = deltaSum;
			best	 = i;
		}
	}

	blockDimX = min( threadWidth,  sizes[ best     ] );
	blockDimY = min( threadHeight, sizes[ 9 - best ] );
	
	assert( blockDimX * blockDimY <= ( unsigned int )deviceProp.maxThreadsPerBlock, "Max number of threads per block reached." );

	gridDimX = ( int )ceil( threadWidth  / ( float )blockDimX );
	gridDimY = ( int )ceil( threadHeight / ( float )blockDimY );

	assert( gridDimX <= ( unsigned int )maxGridDimX, "Max width of the grid reached." );
	assert( gridDimY <= ( unsigned int )maxGridDimY, "Max height of the grid reached." );
}
