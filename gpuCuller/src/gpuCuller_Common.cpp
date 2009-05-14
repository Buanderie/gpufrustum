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

	check_cuda_error();
}

void CopyArrayToDeviceMemory( GCULvoid* array, const ArrayInfo& info )
{
	int size = info.size * info.elementWidth * SizeInBytes( info.type );

	cudaMemcpy( array, info.pointer, size, cudaMemcpyHostToDevice);

	check_cuda_error();
}

void AllocResultDeviceMemory( GCULvoid** memory, const ArrayInfo& frustumInfo, const ArrayInfo& boundingInfo )
{
	cudaMalloc( memory, frustumInfo.size * boundingInfo.size * sizeof( GCUL_Classification ) );

	check_cuda_error();
}

void FreeDeviceMemory( GCULvoid* memory )
{
	if( memory != NULL )
	{
		cudaFree( memory );

		check_cuda_error();
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

void ComputeGridSizes( int threadWidth, int threadHeight, const DeviceFunctionEnv& functionEnv, unsigned int& gridDimX, unsigned int& gridDimY, unsigned int& blockDimX, unsigned int& blockDimY )
{
	static cudaDeviceProp deviceProp;
	cutilSafeCall(cudaGetDeviceProperties(&deviceProp, cutGetMaxGflopsDeviceId()));

	// Get some information about the device.
	static int maxGridDimX			= deviceProp.maxGridSize[0];
	static int maxGridDimY			= deviceProp.maxGridSize[1];
	static int maxThreadPerBlock	= deviceProp.maxThreadsPerBlock;
	static int maxRegisterPerBlock	= deviceProp.regsPerBlock;

	// Choose the right number of thread per block according to
	// the number of register used by the kernel.
	int preferedThreadPerBlock = min( maxThreadPerBlock, ( int )floor( ( float )maxRegisterPerBlock / functionEnv.registerPerThread ) );

	static int sizes[] = { 512, 256, 128, 64, 32, 16, 8, 4, 2, 1 };
	
	// Test some different block sizes to get the best.
	float minDelta  = FLT_MAX;
	int   bestX		= -1;
	int   bestY		= -1;
	for( int i = 0; i < 10 ; ++i )
	{
		int sizeX = sizes[ i ];
		
		for( int j = 0; j < i + 1 ; ++j )
		{
			int sizeY = sizes[ 9 - j ];

			// Make sure we do not reach the maximum of thread per block.
			if( sizeX * sizeY <= preferedThreadPerBlock && sizeX * sizeY > 16 ) 
			{
				float blockCountX = ( float )threadWidth  / sizeX;
				float blockCountY = ( float )threadHeight / sizeY;

				float deltaX = ( ceil( blockCountX ) - blockCountX ) * sizeX;
				float deltaY = ( ceil( blockCountY ) - blockCountY ) * sizeY;

				float deltaSum = deltaX + deltaY;
				if( deltaSum < minDelta )
				{
					minDelta = deltaSum;
					bestX	 = i;
					bestY	 = 9 - j;
				}
			}
		}
	}

	if( bestX == -1 && bestY == -1 )
	{
		assert( false, "Can not compute grid and block sizes." ); 
		return;
	}

	blockDimX = min( threadWidth,  sizes[ bestX ] );
	blockDimY = min( threadHeight, sizes[ bestY ] );
	
	assert( blockDimX * blockDimY <= ( unsigned int )deviceProp.maxThreadsPerBlock, "Max number of threads per block reached." );

	gridDimX = ( int )ceil( threadWidth  / ( float )blockDimX );
	gridDimY = ( int )ceil( threadHeight / ( float )blockDimY );

	assert( gridDimX <= ( unsigned int )maxGridDimX, "Max width of the grid reached." );
	assert( gridDimY <= ( unsigned int )maxGridDimY, "Max height of the grid reached." );
}
