#ifndef __GPUCULLER_INTERNAL_H__
#define __GPUCULLER_INTERNAL_H__

#include <gpuCuller.h>
#include <iostream>
#include <windows.h>

#ifdef _DEBUG
	#define assert( condition, message ) if( !(condition) ) { std::cerr<< message << __FILE__ << __LINE__ << std::endl; DebugBreak(); }
#else
	#define assert( condition, message )
#endif

#ifdef _DEBUG
	#define check_cuda_error( ) cudaError_t err = cudaGetLastError(); assert( cudaSuccess == err, "Cuda error " +  cudaGetErrorString( err ) )
#else
	#define check_cuda_error( ) 
#endif

struct ArrayInfo
{
	GCULuint			size;
	GCULint				elementWidth;
	GCUL_ArrayDataType	type;
	const GCULvoid*		pointer;
};
#define UndefinedArrayInfo( elementWidth ) { 0, elementWidth, GCUL_UNKNOWN, NULL }

enum FrustumType
{
	FRUSTUMTYPE_PYRAMIDAL = 0,
	FRUSTUMTYPE_SPHERICAL,
	FRUSTUMTYPE_UNDEFINED
};

enum BoundingVolumeType
{
	BOUNDINGVOLUMETYPE_BOX = 0,
	BOUNDINGVOLUMETYPE_SPHERE,
	BOUNDINGVOLUMETYPE_UNDEFINED
};

//--------------------
// Functions
//--------------------

FrustumType CurrentFrustumType( void );

BoundingVolumeType CurrentBoundingVolumeType( void );

void AllocArrayDeviceMemory( GCULvoid** array, const ArrayInfo& info );

void CopyArrayToDeviceMemory( GCULvoid* array, const ArrayInfo& info );

void AllocResultDeviceMemory( GCULvoid** memory, const ArrayInfo& frustumInfo, const ArrayInfo& boundingInfo );

void FreeDeviceMemory( GCULvoid* memory );

int SizeInBytes( GCUL_ArrayDataType type );

int ProcessPyramidalFrustumAABBoxCulling( GCUL_Classification* result );

void ComputeGridSizes( int threadWidth, int threadHeight, unsigned int& gridDimX, unsigned int& gridDimY, unsigned int& blockDimX, unsigned int& blockDimY );

#endif // __GPUCULLER_INTERNAL_H__