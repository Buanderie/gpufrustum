#ifndef __GPUCULLER_INTERNAL_H__
#define __GPUCULLER_INTERNAL_H__

#include <gpuCuller.h>
#include <cutil_inline.h>
#include <iostream>
#include <windows.h>

#ifdef _DEBUG
	#define assert( condition, message ) if( !(condition) ) { std::cerr<< message << __FILE__ << __LINE__ << std::endl; DebugBreak(); }
#else
	#define assert( condition, message )
#endif

#ifdef _DEBUG
	#define check_cuda_error( ) cudaError_t err = cudaGetLastError(); assert( cudaSuccess == err, cudaGetErrorString( err ) )
#else
	#define check_cuda_error( ) 
#endif

#ifdef _DEBUG
	#define cuda_call( cudafunc ) cudaError_t err = cudafunc; assert( err == cudaSuccess, cudaGetErrorString( err ) )
#else
	#define cuda_call( cudafunc ) cudafunc
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

struct DeviceFunctionEnv
{
	unsigned int registerPerThread;
	unsigned int sharedMemorySize;
	unsigned int desiredThreadPerBlock;
};


typedef struct occlusionray
{
	float3 start;
	float3 dir;
} occlusionray_t;

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

int ProcessPyramidalFrustumSphereCulling( GCUL_Classification* result );

int ProcessSphericalFrustumSphereCulling( GCUL_Classification* result );

int ProcessSphericalFrustumAABoxCulling( GCUL_Classification* result );

int ProcessPyramidalFrustumAABBOcclusionCulling(float* boxPoints, float* frustumCorners, int boxCount, int frustumCount, int rayCoverageWidth, int rayCoverageHeight, int* classificationResult, int* classificationResultHost);

void ComputeGridSizes( int threadWidth, int threadHeight, int desiredThreadPerBlock, unsigned int& gridDimX, unsigned int& gridDimY, unsigned int& blockDimX, unsigned int& blockDimY );

#endif // __GPUCULLER_INTERNAL_H__