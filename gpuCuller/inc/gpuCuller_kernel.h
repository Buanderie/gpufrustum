#ifndef __GPUCULLER_KERNEL_H__
#define __GPUCULLER_KERNEL_H__

#include <gpuCuller_internal.h>

//--------------------
// Types
//--------------------

struct __align__(16) int6 
{
	int a, b, c, d, e, f;
};

struct __align__(4) char6 
{
	char a, b, c, d, e, f;
};

//--------------------
// Public interface
//--------------------

void ClassifyPlanesPoints( dim3 gridSize, dim3 blockSize, const void* iplanes, const void* ipoints, int nPlane, int nPoint, char* out );

void ClassifyPyramidalFrustumBoxes( dim3 gridSize, dim3 blockSize, const float* frustumCorners, const float* boxPoints, const char* planePointClassification, int planeCount, int pointCount, int* out );

void InverseClassifyPyramidalFrustumBoxes( dim3 gridSize, dim3 blockSize, const float* frustumCorners, const float* boxPoints, int planeCount, int pointCount, int* out );

void ClassifyPlanesSpheres( dim3 gridSize, dim3 blockSize, const void* planes, const void* spheres, int planeCount, int sphereCount, char* out );

void ClassifyPyramidalFrustumSpheres( dim3 gridSize, dim3 blockSize, const char* planeSphereClassification, int frustumCount, int sphereCount, int* out );

//--------------------
// Kernels
//--------------------

__global__ void
ClassifyPlanesPoints( const float4* iplanes, const float3* ipoints, int planeCount, int pointCount, char* out );

__global__ void
ClassifyPyramidalFrustumBoxes( const float3* frustumCorners, const float3* boxPoints, const char* planePointClassification, int planeCount, int pointCount, int* out );

__global__ void
InverseClassifyPyramidalFrustumBoxes( const float3* frustumCorners, const float3* boxPoints, int planeCount, int pointCount, int* out );

__global__ void
ClassifyPlanesSpheres( const float4* planes, const float4* spheres, int planeCount, int sphereCount, char* out );

__global__ void
ClassifyPyramidalFrustumSpheres( const char6* planeSphereClassification, int frustumCount, int sphereCount, int* out );

//--------------------
// Device functions
//--------------------

__device__ int 
SumArrayElements( const int* array, int elementCount );

__device__ int
CountArrayElementValue( const int* array, int elementCount, int elementValue );

__device__ float3
UpperPoint( const float3* box );

__device__ float3
LowerPoint( const float3* box );


#endif // __GPUCULLER_KERNEL_H__