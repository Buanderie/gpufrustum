#ifndef __GPUCULLER_KERNEL_H__
#define __GPUCULLER_KERNEL_H__

#include <gpuCuller_internal.h>

struct plane
{
	float a;
	float b;
	float c;
	float d;
};

struct point3d
{
	float x;
	float y;
	float z;
};

void ClassifyPlanesPoints( dim3 gridSize, dim3 blockSize, const void* iplanes, const void* ipoints, int nPlane, int nPoint, int* out );

void ClassifyPyramidalFrustumBoxes( dim3 gridSize, dim3 blockSize, const float* frustumCorners, const float* boxPoints, const int* planePointClassification, int planeCount, int pointCount, int* out );

__global__ void
ClassifyPlanesPoints( const float4* iplanes, const float3* ipoints, int planeCount, int pointCount, int* out );

__global__ void
ClassifyPyramidalFrustumBoxes( const point3d* frustumCorners, const point3d* boxPoints, const int* planePointClassification, int planeCount, int pointCount, int* out );

__device__ int 
SumArrayElements( const int* array, int elementCount );

__device__ int
CountArrayElementValue( const int* array, int elementCount, int elementValue );

__device__ point3d
UpperPoint( const point3d* box );

__device__ point3d
LowerPoint( const point3d* box );


#endif // __GPUCULLER_KERNEL_H__