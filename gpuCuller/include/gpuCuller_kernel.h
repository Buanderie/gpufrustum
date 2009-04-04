#ifndef __GPUCULLER_KERNEL_H__
#define __GPUCULLER_KERNEL_H__

struct plane_t
{
	float a;
	float b;
	float c;
	float d;
};

struct point3d_t
{
	float x;
	float y;
	float z;
};

void ClassifyPlanesPoints( dim3 gridSize, dim3 blockSize, const void* iplanes, const void* ipoints, int nPlane, int nPoint, int* out );

__global__ void
classifyPlanePoint( const plane_t* iplanes, const point3d_t* ipoints, int nPlane, int nPoint, int* out );



#endif // __GPUCULLER_KERNEL_H__