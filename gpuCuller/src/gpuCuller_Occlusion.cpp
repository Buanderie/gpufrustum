#include <cutil_inline.h>
#include <gpuCuller.h>
#include <gpuCuller_internal.h>
#include <gpuCuller_kernel.h>

int ProcessPyramidalFrustumAABBOcclusionCulling(float* boxPoints, float* frustumCorners, int boxCount, int frustumCount, int rayCoverageWidth, int rayCoverageHeight, int* classificationResult)
{
	GCULvoid* rays_d = NULL;
	//occlusionray_t* rays_h = new occlusionray_t[ rayCoverageWidth*rayCoverageHeight*frustumCount ];
	cuda_call( cudaMalloc( &rays_d, sizeof( occlusionray_t )*rayCoverageWidth*rayCoverageHeight*frustumCount) );
	dim3 gridSize;
	dim3 blockSize;
	gridSize.x = frustumCount;
	gridSize.y = 1;
	blockSize.x = rayCoverageWidth;
	blockSize.y = rayCoverageHeight;
	GenerateOcclusionRay( gridSize, blockSize, boxPoints, frustumCorners, boxCount, frustumCount, rayCoverageWidth, rayCoverageHeight, classificationResult, (occlusionray_t*)rays_d);
	//cudaMemcpy( rays_h, rays_d, sizeof(occlusionray_t)*(rayCoverageWidth*rayCoverageHeight*frustumCount), cudaMemcpyDeviceToHost);
	cudaFree( rays_d );
	return 0;
}