#include <cutil_inline.h>
#include <gpuCuller.h>
#include <gpuCuller_internal.h>
#include <gpuCuller_kernel.h>

int ProcessPyramidalFrustumAABBOcclusionCulling(float* boxPoints, float* frustumCorners, int boxCount, int frustumCount, int rayCoverageWidth, int rayCoverageHeight, int* classificationResult, int* classificationResultHost)
{
	GCULvoid* rays_d = NULL;
	cuda_call( cudaMalloc( &rays_d, sizeof( occlusionray_t )*rayCoverageWidth*rayCoverageHeight*frustumCount) );
	dim3 gridSize;
	dim3 blockSize;
	gridSize.x = frustumCount;
	gridSize.y = 1;
	blockSize.x = rayCoverageWidth;
	blockSize.y = rayCoverageHeight;
	GenerateOcclusionRay( gridSize, blockSize, boxPoints, frustumCorners, boxCount, frustumCount, rayCoverageWidth, rayCoverageHeight, classificationResult, (occlusionray_t*)rays_d);
	
	int nbThreadY = 2;
	gridSize.x = frustumCount;
	gridSize.y = boxCount / nbThreadY;
	blockSize.x = rayCoverageWidth * rayCoverageHeight;
	blockSize.y = nbThreadY;
	
	float* collisionDistance_h = new float[ boxCount * frustumCount * rayCoverageWidth * rayCoverageHeight ];
	void* collisionDistance_d;
	cudaMalloc( &collisionDistance_d, sizeof( float )* boxCount * frustumCount * rayCoverageWidth * rayCoverageHeight );

	OcclusionRayIntersect(gridSize, blockSize, boxPoints, boxCount, frustumCount, rayCoverageWidth, rayCoverageHeight, (float*)collisionDistance_d, classificationResult, (occlusionray_t*)rays_d);
	
	cudaMemcpy( collisionDistance_h, collisionDistance_d, sizeof( float )* boxCount * frustumCount * rayCoverageWidth * rayCoverageHeight, cudaMemcpyDeviceToHost);
	
	cudaFree( rays_d );
	cudaFree( collisionDistance_d );

	//Passe de calcul de minimum...
	//Pour chaque rayon
	for( int i = 0; i < frustumCount*rayCoverageWidth*rayCoverageHeight; ++i )
	{
		float tmin = 40000000;
		int boxMin = 0;
		//Pour chaque boite
		for( int j = 0; j < boxCount; ++j )
		{
			float dist = collisionDistance_h[j*frustumCount*rayCoverageWidth*rayCoverageHeight+i];
			if( dist < tmin && dist > 0 )
			{
				tmin = dist;
				boxMin = j;
			}
		}

		int frustumIndex;
		frustumIndex = i/(rayCoverageWidth*rayCoverageHeight);
		
		//classificationResultHost[ resultOffset ] = GCUL_INSIDE;
		for( int k = 0; k < boxCount; ++k )
		{
			int resultOffset = frustumIndex * boxCount + k;
			if( classificationResultHost[ frustumCount * k + frustumIndex ] == GCUL_SPANNING ||
				classificationResultHost[ frustumCount * k + frustumIndex ] == GCUL_INSIDE )
			{
				if( k != boxMin )
					classificationResultHost[ resultOffset ] = GCUL_OCCLUDED;
			}
		}
	}
	//

	//;
	delete collisionDistance_h;
	//

	return 0;
}