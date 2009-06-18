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
	
	int nbThreadY = floor((float)256/(float)(rayCoverageWidth * rayCoverageHeight));
	gridSize.x = frustumCount;
	gridSize.y = ceil((float)boxCount / (float)nbThreadY);
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
	for( int i = 0; i < boxCount; ++i )
	{
		for( int j = 0; j < frustumCount; ++j )
		{
			if( classificationResultHost[ frustumCount*i + j ] == GCUL_INSIDE ||
				classificationResultHost[ frustumCount*i + j ] == GCUL_SPANNING )
			{
				classificationResultHost[ frustumCount*i + j ] = GCUL_OCCLUDED;
			}
		}
	}

	for( int i = 0; i < frustumCount*rayCoverageWidth*rayCoverageHeight; ++i )
	{
		float tmin = 40000000;
		int boxMin = -1;

		int frustumIndex;
		frustumIndex = i/(rayCoverageWidth*rayCoverageHeight);

		if( frustumIndex != 0 )
			int lol = 67;

		//Pour chaque boite
		for( int j = 0; j < boxCount; ++j )
		{
			int collDistIndex = (j*(frustumCount*rayCoverageWidth*rayCoverageHeight))+i;
			float dist = collisionDistance_h[collDistIndex];

			if( dist < tmin && dist > 0 )
			{
				tmin = dist;
				boxMin = j;
			}
		}


		if( boxMin != -1 )
		{
			//printf("Inclusion - F#%d B#%d\n", frustumIndex, boxMin );
			//classificationResultHost[ frustumIndex * boxCount + boxMin ] = GCUL_INSIDE;
			classificationResultHost[ frustumCount * boxMin + frustumIndex ] = GCUL_INSIDE; 
		}
	}
	//

	//;
	delete collisionDistance_h;
	//

	return 0;
}