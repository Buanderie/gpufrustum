#ifndef __GPUCULLER_LBVH_KERNEL_H__
#define __GPUCULLER_LBVH_KERNEL_H__

#include <gpuCuller_internal.h>

//LBVH Device Code

__global__ void AssignMortonCode(	aabb_t* aabbPtr, bvhnode_t* bvhPtr, unsigned int elementCount, unsigned int depth,
									float min_x,
									float min_y,
									float min_z,
									float max_x,
									float max_y,
									float max_z)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(i > elementCount )
		return;

	//Compute AABB centroid
	float cx = (aabbPtr[ i ].min_x + aabbPtr[ i ].max_x)/2.0f;
	float cy = (aabbPtr[ i ].min_y + aabbPtr[ i ].max_y)/2.0f;
	float cz = (aabbPtr[ i ].min_z + aabbPtr[ i ].max_z)/2.0f;
	bvhPtr[ i ].centroidX = cx;
	bvhPtr[ i ].centroidY = cy;
	bvhPtr[ i ].centroidZ = cz;
	//

	//Compute Morton Code
	unsigned int morton = 0;
	for( int k = depth-1; k >= 0; --k )
	{
		unsigned int tmp_code = 0;
		//Check X Axis
		if( cx >= (min_x+max_x)/2.0f )
		{
			tmp_code += 2;
			min_x = min_x + ((max_x-min_x)/2.0f);
		}
		else
			max_x = min_x + ((max_x-min_x)/2.0f);

		//Check Y Axis
		if( cy >= (min_y+max_y)/2.0f )
		{
			tmp_code += 1;
			min_y = min_y + ((max_y-min_y)/2.0f);
		}
		else
			max_y = min_y + ((max_y-min_y)/2.0f);
		
		morton += (unsigned int)(ceilf(powf( 10.0f, (float)k ))) * tmp_code;
	}
	bvhPtr[ i ].mortonCode = morton;
	//printf( "Morton Code = %i \n", morton );
	//
}

#endif