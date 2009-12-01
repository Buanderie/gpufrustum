#ifndef __GPUCULLER_LBVH_KERNEL_H__
#define __GPUCULLER_LBVH_KERNEL_H__

#include <gpuCuller_internal.h>

//LBVH Device Utility Code
__device__ unsigned int extractDigit( unsigned number, const unsigned pos )
{
    for(int i = 0; i < pos; ++i)
        number /= 10;    //get rid of the preceding digits
    return number % 10;  //now ignore all of the following ones
}
//

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

	//Store the primitive index
	bvhPtr[ i ].primIndex = i;

	//Compute AABB centroid
	float cx = (aabbPtr[ i ].min.x + aabbPtr[ i ].max.x)/2.0f;
	float cy = (aabbPtr[ i ].min.y + aabbPtr[ i ].max.y)/2.0f;
	float cz = (aabbPtr[ i ].min.z + aabbPtr[ i ].max.z)/2.0f;
	bvhPtr[ i ].centroid.x = cx;
	bvhPtr[ i ].centroid.y = cy;
	bvhPtr[ i ].centroid.y = cz;
	//

	//Copy AABB data
	bvhPtr[ i ].bbox.min = aabbPtr[i].min;
	bvhPtr[ i ].bbox.max = aabbPtr[i].max;
	//

	//Compute Morton Code
	unsigned int morton = 0;
	for( int k = depth-1; k >= 0; --k )
	{
		unsigned int tmp_code = 1;
		//Check X Axis
		if( cx >= (min_x+max_x)/2.0f )
		{
			tmp_code += 2;
			min_x = ((max_x+min_x)/2.0f);
		}
		else
		{
			//tmp_code += 2;
			max_x = ((max_x+min_x)/2.0f);
		}

		//Check Y Axis
		if( cy >= (min_y+max_y)/2.0f )
		{
			tmp_code += 1;
			min_y = ((max_y+min_y)/2.0f);
		}
		else
		{
			//tmp_code += 1;
			max_y = ((max_y+min_y)/2.0f);
		}
		
		morton += (unsigned int)(ceilf(powf( 10.0f, (float)k ))) * tmp_code;
	}
	bvhPtr[ i ].mortonCode = morton;
	//printf( "Morton Code = %i \n", morton );
	//
}

__global__ void ComputeSplitLevel( bvhnode_t* bvhPtr, lbvhsplit_t* split, unsigned int elementCount, unsigned int maxDepth )
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	unsigned int code_a;
	unsigned int code_b;
	unsigned int a;
	unsigned int b;

	code_a = bvhPtr[ i ].mortonCode;
//
	//Nice stuff going on here
	if( i > 0 && i < elementCount-1 )
	{
		code_b = bvhPtr[ i+1 ].mortonCode;
	}
	else //Mother f-ing boundaries 
	{
		code_b = 999999999;
	}
	//
//

	for( int k = maxDepth-1; k>=0; --k )
	{
		//Extract the k-th digit of the two guys
		a = extractDigit( code_a, k );
		b = extractDigit( code_b, k );

		const unsigned splitInd = (maxDepth-k-1)*elementCount + i;
		//Compare the two digits
		if( a != b )
		{
			split[ splitInd ].level = (maxDepth-k);
			split[ splitInd ].primIndex = i;
		}
		else
		{
			split[ splitInd ].level = 999999999;
			split[ splitInd ].primIndex = 99999999;
		}
	}
	return;
}


__global__ void ComputeHNodeIntervals( lbvhsplit_t* split, hnode_t* hierarchy, unsigned int elementCount, unsigned int hierarchySize, unsigned maxDepth )
{
	
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if( i >= hierarchySize )
		return;

	//Assign unique ID to every hierarchy node
	hierarchy[ i ].ID = i;

	//If the split is a valid one...
	if( split[i].level <= maxDepth )
	{
		//Two splits of the same level give us one node
		if( split[ i ].level == split[ i+1 ].level )
		{
			hierarchy[ i ].splitLevel = split[ i ].level;
			
			//Special case for "most-left" nodes
			if( split[i].primIndex == 0 )
				hierarchy[ i ].primStart = split[ i ].primIndex;
			else
				hierarchy[ i ].primStart = split[ i ].primIndex + 1;
			//

			hierarchy[ i ].primStop = split[ i+1 ].primIndex;
		}
		else
		{
			//Discard the node
			//hierarchy[ i ].ID = 99999999;
			hierarchy[ i ].primStart = 99999999;
			hierarchy[ i ].primStop = 99999999;
			hierarchy[ i ].splitLevel = 99999999;
		}
	}
	else
	{
		//Discard the node
		//hierarchy[ i ].ID = 99999999;
		hierarchy[ i ].primStart = 99999999;
		hierarchy[ i ].primStop = 99999999;
		hierarchy[ i ].splitLevel = 99999999;
	}

	//Last check...
	//If the lower primitive bound is greater than or equal to the upper bound
	//we discard the node (~ ID = 99999999 and primStart=99999999 and primStop=99999999)
	if( hierarchy[ i ].primStart >= hierarchy[ i ].primStop )
	{
		//hierarchy[ i ].ID = 99999999;
		hierarchy[ i ].primStart = 99999999;
		hierarchy[ i ].primStop = 99999999;
		hierarchy[ i ].splitLevel = 99999999;
	}

	return;
}

__global__ void ComputeChildrenStart( hnode_t* h, unsigned int elementCount, unsigned hierarchySize, unsigned int maxDepth )
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if( i >= hierarchySize )
		return;

	h[ i ].visible = false;

	if( h[ i ].splitLevel >= maxDepth )
	{
		h[i].childrenStart = 99999999;
		return;
	}

	if( h[ i+1 ].splitLevel == h[ i ].splitLevel + 1 )
	{	
		h[ i ].childrenStart = h[ i+1 ].ID;
		return;
	}

	return;
}

__global__ void ComputeChildrenStop( hnode_t* h, unsigned int elementCount, unsigned hierarchySize, unsigned int maxDepth )
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if( i >= hierarchySize )
		return;

	if( h[ i ].splitLevel >= maxDepth )
	{
		h[i].childrenStop = 99999999;
		return;
	}

	if( h[ i ]. splitLevel == h[ i + 1 ]. splitLevel )
	{
		h[i].childrenStop = h[i+1].childrenStart-1;
	}
	else
		h[i].childrenStop = h[i].childrenStart+3;
}

__global__ void ComputeBVHRefit( hnode_t* h, bvhnode_t* prim, unsigned int elementCount, unsigned hierarchySize, unsigned int l, unsigned int maxDepth )
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if( i > hierarchySize )
		return;

	if( h[i].splitLevel > maxDepth )
		return;

	if( h[i].splitLevel == l )
	{
		//It's a leaf, lol
		if( l == maxDepth )
		{
			float min_x = 99999999;
			float min_y = 99999999;
			float min_z = 99999999;
			float max_x = -99999999;
			float max_y = -99999999;
			float max_z = -99999999;
			for( int k = h[i].primStart; k <= h[i].primStop; ++k )
			{
				min_x = fmin( prim[k].bbox.min.x, min_x );
				min_y = fmin( prim[k].bbox.min.y, min_y );
				min_z = fmin( prim[k].bbox.min.z, min_z );
				max_x = fmax( prim[k].bbox.max.x, max_x );
				max_y = fmax( prim[k].bbox.max.y, max_y );
				max_z = fmax( prim[k].bbox.max.z, max_z );
			}
			h[i].bbox.min.x = min_x;
			h[i].bbox.min.y = min_y;
			h[i].bbox.min.z = min_z;
			h[i].bbox.max.x = max_x;
			h[i].bbox.max.y = max_y;
			h[i].bbox.max.z = max_z;
			return;
		}
		else //It's a branch
		{
			//well, cool
			float min_x = 99999999;
			float min_y = 99999999;
			float min_z = 99999999;
			float max_x = -99999999;
			float max_y = -99999999;
			float max_z = -99999999;
			for( int k = h[i].childrenStart; k <= min(h[i].childrenStop, hierarchySize ); ++k )
			{
				if( h[k].splitLevel <= maxDepth )
				{
					min_x = fmin( h[k].bbox.min.x, min_x );
					min_y = fmin( h[k].bbox.min.y, min_y );
					min_z = fmin( h[k].bbox.min.z, min_z );
					max_x = fmax( h[k].bbox.max.x, max_x );
					max_y = fmax( h[k].bbox.max.y, max_y );
					max_z = fmax( h[k].bbox.max.z, max_z );
				}
			}
			h[i].bbox.min.x = min_x;
			h[i].bbox.min.y = min_y;
			h[i].bbox.min.z = min_z;
			h[i].bbox.max.x = max_x;
			h[i].bbox.max.y = max_y;
			h[i].bbox.max.z = max_z;
			return;
		}
	}

	return;
}

#endif