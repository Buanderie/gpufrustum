#ifndef __GPUCULLER_FRUSTUMCULLING_KERNEL_H__
#define __GPUCULLER_FRUSTUMCULLING_KERNEL_H__

#include <cutil.h>
#include <cutil_inline.h>

#include <gpuCuller_internal.h>

texture<unsigned int, 1, cudaReadModeElementType> tex_trav_hnodeSplitLevel;
texture<unsigned int, 1, cudaReadModeElementType> tex_trav_hnodePrimStart;
texture<unsigned int, 1, cudaReadModeElementType> tex_trav_hnodePrimStop;
texture<unsigned int, 1, cudaReadModeElementType> tex_trav_hnodeID;
texture<unsigned int, 1, cudaReadModeElementType> tex_trav_hnodeChildrenStart;
texture<unsigned int, 1, cudaReadModeElementType> tex_trav_hnodeChildrenStop;
texture<float4, 1, cudaReadModeElementType> tex_trav_hnodeAABBMin;
texture<float4, 1, cudaReadModeElementType> tex_trav_hnodeAABBMax;
texture<unsigned int, 1, cudaReadModeElementType> tex_trav_primIndex;
texture<float4, 1, cudaReadModeElementType> tex_trav_primAABBMin;
texture<float4, 1, cudaReadModeElementType> tex_trav_primAABBMax;

__device__ float planeDistance( vec3& v, plane_t& p )
{
	return (v.x*p.a + v.y*p.b + v.z*p.c + p.d );
}

__device__ bool AABBcontainsPoint( aabb_t& a, vec3_t& p )
{
	return (p.x >= a.min.x && p.x <= a.max.x) &&
           (p.y >= a.min.y && p.y <= a.max.y) &&
		   (p.z >= a.min.z && p.z <= a.max.z);
}

__device__ bool AABBenclosing( aabb_t& a, pyrcorners_t& c )
{
	for( int i = 0; i < 8; i++ )
		if( AABBcontainsPoint( a, c.points[i] ) )
			return true;
	return false;
}

__device__ bool Intersect( pyrfrustum_t& f, aabb_t& a )
{

	//if( AABBenclosing( a, c ) )
	//	return true;

	vec3 box[8];
	box[0].x = a.min.x; box[0].y = a.min.y; box[0].z = a.min.z;
	box[1].x = a.max.x; box[1].y = a.min.y; box[1].z = a.min.z;
	box[2].x = a.min.x; box[2].y = a.max.y; box[2].z = a.min.z;
	box[3].x = a.max.x; box[3].y = a.max.y; box[3].z = a.min.z;
	box[4].x = a.min.x; box[4].y = a.min.y; box[4].z = a.max.z;
	box[5].x = a.max.x; box[5].y = a.min.y; box[5].z = a.max.z;
	box[6].x = a.min.x; box[6].y = a.max.y; box[6].z = a.max.z;
	box[7].x = a.max.x; box[7].y = a.max.y; box[7].z = a.max.z;

	int iTotalIn = 0;

	// test all 8 corners against the 6 sides 
	// if all points are behind 1 specific plane, we are out
	// if we are in with all points, then we are fully in
	for(int p = 0; p < 6; ++p) {
	
		int iInCount = 8;
		int iPtIn = 1;

		for(int i = 0; i < 8; ++i) {

			// test this point against the planes
			if(planeDistance( box[i], f.planes[p] ) >= 0 ) {
				iPtIn = 0;
				--iInCount;
			}
		}

		// were all the points outside of plane p?
		if(iInCount == 0)
			return false;

		// check if they were all on the right side of the plane
		iTotalIn += iPtIn;
	}

	// so if iTotalIn is 6, then all are inside the view
	if(iTotalIn == 6)
		return true;

	// we must be partly in then otherwise
	return true;
}

__device__ hnode_t fetchFromTex( unsigned int i )
{
	hnode_t ret;
	ret.splitLevel = tex1Dfetch(tex_trav_hnodeSplitLevel, i);
	ret.primStart = tex1Dfetch(tex_trav_hnodePrimStart, i );
	ret.primStop = tex1Dfetch(tex_trav_hnodePrimStop, i );
	ret.ID = tex1Dfetch(tex_trav_hnodeID, i );
	ret.childrenStart = tex1Dfetch(tex_trav_hnodeChildrenStart, i );
	ret.childrenStop = tex1Dfetch(tex_trav_hnodeChildrenStop, i );
	float4 hmin = tex1Dfetch(tex_trav_hnodeAABBMin, i );
	float4 hmax = tex1Dfetch(tex_trav_hnodeAABBMax, i );
	ret.bbox.min.x = hmin.x;
	ret.bbox.min.y = hmin.y;
	ret.bbox.min.z = hmin.z;
	ret.bbox.max.x = hmax.x;
	ret.bbox.max.y = hmax.y;
	ret.bbox.max.z = hmax.z;
	return ret;
}

__device__ bvhnode_t fetchPrimFromTex( unsigned int i )
{
	bvhnode_t b;
	float4 min = tex1Dfetch(tex_trav_primAABBMin, i );
	float4 max = tex1Dfetch(tex_trav_primAABBMax, i );
	b.bbox.min.x = min.x;
	b.bbox.min.y = min.y;
	b.bbox.min.z = min.z;
	b.bbox.max.x = max.x;
	b.bbox.max.y = max.y;
	b.bbox.max.z = max.z;
	b.primIndex = tex1Dfetch(tex_trav_primIndex, i );
	return b;
}

__global__ void ProcessFrustumCulling __traceable__ ( pyrfrustum_t* f, hnode_t* h, bvhnode_t* prims, unsigned int* out, unsigned int frustumCount, unsigned int primCount, unsigned int hierarchySize, unsigned int maxDepth )
{
	//Frustum array index
	int idt = blockDim.x * blockIdx.x + threadIdx.x;
	//

	if( idt > frustumCount )
		return;

	//stack ^^
	unsigned int stack[200];
	int stack_top = -1;
	//

	//store frustum info
	pyrfrustum_t pyr = f[idt];

	//Add the first level
	for( int i = 0; i < 4; ++i )
	{
		hnode_t n = fetchFromTex(i);
		stack_top++;
		stack[ stack_top ] = n.ID;
	}

	//Parse the tree
	while( true )
	{
		if( stack_top < 0 || stack_top == 200 )
			break;

		//extract the top node
		hnode_t n = fetchFromTex( stack[ stack_top ] );
		stack_top--;
		//

		//check if visible
		if( Intersect( pyr, n.bbox ) )
		{
			//if it's a leaf, check the associated primitives
			if( n.splitLevel == maxDepth )
			{
				for( int i = n.primStart; i <= n.primStop; ++i )
				{
					bvhnode_t b = fetchPrimFromTex(i);
					if( Intersect( pyr, b.bbox ) )
					{
						out[ idt*primCount + b.primIndex ] = 7777777;
					}
				}
				continue;
			}
			
			//if it's a branch, add its children to the stack
			if( n.splitLevel < maxDepth )
			{
				for( int i = n.childrenStart; i <= n.childrenStop; ++i )
				{
					stack_top++;
					stack[ stack_top ] = i;
				}
				continue;
			}
		}
	}

}

__global__ void DummyKernel(hnode_t * out, unsigned int len )
{
	for( int i = 0; i < len; ++i )
		out[i] = fetchFromTex(i);
}

#endif