#include <gpuCuller_internal.h>
#include <gpuCuller_LBVH_kernel.cu>

void LBVH_assign_morton_code()
{
	//Retrieve data pointers !
	aabb_t*		aabb_raw	= thrust::raw_pointer_cast(d_AABB);
	bvhnode_t*	bvhnode_raw	= thrust::raw_pointer_cast(d_BVHNODE);

	//Compute grid/block sizes...
	
	//Execute kernel
	AssignMortonCode<<< 32, 32 >>>( aabb_raw, bvhnode_raw, universeElementCount, bvhDepth,
									universeAABB.min_x,
									universeAABB.min_y,
									universeAABB.min_z,
									universeAABB.max_x,
									universeAABB.max_y,
									universeAABB.max_z);
}