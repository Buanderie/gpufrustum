#ifndef __GPUCULLER_INTERNAL_H__
#define __GPUCULLER_INTERNAL_H__

#include <thrust/device_ptr.h>

//-------- Data structures --------
//AABB
typedef struct aabb{
	float min_x, min_y, min_z;
	float max_x, max_y, max_z;
} aabb_t;

//BVH Node
typedef struct bvhnode{
	unsigned int primIndex;
	unsigned int mortonCode;
	float centroidX, centroidY, centroidZ;
} bvhnode_t;
//
//---------------------------------

//-------- Data References --------
extern thrust::device_ptr<aabb_t> d_AABB;
extern thrust::device_ptr<bvhnode_t> d_BVHNODE;
extern unsigned int universeElementCount;
extern unsigned int bvhDepth;
extern aabb_t universeAABB;
//---------------------------------

//-------- LBVH Code --------------
void LBVH_assign_morton_code();
//---------------------------------

#endif