#ifndef __GPUCULLER_INTERNAL_H__
#define __GPUCULLER_INTERNAL_H__

#include <thrust/device_ptr.h>

//-------- Macros & Defines -------
#define REPORT_FAILURE(__DEATH_MESSAGE__) printf("%s.%d : FAILURE : %s\n", __FILE__, __LINE__, __DEATH_MESSAGE__);exit(-1);
#define REPORT_MEM_OPS
//---------------------------------

//-------- Data structures --------
//3D Vector
typedef struct vec3{
	float x;
	float y;
	float z;
} vec3_t;

//AABB (AoS)
typedef struct aabb{
	vec3_t min;
	vec3_t max;
} aabb_t;

//BVH Node (AoS) // In that case, SoA would mean launching 2 consecutive thrust::sort...
typedef struct bvhnode{
	unsigned int primIndex;
	unsigned int mortonCode;
	unsigned int split;
	vec3_t centroid;
	aabb_t bbox;
} bvhnode_t;

//Split structure
typedef struct lbvhsplit
{
	unsigned int level;
	unsigned int primIndex;
} lbvhsplit_t;

//Hierarchy node structure
typedef struct hnode
{
	unsigned int splitLevel;
	unsigned int primStart;
	unsigned int primStop;
	unsigned int ID;
	unsigned int childrenStart;
	unsigned int childrenStop;
	bool visible;
	aabb_t bbox;
} hnode_t;
//

//Frustum plane
typedef struct plane
{
	float a;
	float b;
	float c;
	float d;
} plane_t;
//

//Pyramidal Frustum
typedef struct pyrfrustum
{
	plane_t planes[6];
} pyrfrustum_t;
//

//Pyramidal Frustum Corners
typedef struct pyrcorners
{
	vec3_t points[8];
} pyrcorners_t;
//

//---------------------------------

//-------- Data References --------
extern thrust::device_ptr<aabb_t> d_AABB;
extern thrust::device_ptr<bvhnode_t> d_BVHNODE;
extern thrust::device_ptr<lbvhsplit_t> d_SPLITSLIST;
extern thrust::device_ptr<hnode_t> d_HIERARCHY;
extern thrust::device_ptr<pyrfrustum_t> d_PYRFRUSTUM;
extern thrust::device_ptr<pyrcorners_t> d_PYRCORNERS;
extern thrust::device_ptr<unsigned int> d_OUTPUT;
extern unsigned int universeElementCount;
extern unsigned int bvhDepth;
extern unsigned int pyrFrustumCount;
extern aabb_t universeAABB;
texture<hnode_t, 1, cudaReadModeElementType> texHIERARCHY;
//---------------------------------

//-------- LBVH Code --------------
void LBVH_assign_morton_code();
void LBVH_sort_by_code();
void LBVH_CheckNodeData();
void LBVH_compute_split_levels();
void LBVH_sort_split_list();
void LBVH_build_hierarchy1();
void LBVH_build_hierarchy2();
unsigned int LBVH_compute_hierachy_mem_size();
void LBVH_BVH_Refit();
void LBVH_Cleanup();
void LBVH_Build();
//---------------------------------

//--------- Frustum Culling -------
void FrustumCulling();
//---------------------------------

//--------- Utils -----------------
void DotOutput( char* fileName, hnode_t* h, int sz, int maxDepth );
//---------------------------------

#endif