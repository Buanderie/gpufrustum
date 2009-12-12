#include <gpuCuller_internal.h>

#include <thrust/device_ptr.h>

//-------- Data References --------
thrust::device_ptr<aabb_t> d_AABB;
thrust::device_ptr<bvhnode_t> d_BVHNODE;
thrust::device_ptr<lbvhsplit_t> d_SPLITSLIST;
thrust::device_ptr<hnode_t> d_HIERARCHY;
thrust::device_ptr<pyrfrustum_t> d_PYRFRUSTUM;
thrust::device_ptr<pyrcorners_t> d_PYRCORNERS;
thrust::device_ptr<char> d_OUTPUT;
unsigned int universeElementCount;
unsigned int pyrFrustumCount;
unsigned int bvhDepth;
aabb_t universeAABB;
//---------------------------------

//Traversal Data References
thrust::device_ptr<unsigned int> trav_hnodeSplitLevel;
thrust::device_ptr<unsigned int> trav_hnodePrimStart;
thrust::device_ptr<unsigned int> trav_hnodePrimStop;
thrust::device_ptr<unsigned int> trav_hnodeID;
thrust::device_ptr<unsigned int> trav_hnodeChildrenStart;
thrust::device_ptr<unsigned int> trav_hnodeChildrenStop;
thrust::device_ptr<float4> trav_hnodeAABBMin;
thrust::device_ptr<float4> trav_hnodeAABBMax;
thrust::device_ptr<unsigned int> trav_primIndex;
thrust::device_ptr<float4> trav_primAABBMin;
thrust::device_ptr<float4> trav_primAABBMax;
//