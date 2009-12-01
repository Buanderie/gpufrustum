#include <gpuCuller_internal.h>

#include <thrust/device_ptr.h>

//-------- Data References --------
thrust::device_ptr<aabb_t> d_AABB;
thrust::device_ptr<bvhnode_t> d_BVHNODE;
thrust::device_ptr<lbvhsplit_t> d_SPLITSLIST;
thrust::device_ptr<hnode_t> d_HIERARCHY;
thrust::device_ptr<pyrfrustum_t> d_PYRFRUSTUM;
thrust::device_ptr<pyrcorners_t> d_PYRCORNERS;
thrust::device_ptr<unsigned int> d_OUTPUT;
unsigned int universeElementCount;
unsigned int pyrFrustumCount;
unsigned int bvhDepth;
aabb_t universeAABB;
//---------------------------------