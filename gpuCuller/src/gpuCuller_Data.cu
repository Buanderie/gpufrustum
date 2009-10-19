#include <gpuCuller_internal.h>

#include <thrust/device_ptr.h>

//-------- Data References --------
thrust::device_ptr<aabb_t> d_AABB;
thrust::device_ptr<bvhnode_t> d_BVHNODE;
//---------------------------------