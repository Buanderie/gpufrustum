#include <iostream>
//
#include <cutil.h>
#include <cutil_inline.h>
#include <gpuCuller_internal.h>
#include <gpuCuller_LBVH_kernel.cu>
//Thrust
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>

//

using namespace std;

//Custom compare function for sorting
template<typename T>
struct dev_cmp_custom_key : public thrust::binary_function<T,T,bool>
{
  /*! Function call operator. The return value is <tt>lhs < rhs</tt>.
   */
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const
  {
    unsigned int a=lhs.mortonCode;
    unsigned int b=rhs.mortonCode;
    return (a < b);
  }
}; // end compare
//

// extract a mortonCode from a bvhnode_t
struct bvhnode_to_mortonCode
{
    __host__ __device__
    unsigned int operator()(const bvhnode_t& node)
    {
        return node.mortonCode;
    }
};
//

void LBVH_assign_morton_code()
{
	//Retrieve data pointers !
	aabb_t*		aabb_raw	= thrust::raw_pointer_cast(d_AABB);
	bvhnode_t*	bvhnode_raw	= thrust::raw_pointer_cast(d_BVHNODE);

	//Compute grid/block sizes...
	// setup execution parameters
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
	const unsigned int maxGridDim = deviceProp.maxGridSize[0];
	const unsigned int maxBlockDim = deviceProp.maxThreadsDim[0];
	const unsigned int gridDim = max( 1, universeElementCount / maxBlockDim );

	dim3  grid( gridDim, 1 );
    dim3  threads( maxBlockDim, 1 );

	//Execute kernel
	AssignMortonCode<<< grid, threads >>>(	aabb_raw, bvhnode_raw, universeElementCount, bvhDepth,
											universeAABB.min_x,
											universeAABB.min_y,
											universeAABB.min_z,
											universeAABB.max_x,
											universeAABB.max_y,
											universeAABB.max_z);
	cudaThreadSynchronize();
}

void LBVH_CheckNodeData()
{
	bvhnode_t* pol = new bvhnode_t[ universeElementCount ];
	cudaMemcpy(pol, thrust::raw_pointer_cast(d_BVHNODE), sizeof(bvhnode_t)*universeElementCount, cudaMemcpyDeviceToHost);
	for( int i = 0; i < universeElementCount; ++i )
	{
		cout << pol[i].mortonCode << endl;
	}
}

void LBVH_sort_by_code()
{
	// strip out the morton codes from each bvhnode
    thrust::device_vector<unsigned int> codes(universeElementCount);
    thrust::transform(d_BVHNODE, d_BVHNODE + universeElementCount, codes.begin(), bvhnode_to_mortonCode());
	// sort by the mortonCodes
    thrust::sort_by_key(codes.begin(), codes.end(), d_BVHNODE);

	//bvhnode_t*	rawb	= thrust::raw_pointer_cast(d_BVHNODE);
	//thrust::sort(rawb, rawb+(universeElementCount-1)*sizeof(bvhnode_t), dev_cmp_custom_key<bvhnode_t>());
	cudaThreadSynchronize();
}