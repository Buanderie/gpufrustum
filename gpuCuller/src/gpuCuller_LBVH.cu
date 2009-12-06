#include <math.h>
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

// extract a level from a lbvhsplit_t
struct lbvhsplit_to_level
{
    __host__ __device__
    unsigned int operator()(const lbvhsplit_t& split)
    {
        return split.level;
    }
};
//

// extract a start index from a hnode_t
struct hnode_to_startind
{
    __host__ __device__
    unsigned int operator()(const hnode_t& node)
    {
        return node.primStart;
    }
};

// extract an ID from a hnode_t
struct hnode_to_ID
{
    __host__ __device__
    unsigned int operator()(const hnode_t& node)
    {
        return node.ID;
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
	const unsigned int gridDim = max( 1, (int)ceil((float)universeElementCount / (float)maxBlockDim) );

	dim3  grid( gridDim, 1 );
    dim3  threads( maxBlockDim, 1 );

	//Execute kernel
	AssignMortonCode<<< grid, threads >>>(	aabb_raw, bvhnode_raw, universeElementCount, bvhDepth,
											universeAABB.min.x,
											universeAABB.min.y,
											universeAABB.min.z,
											universeAABB.max.x,
											universeAABB.max.y,
											universeAABB.max.z);
	cudaThreadSynchronize();
}

void LBVH_CheckNodeData()
{
	//
	lbvhsplit_t* pol1 = new lbvhsplit_t[ universeElementCount*bvhDepth ];
	cudaMemcpy(pol1, thrust::raw_pointer_cast(d_SPLITSLIST), sizeof(lbvhsplit_t)*universeElementCount*bvhDepth, cudaMemcpyDeviceToHost);
	for( int i = 0; i < universeElementCount*bvhDepth; ++i )
	{
		if( pol1[i].level < 100000 )
		cout << pol1[i].level << "[ " << pol1[i].primIndex << " ]" << endl;
	}
	//

	//
	bvhnode_t* pol = new bvhnode_t[ universeElementCount ];
	cudaMemcpy(pol, thrust::raw_pointer_cast(d_BVHNODE), sizeof(bvhnode_t)*universeElementCount, cudaMemcpyDeviceToHost);
	for( int i = 0; i < universeElementCount; ++i )
	{
		cout	<< i << "|" << pol[i].mortonCode << "- (" << pol[i].bbox.min.x << "," << pol[i].bbox.min.y << "," << pol[i].bbox.min.z << ") - ("
				<< pol[i].bbox.max.x << "," << pol[i].bbox.max.y << "," << pol[i].bbox.max.z << ")" << endl;

	}
	//

	//
	hnode_t* bak = new hnode_t[ LBVH_compute_hierachy_mem_size() ];
	cudaMemcpy(bak, thrust::raw_pointer_cast(d_HIERARCHY), sizeof(hnode_t)*(LBVH_compute_hierachy_mem_size()), cudaMemcpyDeviceToHost);
	for( int i = 0; i < LBVH_compute_hierachy_mem_size(); ++i )
	{
		cout	<< bak[i].ID << " - lvl=" << bak[i].splitLevel << " [ " << bak[i].primStart << " ; " << bak[i].primStop << " ] ~ " 
				<< "{ " << bak[i].childrenStart << " ; " << bak[i].childrenStop << " }" 
				<< " - (" << bak[i].bbox.min.x << ";" << bak[i].bbox.min.z << ") "
				<< " - (" << bak[i].bbox.max.x << ";" << bak[i].bbox.max.z << ") "<< endl;
	}
	//
}

void LBVH_sort_by_code()
{
	// strip out the morton codes from each bvhnode
    thrust::device_vector<unsigned int> codes(universeElementCount);
    thrust::transform(d_BVHNODE, d_BVHNODE + universeElementCount, codes.begin(), bvhnode_to_mortonCode());
	
	// sort by the mortonCodes
    thrust::sort_by_key(codes.begin(), codes.end(), d_BVHNODE);

	cudaThreadSynchronize();
}

void LBVH_compute_split_levels()
{
	//Prepare memory for splits list (size = depth * elementCount)
	lbvhsplit_t * splitslist_raw_ptr;
	const unsigned splitListSize = bvhDepth * universeElementCount;
    cudaMalloc((void **) &splitslist_raw_ptr, splitListSize * sizeof(lbvhsplit_t));
	d_SPLITSLIST = thrust::device_ptr<lbvhsplit_t>(splitslist_raw_ptr);

	//Retrieve data pointers !
	bvhnode_t*	bvhnode_raw	= thrust::raw_pointer_cast(d_BVHNODE);

	//Compute grid/block sizes...
	// setup execution parameters
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
	const unsigned int maxGridDim = deviceProp.maxGridSize[0];
	const unsigned int maxBlockDim = deviceProp.maxThreadsDim[0];
	const unsigned int gridDim = max( 1, (int)ceil((float)universeElementCount / (float)maxBlockDim) );

	dim3  grid( gridDim, 1 );
    dim3  threads( maxBlockDim, 1 );

	//Execute kernel
	ComputeSplitLevel<<< grid, threads >>>(	bvhnode_raw, splitslist_raw_ptr, universeElementCount, bvhDepth );
	cudaThreadSynchronize();
}

void LBVH_sort_split_list()
{
	// strip out the split level from each split
    thrust::device_vector<unsigned int> levels(universeElementCount*bvhDepth);
    thrust::transform(d_SPLITSLIST, d_SPLITSLIST + (universeElementCount*bvhDepth), levels.begin(), lbvhsplit_to_level());
	
	// sort by the level
    thrust::sort_by_key(levels.begin(), levels.end(), d_SPLITSLIST);

	cudaThreadSynchronize();
}

unsigned int LBVH_compute_hierachy_mem_size()
{
	float fu1 = (1.0f - pow(4.0f, (float)(bvhDepth+1)));
	float fu2 = (-3.0f);
	unsigned int ret = (unsigned int)(fu1/fu2) - 1;
	
	return ret+bvhDepth;
}

void LBVH_build_hierarchy1()
{
	//Okay... what do we need ?
	lbvhsplit_t * splitslist_raw_ptr = thrust::raw_pointer_cast(d_SPLITSLIST);

	//Prepare memory for hierarchy
	//Memory consumption growth follows a geometric serie
	//
	unsigned int sz = LBVH_compute_hierachy_mem_size();
	hnode_t* h_raw_ptr;
	cudaMalloc((void **) &h_raw_ptr, sz * sizeof(hnode_t));
	d_HIERARCHY = thrust::device_ptr<hnode_t>(h_raw_ptr);

	//Now, we have to compute the intervals over the primitive list for every hierarchy node...
	//Compute grid/block sizes...
	// setup execution parameters
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
	const unsigned int maxGridDim = deviceProp.maxGridSize[0];
	const unsigned int maxBlockDim = deviceProp.maxThreadsDim[0];
	const unsigned int gridDim = max( 1, (int)ceil((float)sz / (float)maxBlockDim) );

	dim3  grid( gridDim, 1 );
    dim3  threads( maxBlockDim, 1 );

	cout << "Hierarchy Size = " << sz << endl;

	ComputeHNodeIntervals<<< grid, threads >>>(	splitslist_raw_ptr, h_raw_ptr, universeElementCount, sz, bvhDepth );
	cudaThreadSynchronize();

	return;
}

void LBVH_build_hierarchy2()
{
	unsigned int sz = LBVH_compute_hierachy_mem_size();

	// strip out the START primIndex from each hnode
    thrust::device_vector<unsigned int> starts(sz);
    thrust::transform(d_HIERARCHY, d_HIERARCHY + (sz), starts.begin(), hnode_to_startind());
	
	// sort by the start index
    thrust::sort_by_key(starts.begin(), starts.end(), d_HIERARCHY);

	// Launch the kernel to get the child start pointers... ????
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
	const unsigned int maxGridDim = deviceProp.maxGridSize[0];
	const unsigned int maxBlockDim = deviceProp.maxThreadsDim[0];
	const unsigned int gridDim = max( 1, (int)ceil((float)sz / (float)maxBlockDim) );
	dim3  grid( gridDim, 1 );
    dim3  threads( maxBlockDim, 1 );

	cout << "POL = " << gridDim << " BAK = " << maxBlockDim << endl;
	ComputeChildrenStart<<< grid, threads >>>( thrust::raw_pointer_cast(d_HIERARCHY), universeElementCount, sz, bvhDepth );
	cudaThreadSynchronize();
	//

	
	// strip out the ID from each hnode
    thrust::device_vector<unsigned int> idlol(sz);
    thrust::transform(d_HIERARCHY, d_HIERARCHY + (sz), idlol.begin(), hnode_to_ID());
	//sort the shit by ID...
	thrust::sort_by_key(idlol.begin(), idlol.end(), d_HIERARCHY);
	//
	cudaThreadSynchronize();
	
	ComputeChildrenStop<<< grid, threads >>>( thrust::raw_pointer_cast(d_HIERARCHY), universeElementCount, sz, bvhDepth );
	cudaThreadSynchronize();
	
	return;
}

void LBVH_BVH_Refit()
{
	
	//Reduction... BVH REFIT

	//Prepare the shit
	unsigned int sz = LBVH_compute_hierachy_mem_size();
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
	const unsigned int maxGridDim = deviceProp.maxGridSize[0];
	const unsigned int maxBlockDim = deviceProp.maxThreadsDim[0];
	const unsigned int gridDim = max( 1, (int)ceil((float)sz / (float)maxBlockDim) );
	dim3  grid( gridDim, 1 );
    dim3  threads( maxBlockDim, 1 );
	//

	//For each level of hierarchy
	for( int i = bvhDepth; i > 0; i-- )
	{
		ComputeBVHRefit<<< grid, threads >>>( thrust::raw_pointer_cast(d_HIERARCHY), thrust::raw_pointer_cast(d_BVHNODE), universeElementCount, sz, i, bvhDepth );
		cudaThreadSynchronize();
	}
}

void LBVH_Cleanup()
{
	//Release memory used by temporary data
	//Release original AABB data
	thrust::device_free(d_AABB);
	//Release split list
	thrust::device_free(d_SPLITSLIST);
	//
	//PROFIT

	// As hierarchy is built, prepare memory for culling result
	unsigned int * dpolbak = 0;
	cudaMalloc((void **) &dpolbak, pyrFrustumCount*universeElementCount*sizeof(unsigned int));
	d_OUTPUT = thrust::device_ptr<unsigned int>(dpolbak);
	thrust::fill(d_OUTPUT, d_OUTPUT + pyrFrustumCount*universeElementCount, 0);
	cudaThreadSynchronize();
	//
}

void LBVH_Build()
{
	//First step: Assign Morton Codes to BVH Nodes
	LBVH_assign_morton_code();
	//

	//Second step: Sort the BVH Nodes according to their morton codes...
	//Use the thrust::sort function...
	//
	LBVH_sort_by_code();

	//Prepare and fill the splits list
	LBVH_compute_split_levels();


	//Sort the splits list by level of split
	LBVH_sort_split_list();

	//First phase of hierarchy construction : Compute hierarchy nodes Primitive Intervals
	LBVH_build_hierarchy1();

	//Second phase of hierarchy construction : Building children pointers
	LBVH_build_hierarchy2();

	//Last phase : BVH Refit
	LBVH_BVH_Refit();

	//LBVH_CheckNodeData();

	LBVH_Cleanup();
}