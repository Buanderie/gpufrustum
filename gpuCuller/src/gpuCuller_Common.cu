#include <windows.h>
#include <cutil_inline.h>
#include <gpuCuller.h>
#include <gpuCuller_internal.h>
#include <thrust/device_vector.h>
#include <thrust/version.h>
#include <iostream>

using namespace std;

void __stdcall gculInitialize( int argc, char** argv )
{
	int major = THRUST_MAJOR_VERSION;
    int minor = THRUST_MINOR_VERSION;
    int subminor = THRUST_SUBMINOR_VERSION;
	cout << "Initializing gpuCuller..." << endl;
	std::cout << "Using Thrust v" << major << "." << minor << "." << subminor << std::endl;

	printf("Initializing CUDA...\n");
	// Initializes CUDA device
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );
	printf("CUDA Initialized, using device #%i\n", cutGetMaxGflopsDeviceId());
	
	//Memory information
	cout << "BVH Hierarchy Node Size = " << sizeof(hnode_t) << endl;
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
	cout << "Device Shared Memory Size per block = " << deviceProp.sharedMemPerBlock << endl;
	//
}

void __stdcall gculLoadAABB( unsigned int N, const void* ptr )
{
	//Load AABB data onto Device (AoS code)
	aabb_t * aabb_raw_ptr;
    cudaMalloc((void **) &aabb_raw_ptr, N * sizeof(aabb_t));
	cudaMemcpy(aabb_raw_ptr, ptr, sizeof(aabb_t)*N, cudaMemcpyHostToDevice);
	d_AABB = thrust::device_ptr<aabb_t>(aabb_raw_ptr);
	

	//Prepare memory for BVH Nodes (AoS code)
	bvhnode_t * bvhnode_raw_ptr; 
	cudaMalloc((void **) &bvhnode_raw_ptr, N * sizeof(bvhnode_t));
	d_BVHNODE = thrust::device_ptr<bvhnode_t>(bvhnode_raw_ptr);	

	//Store number of elements
	universeElementCount = N;
}

void __stdcall gculSetBVHDepth( unsigned int depth )
{
	bvhDepth = depth;
}

void __stdcall gculSetUniverseAABB( float min_x, float min_y, float min_z, float max_x, float max_y, float max_z )
{
	universeAABB.min.x = min_x;
	universeAABB.min.y = min_y;
	universeAABB.min.z = min_z;
	universeAABB.max.x = max_x;
	universeAABB.max.y = max_y;
	universeAABB.max.z = max_z;
}


void __stdcall gculBuildLBVH()
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
	
	LBVH_CheckNodeData();
}

unsigned int __stdcall gculGetHierarchySize()
{
	return LBVH_compute_hierachy_mem_size();
}

void __stdcall gculGetHierarchyInformation( void* data )
{
	cudaMemcpy(data, thrust::raw_pointer_cast(d_HIERARCHY), sizeof(hnode_t)*(LBVH_compute_hierachy_mem_size()), cudaMemcpyDeviceToHost);
}