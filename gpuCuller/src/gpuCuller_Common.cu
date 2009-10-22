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
}

void __stdcall gculLoadAABB( unsigned int N, const void* ptr )
{
	//Load AABB data onto Device
	aabb_t * aabb_raw_ptr;
    cudaMalloc((void **) &aabb_raw_ptr, N * sizeof(aabb_t));
	cudaMemcpy(aabb_raw_ptr, ptr, sizeof(aabb_t)*N, cudaMemcpyHostToDevice);
	d_AABB = thrust::device_ptr<aabb_t>(aabb_raw_ptr);

	//Prepare memory for BVH Nodes
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
	universeAABB.min_x = min_x;
	universeAABB.min_y = min_y;
	universeAABB.min_z = min_z;
	universeAABB.max_x = max_x;
	universeAABB.max_y = max_y;
	universeAABB.max_z = max_z;
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

	
	//LBVH_CheckNodeData();
}