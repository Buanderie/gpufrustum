#include <windows.h>
#include <cutil_inline.h>
#include <gpuCuller.h>
#include <gpuCuller_internal.h>
#include <thrust/device_vector.h>
#include <iostream>

using namespace std;

void __stdcall gculInitialize( int argc, char** argv )
{
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
}

void __stdcall gculBuildLBVH()
{
	//First step: Assign Morton Codes to BVH Nodes

}