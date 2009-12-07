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
}

void __stdcall gculLoadAABB( unsigned int N, const void* ptr )
{
	//Load AABB data onto Device (AoS code)
	aabb_t * aabb_raw_ptr;

#ifdef REPORT_MEM_OPS
	printf("cudaMalloc : AABB\n");
#endif
    cudaMalloc((void **) &aabb_raw_ptr, N * sizeof(aabb_t));
	cudaMemcpy(aabb_raw_ptr, ptr, sizeof(aabb_t)*N, cudaMemcpyHostToDevice);
	d_AABB = thrust::device_ptr<aabb_t>(aabb_raw_ptr);
	

	//Prepare memory for BVH Nodes (AoS code)
	bvhnode_t * bvhnode_raw_ptr; 

#ifdef REPORT_MEM_OPS
	printf("cudaMalloc : BVHNODE\n");
#endif
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

void __stdcall gculBuildHierarchy()
{
	//choose among different building strategies ?
	//for now, only one...
	LBVH_Build();
	return;
}

unsigned int __stdcall gculGetHierarchySize()
{
	return LBVH_compute_hierachy_mem_size();
}

void __stdcall gculGetHierarchyInformation( void* data )
{
	cudaMemcpy(data, thrust::raw_pointer_cast(d_HIERARCHY), sizeof(hnode_t)*(LBVH_compute_hierachy_mem_size()), cudaMemcpyDeviceToHost);
}

void __stdcall gculFreeHierarchy()
{
	//Release memory used by temporary data
	//Release original AABB data
#ifdef REPORT_MEM_OPS
	printf("cudaFree : BVHNODE\n");
#endif
	thrust::device_free(d_BVHNODE);
	//Release split list
#ifdef REPORT_MEM_OPS
	printf("cudaFree : HIERARCHY\n");
#endif
	thrust::device_free(d_HIERARCHY);
	//
#ifdef REPORT_MEM_OPS
	printf("cudaFREE : OUTPUT\n");
#endif
	thrust::device_free(d_OUTPUT);
	//PROFIT
}

void __stdcall gculFreeAABB()
{
#ifdef REPORT_MEM_OPS
	printf("cudaFree : AABB\n");
#endif
	thrust::device_free(d_AABB);
}

void __stdcall gculFreeFrustumPlanes()
{
#ifdef REPORT_MEM_OPS
	printf("cudaFree : PYRFRUSTUM\n");
#endif
	thrust::device_free(d_PYRFRUSTUM);
}

void __stdcall gculFreeFrustumCorners()
{
#ifdef REPORT_MEM_OPS
	printf("cudaFree : PYRCORNERS\n");
#endif
	thrust::device_free(d_PYRCORNERS);
}

void __stdcall gculLoadFrustumPlanes( unsigned int N, const void* ptr )
{
	//Load Pyramidal Frustum data onto Device (AoS code)
	pyrfrustum_t * pyr_raw_ptr;
#ifdef REPORT_MEM_OPS
	printf("cudaMalloc : PYRFRUSTUM\n");
#endif
    cudaMalloc((void **) &pyr_raw_ptr, N * sizeof(pyrfrustum_t));
	cudaMemcpy(pyr_raw_ptr, ptr, sizeof(pyrfrustum_t)*N, cudaMemcpyHostToDevice);
	d_PYRFRUSTUM = thrust::device_ptr<pyrfrustum_t>(pyr_raw_ptr);
	pyrFrustumCount = N;
	return;
}

void __stdcall gculLoadFrustumCorners( unsigned int N, const void* ptr )
{
	//Load Pyramidal Frustum data onto Device (AoS code)
	pyrcorners_t * pyr_raw_ptr;
#ifdef REPORT_MEM_OPS
	printf("cudaMalloc : PYRCORNERS\n");
#endif
    cudaMalloc((void **) &pyr_raw_ptr, N * sizeof(pyrcorners_t));
	cudaMemcpy(pyr_raw_ptr, ptr, sizeof(pyrcorners_t)*N, cudaMemcpyHostToDevice);
	d_PYRCORNERS = thrust::device_ptr<pyrcorners_t>(pyr_raw_ptr);
	pyrFrustumCount = N;
	return;
}

void __stdcall gculProcessCulling()
{
	FrustumCulling();
	return;
}

void __stdcall gculGetResults(void* data)
{
	cudaMemcpy(data, thrust::raw_pointer_cast(d_OUTPUT), sizeof(unsigned int)*universeElementCount*pyrFrustumCount, cudaMemcpyDeviceToHost);
}

void __stdcall gculSaveHierarchyGraph(char* outputFile)
{
	hnode_t* data = new hnode_t[ LBVH_compute_hierachy_mem_size() ];
	cudaMemcpy(data, thrust::raw_pointer_cast(d_HIERARCHY), sizeof(hnode_t)*(LBVH_compute_hierachy_mem_size()), cudaMemcpyDeviceToHost);
	DotOutput( outputFile, data, universeElementCount, bvhDepth );
	delete data;
}