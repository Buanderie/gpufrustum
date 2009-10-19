#ifndef __GPUCULLER_H__
#define __GPUCULLER_H__

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#  ifdef DLL
#    define GPUCULLER_API  __declspec(dllexport)
#  else
#    define GPUCULLER_API  __declspec(dllimport)
#  endif
#else 
#  define GPUCULLER_API 
#endif

GPUCULLER_API
void __stdcall gculInitialize( int argc, char** argv );

GPUCULLER_API
void __stdcall gculLoadAABB( unsigned int size, const void* ptr );

GPUCULLER_API
void __stdcall gculReleaseAABB();

GPUCULLER_API
void __stdcall gculBuildLBVH();

GPUCULLER_API
void __stdcall gculSetBVHDepth( unsigned int depth );

GPUCULLER_API
void __stdcall gculSetUniverseAABB( float min_x, float min_y, float min_z, float max_x, float max_y, float max_z );

#endif

#ifdef __cplusplus
}
#endif  // __cplusplus