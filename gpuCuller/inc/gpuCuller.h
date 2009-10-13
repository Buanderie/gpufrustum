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

#endif

#ifdef __cplusplus
}
#endif  // __cplusplus