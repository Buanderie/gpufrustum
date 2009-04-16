#ifndef TEMPLATE_H
#define TEMPLATE_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#  ifdef DLL
#    define DLL_EXP  __declspec(dllexport)
#  else
#    define DLL_EXP  __declspec(dllimport)
#  endif
#else 
#  define DLL_EXP 
#endif

DLL_EXP
void __cdecl run_Bingo( int argc, char** argv); 

#ifdef __cplusplus
}
#endif  // #ifdef _DEBUG (else branch)

#endif