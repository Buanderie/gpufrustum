#ifndef __GPUCULLER_INTERNAL_H__
#define __GPUCULLER_INTERNAL_H__

#include <gpuCuller.h>
#include <iostream>
#include <windows.h>

#ifndef NDEBUG
#define assert( condition, message ) if( !condition ) { std::cerr<< message << __FILE__ << __LINE__ << std::endl; DebugBreak(); }
#else
	#define assert( condition, message )
#endif

struct ArrayInfo
{
	GCULuint			size;
	GCULint				elementWidth;
	GCUL_ArrayDataType	type;
	GCULsizei			stride;
	const GCULvoid*		pointer;
};
#define UndefinedArrayInfo( elementWidth ) { 0, elementWidth, GCUL_UNKNOWN, NULL, 0 }

//--------------------
// Functions
//--------------------

GCUL_Array CurrentFrustumArray( void );

GCUL_Array CurrentBoundingObjectArray( void );

void AllocArrayDeviceMemory( GCULvoid** array, const ArrayInfo& info );

void CopyArrayToDeviceMemory( GCULvoid* array, const ArrayInfo& info );

void AllocResultDeviceMemory( GCULvoid** memory, const ArrayInfo& frustumInfo, const ArrayInfo& boundingInfo );

void FreeDeviceMemory( GCULvoid* memory );

int SizeInBytes( GCUL_ArrayDataType type );

#endif // __GPUCULLER_INTERNAL_H__