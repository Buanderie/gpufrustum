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
	const GCULvoid*		pointer;
};
#define UndefinedArrayInfo( elementWidth ) { 0, elementWidth, GCUL_UNKNOWN, NULL }

enum FrustumType
{
	FRUSTUMTYPE_PYRAMIDAL = 0,
	FRUSTUMTYPE_SPHERICAL,
	FRUSTUMTYPE_UNDEFINED
};

enum BoundingVolumeType
{
	BOUNDINGVOLUMETYPE_BOX = 0,
	BOUNDINGVOLUMETYPE_SPHERE,
	BOUNDINGVOLUMETYPE_UNDEFINED
};

//--------------------
// Functions
//--------------------

FrustumType CurrentFrustumType( void );

BoundingVolumeType CurrentBoundingVolumeType( void );

void AllocArrayDeviceMemory( GCULvoid** array, const ArrayInfo& info );

void CopyArrayToDeviceMemory( GCULvoid* array, const ArrayInfo& info );

void AllocResultDeviceMemory( GCULvoid** memory, const ArrayInfo& frustumInfo, const ArrayInfo& boundingInfo );

void FreeDeviceMemory( GCULvoid* memory );

int SizeInBytes( GCUL_ArrayDataType type );

int ProcessPyramidalFrustumAABBoxCulling( const GCULuint gridSize[ 2 ], const GCULuint blockSize[ 3 ], GCUL_Classification* result );

#endif // __GPUCULLER_INTERNAL_H__