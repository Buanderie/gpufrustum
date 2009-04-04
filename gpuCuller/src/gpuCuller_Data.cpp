#include <gpuCuller_internal.h>

bool ArrayStates[ GCUL_END_ARRAY ] = 
{
	false,	// GCUL_BOXES_ARRAY
	false,	// GCUL_PYRAMIDALFRUSTUM_ARRAY
	false	// GCUL_SPHERICALFRUSTUM_ARRAY
};

ArrayInfo ArrayInfos[ GCUL_END_ARRAY ] = 
{
	UndefinedArrayInfo( 24 ),	// GCU_BOXES_ARRAY
	UndefinedArrayInfo( 24 ),	// GCU_PYRAMIDALFRUSTUM_ARRAY
	UndefinedArrayInfo( 12 )	// GCU_SPHERICALFRUSTUM_ARRAY
};