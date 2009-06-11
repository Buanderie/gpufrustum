#include <gpuCuller_internal.h>

bool ArrayStates[ GCUL_END_ARRAY ] = 
{
	false,	// GCUL_BOXES_ARRAY
	false,	// GCUL_BSPHERES_ARRAY
	false,	// GCUL_PYRAMIDALFRUSTUMPLANES_ARRAY
	false,  // GCUL_PYRAMIDALFRUSTUMCORNERS_ARRAY
	false	// GCUL_SPHERICALFRUSTUM_ARRAY
};

ArrayInfo ArrayInfos[ GCUL_END_ARRAY ] = 
{
	UndefinedArrayInfo( 24 ),	// GCU_BOXES_ARRAY						24 = 8 corners * 3 coordinates
	UndefinedArrayInfo(  4 ),	// GCUL_BSPHERES_ARRAY					 4 = 1 center * 3 coordinates + 1 radius
	UndefinedArrayInfo( 24 ),	// GCUL_PYRAMIDALFRUSTUMPLANES_ARRAY	24 = 6 planes * 4
	UndefinedArrayInfo( 24 ),	// GCUL_PYRAMIDALFRUSTUMCORNERS_ARRAY	24 = 8 corners * 3 coordinates
	UndefinedArrayInfo(  4 )	// GCU_SPHERICALFRUSTUM_ARRAY			 4 = 1 center * 3 coordinates + 1 radius
};

DeviceFunctionEnv ClassifyPlanesPointsEnv =
{
	10,
	48,
	364
};

DeviceFunctionEnv ClassifyPyramidalFrustumBoxesEnv =
{
	29,
	48,
	372
};

DeviceFunctionEnv InverseClassifyPyramidalFrustumBoxesEnv =
{
	29,
	48,
	232
};

DeviceFunctionEnv ClassifyPlanesSpheresEnv =
{
	10,
	48,
	256
};

DeviceFunctionEnv ClassifyPyramidalFrustumSpheresEnv =
{
	8,
	32,
	256
};

DeviceFunctionEnv ClassifySphericalFrustumSpheresEnv =
{
	8,
	32,
	256
};