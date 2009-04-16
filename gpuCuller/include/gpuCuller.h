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

//--------------------
// Typedefs
//--------------------
typedef unsigned int	GCULenum;
typedef unsigned char	GCULboolean;
typedef unsigned int	GCULbitfield;
typedef signed char		GCULbyte;
typedef short			GCULshort;
typedef int				GCULint;
typedef int				GCULsizei;
typedef unsigned char	GCULubyte;
typedef unsigned short	GCULushort;
typedef unsigned int	GCULuint;
typedef float			GCULfloat;
typedef float			GCULclampf;
typedef double			GCULdouble;
typedef double			GCULclampd;
typedef void			GCULvoid;

//--------------------
// Constants
//--------------------
enum GCUL_ArrayDataType
{
	GCUL_INT = 0,
	GCUL_FLOAT,
	GCUL_DOUBLE,
	GCUL_UNKNOWN
};

enum GCUL_Classification
{
	GCUL_ENCOSING = 0,
	GCUL_INSIDE,
	GCUL_SPANNING,
	GCUL_OUTSIDE	
};

enum GCUL_Array
{
	GCUL_BBOXES_ARRAY = 0,			
	GCUL_BSPHERES_ARRAY,			
	GCUL_PYRAMIDALFRUSTUMPLANES_ARRAY,	
	GCUL_PYRAMIDALFRUSTUMCORNERS_ARRAY,	
	GCUL_SPHERICALFRUSTUM_ARRAY,	
	GCUL_END_ARRAY				
}; 

//--------------------
// Functions
//--------------------

/**
*/
GPUCULLER_API
void __stdcall gculInitialize( int argc, char** argv );

/**
	Specifies the array to enable.
	Symbolic constants GCU_BOXES_ARRAY, GCU_PYRAMIDALFRUSTUM_ARRAY and GCU_SPHERICALFRUSTUM_ARRAY are accepted. 
*/
GPUCULLER_API
void __stdcall gculEnableArray( GCULenum array );

/**
	Specifies the array to disable.
	Symbolic constants GCU_BOXES_ARRAY, GCU_PYRAMIDALFRUSTUM_ARRAY and GCU_SPHERICALFRUSTUM_ARRAY are accepted. 
*/
GPUCULLER_API
void __stdcall gculDisableArray( GCULenum array );

/**
	Specifies planes for pyramidal frustums.
	A frustum is defined by 6 planes {near, far, top, left, right, bottom}.
	A plan is defined by 4 numbers {a, b, c, d}.
	@param size		: Specifies the number of pyramidal frustums.
	@param type		: Specifies the data type of each coordinate in the
					  array.  Symbolic constants GCU_INT, GCU_FLOAT, 
					  and GCU_DOUBLE are accepted. 
	@param pointer	: Specifies a pointer to the first coordinate of the
					  first plane in the array.
*/
GPUCULLER_API
void __stdcall gculPyramidalFrustumPlanesPointer( GCULuint size, GCULenum type, const GCULvoid* pointer );

/**
	Specifies corners for pyramidal frustums.
	A frustum is defined by 8 points {neartopleft, neartopright, nearbottomleft, nearbottomright, fartopleft, fartopright, farbottomleft, farbottomright} 
	A corner is defined by 3 numbers {x, y, z}.
	@param size		: Specifies the number of pyramidal frustums.
	@param type		: Specifies the data type of each coordinate in the
					  array.  Symbolic constants GCU_INT, GCU_FLOAT, 
					  and GCU_DOUBLE are accepted. 
	@param pointer	: Specifies a pointer to the first coordinate of the
					  first corner in the array.
*/
GPUCULLER_API
void __stdcall gculPyramidalFrustumCornersPointer( GCULuint size, GCULenum type, const GCULvoid* pointer );

/**
	Specifies planes and corners for pyramidal frustums.
	@see gculPyramidalFrustumPlanesPointer.
	@see gculPyramidalFrustumCornersPointer.
*/
GPUCULLER_API
void __stdcall gculPyramidalFrustumPointers( GCULuint size, GCULenum type, const GCULvoid* planes, const GCULvoid* corners );

/**
	Specifies data for boxes.
	A boxes is defined by 8 points.
	A point is defined by 3 coordinates {x, y, z}.
	@param size		: Specifies the number of boxes.
	@param type		: Specifies the data type of each coordinate in the
					  array.  Symbolic constants GCU_INT, GCU_FLOAT, 
					  and GCU_DOUBLE are accepted. 
	@param pointer	: Specifies a pointer to the first coordinate of the
					  first box in the array.
*/
GPUCULLER_API
void __stdcall gculBoxesPointer( GCULuint size, GCULenum type, const GCULvoid* pointer );

/**
	Process the frustum culling operation between the frustums array and the boxes array.
	@param gridSize		: Specifies the dimension and size of the grid, such that 
						  gridSize[ 0 ] * gridSize[ 1 ] equals the number of blocks being launched.
						  If gridSize is {0, 0}, the system determines itself the grid size according to the data.
	@param blockSize	: Specifies the dimension and size of each block, such that 
						  blockSize[ 0 ] * blockSize[ 1 ] * blockSize[ 2 ] equals the number of 
						  threads per block.
						  If blockSize is {0, 0}, the system determines itself the grid size according to the data.
	@param result		: The classification of boxes per frustum. 
						  Classification of the frustum i with the box j is available at the index j + i * number of boxes.
						  Symbolic constants GCU_ENCOSING, GCU_INSIDE, GCU_SPANNING and GCU_OUTSIDE are accepted. 
						  The array size must be greater or equal to number of frustums by the number of boxes. 
	@return 0 if successful or the error code.
*/
GPUCULLER_API
GCULint __stdcall gculProcessFrustumCulling( const GCULuint gridSize[ 2 ], const GCULuint blockSize[ 3 ], GCUL_Classification* result );

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif	// __GPUCULLER_H__