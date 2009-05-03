#ifndef __GLMATRIX4F_H__
#define __GLMATRIX4F_H__

#include <windows.h>		// Header File For Windows
#include <gl\gl.h>			// Header File For The OpenGL32 Library
#include <gl\glu.h>			// Header File For The GLu32 Library

#include "glVector4f.h"

class glMatrix4f
{
public:
	float elem[4][4];
	glVector4f& MatVecProduct(glVector4f& vec);
	glMatrix4f( float* values );
};

#endif