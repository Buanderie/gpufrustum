// glVector.h: interface for the glVector class.
//
//////////////////////////////////////////////////////////////////////

#ifndef __GLVECTOR_H__
#define __GLVECTOR_H__

#include <windows.h>		// Header File For Windows
#include <gl\gl.h>			// Header File For The OpenGL32 Library
#include <gl\glu.h>			// Header File For The GLu32 Library

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

class glVector  
{
public:
	void operator *=(GLfloat scalar);
	glVector();
	virtual ~glVector();

	GLfloat k;
	GLfloat j;
	GLfloat i;
};

#endif // !defined(AFX_GLVECTOR_H__F526A5CF_89B5_4F20_8F2C_517D83879D35__INCLUDED_)
