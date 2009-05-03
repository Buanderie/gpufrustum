#ifndef __GLLINE_H__
#define __GLLINE_H__

#include "glVector4f.h"

class glLine
{
public:
	glVector4f m_POO;
	glVector4f m_Vec;
	glLine(){};
	glLine(glVector4f PointOfOrigin, glVector4f Direction);
};

#endif