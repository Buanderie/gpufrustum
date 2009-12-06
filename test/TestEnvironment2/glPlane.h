#ifndef __GLPLANE_H__
#define __GLPLANE_H__

#include "glVector4f.h"

class glPlane
{
public:
	glPlane();
	glPlane( glVector4f Normal );
	glVector4f m_Normal;
	glVector4f intersect2Planes( glPlane p1, glPlane p2 );
};

#endif