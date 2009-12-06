#include "glPlane.h"
#include "glVector4f.h"

glPlane::glPlane( glVector4f Normal )
{
	m_Normal = Normal;
}

glPlane::glPlane()
{

}

glVector4f glPlane::intersect2Planes( glPlane p1, glPlane p2 )
{
	glVector4f res;
	res =	(p1.m_Normal.crossProduct(p2.m_Normal))*-m_Normal[3]+
			(p2.m_Normal.crossProduct(m_Normal))*-p1.m_Normal[3] +
			(m_Normal.crossProduct(p1.m_Normal))*-p2.m_Normal[3];
	
	glVector4f r2;
	r2 = p1.m_Normal.crossProduct(p2.m_Normal);

	float scal = m_Normal.scalarProduct(r2);
	scal = 1.0f/scal;
	res = res * scal;

	return res;
}