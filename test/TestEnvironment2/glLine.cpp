#include "glLine.h"

glLine::glLine(glVector4f PointOfOrigin, glVector4f Direction)
{
	m_POO = PointOfOrigin;
	m_Vec = Direction;
	m_POO.normalize();
	m_Vec.normalize();
}