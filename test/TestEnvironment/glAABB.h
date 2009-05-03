#ifndef __GLAABB_H__
#define __GLAABB_H__

#include "glVector4f.h"

class glAABB
{
private:
	glVector4f m_MinPos;
	glVector4f m_MaxPos;

public:
	glAABB(glVector4f minPos, glVector4f maxPos);
	void draw();
};

#endif