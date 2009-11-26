#ifndef __GLAABB_H__
#define __GLAABB_H__

#include "glVector4f.h"

class glAABB
{
public:
	glVector4f m_MinPos;
	glVector4f m_MaxPos;

public:
	glAABB(glVector4f minPos, glVector4f maxPos);
	glAABB(const glAABB& val);
	void draw();
	void drawColor(float r, float g, float b);
	void extractCornersData( float* out );
	bool isInsideFrustum;
};

#endif