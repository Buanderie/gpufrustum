#include <windows.h>		// Header File For Windows
#include <gl\gl.h>			// Header File For The OpenGL32 Library
#include <gl\glu.h>			// Header File For The GLu32 Library
#include "glAABB.h"

glAABB::glAABB(glVector4f minPos, glVector4f maxPos)
{
	m_MinPos = minPos;
	m_MaxPos = maxPos;
	isInsideFrustum = false;
}

glAABB::glAABB(const glAABB& val)
{
	(*this) = val;
}

void glAABB::draw()
{
	if( isInsideFrustum )
		glColor3f(0.0f,1.0f,0.0f);
	else
		glColor3f(1.0f,0.0f,0.0f);
	glBegin(GL_QUADS);
	glVertex3f(m_MinPos.x, m_MinPos.y, m_MinPos.z);
	glVertex3f(m_MinPos.x, m_MaxPos.y, m_MinPos.z);
	glVertex3f(m_MaxPos.x, m_MaxPos.y, m_MinPos.z);
	glVertex3f(m_MaxPos.x, m_MinPos.y, m_MinPos.z);

	glVertex3f(m_MinPos.x, m_MinPos.y, m_MaxPos.z);
	glVertex3f(m_MinPos.x, m_MaxPos.y, m_MaxPos.z);
	glVertex3f(m_MaxPos.x, m_MaxPos.y, m_MaxPos.z);
	glVertex3f(m_MaxPos.x, m_MinPos.y, m_MaxPos.z);

	glVertex3f(m_MinPos.x, m_MinPos.y, m_MinPos.z);
	glVertex3f(m_MinPos.x, m_MinPos.y, m_MaxPos.z);
	glVertex3f(m_MinPos.x, m_MaxPos.y, m_MaxPos.z);
	glVertex3f(m_MinPos.x, m_MaxPos.y, m_MinPos.z);

	glVertex3f(m_MaxPos.x, m_MinPos.y, m_MinPos.z);
	glVertex3f(m_MaxPos.x, m_MaxPos.y, m_MinPos.z);
	glVertex3f(m_MaxPos.x, m_MaxPos.y, m_MaxPos.z);
	glVertex3f(m_MaxPos.x, m_MinPos.y, m_MaxPos.z);

	glVertex3f(m_MinPos.x, m_MinPos.y, m_MinPos.z);
	glVertex3f(m_MinPos.x, m_MinPos.y, m_MaxPos.z);
	glVertex3f(m_MaxPos.x, m_MinPos.y, m_MaxPos.z);
	glVertex3f(m_MaxPos.x, m_MinPos.y, m_MinPos.z);

	glVertex3f(m_MinPos.x, m_MaxPos.y, m_MinPos.z);
	glVertex3f(m_MinPos.x, m_MaxPos.y, m_MaxPos.z);
	glVertex3f(m_MaxPos.x, m_MaxPos.y, m_MaxPos.z);
	glVertex3f(m_MaxPos.x, m_MaxPos.y, m_MinPos.z);
	glEnd();

	glColor3f(1.0f, 1.0f, 1.0f);
	glBegin(GL_LINES);
	glVertex3f( m_MinPos.x, m_MinPos.y, m_MinPos.z );
	glVertex3f( m_MaxPos.x, m_MinPos.y, m_MinPos.z );
	glVertex3f( m_MaxPos.x, m_MinPos.y, m_MinPos.z );
	glVertex3f( m_MaxPos.x, m_MaxPos.y, m_MinPos.z );
	glVertex3f( m_MaxPos.x, m_MaxPos.y, m_MinPos.z );
	glVertex3f( m_MinPos.x, m_MaxPos.y, m_MinPos.z );
	glVertex3f( m_MinPos.x, m_MaxPos.y, m_MinPos.z );
	glVertex3f( m_MinPos.x, m_MinPos.y, m_MinPos.z );
	//
	glVertex3f( m_MinPos.x, m_MinPos.y, m_MaxPos.z );
	glVertex3f( m_MaxPos.x, m_MinPos.y, m_MaxPos.z );
	glVertex3f( m_MaxPos.x, m_MinPos.y, m_MaxPos.z );
	glVertex3f( m_MaxPos.x, m_MaxPos.y, m_MaxPos.z );
	glVertex3f( m_MaxPos.x, m_MaxPos.y, m_MaxPos.z );
	glVertex3f( m_MinPos.x, m_MaxPos.y, m_MaxPos.z );
	glVertex3f( m_MinPos.x, m_MaxPos.y, m_MaxPos.z );
	glVertex3f( m_MinPos.x, m_MinPos.y, m_MaxPos.z );
	//
	glVertex3f( m_MinPos.x, m_MinPos.y, m_MinPos.z );
	glVertex3f( m_MinPos.x, m_MinPos.y, m_MaxPos.z );
	glVertex3f( m_MaxPos.x, m_MinPos.y, m_MinPos.z );
	glVertex3f( m_MaxPos.x, m_MinPos.y, m_MaxPos.z );

	glVertex3f( m_MinPos.x, m_MaxPos.y, m_MinPos.z );
	glVertex3f( m_MinPos.x, m_MaxPos.y, m_MaxPos.z );
	glVertex3f( m_MaxPos.x, m_MaxPos.y, m_MinPos.z );
	glVertex3f( m_MaxPos.x, m_MaxPos.y, m_MaxPos.z );
	glEnd();
}

void glAABB::extractCornersData( float* out )
{
	out[0] = m_MinPos.x;
	out[1] = m_MinPos.y;
	out[2] = m_MinPos.z;

	out[3] = m_MaxPos.x;
	out[4] = m_MinPos.y;
	out[5] = m_MinPos.z;

	out[6] = m_MaxPos.x;
	out[7] = m_MaxPos.y;
	out[8] = m_MinPos.z;

	out[9] = m_MinPos.x;
	out[10] = m_MaxPos.y;
	out[11] = m_MinPos.z;

	out[12] = m_MinPos.x;
	out[13] = m_MinPos.y;
	out[14] = m_MaxPos.z;

	out[15] = m_MaxPos.x;
	out[16] = m_MinPos.y;
	out[17] = m_MaxPos.z;

	out[18] = m_MaxPos.x;
	out[19] = m_MaxPos.y;
	out[20] = m_MaxPos.z;

	out[21] = m_MinPos.x;
	out[22] = m_MaxPos.y;
	out[23] = m_MaxPos.z;
}