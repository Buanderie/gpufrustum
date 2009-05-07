#include <windows.h>		// Header File For Windows
#include <gl\gl.h>			// Header File For The OpenGL32 Library
#include <gl\glu.h>			// Header File For The GLu32 Library
#include "glAABB.h"

glAABB::glAABB(glVector4f minPos, glVector4f maxPos)
{
	m_MinPos = minPos;
	m_MaxPos = maxPos;
}

glAABB::glAABB(const glAABB& val)
{
	(*this) = val;
}

void glAABB::draw()
{
	glColor3f(0.0f,1.0f,0.0f);
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

	glColor3f(1.0f, 0.0f, 0.0f);
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