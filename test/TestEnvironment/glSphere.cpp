#include "glSphere.h"
#include <Glut.h>

glSphere::glSphere( glPoint center, float radius ) :
	m_Center			( center ),
	m_Radius			( radius ),
	m_IsInsideFrustum	( false  )
{

}

void glSphere::Draw()
{
	glPushMatrix();

	glTranslatef( m_Center.x, m_Center.y, m_Center.z );

	glColor3f( 1.f, 1.f, 1.f );

	glutWireSphere( m_Radius + 0.01f, 8, 8 );

	if( m_IsInsideFrustum )
	{
		glColor3f( 0.f, 1.f, 0.f );
	}
	else
	{
		glColor3f( 1.f, 0.f, 0.f );
	}
	
	glutSolidSphere( m_Radius, 10, 10 );

	glPopMatrix();
}