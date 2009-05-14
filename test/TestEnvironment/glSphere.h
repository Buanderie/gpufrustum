#pragma once

#include "glPoint.h"

class glSphere
{
public:

	glSphere( glPoint center, float radius );

	glPoint GetCenter() { return m_Center; }

	float GetRadius() { return m_Radius; }

	void SetCenter( const glPoint center ) { m_Center = center; }

	void SetRadius( float radius ) { m_Radius = radius; }

	void Draw( );

	void SetInsideFrustum( bool value ) { m_IsInsideFrustum = value; }

private:

	glPoint m_Center;
	float	m_Radius;
	bool	m_IsInsideFrustum;

};