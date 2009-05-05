#ifndef __GLFRUSTUM_H__
#define __GLFRUSTUM_H__

#include "glPlane.h"
#include "glMatrix4f.h"

class glPyramidalFrustum
{
public:
	
	glPlane m_Left;
	glPlane m_Right;
	glPlane m_Up;
	glPlane m_Down;
	glPlane m_Near;
	glPlane m_Far;
	
	glVector4f p[8];

	float m_FOV;
	float m_Neard;
	float m_Fard;
	float m_AspectRatio;
	float m_RotX;
	float m_RotY;
	float m_RotZ;
	glVector4f m_Position;

	glPyramidalFrustum();
	void computePointsAndPlanes();
	glPyramidalFrustum( float FOV, float Near, float Far, float AspectRatio, glVector4f Pos, float RotX, float RotY, float RotZ );
	glPyramidalFrustum(const glPyramidalFrustum& val);
	glPyramidalFrustum& glPyramidalFrustum::operator =(const glPyramidalFrustum& val);
	~glPyramidalFrustum();
	
	static void fromModelViewMatrix( float* mat, glPyramidalFrustum& frustum );
	static void fromModelViewMatrix( glMatrix4f& mat, glPyramidalFrustum& frustum );
	
	void DoSomething();
	
	void draw();

	void extractPlanesData( float* out );
	void extractCornersData( float* out );

};

#endif