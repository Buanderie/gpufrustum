#include <windows.h>		// Header File For Windows
#include <gl\gl.h>			// Header File For The OpenGL32 Library
#include <gl\glu.h>			// Header File For The GLu32 Library

#include <cmath>
#include <cstdio>
#include "glPyramidalFrustum.h"
#include "glPlane.h"
#include "glQuaternion.h"
#include "glMatrix4f.h"

glPyramidalFrustum::glPyramidalFrustum()
{

}

glPyramidalFrustum::glPyramidalFrustum( float FOV, float Near, float Far, float AspectRatio, glVector4f Pos, float RotX, float RotY, float RotZ )
{
	m_Position = Pos;
	m_RotX = RotX;
	m_RotY = RotY;
	m_RotZ = RotZ;
	m_FOV = FOV*0.0174532925f;
	m_Neard = Near;
	m_Fard = Far;
	m_AspectRatio = AspectRatio;

	computePointsAndPlanes();
}

void glPyramidalFrustum::computePointsAndPlanes()
{
	//Calcule les coordonnees des coins
	float sinus = sin(m_FOV/2);
	float shiftxN = sinus*m_Neard;
	float shiftyN = m_AspectRatio*shiftxN;
	float shiftxF = sinus*m_Fard;
	float shiftyF = m_AspectRatio*shiftxF;

	p[0] = glVector4f(-shiftxN, shiftyN, m_Neard, 1);
	p[1] = glVector4f(shiftxN, shiftyN, m_Neard, 1);
	p[2] = glVector4f(-shiftxN, -shiftyN, m_Neard, 1);
	p[3] = glVector4f(shiftxN, -shiftyN, m_Neard, 1);
	p[4] = glVector4f(-shiftxF, shiftyF, m_Fard, 1);
	p[5] = glVector4f(shiftxF, shiftyF, m_Fard, 1);
	p[6] = glVector4f(-shiftxF, -shiftyF, m_Fard, 1);
	p[7] = glVector4f(shiftxF, -shiftyF, m_Fard, 1);

	//Rotation
	glQuaternion rotY, rotX, rotZ, rotFinal;
	rotY.CreateFromAxisAngle(0.0f, 1.0f, 0.0f, m_RotY);
	rotX.CreateFromAxisAngle(1.0f, 0.0f, 0.0f, m_RotX);
	rotZ.CreateFromAxisAngle(0.0f, 0.0f, 1.0f, m_RotZ);
	rotFinal = rotX * rotY * rotZ;
	float mat[16];
	rotFinal.CreateMatrix(mat);
	glMatrix4f mato(mat);
	
	//Application + Translation
	for(int i = 0; i < 8; ++i )
	{
		p[i] = (mato.MatVecProduct(p[i]));
		p[i] = p[i] + m_Position;
	}

	//Compute plane normal vectors now...
	glVector4f v, u, n;
	//Near
	u = p[1] - p[0];
	v = p[2] - p[0];
	n = u.crossProduct(v);
	n.normalize();
	n.w = -(n.x*p[0].x + n.y*p[0].y + n.z*p[0].z);
	m_Near = glPlane( n );

	//Far
	u = p[5] - p[4];
	v = p[6] - p[4];
	n = v.crossProduct(u);
	n.normalize();
	n.w = -(n.x*p[4].x + n.y*p[4].y + n.z*p[4].z);
	m_Far = glPlane( n );

	//Up
	u = p[1] - p[0];
	v = p[4] - p[0];
	n = v.crossProduct(u);
	n.normalize();
	n.w = -(n.x*p[0].x + n.y*p[0].y + n.z*p[0].z);
	m_Up = glPlane( n );

	//Down
	u = p[6] - p[2];
	v = p[3] - p[2];
	n = v.crossProduct(u);
	n.normalize();
	n.w = -(n.x*p[2].x + n.y*p[2].y + n.z*p[2].z);
	m_Down = glPlane( n );

	//Left
	u = p[4] - p[0];
	v = p[2] - p[0];
	n = v.crossProduct(u);
	n.normalize();
	n.w = -(n.x*p[2].x + n.y*p[2].y + n.z*p[2].z);
	m_Left = glPlane( n );

	//Right
	u = p[3] - p[1];
	v = p[5] - p[1];
	n = v.crossProduct(u);
	n.normalize();
	n.w = -(n.x*p[1].x + n.y*p[1].y + n.z*p[1].z);
	m_Right = glPlane( n );
}

glPyramidalFrustum::glPyramidalFrustum(const glPyramidalFrustum& val)
{
	(*this) = val;
}

glPyramidalFrustum& glPyramidalFrustum::operator =(const glPyramidalFrustum& val)
{
	m_Up = val.m_Up;
	m_Down = val.m_Down;
	m_Left = val.m_Left;
	m_Right = val.m_Right;
	m_Near = val.m_Near;
	m_Far = val.m_Far;
	
	for(int i = 0; i < 8; ++i )
		p[i] = val.p[i];

	return (*this);
}

glPyramidalFrustum::~glPyramidalFrustum()
{

}

void ExtractPlane(glPlane& plane, float *mat, int row)
{
	int scale = (row < 0) ? -1 : 1;
	row = abs(row) - 1;
	
	plane.m_Normal[0] = mat[3] + scale * mat[row];
	plane.m_Normal[1] = mat[7] + scale * mat[row + 4];
	plane.m_Normal[2] = mat[11] + scale * mat[row + 8];
	plane.m_Normal[3] = mat[15] + scale * mat[row + 12];
	

	//if( plane.m_Normal[3] != 0 )
	//plane.m_Normal *= (1.0f/plane.m_Normal[3]);
	float length = sqrtf(plane.m_Normal[0] * plane.m_Normal[0] + plane.m_Normal[1] * plane.m_Normal[1] + plane.m_Normal[2] * plane.m_Normal[2]);
	
	plane.m_Normal[0] /= length;
	plane.m_Normal[1] /= length;
	plane.m_Normal[2] /= length;
	plane.m_Normal[3] /= length;
	
}

void glPyramidalFrustum::fromModelViewMatrix( float* mat, glPyramidalFrustum& frustum )
{
	ExtractPlane(frustum.m_Left, mat, 1);
	ExtractPlane(frustum.m_Right, mat, -1);
	ExtractPlane(frustum.m_Down, mat, 2);
	ExtractPlane(frustum.m_Up, mat, -2);
	ExtractPlane(frustum.m_Near, mat, 3);
	ExtractPlane(frustum.m_Far, mat, -3);
}

void glPyramidalFrustum::fromModelViewMatrix( glMatrix4f& mat, glPyramidalFrustum& frustum )
{
	frustum.m_Left.m_Normal[0] = mat.elem[3][0] + mat.elem[0][0];
	frustum.m_Left.m_Normal[1] = mat.elem[3][1] + mat.elem[0][1];
	frustum.m_Left.m_Normal[2] = mat.elem[3][2] + mat.elem[0][2];
	frustum.m_Left.m_Normal[3] = mat.elem[3][3] + mat.elem[0][3];

	frustum.m_Right.m_Normal[0] = mat.elem[3][0] - mat.elem[0][0];
	frustum.m_Right.m_Normal[1] = mat.elem[3][1] - mat.elem[0][1];
	frustum.m_Right.m_Normal[2] = mat.elem[3][2] - mat.elem[0][2];
	frustum.m_Right.m_Normal[3] = mat.elem[3][3] - mat.elem[0][3];

	frustum.m_Up.m_Normal[0] = mat.elem[3][0] - mat.elem[1][0];
	frustum.m_Up.m_Normal[1] = mat.elem[3][1] - mat.elem[1][1];
	frustum.m_Up.m_Normal[2] = mat.elem[3][2] - mat.elem[1][2];
	frustum.m_Up.m_Normal[3] = mat.elem[3][3] - mat.elem[1][3];

	frustum.m_Down.m_Normal[0] = mat.elem[3][0] + mat.elem[1][0];
	frustum.m_Down.m_Normal[1] = mat.elem[3][1] + mat.elem[1][1];
	frustum.m_Down.m_Normal[2] = mat.elem[3][2] + mat.elem[1][2];
	frustum.m_Down.m_Normal[3] = mat.elem[3][3] + mat.elem[1][3];

	frustum.m_Near.m_Normal[0] = mat.elem[3][0] + mat.elem[2][0];
	frustum.m_Near.m_Normal[1] = mat.elem[3][1] + mat.elem[2][1];
	frustum.m_Near.m_Normal[2] = mat.elem[3][2] + mat.elem[2][2];
	frustum.m_Near.m_Normal[3] = mat.elem[3][3] + mat.elem[2][3];

	frustum.m_Far.m_Normal[0] = mat.elem[3][0] - mat.elem[2][0];
	frustum.m_Far.m_Normal[1] = mat.elem[3][1] - mat.elem[2][1];
	frustum.m_Far.m_Normal[2] = mat.elem[3][2] - mat.elem[2][2];
	frustum.m_Far.m_Normal[3] = mat.elem[3][3] - mat.elem[2][3];

}

void glPyramidalFrustum::DoSomething()
{
	glVector4f kokiko(1,0,0,24);
	glVector4f kikoki(0,1,0,30);
	glVector4f bak(0,0,1,40);
	glPlane A;
	A.m_Normal = kokiko;
	glPlane B; B.m_Normal = kikoki;
	glPlane C; C.m_Normal = bak;
	
	glVector4f pol = A.intersect2Planes( B, C );
	int kikoo = 56;
}

void glPyramidalFrustum::draw()
{
	glColor3f(0.0f, 0.0f, 1.0f);
	glBegin(GL_QUADS);
	//Near
	glVertex3f( p[0].x, p[0].y, p[0].z );
	glVertex3f( p[1].x, p[1].y, p[1].z );
	glVertex3f( p[3].x, p[3].y, p[3].z );
	glVertex3f( p[2].x, p[2].y, p[2].z );

	//Far
	glVertex3f( p[4].x, p[4].y, p[4].z );
	glVertex3f( p[5].x, p[5].y, p[5].z );
	glVertex3f( p[7].x, p[7].y, p[7].z );
	glVertex3f( p[6].x, p[6].y, p[6].z );

	//Up
	glVertex3f( p[0].x, p[0].y, p[0].z );
	glVertex3f( p[1].x, p[1].y, p[1].z );
	glVertex3f( p[5].x, p[5].y, p[5].z );
	glVertex3f( p[4].x, p[4].y, p[4].z );

	//Down
	glVertex3f( p[2].x, p[2].y, p[2].z );
	glVertex3f( p[3].x, p[3].y, p[3].z );
	glVertex3f( p[7].x, p[7].y, p[7].z );
	glVertex3f( p[6].x, p[6].y, p[6].z );

	//Left
	glVertex3f( p[0].x, p[0].y, p[0].z );
	glVertex3f( p[2].x, p[2].y, p[2].z );
	glVertex3f( p[6].x, p[6].y, p[6].z );
	glVertex3f( p[4].x, p[4].y, p[4].z );

	//Right
	glVertex3f( p[1].x, p[1].y, p[1].z );
	glVertex3f( p[5].x, p[5].y, p[5].z );
	glVertex3f( p[7].x, p[7].y, p[7].z );
	glVertex3f( p[3].x, p[3].y, p[3].z );

	glEnd();

	glColor3f(1.0,0,0);
	glBegin(GL_LINES);
	//Near
	glVertex3f( p[0].x, p[0].y, p[0].z );
	glVertex3f( p[1].x, p[1].y, p[1].z );
	glVertex3f( p[1].x, p[1].y, p[1].z );
	glVertex3f( p[3].x, p[3].y, p[3].z );
	glVertex3f( p[3].x, p[3].y, p[3].z );
	glVertex3f( p[2].x, p[2].y, p[2].z );
	glVertex3f( p[2].x, p[2].y, p[2].z );
	glVertex3f( p[0].x, p[0].y, p[0].z );
	//Far
	glVertex3f( p[4].x, p[4].y, p[4].z );
	glVertex3f( p[5].x, p[5].y, p[5].z );
	glVertex3f( p[5].x, p[5].y, p[5].z );
	glVertex3f( p[7].x, p[7].y, p[7].z );
	glVertex3f( p[7].x, p[7].y, p[7].z );
	glVertex3f( p[6].x, p[6].y, p[6].z );
	glVertex3f( p[6].x, p[6].y, p[6].z );
	glVertex3f( p[4].x, p[4].y, p[4].z );

	//Other sides
	glVertex3f( p[0].x, p[0].y, p[0].z );
	glVertex3f( p[4].x, p[4].y, p[4].z );
	glVertex3f( p[1].x, p[1].y, p[1].z );
	glVertex3f( p[5].x, p[5].y, p[5].z );
	glVertex3f( p[2].x, p[2].y, p[2].z );
	glVertex3f( p[6].x, p[6].y, p[6].z );
	glVertex3f( p[3].x, p[3].y, p[3].z );
	glVertex3f( p[7].x, p[7].y, p[7].z );

	glEnd();
}

void glPyramidalFrustum::extractPlanesData( float* out )
{
	out[0] = m_Near.m_Normal.x;
	out[1] = m_Near.m_Normal.y;
	out[2] = m_Near.m_Normal.z;
	out[3] = m_Near.m_Normal.w;

	out[4] = m_Far.m_Normal.x;
	out[5] = m_Far.m_Normal.y;
	out[6] = m_Far.m_Normal.z;
	out[7] = m_Far.m_Normal.w;

	out[8] = m_Left.m_Normal.x;
	out[9] = m_Left.m_Normal.y;
	out[10] = m_Left.m_Normal.z;
	out[11] = m_Left.m_Normal.w;

	out[12] = m_Right.m_Normal.x;
	out[13] = m_Right.m_Normal.y;
	out[14] = m_Right.m_Normal.z;
	out[15] = m_Right.m_Normal.w;

	out[16] = m_Up.m_Normal.x;
	out[17] = m_Up.m_Normal.y;
	out[18] = m_Up.m_Normal.z;
	out[19] = m_Up.m_Normal.w;

	out[20] = m_Down.m_Normal.x;
	out[21] = m_Down.m_Normal.y;
	out[22] = m_Down.m_Normal.z;
	out[23] = m_Down.m_Normal.w;
}

void glPyramidalFrustum::extractCornersData( float* out )
{
	for(int i = 0; i < 8; ++i )
	{
		for(int j = 0; j < 3; ++j )
			out[i*3 + j] = p[i][j];
	}
}