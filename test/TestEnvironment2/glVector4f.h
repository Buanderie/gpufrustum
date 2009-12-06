// glVector.h: interface for the glVector class.
//
//////////////////////////////////////////////////////////////////////

#ifndef __GLVECTOR4F_H__
#define __GLVECTOR4F_H__


class glVector4f
{
public:
	void operator *=(float scalar);
	glVector4f operator *(float scalar);
	glVector4f operator +(glVector4f& v2);
	glVector4f operator -(glVector4f& v2);
	glVector4f& operator =(const glVector4f& v);
	float& operator [](unsigned int i);
	glVector4f();
	glVector4f( float X, float Y, float Z, float W );
	glVector4f(glVector4f& val);
	virtual ~glVector4f();

	glVector4f crossProduct( glVector4f c );
	float scalarProduct( glVector4f c );
	void normalize();

	float x;
	float y;
	float z;
	float w;
};

#endif
