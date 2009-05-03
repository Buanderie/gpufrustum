#include <cmath>
#include "glVector4f.h"

float& glVector4f::operator [](unsigned int i)
{
	switch(i)
	{
	case 0:
		return x;
	case 1:
		return y;
	case 2:
		return z;
	case 3:
		return w;
	default:
		return w;
	}
}

glVector4f::~glVector4f()
{

}

glVector4f::glVector4f(void)
{
	glVector4f(0,0,0,0);
}

glVector4f::glVector4f(glVector4f& val)
{
	x = val.x;
	y = val.y;
	z = val.z;
	w = val.w;
}

glVector4f::glVector4f(float X, float Y, float Z, float W)
:x(X), y(Y), z(Z), w(W)
{}

glVector4f glVector4f::crossProduct( glVector4f c )
{
	//Precompute some 2x2 matrix determinants for speed
	/*float Pxy = x*c.y - c.x*y;
	float Pxz = x*c.z - c.x*z;
	float Pxw = x*c.w - c.x*w;
	float Pyz = y*c.z - c.y*z;
	float Pyw = y*c.w - c.y*w;
	float Pzw = z*c.w - c.z*w;
	glVector4f pol(
	y*Pzw - z*Pyw + w*Pyz,    //Note the lack of 'x' in this line
	-z*Pxw + x*Pzw - w*Pxz,    //y, Etc.
	x*Pyw - y*Pxw + w*Pxy,
	-y*Pxz + x*Pyz - z*Pxy
	);
	return pol;*/
	//normalize();
	//c.normalize();
	glVector4f ret(		y*c.z - z*c.y,
						z*c.x - x*c.z,
						x*c.y - y*c.x,
						0
						);
	return ret;		
}

void glVector4f::operator *=(float scalar)
{
	x *= scalar;
	y *= scalar;
	z *= scalar;
	w *= scalar;
}

glVector4f& glVector4f::operator =(const glVector4f& v)
{
	x = v.x;
	y = v.y;
	z = v.z;
	w = v.w;
	return glVector4f(x,y,z,w);
}

glVector4f glVector4f::operator *(float scalar)
{
	return glVector4f( x*scalar, y*scalar, z*scalar, w*scalar );
}

glVector4f glVector4f::operator +(glVector4f& v2)
{
	glVector4f pol( x + v2.x, y+v2.y, z+v2.z, w+v2.w );
	return pol;
}

glVector4f glVector4f::operator -(glVector4f& v2)
{
	glVector4f pol( x - v2.x, y-v2.y, z-v2.z, w-v2.w );
	return pol;
}

float glVector4f::scalarProduct( glVector4f c )
{
	return (x*c.x + y*c.y + z+c.z);
}

void glVector4f::normalize()
{
	float norm = sqrt( x*x + y*y + z*z );
	x = x/norm;
	y = y/norm;
	z = z/norm;
}