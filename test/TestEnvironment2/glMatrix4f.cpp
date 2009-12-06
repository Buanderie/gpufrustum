#include "glMatrix4f.h"
#include "glVector4f.h"

glMatrix4f::glMatrix4f(float* values)
{
	for(int i = 0; i < 16; ++i )
	{
		int ligne = i/4;
		int colonne = i-(ligne*4);
		elem[ligne][colonne] = values[i];
	}
}

glVector4f& glMatrix4f::MatVecProduct(glVector4f &vin)
{
   float v0 =   this->elem[0][0]*vin[0] + this->elem[0][1]*vin[1] + 
                     this->elem[0][2]*vin[2] + this->elem[0][3]*vin[3];
   float v1 =  this->elem[1][0]*vin[0] + this->elem[1][1]*vin[1] +
                     this->elem[1][2]*vin[2] + this->elem[1][3]*vin[3];
   float v2 =  this->elem[2][0]*vin[0] + this->elem[2][1]*vin[1] +
                     this->elem[2][2]*vin[2] + this->elem[2][3]*vin[3];
   float v3 =  this->elem[3][0]*vin[0] + this->elem[3][1]*vin[1] + 
                     this->elem[3][2]*vin[2] + this->elem[3][3]*vin[3];
   return (glVector4f(v0,v1,v2,v3)*(1.0f/v3));
}