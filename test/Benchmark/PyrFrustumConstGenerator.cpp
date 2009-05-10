#include "PyrFrustumGenerator.h"
#include "PyrFrustumConstGenerator.h"
#include <SFML/System/Randomizer.hpp>
#include <math.h>
#include "glVector4f.h"
#include "glVector.h"
#include "glPyramidalFrustum.h"

namespace Bench
{

PyrFrustumConstGenerator::PyrFrustumConstGenerator( float worldDimX, float worldDimY ) :
	PyrFrustumGenerator( worldDimX, worldDimY )
{

}

void PyrFrustumConstGenerator::SetVolumeDistances( float depth, float height, float width )
{
	m_Depth = depth; m_Height = height; m_Width = width;
}

void PyrFrustumConstGenerator::Generate( unsigned int count, float* data )
{
	float nearDistance  = 1.0f;
	float farDistance	= m_Depth - nearDistance;
	float fov			= 2 * atan( m_Width / ( 2 * m_Depth ) );
	float ratio			= m_Width / m_Height;

	for( int i = 0; i < count ; ++i )
	{
		glVector4f	position	= GetRandomPosition();
		glVector	rotations	= GetRandomRotations(); 

		glPyramidalFrustum frustum( fov, nearDistance, farDistance, ratio, position, rotations.i, rotations.j, rotations.k );

		int cornerOffset = count * 6 * 4;

		frustum.extractPlanesData ( &data[				  i * 6 * 4 ] );
		frustum.extractCornersData( &data[ cornerOffset + i * 8 * 4 ] );
	}
}

}