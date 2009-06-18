#include "PyrFrustumGenerator.h"
#include <SFML/System/Randomizer.hpp>
#include "glVector4f.h"
#include "glPyramidalFrustum.h"
#include "glVector.h"
#include <math.h>

namespace Bench
{

PyrFrustumGenerator::PyrFrustumGenerator( float worldDimX, float worldDimY ) :
	Generator( worldDimX, worldDimY )
{

}

void PyrFrustumGenerator::SetVolumeDistances( float depth, float height, float width )
{
	m_Depth = depth; m_Height = height; m_Width = width;
}

void PyrFrustumGenerator::Generate( unsigned int count, float* data )
{
	float nearDistance  = 1.0f;
	float farDistance	= m_Depth - nearDistance;
	float fov			= 2 * atan( m_Width / ( 2 * m_Depth ) );
	float ratio			= m_Width / m_Height;

	for( unsigned int i = 0; i < count ; ++i )
	{
		glVector4f	position	= GetRandomPosition();
		glVector	rotations	= GetRandomRotations(); 

		glPyramidalFrustum frustum( fov, nearDistance, farDistance, ratio, position, rotations.i, rotations.j, rotations.k );

		int cornerOffset = count * 6 * 4;

		frustum.extractPlanesData ( &data[ 0			] );
		frustum.extractCornersData( &data[ cornerOffset ] );
	}
}

glVector PyrFrustumGenerator::GetRandomRotations( )
{
	glVector rotations;

	rotations.i = sf::Randomizer::Random( 0.f, 360.f );
	rotations.j = sf::Randomizer::Random( 0.f, 360.f );
	rotations.k = sf::Randomizer::Random( 0.f, 360.f );

	return rotations;
}

}