#include "SphereGenerator.h"
#include "glVector4f.h"
#include <SFML/System/Randomizer.hpp>

namespace Bench
{

SphereGenerator::SphereGenerator( float worldDimX, float worldDimY ) :
	Generator( worldDimX, worldDimY )
{
}

void SphereGenerator::SetSphereRadius( float radiusMin, float radiusMax )
{
	m_RadiusMin = radiusMin; 
	m_RadiusMax = radiusMax;
}

void SphereGenerator::Generate( unsigned int count, float* data )
{
	for( int i = 0; i < count; ++i )
	{
		float r = sf::Randomizer::Random( m_RadiusMin, m_RadiusMax );

		glVector4f pos = GetRandomPosition( );

		data[ i * 4		] = pos.x;
		data[ i * 4 + 1 ] = pos.y;
		data[ i * 4 + 2 ] = pos.z;
		data[ i * 4 + 3 ] = r;
	}
}

}
