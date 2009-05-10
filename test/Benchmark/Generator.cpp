#include "Generator.h"
#include "glVector4f.h"
#include <SFML/System/Randomizer.hpp>

namespace Bench
{

Generator::Generator( float worldDimX, float worldDimY ) :
	m_WorldDimX( worldDimX ),
	m_WorldDimY( worldDimY )
{

}

float Generator::GetWorldDimX( )
{
	return m_WorldDimX;
}

float Generator::GetWorldDimY( )
{
	return m_WorldDimY;
}

glVector4f Generator::GetRandomPosition( )
{
	glVector4f position;

	position.x = sf::Randomizer::Random( 0.f, GetWorldDimX() );
	position.y = sf::Randomizer::Random( 0.f, GetWorldDimY() );
	position.z = 0.f;
	position.w = 1.f;

	return position;
}

}