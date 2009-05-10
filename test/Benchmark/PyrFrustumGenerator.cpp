#include "PyrFrustumGenerator.h"
#include <SFML/System/Randomizer.hpp>
#include "glVector4f.h"
#include "glVector.h"

namespace Bench
{

PyrFrustumGenerator::PyrFrustumGenerator( float worldDimX, float worldDimY ) :
	Generator( worldDimX, worldDimY )
{
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