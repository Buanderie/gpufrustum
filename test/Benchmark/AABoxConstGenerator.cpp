#include "AABoxConstGenerator.h"
#include "glVector4f.h"
#include "glAABB.h"

namespace Bench
{

AABoxConstGenerator::AABoxConstGenerator( float worldDimX, float worldDimY ) :
	AABoxGenerator( worldDimX, worldDimY )
{
}

void AABoxConstGenerator::SetBoxDimensions( float width, float height, float depth )
{
	m_BoxWidth	= width;
	m_BoxHeight = height;
	m_BoxDepth	= depth;
}

void AABoxConstGenerator::Generate( unsigned int count, float* data )
{
	glVector4f offset( m_BoxWidth / 2, m_BoxHeight / 2, m_BoxDepth / 2, 1.f );

	for( int i = 0; i < count ; ++i )
	{
		glVector4f center = GetRandomPosition();

		glAABB box( center - offset, center + offset );

		box.extractCornersData( &data[ i * 8 * 3 ] );
	}
}

}