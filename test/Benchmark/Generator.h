#pragma once

class glVector4f;

namespace Bench
{
	class Generator
	{
	public:

		Generator( float worldDimX, float worldDimY );

		virtual void Generate( unsigned int count, float* data ) = 0;

		float GetWorldDimX( );

		float GetWorldDimY( );

	protected:

		glVector4f GetRandomPosition( );

	private:

		float m_WorldDimX;
		float m_WorldDimY;
	};
}