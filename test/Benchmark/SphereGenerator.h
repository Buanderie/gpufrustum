#pragma once

#include "Generator.h"

namespace Bench
{

	class SphereGenerator :
		public Generator
	{
	public:
		SphereGenerator( float worldDimX, float worldDimY );

		void SetSphereRadius( float radiusMin, float radiusMax );

		void Generate( unsigned int count, float* data );

	private:

		float m_RadiusMin;
		float m_RadiusMax;

	};

}
