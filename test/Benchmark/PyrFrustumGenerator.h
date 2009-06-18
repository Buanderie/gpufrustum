#pragma once

#include "Generator.h"

class glVector4f;
class glVector;

namespace Bench
{
	class PyrFrustumGenerator :
		public Generator
	{
	public:

		PyrFrustumGenerator( float worldDimX, float worldDimY );

		void SetVolumeDistances( float depth, float height, float width );

		void Generate( unsigned int count, float* data );

	private:

		glVector PyrFrustumGenerator::GetRandomRotations( );

	private:

		float m_Depth;
		float m_Height;
		float m_Width;
	};
}
