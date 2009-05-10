#pragma once

#include "PyrFrustumGenerator.h"

namespace Bench
{
	class PyrFrustumConstGenerator : 
		public PyrFrustumGenerator
	{
	public:

		PyrFrustumConstGenerator( float worldDimX, float worldDimY );

		void SetVolumeDistances( float depth, float height, float width );

		void Generate( unsigned int count, float* data );

	private:

		float m_Depth;
		float m_Height;
		float m_Width;
	};

}