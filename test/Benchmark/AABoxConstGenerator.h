#pragma once

#include "AABoxGenerator.h"

namespace Bench
{

	class AABoxConstGenerator :
		public AABoxGenerator
	{
	public:

		AABoxConstGenerator( float worldDimX, float worldDimY );

		void SetBoxDimensions( float width, float height, float depth );

		void Generate( unsigned int count, float* data );

	private:

		float m_BoxWidth;
		float m_BoxHeight;
		float m_BoxDepth;
	};

}