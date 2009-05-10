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

	protected:

		glVector GetRandomRotations( );
	};
}
