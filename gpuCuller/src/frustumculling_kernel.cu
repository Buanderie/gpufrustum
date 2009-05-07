#include <gpuCuller_kernel.h>

void ClassifyPlanesPoints( dim3 gridSize, dim3 blockSize, const void* iplanes, const void* ipoints, int nPlane, int nPoint, int* out )
{
	ClassifyPlanesPoints<<< gridSize, blockSize >>>( ( plane* )iplanes, ( point3d* )ipoints, nPlane, nPoint, out );
}

void ClassifyPyramidalFrustumBoxes( dim3 gridSize, dim3 blockSize, const float* frustumCorners, const float* boxPoints, const int* planePointClassification, int planeCount, int pointCount, int* out )
{
	ClassifyPyramidalFrustumBoxes<<< gridSize, blockSize >>>( (point3d*)frustumCorners, (point3d*)boxPoints, planePointClassification, planeCount, pointCount, out );
}

/**
		p0	   p1     ...  pn	
	v0 [ i00 ][ i10 ] ... [ in0 ]
	v1 [ i01 ][ i11 ] ... [ in1 ]
	...  ...    ...   ...   ...
	vm [ i0m ][ i1m ] ... [ inm ]

	pI = plane number I. {a, b, c, d}
	vJ = point number J. {x, y, z}

	iIJ = intersection between the plane I and the point J. -1 = Back 1 = Front.

	n = plane count - 1.
	m = point count - 1.
*/
__global__ void
ClassifyPlanesPoints( const plane* iplanes, const point3d* ipoints, int planeCount, int pointCount, int* out )
{
	//On recupere l'indice du resultat de classification dans la matrice resultat
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	//Si le thread travaille en dehors des dimensions de la matrice, on ne fait rien
	if( x >= planeCount || y >= pointCount )
		return;

	/*
	//Transfert Global Memory -> Shared Memory
	// Shared memory pour les plans
	__shared__ float splanes[blockDim.y];
	// Shared memory pour les points
	__shared__ float spoints[blockDim.x];
	// Shared memory pour la sortie
	__shared__ float sout[BLOCK_SIZE][BLOCK_SIZE];
	*/

	float p = iplanes[x].a * ipoints[y].x + iplanes[x].b * ipoints[y].y + iplanes[x].c * ipoints[y].z + iplanes[x].d;

	//BACK
	if( p >= 0 )
	{
		out[ planeCount * y + x ] = -1;
		return;
	}
	else //FRONT
	{
		out[ planeCount * y + x ] = 1;
		return;
	}
}

/**
		 p0	    p1	   p2	  p3	 p4	    p5	
	v0	[ r00 ][ r10 ][ r20 ][ r30 ][ r40 ][ r50 ]
	v1	[ r01 ][ r11 ][ r21 ][ r31 ][ r41 ][ r51 ]
	v2	[ r02 ][ r12 ][ r22 ][ r32 ][ r42 ][ r52 ]		  f0
	v3	[ r03 ][ r13 ][ r23 ][ r33 ][ r43 ][ r53 ]	-> b0[ c00 ]
	v4	[ r04 ][ r14 ][ r24 ][ r34 ][ r44 ][ r54 ]
	v5	[ r05 ][ r15 ][ r25 ][ r35 ][ r45 ][ r55 ]
	v6	[ r06 ][ r16 ][ r26 ][ r36 ][ r46 ][ r56 ]
	v7	[ r07 ][ r17 ][ r27 ][ r37 ][ r47 ][ r57 ]
		[ s0  ][ s1  ][ s2  ][ s3  ][ s4  ][ s5  ]	

	vI = point number I
	pJ = plane number J

	rIJ = intersection between the point I and the plane J { 0, 1 }

	fK = frustum number K
	bL = box number L

	sJ = sum of rJx values. Equals to 8 if all box points are in front. Equals to -8 if all box points are in back.

	cKL = classification { GCU_ENCOSING, GCU_INSIDE, GCU_SPANNING, GCU_OUTSIDE }

	One thread is assigned to the classification of one box with one frustum.
*/
__global__ void
ClassifyPyramidalFrustumBoxes( const point3d* frustumCorners, const point3d* boxPoints, const int* planePointClassification, int planeCount, int pointCount, int* out )
{
	int threadX = blockIdx.x * blockDim.x + threadIdx.x;
	int threadY = blockIdx.y * blockDim.y + threadIdx.y;

	int frustumCount = planeCount / 6;
	int boxCount	 = pointCount / 8;

	if( threadX >= frustumCount || threadY >= boxCount )
		return;

	//--------------------
	// Step 1 : Sum each 
	// column

	int sums[ 6 ];

	// For each frustum plane.
	for( int i = 0; i < 6; ++i )
	{
		sums[ i ] = 0;

		// For each point.
		for( int j = 0; j < 8; ++j )
		{
			int index = ( threadX * 6 + i ) + ( ( threadY + j ) * planeCount );

			sums[ i ] += planePointClassification[ index ];
		}
	}

	//--------------------
	// Step 2 : Determine 
	// the classification

	int outIndex = threadX + threadY * frustumCount;

	int arrayCountEight = CountArrayElementValue( sums, 6, 8 );
	
	if( arrayCountEight == 6 )
	{
		// All points are inside.
		out[ 0 ] = 1; // GCU_INSIDE
		return;
	}
	else
	{
		int arrayCountMinusEight = CountArrayElementValue( sums, 6, -8 );

		if( arrayCountMinusEight > 0 )
		{
			// The box is outside to one or several planes.
			out[ outIndex ] = 3; // GCU_OUTSIDE
		}
		else
		{
			int frustumIndex = threadX;
			int boxIndex	 = threadY;

			bool spanning = false;

			for( int p = 0; p < 8; ++p )
			{
				int frustumCornerIndex = frustumIndex * 8 + p;

				point3d currentCorner = frustumCorners[ frustumCornerIndex ];

				point3d upperBoxPoint = UpperPoint( &boxPoints[ boxIndex ] );
				point3d lowerBoxPoint = LowerPoint( &boxPoints[ boxIndex ] );

				if( ( lowerBoxPoint.x > currentCorner.x ) || ( currentCorner.x > upperBoxPoint.x )
        			||
        			( lowerBoxPoint.y > currentCorner.y ) || ( currentCorner.y > upperBoxPoint.y )
        			||
        			( lowerBoxPoint.z > currentCorner.z ) || ( currentCorner.x > upperBoxPoint.z ) ) 
				{
        			// The box intersects several planes.
					out[ outIndex ] = 2; // GCU_SPANNING

					spanning = true;
        		} 
			}

			if( !spanning )
			{
				// default case
				out[ outIndex ] = 0; // GCU_ENCOSING
			}
		}
	}
}

__device__ int 
SumArrayElements( const int* array, int elementCount )
{
	int sum = 0;
	for( int i = 0; i < elementCount; ++i )
	{
		sum += array[ i ];
	}
	return sum;
}

__device__ point3d
UpperPoint( const point3d* box )
{
	float maxX, maxY, maxZ;

	maxX = maxY = maxZ = -1.175494351e-38; // min float (4bytes)

	for( int i = 0; i < 8; ++i )
	{
		if( box[ i ].x > maxX ) { maxX = box[ i ].x; }
		if( box[ i ].y > maxY ) { maxY = box[ i ].y; }
		if( box[ i ].z > maxZ ) { maxZ = box[ i ].z; }
	}

	point3d result = { maxX, maxY, maxZ };
	return result;
}

__device__ point3d
LowerPoint( const point3d* box )
{
	float minX, minY, minZ;

	minX = minY = minZ = 3.402823466e+38; // max float (4bytes)

	for( int i = 0; i < 8; ++i )
	{
		if( box[ i ].x < minX ) { minX = box[ i ].x; }
		if( box[ i ].y < minY ) { minY = box[ i ].y; }
		if( box[ i ].z < minZ ) { minZ = box[ i ].z; }
	}

	point3d result = { minX, minY, minZ };
	return result;
}

__device__ int
CountArrayElementValue( const int* array, int elementCount, int elementValue )
{
	int count = 0;
	for( int i = 0; i < elementCount; ++i )
	{
		if( array[ i ] == elementValue )
		{
			++count;
		}
	}
	return count;
}