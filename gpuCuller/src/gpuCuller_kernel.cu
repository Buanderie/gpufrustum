#include <stdio.h>
#include <gpuCuller_kernel.h>
#include <cutil_inline.h>

void ClassifyPlanesPoints( dim3 gridSize, dim3 blockSize, const void* iplanes, const void* ipoints, int nPlane, int nPoint, char* out )
{
	size_t sizeOfSharedMemory = ( blockSize.x * 4 + blockSize.y * 3 ) * sizeof( float );

	ClassifyPlanesPoints<<< gridSize, blockSize, sizeOfSharedMemory >>>( ( float4* )iplanes, ( float3* )ipoints, nPlane, nPoint, out );

	check_cuda_error();
}

void ClassifyPyramidalFrustumBoxes( dim3 gridSize, dim3 blockSize, const float* frustumCorners, const float* boxPoints, const char* planePointClassification, int planeCount, int pointCount, int* out )
{
	ClassifyPyramidalFrustumBoxes<<< gridSize, blockSize >>>( ( float3* )frustumCorners, ( float3* )boxPoints, planePointClassification, planeCount, pointCount, out );

	check_cuda_error();
}

void InverseClassifyPyramidalFrustumBoxes( dim3 gridSize, dim3 blockSize, const float* frustumCorners, const float* boxPoints, int planeCount, int pointCount, int* out )
{
	InverseClassifyPyramidalFrustumBoxes<<< gridSize, blockSize >>>( ( float3* )frustumCorners, ( float3* )boxPoints, planeCount, pointCount, out );

	check_cuda_error();
}

void ClassifyPlanesSpheres( dim3 gridSize, dim3 blockSize, const void* planes, const void* spheres, int planeCount, int sphereCount, char* out )
{
	size_t sizeOfSharedMemory = ( blockSize.x * 4 + blockSize.y * 4 ) * sizeof( float );

	ClassifyPlanesSpheres<<< gridSize, blockSize, sizeOfSharedMemory >>>( ( float4* )planes, ( float4* )spheres, planeCount, sphereCount, out );

	check_cuda_error();
}

void ClassifyPyramidalFrustumSpheres( dim3 gridSize, dim3 blockSize, const char* planeSphereClassification, int frustumCount, int sphereCount, int* out )
{
	size_t sizeOfSharedMemory = blockSize.x * blockSize.y * sizeof( char6 );

	ClassifyPyramidalFrustumSpheres<<< gridSize, blockSize, sizeOfSharedMemory >>>( ( char6* )planeSphereClassification, frustumCount, sphereCount, out );

	check_cuda_error();
}

void ClassifySphericalFrustumSpheres( dim3 gridSize, dim3 blockSize, const float* sphericalFrustums, const float* spheres, int frustumCount, int sphereCount, int* out )
{
	size_t sizeOfSharedMemory = ( blockSize.x * 4 + blockSize.y * 4 ) * sizeof( float );

	ClassifySphericalFrustumSpheres<<< gridSize, blockSize, sizeOfSharedMemory >>>( ( float4* )sphericalFrustums, ( float4* )spheres, frustumCount, sphereCount, out );

	check_cuda_error();
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
ClassifyPlanesPoints( const float4* iplanes, const float3* ipoints, int planeCount, int pointCount, char* out )
{
	//On recupere l'indice du resultat de classification dans la matrice resultat
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	//Si le thread travaille en dehors des dimensions de la matrice, on ne fait rien
	if( x >= planeCount || y >= pointCount )
	{
		return;
	}

	// Shared memory used to read input data.
	// Shared memory is only used by same block threads. 
	// * 0 to m   = planes
	// * n+1 to n = points 
	// The size of this array must be number_of_threads_per_block * sizeof( float ).
	extern __shared__ float sharedMemory[];

	float4* sharedPlanes = ( float4* )&sharedMemory[ 0			    ];
	float3* sharedPoints = ( float3* )&sharedMemory[ blockDim.x * 4 ];

	int planeShIndex = threadIdx.x;	// offset to the first coordinate of the plane.
	int pointShIndex = threadIdx.y;	// offset to the first coordinate of the point.

	// Only threads which are owned by the first row and the first column load data.
	if( threadIdx.x == 0 && threadIdx.y == 0 )
	{
		// Load the 1st plane and the 1st point
		sharedPlanes[ 0 ] = iplanes[ x ];
		sharedPoints[ 0 ] = ipoints[ y ];
	}
	else if( threadIdx.x == 0 )
	{
		// Load the n-th point
		sharedPoints[ pointShIndex ] = ipoints[ y ];
	}
	else if( threadIdx.y == 0 )
	{
		// Load the n-th planes
		sharedPlanes[ planeShIndex ] = iplanes[ x ];
	}
	
	// Wait all reading thread.
	__syncthreads();

	// Compute the multiplication between the point and the plane.
	// P = <N.Pt> + D
	float p =	sharedPlanes[ planeShIndex ].x * sharedPoints[ pointShIndex ].x + 
				sharedPlanes[ planeShIndex ].y * sharedPoints[ pointShIndex ].y + 
				sharedPlanes[ planeShIndex ].z * sharedPoints[ pointShIndex ].z + 
				sharedPlanes[ planeShIndex ].w;

	int outIndex = planeCount * y + x;

	//BACK
	if( p >= 0 )
	{
		out[ outIndex ] = -1;
	}
	else //FRONT
	{
		out[ outIndex ] = 1;
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
ClassifyPyramidalFrustumBoxes( const float3* frustumCorners, const float3* boxPoints, const char* planePointClassification, int planeCount, int pointCount, int* out )
{
	int threadX = blockIdx.x * blockDim.x + threadIdx.x;
	int threadY = blockIdx.y * blockDim.y + threadIdx.y;

	int frustumCount = planeCount / 6;
	int boxCount	 = pointCount / 8;

	if( threadX >= frustumCount || threadY >= boxCount )
		return;

	// TODO :	load frustum and box data to shared memory if the memory size is appropriate
	//			or use device memory with a constant acces (to take advantage of the cache memory)

	//--------------------
	// Step 1 : Sum each 
	// column

	int sums[ 6 ];

	// For each frustum plane.
	for( int i = 0; i < 6; ++i )
	{
		sums[ i ] = 0;

		int planeIndex = threadX * 6 + i;

		// For each point.
		for( int j = 0; j < 8; ++j )
		{
			int pointIndex = threadY * 8 + j;

			int index = pointIndex * planeCount + planeIndex;

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
		out[ outIndex ] = 0; // GCU_INSIDE
		return;
	}
	else
	{
		int arrayCountMinusEight = CountArrayElementValue( sums, 6, -8 );

		if( arrayCountMinusEight > 0 )
		{
			// The box is outside to one or several planes.
			out[ outIndex ] = 1; // GCU_OUTSIDE
		}
		else
		{
			out[ outIndex ] = GCUL_UNDEFINED;
		}
	}
}

__global__ void
InverseClassifyPyramidalFrustumBoxes( const float3* frustumCorners, const float3* boxPoints, int planeCount, int pointCount, int* out )
{
			
	int threadX = blockIdx.x * blockDim.x + threadIdx.x;
	int threadY = blockIdx.y * blockDim.y + threadIdx.y;

	int frustumIndex = threadX;
	int boxIndex	 = threadY;

	int frustumCount = planeCount / 6;
	int boxCount	 = pointCount / 8;

	if( threadX >= frustumCount || threadY >= boxCount )
		return;

	int outIndex = threadX + threadY * frustumCount;

	if( out[ outIndex ] == GCUL_UNDEFINED )
	{

			// Get the upper and lower point of the box.
			float3 upperBoxPoint = UpperPoint( &boxPoints[ boxIndex ] );
			float3 lowerBoxPoint = LowerPoint( &boxPoints[ boxIndex ] );

			bool spanning = false; // by default.

			// For each corner of the frustum.
			for( int p = 0; p < 8; ++p )
			{
				int frustumCornerIndex = frustumIndex * 8 + p;

				float3 currentCorner = frustumCorners[ frustumCornerIndex ];

				// If a frustum corner is outside the box.
				if( ( lowerBoxPoint.x > currentCorner.x ) || ( currentCorner.x > upperBoxPoint.x )
        			||
        			( lowerBoxPoint.y > currentCorner.y ) || ( currentCorner.y > upperBoxPoint.y )
        			||
        			( lowerBoxPoint.z > currentCorner.z ) || ( currentCorner.x > upperBoxPoint.z ) ) 
				{
        			// The frustum intersects the box.
					out[ outIndex ] = GCUL_SPANNING; // GCU_SPANNING
					spanning = true;
        		} 
			}

			if( !spanning )
			{
				// default case
				out[ outIndex ] = GCUL_ENCLOSING; // GCU_ENCOSING
			}
	}
}

/**
*/
__global__ void
ClassifyPlanesSpheres( const float4* planes, const float4* spheres, int planeCount, int sphereCount, char* out )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Exit if out of bounds.
	if( x >= planeCount || y >= sphereCount )
	{
		return;
	}

	// Shared memory used to read input data.
	// Shared memory is only used by same block threads. 
	// * 0 to m   = planes
	// * n+1 to n = spheres 
	// The size of this array must be number_of_threads_per_block * sizeof( float ).
	extern __shared__ float sharedMemory[];

	float4* sharedPlanes  = ( float4* )&sharedMemory[ 0			    ];
	float4* sharedSpheres = ( float4* )&sharedMemory[ blockDim.x * 4 ];

	int planeShIndex  = threadIdx.x;	// offset to the first coordinate of the plane.
	int sphereShIndex = threadIdx.y;	// offset to the first coordinate of the sphere.

	// Only threads which are owned by the first row and the first column load data.
	if( threadIdx.x == 0 && threadIdx.y == 0 )
	{
		// Load the 1st plane and the 1st sphere
		sharedPlanes [ 0 ] = planes [ x ];
		sharedSpheres[ 0 ] = spheres[ y ];
	}
	else if( threadIdx.x == 0 )
	{
		// Load the n-th sphere
		sharedSpheres[ sphereShIndex ] = spheres[ y ];
	}
	else if( threadIdx.y == 0 )
	{
		// Load the n-th planes
		sharedPlanes[ planeShIndex ] = planes[ x ];
	}
	
	// Wait all reading thread.
	__syncthreads();

	// Compute the multiplication between the sphere position and the plane.
	// P = <N.Pt> + D
	float p =	sharedPlanes[ planeShIndex ].x * sharedSpheres[ sphereShIndex ].x + 
				sharedPlanes[ planeShIndex ].y * sharedSpheres[ sphereShIndex ].y + 
				sharedPlanes[ planeShIndex ].z * sharedSpheres[ sphereShIndex ].z + 
				sharedPlanes[ planeShIndex ].w;

	int outIndex = planeCount * y + x;

	if( p > 0 )
	{
		// Center back.
		if( abs( p ) >= sharedSpheres[ sphereShIndex ].w )
		{
			out[ outIndex ] = -1; // back
		}
		else
		{
			out[ outIndex ] = 0; // across
		}
	}
	else // p <= 0
	{
		// Center front.
		if( abs( p ) >= sharedSpheres[ sphereShIndex ].w )
		{
			out[ outIndex ] = 1; // front
		}
		else
		{
			out[ outIndex ] = 0; // across
		} 
	}

	//if( p >= sharedSpheres[ sphereShIndex ].w ) // distance >= radius
	//{
	//	out[ outIndex ] = 1; // front
	//}
	//else if( p <= -sharedSpheres[ sphereShIndex ].w )
	//{
	//	out[ outIndex ] = -1; // back
	//}
	//else
	//{
	//	out[ outIndex ] = 0; // across
	//}
}

/**
*/
__global__ void
ClassifyPyramidalFrustumSpheres( const char6* planeSphereClassification, int frustumCount, int sphereCount, int* out )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= frustumCount || y >= sphereCount )
		return;

	extern __shared__ char6 subClassification[];

	int classShIndex = threadIdx.y * blockDim.x + threadIdx.x;

	// Each thread load six values from the classification to the shared memory.
	subClassification[ classShIndex ] = planeSphereClassification[ frustumCount * y + x ];

	int outIndex = frustumCount * y + x;

	/*printf( "%d\n", (int)subClassification[ classShIndex ].a );
	printf( "%d\n", (int)subClassification[ classShIndex ].b );
	printf( "%d\n", (int)subClassification[ classShIndex ].c );
	printf( "%d\n", (int)subClassification[ classShIndex ].d );
	printf( "%d\n", (int)subClassification[ classShIndex ].e );
	printf( "%d\n", (int)subClassification[ classShIndex ].f );*/

	if( 
		subClassification[ classShIndex ].a == -1 ||
		subClassification[ classShIndex ].b == -1 ||
		subClassification[ classShIndex ].c == -1 ||
		subClassification[ classShIndex ].d == -1 ||
		subClassification[ classShIndex ].e == -1 ||
		subClassification[ classShIndex ].f == -1
		)
	{
		out[ outIndex ] = GCUL_OUTSIDE;
	}
	else
	{
		int count = 0;

		if( subClassification[ classShIndex ].a == 1 ) { ++count; } 
		if( subClassification[ classShIndex ].b == 1 ) { ++count; } 
		if( subClassification[ classShIndex ].c == 1 ) { ++count; } 
		if( subClassification[ classShIndex ].d == 1 ) { ++count; } 
		if( subClassification[ classShIndex ].e == 1 ) { ++count; } 
		if( subClassification[ classShIndex ].f == 1 ) { ++count; } 

		if( count == 0 )
		{
			out[ outIndex ] = GCUL_INSIDE;
		}
		else if( count == 6 )
		{
			out[ outIndex ] = GCUL_INSIDE;
		}
		else
		{
			out[ outIndex ] = GCUL_SPANNING;
		}
	}
}

__global__ void
ClassifySphericalFrustumSpheres( const float4* sphericalFrustums, const float4* spheres, int frustumCount, int sphereCount, int* out )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Exit if out of bounds.
	if( x >= frustumCount || y >= sphereCount )
	{
		return;
	}

	// Shared memory used to read input data.
	// Shared memory is only used by same block threads. 
	// * 0 to m   = planes
	// * n+1 to n = spheres 
	// The size of this array must be number_of_threads_per_block * sizeof( float ).
	extern __shared__ float sharedMemory[];

	float4* sharedFrustums = ( float4* )&sharedMemory[ 0			  ];
	float4* sharedSpheres  = ( float4* )&sharedMemory[ blockDim.x * 4 ];

	int frustumShIndex  = threadIdx.x;	// offset to the first coordinate of the frustum.
	int sphereShIndex	= threadIdx.y;	// offset to the first coordinate of the sphere.

	// Only threads which are owned by the first row and the first column load data.
	if( threadIdx.x == 0 && threadIdx.y == 0 )
	{
		// Load the 1st plane and the 1st sphere.
		sharedFrustums	[ 0 ] = sphericalFrustums	[ x ];
		sharedSpheres	[ 0 ] = spheres				[ y ];
	}
	else if( threadIdx.x == 0 )
	{
		// Load the n-th sphere.
		sharedSpheres[ sphereShIndex ] = spheres[ y ];
	}
	else if( threadIdx.y == 0 )
	{
		// Load the n-th frustum.
		sharedFrustums[ frustumShIndex ] = sphericalFrustums[ x ];
	}
	
	// Wait all reading threads.
	__syncthreads();

	float3 vec = 
	{
		sharedSpheres[ sphereShIndex ].x - sharedFrustums[ frustumShIndex ].x,
		sharedSpheres[ sphereShIndex ].y - sharedFrustums[ frustumShIndex ].y,
		sharedSpheres[ sphereShIndex ].z - sharedFrustums[ frustumShIndex ].z
	};

	float distance = sqrt( vec.x * vec.x + vec.y * vec.y + vec.z * vec.z );

	int outIndex = frustumCount * y + x;

	if( distance > sharedSpheres[ sphereShIndex ].w + sharedFrustums[ frustumShIndex ].w )
	{
		out[ outIndex ] = GCUL_OUTSIDE;
	}
	else if( distance + sharedSpheres[ sphereShIndex ].w < sharedFrustums[ frustumShIndex ].w )
	{
		out[ outIndex ] = GCUL_INSIDE;
	}
	else if( distance + sharedFrustums[ frustumShIndex ].w < sharedSpheres[ sphereShIndex ].w )
	{
		out[ outIndex ] = GCUL_ENCLOSING;
	}
	else
	{
		out[ outIndex ] = GCUL_SPANNING;
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

__device__ float3
UpperPoint( const float3* box )
{
	float maxX, maxY, maxZ;

	maxX = maxY = maxZ = -1.175494351e-38; // min float (4bytes)

	for( int i = 0; i < 8; ++i )
	{
		if( box[ i ].x > maxX ) { maxX = box[ i ].x; }
		if( box[ i ].y > maxY ) { maxY = box[ i ].y; }
		if( box[ i ].z > maxZ ) { maxZ = box[ i ].z; }
	}

	float3 result = { maxX, maxY, maxZ };
	return result;
}

__device__ float3
LowerPoint( const float3* box )
{
	float minX, minY, minZ;

	minX = minY = minZ = 3.402823466e+38; // max float (4bytes)

	for( int i = 0; i < 8; ++i )
	{
		if( box[ i ].x < minX ) { minX = box[ i ].x; }
		if( box[ i ].y < minY ) { minY = box[ i ].y; }
		if( box[ i ].z < minZ ) { minZ = box[ i ].z; }
	}

	float3 result = { minX, minY, minZ };
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