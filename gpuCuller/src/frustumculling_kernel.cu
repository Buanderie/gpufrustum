#ifndef __FRUSTUM_CULLING_KERNEL_CU__
#define __FRUSTUM_CULLING_KERNEL_CU__

#include <gpuCuller_kernel.h>

void ClassifyPlanesPoints( dim3 gridSize, dim3 blockSize, const void* iplanes, const void* ipoints, int nPlane, int nPoint, int* out )
{
	classifyPlanePoint<<< gridSize, blockSize >>>( ( plane_t* )iplanes, ( point3d_t* )ipoints, nPlane, nPoint, out );
}

/* classifyPlanePoint
                 [X][ ............. ]
				 [Y][ ............. ]
[ Planes/Points ]
[ a,b,c,d       ][                  ]
[ a,b,c,d       ][                  ]
[   ...         ][        out       ]
    ...          [                  ]
[               ][                  ]

Le resultat est une matrice de nPlane lignes x nPoint colonnes... si == 1 alors FRONT, si == 0, alors BACK.
*/
__global__ void
classifyPlanePoint( const plane_t* iplanes, const point3d_t* ipoints, int nPlane, int nPoint, int* out )
{
	//On recupere l'indice du resultat de classification dans la matrice resultat
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	//Si le thread travaille en dehors des dimensions de la matrice, on ne fait rien
	if( x >= nPoint || y >= nPlane )
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

	float p = iplanes[y].a*ipoints[x].x + iplanes[y].b*ipoints[x].y + iplanes[y].c*ipoints[x].z + iplanes[y].d;

	//FRONT
	if( p < 0 )
	{
		out[ nPoint * y + x ] = 1;
		return;
	}
	else
	//BACK
	if( p > 0 )
	{
		out[ nPoint * y + x ] = 0;
		return;
	}

	//Cas ou le point est confondu au plan... Considere comme FRONT
	out[ nPoint * y + x ] = 1;
	return;
}


//Classification Frustum Pyramidal / AABB
/*
Ce test se fait en deux passes:
	-Passe 1: Reduction de la matrice resultat de classifyPlanePoint sur l'axe X. On somme les resultats, de maniere a obtenir le nombre de points de chaque cote des plans
	-Passe 2: Test sur le nombre de points de chaque cote des plans, et obtention du resultat
*/
__global__ void 
intersectPyrFrustumAABB_pass1( )
{
 
}
#endif