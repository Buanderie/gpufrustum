#ifndef __FRUSTUM_CULLING_KERNEL_H__
#define __FRUSTUM_CULLING_KERNEL_H__

#include "structs.h"

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
classifiyPlanePoint( plane_t* iplanes, point3d_t* ipoints, int nPoint, int nPlane, int* out )
{
	//On recupere l'indice du resultat de classification dans la matrice resultat
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	//Si le thread travaille en dehors des dimensions de la matrice, on ne fait rien
	if( x >= nPoint || y >= nPlane )
		return;

	//Transfert Device Memory -> Shared Memory ? A voir...

	//
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

#endif