#ifndef _CFRUSTUM_H_
#define _CFRUSTUM_H_

#include <windows.h>
#include <gl\gl.h>
#include <math.h>

/* les positions possibles d un objet dans le plan : 
dedans, dehors ou a cheval sur les deux */
enum Frustum_Pos
{
	dehors = 0,
	part_dedans,
	dedans
};

class CFrustum
{
public:
	void Extrait(bool extraire_mat);
	bool SameMatrix();
	void SetChanged(bool b_changed);
	
	int PointInFrustum( float x, float y, float z );
	int SphereInFrustum( float x, float y, float z, float radius );
	int CubeInFrustum( float x, float y, float z, float size );
	
		float frustum[6][4];	// 6 plans
private:

	float old_ModelView[16], // ModelView sauvegardée
		old_Projection[16],// Projection sauvegardée
		modl[16], 		// ModelView
		proj[16], 		// Projection
		clip[16];		// le frustum de vue
	bool change;		// est ce que l une des 2 matrices a changée ?
};

#endif
