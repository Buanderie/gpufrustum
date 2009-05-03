#include "cfrustum.h"

/*
Extrait le frustum de vue
On va faire ça en 4 etapes : 
1) on recupere les matricezs en cours
2) on les combine
3) on calcule le frustum
4) on normalise
et Pourquoi j ai mis un paramatre ? be tout simplement, si vous utilisez SameMatrix(), 
les matrices de Modelisation et de projection sont extraites, mais sinon, vous devez les
extraire vous meme ...
*/
void CFrustum :: Extrait (bool extraire_mat)
{
   if (extraire_mat == true)
   {
	glGetFloatv( GL_MODELVIEW_MATRIX, modl );   /* Récupere la matrice de modelisation */
	glGetFloatv( GL_PROJECTION_MATRIX, proj );  /* Récupere la matrice de projection */
   }

   float   t;

   /* On combine les 2, en multipliant la matrice de projection par celle de modelisation*/
   clip[ 0] = modl[ 0] * proj[ 0] + modl[ 1] * proj[ 4] + modl[ 2] * proj[ 8] + modl[ 3] * proj[12];
   clip[ 1] = modl[ 0] * proj[ 1] + modl[ 1] * proj[ 5] + modl[ 2] * proj[ 9] + modl[ 3] * proj[13];
   clip[ 2] = modl[ 0] * proj[ 2] + modl[ 1] * proj[ 6] + modl[ 2] * proj[10] + modl[ 3] * proj[14];
   clip[ 3] = modl[ 0] * proj[ 3] + modl[ 1] * proj[ 7] + modl[ 2] * proj[11] + modl[ 3] * proj[15];

   clip[ 4] = modl[ 4] * proj[ 0] + modl[ 5] * proj[ 4] + modl[ 6] * proj[ 8] + modl[ 7] * proj[12];
   clip[ 5] = modl[ 4] * proj[ 1] + modl[ 5] * proj[ 5] + modl[ 6] * proj[ 9] + modl[ 7] * proj[13];
   clip[ 6] = modl[ 4] * proj[ 2] + modl[ 5] * proj[ 6] + modl[ 6] * proj[10] + modl[ 7] * proj[14];
   clip[ 7] = modl[ 4] * proj[ 3] + modl[ 5] * proj[ 7] + modl[ 6] * proj[11] + modl[ 7] * proj[15];

   clip[ 8] = modl[ 8] * proj[ 0] + modl[ 9] * proj[ 4] + modl[10] * proj[ 8] + modl[11] * proj[12];
   clip[ 9] = modl[ 8] * proj[ 1] + modl[ 9] * proj[ 5] + modl[10] * proj[ 9] + modl[11] * proj[13];
   clip[10] = modl[ 8] * proj[ 2] + modl[ 9] * proj[ 6] + modl[10] * proj[10] + modl[11] * proj[14];
   clip[11] = modl[ 8] * proj[ 3] + modl[ 9] * proj[ 7] + modl[10] * proj[11] + modl[11] * proj[15];

   clip[12] = modl[12] * proj[ 0] + modl[13] * proj[ 4] + modl[14] * proj[ 8] + modl[15] * proj[12];
   clip[13] = modl[12] * proj[ 1] + modl[13] * proj[ 5] + modl[14] * proj[ 9] + modl[15] * proj[13];
   clip[14] = modl[12] * proj[ 2] + modl[13] * proj[ 6] + modl[14] * proj[10] + modl[15] * proj[14];
   clip[15] = modl[12] * proj[ 3] + modl[13] * proj[ 7] + modl[14] * proj[11] + modl[15] * proj[15];

   /* Extrait le plan de DROITE */
   frustum[0][0] = clip[ 3] - clip[ 0];
   frustum[0][1] = clip[ 7] - clip[ 4];
   frustum[0][2] = clip[11] - clip[ 8];
   frustum[0][3] = clip[15] - clip[12];

   /* Calcul des normales */
   t = sqrt( frustum[0][0] * frustum[0][0] + frustum[0][1] * frustum[0][1] + frustum[0][2] * frustum[0][2] );
   frustum[0][0] /= t;
   frustum[0][1] /= t;
   frustum[0][2] /= t;
   frustum[0][3] /= t;

   /* Extrait le plan de GAUCHE */
   frustum[1][0] = clip[ 3] + clip[ 0];
   frustum[1][1] = clip[ 7] + clip[ 4];
   frustum[1][2] = clip[11] + clip[ 8];
   frustum[1][3] = clip[15] + clip[12];

   /* Calcul des normales */
   t = sqrt( frustum[1][0] * frustum[1][0] + frustum[1][1] * frustum[1][1] + frustum[1][2] * frustum[1][2] );
   frustum[1][0] /= t;
   frustum[1][1] /= t;
   frustum[1][2] /= t;
   frustum[1][3] /= t;

   /* Extrait le plan du BAS */
   frustum[2][0] = clip[ 3] + clip[ 1];
   frustum[2][1] = clip[ 7] + clip[ 5];
   frustum[2][2] = clip[11] + clip[ 9];
   frustum[2][3] = clip[15] + clip[13];

   /* Calcul des normales */
   t = sqrt( frustum[2][0] * frustum[2][0] + frustum[2][1] * frustum[2][1] + frustum[2][2] * frustum[2][2] );
   frustum[2][0] /= t;
   frustum[2][1] /= t;
   frustum[2][2] /= t;
   frustum[2][3] /= t;

   /* Extrait le plan du HAUT */
   frustum[3][0] = clip[ 3] - clip[ 1];
   frustum[3][1] = clip[ 7] - clip[ 5];
   frustum[3][2] = clip[11] - clip[ 9];
   frustum[3][3] = clip[15] - clip[13];

   /* Calcul des normales */
   t = sqrt( frustum[3][0] * frustum[3][0] + frustum[3][1] * frustum[3][1] + frustum[3][2] * frustum[3][2] );
   frustum[3][0] /= t;
   frustum[3][1] /= t;
   frustum[3][2] /= t;
   frustum[3][3] /= t;

   /* Extrait le plan ELOIGNE */
   frustum[4][0] = clip[ 3] - clip[ 2];
   frustum[4][1] = clip[ 7] - clip[ 6];
   frustum[4][2] = clip[11] - clip[10];
   frustum[4][3] = clip[15] - clip[14];

   /* Calcul des normales */
   t = sqrt( frustum[4][0] * frustum[4][0] + frustum[4][1] * frustum[4][1] + frustum[4][2] * frustum[4][2] );
   frustum[4][0] /= t;
   frustum[4][1] /= t;
   frustum[4][2] /= t;
   frustum[4][3] /= t;

   /* Extrait le plan PROCHE */
   frustum[5][0] = clip[ 3] + clip[ 2];
   frustum[5][1] = clip[ 7] + clip[ 6];
   frustum[5][2] = clip[11] + clip[10];
   frustum[5][3] = clip[15] + clip[14];

   /* Calcul des normales */
   t = sqrt( frustum[5][0] * frustum[5][0] + frustum[5][1] * frustum[5][1] + frustum[5][2] * frustum[5][2] );
   frustum[5][0] /= t;
   frustum[5][1] /= t;
   frustum[5][2] /= t;
   frustum[5][3] /= t;
	
}

/*
Calcule  si un point est sur un des 6 plans du frustum
avec la formule distance = A * X + B * Y + C * Z + D
où A, B, C, et D sont les 4 nombres qui definissent le plan et X, Y, et Z
sont les coordonnees du point
*/
int CFrustum :: PointInFrustum( float x, float y, float z )
{
   int p;

   for( p = 0; p < 6; p++ )	// pour tous les plans
	   // si le point est en dehors d un des plans
      if( frustum[p][0] * x + frustum[p][1] * y + frustum[p][2] * z + frustum[p][3] <= 0 )
         return dehors;	// on se barre
   return dedans;	// sinon c ok
}


/*
Calcule si une sphere de centre x,y,z et de rayon radius est dans le frustum de vue
et renvoie la variable adequate
*/
int CFrustum :: SphereInFrustum( float x, float y, float z, float radius )
{
   int p;
   int c = 0;	/* le nombre de fois qu on est dans le frustum : 
   si on y est 6 fois, on est completement
   si on y est 0 fois, on est en dehors
   sinon, on est partiellement dedans */
   float d;

   for( p = 0; p < 6; p++ )	// pour tous les plans
   {
      d = frustum[p][0] * x + frustum[p][1] * y + frustum[p][2] * z + frustum[p][3];
      if( d <= -radius )	// si on est en dehors d un des plans
         return dehors;		// on est en dehors
      if( d > radius )		// sinon on incremente le nombre de fois qu on est dans le plan
         c++;
   }
   return (c == 6) ? dedans : part_dedans; // voir plus haut pour comprendre ce test
}

/*
Calcul pour le cube
Sans commentaires, c juste un test avec 8 points ...
*/
int CFrustum :: CubeInFrustum( float x, float y, float z, float size )
{
	int p;

	for( p = 0; p < 6; p++ )
	{
		if( frustum[p][0] * (x - size) + frustum[p][1] * (y - size) + frustum[p][2] * (z - size) + frustum[p][3] > 0 )
			continue;
		if( frustum[p][0] * (x + size) + frustum[p][1] * (y - size) + frustum[p][2] * (z - size) + frustum[p][3] > 0 )
			continue;
		if( frustum[p][0] * (x - size) + frustum[p][1] * (y + size) + frustum[p][2] * (z - size) + frustum[p][3] > 0 )
			continue;
		if( frustum[p][0] * (x + size) + frustum[p][1] * (y + size) + frustum[p][2] * (z - size) + frustum[p][3] > 0 )
			continue;
		if( frustum[p][0] * (x - size) + frustum[p][1] * (y - size) + frustum[p][2] * (z + size) + frustum[p][3] > 0 )
			continue;
		if( frustum[p][0] * (x + size) + frustum[p][1] * (y - size) + frustum[p][2] * (z + size) + frustum[p][3] > 0 )
			continue;
		if( frustum[p][0] * (x - size) + frustum[p][1] * (y + size) + frustum[p][2] * (z + size) + frustum[p][3] > 0 )
			continue;
		if( frustum[p][0] * (x + size) + frustum[p][1] * (y + size) + frustum[p][2] * (z + size) + frustum[p][3] > 0 )
			continue;
		return dehors;
	}
	return dedans;
}

/*
Teste si la matrice ModelView a changé ...
renvoie 0 si oui, et 1 si rien n' a bougé ...
*/
bool CFrustum :: SameMatrix( )
{
	static int i = 0;
	glGetFloatv( GL_MODELVIEW_MATRIX, modl );   /* Récupere la matrice de modelisation */
	for (i=0; i < 16; i++)	// Pour les 2 matrices
	{
	if ( modl[i] == old_ModelView[i] ) // si cet element de la matrice est egal a celui de la derniere sauvegarde, 
		continue;	// on passe a l element suivant...
	else
	{
		for (i =0; i > 16; i++)
			old_ModelView[i] = modl[i];	
		change = true;
		return 0;	// sinon, on retourne 0, parce que la matrice a été modifié ...
	}	
	}
	
	glGetFloatv( GL_PROJECTION_MATRIX, proj );  /* Récupere la matrice de projection */
	for (i=0; i < 16; i++)	// Pour les 2 matrices
	{
	if ( proj[i] == old_Projection[i] ) // si cet element de la matrice est egal a celui de la derniere sauvegarde, 
		continue;	// on passe a l element suivant...
	else
	{
		for (i =0; i > 16; i++)
			old_Projection[i] = proj[i];
		change = true;
		return 0;	// sinon, on retourne 0, parce que la matrice a été modifié ...
	}
	}

change = false;
return 1;	// et si les 2 matrices sont egales, alors c est que la matrice na pas changé ... 
//donc il ne reste plus qu' a tester si les elements sont visible ... et la c a vous ....
}

/*
A quoi sert cette fonction, que je n ai meme pas pris la peine de documenter ? 
Tout simplement, cela permet de mettre change a true, tout cela pour eviter de passer par le test des 2 matrices au dessus..
comme ca, quand vous savez que lune des matrices va changer, par ex quand on bouge la souris, 
vous faites un SetChanged(true); et vous ne faites pas le test de SameMatrix ... 
comme cela, au lieu de tester 32 float, vous testez juste un booleen ... Elle est pas belle la vie ?
*/
void CFrustum :: SetChanged (bool b_change)
{
	//change = b_changed;
}