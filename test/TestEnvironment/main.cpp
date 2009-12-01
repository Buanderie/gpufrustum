////////////////////////////////////////////////////////////
// Headers
////////////////////////////////////////////////////////////
#include <vector>
#include <SFML/Window.hpp>
#include <iostream>

#include <gpuCuller.h>

#include "rendering.h"
#include "glCamera.h"
#include "glPoint.h"
#include "glSphere.h"
#include "glMatrix4f.h"
#include "glPyramidalFrustum.h"
#include "utils.h"

using namespace std;

///////////////// Variables Globales ///////////////////////
glCamera* cam;
vector<glPyramidalFrustum> frustumList;
vector<glAABB> aabbList;
vector<glSphere> sphereList;
vector<glSphere> sphericalFrustumList;
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
typedef struct aabb{
	float min_x, min_y, min_z;
	float max_x, max_y, max_z;
} aabb_t;

typedef struct hnode
{
	unsigned int splitLevel;
	unsigned int primStart;
	unsigned int primStop;
	unsigned int ID;
	unsigned int childrenStart;
	unsigned int childrenStop;
	bool visible;
	aabb_t bbox;
} hnode_t;

//Frustum plane
typedef struct plane
{
	float a;
	float b;
	float c;
	float d;
} plane_t;
//

//Pyramidal Frustum
typedef struct pyrfrustum
{
	plane_t planes[6];
} pyrfrustum_t;
//
/////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    //Create the main window
	sf::Window App(sf::VideoMode::GetDesktopMode(), string("gpuCuller Demo"));

    //Create a clock for measuring time elapsed
    sf::Clock Clock;
	sf::Clock FrameRateCpt;

    //Set color and depth clear value
    glClearDepth(1.f);
    glClearColor(0.0f, 0.0f, 0.0f, 0.5f);				// Black Background
	glClearDepth(1.0f);									// Depth Buffer Setup
	glEnable(GL_DEPTH_TEST);							// Enables Depth Testing
	glDepthFunc(GL_LEQUAL);								// The Type Of Depth Testing To Do
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);	// Really Nice Perspective Calculations

    //Setup a perspective projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(90.f, 1.f, 1.f, 100.f);

	//Mise en place des parametres camera
	cam = new glCamera();
	cam->m_Position.x=0;
	cam->m_Position.y = 0;
	cam->m_Position.z = 0;
	cam->m_MaxForwardVelocity = 5.0f;
	cam->m_MaxPitchRate = 5.0f;
	cam->m_MaxHeadingRate = 5.0f;
	cam->m_PitchDegrees = 0.0f;
	cam->m_HeadingDegrees = 0.0f;

	srand( time( NULL ) );

	int nAABB = 2048;
	int nFrustum = 5;

	generateRandomAABBs(	nAABB,
							1.0f, 7.0f,
							1.0f, 7.0f,
							1.0f, 7.0f,
							-90, 90,
							0, 10,
							-90, 90,
							aabbList
							);

	generateRandomPyrFrustums(	nFrustum,
								45, 90,
								1, 2,
								5, 50,
								3.0f/4.0f, 3.0f/4.0f,
								-50, 50,
								0, 0,
								0, 50,
								0, 0,
								-180, 180,
								0, 0,
								frustumList ); 
	
	//Initialize gpuCuller
	gculInitialize( argc, argv );
	gculSetBVHDepth( 3 );
	gculSetUniverseAABB( -100, -100, -100, 100, 100, 100 );
	//Load AABB
	aabb_t* pol = new aabb_t[nAABB];
	for( int i = 0; i < aabbList.size(); ++i )
	{
		pol[ i ].min_x = aabbList[i].m_MinPos.x;
		pol[ i ].min_y = aabbList[i].m_MinPos.z;
		pol[ i ].min_z = aabbList[i].m_MinPos.y;
		pol[ i ].max_x = aabbList[i].m_MaxPos.x;
		pol[ i ].max_y = aabbList[i].m_MaxPos.z;
		pol[ i ].max_z = aabbList[i].m_MaxPos.y;
	}
	gculLoadAABB( nAABB, (void*)pol );
	gculBuildLBVH();

	pyrfrustum_t* frustumPlanesData = new pyrfrustum_t[nFrustum];
	float* frustumCornersData = new float[nFrustum*24];
	getFrustumPlanesArray( frustumList, (float*)frustumPlanesData );
	gculLoadFrustumPlanes( nFrustum, (void*)frustumPlanesData );
	getFrustumCornersArray( frustumList, frustumCornersData );
	gculLoadFrustumCorners( nFrustum, (void*)frustumCornersData );

	//Get the hierarchy information
	hnode_t* bak = new hnode_t[ gculGetHierarchySize() ];
	gculGetHierarchyInformation( (void*)bak );
	for( int i = 0; i < gculGetHierarchySize(); ++i )
	{
		cout	<< "lvl=" << bak[i].splitLevel
				<< " min=(" << bak[i].bbox.min_x << "," << bak[i].bbox.min_y << "," << bak[i].bbox.min_z << ")"
				<< " max=(" << bak[i].bbox.max_x << "," << bak[i].bbox.max_y << "," << bak[i].bbox.max_z << ")"
				<< endl;
	}

    // Start game loop
    while (App.IsOpened())
    {
		//Frustum Culling
		Clock.Reset();

        // Process events
        sf::Event Event;
        while (App.GetEvent(Event))
        {
            // Close window : exit
            if (Event.Type == sf::Event::Closed)
                App.Close();

            // Escape key : exit
            if ((Event.Type == sf::Event::KeyPressed) && (Event.Key.Code == sf::Key::Escape))
                App.Close();

			if( Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Up )
				cam->m_ForwardVelocity = 10.0f*App.GetFrameTime();
			if( Event.Type == sf::Event::KeyReleased && Event.Key.Code == sf::Key::Up )
				cam->m_ForwardVelocity = 0;

			if( Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::Down )
				cam->m_ForwardVelocity = -10.0f*App.GetFrameTime();
			if( Event.Type == sf::Event::KeyReleased && Event.Key.Code == sf::Key::Down )
				cam->m_ForwardVelocity = 0;

			if( Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::C )
			{

			}
			
			if( Event.Type == sf::Event::MouseMoved )
			{
				cam->ChangeHeading(0.2f*(float)(Event.MouseMove.X-((int)App.GetWidth()/2)));
				cam->ChangePitch(0.2f*(float)(Event.MouseMove.Y-((int)App.GetHeight()/2)));
				App.SetCursorPosition((int)App.GetWidth()/2, (int)App.GetHeight()/2); 
			}

            // Resize event : adjust viewport
            if (Event.Type == sf::Event::Resized)
                glViewport(0, 0, Event.Size.Width, Event.Size.Height);
        }
		//

        // Set the active window before using OpenGL commands
        // It's useless here because active window is always the same,
        // but don't forget it if you use multiple windows or controls
        App.SetActive();

        // Clear color and depth buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Apply some transformations
        glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		cam->SetPrespective();

		drawFloorGrid( );

		//God help us
	//
		unsigned int * tarace = new unsigned int[nAABB];
		gculProcessCulling();
		gculGetResults(tarace);
		//for( int i = 0 ;i < 512; ++i )
		//	cout << tarace[i] << endl;
	//

		//display aabbs
		for( int i = 0; i < aabbList.size(); ++i )
		{
			if( tarace[i] == 777 )
				aabbList[i].isInsideFrustum = true;
			else
				aabbList[i].isInsideFrustum = false;
			aabbList[i].draw();
		}

		/*for(int i = 0; i < gculGetHierarchySize(); ++i )
		{
			glAABB chatte(	glVector4f(bak[i].bbox.min_x,bak[i].bbox.min_z,bak[i].bbox.min_y,0),
							glVector4f(bak[i].bbox.max_x,bak[i].bbox.max_z,bak[i].bbox.max_y,0)
							);
			if( bak[i].visible )
				chatte.isInsideFrustum = true;
			if( bak[i].splitLevel == 3)
				chatte.draw();
		}*/

		for( int i = 0; i < frustumList.size(); ++i )
			frustumList[i].draw();
		//

        // Finally, display rendered frame on screen
        App.Display();

		FrameRateCpt.Reset();
    }
	
	delete cam;

    return EXIT_SUCCESS;

 }