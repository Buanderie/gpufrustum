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
GCUL_Classification* gculResultsPyrBoxes;
GCUL_Classification* gculResultsPyrSpheres;
GCUL_Classification* gculResultsSphericalFrustumsSpheres;
GCUL_Classification* gculResultsSphericalFrustumsBoxes;
////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    //Create the main window
	sf::Window App(sf::VideoMode(800,600,32), string("gpuCuller Demo"));

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
	
	int nFrustum;
	int nAABB;
	int nSpheres;
	int nSphericalFrustums;
	if( argc == 5 )
	{
		nFrustum = atoi(argv[1]);
		nAABB = atoi(argv[3]);
		nSpheres = atoi(argv[4]);
		nSphericalFrustums = atoi(argv[2]);
	}
	else
	{
		nFrustum = 25;
		nSpheres = 100;
		nAABB = 100;
		nSphericalFrustums = 25;
	}

	srand( time( NULL ) );
	//srand( 10 );


	//glPyramidalFrustum frus(90.0f, 1.0f, 100.0f, 0.75f, glVector4f(0,0,0,0), 0, 0, 0);
	//glPyramidalFrustum frus2(90.0f, 1.0f, 50.0f, 0.75f, glVector4f(0,0,0,0), 0, 180.0f, 0);

	//frustumList.push_back(frus);
	//frustumList.push_back(frus2);

	generateRandomAABBs(	nAABB,
							1.0f, 7.0f,
							1.0f, 7.0f,
							1.0f, 7.0f,
							-100, 0,
							0, 0,
							-100, 100,
							aabbList
							);

	generateRandomPyrFrustums(	nFrustum,
								45, 90,
								1, 2,
								5, 50,
								3.0f/4.0f, 3.0f/4.0f,
								-100, 100,
								0, 0,
								0, 100,
								0, 0,
								-180, 180,
								0, 0,
								frustumList );
								
	generateRandomSpheres( nSpheres,
							0.5f, 2.f,
							0, 100,
							0, 0,
							-100, 100,
							sphereList
							);

	generateRandomSpheres( nSphericalFrustums,
							10.f, 10.f,
							-100, 100,
							0, 0,
							-100, 0,
							sphericalFrustumList
							);
						

	//gpuCuller
	float* frustumPlanesData = new float[nFrustum*24];
	float* frustumCornersData = new float[nFrustum*24];
	float* aabbCornersData = new float[nAABB*24];
	float* spheresData = new float[nSpheres*4];
	float* sphericalFrustumData = new float[nSphericalFrustums*4];

	getFrustumPlanesArray( frustumList, frustumPlanesData );
	getFrustumCornersArray( frustumList, frustumCornersData );
	getAABBCornersArray( aabbList, aabbCornersData );
	getSpheresArray( sphereList, spheresData );
	getSpheresArray( sphericalFrustumList, sphericalFrustumData );

	//Initialize gpuCuller
	gculInitialize( argc, argv );

	//gculEnable(GCUL_OCCLUSION_CULLING);

	//Initialize data
	gculPyramidalFrustumCornersPointer( nFrustum, GCUL_FLOAT,frustumCornersData );
	gculPyramidalFrustumPlanesPointer( nFrustum, GCUL_FLOAT, frustumPlanesData );
	gculSphericalFrustumPointer( nSphericalFrustums, GCUL_FLOAT, sphericalFrustumData );
	gculBoxesPointer( nAABB, GCUL_FLOAT, aabbCornersData );
	gculSpheresPointer( nSpheres, GCUL_FLOAT, spheresData );

	//Prepare output
	gculResultsPyrBoxes	= new GCUL_Classification[nFrustum * nAABB	 ];
	gculResultsPyrSpheres	= new GCUL_Classification[nFrustum * nSpheres];
	gculResultsSphericalFrustumsSpheres = new GCUL_Classification[nSphericalFrustums * nSpheres];
	gculResultsSphericalFrustumsBoxes = new GCUL_Classification[nSphericalFrustums * nAABB];
	//


    // Start game loop
    while (App.IsOpened())
    {
		//Frustum Culling
		Clock.Reset();

		/*gculEnableArray( GCUL_Array::GCUL_PYRAMIDALFRUSTUMCORNERS_ARRAY );
		gculEnableArray( GCUL_Array::GCUL_PYRAMIDALFRUSTUMPLANES_ARRAY );
		gculEnableArray( GCUL_Array::GCUL_BBOXES_ARRAY );*/

		//Pyr / AABB
		gculEnableArray( GCUL_BBOXES_ARRAY			 );
		gculEnableArray( GCUL_Array::GCUL_PYRAMIDALFRUSTUMCORNERS_ARRAY );
		gculEnableArray( GCUL_Array::GCUL_PYRAMIDALFRUSTUMPLANES_ARRAY );
		gculProcessFrustumCulling( gculResultsPyrBoxes );
		gculDisableArray( GCUL_BBOXES_ARRAY			 );
		gculDisableArray( GCUL_Array::GCUL_PYRAMIDALFRUSTUMCORNERS_ARRAY );
		gculDisableArray( GCUL_Array::GCUL_PYRAMIDALFRUSTUMPLANES_ARRAY );
		//

		//Pyr / Sphere
		gculEnableArray( GCUL_Array::GCUL_BSPHERES_ARRAY			 );
		gculEnableArray( GCUL_Array::GCUL_PYRAMIDALFRUSTUMCORNERS_ARRAY );
		gculEnableArray( GCUL_Array::GCUL_PYRAMIDALFRUSTUMPLANES_ARRAY );
		gculProcessFrustumCulling( gculResultsPyrSpheres );
		gculDisableArray( GCUL_BSPHERES_ARRAY			 );
		gculDisableArray( GCUL_Array::GCUL_PYRAMIDALFRUSTUMCORNERS_ARRAY );
		gculDisableArray( GCUL_Array::GCUL_PYRAMIDALFRUSTUMPLANES_ARRAY );	
		//

		//Sphere / AABB
		gculEnableArray( GCUL_Array::GCUL_SPHERICALFRUSTUM_ARRAY );
		gculEnableArray( GCUL_Array::GCUL_BBOXES_ARRAY );
		gculProcessFrustumCulling( gculResultsSphericalFrustumsBoxes );
		gculDisableArray( GCUL_Array::GCUL_BBOXES_ARRAY );
		gculDisableArray( GCUL_Array::GCUL_SPHERICALFRUSTUM_ARRAY );
		//

		//Sphere / Sphere
		gculEnableArray( GCUL_Array::GCUL_SPHERICALFRUSTUM_ARRAY );
		gculEnableArray( GCUL_Array::GCUL_BSPHERES_ARRAY );
		gculProcessFrustumCulling( gculResultsSphericalFrustumsSpheres );
		gculDisableArray( GCUL_Array::GCUL_BSPHERES_ARRAY );
		gculDisableArray( GCUL_Array::GCUL_SPHERICALFRUSTUM_ARRAY );		
		//


		/*float t = Clock.GetElapsedTime();
		float cullingpersecond = ( nSphericalFrustums * nAABB )/ t;

		printf("delta_t = %f ms | %fM culling operations per second\n", t * 1000.0f, cullingpersecond / 1000000.0f );
		*/

		//Results Pyr / AABB
		for( int i = 0; i < nAABB; ++i )
		{
			for(int j = 0; j < nFrustum; ++j)
			{
				//printf("Frustum %d Box %d Resut %d\n ", j, i, gculResultsSphericalFrustumsBoxes[nFrustum*i +j]);
				if( gculResultsPyrBoxes[nFrustum*i + j] == GCUL_INSIDE || gculResultsPyrBoxes[nFrustum*i + j] == GCUL_SPANNING )
				{
					aabbList[i].isInsideFrustum = true; //SetInsideFrustum( true );
				}
			}
		}

		//Results Sphere / AABB
		for( int i = 0; i < nAABB; ++i )
		{
			for(int j = 0; j < nSphericalFrustums; ++j)
			{
				//printf("Frustum %d Box %d Resut %d\n ", j, i, gculResultsSphericalFrustumsBoxes[nFrustum*i +j]);
				if( gculResultsSphericalFrustumsBoxes[nSphericalFrustums*i + j] == GCUL_INSIDE || gculResultsSphericalFrustumsBoxes[nSphericalFrustums*i + j] == GCUL_SPANNING )
				{
					aabbList[i].isInsideFrustum = true; //SetInsideFrustum( true );
				}
			}
		}

		//Results Pyr / Sphere
		for( int i = 0; i < nSpheres; ++i )
		{
			for(int j = 0; j < nFrustum; ++j)
			{
				//printf("Frustum %d Box %d Resut %d\n ", j, i, gculResultsSphericalFrustumsBoxes[nFrustum*i +j]);
				if( gculResultsPyrSpheres[nFrustum*i + j] == GCUL_INSIDE || gculResultsPyrSpheres[nFrustum*i + j] == GCUL_SPANNING )
				{
					sphereList[i].SetInsideFrustum( true );
				}
			}
		}
		
		//Results Sphere / Sphere
		for( int i = 0; i < nSpheres; ++i )
		{
			for(int j = 0; j < nSphericalFrustums; ++j)
			{
				//printf("Frustum %d Box %d Resut %d\n ", j, i, gculResultsSphericalFrustumsBoxes[nFrustum*i +j]);
				if( gculResultsSphericalFrustumsSpheres[nSphericalFrustums*i + j] == GCUL_INSIDE || gculResultsSphericalFrustumsSpheres[nSphericalFrustums*i + j] == GCUL_SPANNING )
				{
					sphereList[i].SetInsideFrustum( true );
				}
			}
		}
		


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

		//display the frustums
		for(int i = 0; i < frustumList.size(); ++i )
			frustumList[i].draw();

		for(int i = 0; i < aabbList.size(); ++i )
			aabbList[i].draw();

		for(  int i = 0; i < sphereList.size(); ++i )
		{
			sphereList[ i ].Draw();
		}

		
		for( int i = 0; i < sphericalFrustumList.size(); ++i )
		{
			sphericalFrustumList[ i ].SetFrustum( true );
			sphericalFrustumList[ i ].Draw();
		}

        // Finally, display rendered frame on screen
        App.Display();

		FrameRateCpt.Reset();
    }

	// delete arrays
	delete[] frustumPlanesData;
	delete[] frustumCornersData;
	delete[] aabbCornersData;
	delete[] gculResultsPyrSpheres;
	delete[] gculResultsPyrBoxes;
	
	delete cam;

    return EXIT_SUCCESS;
 }