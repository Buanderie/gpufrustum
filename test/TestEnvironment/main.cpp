////////////////////////////////////////////////////////////
// Headers
////////////////////////////////////////////////////////////
#include <vector>
#include <SFML/Window.hpp>

#include "rendering.h"
#include "glCamera.h"
#include "glPoint.h"
#include "glMatrix4f.h"
#include "glPyramidalFrustum.h"
#include "utils.h"

using namespace std;

///////////////// Variables Globales ///////////////////////
glCamera* cam;
vector<glPyramidalFrustum> frustumList;
////////////////////////////////////////////////////////////

int main()
{
    //Create the main window
	sf::Window App(sf::VideoMode::GetDesktopMode(), "gpuCuller Demo");

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
	
	generateRandomPyrFrustums(	10000,
								45, 90,
								1, 2,
								5, 10,
								3.0f/4.0f, 3.0f/4.0f,
								-1000, 1000,
								0, 0,
								-1000, 1000,
								0, 0,
								-180, 180,
								0, 0,
								frustumList );

    // Start game loop
    while (App.IsOpened())
    {
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

		//Display an AABB
		glAABB a(glVector4f(0,0,0,0), glVector4f(1,1,1,1));
		a.draw();

        // Finally, display rendered frame on screen
        App.Display();

		FrameRateCpt.Reset();
    }

    return EXIT_SUCCESS;
}

