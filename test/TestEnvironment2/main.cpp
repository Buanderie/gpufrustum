////////////////////////////////////////////////////////////
// Headers
////////////////////////////////////////////////////////////
//STD
#include <time.h>

//STL
#include <vector>
#include <iostream>

//GLFW & AntTweakBar
#include <AntTweakBar.h>
#define GLFW_DLL
#include <glfw.h>

//gpuCuller
#include <gpuCuller.h>

//internals
#include "rendering.h"
#include "glCamera.h"
#include "glPoint.h"
#include "glMatrix4f.h"
#include "glPyramidalFrustum.h"
#include "utils.h"

using namespace std;

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

typedef enum distrib
{ 
	UNIFORM = 1,
	CLUSTERS,
	MIX
} distrib_t;
/////////////////////////////////////////////////////////////

///////////////// Variables Globales ///////////////////////
glCamera* cam;
vector<glPyramidalFrustum> frustumList;
vector<glAABB> aabbList;
int winWidth, winHeight;

//results
float bvhBuildingTime;
float cullingTime;
float frameTime;
float frameRate;
float cullingOps;
unsigned int * cullingResult;
//

//universe parameters
float universeMinX;
float universeMinY;
float universeMinZ;
float universeMaxX;
float universeMaxY;
float universeMaxZ;
//

//frustum parameters
distrib_t frustumDistribution;
unsigned int nFrustum;
float fMinFOV;
float fMaxFOV;
float fMinNear;
float fMaxNear;
float fMinFar;
float fMaxFar;
float fMinAspectRatio;
float fMaxAspectRatio;
float fMinRotX;
float fMaxRotX;
float fMinRotY;
float fMaxRotY;
float fMinRotZ;
float fMaxRotZ;
//

//aabb parameters
distrib_t aabbDistribution;
unsigned int nAABB;
float aabbMinWidth;
float aabbMaxWidth;
float aabbMinHeight;
float aabbMaxHeight;
float aabbMinDepth;
float aabbMaxDepth;
//

//BVH parameters
bool isCulling;
bool isRendering;
bool dynamicMode;
unsigned int bvhDepth;
//

////////////////////////////////////////////////////////////

// Callback function called by GLFW when window size changes
void GLFWCALL WindowSizeCB(int width, int height)
{
    // Set OpenGL viewport and camera
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(90.f, 1.f, 1.f, 1000.f);
    
    // Send the new window size to AntTweakBar
    TwWindowSize(width, height);

	winWidth = width;
	winHeight = height;
}

//Set default values for main parameters
void setDefaultParameters(char** arg)
{
	//universe
	universeMinX = -300;
	universeMinY = -10;
	universeMinZ = -300;
	universeMaxX = 300;
	universeMaxY = 10;
	universeMaxZ = 300;
	//
	cullingResult = 0;
	//
	bvhDepth = 5;
	dynamicMode = false;
	isCulling = true;
	isRendering = false;
	//
	aabbDistribution = UNIFORM;
	nAABB = 10000;
	aabbMinWidth = 1.0f;
	aabbMaxWidth = 7.0f;
	aabbMinHeight = 1.0f;
	aabbMaxHeight = 7.0f;
	aabbMinDepth = 1.0f;
	aabbMaxDepth = 7.0f;
	//
	frustumDistribution = UNIFORM;
	nFrustum = 100;
	fMinFOV = 45.0f;
	fMaxFOV = 90.0f;
	fMinNear = 1;
	fMaxNear = 2;
	fMinFar = 5;
	fMaxFar = 50;
	fMinAspectRatio = 3.0f/4.0f; 
	fMaxAspectRatio = 3.0f/4.0f;
	fMinRotX = 0;
	fMaxRotX = 0;
	fMinRotY = -180;
	fMaxRotY = 180;
	fMinRotZ = 0;
	fMaxRotZ = 0;
	//
}

//Function which creates AABB distribution
void createAABB( distrib_t aabbDistrib )
{
	aabbList.clear();
	switch( aabbDistrib )
	{
	case UNIFORM:
		generateRandomAABBs(	nAABB,
								aabbMinWidth, aabbMaxWidth,
								aabbMinHeight, aabbMaxHeight,
								aabbMinDepth, aabbMaxDepth,
								universeMinX + 1.0f, universeMaxX - 1.0f,
								universeMinY + 1.0f, universeMaxY - 1.0f,
								universeMinZ + 1.0f, universeMaxZ - 1.0f,
								aabbList
								);
	break;
	case CLUSTERS:
		generateClusteredAABBs(	nAABB,
								aabbMinWidth, aabbMaxWidth,
								aabbMinHeight, aabbMaxHeight,
								aabbMinDepth, aabbMaxDepth,
								universeMinX + 1.0f, universeMaxX - 1.0f,
								universeMinY + 1.0f, universeMaxY - 1.0f,
								universeMinZ + 1.0f, universeMaxZ - 1.0f,
								aabbList
								);
	break;
	case MIX:
		generateMixedAABBs(		nAABB,
								aabbMinWidth, aabbMaxWidth,
								aabbMinHeight, aabbMaxHeight,
								aabbMinDepth, aabbMaxDepth,
								universeMinX + 1.0f, universeMaxX - 1.0f,
								universeMinY + 1.0f, universeMaxY - 1.0f,
								universeMinZ + 1.0f, universeMaxZ - 1.0f,
								aabbList
								);
	break;
	}
}

//Function which creates Frustum distribution
void createFrustum( distrib_t frustumDistrib )
{
	frustumList.clear();

	switch( frustumDistrib )
	{
	case UNIFORM:
	generateRandomPyrFrustums(	nFrustum,
								fMinFOV, fMaxFOV,
								fMinNear, fMaxNear,
								fMinFar, fMaxFar,
								fMinAspectRatio, fMaxAspectRatio,
								universeMinX + 1.0f, universeMaxX - 1.0f,
								universeMinY + 1.0f, universeMaxY - 1.0f,
								universeMinZ + 1.0f, universeMaxZ - 1.0f,
								fMinRotX, fMaxRotX,
								fMinRotY, fMaxRotY,
								fMinRotZ, fMaxRotZ,
								frustumList );
	break;
	case CLUSTERS:
		generateClusteredPyrFrustums(	nFrustum,
									fMinFOV, fMaxFOV,
									fMinNear, fMaxNear,
									fMinFar, fMaxFar,
									fMinAspectRatio, fMaxAspectRatio,
									universeMinX + 1.0f, universeMaxX - 1.0f,
									universeMinY + 1.0f, universeMaxY - 1.0f,
									universeMinZ + 1.0f, universeMaxZ - 1.0f,
									fMinRotX, fMaxRotX,
									fMinRotY, fMaxRotY,
									fMinRotZ, fMaxRotZ,
									frustumList );
	break;
	case MIX:
		generateMixedPyrFrustums(	nFrustum,
								fMinFOV, fMaxFOV,
								fMinNear, fMaxNear,
								fMinFar, fMaxFar,
								fMinAspectRatio, fMaxAspectRatio,
								universeMinX + 1.0f, universeMaxX - 1.0f,
								universeMinY + 1.0f, universeMaxY - 1.0f,
								universeMinZ + 1.0f, universeMaxZ - 1.0f,
								fMinRotX, fMaxRotX,
								fMinRotY, fMaxRotY,
								fMinRotZ, fMaxRotZ,
								frustumList );
	break;
	}
}

//Function which rebuild the BVH from the new frustum/aabb list
void buildBVH()
{
	gculSetBVHDepth( bvhDepth );
	gculSetUniverseAABB( universeMinX, universeMinY, universeMinZ, universeMaxX, universeMaxY, universeMaxZ );
	//Load AABB
	aabb_t* pol = new aabb_t[nAABB];
	for( int i = 0; i < aabbList.size(); ++i )
	{
		pol[ i ].min_x = aabbList[i].m_MinPos.x;
		pol[ i ].min_y = aabbList[i].m_MinPos.y;
		pol[ i ].min_z = aabbList[i].m_MinPos.z;
		pol[ i ].max_x = aabbList[i].m_MaxPos.x;
		pol[ i ].max_y = aabbList[i].m_MaxPos.y;
		pol[ i ].max_z = aabbList[i].m_MaxPos.z;
	}
	
	//Load AABB to the GPU
	gculLoadAABB( nAABB, (void*)pol );

	pyrfrustum_t* frustumPlanesData = new pyrfrustum_t[nFrustum];
	float* frustumCornersData = new float[nFrustum*24];
	
	//Load Frustum data to the GPU
	getFrustumPlanesArray( frustumList, (float*)frustumPlanesData );
	gculLoadFrustumPlanes( nFrustum, (void*)frustumPlanesData );
	getFrustumCornersArray( frustumList, frustumCornersData );
	gculLoadFrustumCorners( nFrustum, (void*)frustumCornersData );

	//Build the BVH
	float timeBuffer = glfwGetTime();
	gculBuildHierarchy();
	bvhBuildingTime = (glfwGetTime() - timeBuffer)*1000.0f;
	//gculSaveHierarchyGraph("kikoo.dot");
	//

	//Delete CPU data
	delete pol;
	delete frustumPlanesData;
	delete frustumCornersData;
}

void buildUniverse()
{
	createAABB( aabbDistribution );
	createFrustum( frustumDistribution );
}

void processCulling()
{
	if( cullingResult != 0 )
		delete cullingResult;

	cullingResult = new unsigned int[ nAABB * nFrustum ];
	gculProcessCulling();
	gculGetResults(cullingResult);
}

void drawUniverse()
{
	for( int j = 0; j < frustumList.size(); ++j )
	{
		for( int i = 0; i < aabbList.size(); ++i )
		{
			if( cullingResult[j*nAABB + i] == 7777777 )
			{
				aabbList[i].isInsideFrustum = true;
			}
		}
		frustumList[j].draw();
	}

	for( int i = 0; i < aabbList.size(); ++i )
	{
		aabbList[i].draw();
	}
}

//RE-BUILD BUTTON CALLBACK
void TW_CALL RunbuildBVH(void * clientData)
{ 
	buildUniverse();
	gculFreeHierarchy();
	gculFreeFrustumPlanes();
	gculFreeFrustumCorners();
	buildBVH();
}

//SAVE DOT FILE CALLBACK
void TW_CALL GenerateDOTFile(void * clientData )
{
	gculSaveHierarchyGraph("hierarchy.dot");
}

// Main
int main(int argc, char** argv ) 
{
	srand( time(NULL) );

	//GLFW Stuff
    GLFWvidmode mode;   // GLFW video mode
    TwBar *bar;         // Pointer to a tweak bar
	TwBar *resultBar;
    //

	//Parameters values
    float bgColor[] = { 0.1f, 0.2f, 0.4f };         // Background color 
	//

	//OpenGL and Camera settings
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
    gluPerspective(90.f, 1.f, 1.f, 1000.f);
	//
	cam = new glCamera();
	cam->m_Position.x=0;
	cam->m_Position.y = 10;
	cam->m_Position.z = -10;
	cam->m_MaxForwardVelocity = 5.0f;
	cam->m_MaxPitchRate = 5.0f;
	cam->m_MaxHeadingRate = 5.0f;
	cam->m_PitchDegrees = 0.0f;
	cam->m_HeadingDegrees = 0.0f;
	//

    // Intialize GLFW   
    if( !glfwInit() )
    {
        // A fatal error occured
        fprintf(stderr, "GLFW initialization failed\n");
        return 1;
    }

    // Create a window
    glfwGetDesktopMode(&mode);
	winWidth = 800;
	winHeight = 600;
	if( !glfwOpenWindow(winWidth, winHeight, mode.RedBits, mode.GreenBits, mode.BlueBits, 
                        0, 16, 0, GLFW_WINDOW /* or GLFW_FULLSCREEN */) )
    {
        // A fatal error occured    
        fprintf(stderr, "Cannot open GLFW window\n");
        glfwTerminate();
        return 1;
    }
    glfwEnable(GLFW_MOUSE_CURSOR);
    glfwEnable(GLFW_KEY_REPEAT);

    glfwSetWindowTitle("gpuCuller Demo Environment");

    // Initialize AntTweakBar
    if( !TwInit(TW_OPENGL, NULL) )
    {
        // A fatal error occured    
        fprintf(stderr, "AntTweakBar initialization failed: %s\n", TwGetLastError());
        glfwTerminate();
        return 1;
    }

	setDefaultParameters(argv);

    // Create a tweak bar
    bar = TwNewBar("mybar");
    TwDefine(" GLOBAL help='This example shows how to integrate AntTweakBar with GLFW and OpenGL.' "); // Message added to the help bar.
	TwDefine(" mybar size='240 500' label='Settings'");

    // Add 'bgColor' to 'bar': it is a modifable variable of type TW_TYPE_COLOR3F (3 floats color)
    TwAddVarRW(bar, "bgColor", TW_TYPE_COLOR3F, &bgColor, "group='Graphics' label='Background color' ");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "universeMinX", TW_TYPE_FLOAT, &universeMinX, "group='Universe' max=1000 label='Universe min. X'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "universeMaxX", TW_TYPE_FLOAT, &universeMaxX, "group='Universe' max=1000 label='Universe max. X'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "universeMinY", TW_TYPE_FLOAT, &universeMinY, "group='Universe' max=1000 label='Universe min. Y'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "universeMaxY", TW_TYPE_FLOAT, &universeMaxY, "group='Universe' max=1000 label='Universe max. Y'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "universeMinZ", TW_TYPE_FLOAT, &universeMinZ, "group='Universe' max=1000 label='Universe min. Z'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "universeMaxZ", TW_TYPE_FLOAT, &universeMaxZ, "group='Universe' max=1000 label='Universe max. Z'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "nAABB", TW_TYPE_UINT32, &nAABB, "group='Primitives' max=100000 step=10 label='Number of primitives'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "aabbMinWidth", TW_TYPE_FLOAT, &aabbMinWidth, "group='Primitives' max=1000 label='AABB min. width'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "aabbMaxWidth", TW_TYPE_FLOAT, &aabbMaxWidth, "group='Primitives' max=1000 label='AABB max. width'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "aabbMinHeight", TW_TYPE_FLOAT, &aabbMinHeight, "group='Primitives' max=1000 label='AABB min. height'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "aabbMaxHeight", TW_TYPE_FLOAT, &aabbMaxHeight, "group='Primitives' max=1000 label='AABB max. height'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "aabbMinDepth", TW_TYPE_FLOAT, &aabbMinDepth, "group='Primitives' max=1000 label='AABB min. depth'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "aabbMaxDepth", TW_TYPE_FLOAT, &aabbMaxDepth, "group='Primitives' max=1000 label='AABB min. width'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "nFrustum", TW_TYPE_UINT32, &nFrustum, "group='Frustum' max=10000 step=10 label='Number of Frustum'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "fMinFOV", TW_TYPE_FLOAT, &fMinFOV, "group='Frustum' max=360 label='Frustum min. FOV'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "fMaxFOV", TW_TYPE_FLOAT, &fMaxFOV, "group='Frustum' max=360 label='Frustum max. FOV'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "fMinNear", TW_TYPE_FLOAT, &fMinNear, "group='Frustum' max=500 label='Frustum min. Near plane dist.'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "fMaxNear", TW_TYPE_FLOAT, &fMaxNear, "group='Frustum' max=500 label='Frustum max. Near plane dist.'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "fMinFar", TW_TYPE_FLOAT, &fMinFar, "group='Frustum' max=500 label='Frustum min. Far plane dist.'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "fMaxFar", TW_TYPE_FLOAT, &fMaxFar, "group='Frustum' max=500 label='Frustum min. Far plane dist.'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "fMinAspectRatio", TW_TYPE_FLOAT, &fMinAspectRatio, "group='Frustum' max=4 label='Frustum min. Aspect Ratio'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "fMaxAspectRatio", TW_TYPE_FLOAT, &fMaxAspectRatio, "group='Frustum' max=4 label='Frustum max. Aspect Ratio'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "fMinRotY", TW_TYPE_FLOAT, &fMinRotY, "group='Frustum' max=4 label='Frustum min. rotation'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "fMaxRotY", TW_TYPE_FLOAT, &fMaxRotY, "group='Frustum' max=4 label='Frustum max. rotation'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "bvhDepth", TW_TYPE_UINT32, &bvhDepth, "group='Hierarchy' max=20 step=1 label='Depth of BVH'");
	
	// Defining an empty distribution enum type
	TwType DistributionType = TwDefineEnum("DistributionType", NULL, 0);

	// Adding season to bar and defining seasonType enum values
	TwAddVarRW(bar, "AABB Distribution", DistributionType, &aabbDistribution, "group='Primitives' enum='1 {Uniform}, 2 {Clusters}, 3 {Mix}' ");

	// Adding season to bar and defining seasonType enum values
	TwAddVarRW(bar, "Frustum Distribution", DistributionType, &frustumDistribution, "group='Frustum' enum='1 {Uniform}, 2 {Clusters}, 3 {Mix}' ");

	//Adding a button to build Hierarchy
	TwAddButton(bar, "Rebuild", RunbuildBVH, NULL, "group='Hierarchy' label='Rebuild Universe' ");

	//Adding a button to build Hierarchy
	TwAddButton(bar, "DOTDOT", GenerateDOTFile, NULL, "group='Hierarchy' label='Generate DOT file' ");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "dynamicMode", TW_TYPE_BOOLCPP, &dynamicMode, "group='Hierarchy' label='Dynamic Mode (rebuild every frame)'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "isCulling", TW_TYPE_BOOLCPP, &isCulling, "group='Hierarchy' label='Culling Active'");

	// Add 'nAABB' to 'bar': it is a modifable variable of type TW_TYPE_UINT32
    TwAddVarRW(bar, "isRendering", TW_TYPE_BOOLCPP, &isRendering, "group='Graphics' label='Rendering Active'");

	//ANT TWEAK BAR FOR RESULT DISPLAY
	resultBar = TwNewBar("resultBar");
    TwDefine(" GLOBAL help='This example shows how to integrate AntTweakBar with GLFW and OpenGL.' "); // Message added to the help bar.
	TwDefine(" mybar size='240 500' label='Results'");
    TwAddVarRO(resultBar, "bvhBuildingTime", TW_TYPE_FLOAT, &bvhBuildingTime, "label='BVH building time (ms)'");
	TwAddVarRO(resultBar, "cullingTime", TW_TYPE_FLOAT, &cullingTime, "label='Culling Time (ms)'");
	TwAddVarRO(resultBar, "frameRate", TW_TYPE_FLOAT, &frameRate, "label='Frame Rate (FPS)'");
	TwAddVarRO(resultBar, "cullingOps", TW_TYPE_FLOAT, &cullingOps, "label='Culling Ops (Millions/s).'");
	//

    // Set GLFW event callbacks
    // - Redirect window size changes to the callback function WindowSizeCB
    glfwSetWindowSizeCallback(WindowSizeCB);
    // - Directly redirect GLFW mouse button events to AntTweakBar
    glfwSetMouseButtonCallback((GLFWmousebuttonfun)TwEventMouseButtonGLFW);
    // - Directly redirect GLFW mouse position events to AntTweakBar
    glfwSetMousePosCallback((GLFWmouseposfun)TwEventMousePosGLFW);
    // - Directly redirect GLFW mouse wheel events to AntTweakBar
    glfwSetMouseWheelCallback((GLFWmousewheelfun)TwEventMouseWheelGLFW);
    // - Directly redirect GLFW key events to AntTweakBar
    glfwSetKeyCallback((GLFWkeyfun)TwEventKeyGLFW);
    // - Directly redirect GLFW char events to AntTweakBar
    glfwSetCharCallback((GLFWcharfun)TwEventCharGLFW);


    // Initialize timers
    double timeBuffer1 = 0;
	double timeBuffer2 = 0;
	//

	//Initialize gpuCuller
	gculInitialize( argc, argv );
	//Create initial universe
	buildUniverse();
	//Create initial BVH
	buildBVH();
	//

    // Main loop (repeated while window is not closed and [ESC] is not pressed)
    while( glfwGetWindowParam(GLFW_OPENED) && !glfwGetKey(GLFW_KEY_ESC) )
    {
		timeBuffer2 = glfwGetTime();

		//Check for keyboard input
		if( glfwGetKey( GLFW_KEY_UP ) == GLFW_PRESS )
			cam->m_ForwardVelocity = 10.0f*frameTime;
		if( glfwGetKey( GLFW_KEY_DOWN ) == GLFW_PRESS )
			cam->m_ForwardVelocity = -10.0f*frameTime;
		if( glfwGetKey( GLFW_KEY_DOWN ) == GLFW_RELEASE && glfwGetKey( GLFW_KEY_UP ) == GLFW_RELEASE )
			cam->m_ForwardVelocity = 0;
		//

		//Check for mouse input
		if( glfwGetMouseButton( GLFW_MOUSE_BUTTON_RIGHT ) == GLFW_PRESS )
		{
			int xpos, ypos;
			glfwGetMousePos( &xpos, &ypos );
			cam->ChangeHeading(0.2f*(float)(xpos-((int)winWidth/2)));
			cam->ChangePitch(0.2f*(float)(ypos-((int)winHeight/2)));
			glfwSetMousePos( winWidth/2, winHeight/2 );
		}
		//

        // Clear frame buffer using bgColor
        glClearColor(bgColor[0], bgColor[1], bgColor[2], 1);
        glClear( GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT );

		//Do stuff
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		cam->SetPrespective();
		//

		//
		if( dynamicMode )
		{
			gculFreeHierarchy();
			gculFreeFrustumPlanes();
			gculFreeFrustumCorners();
			buildBVH();
		}
		//

		//Draw the grid on the "floor"
		drawFloorGrid();

		if( isCulling )
		{
			timeBuffer1 = glfwGetTime();
			processCulling();
			cullingTime = (glfwGetTime() - timeBuffer1)*1000.0f;
			cout << cullingTime << endl;
			cullingOps = (((float)(nFrustum))/(cullingTime/1000.0f))/1000000.0f;
			//
			//Draw the universe
			if( isRendering )
				drawUniverse();
		}

        // Draw tweak bars
        TwDraw();

        // Present frame buffer
        glfwSwapBuffers();

		frameTime = (glfwGetTime() - timeBuffer2);
		frameRate = 1.0f/frameTime;
    }

    // Terminate AntTweakBar and GLFW
    TwTerminate();
    glfwTerminate();

    return 0;
}

//