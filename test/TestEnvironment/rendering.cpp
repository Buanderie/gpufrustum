#include "rendering.h"

void drawFloorGrid()
{
	glEnable(GL_LINE_SMOOTH);
	glLineWidth(1.0f);

	glEnable(GL_BLEND);		// Turn Blending On
	glDisable(GL_DEPTH_TEST);	// Turn Depth Testing Off

	glColor4f(.4,.4,.4,0.8);
	glBlendFunc(GL_SRC_ALPHA,GL_ONE);
	glBegin(GL_QUADS);
	glVertex3f( -1000,-0.001, -1000);
	glVertex3f( -1000,-0.001,1000);
	glVertex3f(1000,-0.001,1000);
	glVertex3f(1000,-0.001, -1000);
	glEnd();

	glEnable(GL_BLEND);		// Turn Blending On
	glDisable(GL_DEPTH_TEST);	// Turn Depth Testing Off

	for(int i=-1000;i<=1000;i++) {
		if (i==0) { glColor3f(1.0,0.0,0.0); glLineWidth(2.0f);} else { glColor3f(0.50,.50,.50); glLineWidth(1.0f);};
		glBegin(GL_LINES);
		glVertex3f(i,0,-1000);
		glVertex3f(i,0,1000);
		glEnd();
		if (i==0) { glColor3f(0.0,0.0,1.0); glLineWidth(2.0f);} else { glColor3f(0.50,.50,.50); glLineWidth(1.0f);};
		glBegin(GL_LINES);
		glVertex3f(1000,0,i);
		glVertex3f(-1000,0,i);
		glEnd();
	}

	glLineWidth(2.0f);
	glColor3f(0.0,1.0,0.0);
	glBegin(GL_LINES);
	glVertex3f(0,-1000,0);
	glVertex3f(0,1000,0);
	glEnd();

}