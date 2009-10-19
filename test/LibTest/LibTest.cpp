// LibTest.cpp : Defines the entry point for the console application.
//

#include <gpuCuller.h>

int main(int argc, char** argv)
{
	float* pol = new float[50*6];

	gculInitialize( argc, argv );

	gculLoadAABB( 50, (void*)pol );

	return 0;
}

