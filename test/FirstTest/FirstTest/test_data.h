#ifndef __TEST_DATA_H__
#define __TEST_DATA_H__

float frustum_planes[24] = {	
0.00000000,
0.00000000,
-1.0000000,
1.0000000,
-0.00000000,
0.00000000,
1.0000000,
-10.000000,
-0.81649655,
-0.00000000,
-0.57735026,
0.00000000,
0.81649655,
0.00000000,
-0.57735026,
0.00000000,
0.00000000,
0.81649655,
-0.57735026,
0.00000000,
0.00000000,
-0.81649655,
-0.57735026,
0.00000000
};

float frustum_corners[24] = {
-0.70710677,
0.70710677,
1.0000000,
1.0000000,
0.70710677,
0.70710677,
1.0000000,
1.0000000,
-0.70710677,
-0.70710677,
1.0000000,
1.0000000,
0.70710677,
-0.70710677,
1.0000000,
1.0000000,
-7.0710678,
7.0710678,
10.000000,
1.0000000,
7.0710678,
7.0710678,
10.000000,
1.0000000
		};

float boxA_points[24] = {	-1.0f,	1.0f, 8.0f,
							1.0f,	1.0f, 8.0f,	
							-1.0f,	1.0f, 6.0f,
							1.0f,	1.0f, 6.0f,
							-1.0f,	-1.0f, 8.0f,
							1.0f,	-1.0f, 8.0f,
							-1.0f,	-1.0f, 6.0f,
							1.0f,	-1.0f, 6.0f
						};
#endif