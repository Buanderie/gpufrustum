#ifndef __GPUCULLER_BVH__
#define __GPUCULLER_BVH__


class BVHNode
{
private:
	BVHNode* leftChild;
	BVHNode* rightChild;
};

class BVHTree
{
private: 
	BVHNode* rootNode;

public:
	void convertToArray();
};

#endif