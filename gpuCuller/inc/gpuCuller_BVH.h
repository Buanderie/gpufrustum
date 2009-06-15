#ifndef __GPUCULLER_BVH__
#define __GPUCULLER_BVH__

#include <cutil_inline.h>

typedef struct aabb_bvh
{
	float3 min, max;
} aabb_bvh_t;

typedef struct bintreenode
{
	float3 minpoint;
	float3 maxpoint;
	unsigned int skiplink;
	unsigned int itemInd;

} bintreenode_t;

class BVHNode
{
public:
	
	typedef enum nodetype
	{
		BVH_ROOT, BVH_LEFT, BVH_RIGHT
	} nodetype_t;

	BVHNode* leftChild;
	BVHNode* rightChild;
	BVHNode* parent;
	aabb_bvh_t nodeAABB;
	int AABBleafindex;
	int ThreadedbinTreeIndex;
	BVHNode( BVHNode* parent );
	BVHNode* getRightNode();
	BVHNode* getLeftNode();
	BVHNode* getRightBrother();
	BVHNode( float3 minPos, float3 maxPos, BVHNode* LeftNode, BVHNode* RightNode );
	nodetype_t Type;
};

class BVHTree
{
public: 
	BVHTree();
	BVHNode* rootNode;
	aabb_bvh_t* boxData;
	void splitOnDim(aabb_bvh_t *data, int start, int end, int dim, BVHNode *node);
	void sortAABBOnDim(aabb_bvh_t* data, int start, int end, int dim);
	aabb_bvh_t MergeAABB( aabb_bvh_t* data, int start, int end );
	void convertToArray();
	void buildFromAABBList(aabb_bvh_t* data, int naabb);
	BVHNode* getRoot();
	static bool aabbSortFunctionX (aabb_bvh_t i,aabb_bvh_t j);
	static bool aabbSortFunctionY (aabb_bvh_t i,aabb_bvh_t j);
	static bool aabbSortFunctionZ (aabb_bvh_t i,aabb_bvh_t j);
	static void updateThreadLink( BVHNode* n, int index );
	bintreenode_t* getThreadedBinaryTree();
};

#endif