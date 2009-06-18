#include <algorithm>

#include <gpuCuller_BVH.h>

using namespace std;

BVHNode::BVHNode(BVHNode* parent)
{
	parent = parent;
}

BVHNode::BVHNode( float3 minPos, float3 maxPos, BVHNode* LeftNode, BVHNode* RightNode )
{
	
}

BVHNode* BVHNode::getRightNode()
{
	return rightChild;
}
	
BVHNode* BVHNode::getLeftNode()
{
	return leftChild;
}

BVHNode* BVHNode::getRightBrother()
{
	if( parent != 0 )
		return parent->getLeftNode();
	else
		return 0;
}

BVHTree::BVHTree()
{
	rootNode = new BVHNode(0);
	rootNode->Type = BVHNode::nodetype::BVH_ROOT;
}

void BVHTree::buildFromAABBList(aabb_bvh_t* data, int naabb)
{
	boxData = (aabb_bvh_t*)malloc(naabb*sizeof(aabb_bvh_t));
	memcpy((void*)boxData, (void*)data, naabb*sizeof(aabb_bvh_t));
	splitOnDim(boxData, 0, naabb-1, 0, rootNode);
}

void BVHTree::splitOnDim(aabb_bvh_t *data, int start, int end, int dim, BVHNode *node)
{
	if( start - end == 0 )
	{
		node->leftChild = 0;
		node->rightChild = 0;
		node->nodeAABB.min = data[start].min;
		node->nodeAABB.max = data[start].max;
		node->AABBleafindex = start;
		return;
	}
	
	//We are not on a leaf node, so compute an enclosing AABB
	node->nodeAABB = MergeAABB(data, start, end);

	//Sort the AABBs along dim and find the split
	sortAABBOnDim(data, start, end, dim);
	int split = start + (end-start)/2;

	//Launch new splits
	node->leftChild = new BVHNode(node);
	node->leftChild->Type = BVHNode::BVH_LEFT;
	node->rightChild = new BVHNode(node);
	node->rightChild->Type = BVHNode::BVH_RIGHT;
	splitOnDim( data, start, split, (dim+1)%3, node->getLeftNode());
	splitOnDim( data, split+1, end, (dim+1)%3, node->getRightNode());
}

bool BVHTree::aabbSortFunctionX (aabb_bvh_t i,aabb_bvh_t j)
{
	float centroX1 = (i.min.x) + (i.max.x - i.min.x);
	float centroX2 = (j.min.x) + (j.max.x - j.min.x);
	return (centroX1 < centroX2);
}

bool BVHTree::aabbSortFunctionY (aabb_bvh_t i,aabb_bvh_t j)
{
	float centroY1 = (i.min.y) + (i.max.y - i.min.y);
	float centroY2 = (j.min.y) + (j.max.y - j.min.y);
	return (centroY1 < centroY2);
}

bool BVHTree::aabbSortFunctionZ (aabb_bvh_t i,aabb_bvh_t j)
{
	float centroZ1 = (i.min.z) + (i.max.z - i.min.z);
	float centroZ2 = (j.min.z) + (j.max.z - j.min.z);
	return (centroZ1 < centroZ2);
}

void BVHTree::sortAABBOnDim(aabb_bvh_t* data, int start, int end, int dim)
{
	switch(dim)
	{
	case 0:
		sort (data+start, data+end, aabbSortFunctionX);
		break;
	case 1:
		sort (data+start, data+end, aabbSortFunctionY);
		break;
	case 2:
		sort (data+start, data+end, aabbSortFunctionZ);
		break;
	}
}


aabb_bvh_t BVHTree::MergeAABB( aabb_bvh_t* data, int start, int end )
{
    aabb_bvh_t ret;
    for( int dim = 0; dim < 3; ++dim )
    {
        float minval = 40000000;
        float maxval = -40000000;
        for( int i = start; i <= end; ++i )
        {
            float curminval;
            float curmaxval;
            switch( dim )
            {
				case 0:
                   curminval = data[i].min.x;
                   curmaxval = data[i].max.x;
				case 1:
                   curminval = data[i].min.y;
                   curmaxval = data[i].max.y;
				case 2:
                   curminval = data[i].min.z;
                   curmaxval = data[i].max.z;
            }
            if( curminval < minval )
                minval = curminval;
            if( curmaxval > maxval )
                maxval = curmaxval;
        }
        switch( dim )
        {
            case 0:
                ret.min.x = minval;
                ret.max.x = maxval;
			case 1:
                ret.min.y = minval;
                ret.max.y = maxval;
			case 2:
                ret.min.z = minval;
                ret.max.z = maxval;
        }
    }
    return ret;
}

void BVHTree::updateThreadLink( BVHNode* n, int index )
{
	if( n == 0 )
		return;

	n->ThreadedbinTreeIndex = index;
	updateThreadLink( n->getRightNode(), index );
}

bintreenode_t* BVHTree::getThreadedBinaryTree()
{
	return 0;
}