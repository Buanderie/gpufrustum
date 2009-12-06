#include <deque>
#include <string>
#include <fstream>

#include <gpuCuller_internal.h>

using namespace std;

void DotOutput( char* fileName, hnode_t* h, int sz, int maxDepth )
{
	//File output
	ofstream f (fileName);

	//Create the queue for breadth-first parsing
	deque< hnode_t > q;

	//Header ?
	f << "digraph G1337 {\n";

	//Current node
	hnode_t n;

	//Parsing for Vertices
	//Add the first nodes to the queue lol
	for( int i = 0; i < 4; ++i )
		if( h[i].splitLevel <= maxDepth )
		q.push_back(h[i]);

	while( !q.empty() )
	{
		n = q.front();
		q.pop_front();

		f	<< "\t"
			<< "NODE" << n.ID
			<< " [label=\"[" << n.primStart << ";" << n.primStop << "]\"]\n";
		if( n.splitLevel < maxDepth )
		{
			for( int i = n.childrenStart; i <= n.childrenStop; ++i )
			{
				q.push_back( h[i] );
			}
		}
	}

	for( int i = 0; i < sz; ++i )
	{
		f	<< "\t"
			<< "NODE" << i + sz
			<< " [label=\"" << i << "\" shape=\"box\"]\n";
	}

	f << "\n";
	//
	
	q.clear();

	//Parsing for Edges
	//Add the first nodes to the queue lol
	for( int i = 0; i < 4; ++i )
		if( n.primStart < sz && n.primStop < sz )
		q.push_back(h[i]);

	while( !q.empty() )
	{
		n = q.front();
		q.pop_front();

		//f	<< "NODE" << n.ID
		//	<< " [label=\"[" << n.primStart << ";" << n.primStop << "]\"]\n";
		if( n.splitLevel < maxDepth )
		{
			for( int i = n.childrenStart; i <= n.childrenStop; ++i )
			{
				if( n.primStart < sz && n.primStop < sz )
				{
					q.push_back( h[i] );
					f << "\t";
					f << "NODE" << n.ID;
					f << "->";
					f << "NODE" << h[i].ID;
					f << " [label=\"" << h[i].ID << "\"]\n";
				}
			}
		}
		else
		{
			for( int i = n.primStart; i <= n.primStop; ++i )
			{
				f << "\t";
				f << "NODE" << n.ID;
				f << "->";
				f << "NODE" << i + sz;
				f << " [label=\"" << "" << "\"]\n";
			}
		}
	}
	//
	f << "}\n\n";
}