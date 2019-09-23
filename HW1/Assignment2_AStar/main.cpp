#include<iostream>
#include<set>
#include<string>
#include<iomanip>
#include"graph.hpp"

using namespace std;

string name[] = { "Arad","Bucharset","Craiova","Dobreta",
"Eforie","Fagaras","Giurgiu","Hirsova","Iasi","Lugoj","Mehadia","Neamt",
"Oradea","Pitesti","Rimnicu_Vilcea","Sibiu","Timisoara","Urziceni","Vaslui","Zerind" };

void AStar(Graph newGraph, city_name_set start,city_name_set dest);

int main() {
	Graph newGraph;
	newGraph.init();
	cout << "Please specify the departure city:" << endl;
	for (int i = 0; i < 20; i++) {
		cout << setw(15) << name[i] << "->" << setw(2) << i;
		if (i % 2)
			cout << endl;
		else
			cout << "       ";
	}
	int start;
	cin >> start;
	AStar(newGraph, (city_name_set)start, Bucharset);
}

void AStar(Graph newGraph, city_name_set start,city_name_set dest) {
	set<City> openList;
	bool isClosed[20] = {false};
	bool isOpen[20] = {false};
	city_name_set parent[20];

	for (int i = 0; i < 20; i++)parent[i] = (city_name_set)i;

	openList.emplace(City(start));
	isOpen[start] = true;
	while (!openList.empty()) {
		City current = *openList.begin();
		if (current.getName() == dest) {
			break;
		}
		openList.erase(openList.begin());
		isOpen[current.getName()] = false;
		isClosed[current.getName()] = true;
		for (int i = 0; i < 20; i++) {
			if (isClosed[i])continue;
			int weight = newGraph.getEdgeWeight(current.getName(), (city_name_set)i);
			if (weight == 0)continue;
			if (isOpen[i]) {
				for (set<City>::iterator iter = openList.begin(); iter != openList.end(); iter++) {
					if (iter->getName() == i) {
						if (iter->getf() > weight + current.getg() + newGraph.getStraight((city_name_set)i))
						{
							openList.erase(iter);
							openList.emplace(City((city_name_set)i, weight + current.getg(), newGraph.getStraight((city_name_set)i)));
							parent[(city_name_set)i] = current.getName();
						}
					}
				}
			}
			else {
				openList.emplace(City((city_name_set)i, weight + current.getg(), newGraph.getStraight((city_name_set)i)));
				isOpen[i] = true;
				parent[(city_name_set)i] = current.getName();
			}
		}
	}

	int last = dest;
	cout << "Path" << endl << name[last];
	while (true) {
		int temp = parent[last];
		cout << " <- " << name[temp];
		if (parent[temp] == temp) {
			break;
		}
		last = temp;
	}
}