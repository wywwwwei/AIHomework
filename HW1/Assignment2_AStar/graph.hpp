#pragma once

enum city_name_set {Arad,Bucharset,Craiova,Dobreta,
	Eforie,Fagaras,Giurgiu,Hirsova,Iasi,Lugoj,Mehadia,Neamt,
	Oradea,Pitesti,Rimnicu_Vilcea,Sibiu,Timisoara,Urziceni,Vaslui,Zerind};

class  City
{
	city_name_set id_name;
	int f;
	int g;
	int h;

public:
	City(city_name_set _id_name, int _g = 0, int _h = 0) {
		id_name = _id_name;
		f = _g + _h;
		g = _g;
		h = _h;
	}

	city_name_set getName() const{
		return id_name;
	}

	int getf()const{
		return f;
	}
	int getg() const {
		return g;
	}

	bool operator<(const City&temp)const {
		return this->f < temp.f;
	}
};

class Graph {
private:
	int graph[20][20];
	int straightToBucharest[20];

public:
	Graph() {
		for (int i = 0; i < 20; i++)
			for (int j = 0; j < 20; j++)
				graph[i][j] = 0;
		for (int i = 0; i < 20; i++)
			straightToBucharest[i] = 0;
	}

	void addAnEdge(city_name_set cityA, city_name_set cityB, int weight) {
		graph[cityA][cityB] = graph[cityB][cityA] = weight;
	}

	int getEdgeWeight(city_name_set cityA, city_name_set cityB) {
		return graph[cityA][cityB];
	}

	void addStraight(city_name_set cityA, int weight) {
		straightToBucharest[cityA] = weight;
	}

	int getStraight(city_name_set cityA) {
		return straightToBucharest[cityA];
	}

	void init() {
		//Arad
		this->addAnEdge(Arad, Zerind, 75);
		this->addAnEdge(Arad, Sibiu, 140);
		this->addAnEdge(Arad, Timisoara, 118);
		//Bucharset
		this->addAnEdge(Bucharset, Fagaras, 211);
		this->addAnEdge(Bucharset, Pitesti, 101);
		this->addAnEdge(Bucharset, Urziceni, 85);
		this->addAnEdge(Bucharset, Giurgiu, 90);
		//Craiova
		this->addAnEdge(Craiova, Dobreta, 120);
		this->addAnEdge(Craiova, Rimnicu_Vilcea, 146);
		this->addAnEdge(Craiova, Pitesti, 138);
		//Dobreta
		this->addAnEdge(Dobreta, Mehadia, 75);
		//Eforie
		this->addAnEdge(Eforie, Hirsova, 86);
		//Fagaras
		this->addAnEdge(Fagaras, Sibiu, 99);
		//Giurgiu
		//Hirsova
		this->addAnEdge(Hirsova, Urziceni, 98);
		//Iasi
		this->addAnEdge(Iasi, Neamt, 87);
		this->addAnEdge(Iasi, Vaslui, 92);
		//Lugoj
		this->addAnEdge(Lugoj, Timisoara, 111);
		this->addAnEdge(Lugoj, Mehadia, 70);
		//Mehadia
		//Neamt
		//Oradea
		this->addAnEdge(Oradea, Zerind, 71);
		this->addAnEdge(Oradea, Sibiu, 151);
		//Pitesti
		this->addAnEdge(Pitesti, Rimnicu_Vilcea, 97);
		//Rimnicu Vilcea
		this->addAnEdge(Rimnicu_Vilcea, Sibiu, 80);
		//Sibiu
		//Timisoara
		//Urziceni
		this->addAnEdge(Urziceni, Vaslui, 142);
		//Vaslui
		//Zerind


		this->addStraight(Arad, 366);
		this->addStraight(Bucharset, 0);
		this->addStraight(Craiova, 160);
		this->addStraight(Dobreta, 242);
		this->addStraight(Eforie, 161);
		this->addStraight(Fagaras, 178);
		this->addStraight(Giurgiu, 77);
		this->addStraight(Hirsova, 151);
		this->addStraight(Iasi, 226);
		this->addStraight(Lugoj, 244);
		this->addStraight(Mehadia, 241);
		this->addStraight(Neamt, 234);
		this->addStraight(Oradea, 380);
		this->addStraight(Pitesti, 98);
		this->addStraight(Rimnicu_Vilcea, 193);
		this->addStraight(Sibiu, 253);
		this->addStraight(Timisoara, 329);
		this->addStraight(Urziceni, 80);
		this->addStraight(Vaslui, 199);
		this->addStraight(Zerind, 374);
	}

};