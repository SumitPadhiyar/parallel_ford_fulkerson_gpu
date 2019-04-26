#include <utility>
#include <bits/stdc++.h>
#include "graph_generator.h"

using namespace std;

class Network {
private:
    unsigned short vertex_count = 0;
    //vector<string> vertex_string_representation;

    vector<vector<unsigned short> > adjacency_matrix;

    bool bfs(unsigned short vertexS, unsigned short vertexT, vector<unsigned short> &parent) {
        vector<bool> visited(vertex_count);

        queue<unsigned short> vertexQueue;
        vertexQueue.push(vertexS);
        visited[vertexS] = true;
        parent[vertexS] = -1;

        while (!vertexQueue.empty()) {
            int vertexU = vertexQueue.front();
            vertexQueue.pop();

            for (int vertexV = 0; vertexV < vertex_count; vertexV++) {
                if (!visited[vertexV] && adjacency_matrix[vertexU][vertexV] > 0) {
                    //cout << "visiting " << vertexV << ",";
                    vertexQueue.push(vertexV);
                    parent[vertexV] = vertexU;
                    visited[vertexV] = true;
                }
            }
        }

        //cout << "returning- " << visited[vertexT] << endl;

        return visited[vertexT];
    }

public:

    explicit Network(unsigned short vertex_count) : vertex_count(vertex_count), adjacency_matrix(
            vector<vector<unsigned short> >(vertex_count, vector<unsigned short>(vertex_count))) {
        this->vertex_count = vertex_count;
        //vertex_string_representation.resize(vertex_count);
    }

    void add_vertex_string_representation(int index, string str) {
        //vertex_string_representation[index] = std::move(str);
    }

    void add_edge(unsigned short from, unsigned short to, unsigned short capacity) {
        if (from >= vertex_count || to >= vertex_count) {
            return;
        }

        adjacency_matrix[from][to] = capacity;
    }

    int maxFlow(unsigned short vertexS, unsigned short vertexT) {
        unsigned int maxFlow = 0;
        vector<unsigned short> parent(vertex_count);
        unsigned short vertexU = 0;
        unsigned short vertexV = 0;

        while (bfs(vertexS, vertexT, parent)) {
            string pathString;

            unsigned short bottleneckFlow = USHRT_MAX;
            for (vertexV = vertexT; vertexV != vertexS; vertexV = parent[vertexV]) {
                vertexU = parent[vertexV];
                bottleneckFlow = min(bottleneckFlow, adjacency_matrix[vertexU][vertexV]);

//                pathString.insert(0, to_string(vertexV));
//                pathString.insert(0, " --> ");
            }

//            pathString.insert(0, "S");

//            cout << "Augmentation path: " << pathString << "\t";
//            cout << " bottleneck (min flow on path added to max flow) = " << bottleneckFlow << endl;

            for (vertexV = vertexT; vertexV != vertexS; vertexV = parent[vertexV]) {
                vertexU = parent[vertexV];
                adjacency_matrix[vertexU][vertexV] -= bottleneckFlow;
                adjacency_matrix[vertexV][vertexU] += bottleneckFlow;
            }

            maxFlow += bottleneckFlow;
        }

        return maxFlow;
    }
};


int main(int argc, char **argv) {
    auto start = chrono::high_resolution_clock::now();

    if (argc < 1) {
        cout << "Input Filename is required" << endl;
        return 0;
    }

    ifstream file;
    file.open(argv[1]);

    if (!file) {
        cout << "Error reading file!";
        return 0;
    }

    int vertexCount = atoi(argv[2]);

    int vertexS = 0;
    int vertexT = vertexCount - 1;

    Network network(vertexCount);
    //map human readable names to each vertex, not just array indexes
//    network.add_vertex_string_representation(0, "S");
//    network.add_vertex_string_representation(1, "2");
//    network.add_vertex_string_representation(2, "3");
//    network.add_vertex_string_representation(3, "4");
//    network.add_vertex_string_representation(4, "5");
//    network.add_vertex_string_representation(5, "6");
//    network.add_vertex_string_representation(6, "7");
//    network.add_vertex_string_representation(7, "I");
//
//    network.add_edge(0, 0, 0);
//    network.add_edge(0, 1, 10);
//    network.add_edge(0, 2, 5);
//    network.add_edge(0, 3, 15);
//    network.add_edge(0, 4, 0);
//    network.add_edge(0, 5, 0);
//    network.add_edge(0, 6, 0);
//    network.add_edge(0, 7, 0);
//    network.add_edge(1, 0, 0);
//    network.add_edge(1, 1, 0);
//    network.add_edge(1, 2, 4);
//    network.add_edge(1, 3, 0);
//    network.add_edge(1, 4, 9);
//    network.add_edge(1, 5, 15);
//    network.add_edge(1, 6, 0);
//    network.add_edge(1, 7, 0);
//    network.add_edge(2, 0, 0);
//    network.add_edge(2, 1, 0);
//    network.add_edge(2, 2, 0);
//    network.add_edge(2, 3, 4);
//    network.add_edge(2, 4, 0);
//    network.add_edge(2, 5, 8);
//    network.add_edge(2, 6, 0);
//    network.add_edge(2, 7, 0);
//    network.add_edge(3, 0, 0);
//    network.add_edge(3, 1, 0);
//    network.add_edge(3, 2, 0);
//    network.add_edge(3, 3, 0);
//    network.add_edge(3, 4, 0);
//    network.add_edge(3, 5, 0);
//    network.add_edge(3, 6, 30);
//    network.add_edge(3, 7, 0);
//    network.add_edge(4, 0, 0);
//    network.add_edge(4, 1, 0);
//    network.add_edge(4, 2, 0);
//    network.add_edge(4, 3, 0);
//    network.add_edge(4, 4, 0);
//    network.add_edge(4, 5, 15);
//    network.add_edge(4, 6, 0);
//    network.add_edge(4, 7, 10);
//    network.add_edge(5, 0, 0);
//    network.add_edge(5, 1, 0);
//    network.add_edge(5, 2, 0);
//    network.add_edge(5, 3, 0);
//    network.add_edge(5, 4, 0);
//    network.add_edge(5, 5, 0);
//    network.add_edge(5, 6, 15);
//    network.add_edge(5, 7, 10);
//    network.add_edge(6, 0, 0);
//    network.add_edge(6, 1, 0);
//    network.add_edge(6, 2, 6);
//    network.add_edge(6, 3, 0);
//    network.add_edge(6, 4, 0);
//    network.add_edge(6, 5, 0);
//    network.add_edge(6, 6, 0);
//    network.add_edge(6, 7, 10);
//    network.add_edge(7, 0, 0);
//    network.add_edge(7, 1, 0);
//    network.add_edge(7, 2, 0);
//    network.add_edge(7, 3, 0);
//    network.add_edge(7, 4, 0);
//    network.add_edge(7, 5, 0);
//    network.add_edge(7, 6, 0);
//    network.add_edge(7, 7, 0);

    string line;
    unsigned short source, destination, capacity;

    while (file) {
        getline(file, line);
        if (line.empty()) {
            continue;
        }

        std::stringstream linestream(line);
        linestream >> source >> destination >> capacity;

        //cout << source << "\t" << destination << "\t" << capacity << endl;

        network.add_edge(source, destination, capacity);
    }

    file.close();

    int maxflow;

    maxflow = network.maxFlow(vertexS, vertexT);

    auto duration = chrono::duration_cast<chrono::milliseconds>(start - chrono::high_resolution_clock::now());

    cout << "Network maxflow: " << maxflow << " duration -" << duration.count() << " ms" << endl;

    return 0;
}
