#include <bits/stdc++.h>
using namespace std;
int **graphMatrix, **residualGraph, vertexCount;

bool bfs(int vertexS, int vertexT, int* parent)
{
    bool visited[vertexCount];

    queue<int> vertexQueue;
    vertexQueue.push(vertexS);
    visited[vertexS] = true;
    parent[vertexS] = -1;

    while (!vertexQueue.empty()) {
        int vertexU = vertexQueue.front();
        vertexQueue.pop();

        for (int vertexV = 0; vertexV < vertexCount; vertexV++) {
            if (visited[vertexV] == false && residualGraph[vertexU][vertexV] > 0) {
                vertexQueue.push(vertexV);
                parent[vertexV] = vertexU;
                visited[vertexV] = true;
            }
        }
    }
    return visited[vertexT];
}

int maxFlow(string* arrayIndexStringEquivalents, int vertexS, int vertexT)
{
    int maxFlow = 0;
    int parent[vertexCount];
    int vertexU = 0;
    int vertexV = 0;

    for (vertexU = 0; vertexU < vertexCount; vertexU++) {
        for (vertexV = 0; vertexV < vertexCount; vertexV++) {
            residualGraph[vertexU][vertexV] = graphMatrix[vertexU][vertexV];
        }
    }

    while (bfs(vertexS, vertexT, parent)) {
        string pathString = "";

        int bottleneckFlow = INT_MAX;
        for (vertexV = vertexT; vertexV != vertexS; vertexV = parent[vertexV]) {
            vertexU = parent[vertexV];
            bottleneckFlow = min(bottleneckFlow, residualGraph[vertexU][vertexV]);

            pathString = " --> " + arrayIndexStringEquivalents[vertexV] + pathString;
        }
        pathString = "S" + pathString;
        cout << "Augmentation path \n" << pathString;
        cout << "bottleneck (min flow on path added to max flow) = " << bottleneckFlow << endl;

        for (vertexV = vertexT; vertexV != vertexS; vertexV = parent[vertexV]) {
            vertexU = parent[vertexV];
            residualGraph[vertexU][vertexV] -= bottleneckFlow;
            residualGraph[vertexV][vertexU] += bottleneckFlow;
        }

        maxFlow += bottleneckFlow;
    }

    //cout << maxFlow;

    return maxFlow;
}

int main()
{
    vertexCount = 8;
    string arrayIndexStringEquivalents[vertexCount] = { "S", "2", "3", "4", "5", "6", "7", "T" }; //map human readable names to each vertex, not just array indexes
    graphMatrix = (int**)malloc(sizeof(int*) * vertexCount), residualGraph = (int**)malloc(sizeof(int*) * vertexCount);
    for (int i = 0; i < vertexCount; i++)
        graphMatrix[i] = (int*)malloc(sizeof(int) * vertexCount), residualGraph[i] = (int*)malloc(sizeof(int) * vertexCount);
    graphMatrix[0][0] = 0, graphMatrix[0][1] = 10, graphMatrix[0][2] = 5, graphMatrix[0][3] = 15, graphMatrix[0][4] = 0, graphMatrix[0][5] = 0, graphMatrix[0][6] = 0, graphMatrix[0][7] = 0, //,		//edges FROM S TO anything
        graphMatrix[1][0] = 0, graphMatrix[1][1] = 0, graphMatrix[1][2] = 4, graphMatrix[1][3] = 0, graphMatrix[0][4] = 9, graphMatrix[0][5] = 15, graphMatrix[0][6] = 0, graphMatrix[0][7] = 0,
    graphMatrix[2][0] = 0, graphMatrix[2][1] = 0, graphMatrix[2][2] = 0, graphMatrix[2][3] = 4, graphMatrix[0][4] = 0, graphMatrix[0][5] = 8, graphMatrix[0][6] = 0, graphMatrix[0][7] = 0,
    graphMatrix[3][0] = 0, graphMatrix[3][1] = 0, graphMatrix[3][2] = 0, graphMatrix[3][3] = 0, graphMatrix[0][4] = 0, graphMatrix[0][5] = 0, graphMatrix[0][6] = 30, graphMatrix[0][7] = 0,
    graphMatrix[4][0] = 0, graphMatrix[4][1] = 0, graphMatrix[4][2] = 0, graphMatrix[4][3] = 0, graphMatrix[0][4] = 0, graphMatrix[0][5] = 15, graphMatrix[0][6] = 0, graphMatrix[0][7] = 10,
    graphMatrix[5][0] = 0, graphMatrix[5][1] = 0, graphMatrix[5][2] = 0, graphMatrix[5][3] = 0, graphMatrix[0][4] = 0, graphMatrix[0][5] = 0, graphMatrix[0][6] = 15, graphMatrix[0][7] = 10,
    graphMatrix[6][0] = 0, graphMatrix[6][1] = 0, graphMatrix[6][2] = 6, graphMatrix[6][3] = 0, graphMatrix[0][4] = 0, graphMatrix[0][5] = 0, graphMatrix[0][6] = 0, graphMatrix[0][7] = 10,
    graphMatrix[7][0] = 0, graphMatrix[7][1] = 0, graphMatrix[7][2] = 0, graphMatrix[7][3] = 0, graphMatrix[0][4] = 0, graphMatrix[0][5] = 0, graphMatrix[0][6] = 0, graphMatrix[0][7] = 0;

    int vertexS = 0;
    int vertexT = vertexCount - 1;
    cout << maxFlow(arrayIndexStringEquivalents, vertexS, vertexT);

    return 0;
}
