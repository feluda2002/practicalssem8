#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>

using namespace std;

class Graph {
public:
    Graph(int vertices);
    void addEdge(int v, int w);
    void dfs(int start);
    void bfs(int start);

private:
    int V;
    vector<vector<int>> adjList;
};

Graph::Graph(int vertices) : V(vertices) {
    adjList.resize(V);
}

void Graph::addEdge(int v, int w) {
    adjList[v].push_back(w);
    adjList[w].push_back(v);
}

void Graph::dfs(int start) {
    vector<bool> visited(V, false);
    stack<int> s;

    s.push(start);

    while (!s.empty()) {
        int curr;
        #pragma omp critical
        {
            if (!s.empty()) {
                curr = s.top();
                s.pop();
            }
        }

        if (!visited[curr]) {
            cout << curr << " ";
            visited[curr] = true;

            #pragma omp parallel for
            for (int neighbor : adjList[curr]) {
                if (!visited[neighbor]) {
                    #pragma omp critical
                    {
                        if (!visited[neighbor]) {
                            s.push(neighbor);
                        }
                    }
                }
            }
        }
    }
}

void Graph::bfs(int start) {
    vector<bool> visited(V, false);
    queue<int> q;

    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int curr;
        #pragma omp critical
        {
            if (!q.empty()) {
                curr = q.front();
                q.pop();
            }
        }

        cout << curr << " ";

        #pragma omp parallel for
        for (int neighbor : adjList[curr]) {
            #pragma omp critical
            {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }
    }
}

int main() {
    Graph g(6);

    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);

    cout << "DFS traversal starting from vertex 0: ";
    g.dfs(0);
    cout << endl;

    cout << "BFS traversal starting from vertex 0: ";
    g.bfs(0);
    cout << endl;

    return 0;
}

