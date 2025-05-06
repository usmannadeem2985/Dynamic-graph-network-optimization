#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <string>
#include <unordered_map>
#include <metis.h>

struct Edge {
    int to;
    std::vector<double> weights; // Multi-objective weights
};

struct Path {
    std::vector<float> objectives;
    std::vector<int> nodes;
};

class Graph {
public:
    Graph();
    bool loadFromMetis(const std::string& filename, int num_objectives = 1);
    bool loadFromEdgeList(const std::string& filename, int num_objectives = 1);
    void addEdge(int from, int to, const std::vector<double>& weights);
    void removeEdge(int from, int to);
    void updateEdgeWeight(int from, int to, const std::vector<double>& new_weights);
    int numNodes() const;
    int numEdges() const;
    const std::vector<Edge>& neighbors(int node) const;
    std::vector<int> partitionWithMetis(int num_partitions);
    void setNumNodes(int n);
    void computeParetoFronts(const std::vector<int>& owned_nodes, int source, std::vector<std::vector<std::vector<double>>>& pareto_fronts);
    std::unordered_map<int, std::vector<Path>> paretoPaths;
    std::vector<bool> isUpdated;

private:
    std::vector<std::vector<Edge>> adjList;
    int n_nodes;
    int n_edges;
    int n_objectives;
    std::unordered_map<int, std::unordered_map<int, size_t>> edgeIndex; // from -> (to -> index in adjList)
};

bool isNonDominated(const Path& candidate, const std::vector<Path>& currentSet);

#endif // GRAPH_H 