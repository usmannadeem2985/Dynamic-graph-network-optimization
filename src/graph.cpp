#include "../include/graph.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <queue>
#include <limits>
#include <algorithm>

Graph::Graph() : n_nodes(0), n_edges(0), n_objectives(1) {}

bool Graph::loadFromMetis(const std::string& filename, int num_objectives) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening METIS file: " << filename << std::endl;
        return false;
    }
    this->n_objectives = num_objectives;
    std::string line;
    // Skip comments
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '%') continue;
        std::istringstream iss(line);
        iss >> n_nodes >> n_edges;
        break;
    }
    adjList.assign(n_nodes, std::vector<Edge>());
    int node = 0;
    while (std::getline(infile, line) && node < n_nodes) {
        if (line.empty() || line[0] == '%') continue;
        std::istringstream iss(line);
        int neighbor;
        while (iss >> neighbor) {
            // METIS is 1-based, convert to 0-based
            adjList[node].push_back({neighbor - 1, std::vector<double>(n_objectives, 1.0)});
            edgeIndex[node][neighbor - 1] = adjList[node].size() - 1;
        }
        node++;
    }
    return true;
}

bool Graph::loadFromEdgeList(const std::string& filename, int num_objectives) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening edge list file: " << filename << std::endl;
        return false;
    }
    this->n_objectives = num_objectives;
    std::string line;
    int max_node = -1;
    std::vector<std::pair<int, int>> edges;
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        int from, to;
        if (!(iss >> from >> to)) continue;
        edges.emplace_back(from, to);
        if (from > max_node) max_node = from;
        if (to > max_node) max_node = to;
    }
    n_nodes = max_node + 1;
    adjList.assign(n_nodes, std::vector<Edge>());
    n_edges = 0;
    for (const auto& e : edges) {
        addEdge(e.first, e.second, std::vector<double>(n_objectives, 1.0));
    }
    return true;
}

void Graph::addEdge(int from, int to, const std::vector<double>& weights) {
    adjList[from].push_back({to, weights});
    edgeIndex[from][to] = adjList[from].size() - 1;
    n_edges++;
}

void Graph::removeEdge(int from, int to) {
    auto& edges = adjList[from];
    auto it = edgeIndex[from].find(to);
    if (it != edgeIndex[from].end()) {
        size_t idx = it->second;
        edges.erase(edges.begin() + idx);
        edgeIndex[from].erase(to);
        n_edges--;
        // Rebuild index
        for (size_t i = 0; i < edges.size(); ++i) {
            edgeIndex[from][edges[i].to] = i;
        }
    }
}

void Graph::updateEdgeWeight(int from, int to, const std::vector<double>& new_weights) {
    auto it = edgeIndex[from].find(to);
    if (it != edgeIndex[from].end()) {
        adjList[from][it->second].weights = new_weights;
    }
}

int Graph::numNodes() const {
    return n_nodes;
}

int Graph::numEdges() const {
    return n_edges;
}

const std::vector<Edge>& Graph::neighbors(int node) const {
    return adjList[node];
}

std::vector<int> Graph::partitionWithMetis(int num_partitions) {
    std::vector<idx_t> xadj(n_nodes + 1, 0);
    std::vector<idx_t> adjncy;
    idx_t edge_count = 0;
    for (int i = 0; i < n_nodes; ++i) {
        for (const auto& edge : adjList[i]) {
            adjncy.push_back(edge.to);
            ++edge_count;
        }
        xadj[i + 1] = edge_count;
    }
    idx_t nVertices = n_nodes;
    idx_t nParts = num_partitions;
    idx_t ncon = 1;
    std::vector<idx_t> part(n_nodes, 0);
    idx_t objval;
    int options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    int ret = METIS_PartGraphKway(&nVertices, &ncon, xadj.data(), adjncy.data(),
                                  nullptr, nullptr, nullptr, &nParts, nullptr, nullptr, options,
                                  &objval, part.data());
    if (ret != METIS_OK) {
        std::cerr << "METIS_PartGraphKway failed!" << std::endl;
        return std::vector<int>();
    }
    std::vector<int> result(part.begin(), part.end());
    return result;
}

void Graph::setNumNodes(int n) {
    n_nodes = n;
    adjList.resize(n_nodes);
}

void Graph::computeParetoFronts(const std::vector<int>& owned_nodes, int source, std::vector<std::vector<std::vector<double>>>& pareto_fronts) {
    // Simple multi-objective Dijkstra-like algorithm (for demonstration)
    int n = numNodes();
    int m = n_objectives;
    pareto_fronts.assign(n, {});
    using Path = std::vector<double>;
    auto dominates = [](const Path& a, const Path& b) {
        bool strictly_better = false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i] > b[i]) return false;
            if (a[i] < b[i]) strictly_better = true;
        }
        return strictly_better;
    };
    std::vector<std::vector<Path>> fronts(n);
    std::priority_queue<std::pair<Path, int>, std::vector<std::pair<Path, int>>, std::greater<>> pq;
    pq.push({std::vector<double>(m, 0.0), source});
    fronts[source].push_back(std::vector<double>(m, 0.0));
    while (!pq.empty()) {
        auto [cost, u] = pq.top(); pq.pop();
        for (const auto& edge : adjList[u]) {
            Path new_cost(m);
            for (int i = 0; i < m; ++i) new_cost[i] = cost[i] + edge.weights[i];
            // Check if new_cost is dominated in fronts[edge.to]
            bool dominated = false;
            auto& pf = fronts[edge.to];
            for (const auto& existing : pf) {
                if (dominates(existing, new_cost) || existing == new_cost) {
                    dominated = true;
                    break;
                }
            }
            if (!dominated) {
                // Remove all paths dominated by new_cost
                pf.erase(std::remove_if(pf.begin(), pf.end(), [&](const Path& p) { return dominates(new_cost, p); }), pf.end());
                pf.push_back(new_cost);
                pq.push({new_cost, edge.to});
            }
        }
    }
    // Only keep results for owned_nodes
    for (int node : owned_nodes) {
        pareto_fronts[node] = fronts[node];
    }
}

bool isNonDominated(const Path& candidate, const std::vector<Path>& currentSet) {
    for (const auto& p : currentSet) {
        bool dominates = true, strictly_better = false;
        for (size_t i = 0; i < p.objectives.size(); ++i) {
            if (p.objectives[i] < candidate.objectives[i]) strictly_better = true;
            if (p.objectives[i] > candidate.objectives[i]) { dominates = false; break; }
        }
        if (dominates && strictly_better) return false;
    }
    return true;
} 