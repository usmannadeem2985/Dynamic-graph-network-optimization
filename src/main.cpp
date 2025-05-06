#include "../include/graph.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int max_nodes = 1000;
    Graph g;
    g.setNumNodes(max_nodes);
    std::ifstream infile("dataset/dataset.txt");
    std::string line;
    std::unordered_set<int> valid_nodes;
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        int from, to;
        if (!(iss >> from >> to)) continue;
        if (from < max_nodes && to < max_nodes) {
            g.addEdge(from, to, {1.0, 1.0});
            valid_nodes.insert(from);
            valid_nodes.insert(to);
        }
    }
    std::vector<int> partitions;
    if (world_rank == 0) {
        std::cout << "Loaded subgraph with up to " << max_nodes << " nodes and " << g.numEdges() << " edges.\n";
        int num_partitions = world_size;
        std::cout << "Partitioning subgraph into " << num_partitions << " parts using METIS...\n";
        partitions = g.partitionWithMetis(num_partitions);
    }
    // Broadcast partition assignments to all ranks
    if (world_rank == 0) {
        MPI_Bcast(partitions.data(), max_nodes, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        partitions.resize(max_nodes);
        MPI_Bcast(partitions.data(), max_nodes, MPI_INT, 0, MPI_COMM_WORLD);
    }
    // Each rank prints the nodes it owns
    std::cout << "Rank " << world_rank << " owns nodes: ";
    for (int i = 0; i < max_nodes; ++i) {
        if (partitions[i] == world_rank) {
            std::cout << i << " ";
        }
    }
    std::cout << std::endl;
    // Gather owned nodes for this rank
    std::vector<int> owned_nodes;
    for (int i = 0; i < max_nodes; ++i) {
        if (partitions[i] == world_rank) {
            owned_nodes.push_back(i);
        }
    }
    // Parallel multi-objective shortest path computation (from source 0)
    std::vector<std::vector<std::vector<double>>> pareto_fronts;
    double t0 = MPI_Wtime();
    g.computeParetoFronts(owned_nodes, 0, pareto_fronts);
    double t1 = MPI_Wtime();
    std::cout << "Rank " << world_rank << " MSPA time: " << (t1-t0) << " seconds." << std::endl;
    // Print Pareto fronts for first 3 owned nodes
    int printed = 0;
    for (int node : owned_nodes) {
        if (printed++ >= 3) break;
        std::cout << "Rank " << world_rank << " node " << node << " Pareto front: ";
        for (const auto& path : pareto_fronts[node]) {
            std::cout << "[";
            for (size_t i = 0; i < path.size(); ++i) {
                std::cout << path[i];
                if (i + 1 < path.size()) std::cout << ",";
            }
            std::cout << "] ";
        }
        std::cout << std::endl;
    }
    // Example: Insert edge 0->604 with objectives {1.0, 1.0} if owned by this rank
    int update_from = 0, update_to = 604;
    std::vector<float> new_obj = {1.0, 1.0};
    if (partitions[update_from] == world_rank) {
        g.addEdge(update_from, update_to, {1.0, 1.0});
        // Mark update_from as updated
        g.isUpdated.assign(max_nodes, false);
        g.isUpdated[update_from] = true;
        // Incremental MOSP update: propagate from update_from
        #pragma omp parallel for
        for (int i = 0; i < max_nodes; ++i) {
            if (g.isUpdated[i]) {
                for (const auto& edge : g.neighbors(i)) {
                    Path newPath;
                    newPath.objectives = new_obj;
                    newPath.nodes = {i, edge.to};
                    if (isNonDominated(newPath, g.paretoPaths[edge.to])) {
                        g.paretoPaths[edge.to].push_back(newPath);
                        g.isUpdated[edge.to] = true;
                    }
                }
            }
        }
        // Print updated Pareto front for update_to
        std::cout << "Rank " << world_rank << " node " << update_to << " updated Pareto front: ";
        for (const auto& path : g.paretoPaths[update_to]) {
            std::cout << "[";
            for (size_t i = 0; i < path.objectives.size(); ++i) {
                std::cout << path.objectives[i];
                if (i + 1 < path.objectives.size()) std::cout << ",";
            }
            std::cout << "] ";
        }
        std::cout << std::endl;
    }
    MPI_Finalize();
    return 0;
} 