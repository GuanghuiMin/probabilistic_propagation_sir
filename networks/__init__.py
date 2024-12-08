import torch
from torch_geometric.utils import from_networkx
import networkx as nx
from visualize import visualize_pyg_graph


def generate_pyg_graph(graph_type, nodes=1000, avg_degree=10, rewiring_prob=0.1, seed=10):
    if graph_type == "ba":  # Barabási–Albert Graph
        m = avg_degree // 2
        graph = nx.barabasi_albert_graph(nodes, m, seed=seed).to_directed()
    elif graph_type == "er":  # Erdős–Rényi Graph
        p = avg_degree / (nodes - 1)
        graph = nx.erdos_renyi_graph(nodes, p, directed=True, seed=seed)
    elif graph_type == "powerlaw":  # Power-law Graph
        graph = nx.scale_free_graph(nodes, seed=seed).to_directed()
    elif graph_type == "smallworld":  # Small-World Graph
        k = avg_degree // 2
        graph = nx.watts_strogatz_graph(nodes, k, rewiring_prob, seed=seed)
        graph = nx.DiGraph(graph)
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")

    pyg_graph = from_networkx(graph)
    pyg_graph.x = torch.eye(graph.number_of_nodes())  # Feature matrix as identity
    pyg_graph.edge_index = pyg_graph.edge_index.to('cuda')
    pyg_graph.x = pyg_graph.x.to('cuda')

    return pyg_graph


if __name__ == '__main__':
    graph_type = "ba"  # Choose from "ba", "er", "powerlaw", "smallworld"
    pyg_graph = generate_pyg_graph(graph_type, nodes=1000, avg_degree=10, rewiring_prob=0.1, seed=10)

    visualize_pyg_graph(pyg_graph, subgraph_size=50, save=True, filename="subgraph.png")