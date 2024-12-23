import networkx as nx
import matplotlib.pyplot as plt
import gzip
from enum import Enum
import os

def generate_graph(graph_type, nodes=1000, avg_degree=10, rewiring_prob=0.1, seed=10, file_path=None):
    if graph_type == "ba":
        m = avg_degree // 2
        graph = nx.barabasi_albert_graph(nodes, m, seed=seed).to_directed()
    elif graph_type == "er":
        p = avg_degree / (nodes - 1)
        graph = nx.erdos_renyi_graph(nodes, p=p, directed=True, seed=seed)
    elif graph_type == "powerlaw":
        graph = nx.scale_free_graph(nodes, seed=seed).to_directed()
    elif graph_type == "smallworld":
        k = avg_degree // 2
        graph = nx.watts_strogatz_graph(nodes, k, rewiring_prob, seed=seed)
        graph = nx.DiGraph(graph)
    else:
        raise ValueError("Invalid graph type or missing parameters.")
    return graph

def load_real_data(data_dir, file_name):
    G = nx.DiGraph()
    file_path = os.path.join(data_dir, file_name)
    print(f"Loading dataset from {data_dir}/{file_name}...")
    with gzip.open(file_path, 'rt') as f:
        if file_name=="com-friendster.top5000.cmty.txt.gz":
            for line in f:
                # Parse each line as a list of node IDs (representing a community)
                nodes = list(map(int, line.strip().split()))
                # Add edges between every pair of nodes in the community
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        G.add_edge(nodes[i], nodes[j])  # Directed edge
                        G.add_edge(nodes[j], nodes[i])  # Reverse edge for bidirectionality
        else:
            with gzip.open(file_path, 'rt') as f:
                for line in f:
                    if line.startswith('#'):  # Skip comments
                        continue
                    source, target = map(int, line.strip().split())
                    G.add_edge(source, target)
    print("Dataset loaded.")
    G = nx.convert_node_labels_to_integers(G, label_attribute="original_id")
    return G


def visualize_graph(graph, title="Directed Graph", nodes_to_draw=500):
    subgraph = graph.subgraph(list(graph.nodes)[:nodes_to_draw])
    plt.figure(figsize=(8, 8))
    nx.draw(subgraph, node_size=10, edge_color='gray', with_labels=False)
    plt.title(f"{title} ({nodes_to_draw} Nodes Subgraph)")
    plt.show()



if __name__ == '__main__':
    graph3 = generate_graph("smallworld", nodes=2000,avg_degree=4, seed=10)
    visualize_graph(graph3, title="Power-law Directed Graph")

