import os
import pandas as pd
import networkx as nx

def save_args_to_csv(args, output_dir):
    """Save all arguments to a CSV file."""
    args_dict = vars(args)
    args_df = pd.DataFrame(args_dict.items(), columns=["Argument", "Value"])
    args_df.to_csv(os.path.join(output_dir, "args_settings.csv"), index=False)
    print(f"Arguments saved to: {os.path.join(output_dir, 'args_settings.csv')}")


import networkx as nx
import random
import warnings


def calculate_centralities(graph,
                           centralities_to_compute=["Degree", "Eigenvector", "Katz", "PageRank", "Betweenness",
                                                    "Closeness"],
                           approx_betweenness=True,
                           betweenness_samples=100,
                           use_approx_for_closeness=True,
                           closeness_samples=100):
    centralities = {}

    # Check for directed or disconnected graphs
    if graph.is_directed():
        warnings.warn("Graph is directed; certain centralities may not be meaningful.")

    if not nx.is_connected(graph):
        warnings.warn("Graph is disconnected; closeness and betweenness centralities may not be accurate.")

    # Degree Centrality
    if "Degree" in centralities_to_compute:
        degree_centrality = nx.degree_centrality(graph)
        centralities["Degree"] = degree_centrality

    # Eigenvector Centrality
    if "Eigenvector" in centralities_to_compute:
        eigenvector_centrality = nx.eigenvector_centrality(
            graph,
            max_iter=1000,
            tol=1e-06,
            weight=None
        )
        centralities["Eigenvector"] = eigenvector_centrality

    # Katz Centrality
    if "Katz" in centralities_to_compute:
        katz_centrality = nx.katz_centrality(
            graph,
            alpha=0.1,
            beta=1.0,
            max_iter=1000,
            tol=1e-06
        )
        centralities["Katz"] = katz_centrality

    # PageRank
    if "PageRank" in centralities_to_compute:
        pagerank = nx.pagerank(
            graph,
            alpha=0.85,
            max_iter=1000,
            tol=1e-06
        )
        centralities["PageRank"] = pagerank

    # Betweenness Centrality
    if "Betweenness" in centralities_to_compute:
        if approx_betweenness:
            betweenness_centrality = nx.approximate_betweenness_centrality(
                graph,
                k=betweenness_samples,
                normalized=True
            )
        else:
            betweenness_centrality = nx.betweenness_centrality(graph)
        centralities["Betweenness"] = betweenness_centrality

    # Closeness Centrality
    if "Closeness" in centralities_to_compute:
        if use_approx_for_closeness:
            sample_nodes = random.sample(graph.nodes(), min(closeness_samples, len(graph)))
            closeness_centrality = {}
            for node in graph.nodes():
                total_distance = 0
                count = 0
                for sample in sample_nodes:
                    try:
                        distance = nx.shortest_path_length(graph, source=sample, target=node)
                        total_distance += distance
                        count += 1
                    except nx.NetworkXNoPath:
                        continue
                if count > 0:
                    average_distance = total_distance / count
                    closeness_centrality[node] = 1.0 / average_distance
                else:
                    closeness_centrality[node] = 0.0
        else:
            closeness_centrality = nx.closeness_centrality(graph)
        centralities["Closeness"] = closeness_centrality

    return centralities