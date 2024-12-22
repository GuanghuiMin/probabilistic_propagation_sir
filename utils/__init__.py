import os
import pandas as pd
import networkx as nx

def save_args_to_csv(args, output_dir):
    """Save all arguments to a CSV file."""
    args_dict = vars(args)
    args_df = pd.DataFrame(args_dict.items(), columns=["Argument", "Value"])
    args_df.to_csv(os.path.join(output_dir, "args_settings.csv"), index=False)
    print(f"Arguments saved to: {os.path.join(output_dir, 'args_settings.csv')}")


def calculate_centralities(graph):
    degree_centrality = nx.degree_centrality(graph)
    eigenvector_centrality = nx.eigenvector_centrality_numpy(graph)
    katz_centrality = nx.katz_centrality_numpy(graph, alpha=0.1)
    pagerank = nx.pagerank(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    closeness_centrality = nx.closeness_centrality(graph)

    return {
        "Degree": degree_centrality,
        "Eigenvector": eigenvector_centrality,
        "Katz": katz_centrality,
        "PageRank": pagerank,
        "Betweenness": betweenness_centrality,
        "Closeness": closeness_centrality,
    }