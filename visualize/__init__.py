import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

def visualize_pyg_graph(pyg_graph, subgraph_size=500, save=False, filename="subgraph.png"):
    subgraph_nodes = list(range(subgraph_size))
    subgraph_nodes = torch.tensor(subgraph_nodes, device=pyg_graph.edge_index.device)
    
    mask = torch.isin(pyg_graph.edge_index[0, :], subgraph_nodes) & \
           torch.isin(pyg_graph.edge_index[1, :], subgraph_nodes)
    
    subgraph_edge_index = pyg_graph.edge_index[:, mask]

    subgraph = nx.DiGraph()
    subgraph.add_edges_from(subgraph_edge_index.cpu().numpy().T)
    pos = nx.spring_layout(subgraph)

    plt.figure(figsize=(8, 8))
    nx.draw(subgraph, pos, node_size=10, edge_color='gray', with_labels=False)
    plt.title(f"Subgraph with {subgraph_size} Nodes")

    if save:
        os.makedirs("./figures", exist_ok=True)
        save_path = os.path.join("./figures", filename)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Subgraph visualization saved to: {save_path}")

    plt.show()


def plot_results(mc_s_trajectories, ground_truth_s, approx_exp_s, nodes, output_dir="./figures"):
    """Plot trajectories and save the figure."""
    max_length = max(
        max(len(trajectory) for trajectory in mc_s_trajectories),
        len(ground_truth_s),
        len(approx_exp_s)
    )
    mc_s_trajectories_padded = [
        trajectory + [trajectory[-1]] * (max_length - len(trajectory)) for trajectory in mc_s_trajectories
    ]
    mc_s_trajectories_padded = np.array(mc_s_trajectories_padded)

    mean_s = np.mean(mc_s_trajectories_padded, axis=0)
    lower_bound = np.percentile(mc_s_trajectories_padded, 2.5, axis=0)
    upper_bound = np.percentile(mc_s_trajectories_padded, 97.5, axis=0)

    plt.figure(figsize=(12, 8))
    for trajectory in mc_s_trajectories_padded:
        plt.plot(trajectory, color='gray', alpha=0.2)
    plt.plot(mean_s, color='blue', label='MC Mean S', linewidth=2)
    plt.fill_between(range(len(mean_s)), lower_bound, upper_bound, color='blue', alpha=0.3, label='95% Confidence Interval')
    plt.plot(ground_truth_s, color='green', label='Ground Truth S', linewidth=2)
    plt.plot(approx_exp_s, color='red', label='Approx Exp S', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Number of Susceptible Individuals')
    plt.title('SIR Model: Susceptible (S) State Trajectories, Mean, and Confidence Interval')
    plt.legend()
    plt.grid(True)
    plt.ylim([0, nodes])

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "sir_trajectories.png"))
    plt.close()


