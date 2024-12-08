import torch


def calculate_infection_probability(results_list, num_nodes):
    infection_counts = torch.zeros(num_nodes, device='cuda')
    num_simulations = len(results_list)

    for results in results_list:
        for result in results:
            for node, status in result['status'].items():
                if status == 1 or status == 2:
                    infection_counts[node] += 1

    infection_probability = infection_counts / num_simulations
    return infection_probability


def calculate_graph_properties(data):
    """Calculate graph properties: max in-degree, 2-norm, spectral radius."""
    adj_matrix = torch.sparse_coo_tensor(
        data.edge_index,
        torch.ones(data.edge_index.size(1), device='cuda'),
        (data.num_nodes, data.num_nodes),
        device='cuda'
    ).to_dense()
    max_in_degree = data.edge_index[1].bincount(minlength=data.num_nodes).max().item()
    two_norm = torch.linalg.norm(adj_matrix, ord=2).item()
    eigenvalues = torch.linalg.eigvals(adj_matrix)
    spectral_radius = eigenvalues.abs().max().item()
    return max_in_degree, two_norm, spectral_radius