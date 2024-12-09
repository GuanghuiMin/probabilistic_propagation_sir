import torch
from torch_geometric.data import Data

def load_external_dataset(dataset_path):
    edge_list = []
    with open(dataset_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            u, v = map(int, line.split())
            # 如果需要从1开始编号改为0开始：
            # u, v = u - 1, v - 1
            edge_list.append([u, v])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    num_nodes = edge_index.max().item() + 1
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    return data

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



def approximate_largest_eigenvalue(edge_index, num_nodes, num_iters=100, device='cuda'):
    """Approximate largest eigenvalue of the adjacency matrix using power iteration."""
    values = torch.ones(edge_index.size(1), dtype=torch.float32, device=device)
    A = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes), device=device).coalesce()


    x = torch.randn(num_nodes, device=device)
    x = x / torch.norm(x)

    for _ in range(num_iters):
        y = torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)

        norm_y = torch.norm(y)
        if norm_y == 0:
            return 0.0
        x = y / norm_y

    largest_eigenvalue = norm_y.item()
    return largest_eigenvalue


def calculate_graph_properties(data):
    """Calculate graph properties: max in-degree, 2-norm, spectral radius using approximations."""
    device = data.edge_index.device

    max_in_degree = data.edge_index[1].bincount(minlength=data.num_nodes).max().item()

    largest_eigenvalue = approximate_largest_eigenvalue(data.edge_index, data.num_nodes, num_iters=100, device=device)

    two_norm = largest_eigenvalue
    spectral_radius = largest_eigenvalue

    return max_in_degree, two_norm, spectral_radius