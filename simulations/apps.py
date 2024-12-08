import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
import networkx as nx
import matplotlib.pyplot as plt


def approx_conditional_probability_iteration_exp(data, beta, gamma, initial_infected_nodes, tol=0.001):
    N = data.num_nodes
    status = torch.zeros((N, 3), device='cuda')

    status[initial_infected_nodes, 1] = 1
    status[:, 0] = (status.sum(dim=1) == 0).float()

    s = status[:, 0]
    i = status[:, 1]
    r = status[:, 2]

    iteration_results = []
    t = 0

    iteration_results.append({
        'iteration': t,
        'susceptible': s.clone().cpu().tolist(),
        'infected': i.clone().cpu().tolist(),
        'recovered': r.clone().cpu().tolist()
    })

    while i.max().item() > tol:
        t += 1

        edge_src, edge_dst = data.edge_index
        infective_contributions = torch.sparse_coo_tensor(
            torch.stack([edge_dst, edge_src]),
            i[edge_src],
            (N, N),
            device='cuda'
        ).to_dense().sum(dim=1)

        s_new = s * torch.exp(-beta * infective_contributions)
        i_new = s - s_new + i * (1 - gamma)
        r_new = r + i * gamma

        s, i, r = s_new, i_new, r_new

        iteration_results.append({
            'iteration': t,
            'susceptible': s.clone().cpu().tolist(),
            'infected': i.clone().cpu().tolist(),
            'recovered': r.clone().cpu().tolist()
        })

    return iteration_results


def plot_sir_trends(results):
    iterations = [result['iteration'] for result in results]

    S_counts = [sum(result['susceptible']) for result in results]
    I_counts = [sum(result['infected']) for result in results]
    R_counts = [sum(result['recovered']) for result in results]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, S_counts, label='Susceptible (S)', color='blue')
    plt.plot(iterations, I_counts, label='Infected (I)', color='red')
    plt.plot(iterations, R_counts, label='Recovered (R)', color='green')
    plt.xlabel('Iterations')
    plt.ylabel('Number of individuals (Expected)')
    plt.title('SIR Model: Expected Number of Individuals in Each State')
    plt.legend()
    plt.grid(True)
    plt.show()


def test_conditional_probability_iteration_estimation():
    G = nx.barabasi_albert_graph(100, 2).to_directed()
    data = from_networkx(G).to('cuda')

    beta = 0.03
    gamma = 0.01
    initial_infected_nodes = torch.tensor([0, 1], device='cuda')
    tol = 0.001

    results = approx_conditional_probability_iteration_exp(data, beta, gamma, initial_infected_nodes, tol)

    plot_sir_trends(results)


if __name__ == '__main__':
    test_conditional_probability_iteration_estimation()