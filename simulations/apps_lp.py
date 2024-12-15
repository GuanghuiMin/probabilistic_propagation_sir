import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_scatter import scatter

def build_neighbor_structures(edge_index, num_nodes, device):
    edge_index_cpu = edge_index.cpu()
    src, dst = edge_index_cpu[0], edge_index_cpu[1]

    # In-neighbors
    in_degree = torch.zeros(num_nodes, dtype=torch.long)
    in_degree.index_add_(0, dst, torch.ones_like(dst))
    in_ptr = torch.zeros(num_nodes+1, dtype=torch.long)
    in_ptr[1:] = torch.cumsum(in_degree, dim=0)
    in_indices = torch.zeros(in_ptr[-1], dtype=torch.long)
    idx_pos = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(dst.size(0)):
        d = dst[i].item()
        pos = in_ptr[d] + idx_pos[d]
        in_indices[pos] = src[i]
        idx_pos[d] += 1

    # Out-neighbors
    out_degree = torch.zeros(num_nodes, dtype=torch.long)
    out_degree.index_add_(0, src, torch.ones_like(src))
    out_ptr = torch.zeros(num_nodes+1, dtype=torch.long)
    out_ptr[1:] = torch.cumsum(out_degree, dim=0)
    out_indices = torch.zeros(out_ptr[-1], dtype=torch.long)
    idx_pos = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(src.size(0)):
        s = src[i].item()
        pos = out_ptr[s] + idx_pos[s]
        out_indices[pos] = dst[i]
        idx_pos[s] += 1

    return in_ptr.to(device), in_indices.to(device), out_ptr.to(device), out_indices.to(device)

def approx_conditional_probability_iteration_local_push(G, beta, gamma, initial_infected_nodes, tol=0.001, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    N = G.num_nodes
    edge_index = G.edge_index.to(device)
    in_ptr, in_indices, out_ptr, out_indices = build_neighbor_structures(edge_index, N, device)

    S = torch.ones(N, device=device)
    I = torch.zeros(N, device=device)
    R = torch.zeros(N, device=device)
    initial_infected_nodes = torch.tensor(initial_infected_nodes, dtype=torch.long, device=device)
    S[initial_infected_nodes] = 0
    I[initial_infected_nodes] = 1

    residuals = torch.zeros(N, device=device)
    initial_res = beta * I[edge_index[0]]
    scatter(initial_res, edge_index[1], out=residuals, reduce='add')

    iteration_results = []
    t = 0
    iteration_results.append({
        'iteration': t,
        'susceptible': S.clone().cpu().tolist(),
        'infected': I.clone().cpu().tolist(),
        'recovered': R.clone().cpu().tolist()
    })

    while True:
        t += 1
        updated_this_round = False

        while True:
            max_val, v = torch.max(residuals, dim=0)
            max_val = max_val.item()
            v = v.item()
            if max_val <= tol:
                break

            updated_this_round = True
            v_t = torch.tensor([v], device=device, dtype=torch.long)
            # sum_m_v = sum of beta * I[u] for in_neighbors of v
            start_in = in_ptr[v]
            end_in = in_ptr[v+1]
            in_neigh = in_indices[start_in:end_in]
            sum_m_v = beta * torch.sum(I[in_neigh])

            S_prev = S[v_t]
            I_prev = I[v_t]

            S_new = S_prev * torch.exp(-sum_m_v)
            I_new = S_prev - S_new + I_prev * (1 - gamma)
            R_new = R[v_t] + I_prev * gamma

            diff_s = torch.abs(S_new - S_prev)
            diff_i = torch.abs(I_new - I_prev)

            if (diff_s > tol) or (diff_i > tol):
                S[v] = S_new
                I[v] = I_new
                R[v] = R_new

                delta_I = (I[v_t] - I_prev).item()
                if abs(delta_I) > 0:
                    add_val = beta * abs(delta_I)
                    start_out = out_ptr[v]
                    end_out = out_ptr[v+1]
                    out_neigh = out_indices[start_out:end_out]
                    residuals[out_neigh] += add_val

                residuals[v] = 0.0
            else:
                residuals[v] = 0.0

        iteration_results.append({
            'iteration': t,
            'susceptible': S.clone().cpu().numpy().tolist(),
            'infected': I.clone().cpu().numpy().tolist(),
            'recovered': R.clone().cpu().numpy().tolist()
        })

        if not updated_this_round:
            break

    return iteration_results

def plot_sir_trends(results):
    iterations = [res['iteration'] for res in results]
    S_counts = [np.sum(res['susceptible']) for res in results]
    I_counts = [np.sum(res['infected']) for res in results]
    R_counts = [np.sum(res['recovered']) for res in results]

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
    from torch_geometric.utils import barabasi_albert_graph
    edge_index = barabasi_albert_graph(num_nodes=100, num_edges=2)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    G = Data(edge_index=edge_index, num_nodes=100)

    beta = 0.03
    gamma = 0.01
    initial_infected_nodes = [0, 1]
    tol = 0.001

    results = approx_conditional_probability_iteration_local_push(
        G, beta, gamma, initial_infected_nodes, tol, device='cuda'
    )

    for result in results:
        print(f"Iteration {result['iteration']}:")
        print("  Susceptible:", result['susceptible'])
        print("  Infected:", result['infected'])
        print("  Recovered:", result['recovered'], "\n")

    plot_sir_trends(results)

if __name__ == '__main__':
    test_conditional_probability_iteration_estimation()