import torch
from torch_geometric.utils import from_networkx
import matplotlib.pyplot as plt
import random
import networkx as nx


def run_sir_simulation(data, beta, gamma, initial_infected_nodes):
    N = data.num_nodes
    susceptible = torch.ones(N, dtype=torch.bool, device='cuda')
    infected = torch.zeros(N, dtype=torch.bool, device='cuda')
    recovered = torch.zeros(N, dtype=torch.bool, device='cuda')

    susceptible[initial_infected_nodes] = False
    infected[initial_infected_nodes] = True

    iteration_results = []
    iteration = 0

    iteration_results.append({
        'iteration': iteration,
        'status': {i: (0 if susceptible[i].item() else 1 if infected[i].item() else 2) for i in range(N)}
    })

    while infected.sum().item() > 0:
        iteration += 1
        new_infected = infected.clone()
        new_recovered = recovered.clone()

        for node in range(N):
            if infected[node]:
                if random.random() < gamma:
                    new_infected[node] = False
                    new_recovered[node] = True
            elif susceptible[node]:
                neighbors = data.edge_index[0][data.edge_index[1] == node]
                infective_neighbors = infected[neighbors].sum().item()
                if infective_neighbors >= 1:
                    for _ in range(infective_neighbors):
                        if random.random() < beta:
                            new_infected[node] = True
                            susceptible[node] = False
                            break

        infected = new_infected
        recovered = new_recovered

        iteration_results.append({
            'iteration': iteration,
            'status': {i: (0 if susceptible[i].item() else 1 if infected[i].item() else 2) for i in range(N)}
        })

    return iteration_results


def plot_sir(S, I, R):
    plt.figure(figsize=(10, 6))
    plt.plot(S, label='Susceptible (S)', color='blue')
    plt.plot(I, label='Infected (I)', color='red')
    plt.plot(R, label='Recovered (R)', color='green')
    plt.xlabel('Iterations')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model Simulation')
    plt.legend()
    plt.grid(True)
    plt.show()


def test_sir_simulation():
    G = nx.barabasi_albert_graph(100, 3).to_directed()
    data = from_networkx(G)
    data = data.to('cuda')

    beta = 0.03
    gamma = 0.01
    initial_infected_nodes = random.sample(range(data.num_nodes), 5)

    results = run_sir_simulation(data, beta, gamma, initial_infected_nodes)

    S, I, R = [], [], []
    for result in results:
        status = result['status']
        S.append(sum(1 for state in status.values() if state == 0))
        I.append(sum(1 for state in status.values() if state == 1))
        R.append(sum(1 for state in status.values() if state == 2))

    plot_sir(S, I, R)


if __name__ == '__main__':
    test_sir_simulation()