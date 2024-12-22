import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from tqdm import tqdm


def run_sir_simulation(G, beta, gamma, initial_infected_nodes):
    N = G.number_of_nodes()
    susceptible = np.ones(N, dtype=bool)
    infected = np.zeros(N, dtype=bool)
    recovered = np.zeros(N, dtype=bool)

    susceptible[initial_infected_nodes] = False
    infected[initial_infected_nodes] = True

    iteration_results = []
    iteration = 0


    iteration_results.append({
        'iteration': iteration,
        'status': {i: (0 if susceptible[i] else 1 if infected[i] else 2) for i in range(N)}
    })

    while infected.sum() > 0:
        iteration += 1
        new_infected = infected.copy()
        new_recovered = recovered.copy()

        for node in range(N):
            if infected[node]:
                if random.random() < gamma:
                    new_infected[node] = False
                    new_recovered[node] = True
            elif susceptible[node]:
                neighbors = list(G.predecessors(node))
                infective_neighbors = sum(infected[neighbor] for neighbor in neighbors)
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
            'status': {i: (0 if susceptible[i] else 1 if infected[i] else 2) for i in range(N)}
        })

    return iteration_results


# visualization
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


# test
def test_sir_simulation():
    G = nx.barabasi_albert_graph(100, 3)
    G = G.to_directed()

    beta = 0.03
    gamma = 0.01
    initial_infected_nodes = random.sample(list(G.nodes), 5)

    results = run_sir_simulation(G, beta, gamma, initial_infected_nodes)

    S = []
    I = []
    R = []
    for result in results:
        status = result['status']
        S.append(sum(1 for state in status.values() if state == 0))
        I.append(sum(1 for state in status.values() if state == 1))
        R.append(sum(1 for state in status.values() if state == 2))

    plot_sir(S, I, R)

def run_monte_carlo_simulations(graph, beta, gamma, initial_infected_nodes, num_simulations):
    results_list, mc_trajectories = [], []
    print("Running Monte Carlo simulations...")

    start_time = time.time()  # Start timing
    for _ in tqdm(range(num_simulations)):
        results = run_sir_simulation(graph, beta, gamma, initial_infected_nodes)
        mc_trajectories.append(
            [sum(1 for state in result['status'].values() if state == 0) for result in results]
        )
        results_list.append(results)
    end_time = time.time()  # End timing

    total_time = end_time - start_time  # Total runtime
    print(f"Monte Carlo simulations complete. Total time: {total_time:.2f} seconds.")
    return results_list, mc_trajectories, total_time


if __name__ == '__main__':
    test_sir_simulation()