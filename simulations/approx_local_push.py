import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import heapq


def approx_conditional_probability_iteration_local_push(G, beta, gamma, initial_infected_nodes, tol=0.001):
    N = G.number_of_nodes()
    S = np.ones(N)
    I = np.zeros(N)
    R = np.zeros(N)
    for node in initial_infected_nodes:
        S[node] = 0
        I[node] = 1

    # Precompute in-neighbors for each node
    in_neighbors = {v: list(G.predecessors(v)) for v in G.nodes()}
    out_neighbors = {v: list(G.successors(v)) for v in G.nodes()}

    # Initialize residuals
    residuals = {v: 0.0 for v in G.nodes()}
    for u in initial_infected_nodes:
        for v in out_neighbors[u]:
            residuals[v] += beta * I[u]

    # Use a heap to prioritize nodes with larger residuals
    heap = []
    for v in G.nodes():
        if residuals[v] > tol:
            # Use negative residual for max-heap effect
            heapq.heappush(heap, (-residuals[v], v))

    iteration_results = []
    t = 0

    iteration_results.append({
        'iteration': t,
        'susceptible': {node: S[node] for node in G.nodes()},
        'infected': {node: I[node] for node in G.nodes()},
        'recovered': {node: R[node] for node in G.nodes()}
    })

    while heap:
        t += 1
        # Process nodes in order of decreasing residual
        while heap:
            neg_residual, v = heapq.heappop(heap)
            residual = -neg_residual
            if residual <= tol:
                continue  # Skip if residual is below tolerance
            # Compute new S, I, R
            S_prev = S[v]
            I_prev = I[v]
            # Compute sum_m[v] from incoming messages
            sum_m_v = 0.0
            for u in in_neighbors[v]:
                sum_m_v += beta * I[u]
            S_new = S_prev * np.exp(-sum_m_v)
            I_new = S_prev - S_new + I_prev * (1 - gamma)
            R_new = R[v] + I_prev * gamma
            # Check for convergence
            if np.abs(S_new - S[v]) > tol or np.abs(I_new - I[v]) > tol:
                # Update states
                S[v] = S_new
                I[v] = I_new
                R[v] = R_new
                # Compute residual for neighboring nodes
                for u in out_neighbors[v]:
                    residuals[u] += beta * np.abs(I[v] - I_prev)
                    if residuals[u] > tol:
                        heapq.heappush(heap, (-residuals[u], u))
                # Reset residual for current node
                residuals[v] = 0.0
            else:
                # If converged, set residual to zero
                residuals[v] = 0.0

        iteration_results.append({
            'iteration': t,
            'susceptible': {node: S[node] for node in G.nodes()},
            'infected': {node: I[node] for node in G.nodes()},
            'recovered': {node: R[node] for node in G.nodes()}
        })

    return iteration_results

def plot_sir_trends(results, G):
    iterations = [result['iteration'] for result in results]
    S_counts = [sum(result['susceptible'].values()) for result in results]
    I_counts = [sum(result['infected'].values()) for result in results]
    R_counts = [sum(result['recovered'].values()) for result in results]

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
    beta = 0.03
    gamma = 0.01
    initial_infected_nodes = [0, 1]
    tol = 0.001

    results = approx_conditional_probability_iteration_local_push(G, beta, gamma, initial_infected_nodes, tol)

    for result in results:
        print(f"Iteration {result['iteration']}:")
        print(f"  Susceptible: {result['susceptible']}")
        print(f"  Infected: {result['infected']}")
        print(f"  Recovered: {result['recovered']}\n")

    plot_sir_trends(results, G)

if __name__ == '__main__':
    test_conditional_probability_iteration_estimation()