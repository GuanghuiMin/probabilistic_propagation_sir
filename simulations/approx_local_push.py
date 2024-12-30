import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import heapq

def global_preheat_steps(S, I, R, in_neighbors, beta, gamma, steps=3):
    """
    Do 'steps' rounds of global SIR-like updates (exponential approximation),
    to warm up the state before LocalPush.
    """
    N = len(S)
    for _ in range(steps):
        S_new = S.copy()
        I_new = I.copy()
        R_new = R.copy()
        for v in range(N):
            sum_in_I = 0.0
            for u in in_neighbors[v]:
                sum_in_I += I[u]
            s_prev = S[v]
            i_prev = I[v]
            s_cur = s_prev * np.exp(-beta * sum_in_I)
            i_cur = (s_prev - s_cur) + i_prev * (1 - gamma)
            r_cur = R[v] + i_prev * gamma
            S_new[v] = s_cur
            I_new[v] = i_cur
            R_new[v] = r_cur
        S[:] = S_new
        I[:] = I_new
        R[:] = R_new

def approx_conditional_probability_iteration_local_push(
    G, beta, gamma, initial_infected_nodes, tol=0.001,
    preheat_steps=10, num_substeps=10
):
    """
    LocalPush with multi-step update:
      - First, do 'preheat_steps' global updates to warm up distant nodes.
      - Then, in each pop from the heap, do 'num_substeps' smaller updates
        instead of one big step.
    """
    N = G.number_of_nodes()
    S = np.ones(N)
    I = np.zeros(N)
    R = np.zeros(N)
    for node in initial_infected_nodes:
        S[node] = 0
        I[node] = 1

    in_neighbors = {v: list(G.predecessors(v)) for v in G.nodes()}
    out_neighbors = {v: list(G.successors(v)) for v in G.nodes()}

    # 1) 全局预热
    global_preheat_steps(S, I, R, in_neighbors, beta, gamma, steps=preheat_steps)

    # 2) 根据预热状态初始化 residuals
    residuals = {v: 0.0 for v in G.nodes()}
    for u in G.nodes():
        if I[u] > 0:
            for w in out_neighbors[u]:
                residuals[w] += beta * I[u]

    in_heap = {v: False for v in G.nodes()}
    heap = []

    def push_node(u):
        if residuals[u] > tol and not in_heap[u]:
            in_heap[u] = True
            heapq.heappush(heap, (-residuals[u], u))

    # 将初始 residuals 超过 tol 的节点放到堆中
    for v in G.nodes():
        if residuals[v] > tol:
            in_heap[v] = True
            heapq.heappush(heap, (-residuals[v], v))

    iteration_results = []
    t = 0
    iteration_results.append({
        'iteration': t,
        'susceptible': {node: S[node] for node in G.nodes()},
        'infected': {node: I[node] for node in G.nodes()},
        'recovered': {node: R[node] for node in G.nodes()}
    })

    beta_sub = beta / num_substeps
    gamma_sub = gamma / num_substeps

    while heap:
        t += 1
        while heap:
            neg_residual, v = heapq.heappop(heap)
            in_heap[v] = False
            residual_val = -neg_residual
            if residual_val <= tol:
                continue

            S_prev = S[v]
            I_prev = I[v]
            R_prev = R[v]

            sum_in_I_v = 0.0
            for u in in_neighbors[v]:
                sum_in_I_v += I[u]

            S_temp = S_prev
            I_temp = I_prev
            R_temp = R_prev
            for _ in range(num_substeps):
                s_new = S_temp * np.exp(-beta_sub * sum_in_I_v)
                i_new = (S_temp - s_new) + I_temp * (1 - gamma_sub)
                r_new = R_temp + I_temp * gamma_sub
                S_temp, I_temp, R_temp = s_new, i_new, r_new

            S_new = S_temp
            I_new = I_temp
            R_new = R_temp

            # 如有明显变化，再向邻居推残差
            if (abs(S_new - S[v]) > tol) or (abs(I_new - I[v]) > tol):
                S[v] = S_new
                I[v] = I_new
                R[v] = R_new
                delta_i = abs(I[v] - I_prev)

                for nbr in out_neighbors[v]:
                    old_val = residuals[nbr]
                    residuals[nbr] += beta * delta_i  # note: 这里仍是 beta
                    if residuals[nbr] > tol and (residuals[nbr] > old_val):
                        push_node(nbr)
                residuals[v] = 0.0
            else:
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

    results = approx_conditional_probability_iteration_local_push(
        G, beta, gamma, initial_infected_nodes, tol,
        preheat_steps=5, num_substeps=3
    )
    for result in results:
        print(f"Iteration {result['iteration']}:")
        print(f"  Susceptible: {result['susceptible']}")
        print(f"  Infected: {result['infected']}")
        print(f"  Recovered: {result['recovered']}\n")

    plot_sir_trends(results, G)

if __name__ == '__main__':
    test_conditional_probability_iteration_estimation()