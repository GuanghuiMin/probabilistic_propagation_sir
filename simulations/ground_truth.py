import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def conditional_probability_iteration(G, beta, gamma, initial_infected_nodes, tol=0.001):
    N = G.number_of_nodes()
    status = np.zeros((N, 3))

    for node in G.nodes():
        if node in initial_infected_nodes:
            status[node, 1] = 1
        else:
            status[node, 0] = 1

    s = status[:, 0]
    i = status[:, 1]
    r = status[:, 2]

    iteration_results = []  # 存储每次迭代的结果
    t = 0  # 初始化迭代次数

    iteration_results.append({
        'iteration': t,
        'susceptible': {node: s[node] for node in G.nodes()},
        'infected': {node: i[node] for node in G.nodes()},
        'recovered': {node: r[node] for node in G.nodes()}
    })

    while max(i) > tol:
        t += 1
        for node in G.nodes():
            in_neighbors = list(G.predecessors(node))  # 获取所有入邻居节点
            s_new = np.prod([1 - beta * i[x] for x in in_neighbors]) * s[node]  # 计算新的易感概率
            i_new = s[node] - s_new + i[node] * (1 - gamma)  # 计算新的感染概率
            r_new = r[node] + i[node] * gamma  # 计算恢复概率

            s[node] = s_new
            i[node] = i_new
            r[node] = r_new

        iteration_results.append({
            'iteration': t,
            'susceptible': {node: s[node] for node in G.nodes()},
            'infected': {node: i[node] for node in G.nodes()},
            'recovered': {node: r[node] for node in G.nodes()}
        })

    return iteration_results

# visualization
def plot_sir_trends(results, G):
    iterations = [result['iteration'] for result in results]

    S_counts = []
    I_counts = []
    R_counts = []

    for result in results:
        S_counts.append(sum(result['susceptible'].values()))
        I_counts.append(sum(result['infected'].values()))
        R_counts.append(sum(result['recovered'].values()))

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

# test
def test_conditional_probability_iteration():
    G = nx.barabasi_albert_graph(100, 2).to_directed()

    beta = 0.05
    gamma = 0.3
    initial_infected_nodes = [0, 1]
    tol = 1e-3

    results = conditional_probability_iteration(G, beta, gamma, initial_infected_nodes, tol)

    for result in results:
        print(f"Iteration {result['iteration']}:")
        print(f"  Susceptible: {result['susceptible']}")
        print(f"  Infected: {result['infected']}")
        print(f"  Recovered: {result['recovered']}\n")

    plot_sir_trends(results, G)

    print(f"lower bound: {np.log(tol)/np.log(1-gamma)}")
    print(f"max in_degree: {max(dict(G.in_degree()).values())}")
    print(f"diameter: {nx.diameter(G)}")
    print(-result['iteration']/np.log(gamma))

if __name__ == '__main__':
    test_conditional_probability_iteration()