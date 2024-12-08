import torch
import torch.multiprocessing as mp
from torch_geometric.utils import from_networkx
import networkx as nx
import time
from tqdm import tqdm


def run_sir_simulation(data, beta, gamma, initial_infected_nodes, device):
    N = data.num_nodes

    susceptible = torch.ones(N, dtype=torch.bool, device=device)
    infected = torch.zeros(N, dtype=torch.bool, device=device)
    recovered = torch.zeros(N, dtype=torch.bool, device=device)

    susceptible[initial_infected_nodes] = False
    infected[initial_infected_nodes] = True

    src = data.edge_index[0].to(device)
    dst = data.edge_index[1].to(device)

    iteration_results = []
    iteration = 0

    iteration_results.append({
        'iteration': iteration,
        'status': {i: (0 if susceptible[i].item() else (1 if infected[i].item() else 2)) for i in range(N)}
    })

    while infected.any():
        iteration += 1

        recover_probs = torch.rand(N, device=device)
        newly_recovered = infected & (recover_probs < gamma)

        infected_after_recovery = infected & (~newly_recovered)
        recovered = recovered | newly_recovered

        infected_src_mask = infected_after_recovery[src]
        infected_neighbors_count = torch.zeros(N, dtype=torch.int64, device=device)
        infected_neighbors_count.scatter_add_(0, dst[infected_src_mask],
                                              torch.ones(infected_src_mask.sum(), device=device, dtype=torch.int64))

        with torch.no_grad():
            has_infected_neighbor = (infected_neighbors_count > 0) & susceptible
            neighbors_count = infected_neighbors_count[has_infected_neighbor].float()

            infection_prob = 1 - torch.pow((1 - beta), neighbors_count)

            infection_draw = torch.rand(has_infected_neighbor.sum(), device=device)
            new_infections_mask = (infection_draw < infection_prob)

            new_infections = torch.zeros(N, dtype=torch.bool, device=device)
            idx_has_infected_neighbor = has_infected_neighbor.nonzero().flatten()
            new_infections[idx_has_infected_neighbor[new_infections_mask]] = True

        infected = infected_after_recovery | new_infections
        susceptible = susceptible & (~new_infections)

        iteration_results.append({
            'iteration': iteration,
            'status': {i: (0 if susceptible[i].item() else (1 if infected[i].item() else 2)) for i in range(N)}
        })

    return iteration_results


def run_monte_carlo(data, beta, gamma, initial_infected_nodes, num_simulations, device):
    results_list = []
    mc_s_trajectories = []
    start_time = time.time()
    for _ in tqdm(range(num_simulations), desc=f"Simulations on {device}"):
        results = run_sir_simulation(data, beta, gamma, initial_infected_nodes, device)
        s_trajectory = [sum(1 for state in result['status'].values() if state == 0) for result in results]
        mc_s_trajectories.append(s_trajectory)
        results_list.append(results)
    mc_time = time.time() - start_time
    return results_list, mc_s_trajectories, mc_time


def worker(rank, world_size, data, beta, gamma, initial_infected_nodes, total_simulations, return_dict):
    device = torch.device(f'cuda:{rank}')
    data = data.to(device)
    sims_per_worker = total_simulations // world_size
    if rank == world_size - 1:
        sims_per_worker += total_simulations % world_size

    start = time.time()
    results_list, mc_s_trajectories, mc_time = run_monte_carlo(
        data, beta, gamma, initial_infected_nodes, sims_per_worker, device
    )
    end = time.time()
    elapsed = end - start
    return_dict[rank] = (results_list, mc_s_trajectories, mc_time, elapsed)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    world_size = 6
    N = 1000
    p = 0.0001

    G = nx.erdos_renyi_graph(N, p)
    data = from_networkx(G)

    beta = 0.3
    gamma = 0.1
    initial_infected_nodes = [0]
    total_simulations = 500

    manager = mp.Manager()
    return_dict = manager.dict()

    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(
        rank, world_size, data, beta, gamma, initial_infected_nodes, total_simulations, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    all_results = []
    all_trajectories = []
    total_time = 0.0
    total_elapsed = 0.0
    for rank in range(world_size):
        res_list, traj_list, mc_t, elapsed = return_dict[rank]
        all_results.extend(res_list)
        all_trajectories.extend(traj_list)
        total_time += mc_t
        total_elapsed += elapsed

    print("Monte Carlo simulations completed.")
    print(f"Total number of simulations: {len(all_results)}")
    print(f"Sum of all processes' compute times: {total_time:.2f} s")
    print(f"Wall-clock time (approx): {total_elapsed / world_size:.2f} s per worker")