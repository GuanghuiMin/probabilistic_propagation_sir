import argparse
import os
import torch
import torch.multiprocessing as mp
import time
import numpy as np
import pandas as pd
from networks import generate_pyg_graph
from simulations.mc_multiprocess import worker
from utils import calculate_graph_properties, calculate_infection_probability, load_external_dataset
from eval import compare_mc_and_ranking
from visualize import plot_results
from simulations.pps import conditional_probability_iteration
from simulations.apps import approx_conditional_probability_iteration_exp
from simulations.apps_lp import approx_conditional_probability_iteration_local_push
from simulations.aapps import sor_approx_conditional_probability_iteration_exp


def main(args):
    if args.dataset.strip() == "":
        data = generate_pyg_graph(args.graph_type, nodes=args.nodes)
        print(
            f"Generated a random {args.graph_type} graph with {args.nodes} nodes and {data.edge_index.size(1)} edges.")
    else:
        data = load_external_dataset(args.dataset)
    num_nodes = data.num_nodes
    initial_infected_nodes = torch.tensor(
        np.random.choice(num_nodes, args.initial_infected, replace=False), device='cuda'
    )

    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    avg_degree = num_edges / num_nodes
    # max_in_degree, two_norm, spectral_radius = calculate_graph_properties(data)

    # print(f"beta/gamma: {args.beta/args.gamma}, if epidemic threshold: {args.beta/args.gamma < 1/two_norm}, if error converges: {args.beta/args.gamma < 1/max_in_degree}")
    # print(f"Graph properties: avg degree: {avg_degree}, max in-degree: {max_in_degree}, 2-norm: {two_norm}, spectral_radius: {spectral_radius}")

    mp.set_start_method("spawn", force=True)
    world_size = 6
    manager = mp.Manager()
    return_dict = manager.dict()

    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(
        rank, world_size, data, args.beta, args.gamma, initial_infected_nodes, args.num_simulations, return_dict))
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
    print(f"Wall-clock time approx: {total_elapsed / world_size:.2f} s per worker")
    mc_time = total_time
    infection_probability = calculate_infection_probability(all_results, data.num_nodes)
    mc_convergence_steps = np.mean([len(trajectory) for trajectory in all_trajectories])

    start_time = time.time()
    ground_truth = conditional_probability_iteration(data, args.beta, args.gamma, initial_infected_nodes, args.tol)
    gt_time = time.time() - start_time

    start_time = time.time()
    approx_exp = approx_conditional_probability_iteration_exp(data, args.beta, args.gamma, initial_infected_nodes,
                                                              args.tol)
    approx_time = time.time() - start_time

    start_time = time.time()
    sor_exp = sor_approx_conditional_probability_iteration_exp(data, args.beta, args.gamma, initial_infected_nodes,
                                                               args.tol, args.omega)
    sor_time = time.time() - start_time

    start_time = time.time()
    local_push_exp = approx_conditional_probability_iteration_local_push(data, args.beta, args.gamma,
                                                                         initial_infected_nodes, args.tol)
    local_push_time = time.time() - start_time

    ground_truth_s = [sum(result['susceptible']) for result in ground_truth]
    approx_exp_s = [sum(result['susceptible']) for result in approx_exp]
    sor_exp_s = [sum(result['susceptible']) for result in sor_exp]
    local_push_exp_s = [sum(result['susceptible']) for result in local_push_exp]

    gt_tau, gt_p_value = compare_mc_and_ranking(ground_truth, infection_probability)
    approx_tau, approx_p_value = compare_mc_and_ranking(approx_exp, infection_probability)
    sor_tau, sor_p_value = compare_mc_and_ranking(sor_exp, infection_probability)
    local_push_tau, local_push_p_value = compare_mc_and_ranking(local_push_exp, infection_probability)

    gt_final_s = ground_truth_s[-1]
    approx_final_s = approx_exp_s[-1]
    sor_final_s = sor_exp_s[-1]
    local_push_final_s = local_push_exp_s[-1]

    mc_final_s = [trajectory[-1] if isinstance(trajectory[-1], (int, float)) else sum(trajectory[-1]) for trajectory in
                  all_trajectories]
    mc_s_estimate = np.mean(mc_final_s)

    mc_s_lower_ci = np.percentile(mc_final_s, 2.5)
    mc_s_upper_ci = np.percentile(mc_final_s, 97.5)

    gt_percentile = np.sum(np.array(mc_final_s) < gt_final_s) / len(mc_final_s) * 100
    approx_percentile = np.sum(np.array(mc_final_s) < approx_final_s) / len(mc_final_s) * 100
    sor_percentile = np.sum(np.array(mc_final_s) < sor_final_s) / len(mc_final_s) * 100
    local_push_percentile = np.sum(np.array(mc_final_s) < local_push_final_s) / len(mc_final_s) * 100

    plot_results(all_trajectories, ground_truth_s, approx_exp_s, sor_exp_s, local_push_exp_s, num_nodes,
                 args.output_dir)

    data_dict = {
        "Metric": [
            "Monte Carlo Time (s)",
            "Ground Truth Time (s)",
            "Approximation Time (s)",
            "SOR Time (s)",
            "Local Push Approx Time (s)",
            "MC Average Convergence Steps",
            "GT Convergence Steps",
            "Approx Convergence Steps",
            "SOR Convergence Steps",
            "Local Push Convergence Steps",
            "GT Final S Percentile",
            "Approx Final S Percentile",
            "SOR Final S Percentile",
            "Local Push Final S Percentile",
            "GT Kendall-Tau",
            "GT p-value",
            "Approx Kendall-Tau",
            "Approx p-value",
            "SOR Kendall-Tau",
            "SOR p-value",
            "Local Push Kendall-Tau",
            "Local Push p-value",
            "MC Final S Estimate",
            "MC Final S Lower CI (95%)",
            "MC Final S Upper CI (95%)",
            "GT Final S",
            "Approx Final S",
            "SOR Final S",
            "Local Push Final S"
        ],
        "Value": [
            mc_time,
            gt_time,
            approx_time,
            sor_time,
            local_push_time,
            mc_convergence_steps,
            len(ground_truth),
            len(approx_exp),
            len(sor_exp),
            len(local_push_exp),
            gt_percentile,
            approx_percentile,
            sor_percentile,
            local_push_percentile,
            gt_tau,
            gt_p_value,
            approx_tau,
            approx_p_value,
            sor_tau,
            sor_p_value,
            local_push_tau,
            local_push_p_value,
            mc_s_estimate,
            mc_s_lower_ci,
            mc_s_upper_ci,
            gt_final_s,
            approx_final_s,
            sor_final_s,
            local_push_final_s
        ]
    }
    results_table = pd.DataFrame(data_dict)
    results_table.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)
    print(results_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SIR model simulations.")
    parser.add_argument("--graph_type", type=str, default="er",
                        help="Graph type (e.g., ba, er, powerlaw, smallworld, complete)")
    parser.add_argument("--nodes", type=int, default=10000, help="Number of nodes in the graph")
    parser.add_argument("--beta", type=float, default=1 / 18, help="Infection rate")
    parser.add_argument("--gamma", type=float, default=1 / 9, help="Recovery rate")
    parser.add_argument("--omega", type=float, default=1.3, help="Relaxtion rate")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance for convergence")
    parser.add_argument("--num_simulations", type=int, default=500, help="Number of Monte Carlo simulations")
    parser.add_argument("--initial_infected", type=int, default=50, help="Number of initially infected nodes")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save figures and results")
    parser.add_argument("--dataset", type=str, default="",
                        help="Path to an external dataset file (if empty, use graph_type)")

    args = parser.parse_args()
    main(args)