import argparse
import os
import torch
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from networks import generate_pyg_graph
from simulations.mc import run_sir_simulation
from simulations.pps import conditional_probability_iteration
from simulations.apps import approx_conditional_probability_iteration_exp
from utils import calculate_graph_properties, calculate_infection_probability
from eval import compare_mc_and_ranking
from visualize import plot_results


def run_monte_carlo(data, beta, gamma, initial_infected_nodes, num_simulations):
    """Run Monte Carlo simulations."""
    results_list = []
    mc_s_trajectories = []
    start_time = time.time()
    for _ in tqdm(range(num_simulations), desc="Running Monte Carlo Simulations"):
        results = run_sir_simulation(data, beta, gamma, initial_infected_nodes)
        s_trajectory = [sum(1 for state in result['status'].values() if state == 0) for result in results]
        mc_s_trajectories.append(s_trajectory)
        results_list.append(results)
    mc_time = time.time() - start_time
    return results_list, mc_s_trajectories, mc_time

def main(args):  
    data = generate_pyg_graph(args.graph_type, nodes=args.nodes)
    initial_infected_nodes = torch.tensor(
        np.random.choice(args.nodes, args.initial_infected, replace=False), device='cuda'
    )

    max_in_degree, two_norm, spectral_radius = calculate_graph_properties(data)
    print(f"beta/gamma: {args.beta/args.gamma}, if epidemic threshold: {args.beta/args.gamma < 1/two_norm}, if error converges: {args.beta/args.gamma < 1/max_in_degree}")
    print(f"Graph properties: max in-degree: {max_in_degree}, 2-norm: {two_norm}, spectral_radius: {spectral_radius}")

    results_list, mc_s_trajectories, mc_time = run_monte_carlo(data, args.beta, args.gamma, initial_infected_nodes, args.num_simulations)
    infection_probability = calculate_infection_probability(results_list, data.num_nodes)
    mc_convergence_steps = np.mean([len(trajectory) for trajectory in mc_s_trajectories])

    start_time = time.time()
    ground_truth = conditional_probability_iteration(data, args.beta, args.gamma, initial_infected_nodes, args.tol)
    gt_time = time.time() - start_time

    start_time = time.time()
    approx_exp = approx_conditional_probability_iteration_exp(data, args.beta, args.gamma, initial_infected_nodes, args.tol)
    approx_time = time.time() - start_time

    ground_truth_s = [sum(result['susceptible']) for result in ground_truth]
    approx_exp_s = [sum(result['susceptible']) for result in approx_exp]
    gt_tau, gt_p_value = compare_mc_and_ranking(ground_truth, infection_probability)
    approx_tau, approx_p_value = compare_mc_and_ranking(approx_exp, infection_probability)


    ground_truth_s = [sum(result['susceptible']) for result in ground_truth]
    approx_exp_s = [sum(result['susceptible']) for result in approx_exp]
    gt_final_s = ground_truth_s[-1]
    approx_final_s = approx_exp_s[-1]
    mc_final_s = [trajectory[-1] if isinstance(trajectory[-1], (int, float)) else sum(trajectory[-1]) for trajectory in mc_s_trajectories]
    gt_percentile = np.sum(np.array(mc_final_s) < gt_final_s) / len(mc_final_s) * 100
    approx_percentile = np.sum(np.array(mc_final_s) < approx_final_s) / len(mc_final_s) * 100

    plot_results(mc_s_trajectories, ground_truth_s, approx_exp_s, args.nodes, args.output_dir)

    data_dict = {
    "Metric": [
        "Monte Carlo Time (s)", 
        "Ground Truth Time (s)", 
        "Approximation Time (s)", 
        "MC Average Convergence Steps", 
        "GT Convergence Steps", 
        "Approximation Convergence Steps", 
        "GT Final S Percentile", 
        "Approx Final S Percentile", 
        "GT Kendall-Tau", 
        "GT p-value", 
        "Approx Kendall-Tau", 
        "Approx p-value"
    ],
    "Value": [
        mc_time, 
        gt_time, 
        approx_time, 
        mc_convergence_steps, 
        len(ground_truth), 
        len(approx_exp), 
        gt_percentile, 
        approx_percentile, 
        gt_tau, 
        gt_p_value, 
        approx_tau, 
        approx_p_value
    ]
}
    results_table = pd.DataFrame(data_dict)
    results_table.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)
    print(results_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SIR model simulations.")
    parser.add_argument("--graph_type", type=str, default="er", help="Graph type (e.g., ba, er, powerlaw, smallworld)")
    parser.add_argument("--nodes", type=int, default=50, help="Number of nodes in the graph")
    parser.add_argument("--beta", type=float, default=1/18, help="Infection rate")
    parser.add_argument("--gamma", type=float, default=1/9, help="Recovery rate")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance for convergence")
    parser.add_argument("--num_simulations", type=int, default=500, help="Number of Monte Carlo simulations")
    parser.add_argument("--initial_infected", type=int, default=10, help="Number of initially infected nodes")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save figures and results")
    args = parser.parse_args()
    main(args)