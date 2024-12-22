import argparse
import os
import time
import random
import datetime
import numpy as np
import networkx as nx
import pandas as pd
from networks import generate_graph, load_real_data
from simulations.mc import run_monte_carlo_simulations
from eval import calculate_kendall_tau, calculate_top_k_overlap, calculate_quantile, calculate_mse, calculate_infection_probability
from simulations.approx_exp import approx_conditional_probability_iteration_exp
from simulations.sor_approx import sor_approx_conditional_probability_iteration_exp
from simulations.ground_truth import conditional_probability_iteration
from simulations.approx_local_push import approx_conditional_probability_iteration_local_push
from visual import plot_results
from utils import calculate_centralities,save_args_to_csv

def main(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    save_args_to_csv(args, output_dir)
    if args.file_name:
        graph = load_real_data(args.data_dir, args.file_name)
    else:
        graph = generate_graph(graph_type=args.graph_type,avg_degree=args.average_degree,nodes=args.nodes)
    if len(graph.nodes()) < args.initial_infected:
        raise ValueError(
            f"Graph has {len(graph.nodes())} nodes, but you requested {args.initial_infected} initial infected nodes. "
            "Please reduce --initial_infected or increase the graph size."
        )
    initial_infected_nodes = random.sample(list(graph.nodes()), args.initial_infected)

    mc_results, mc_trajectories, mc_time = run_monte_carlo_simulations(
        graph, args.beta, args.gamma, initial_infected_nodes, args.num_simulations
    )
    mc_infection_prob = calculate_infection_probability(mc_results)
    mc_final_s = [traj[-1] for traj in mc_trajectories]

    # Methods
    methods = {
        "Ground Truth": conditional_probability_iteration,
        "Approx Exp": approx_conditional_probability_iteration_exp,
        "SOR Approx Exp": sor_approx_conditional_probability_iteration_exp,
        "Local Push": approx_conditional_probability_iteration_local_push,
    }

    # Initialize metrics
    metrics = {"Runtime (s)": {}, "Iterations": {}, "Kendall Tau": {}, "p-value": {}, "Top-K Overlap": {},
               "Final S Quantile": {}, "MSE": {}}
    method_trajectories = {}
    for method_name, method_func in methods.items():
        start_time = time.time()
        results_exp = method_func(graph, args.beta, args.gamma, initial_infected_nodes, args.tol)
        runtime = time.time() - start_time
        iterations = len(results_exp)

        final_s = sum(results_exp[-1]['susceptible'].values())
        tau, p_value = calculate_kendall_tau(results_exp, mc_infection_prob)
        recovered_prob = np.array([results_exp[-1]['recovered'][node]+results_exp[-1]['infected'][node] for node in range(len(mc_infection_prob))])
        top_k = calculate_top_k_overlap(recovered_prob, mc_infection_prob)
        quantile = calculate_quantile(final_s, mc_final_s)
        mse = calculate_mse(final_s, mc_final_s)

        metrics["Runtime (s)"][method_name] = runtime
        metrics["Iterations"][method_name] = iterations
        metrics["Kendall Tau"][method_name] = tau
        metrics["p-value"][method_name] = p_value
        metrics["Top-K Overlap"][method_name] = top_k
        metrics["Final S Quantile"][method_name] = quantile
        metrics["MSE"][method_name] = mse
        method_trajectories[method_name] = [sum(result['susceptible'].values()) for result in results_exp]

    # Add Monte Carlo baseline
    metrics["Runtime (s)"]["Monte Carlo"] = mc_time
    metrics["Iterations"]["Monte Carlo"] = np.mean([len(traj) for traj in mc_trajectories])

    # Convert to DataFrame and transpose
    results_df = pd.DataFrame(metrics).T
    results_df.to_csv(os.path.join(output_dir, "results.csv"))
    print(f"Results saved to: {os.path.join(output_dir, 'results.csv')}")
    print(results_df)

    # Save Monte Carlo plot
    # Pad trajectories to the maximum length
    max_length = max(len(traj) for traj in mc_trajectories)
    padded_trajectories = np.array([traj + [traj[-1]] * (max_length - len(traj)) for traj in mc_trajectories])

    # Calculate mean and confidence intervals
    plot_results(
        mc_trajectories,
        method_trajectories["Ground Truth"],
        method_trajectories["Approx Exp"],
        method_trajectories["SOR Approx Exp"],
        method_trajectories["Local Push"],
        output_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SIR model simulations.")
    parser.add_argument("--graph_type", type=str, default="er",
                        help="Graph type (e.g., ba, er, powerlaw, smallworld, complete)")
    parser.add_argument("--nodes", type=int, default=1000, help="Number of nodes in the graph")
    parser.add_argument("--average_degree", type=float, default=10, help="Average degree of the graph")
    parser.add_argument("--beta", type=float, default=1 / 18, help="Infection rate")
    parser.add_argument("--gamma", type=float, default=1 / 9, help="Recovery rate")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance for convergence")
    parser.add_argument("--num_simulations", type=int, default=200, help="Number of Monte Carlo simulations")
    parser.add_argument("--initial_infected", type=int, default=10, help="Number of initially infected nodes")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to load real data")
    parser.add_argument("--file_name", type=str, default="soc-Epinions1.txt.gz", help="File name of the real data")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save results")

    args = parser.parse_args()
    main(args)