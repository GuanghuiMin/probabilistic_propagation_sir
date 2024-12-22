import argparse
import os
import time
import random
import datetime
import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import kendalltau
from networks import generate_graph,load_real_data
from simulations.mc import run_monte_carlo_simulations
from eval import calculate_kendall_tau, calculate_top_k_overlap, calculate_quantile, calculate_mse, calculate_infection_probability
from simulations.approx_exp import approx_conditional_probability_iteration_exp
from simulations.sor_approx import sor_approx_conditional_probability_iteration_exp
from simulations.ground_truth import conditional_probability_iteration
from simulations.approx_local_push import approx_conditional_probability_iteration_local_push
from visual import plot_results
from utils import calculate_centralities,save_args_to_csv

def calculate_error_bars_for_metrics(mc_infection_probs, method_probs, k=200):
    """Calculate Kendall-Tau and Top-K Overlap error bars by comparing method probabilities with MC averages."""
    kendall_taus = []
    top_k_overlaps = []

    for mc_prob in mc_infection_probs:
        # Kendall-Tau
        tau, _ = kendalltau(mc_prob, method_probs)
        kendall_taus.append(tau)

        # Top-K Overlap
        top_k_overlap = calculate_top_k_overlap(method_probs, mc_prob, k)
        top_k_overlaps.append(top_k_overlap)

    # Return mean ± std for both metrics
    return {
        "Kendall Tau": f"{np.mean(kendall_taus):.4f} ± {np.std(kendall_taus):.4f}",
        "Top-K Overlap": f"{np.mean(top_k_overlaps):.4f} ± {np.std(top_k_overlaps):.4f}"
    }


def main(args):
    # Output setup
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    save_args_to_csv(args, output_dir)
    if args.file_name:
        graph = load_real_data(args.data_dir, args.file_name)
    else:
        graph = generate_graph(graph_type=args.graph_type, avg_degree=args.average_degree, nodes=args.nodes)
    if len(graph.nodes()) < args.initial_infected:
        raise ValueError(
            f"Graph has {len(graph.nodes())} nodes, but you requested {args.initial_infected} initial infected nodes. "
            "Please reduce --initial_infected or increase the graph size."
        )
    # if graph.is_multigraph():
    #     graph = nx.DiGraph(graph)
    initial_infected_nodes = random.sample(list(graph.nodes()), args.initial_infected)

    # Monte Carlo simulations with multiple trials
    print("Running Monte Carlo simulations...")
    mc_infection_probs = []
    mc_runtime = []
    mc_iterations = []

    for i in range(5):  # Run Monte Carlo simulations 5 times
        print(f"Monte Carlo Trial {i + 1}")
        start_time = time.time()
        mc_results, mc_trajectories, _ = run_monte_carlo_simulations(
            graph, args.beta, args.gamma, initial_infected_nodes, args.num_simulations
        )
        mc_runtime.append(time.time() - start_time)
        mc_iterations.append(np.mean([len(traj) for traj in mc_trajectories]))
        mc_infection_probs.append(calculate_infection_probability(mc_results))

    mean_mc_infection_prob = np.mean(mc_infection_probs, axis=0)

    # Monte Carlo Metrics
    metrics = {"Runtime": {}, "Iterations": {}, "Kendall Tau": {}, "Top-K Overlap": {}}
    metrics["Runtime"]["Monte Carlo"] = f"{np.mean(mc_runtime):.4f} ± {np.std(mc_runtime):.4f}"
    metrics["Iterations"]["Monte Carlo"] = f"{np.mean(mc_iterations):.2f} ± {np.std(mc_iterations):.2f}"

    # Methods evaluation (only once)
    methods = {
        "Ground Truth": conditional_probability_iteration,
        "Approx Exp": approx_conditional_probability_iteration_exp,
        "SOR Approx Exp": sor_approx_conditional_probability_iteration_exp,
        "Local Push": approx_conditional_probability_iteration_local_push,
    }

    print("Running methods and calculating metrics...")
    for method_name, method_func in methods.items():
        print(f"Running {method_name}...")
        results_exp = method_func(graph, args.beta, args.gamma, initial_infected_nodes, args.tol)

        # Calculate method infection probabilities
        method_probs = np.array([
            results_exp[-1]['recovered'][node] + results_exp[-1]['infected'][node]
            for node in range(len(mean_mc_infection_prob))
        ])

        # Compare with MC infection probabilities
        metrics_with_error = calculate_error_bars_for_metrics(mc_infection_probs, method_probs)
        metrics["Kendall Tau"][method_name] = metrics_with_error["Kendall Tau"]
        metrics["Top-K Overlap"][method_name] = metrics_with_error["Top-K Overlap"]

    # Centralities evaluation
    print("Calculating centralities...")
    centralities = calculate_centralities(graph)
    for name, values in centralities.items():
        print(f"Processing {name} centrality...")
        centrality_probs = np.array([values[node] for node in range(len(mean_mc_infection_prob))])
        metrics_with_error = calculate_error_bars_for_metrics(mc_infection_probs, centrality_probs)

        metrics["Kendall Tau"][name] = metrics_with_error["Kendall Tau"]
        metrics["Top-K Overlap"][name] = metrics_with_error["Top-K Overlap"]
        metrics["Runtime"][name] = "N/A"
        metrics["Iterations"][name] = "N/A"

    print("Centralities calculated.")

    print(f"{mc_infection_probs[-1]}")
    # Save results
    results_df = pd.DataFrame(metrics).T
    results_df.to_csv(os.path.join(output_dir, "results_with_error.csv"))
    print(f"Results saved to: {os.path.join(output_dir, 'results_with_error.csv')}")
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
    parser.add_argument("--graph_type", type=str, default="powerlaw",
                        help="Graph type (e.g., ba, er, powerlaw, smallworld)")
    parser.add_argument("--nodes", type=int, default=5000, help="Number of nodes in the graph")
    parser.add_argument("--average_degree", type=float, default=20, help="Average degree of the graph")
    parser.add_argument("--beta", type=float, default=1 / 18, help="Infection rate")
    parser.add_argument("--gamma", type=float, default=1 / 9, help="Recovery rate")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance for convergence")
    parser.add_argument("--num_simulations", type=int, default=200, help="Number of Monte Carlo simulations")
    parser.add_argument("--initial_infected", type=int, default=50, help="Number of initially infected nodes")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to load real data")
    parser.add_argument("--file_name", type=str, help="File name of the real data")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save results")

    args = parser.parse_args()
    main(args)