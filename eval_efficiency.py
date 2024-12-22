import argparse
import os
import time
import random
import datetime
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from networks import generate_graph
from simulations.mc import run_monte_carlo_simulations
from eval import calculate_kendall_tau, calculate_top_k_overlap, calculate_quantile, calculate_mse, calculate_infection_probability
from simulations.approx_exp import approx_conditional_probability_iteration_exp
from simulations.sor_approx import sor_approx_conditional_probability_iteration_exp
from simulations.ground_truth import conditional_probability_iteration
from simulations.approx_local_push import approx_conditional_probability_iteration_local_push
from utils import save_args_to_csv


def run_multiple_trials_mc(graph, beta, gamma, initial_infected_nodes, num_simulations, num_trials=10):
    """
    Run Monte Carlo simulations multiple times and calculate mean ± std.
    """
    runtimes, iterations = [], []
    for _ in range(num_trials):
        start_time = time.time()
        mc_results, mc_trajectories, _ = run_monte_carlo_simulations(
            graph, beta, gamma, initial_infected_nodes, num_simulations
        )
        runtime = time.time() - start_time
        runtimes.append(runtime)
        iterations.append(np.mean([len(traj) for traj in mc_trajectories]))
    return np.mean(runtimes), np.std(runtimes), np.mean(iterations), np.std(iterations)


def run_single_method_multiple_trials(method_func, graph, beta, gamma, initial_infected_nodes, tol, num_trials=10):
    """
    Run a single method multiple times and calculate mean ± std for runtime and iterations.
    """
    runtimes, iterations = [], []
    for _ in range(num_trials):
        start_time = time.time()
        results_exp = method_func(graph, beta, gamma, initial_infected_nodes, tol)
        runtime = time.time() - start_time
        runtimes.append(runtime)
        iterations.append(len(results_exp))
    return np.mean(runtimes), np.std(runtimes), np.mean(iterations), np.std(iterations)


def main(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    save_args_to_csv(args, output_dir)
    graph = generate_graph(graph_type=args.graph_type, avg_degree=args.average_degree, nodes=args.nodes)
    if len(graph.nodes()) < args.initial_infected:
        raise ValueError(
            f"Graph has {len(graph.nodes())} nodes, but you requested {args.initial_infected} initial infected nodes. "
            "Please reduce --initial_infected or increase the graph size."
        )
    initial_infected_nodes = random.sample(list(graph.nodes()), args.initial_infected)

    # Monte Carlo simulations for different numbers of runs
    mc_simulations = [100, 200, 500]
    mc_metrics = {}
    for num_simulations in mc_simulations:
        print(f"Running Monte Carlo simulations ({num_simulations} runs)...")
        runtime_mean, runtime_std, iter_mean, iter_std = run_multiple_trials_mc(
            graph, args.beta, args.gamma, initial_infected_nodes, num_simulations, num_trials=10
        )
        mc_metrics[f"Monte Carlo ({num_simulations} runs)"] = {
            "Runtime (s)": f"{runtime_mean:.4f} ± {runtime_std:.4f}",
            "Iterations": f"{iter_mean:.2f} ± {iter_std:.2f}"
        }

    # Methods evaluation
    methods = {
        "Ground Truth": conditional_probability_iteration,
        "Approx Exp": approx_conditional_probability_iteration_exp,
        "SOR Approx Exp": sor_approx_conditional_probability_iteration_exp,
        "Local Push": approx_conditional_probability_iteration_local_push,
    }

    metrics = {"Runtime (s)": {}, "Iterations": {}}
    for method_name, method_func in methods.items():
        print(f"Running {method_name}...")
        runtime_mean, runtime_std, iter_mean, iter_std = run_single_method_multiple_trials(
            method_func, graph, args.beta, args.gamma, initial_infected_nodes, args.tol, num_trials=10
        )
        metrics["Runtime (s)"][method_name] = f"{runtime_mean:.4f} ± {runtime_std:.4f}"
        metrics["Iterations"][method_name] = f"{iter_mean:.2f} ± {iter_std:.2f}"
        print(f"{method_name} complete.")

    # Combine Monte Carlo metrics into the final metrics
    for mc_key, mc_data in mc_metrics.items():
        metrics["Runtime (s)"][mc_key] = mc_data["Runtime (s)"]
        metrics["Iterations"][mc_key] = mc_data["Iterations"]

    # Save results
    results_df = pd.DataFrame(metrics).T
    results_df.to_csv(os.path.join(output_dir, "results_runtime_iterations.csv"))
    print(f"Results saved to: {os.path.join(output_dir, 'results_runtime_iterations.csv')}")
    print(results_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SIR model simulations.")
    parser.add_argument("--graph_type", type=str, default="ba",
                        help="Graph type (e.g., ba, er, powerlaw, smallworld)")
    parser.add_argument("--nodes", type=int, default=5000, help="Number of nodes in the graph")
    parser.add_argument("--average_degree", type=float, default=10, help="Average degree of the graph")
    parser.add_argument("--beta", type=float, default=1 / 18, help="Infection rate")
    parser.add_argument("--gamma", type=float, default=1 / 9, help="Recovery rate")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance for convergence")
    parser.add_argument("--initial_infected", type=int, default=50, help="Number of initially infected nodes")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to load real data")
    parser.add_argument("--file_name", type=str, help="File name of the real data")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save results")

    args = parser.parse_args()
    main(args)