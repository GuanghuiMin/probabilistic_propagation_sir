import argparse
import os
import time
import random
import datetime
import numpy as np
import pandas as pd
import json
from scipy.sparse import coo_matrix
from scipy.stats import kendalltau
from networks import generate_graph, load_real_data
from simulations.mc import run_monte_carlo_simulations
from simulations.approx_exp import approx_conditional_probability_iteration_exp
from simulations.sor_approx import sor_approx_conditional_probability_iteration_exp
from simulations.ground_truth import conditional_probability_iteration
from simulations.approx_local_push import approx_conditional_probability_iteration_local_push
from eval import calculate_top_k_overlap, calculate_infection_probability
from utils import save_args_to_csv

def run_multiple_trials(method_func, graph, beta, gamma, initial_infected_nodes, tol, num_trials=3, output_dir=None):
    """Run a method multiple times and calculate runtime, iterations, and final S values."""
    runtimes = []
    iterations = []
    final_s_values = []
    raw_outputs = []

    for trial in range(num_trials):
        print(f"Running trial {trial + 1}/{num_trials} for method.")
        start_time = time.time()
        results_exp = method_func(graph, beta, gamma, initial_infected_nodes, tol)
        runtimes.append(time.time() - start_time)
        iterations.append(len(results_exp))
        final_s = sum(results_exp[-1]["susceptible"].values())
        final_s_values.append(final_s)

        # Convert results to sparse format
        sparse_results = [
            {
                key: {
                    "row": list(data.keys()),
                    "col": [0] * len(data),
                    "data": list(data.values()),
                    "shape": [len(data), 1],
                }
                for key, data in iteration.items() if isinstance(data, dict)
            }
            for iteration in results_exp
        ]
        raw_outputs.append(sparse_results)

    # Save raw outputs in sparse format
    save_sparse_json(
        os.path.join(output_dir, f"{method_func.__name__}_raw_results.json"),
        raw_outputs,
    )

    return {
        "Runtime": f"{np.mean(runtimes):.4f} ± {np.std(runtimes):.4f}",
        "Iterations": f"{np.mean(iterations):.2f} ± {np.std(iterations):.2f}",
        "Final S": f"{np.mean(final_s_values):.4f} ± {np.std(final_s_values):.4f}",
        "Final S Values": final_s_values,  # Return for further analysis
    }


def save_sparse_json(filepath, results):
    """Save sparse results to JSON file."""
    sparse_data = []
    for simulation in results:
        if isinstance(simulation, (np.ndarray, list)):  # Handle array or list
            sparse_simulation = {
                "row": list(range(len(simulation))),
                "col": [0] * len(simulation),
                "data": list(simulation),
                "shape": [len(simulation), 1],
            }
            sparse_data.append(sparse_simulation)
        else:
            raise ValueError(f"Unexpected simulation result structure: {type(simulation)}")
    with open(filepath, "w") as f:
        json.dump(sparse_data, f)


def main(args):
    # Output setup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    save_args_to_csv(args, output_dir)

    # Graph generation
    if args.file_name:
        graph = load_real_data(args.data_dir, args.file_name)
    else:
        graph = generate_graph(graph_type=args.graph_type, avg_degree=args.average_degree, nodes=args.nodes)
    if len(graph.nodes()) < args.initial_infected:
        raise ValueError("The graph does not have enough nodes for the initial infected nodes.")
    initial_infected_nodes = random.sample(list(graph.nodes()), args.initial_infected)

    # Monte Carlo simulations
    print("Running Monte Carlo simulations...")
    mc_runtime = []
    mc_iterations = []
    mc_final_s_values = []
    mc_infection_probs = []

    for i in range(3):  # Run Monte Carlo simulations 3 times
        print(f"Monte Carlo Trial {i + 1}")
        start_time = time.time()
        mc_results, mc_trajectories, _ = run_monte_carlo_simulations(
            graph, args.beta, args.gamma, initial_infected_nodes, args.num_simulations
        )
        mc_runtime.append(time.time() - start_time)
        mc_iterations.append(np.mean([len(traj) for traj in mc_trajectories]))
        mc_final_s_values.extend([traj[-1] for traj in mc_trajectories])
        mc_infection_probs.append(calculate_infection_probability(mc_results))

    # Save Monte Carlo results as sparse format
    save_sparse_json(os.path.join(output_dir, "mc_infection_probs.json"), mc_infection_probs)

    # Monte Carlo Summary
    mean_mc_infection_prob = np.mean(mc_infection_probs, axis=0)
    mc_summary = {
        "Runtime": f"{np.mean(mc_runtime):.4f} ± {np.std(mc_runtime):.4f}",
        "Iterations": f"{np.mean(mc_iterations):.2f} ± {np.std(mc_iterations):.2f}",
        "Final S Mean": f"{np.mean(mc_final_s_values):.4f} ± {np.std(mc_final_s_values):.4f}",
        "Final S Lower CI": f"{np.percentile(mc_final_s_values, 2.5):.4f} ± {np.std([np.percentile(sim, 2.5) for sim in mc_final_s_values]):.4f}",
        "Final S Upper CI": f"{np.percentile(mc_final_s_values, 97.5):.4f} ± {np.std([np.percentile(sim, 97.5) for sim in mc_final_s_values]):.4f}",
    }
    pd.DataFrame([mc_summary]).to_csv(os.path.join(output_dir, "mc_summary.csv"), index=False)

    print("Monte Carlo simulations complete and saved successfully.")

    # Methods evaluation
    methods = {
        "Ground Truth": conditional_probability_iteration,
        "Approx Exp": approx_conditional_probability_iteration_exp,
        "SOR Approx Exp": sor_approx_conditional_probability_iteration_exp,
        "Local Push": approx_conditional_probability_iteration_local_push,
    }

    print("Running methods and calculating metrics...")
    method_results = {}
    for method_name, method_func in methods.items():
        print(f"Running {method_name}...")
        summary = run_multiple_trials(
            method_func, graph, args.beta, args.gamma, initial_infected_nodes, args.tol, num_trials=3, output_dir=output_dir
        )

        # Calculate Kendall Tau and Top-K Overlap using Monte Carlo mean
        kendall_tau_vals = []
        top_k_overlap_vals = []
        for i in range(3):  # Compare each method trial with MC mean
            tau, _ = kendalltau(mean_mc_infection_prob, mc_infection_probs[i])
            kendall_tau_vals.append(tau)
            top_k_overlap_vals.append(
                calculate_top_k_overlap(mean_mc_infection_prob, mc_infection_probs[i], k=200)
            )

        method_results[method_name] = {
            "Runtime": summary["Runtime"],
            "Iterations": summary["Iterations"],
            "Final S": summary["Final S"],
            "Kendall Tau": f"{np.mean(kendall_tau_vals):.4f} ± {np.std(kendall_tau_vals):.4f}",
            "Top-K Overlap": f"{np.mean(top_k_overlap_vals):.4f} ± {np.std(top_k_overlap_vals):.4f}",
        }

    # Save method results
    pd.DataFrame.from_dict(method_results, orient="index").to_csv(
        os.path.join(output_dir, "results_with_error.csv")
    )

    print("Results saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SIR model simulations.")
    parser.add_argument("--graph_type", type=str, default="er",
                        help="Graph type (e.g., ba, er, powerlaw, smallworld, complete)")
    parser.add_argument("--nodes", type=int, default=1000, help="Number of nodes in the graph")
    parser.add_argument("--average_degree", type=float, default=10, help="Average degree of the graph")
    parser.add_argument("--beta", type=float, default=1 / 18, help="Infection rate")
    parser.add_argument("--gamma", type=float, default=1 / 9, help="Recovery rate")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance for convergence")
    parser.add_argument("--num_simulations", type=int, default=2, help="Number of Monte Carlo simulations")
    parser.add_argument("--initial_infected", type=int, default=10, help="Number of initially infected nodes")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to load real data")
    parser.add_argument("--file_name", type=str, default="soc-Epinions1.txt.gz", help="File name of the real data")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save results")

    args = parser.parse_args()
    main(args)