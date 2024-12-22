import argparse
import os
import time
import random
import datetime
import numpy as np
import pandas as pd
from networks import generate_graph
from simulations.mc import run_monte_carlo_simulations
from eval import calculate_quantile, calculate_mse
from simulations.approx_exp import approx_conditional_probability_iteration_exp
from simulations.sor_approx import sor_approx_conditional_probability_iteration_exp
from simulations.ground_truth import conditional_probability_iteration
from simulations.approx_local_push import approx_conditional_probability_iteration_local_push
from utils import save_args_to_csv


def calculate_mc_final_s_statistics(mc_final_s_trials):
    mc_final_s_trials = np.array(mc_final_s_trials)
    mean_s = np.mean(mc_final_s_trials)
    lower_bound = np.percentile(mc_final_s_trials, 2.5)
    upper_bound = np.percentile(mc_final_s_trials, 97.5)

    mean_error = np.std([np.mean(s) for s in mc_final_s_trials])
    lower_error = np.std([np.percentile(s, 2.5) for s in mc_final_s_trials])
    upper_error = np.std([np.percentile(s, 97.5) for s in mc_final_s_trials])

    return mean_s, lower_bound, upper_bound, mean_error, lower_error, upper_error


def calculate_final_s_error_band(mc_final_s_trials, method_final_s):
    mse_values = [calculate_mse(s, method_final_s) for s in mc_final_s_trials]
    quantile_values = [calculate_quantile(method_final_s, s) for s in mc_final_s_trials]

    mse = f"{np.mean(mse_values):.4f} ± {np.std(mse_values):.4f}"
    quantile = f"{np.mean(quantile_values):.4f} ± {np.std(quantile_values):.4f}"
    return mse, quantile


def main(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    save_args_to_csv(args, output_dir)

    graph = generate_graph(graph_type=args.graph_type, avg_degree=args.average_degree, nodes=args.nodes)
    if len(graph.nodes()) < args.initial_infected:
        raise ValueError("The graph does not have enough nodes for the initial infected nodes.")
    initial_infected_nodes = random.sample(list(graph.nodes()), args.initial_infected)

    print("Running Monte Carlo simulations...")
    mc_final_s_trials, mc_runtime = [], []
    for i in range(3):  # Run Monte Carlo simulations 3 times
        print(f"Monte Carlo Trial {i + 1}")
        start_time = time.time()
        _, mc_trajectories, _ = run_monte_carlo_simulations(
            graph, args.beta, args.gamma, initial_infected_nodes, args.num_simulations
        )
        mc_runtime.append(time.time() - start_time)
        mc_final_s_trials.append([traj[-1] for traj in mc_trajectories])

    mean_s, lower_bound, upper_bound, mean_error, lower_error, upper_error = calculate_mc_final_s_statistics(
        mc_final_s_trials
    )

    mc_summary = {
        "Metric": ["Mean S", "Lower Bound (2.5%)", "Upper Bound (97.5%)"],
        "Value": [
            f"{mean_s:.4f} ± {mean_error:.4f}",
            f"{lower_bound:.4f} ± {lower_error:.4f}",
            f"{upper_bound:.4f} ± {upper_error:.4f}",
        ]
    }

    methods = {
        "Ground Truth": conditional_probability_iteration,
        "Approx Exp": approx_conditional_probability_iteration_exp,
        "SOR Approx Exp": sor_approx_conditional_probability_iteration_exp,
        "Local Push": approx_conditional_probability_iteration_local_push,
    }

    method_summary = []
    print("Running methods and calculating metrics...")
    for method_name, method_func in methods.items():
        print(f"Running {method_name}...")
        results_exp = method_func(graph, args.beta, args.gamma, initial_infected_nodes, args.tol)

        method_final_s = sum(results_exp[-1]['susceptible'].values())
        final_s_error = np.std([sum(s) for s in mc_final_s_trials])
        method_summary.append({
            "Metric": f"Final S ({method_name})",
            "Value": f"{method_final_s:.4f} ± {final_s_error:.4f}"
        })

    combined_summary = pd.concat(
        [pd.DataFrame(mc_summary), pd.DataFrame(method_summary)],
        ignore_index=True
    )

    combined_summary.to_csv(os.path.join(output_dir, "mc_simulations_summary.csv"), index=False)
    print(f"Monte Carlo summary saved to: {os.path.join(output_dir, 'mc_simulations_summary.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SIR model simulations.")
    parser.add_argument("--graph_type", type=str, default="smallworld",
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