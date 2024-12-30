import argparse
import os
import gc
import time
import random
import datetime
import numpy as np
import pandas as pd
import networkx as nx

from networks import generate_graph, load_real_data
from simulations.mc import run_monte_carlo_simulations
from eval import (
    calculate_kendall_tau,
    calculate_top_k_overlap,
    calculate_quantile,
    calculate_mse,
    calculate_infection_probability,
)
from simulations.approx_exp import approx_conditional_probability_iteration_exp
from simulations.sor_approx import sor_approx_conditional_probability_iteration_exp
from simulations.ground_truth import conditional_probability_iteration
from simulations.approx_local_push import approx_conditional_probability_iteration_local_push
from visual import plot_results
from utils import calculate_centralities, save_args_to_csv

def main(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print("[Info] Loading or generating graph...")
    if args.file_name:
        graph = load_real_data(args.data_dir, args.file_name)
    else:
        graph = generate_graph(
            graph_type=args.graph_type,
            avg_degree=args.average_degree,
            nodes=args.nodes
        )
    num_nodes = graph.number_of_nodes()
    args.nodes = num_nodes
    print("[Info] Saving arguments...")
    save_args_to_csv(args, output_dir)

    if num_nodes < args.initial_infected:
        raise ValueError(
            f"Graph has {num_nodes} nodes, "
            f"but you requested {args.initial_infected} initial infected nodes. "
            "Please reduce --initial_infected or increase the graph size."
        )

    initial_infected_nodes = random.sample(list(graph.nodes()), args.initial_infected)

    print("[Info] Running MC simulations ...")
    start_time = time.time()
    mc_results, mc_trajectories, _ = run_monte_carlo_simulations(
        graph, args.beta, args.gamma, initial_infected_nodes, args.num_simulations
    )
    mc_time = time.time() - start_time
    mc_infection_prob = calculate_infection_probability(mc_results)
    mc_final_s = [traj[-1] for traj in mc_trajectories]
    del mc_results
    gc.collect()

    methods = {
        "Ground Truth": conditional_probability_iteration,
        "Approx Exp": approx_conditional_probability_iteration_exp,
        "SOR Approx Exp": sor_approx_conditional_probability_iteration_exp,
        "Local Push": approx_conditional_probability_iteration_local_push,
    }

    metrics = {
        "Runtime (s)": {},
        "Iterations": {},
        "Kendall Tau": {},
        "p-value": {},
        "Top-K Overlap": {},
        "Final S Quantile": {},
        "MSE": {},
    }
    method_trajectories = {}

    for method_name, method_func in methods.items():
        print(f"[Method] {method_name} started.")
        start_t = time.time()
        results_exp = method_func(
            graph, args.beta, args.gamma, initial_infected_nodes, args.tol
        )
        runtime = time.time() - start_t
        iterations = len(results_exp)
        final_s = sum(results_exp[-1]["susceptible"].values())

        tau, p_value = calculate_kendall_tau(results_exp, mc_infection_prob)
        recovered_prob = np.array([
            results_exp[-1]["recovered"][node] + results_exp[-1]["infected"][node]
            for node in range(num_nodes)
        ])
        top_k = calculate_top_k_overlap(recovered_prob, mc_infection_prob, k=round(0.1*args.nodes))
        quantile = calculate_quantile(final_s, mc_final_s)
        mse = calculate_mse(final_s, mc_final_s)

        metrics["Runtime (s)"][method_name] = runtime
        metrics["Iterations"][method_name] = iterations
        metrics["Kendall Tau"][method_name] = tau
        metrics["p-value"][method_name] = p_value
        metrics["Top-K Overlap"][method_name] = top_k
        metrics["Final S Quantile"][method_name] = quantile
        metrics["MSE"][method_name] = mse

        method_trajectories[method_name] = [
            sum(result["susceptible"].values()) for result in results_exp
        ]

        print(f"[Method] {method_name} finished. Runtime={runtime:.2f}s, Iterations={iterations}")
        del results_exp
        gc.collect()


    metrics["Runtime (s)"]["Monte Carlo"] = mc_time
    metrics["Iterations"]["Monte Carlo"] = np.mean([len(traj) for traj in mc_trajectories])


    results_df = pd.DataFrame(metrics).T
    results_df_path = os.path.join(output_dir, "results.csv")
    results_df.to_csv(results_df_path)
    print(f"[Info] Results saved to: {results_df_path}")
    print(results_df)


    print("[Info] Plotting results...")
    plot_results(
        mc_trajectories,
        method_trajectories.get("Ground Truth", []),
        method_trajectories.get("Approx Exp", []),
        method_trajectories.get("SOR Approx Exp", []),
        method_trajectories.get("Local Push", []),
        output_dir
    )

    print("[Info] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SIR model simulations.")
    parser.add_argument("--graph_type", type=str, default="er",
                        help="Graph type (e.g., ba, er, powerlaw, smallworld, complete)")
    parser.add_argument("--nodes", type=int, default=1000, help="Number of nodes in the graph")
    parser.add_argument("--average_degree", type=float, default=10, help="Average degree of the graph")
    parser.add_argument("--beta", type=float, default=1 / 18, help="Infection rate")
    parser.add_argument("--gamma", type=float, default=1 / 9, help="Recovery rate")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance for convergence")
    parser.add_argument("--num_simulations", type=int, default=20, help="Number of Monte Carlo simulations")
    parser.add_argument("--initial_infected", type=int, default=10, help="Number of initially infected nodes")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to load real data")
    parser.add_argument("--file_name", type=str, default="gplus_combined.txt.gz", help="File name of the real data")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save results")

    args = parser.parse_args()
    main(args)