import argparse
import os
import gc
import time
import random
import datetime
import numpy as np
import pandas as pd
from pygments.lexer import default
from scipy import sparse
from networks import generate_graph, load_real_data
from simulations.mc import run_monte_carlo_simulations
from scipy.stats import kendalltau
from eval import calculate_kendall_tau, calculate_top_k_overlap, calculate_quantile
from eval import calculate_mse, calculate_infection_probability
from simulations.approx_exp import approx_conditional_probability_iteration_exp
from simulations.sor_approx import sor_approx_conditional_probability_iteration_exp
from simulations.ground_truth import conditional_probability_iteration
from simulations.approx_local_push import approx_conditional_probability_iteration_local_push
from visual import plot_results
from utils import save_args_to_csv, calculate_centralities

def calculate_error_bars_for_metrics(mc_infection_probs, method_probs, k):
    kendall_taus = []
    top_k_overlaps = []
    for mc_prob in mc_infection_probs:
        tau, _ = kendalltau(mc_prob, method_probs)
        kendall_taus.append(tau)
        top_k_overlap = calculate_top_k_overlap(method_probs, mc_prob, k)
        top_k_overlaps.append(top_k_overlap)
    return {
        "Kendall Tau": f"{np.mean(kendall_taus):.4f} ± {np.std(kendall_taus):.4f}",
        "Top-K Overlap": f"{np.mean(top_k_overlaps):.4f} ± {np.std(top_k_overlaps):.4f}"
    }

def calculate_mc_final_s_statistics(mc_final_s_trials):
    mc_final_s_trials = [np.array(x) for x in mc_final_s_trials]
    means = [np.mean(x) for x in mc_final_s_trials]
    lowers = [np.percentile(x, 2.5) for x in mc_final_s_trials]
    uppers = [np.percentile(x, 97.5) for x in mc_final_s_trials]
    mean_s = np.mean(means)
    lower_bound = np.mean(lowers)
    upper_bound = np.mean(uppers)
    mean_error = np.std(means)
    lower_error = np.std(lowers)
    upper_error = np.std(uppers)
    return mean_s, lower_bound, upper_bound, mean_error, lower_error, upper_error

def calculate_error_metrics(mc_infection_probs, method_probs, k):
    kendall_taus = []
    top_k_overlaps = []
    for mc_prob in mc_infection_probs:
        tau, _ = kendalltau(mc_prob, method_probs)
        kendall_taus.append(tau)
        top_k_overlap = calculate_top_k_overlap(method_probs, mc_prob, k)
        top_k_overlaps.append(top_k_overlap)
    return {
        "Kendall Tau": f"{np.mean(kendall_taus):.4f} ± {np.std(kendall_taus):.4f}",
        "Top-K Overlap": f"{np.mean(top_k_overlaps):.4f} ± {np.std(top_k_overlaps):.4f}"
    }

def save_sparse_results(results, filename):
    if isinstance(results, list):
        sparse.save_npz(filename, results[-1])
    else:
        sparse.save_npz(filename, results)

def load_sparse_results(filename):
    return sparse.load_npz(filename)

def run_single_mc_trial(graph, beta, gamma, initial_infected_nodes, num_simulations, trial_idx, temp_dir):
    start_time = time.time()
    all_final_s_values = []
    all_trajectories_lengths = []
    infection_probs = np.zeros(len(graph))
    batch_size = min(20, num_simulations)
    num_batches = (num_simulations + batch_size - 1) // batch_size

    for batch in range(num_batches):
        current_batch_size = min(batch_size, num_simulations - batch * batch_size)
        print(f"[MC Trial {trial_idx}] Batch {batch + 1}/{num_batches}...")
        batch_results, batch_trajectories, _ = run_monte_carlo_simulations(
            graph, beta, gamma, initial_infected_nodes, current_batch_size
        )
        all_final_s_values.extend([traj[-1] for traj in batch_trajectories])
        all_trajectories_lengths.extend([len(traj) for traj in batch_trajectories])
        infection_probs += calculate_infection_probability(batch_results) * (current_batch_size / num_simulations)
        del batch_results, batch_trajectories
        gc.collect()

    runtime = time.time() - start_time
    np.save(os.path.join(temp_dir, f'mc_final_s_trial_{trial_idx}.npy'), all_final_s_values)
    np.save(os.path.join(temp_dir, f'mc_infection_probs_trial_{trial_idx}.npy'), infection_probs)
    return runtime, np.mean(all_trajectories_lengths)

def sparse_dict_to_matrix(dict_data, num_nodes):
    rows = []
    cols = []
    data = []
    if 'iteration' in dict_data:
        dict_data.pop('iteration')
    state_to_col = {'susceptible': 0, 'infected': 1, 'recovered': 2}
    for state, node_dict in dict_data.items():
        if state in state_to_col:
            for node, value in node_dict.items():
                if isinstance(node, (int, str)) and str(node).isdigit():
                    rows.append(int(node))
                    cols.append(state_to_col[state])
                    data.append(float(value))
    return sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, 3))

def matrix_to_dict(sparse_matrix):
    states = ['susceptible', 'infected', 'recovered']
    result = {state: {} for state in states}
    for i, j in zip(*sparse_matrix.nonzero()):
        value = sparse_matrix[i, j]
        if value != 0:
            result[states[j]][str(i)] = float(value)
    return result

def run_method_trial(method_func, graph, beta, gamma, initial_infected_nodes, tol, trial_idx, temp_dir):
    start_time = time.time()
    results = None
    try:
        results = method_func(graph, beta, gamma, initial_infected_nodes, tol)
        runtime = time.time() - start_time
        if not results or not isinstance(results, list):
            raise ValueError(f"Method returned invalid results: {results}")
        final_state = results[-1]
        num_nodes = len(graph)
        sparse_final_state = sparse_dict_to_matrix(final_state, num_nodes)
        save_path = os.path.join(temp_dir, f'method_results_trial_{trial_idx}.npz')
        save_sparse_results(sparse_final_state, save_path)
        final_s = sum(float(v) for v in final_state.get('susceptible', {}).values())
        return {
            "runtime": runtime,
            "final_s": final_s,
            "iterations": len(results)
        }
    except Exception as e:
        print(f"[Error] Method trial {trial_idx} failed: {e}")
        raise
    finally:
        if results is not None:
            del results
        gc.collect()

def load_method_final_probs(trial_idx, temp_dir, num_nodes):
    sparse_final_state = load_sparse_results(
        os.path.join(temp_dir, f'method_results_trial_{trial_idx}.npz')
    )
    final_dict = matrix_to_dict(sparse_final_state)
    method_probs = np.zeros(num_nodes)
    for node_str in final_dict['infected']:
        node = int(node_str)
        method_probs[node] += final_dict['infected'][node_str]
    for node_str in final_dict['recovered']:
        node = int(node_str)
        method_probs[node] += final_dict['recovered'][node_str]
    return method_probs

def main(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    temp_dir = os.path.join(output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    print("[Info] Loading or generating graph...")
    if args.file_name:
        graph = load_real_data(args.data_dir, args.file_name)
    else:
        graph = generate_graph(graph_type=args.graph_type,
                               avg_degree=args.average_degree,
                               nodes=args.nodes)
    args.nodes = len(graph.nodes())
    save_args_to_csv(args, output_dir)
    if len(graph.nodes()) < args.initial_infected:
        raise ValueError("The graph does not have enough nodes for the initial infected nodes.")
    initial_infected_nodes = random.sample(list(graph.nodes()), args.initial_infected)

    print("[Info] Running Monte Carlo simulations (3 trials) ...")
    mc_runtime = []
    mc_iterations = []
    for i in range(3):
        print(f"[MC] Trial {i+1}/3 started.")
        runtime, iterations = run_single_mc_trial(
            graph, args.beta, args.gamma, initial_infected_nodes,
            args.num_simulations, i, temp_dir
        )
        print(f"[MC] Trial {i+1}/3 finished. Runtime={runtime:.2f}s, Iterations={iterations:.2f} (avg).")
        mc_runtime.append(runtime)
        mc_iterations.append(iterations)

    print("[Info] Loading MC trial results from disk...")
    mc_final_s_trials = []
    mc_infection_probs = []
    for i in range(3):
        mc_final_s_trials.append(
            np.load(os.path.join(temp_dir, f'mc_final_s_trial_{i}.npy'))
        )
        mc_infection_probs.append(
            np.load(os.path.join(temp_dir, f'mc_infection_probs_trial_{i}.npy'))
        )

    print("[Info] Calculating MC summary statistics...")
    mean_s, lower_bound, upper_bound, mean_error, lower_error, upper_error = \
        calculate_mc_final_s_statistics(mc_final_s_trials)
    mc_runtime_mean = np.mean(mc_runtime)
    mc_runtime_std = np.std(mc_runtime)
    mc_iterations_mean = np.mean(mc_iterations)
    mc_iterations_std = np.std(mc_iterations)
    mc_summary = {
        "Metric": [
            "Mean S",
            "Lower Bound (2.5%)",
            "Upper Bound (97.5%)",
            "MC Runtime (s)",
            "MC Iterations"
        ],
        "Value": [
            f"{mean_s:.4f} ± {mean_error:.4f}",
            f"{lower_bound:.4f} ± {lower_error:.4f}",
            f"{upper_bound:.4f} ± {upper_error:.4f}",
            f"{mc_runtime_mean:.4f} ± {mc_runtime_std:.4f}",
            f"{mc_iterations_mean:.4f} ± {mc_iterations_std:.4f}",
        ]
    }
    mc_summary_df = pd.DataFrame(mc_summary)
    mc_summary_df.to_csv(os.path.join(output_dir, "mc_summary.csv"), index=False)
    print("[Info] MC summary saved to mc_summary.csv")

    methods = {
        "Ground Truth": conditional_probability_iteration,
        "Approx Exp": approx_conditional_probability_iteration_exp,
        "SOR Approx Exp": sor_approx_conditional_probability_iteration_exp,
        "Local Push": approx_conditional_probability_iteration_local_push,
    }
    method_summary = []
    num_nodes = len(graph)
    for method_name, method_func in methods.items():
        print(f"[Method] {method_name} ...")
        trial_results = []
        for i in range(3):
            print(f"    -> Trial {i + 1}/3 started.")
            trial_result = run_method_trial(
                method_func, graph, args.beta, args.gamma,
                initial_infected_nodes, args.tol, i, temp_dir
            )
            print(f"    -> Trial {i + 1}/3 finished. "
                  f"Runtime={trial_result['runtime']:.2f}s, "
                  f"Iterations={trial_result['iterations']}")
            trial_results.append(trial_result)

        runtimes = [res['runtime'] for res in trial_results]
        iterations = [res['iterations'] for res in trial_results]
        final_s_values = [res['final_s'] for res in trial_results]

        method_probs = load_method_final_probs(trial_idx=2, temp_dir=temp_dir, num_nodes=num_nodes)
        error_metrics = calculate_error_metrics(mc_infection_probs, method_probs, k=round(0.1*args.nodes))
        method_data = {
            "Method": [method_name],
            "Final S": [f"{np.mean(final_s_values):.4f} ± {np.std(final_s_values):.4f}"],
            "Runtime (s)": [f"{np.mean(runtimes):.4f} ± {np.std(runtimes):.4f}"],
            "Iterations": [f"{np.mean(iterations):.2f} ± {np.std(iterations):.2f}"],
            "Kendall Tau": [error_metrics['Kendall Tau']],
            "Top-K Overlap": [error_metrics['Top-K Overlap']]
        }
        method_df = pd.DataFrame(method_data)
        method_filename = f"{method_name.replace(' ', '_')}_summary.csv"
        method_df.to_csv(os.path.join(output_dir, method_filename), index=False)
        print(f"[Info] Saved {method_name} results to {method_filename}")

    # --------------------- FIX: Define 'metrics' and 'mean_mc_infection_prob' ---------------------
    metrics = {
        "Kendall Tau": {},
        "Top-K Overlap": {},
        "Runtime": {},
        "Iterations": {}
    }

    # We compute mean_mc_infection_prob as the average across the 3 MC trials:
    mean_mc_infection_prob = np.mean(mc_infection_probs, axis=0)

    print("[Info] Calculating centralities...")
    centralities = calculate_centralities(graph)
    for name, values in centralities.items():
        print(f"[centrality] Processing {name} centrality...")
        # Use the average MC's length to match node indexing
        centrality_probs = np.array([values[node] for node in range(len(mean_mc_infection_prob))])
        metrics_with_error = calculate_error_bars_for_metrics(mc_infection_probs, centrality_probs, k=round(0.1*args.nodes))
        metrics["Kendall Tau"][name] = metrics_with_error["Kendall Tau"]
        metrics["Top-K Overlap"][name] = metrics_with_error["Top-K Overlap"]
        metrics["Runtime"][name] = "N/A"
        metrics["Iterations"][name] = "N/A"

    print("[Info] Centralities calculated.")
    results_df = pd.DataFrame(metrics).T
    results_df.to_csv(os.path.join(output_dir, "centralities.csv"))
    print(f"[Info] Results saved to: {os.path.join(output_dir, 'centralities.csv')}")

    print("[Info] Cleaning up temporary files...")
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)
    print(f"[Done] All results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SIR model simulations.")
    parser.add_argument("--graph_type", type=str, default="", help="Graph type (ba, er, powerlaw, smallworld)")
    parser.add_argument("--nodes", type=int, default=1000, help="Number of nodes in the graph")
    parser.add_argument("--average_degree", type=float, default=20, help="Average degree of the graph")
    parser.add_argument("--beta", type=float, default=1 / 18, help="Infection rate")
    parser.add_argument("--gamma", type=float, default=1 / 9, help="Recovery rate")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance for convergence")
    parser.add_argument("--num_simulations", type=int, default=100, help="Number of Monte Carlo simulations")
    parser.add_argument("--initial_infected", type=int, default=50, help="Number of initially infected nodes")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to load real data")
    parser.add_argument("--file_name", default="email-EuAll.txt.gz", type=str, help="File name of the real data")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save results")

    args = parser.parse_args()
    main(args)