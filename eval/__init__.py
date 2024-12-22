import numpy as np
from scipy.stats import kendalltau


def calculate_infection_probability(results_list):
    N = len(results_list[0][0]['status'])
    infection_counts = np.zeros(N)
    num_simulations = len(results_list)

    for results in results_list:
        for result in results:
            for node, status in result['status'].items():
                if status == 1 or status == 2:
                    infection_counts[node] += 1

    infection_probability = infection_counts / num_simulations

    return infection_probability

def calculate_kendall_tau(results_exp, infection_probability):
    last_iteration = results_exp[-1]
    recovered_prob = np.array([last_iteration['recovered'][node] + last_iteration['infected'][node]
                               for node in range(len(infection_probability))])
    tau, p_value = kendalltau(infection_probability, recovered_prob)
    return tau, p_value


def calculate_top_k_overlap(recovered_prob, infection_probability, k=100):
    # last_iteration = results_exp[-1]
    # recovered_prob = np.array([last_iteration['recovered'][node] + last_iteration['infected'][node]
    #                            for node in range(len(infection_probability))])
    top_k_mc_nodes = np.argsort(infection_probability)[-k:]
    top_k_exp_nodes = np.argsort(recovered_prob)[-k:]
    overlap_count = len(set(top_k_mc_nodes).intersection(set(top_k_exp_nodes)))
    return overlap_count / k


def calculate_quantile(final_s, mc_final_s):
    return np.sum(np.array(mc_final_s) < final_s) / len(mc_final_s) * 100


def calculate_mse(final_s, mc_final_s):
    return np.mean((np.array(mc_final_s) - final_s) ** 2)
