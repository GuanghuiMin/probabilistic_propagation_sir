import torch
from scipy.stats import kendalltau



def compare_mc_and_ranking(results_exp, infection_probability):
    """Analyze results using Kendall-Tau correlation."""
    last_iteration = results_exp[-1]
    recovered_prob = torch.tensor(
        [last_iteration['recovered'][node] for node in range(len(infection_probability))],
        device='cuda'
    )
    infection_probability_cpu = infection_probability.cpu().numpy()
    recovered_prob_cpu = recovered_prob.cpu().numpy()
    tau, p_value = kendalltau(infection_probability_cpu, recovered_prob_cpu)
    return tau, p_value


def top_k_overlap(results_exp, infection_probability, k):
    last_iteration = results_exp[-1]
    recovered_prob = torch.tensor(
        [last_iteration['recovered'][node] for node in range(len(infection_probability))],
        device='cuda'
    )

    top_k_mc_nodes = torch.topk(infection_probability, k).indices
    top_k_exp_nodes = torch.topk(recovered_prob, k).indices

    overlap_count = len(set(top_k_mc_nodes.cpu().numpy()).intersection(set(top_k_exp_nodes.cpu().numpy())))
    overlap_ratio = overlap_count / k

    return overlap_ratio