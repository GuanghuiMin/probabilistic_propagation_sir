import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results(mc_trajectories, ground_truth_s, approx_exp_s, sor_exp_s, local_push_exp_s, output_dir):
    max_length = max(
        max(len(traj) for traj in mc_trajectories),
        len(ground_truth_s),
        len(approx_exp_s),
        len(sor_exp_s),
        len(local_push_exp_s)
    )

    # Pad Monte Carlo trajectories
    padded_mc_trajectories = np.array(
        [traj + [traj[-1]] * (max_length - len(traj)) for traj in mc_trajectories]
    )

    mean_s = np.mean(padded_mc_trajectories, axis=0)
    lower_bound = np.percentile(padded_mc_trajectories, 2.5, axis=0)
    upper_bound = np.percentile(padded_mc_trajectories, 97.5, axis=0)

    plt.figure(figsize=(12, 8))
    for traj in padded_mc_trajectories:
        plt.plot(traj, color='gray', alpha=0.2)

    plt.plot(mean_s, color='blue', label='MC Mean S', linewidth=2)
    plt.fill_between(range(len(mean_s)), lower_bound, upper_bound, color='blue', alpha=0.3,
                     label='95% Confidence Interval')
    plt.plot(ground_truth_s, color='green', label='Ground Truth S', linewidth=2)
    plt.plot(approx_exp_s, color='red', label='Approx Exp S', linewidth=2)
    plt.plot(sor_exp_s, color='orange', label='SOR Exp S', linewidth=2)
    plt.plot(local_push_exp_s, color='black', label='Local Push Exp S', linewidth=3)

    plt.xlabel('Iterations')
    plt.ylabel('Number of Susceptible Individuals')
    plt.title('SIR Model: Susceptible State Trajectories')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(output_dir, "sir_trajectories.png"))
    plt.close()
    print(f"Plot saved to: {os.path.join(output_dir, 'sir_trajectories.png')}")
