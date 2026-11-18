import random
import numpy as np
import matplotlib.pyplot as plt
from find_s import FindS

ATTRIBUTES = {
    'Sky': ['Sunny', 'Cloudy', 'Rainy'],
    'AirTemp': ['Warm', 'Cold'],
    'Humidity': ['Normal', 'High'],
    'Wind': ['Strong', 'Weak'],
    'Water': ['Warm', 'Cool'],
    'Forecast': ['Same', 'Change']
}

ATTR_KEYS = list(ATTRIBUTES.keys())

def generate_random_instance(rng):
    # Generate a single random instance uniformly across all attributes
    return [rng.choice(ATTRIBUTES[key]) for key in ATTR_KEYS]

def label_instance(instance, target_concept):
    # Assign 'Yes' if instance matches target_concept, otherwise 'No'
    for x_val, t_val in zip(instance, target_concept):
        if t_val != '?' and t_val != x_val:
            return 'No'
    return 'Yes'

def run_single_experiment(seed, target_concept, max_examples=1000):
    # Run FIND-S on random instances until stopping condition (i) or (ii)
    rng = random.Random(seed)
    model = FindS(len(ATTR_KEYS))

    for count in range(1, max_examples + 1):
        instance = generate_random_instance(rng)
        label = label_instance(instance, target_concept)
        model.fit_step(instance, label)

        if model.hypothesis == target_concept:
            return count, 'Condition (i): Exact target learned'

    return max_examples, 'Condition (ii): N_max reached'

def run_all_experiments(num_runs=25, max_examples=1000):
    # Execute multiple independent runs and compute required statistics
    target = ['Sunny', 'Warm', '?', '?', '?', '?']
    successful_counts = []
    failed_seeds = []

    print(f"--- Running {num_runs} Independent Experiments ---")
    for seed in range(num_runs):
        steps, condition = run_single_experiment(seed, target, max_examples)
        print(f"Seed {seed:02d} -> Examples: {steps:3d} | Stopped by: {condition}")

        if 'Condition (i)' in condition:
            successful_counts.append(steps)
        else:
            failed_seeds.append(seed)

    arr = np.array(successful_counts)
    q75, q25 = np.percentile(arr, [75, 25])

    stats = {
        'n_success': len(arr),
        'n_failed': len(failed_seeds),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)),
        'median': float(np.median(arr)),
        'iqr': float(q75 - q25),
        'min': int(np.min(arr)),
        'max': int(np.max(arr)),
        'raw_counts': successful_counts
    }
    return stats

def plot_results(counts, filename='finds_convergence.png'):
    # Plot and save histogram with labelled axes and run count
    plt.figure(figsize=(8, 5))
    plt.hist(counts, bins=10, color='#FF8C00', edgecolor='black', alpha=0.85)
    plt.axvline(np.mean(counts), color='blue', linestyle='dashed', linewidth=1.5,
                label=f"Mean: {np.mean(counts):.2f}")
    plt.axvline(np.median(counts), color='green', linestyle='dotted', linewidth=1.5,
                label=f"Median: {np.median(counts):.2f}")

    plt.title(f"Training Examples Required by FIND-S (N = {len(counts)} runs)")
    plt.xlabel("Number of Random Training Examples")
    plt.ylabel("Frequency (Number of Runs)")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

