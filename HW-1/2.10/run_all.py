from find_s import FindS
from experiment import run_all_experiments, plot_results

def reproduce_table_2_1():
    # Reproduce textbook Section 2.4 trace using Table 2.1 dataset
    table_2_1 = [
        (['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'], 'Yes'),
        (['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'], 'Yes'),
        (['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'], 'No'),
        (['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'], 'Yes')
    ]

    model = FindS(num_attributes=6)
    trace = model.fit_with_trace(table_2_1)

    print("=== Part 1: Textbook Section 2.4 Trace on Table 2.1 ===")
    for step, hypothesis in enumerate(trace):
        print(f"h_{step}: {hypothesis}")
    print("-" * 55)

def main():
    # Execute full pipeline to reproduce all results and figures
    reproduce_table_2_1()

    stats = run_all_experiments(num_runs=25, max_examples=1000)

    print("\n=== Part 2: Summary Statistics (25 Independent Runs) ===")
    print(f"Stopped by Condition (i) [Target Learned]: {stats['n_success']} runs")
    print(f"Stopped by Condition (ii) [N_max Reached]: {stats['n_failed']} runs")
    print(f"Mean ± Std:   {stats['mean']:.2f} ± {stats['std']:.2f}")
    print(f"Median (IQR): {stats['median']:.2f} ({stats['iqr']:.2f})")
    print(f"Min / Max:    {stats['min']} / {stats['max']}")

    plot_results(stats['raw_counts'], filename='finds_convergence.png')
    print("\nFigure saved successfully as 'finds_convergence.png'.")

if __name__ == "__main__":
    main()

