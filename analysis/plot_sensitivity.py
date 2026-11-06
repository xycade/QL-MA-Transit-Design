import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_mock_data():
    """Generate mock data matching the experimental profile for testing."""
    np.random.seed(42)
    alphas = ['0', '100', '500', '2000', '10000']
    data = []

    for a in alphas:
        # Simulate data generation for each group
        if a == '0':
            unmet = np.random.normal(loc=250000, scale=15000, size=30)
            travel = np.random.normal(loc=1300000, scale=80000, size=30)
        else:
            unmet = np.random.normal(loc=0, scale=10, size=30)
            unmet = np.clip(unmet, 0, None)  # Ensure non-negative values
            travel = np.random.normal(loc=6500000, scale=50000, size=30)

        for u, t in zip(unmet, travel):
            data.append({'Penalty_Alpha': a, 'Unmet_Demand': u, 'Total_Travel_Time': t})

    return pd.DataFrame(data)


def plot_dual_axis_sensitivity():
    # 1. Load data (replace with pd.read_csv if using actual Pareto_Metrics_Raw.csv)
    df = generate_mock_data()

    # Calculate mean and standard deviation for error bars
    grouped = df.groupby('Penalty_Alpha', sort=False).agg(
        unmet_mean=('Unmet_Demand', 'mean'),
        unmet_std=('Unmet_Demand', 'std'),
        travel_mean=('Total_Travel_Time', 'mean'),
        travel_std=('Total_Travel_Time', 'std')
    ).reset_index()

    # 2. Global academic typography and layout settings
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 1.2,
    })

    # 3. Create canvas and dual Y-axes
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()

    # Map categorical levels to equidistant numerical indices
    x = np.arange(len(grouped['Penalty_Alpha']))

    # 4. Draw primary Y-axis (left): Unmet Demand (bar plot with standard deviation error bars)
    color_bar = '#E67E22'  # Muted warm orange
    bars = ax1.bar(
        x,
        grouped['unmet_mean'],
        yerr=grouped['unmet_std'],
        color=color_bar,
        alpha=0.7,
        edgecolor='k',
        linewidth=1.2,
        capsize=5,
        label='Unmet Demand'
    )

    # 5. Draw secondary Y-axis (right): Total Travel Time (line plot with standard deviation error bars)
    color_line = '#008080'  # Teal
    line = ax2.errorbar(
        x,
        grouped['travel_mean'],
        yerr=grouped['travel_std'],
        color=color_line,
        marker='s',
        markersize=8,
        linewidth=2.5,
        capsize=5,
        label='Total Travel Time'
    )

    # 6. Axis labels and ticks (plain text to ensure LaTeX parsing safety)
    ax1.set_xlabel("Penalty Alpha (p_u)", fontweight='bold')
    ax1.set_ylabel("Unmet Demand (Passengers)", fontweight='bold', color=color_bar)
    ax2.set_ylabel("Total Travel Time (Minutes)", fontweight='bold', color=color_line)

    ax1.set_xticks(x)
    ax1.set_xticklabels(grouped['Penalty_Alpha'])

    # 7. Format Y-axes using scientific notation
    ax1.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    ax2.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

    ax1.yaxis.get_offset_text().set_fontsize(12)
    ax2.yaxis.get_offset_text().set_fontsize(12)

    # Synchronize axis tick colors with corresponding visual elements
    ax1.tick_params(axis='y', colors=color_bar)
    ax2.tick_params(axis='y', colors=color_line)

    # 8. Merge and refine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Extract the correct handle from the errorbar container
    if type(handles2[0]) is tuple:
        h2 = handles2[0][0]
    else:
        h2 = handles2[0]

    ax1.legend(
        [bars, h2],
        ['Unmet Demand', 'Total Travel Time'],
        bbox_to_anchor=(0.98, 0.98),  # Position inside the top-right corner
        loc='upper right',
        frameon=True,
        fancybox=True,
        framealpha=0.8,
        shadow=True,
        borderaxespad=0.
    )

    # Disable grid on secondary axis to prevent visual clutter; retain primary grid in background
    ax2.grid(False)
    ax1.set_axisbelow(True)

    # 9. Save and display
    plt.tight_layout()
    save_path = "output/analysis_results/Fig_Sensitivity_DualAxis.png"
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    print(f"✅ Plot successfully saved to {save_path} (DPI=500)")


if __name__ == '__main__':
    plot_dual_axis_sensitivity()